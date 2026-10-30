"""Fresh-process reconstruction + raw-RGB inference of a saved real bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import yaml
from safetensors import safe_open

CODE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(CODE))
from openwam.train.goal_visual_history import load_visual_history_bundle
from openwam.train.goal_trainer import _construct,load_component_weights
from openwam.train.goal_curated import prepare_batch
from openwam.train.goal_overfit import _write_json
from overfit_visual_history import evaluate,rgb


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=json.loads((a.run/'protocol.json').read_text());meta=json.loads((a.run/'visual_history.json').read_text())
    smoke=a.root/'outputs/stage1_smoke_history_20261029';cfg=yaml.safe_load((smoke/'config.yaml').read_text())
    vae=_construct(cfg['components']['vae']);load_component_weights(vae,protocol['checkpoint'],'video_backbone.vae.')
    vae.to(device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    checkpoint=a.run/'visual_history.safetensors'
    expert=load_visual_history_bundle(checkpoint,vae=vae,text_dim=meta['dependencies']['text_dim'],
        precomputed_text_fingerprint=meta['dependencies']['text']['fingerprint'],device='cuda',dtype=torch.bfloat16)
    state_exact=True
    with safe_open(str(checkpoint),framework='pt',device='cpu') as f:
        for k,v in expert.state_dict().items():
            state_exact=state_exact and torch.equal(v.detach().cpu(),f.get_tensor(k))
    contexts=torch.load(smoke/'text_contexts.pt',weights_only=True,map_location='cpu')
    items=[torch.load(smoke/'samples'/f'{i:02d}_2.pt',weights_only=False,map_location='cpu') for i in protocol['source_ids']]
    batches=[]
    for r,b in items:
        b['context'],b['context_lens']=contexts[protocol['common_instruction']]
        batches.append(prepare_batch(r,b,device='cuda',dtype=torch.bfloat16))
    targets=[rgb(b['targets'].image) for b in batches];results=[]
    for i,b in enumerate(batches):
        directory=a.run/'report/fresh_reload'/f'case_{i}'
        metrics=evaluate(expert,b,directory,all_targets=targets)
        expected=np.asarray(Image.open(a.run/f'report/final/case_{i}/real/prediction.png'))
        actual=np.asarray(Image.open(directory/'prediction.png'))
        delta=np.abs(expected.astype(np.float32)-actual.astype(np.float32))
        results.append(dict(case=i,pixel_identical=bool(np.array_equal(expected,actual)),mae=metrics['mae'],
            replay_pixel_mae=float(delta.mean()),replay_max_pixel_delta=float(delta.max()),
            nearest_scene=metrics['nearest_scene'],numerical_tolerance_pass=bool(delta.mean()<1. and metrics['nearest_scene']==i)))
    # Diagnose raw-RGB vs precomputed-latent and frozen vs trainable inference.
    b=dict(batches[0]);b['history_latents']=torch.stack([
        expert.codec.encode(b['history_images'][:,j]) if bool(b['history_valid'][0,j])
        else torch.zeros(1,48,1,16,20,device='cuda',dtype=torch.bfloat16) for j in range(10)],dim=1)
    del b['history_images']
    directory=a.run/'report/fresh_reload/cached_path'
    evaluate(expert,b,directory,all_targets=targets)
    raw=np.asarray(Image.open(a.run/'report/fresh_reload/case_0/prediction.png'))
    cached=np.asarray(Image.open(directory/'prediction.png'))
    expert.train_stage_a().eval()
    matched_state=[]
    for i,b in enumerate(batches):
        directory2=a.run/'report/fresh_reload/trainable_eval'/f'case_{i}'
        evaluate(expert,b,directory2,all_targets=targets)
        original=np.asarray(Image.open(a.run/f'report/final/case_{i}/real/prediction.png'))
        trainable=np.asarray(Image.open(directory2/'prediction.png'))
        matched_state.append(bool(np.array_equal(original,trainable)))
    trainable=np.asarray(Image.open(a.run/'report/fresh_reload/trainable_eval/case_0/prediction.png'))
    path_check=dict(raw_cached_pixel_identical=bool(np.array_equal(raw,cached)),
        frozen_trainable_pixel_identical=bool(np.array_equal(raw,trainable)),
        matched_trainability_state_pixel_identical=matched_state,
        finding='Fresh reconstruction is pixel-exact with the original evaluation requires_grad state. Freezing parameters changes BF16 inference numerics slightly; raw/cached history paths agree.')
    expert.freeze()
    h=hashlib.sha256()
    with checkpoint.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    result=dict(fresh_process=True,rebuilt_from_bundle=True,raw_rgb_history=True,vae_text_dependencies_checked=True,
        checkpoint_sha256=h.hexdigest(),state_dict_bit_exact=state_exact,path_check=path_check,
        acceptance='Saved tensors bit-exact; replay image mean absolute difference <1/255 and correct trained scene. Bit-exact pixel replay is reported separately.',
        frozen=not any(p.requires_grad for p in expert.parameters()),cases=results)
    _write_json(a.run/'report/fresh_reload.json',result);_write_json(a.run/'fresh_reload.json',result)
    print(json.dumps(result),flush=True)
    assert state_exact and all(r['numerical_tolerance_pass'] for r in results)
    assert path_check['raw_cached_pixel_identical'] and all(matched_state)


if __name__=='__main__':main()
