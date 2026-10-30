"""Read-only interim Stage1 image/TCP replay on a few fitted smoke anchors."""
import argparse,json,html,sys
from pathlib import Path
import torch,yaml

CODE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(CODE))
from openwam.train.goal_visual_history import load_visual_history_bundle
from openwam.train.goal_trainer import _construct,load_component_weights
from openwam.train.goal_curated import prepare_batch
from openwam.train.goal_overfit import export_inputs,_write_json
from overfit_visual_history import evaluate,rgb,perturb

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    smoke=a.root/'outputs/stage1_smoke_history_20261029';cfg=yaml.safe_load((smoke/'config.yaml').read_text());deps=json.loads((smoke/'dependencies.json').read_text())
    vae=_construct(cfg['components']['vae']);load_component_weights(vae,a.root/'assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors','video_backbone.vae.')
    vae.to(device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    expert=load_visual_history_bundle(a.checkpoint,vae=vae,text_dim=deps['text_dim'],precomputed_text_fingerprint=deps['text']['fingerprint'],device='cuda',dtype=torch.bfloat16)
    contexts=torch.load(smoke/'text_contexts.pt',weights_only=True,map_location='cpu');items=[]
    for index in [8,12,4,26]:
        r,b=torch.load(smoke/'samples'/f'{index:02d}_0.pt',weights_only=False,map_location='cpu');b['context'],b['context_lens']=contexts[r['instruction']]
        items.append((r,prepare_batch(r,b,device='cuda',dtype=torch.bfloat16)))
    export_inputs(a.output,[r for r,b in items],[b for r,b in items]);rows=[]
    for i,(r,b) in enumerate(items):
        values={mode:evaluate(expert,b if mode=='history' else perturb(b,b,'no_history'),a.output/f'case_{i:03d}'/mode,all_targets=[rgb(b['targets'].image)]) for mode in ['single','history']}
        rows.append(dict(case=i,family=r['family'],values=values));print(json.dumps(rows[-1]),flush=True)
    _write_json(a.output/'metrics.json',dict(checkpoint=str(a.checkpoint),training=expert.bundle_metadata.get('training'),cases=rows))
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Stage1 中途预测预览</title><style>body{font:16px system-ui;max-width:1500px;margin:auto}img{width:300px}table{border-collapse:collapse}td{padding:10px}</style><h1>Stage1 中途预测预览</h1><p>当前训练锚点；同一 checkpoint、相同噪声切换单帧/历史。非最终冻结版本，不是泛化评估。</p>']
    for x in rows:
        i=x['case'];parts.append('<h2>'+html.escape(x['family'])+'</h2><table><tr>')
        for name,label in [(f'inputs/case_{i:04d}_observed.png','当前输入'),(f'inputs/case_{i:04d}_target_points.png','GT'),(f'case_{i:03d}/single/prediction.png','单帧预测'),(f'case_{i:03d}/history/prediction.png','历史预测')]:parts.append(f'<td>{label}<br><img src="{name}"></td>')
        parts.append('</tr></table><p>'+html.escape(json.dumps({m:{k:v[k] for k in ['mae','tcp_error_px','valid_tcp']} for m,v in x['values'].items()},ensure_ascii=False))+'</p>')
    (a.output/'index.html').write_text(''.join(parts)+'</html>')

if __name__=='__main__':main()
