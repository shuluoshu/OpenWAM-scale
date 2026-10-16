"""CPU-only real-reader before/after review; verifies GT/FK are unchanged."""
import argparse
import base64
import copy
import hashlib
import html
import io
import json
from pathlib import Path
import random

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import torch

from openwam.dataloader.curated_fixture import CuratedFixtureDataset
from openwam.dataloader.transforms.observation import ObservationAugmentation, TRACE_KEY


def equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal(a[key], b[key])
    elif isinstance(a, (list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif torch.is_tensor(a):
        assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    else:
        assert a==b


def verify_pair(before, after):
    for key in ("action", "action_mask", "video_mask", "prompt"):
        equal(before[key], after[key])
    equal(before.get("fk_supervision"), after.get("fk_supervision"))
    meta=after.get(TRACE_KEY,{})
    if meta.get("state_dropped"):
        assert not after["proprio"].any() and not after["proprio_mask"].any()
    else:
        equal(before["proprio"],after["proprio"])
        equal(before["proprio_mask"],after["proprio_mask"])
    patch=meta.get("patch")
    for frame in after["video"]:
        if patch:
            assert not np.asarray(frame.crop(patch["box"])).any()
        if meta.get("dropped_view") is not None:
            assert not np.asarray(frame.crop(meta["view_boxes"][meta["dropped_view"]])).any()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path,required=True)
    p.add_argument("--config",type=Path,default=Path("configs/augmentation/multiview.yaml"))
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--clips",nargs="+",default=["04_LightWheel","08_abc_130k","22_interndata_a1"])
    args=p.parse_args()
    if args.output_dir.exists():p.error("choose a new output directory")
    torch.set_num_threads(2)
    config=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    clean=CuratedFixtureDataset(args.manifest)
    augmented=CuratedFixtureDataset(args.manifest,augmentation=config)
    assert clean.mapping==augmented.mapping
    checks=[]
    random.seed(420)
    for i,(row,a,b) in enumerate(zip(clean.rows,clean.readers,augmented.readers)):
        np.testing.assert_array_equal(a._valid_source_indices,b._valid_source_indices)
        index=next(k for k,pair in enumerate(clean.mapping) if pair==(i,0))
        before,after=clean[index],augmented[index]
        verify_pair(before,after)
        checks.append(dict(id=row["id"],unchanged_gt_and_fk=True,
                           augmentation=after.get(TRACE_KEY,{}),windows=len(a)))
    modes={"default":config}
    for key in ("color_jitter","patch_mask","view_dropout","state_dropout","combined"):
        cfg=copy.deepcopy(config)
        for name in ("color_jitter","patch_mask","view_dropout","state_dropout"):
            cfg[name]["p"]=1 if key in (name,"combined") else 0
        modes[key]=cfg
    args.output_dir.mkdir(parents=True)
    cards=[];records=[]
    for ident in args.clips:
        i=next(i for i,row in enumerate(clean.rows) if row["id"]==ident)
        index=next(k for k,pair in enumerate(clean.mapping) if pair==(i,0))
        before=clean[index]
        for n,(mode,cfg) in enumerate(modes.items()):
            random.seed(17001+n)
            augmented.readers[i].observation_augmentation=ObservationAugmentation(cfg)
            after=augmented[index]
            verify_pair(before,after)
            meta=after.get(TRACE_KEY,{})
            width,height=before["video"][0].size
            contact=Image.new("RGB",(width*4,height+26),"#152033")
            draw=ImageDraw.Draw(contact)
            views=[("clean t0",before["video"][0]),("aug t0",after["video"][0]),
                   ("aug t4",after["video"][4]),("aug t8",after["video"][8])]
            for col,(title,image) in enumerate(views):
                contact.paste(image,(col*width,26));draw.text((col*width+6,6),title,fill="white")
            buffer=io.BytesIO();contact.save(buffer,format="PNG")
            (args.output_dir/f"{ident}_{mode}.png").write_bytes(buffer.getvalue())
            uri="data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode()
            record=dict(id=ident,mode=mode,augmentation=meta,
                        clean_state=before["proprio"][0].tolist(),
                        augmented_state=after["proprio"][0].tolist(),
                        unchanged_gt_and_fk=True,fk_present="fk_supervision" in after)
            records.append(record)
            cards.append(f'<section><h2>{ident} / {mode}</h2><img src="{uri}">'
                f'<p>state dropped: {meta.get("state_dropped",False)} · GT/action mask/FK anchor unchanged</p>'
                f'<pre>{html.escape(json.dumps(meta,ensure_ascii=False,indent=2))}</pre></section>')
    # Fixed evaluation must neither augment nor consume augmentation RNG.
    with augmented.unaugmented():
        state=random.getstate()
        for k in augmented.eval_indices():
            a,b=clean[k],augmented[k]
            equal(a["proprio"],b["proprio"])
            equal(a["proprio_mask"],b["proprio_mask"])
            for x,y in zip(a["video"],b["video"]):np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
        assert random.getstate()==state
    sources=[Path("openwam/dataloader/transforms/observation.py"),Path("openwam/train/utils/observation_cache.py")]
    report=dict(all_passed=True,config=config,clips=len(clean.rows),windows=len(clean),
                checks=checks,examples=records,clean_evaluation_verified=True,
                source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
                scope="CPU production-reader observations; not a model/GPU training benchmark")
    (args.output_dir/"validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    document='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Observation augmentation</title>
<style>body{max-width:1400px;margin:24px auto;font:16px system-ui}img{max-width:100%}section{border:1px solid #ccc;padding:16px;margin:20px 0}pre{white-space:pre-wrap}</style>
<h1>当前生产 reader：增强前后对比</h1><p>先完成 resize/crop 与三视角拼接，再做颜色/单块遮挡/单视角drop；同clip各帧共享参数。head=上，left/right=下。state dropout只改输入，不改action GT或FK锚点。</p>
<p>default使用配置概率；其余卡片强制该操作发生以便验收；combined将四个概率都设为1。单视角样例保留唯一相机。视频训练目标也使用增强后的帧。未训练/未执行模型forward。</p>'''+''.join(cards)+'</html>'
    (args.output_dir/"index.html").write_text(document,encoding="utf-8")
    print(json.dumps(dict(all_passed=True,clips=len(clean.rows),windows=len(clean),
                         examples=len(records),page=str(args.output_dir/"index.html")),indent=2))


if __name__=="__main__":main()
