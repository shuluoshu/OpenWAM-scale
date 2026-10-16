#!/usr/bin/env python3
"""Export one current prepared-fixture sample, without a model or GPU forward."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

from tools import find_repo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--sample-index", type=int, default=0, help="accepted window within the clip")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    repo = find_repo(args.repo)
    manifest = (repo / args.manifest).resolve()
    output = args.output_dir.expanduser().resolve()
    if not manifest.is_file():
        p.error("prepared-fixture manifest does not exist")
    if output.exists() and any(output.iterdir()):
        p.error("choose a new or empty output directory to preserve earlier evidence")
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from openwam.dataloader.curated_fixture import CuratedFixtureDataset
    from openwam.model.video_backbone.wan.preprocess import preprocess_video

    torch.set_num_threads(2)
    dataset = CuratedFixtureDataset(manifest)
    matches = [i for i, row in enumerate(dataset.rows) if row["id"] == args.clip]
    if len(matches) != 1:
        p.error(f"clip must identify one accepted dataset; found {len(matches)}")
    which = matches[0]
    reader = dataset.readers[which]
    if not 0 <= args.sample_index < len(reader):
        p.error("sample-index outside the accepted windows for this clip")
    reader._return_trace = True
    index = next(k for k, pair in enumerate(dataset.mapping) if pair == (which, args.sample_index))
    sample = dataset[index]
    frames = sample["video"]
    pixels = preprocess_video(frames, dtype=torch.bfloat16, device="cpu")
    output.mkdir(parents=True, exist_ok=True)

    def array(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    arrays = {key: array(sample[key]) for key in
              ["action", "action_mask", "proprio", "proprio_mask", "video_mask"]}
    arrays["pre_vae_pixels_float32"] = pixels.float().cpu().numpy()
    for key, value in sample["trace"].items():
        if key.startswith(("raw_action_", "raw_proprio_", "normalized_action_", "normalized_proprio_")):
            arrays[key] = array(value)
    np.savez_compressed(output / "model_inputs.npz", **arrays)
    width, height = frames[0].size
    sheet = Image.new("RGB", (width*3, (height+24)*((len(frames)+2)//3)), "#101927")
    draw = ImageDraw.Draw(sheet)
    for i, image in enumerate(frames):
        assert image.size == (width, height)
        x, y = (i % 3)*width, (i // 3)*(height+24)
        sheet.paste(image.convert("RGB"), (x, y+24))
        draw.text((x+4, y+4), f"reader frame {i}", fill="white")
    sheet.save(output / "reader_frames.png")
    display = ((pixels[0, :, 0].float().permute(1, 2, 0)+1)*127.5).clamp(0, 255)
    Image.fromarray(np.rint(display.numpy()).astype(np.uint8)).save(output / "pre_vae_frame0.png")

    def safe(x):
        if torch.is_tensor(x): return x.detach().cpu().tolist()
        if isinstance(x, (np.ndarray, np.generic)): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)

    sources = ["openwam/dataloader/utils/gripper.py", "openwam/dataloader/utils/mora_relative.py",
               "openwam/dataloader/mora_relative_lerobot.py", "openwam/dataloader/curated_fixture.py",
               "openwam/model/video_backbone/wan/preprocess.py"]
    row = dataset.rows[which]
    summary = dict(clip=args.clip, sample_index=args.sample_index, source_manifest=str(manifest),
        source_selection={key: row[key] for key in ["repo", "episode", "start", "end", "fps"]},
        source_anchor=int(row["start"]) + int(sample["trace"]["window_offset"]),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), prompt=sample["prompt"],
        contract=reader.representation.contract, trace=sample["trace"],
        gripper_slots=[9, 43], gripper_gt=arrays["action"][:, [9, 43]].tolist(),
        gripper_gt_mask=arrays["action_mask"][:, [9, 43]].tolist(),
        gripper_input=arrays["proprio"][0, [9, 43]].tolist(),
        pre_vae=dict(shape=list(pixels.shape), dtype=str(pixels.dtype), device=str(pixels.device)),
        source_hashes={name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in sources},
        scope="current production reader plus actual CPU pixel preprocessing; no VAE/model forward, training or full source-video re-decode")
    (output / "trace.json").write_text(json.dumps(summary, default=safe, ensure_ascii=False, indent=2)+"\n")
    paths = []
    for side, slot, color in [(0, 9, "#e8be43"), (1, 43, "#6dd9e7")]:
        points = []
        for k, (value, valid) in enumerate(zip(arrays["action"][:, slot], arrays["action_mask"][:, slot])):
            if valid:
                points.append(f"{25+640*k/max(1,len(arrays['action'])-1):.2f},{165-140*float(value):.2f}")
            elif points:
                paths.append(f'<polyline fill="none" stroke="{color}" points="{" ".join(points)}"/>')
                points = []
        if points:
            paths.append(f'<polyline fill="none" stroke="{color}" points="{" ".join(points)}"/>')
    doc = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>{html.escape(args.clip)}</title>
<style>body{{max-width:1100px;margin:24px auto;font:16px system-ui}}img{{max-width:100%}}pre{{white-space:pre-wrap}}svg{{background:#172436}}</style>
<h1>{html.escape(args.clip)} · 当前 reader 输入</h1><p>观测 state 输入；gripper GT 来源和时序见实际 contract。这里只执行 CPU 像素预处理，没有执行 VAE/模型 forward。</p>
<p>指令：{html.escape(str(sample['prompt']))}</p><h2>当前 reader 的完整时间帧和拼接画布</h2><img src="reader_frames.png">
<h2>实际 bf16 pre-VAE 第0帧反变换显示</h2><img src="pre_vae_frame0.png"><pre>{html.escape(str(summary['pre_vae']))}</pre>
<h2>实际 gripper GT，绝对 [0,1] 开合度</h2><p>黄色=左，青色=右；横轴=chunk step，纵轴=0…1。无监督段断开。这是数值曲线，不是相机轨迹投影。</p>
<svg viewBox="0 0 690 190">{''.join(paths)}</svg><p><a href="trace.json">当前 contract、源帧、mask、代码 hash</a> · <a href="model_inputs.npz">模型输入与 raw tensors</a></p>
<details><summary>标签定义</summary><pre>{html.escape(json.dumps(summary['contract'],default=safe,ensure_ascii=False,indent=2))}</pre></details></html>'''
    (output / "index.html").write_text(doc, encoding="utf-8")
    print(json.dumps(dict(page=str(output/"index.html"), clip=args.clip,
        valid_gripper_cells=int(arrays["action_mask"][:, [9, 43]].sum()),
        gripper_sources=reader.gripper_commands.sources if hasattr(reader,"gripper_commands")
                        else reader.representation.gripper_commands.sources), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
