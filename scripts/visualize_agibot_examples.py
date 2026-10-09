"""Three visually selected AgiWorld clips for the existing portable report.

Run: python -m scripts.visualize_agibot_examples
Each sample has its own dataset/episode FK audit; source data stays read-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from scripts.data.fk_audit_agibot import DATASET
from scripts.data.fk_audit_agibot import run as audit_fk
from scripts.data.video_projection_samples import decode_sequence, extract_robot
from scripts.visualize_trajectory_videos import OUTPUT, build_report, render_sample

SAMPLES = (
    dict(task=365, episode=0, anchor=180, count=300, title='整理日用品：抓取与转移',
         reason='桌面背景清楚，夹爪及被抓物体可见，左右臂先后参与转移。'),
    dict(task=362, episode=0, anchor=180, count=300, title='双臂叠短裤：协同提起与折叠',
         reason='不同于单臂擦白板；俯视布面背景，两侧夹爪参与同一操作。'),
    dict(task=410, episode=0, anchor=180, count=300, title='倒水：抓壶与倾转',
         reason='近距离夹爪与壶把接触，末端旋转明显，便于比较腕部与工具原点。'),
)


def generate(output: Path = OUTPUT):
    entries = []
    for sample in SAMPLES:
        task, episode = sample['task'], sample['episode']
        name = f'agibot-task{task}-ep{episode}'
        root = output/'agibot_examples'/name
        root.mkdir(parents=True, exist_ok=True)
        dataset = DATASET.parent.parent/f'task_{task}'/DATASET.name
        # This is an extraction-only configuration, not a training profile.
        profile = root/'source.yaml'
        OmegaConf.save(OmegaConf.create(dict(dataset_dir=str(dataset),
                      head_camera='observation.images.cam_high')), profile)
        metadata = extract_robot('agibotworld', root/'video_samples',
                                 sample['anchor'], sample['count'],
                                 profile_path=profile, episode_index=episode)
        metadata.update(clip_label=f'task{task} ep{episode}',
                        selection_policy=sample['reason'], sample_title=sample['title'],
                        requested_frame_count=sample['count'])
        folder = root/'video_samples/agibotworld'
        metadata.update(decode_sequence(metadata, folder))
        if metadata['frame_count'] != sample['count']:
            raise ValueError(f'{name}: requested clip extends beyond source episode')
        (folder/'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
        audit_fk(root/'agibotworld', dataset=dataset, episode_index=episode)
        render_sample('agibotworld', root)
        entries.append(dict(id=name, title=sample['title'], reason=sample['reason'],
                            root=str(root.relative_to(output)), **{k: sample[k] for k in ('task', 'episode', 'anchor', 'count')}))
    manifest = dict(samples=entries,
                    surveyed_tasks=[327, 356, 362, 365, 385, 410],
                    selection='Visual inspection of six evenly spaced head-camera frames per episode0, then three distinct manipulation scenes.',
                    endpoint='A2D report-only visual TCP: wrist local +X 0.200m, rotation unchanged; CAD-informed manual preview, not factory calibration. 100/128mm remain old named-center diagnostics.')
    (output/'agibot_examples/manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    build_report(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    generate(args.output)
