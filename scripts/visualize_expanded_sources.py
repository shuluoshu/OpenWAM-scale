"""Render and publish the nine requested embodiment samples in the existing HTML."""
import argparse
import json

import numpy as np

from scripts.visualize_trajectory_videos import OUTPUT, build_report, render_sample
from scripts.visualize_wrist_videos import render_wrist

SAMPLES = ('a1_genie1', 'a1_arx_lift2', 'a1_split_aloha', 'robomind_v1_agilex',
           'robomind_v1_fr3', 'robochallenge_v2_aloha', 'robochallenge_v2_dosw1',
           'robomind_v2_agilex', 'abc_130k_yam')


def main(samples=SAMPLES, report_only=False, no_report=False):
    for name in samples:
        root = OUTPUT/'expanded_sources'/name
        if not report_only:
            render_sample(name, root)
            render_wrist(name, root, reuse_samples=True)
    if no_report:
        return
    entries = [json.loads((OUTPUT/'expanded_sources'/name/'entry.json').read_text()) for name in SAMPLES]
    coverage_path = OUTPUT/'wrist_samples/coverage.json'
    coverage = [entry for entry in json.loads(coverage_path.read_text()) if entry['sample'] not in SAMPLES]
    for entry in entries:
        root, name = OUTPUT/entry['root'], entry['family']
        metrics = json.loads((root/'trajectory_videos'/name/'metrics.json').read_text())
        assert 'wrist' in metrics['videos'], f'{name}: missing wrist video export'
        with np.load(root/'video_samples'/name/'arrays.npz') as arrays:
            tcp_available = bool(arrays['tcp_valid'].any())
        coverage.append(dict(sample=entry['id'], family=name, root=entry['root'], tcp_available=tcp_available,
            views={side: {key: view[key] for key in ['status', 'video_available', 'projection_available', 'reason']}
                   for side, view in metrics['wrist']['views'].items()}))
    (OUTPUT/'expanded_sources/manifest.json').write_text(json.dumps(dict(samples=entries), ensure_ascii=False, indent=2)+'\n')
    coverage_path.write_text(json.dumps(coverage, ensure_ascii=False, indent=2)+'\n')
    build_report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', nargs='+', choices=SAMPLES, default=SAMPLES)
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--no-report', action='store_true')
    args = parser.parse_args()
    main(args.samples, args.report_only, args.no_report)
