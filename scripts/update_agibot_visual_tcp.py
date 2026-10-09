"""Refresh cached AgiWorld reports with the CAD-informed visual TCP preview.

Uses existing decoded images, poses and per-episode FK; never writes source data
or training configuration. Run after the main report and extra samples exist.
"""
import json

import numpy as np

from scripts.data.agibot_tcp_geometry import derive_evidence
from scripts.data.tcp_endpoint_policy import apply_endpoint_policy
from scripts.visualize_tcp_projections import build_report as build_static_report
from scripts.visualize_trajectory_videos import OUTPUT, build_report, render_sample


def main():
    evidence = OUTPUT/'tcp_origin_audit/agibotworld/visual_tcp_geometry.json'
    evidence.write_text(json.dumps(derive_evidence(), indent=2, ensure_ascii=False)+'\n')
    manifest = json.loads((OUTPUT/'agibot_examples/manifest.json').read_text())
    roots = [OUTPUT, *[OUTPUT/entry['root'] for entry in manifest['samples']]]
    for root in roots:
        for kind in ['projection_samples', 'video_samples']:
            folder = root/kind/'agibotworld'
            if not (folder/'arrays.npz').exists():
                continue
            with np.load(folder/'arrays.npz') as data:
                arrays = dict(data)
            metadata = json.loads((folder/'metadata.json').read_text())
            apply_endpoint_policy('agibotworld', arrays, metadata)
            assert metadata['tcp_status'] == 'visual_tcp_preview'
            np.savez_compressed(folder/'arrays.npz', **arrays)
            (folder/'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
        render_sample('agibotworld', root)
    manifest['endpoint'] = 'A2D report-only visual TCP = canonical wrist @ TransX(+0.200m), rotation unchanged. CAD-informed and manually reviewed, not an official calibrated TCP. Old 100/128mm named centers remain only in before/after comparison.'
    (OUTPUT/'agibot_examples/manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    build_static_report()
    build_report()


if __name__ == '__main__':
    main()
