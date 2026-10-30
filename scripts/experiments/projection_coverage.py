"""Summarize actual GT visibility separately from video/overlay arithmetic tests."""
import argparse
import collections
import html
import json
from pathlib import Path


def classify(*, available, calibrated, gt, out_of_frame):
    if not available:
        return 'no_camera'
    if not calibrated:
        return 'calibration_unavailable'
    if not gt:
        return 'all_gt_outside_view' if out_of_frame else 'no_projectable_gt'
    return 'partial' if calibrated < available else 'visible'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(); report = root / 'report'
    manifest = json.loads((root / 'data/manifest.json').read_text())
    results = []
    for sequence in manifest['sequences']:
        anchors = [r for r in manifest['anchors'] if r['sequence_id'] == sequence['sequence_id']]
        for role in ('head', 'left_wrist', 'right_wrist'):
            row = dict(sequence_id=sequence['sequence_id'], family=sequence['family'], episode=sequence['episode'],
                       role=role, anchors=len(anchors), image_frames=0, camera_frames=0,
                       gt_world_points=0, gt_projectable_points=0, gt_visible_points=0,
                       pred_visible_points=0, gt_outside_points=0, reasons=collections.Counter())
            for anchor in anchors:
                path = report / 'coordinates' / f"joint_{anchor['id']:06d}.json"
                view = json.loads(path.read_text())['views'][role]; branch = view['direct']
                row['image_frames'] += bool(view['image_available'])
                row['camera_frames'] += bool(view['projection_available'])
                for key, field in [('gt_world_points', 'gt_valid'), ('gt_projectable_points', 'gt_projection_valid'),
                                   ('gt_visible_points', 'gt_in_frame'), ('pred_visible_points', 'pred_in_frame')]:
                    row[key] += sum(bool(p) for pair in branch[field] for p in pair)
                row['reasons'][view.get('reason') or 'available'] += 1
            row['gt_outside_points'] = row['gt_projectable_points'] - row['gt_visible_points']
            row['status'] = classify(available=row['image_frames'], calibrated=row['camera_frames'],
                                     gt=row['gt_visible_points'], out_of_frame=row['gt_outside_points'])
            row['reasons'] = dict(row['reasons']); results.append(row)
    result = dict(schema='projection-coverage-v1', status='complete',
                  caveat='Visibility and renderer arithmetic do not certify camera calibration or physical TCP endpoint.', views=results)
    (report / 'projection_coverage.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(row[k]))}</td>' for k in
                   ('family', 'episode', 'role', 'status', 'image_frames', 'camera_frames',
                    'gt_world_points', 'gt_visible_points', 'gt_outside_points', 'reasons')) + '</tr>' for row in results)
    (report / 'projection_coverage.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>GT 投影覆盖</title><style>body{font:16px system-ui;margin:2em}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}</style><h1>实际 GT 投影覆盖</h1><p>按 episode、相机统计；点数按各预测窗口累计，可能重复。可见及数值复算通过，不代表相机物理标定和 TCP 端点已验收。</p><table><tr><th>数据集</th><th>episode</th><th>视角</th><th>状态</th><th>有图帧</th><th>可投影帧</th><th>GT世界点</th><th>画面内GT点</th><th>画面外GT点</th><th>原因（采样帧数）</th></tr>''' + rows + '</table><p><a href="projection_coverage.json">机器可读记录</a></p></html>')
    print(json.dumps(dict(views=len(results), status_counts=dict(collections.Counter(r['status'] for r in results)))))


if __name__ == '__main__':
    main()
