"""CPU-only live report for the full-statistics, time-budgeted overfit run.

Reads producer artifacts; never loads model weights or changes training inputs.
Use the entire RUN as the offline delivery root (videos live in fullspan/report).
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import time

import numpy as np


EXPECTED_VIDEOS = 6


def read_json(path, warnings=None):
    try:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError('expected JSON object')
        return value
    except FileNotFoundError:
        return {}
    except (ValueError, OSError) as error:
        if warnings is not None:
            warnings.append(f'{Path(path).name}: producer artifact unavailable: {error}')
        return {}


def read_events(path, warnings):
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return []
    rows = []
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except ValueError:
            warnings.append(f'{Path(path).parent.name}/{Path(path).name}:{number}: incomplete/invalid event ignored')
    return rows


def write_atomic(path, value):
    temp = path.with_suffix(path.suffix + '.pending')
    temp.write_text(value)
    temp.replace(path)


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def first(*values):
    return next((v for v in values if v is not None), None)


def loss_series(rows, stage):
    result = {}
    for row in rows:
        values = row if stage == 'stage1' else row.get('losses', {})
        for key, value in values.items():
            if stage == 'stage1' and key not in ('loss', 'fm_loss', 'tcp_loss'):
                continue
            if key.endswith('_count') or not numeric(value):
                continue
            result.setdefault(key, []).append([row['step'], float(value)])
    return result


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return dict(count=int(values.size), mean=float(values.mean()) if values.size else None,
                maximum=float(values.max()) if values.size else None,
                sum=float(values.sum()))


def trajectory_metrics(branch):
    """Score world-space direct trajectories once, independently of camera visibility.

    metric_valid already excludes source/native endpoint disagreements. Prediction
    jumps need prediction validity; GT jumps require native metric GT validity.
    Adjacent means within one 32-step chunk and the same hand, never across anchors.
    """
    empty = {key: stats([]) for key in ('world_error_m', 'pred_adjacent_jump_m', 'gt_adjacent_jump_m')}
    if not branch or 'world_xyz' not in branch or 'gt_world_xyz' not in branch:
        return empty
    pred = np.asarray(branch['world_xyz'], dtype=float)
    gt = np.asarray(branch['gt_world_xyz'], dtype=float)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        return empty
    shape = pred.shape[:-1]
    mask = lambda key: np.asarray(branch.get(key, np.zeros(shape, bool)), dtype=bool)
    pv = mask('valid') & np.isfinite(pred).all(-1)
    gv = mask('metric_gt_valid') & np.isfinite(gt).all(-1)
    mv = mask('metric_valid') & np.isfinite(pred).all(-1) & np.isfinite(gt).all(-1)
    return dict(world_error_m=stats(np.linalg.norm(pred - gt, axis=-1)[mv]),
                pred_adjacent_jump_m=stats(np.linalg.norm(np.diff(pred, axis=0), axis=-1)[pv[1:] & pv[:-1]]),
                gt_adjacent_jump_m=stats(np.linalg.norm(np.diff(gt, axis=0), axis=-1)[gv[1:] & gv[:-1]]))


def collect_trajectories(root, manifest, warnings):
    rows = []
    for record in manifest.get('anchors', []):
        for arm in ('joint', 'none'):
            path = root / 'fullspan/report/coordinates' / f'{arm}_{record["id"]:06d}.json'
            coordinate = read_json(path, warnings)
            if not coordinate:
                continue
            if coordinate.get('anchor') != record.get('anchor'):
                warnings.append(f'{path.name}: coordinate anchor mismatch; metrics excluded')
                continue
            branch = coordinate.get('views', {}).get('head', {}).get('direct', {})
            rows.append(dict(arm=arm, id=record['id'], sequence_id=record.get('sequence_id'),
                family=record.get('family'), episode=record.get('episode'), anchor=record['anchor'],
                endpoint=coordinate.get('endpoint', {}), metrics=trajectory_metrics(branch)))
    return rows


def draw_coverage(rows, cases):
    counts = Counter(str(row['sample']) for row in rows if row.get('sample') is not None)
    mapped = [{**{key: case.get(key) for key in ('index', 'family', 'repo', 'episode', 'anchor', 'prompt', 'valid_action_elements')},
               'draw_count': counts[str(case['index'])]} for case in cases if 'index' in case]
    known = {str(case['index']) for case in cases if 'index' in case}
    return dict(available_training_cases=len(cases), logged_draws=sum(counts.values()),
                unique_drawn_cases=len(counts), cases=mapped,
                unknown_case_ids=sorted(set(counts) - known))


def svg_plot(path, title, pairs):
    width, height = 370, 180
    x, y = np.asarray(pairs, dtype=float).T
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo or max(abs(hi) * .1, 1e-8)
    points = ' '.join(f'{48 + 310 * (a - x.min()) / max(1, x.max() - x.min()):.2f},{140 - 105 * (b - lo) / span:.2f}' for a, b in pairs)
    content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="11" fill="#26354c">
<text x="12" y="18">{html.escape(title)}</text><text x="4" y="40">{hi:.3g}</text><text x="4" y="143">{lo:.3g}</text>
<path d="M48 30V145H360" fill="none" stroke="#aaa"/><polyline points="{points}" fill="none" stroke="#2465b2" stroke-width="1"/>
<text x="48" y="162">{int(x.min())}</text><text x="307" y="162">{int(x.max())}</text><text x="140" y="178">optimizer update</text></g></svg>'''
    write_atomic(path, content)


def build_report(root, prior_run=None):
    root = Path(root).resolve()
    out = root / 'report'
    out.mkdir(parents=True, exist_ok=True)
    warnings = []
    read = lambda name: read_json(root / name, warnings)
    plan, supervisor, proof = read('plan.json'), read('supervisor.json'), read('completion.json')
    selection, manifest = read('selection.json'), read('data/manifest.json')
    fullspan = read('fullspan/data/manifest.json')
    teacher, s1summary = read('stage1_identity.json'), read('stage1/summary.json')
    all_s1 = read_events(root / 'stage1/progress.jsonl', warnings)
    startup = next((r for r in all_s1 if r.get('event') == 'started'), {})
    population = s1summary.get('scope', {}).get('finite_epoch', {}) or read('stage1_population.json')
    s1identity = first(s1summary.get('identity'), startup.get('identity'), {})
    training, evaluations = {}, {}
    for stage in ('stage1', 'joint', 'none'):
        rows = ([r for r in all_s1 if r.get('event') == 'update'] if stage == 'stage1'
                else read_events(root / f'stage2/{stage}/updates.jsonl', warnings))
        rows = list({r['step']: r for r in rows if numeric(r.get('step'))}.values())
        rows.sort(key=lambda r: r['step'])
        status = s1summary if stage == 'stage1' else read(f'stage2/{stage}/status.json')
        stageplan = plan.get(stage, {})
        if not isinstance(stageplan, dict):
            stageplan = {}
        requested = (first(status.get('requested_updates'), s1identity.get('training', {}).get('max_updates'), plan.get('stage1_updates'))
                     if stage == 'stage1' else first(status.get('requested_updates'), stageplan.get('max_updates')))
        elapsed = max([float(r.get('elapsed_seconds', r.get('elapsed_s', 0))) for r in rows] + [float(status.get('elapsed_seconds', 0))])
        updates = max([r['step'] for r in rows] + [status.get('step', status.get('completed_steps', 0))])
        losses = loss_series(rows, stage)
        coverage = draw_coverage(rows, manifest.get('cases', [])) if stage != 'stage1' else dict(
            available_training_cases=first(population.get('total_samples'), s1identity.get('total_samples'), plan.get('stage1_population_samples')),
            logged_draws=sum(sum(r.get('family_counts', {}).values()) for r in rows),
            family_draws=dict(sum((Counter(r.get('family_counts', {})) for r in rows), Counter())),
            sample_fps=first(population.get('sample_fps'), plan.get('stage1_sample_fps')),
            unique_drawn_cases=None, note='Per-anchor IDs are not logged by Stage1; exact unique draw coverage unavailable.')
        wall_times = [r['wall_time'] for r in all_s1 if numeric(r.get('wall_time'))] if stage == 'stage1' else []
        count_values = {}
        for row in rows:
            values = row if stage == 'stage1' else row.get('losses', {})
            for key, value in values.items():
                if (key.endswith('_count') or key == 'valid_tcp_points') and numeric(value):
                    count_values.setdefault(key, []).append(value)
        training[stage] = dict(status=status.get('status', 'running' if rows else 'pending'), completed_updates=updates,
            requested_updates=requested, budget_seconds=first(status.get('budget_seconds'), stageplan.get('budget_seconds'),
                plan.get('seconds_per_arm') if stage != 'stage1' else plan.get('stage1_expected_seconds')),
            elapsed_seconds=elapsed, event_wall_span_seconds=max(wall_times)-min(wall_times) if wall_times else None,
            coverage=coverage, supervision_counts={k: stats(v) for k,v in count_values.items()}, losses={k: dict(stats([p[1] for p in v]), first=v[0][1], last=v[-1][1]) for k,v in losses.items()},
            loss_plots=[])
        for index, (key, pairs) in enumerate(losses.items()):
            name = f'loss_{stage}_{index}.svg'
            svg_plot(out / name, f'{stage}: {key}', pairs)
            training[stage]['loss_plots'].append(name)
        if stage != 'stage1':
            evaluations[stage] = []
            for folder in sorted((root / 'stage2' / stage).glob('eval_*')):
                value = read_json(folder / 'metrics.json', warnings)
                if not numeric(value.get('step')):
                    continue
                value['phase'] = ('initial' if value['step'] == 0 else 'final'
                    if status.get('status') == 'complete' and value['step'] == updates else 'intermediate')
                evaluations[stage].append(value)
    video_coverage, validation = read('fullspan/report/coverage.json'), read('fullspan/report/validation.json')
    videos = []
    for video in video_coverage.get('videos', []):
        path = (root / 'fullspan/report' / video.get('video', '')).resolve()
        if path.is_relative_to(root / 'fullspan/report') and path.is_file():
            videos.append({k: video.get(k) for k in ('sequence_id', 'family', 'episode', 'mode', 'kind', 'video', 'fps', 'frame_count', 'sha256')})
    unique_videos = {(v['sequence_id'], v['mode']) for v in videos if v['kind'] == 'direct'}
    expected_pairs = {(s['sequence_id'], arm) for s in fullspan.get('sequences', []) for arm in ('joint', 'none')}
    checks = {v.get('key'): v for v in validation.get('videos', [])}
    current_checks = all(
        checks.get(v['sequence_id'] + '/' + v['mode'], {}).get('status') == 'passed'
        and checks[v['sequence_id'] + '/' + v['mode']].get('video') == v['video']
        and checks[v['sequence_id'] + '/' + v['mode']].get('verified_dependency_hashes', {}).get('report/' + v['video']) == v['sha256']
        for v in videos if v['kind'] == 'direct')
    delivery_check = read('report/delivery_check.json')
    delivery_passed = not (out / 'delivery_check.json').exists() or delivery_check.get('all_passed') is True
    complete = (supervisor.get('status') == 'complete' and delivery_passed and proof.get('status') == 'complete' and video_coverage.get('completed_videos') == EXPECTED_VIDEOS
                and len(unique_videos) == EXPECTED_VIDEOS and unique_videos == expected_pairs and current_checks and validation.get('status') == 'complete'
                and validation.get('verified_videos') == EXPECTED_VIDEOS)
    camera_limitations = sorted({(video.get('family', ''), role, view['reason'])
        for video in video_coverage.get('videos', []) for row in video.get('rows', [])
        for role, view in row.get('views', {}).items() if view.get('reason')})
    trajectories = collect_trajectories(root, fullspan, warnings)
    # Preflight is an explicit producer artifact; absence never implies zero clipping.
    preflight = read('data/preflight.json')
    abc = dict(training={arm: [c for c in training[arm]['coverage']['cases']
               if str(c.get('family', '')).startswith('abc') and c.get('anchor') == 70] for arm in ('joint', 'none')},
               trajectories=[r for r in trajectories if r['anchor'] == 70 and str(r.get('family', '')).startswith('abc')],
               clipping=[r for r in preflight.get('cases', []) if r.get('anchor') == 70 and str(r.get('family', '')).startswith('abc')])
    if prior_run:
        prior = Path(prior_run).resolve()
        prior_manifest = read_json(prior / 'data/manifest.json', warnings)
        prior_fullspan = read_json(prior / 'fullspan/data/manifest.json', warnings)
        prior_fullspan = dict(prior_fullspan, anchors=[r for r in prior_fullspan.get('anchors', [])
            if r.get('anchor') == 70 and str(r.get('family', '')).startswith('abc')])
        abc['prior_trajectories'] = collect_trajectories(prior, prior_fullspan, warnings)
        abc['prior_training'] = {}
        for arm in ('joint', 'none'):
            prior_draws = read_events(prior / f'stage2/{arm}/updates.jsonl', warnings)
            abc['prior_training'][arm] = [c for c in draw_coverage(prior_draws, prior_manifest.get('cases', []))['cases']
                if c.get('anchor') == 70 and str(c.get('family', '')).startswith('abc')]
        abc['prior_note'] = 'No matching case means this exact anchor was not in the prior fixed training case manifest; not a zero physical error.'
    result = dict(schema='openwam.fullstats_overfit_report.v1', updated_unix=time.time(), overall_complete=complete,
        status='complete' if complete else supervisor.get('status', 'pending'), phase=supervisor.get('phase'),
        supervisor_error=supervisor.get('error'), delivery_check=delivery_check,
        training_complete=proof.get('status') == 'complete', training=training, evaluations=evaluations,
        stage1_evaluations=s1summary.get('evaluations', []), teacher=teacher,
        normalization_sha256=manifest.get('normalization_sha256'), selection=selection,
        sequences=fullspan.get('sequences', []) or selection.get('sequences', []), videos=videos, expected_videos=EXPECTED_VIDEOS,
        completed_videos=video_coverage.get('completed_videos', 0), verified_videos=validation.get('verified_videos', 0),
        validation_status=validation.get('status', 'pending'), preflight=preflight, trajectories=trajectories,
        abc_frame70=abc, camera_limitations=camera_limitations, warnings=warnings, prior_run=str(prior_run) if prior_run else None)
    if not complete and result['status'] == 'complete':
        result['status'] = 'awaiting_video_validation'
    render_html(root, result)
    write_atomic(out / 'summary.json', json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    return result


def render_html(root, result):
    esc = lambda v: html.escape(str(v))
    fmt = lambda v: 'pending / unavailable' if v is None else f'{v:.6g}' if isinstance(v, float) else esc(v)
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Full-statistics Stage1 / Stage2 overfit</title><style>body{font:15px system-ui;max-width:1200px;margin:24px auto;padding:0 16px;line-height:1.5;color:#203046}table{border-collapse:collapse;width:100%;margin-bottom:18px}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:left}code{overflow-wrap:anywhere}video{width:100%;max-height:550px}.plots{display:flex;flex-wrap:wrap}.plots img{width:370px;max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#165fa5}.pending{background:#fff1ce;padding:12px}</style>',
        '<h1>Full-statistics Stage1 / Stage2 overfit</h1>',
        f'<p class="pending"><strong>{esc(result["status"])}</strong> · phase: {esc(result["phase"])} · training complete: {result["training_complete"]} · videos {result["completed_videos"]}/{EXPECTED_VIDEOS}, validated {result["verified_videos"]}/{EXPECTED_VIDEOS}.</p>',
        '<p>Completed training does not establish successful overfit. Missing results remain pending. Scores apply only to these selected training instructions; they do not measure generalization.</p>']
    if result['supervisor_error']:
        parts.append('<p class="pending">Supervisor error: ' + esc(result['supervisor_error']) + '</p>')
    if not result['overall_complete']:
        parts.append('<meta http-equiv="refresh" content="60">')
    def table(headers, rows):
        parts.append('<table><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + '</tr>')
        for row in rows:
            parts.append('<tr>' + ''.join(f'<td>{fmt(v)}</td>' for v in row) + '</tr>')
        parts.append('</table>')
    parts.append('<h2>Actual work and budgets</h2>')
    table(['Stage', 'Updates', 'Requested updates', 'Budget / Stage1 estimate (s)', 'Logged elapsed (s)', 'State'],
        [[k, v['completed_updates'], v['requested_updates'], v['budget_seconds'], v['elapsed_seconds'], v['status']] for k,v in result['training'].items()])
    parts.append('<p>Stage2 stops by its worker time budget; its update count is an outcome. Worker elapsed includes in-worker evaluation. Stage1 is update-bounded; event wall span includes logged checkpoint/report work. Unrecorded process startup is not inferred.</p>')
    table(['Stage', 'Available cases', 'Logged optimizer draws', 'Unique drawn cases', 'Stage1 sample FPS', 'Event wall span (s)'],
        [[k,v['coverage']['available_training_cases'],v['coverage']['logged_draws'],v['coverage']['unique_drawn_cases'],v['coverage'].get('sample_fps'),v['event_wall_span_seconds']] for k,v in result['training'].items()])
    parts.append('<p>Stage1 exact unique-anchor draw coverage is unavailable because its update events contain family counts, not anchor IDs. Stage2 draw counts use the actual logged sample IDs.</p>')
    parts.append(f'<p>Frozen teacher SHA-256: <code>{esc(result["teacher"].get("sha256", "pending"))}</code>; frozen for both arms: {esc(result["teacher"].get("frozen_for_both_arms", "pending"))}<br>Normalization registry SHA-256: <code>{esc(result["normalization_sha256"] or "pending")}</code></p>')
    parts.append('<h2>Full instruction GT / prediction videos</h2><p>Every 10 source frames plus the terminal frame. Two rows (GT / prediction), four columns (current head, current left/right wrists, Stage1 target keyframe). Unverified camera calibration does not become trusted through low numerical error.</p>')
    table(['Dataset', 'Episode', 'Range [start,end)', 'Source FPS', 'Instruction'],
          [[s.get('family'),s.get('episode'),f'{s.get("start")}–{s.get("end_exclusive")}',s.get('fps'),s.get('prompt')] for s in result['sequences']])
    for video in result['videos']:
        path = '../fullspan/report/' + video['video']
        parts.append(f'<p>{esc(video["family"])} · {esc(video["mode"])} · {fmt(video["fps"])} encoded FPS · {fmt(video["frame_count"])} frames</p><video controls preload="metadata" src="{esc(path)}"></video>')
    if result['camera_limitations']:
        table(['Dataset', 'Camera', 'Recorded projection limitation'], result['camera_limitations'])
    if not result['videos']:
        parts.append('<p>All six videos pending.</p>')
    parts.append('<h2>Losses: independent linear scales</h2><p>Unsmoothed optimizer-event values; separate scales expose action flow matching, video, FK, and total loss magnitudes. Zero FK loss with zero valid counts is not evidence of accurate FK.</p><div class="plots">')
    for stage in result['training'].values():
        for plot in stage['loss_plots']:
            parts.append(f'<img src="{esc(plot)}" alt="Loss curve with independent scale">')
    parts.append('</div>')
    table(['Stage', 'Loss', 'First', 'Last', 'Mean', 'Maximum'],
          [[stage,key,v['first'],v['last'],v['mean'],v['maximum']] for stage,item in result['training'].items() for key,v in item['losses'].items()])
    table(['Stage', 'Supervision counter', 'Logged sum', 'Mean per logged update'],
          [[stage,key,v['sum'],v['mean']] for stage,item in result['training'].items() for key,v in item['supervision_counts'].items()])
    parts.append('<h2>Real generation: initial / intermediate / final</h2><p>Physical EEF translation MAE uses denormalized valid native action components (m); it is distinct from world-trajectory Euclidean distance. Final is shown only when the evaluation step equals the completed worker step.</p>')
    table(['Arm', 'Phase / update', 'Case / dataset', 'EEF translation MAE (m)', 'Valid components'],
        [[arm,f'{ev["phase"]} / {ev["step"]}',f'{c.get("name")} / {c.get("family")}',
          c.get('metrics', {}).get('eef_delta_translation_m_mae', {}).get('value'),
          c.get('metrics', {}).get('eef_delta_translation_m_mae', {}).get('valid_count')]
         for arm,evs in result['evaluations'].items() for ev in evs for c in ev.get('cases', [])])
    table(['Stage1 update', 'Group', 'Image MAE (0–255)', 'TCP error (px)', 'Valid TCP points'],
          [[ev.get('step'),key,val.get('image_mae_uint8'),val.get('tcp_pixel_error'),val.get('valid_points')]
           for ev in result['stage1_evaluations'] for key,val in ev.get('group_metrics', {}).items()])
    parts.append('<h2>Fullspan world trajectories and adjacent-step jumps</h2><p>Each coordinate sidecar is counted once, independently of camera visibility. World error uses metric_valid; jumps are within a chunk, adjacent source steps, same hand. Endpoint arithmetic is not physical contact-center certification. No image smoothing is applied to these metrics.</p>')
    grouped = {}
    for row in result['trajectories']:
        grouped.setdefault((row['arm'],row['sequence_id']), []).append(row)
    def combined(rows, key):
        count = sum(r['metrics'][key]['count'] for r in rows)
        return sum(r['metrics'][key]['sum'] for r in rows) / count if count else None
    table(['Arm', 'Sequence', 'Anchors with coordinates', 'World error mean (m)', 'Pred adjacent jump mean (m)', 'GT adjacent jump mean (m)'],
          [[arm,seq,len(rows),*(combined(rows,key) for key in ('world_error_m','pred_adjacent_jump_m','gt_adjacent_jump_m'))] for (arm,seq),rows in grouped.items()])
    parts.append('<h2>ABC frame 70: sampling, clipping, physical error</h2>')
    table(['Arm', 'Training case', 'Actual draws', 'Source anchor'],
          [[arm,c['index'],c['draw_count'],c['anchor']] for arm, cases in result['abc_frame70']['training'].items() for c in cases])
    table(['Arm', 'World error mean (m)', 'Valid point count', 'Pred adjacent jump mean (m)'],
          [[r['arm'],r['metrics']['world_error_m']['mean'],r['metrics']['world_error_m']['count'],r['metrics']['pred_adjacent_jump_m']['mean']] for r in result['abc_frame70']['trajectories']])
    if result['prior_run']:
        parts.append('<p>Prior run, ABC frame 70: exact anchor comparison; teacher and statistics differ.</p>')
        table(['Prior arm', 'Case present in training manifest', 'Actual draws at anchor 70'],
            [[arm,bool(cases),sum(c['draw_count'] for c in cases)] for arm,cases in result['abc_frame70'].get('prior_training', {}).items()])
        table(['Prior arm', 'World error mean (m)', 'Pred adjacent jump mean (m)'],
            [[r['arm'],r['metrics']['world_error_m']['mean'],r['metrics']['pred_adjacent_jump_m']['mean']] for r in result['abc_frame70'].get('prior_trajectories', [])])
    parts.append('<h2>Quantile clipping floor: old / full registry</h2><p>Clipping preflight measures representation loss before model inference. A missing clipping record means unavailable, never a zero floor.</p>')
    if result['preflight']:
        def clipping_table(cases):
            rows = []
            for case in cases:
                for version in ('old', 'new'):
                    value = case.get(version, {})
                    tcp = value.get('tcp_position_roundtrip_m', {})
                    rows.append([case.get('family'),case.get('anchor'),case.get('trainable'),version,
                        value.get('clipped_fraction'),tcp.get('local_left', {}).get('mean'),tcp.get('local_right', {}).get('mean')])
            table(['Dataset', 'Anchor', 'Trainable', 'Registry', 'Clipped fraction', 'Local L TCP floor (m)', 'Local R TCP floor (m)'], rows)
        clipping_table(result['abc_frame70']['clipping'])
        parts.append('<details><summary>All selected anchors: clipping and TCP roundtrip floor</summary>')
        clipping_table(result['preflight'].get('cases', []))
        parts.append('</details><p>Per-dimension masks and head/local TCP floor counts are in the preflight JSON. Local TCP floor is Euclidean translation error; it is not directly comparable with component MAE.</p>')
    else:
        parts.append('<p>Clipping comparison pending.</p>')
    parts.append('<h2>Evidence</h2>')
    for relative,label in [('fullspan/report/index.html','Fullspan video list'),('fullspan/report/projection_coverage.html','Per-camera projection coverage'),('fullspan/report/validation.json','Full-video validation'),('data/preflight.json','Clipping preflight'),('data/manifest.json','Training cases / registry identity'),('selection.json','Selected instructions'),('plan.json','Budgets'),('stage1_identity.json','Teacher identity'),('completion.json','Training completion proof')]:
        if (root / relative).is_file():
            parts.append(f'<p><a href="../{relative}">{label}</a></p>')
    if result['warnings']:
        parts.append('<details><summary>Producer artifacts still being written / diagnostics</summary><pre>' + esc('\n'.join(result['warnings'])) + '</pre></details>')
    parts.append('<p><a href="summary.json">All metrics, per-case draw counts, identities, and pending states (JSON)</a></p></html>')
    write_atomic(root / 'report/index.html', '\n'.join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prior-run', type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=60)
    args = parser.parse_args()
    while True:
        summary = build_report(args.run, args.prior_run)
        print(json.dumps({k: summary[k] for k in ('status', 'overall_complete', 'completed_videos', 'verified_videos')}), flush=True)
        if not args.watch or summary['overall_complete']:
            break
        time.sleep(max(1, args.interval))


if __name__ == '__main__':
    main()
