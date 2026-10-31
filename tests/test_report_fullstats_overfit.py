import importlib
import json
from pathlib import Path

import numpy as np


def module():
    return importlib.import_module('scripts.experiments.report_fullstats_overfit')


def put(root, name, value):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value))


def test_missing_and_partial_producers_are_pending(tmp_path):
    (tmp_path / 'supervisor.json').write_text('{"status":')
    summary = module().build_report(tmp_path)
    assert summary['overall_complete'] is False
    assert summary['training']['joint']['requested_updates'] is None
    assert summary['warnings']
    assert (tmp_path / 'report/index.html').is_file()


def test_dynamic_time_budget_actual_coverage_and_all_loss_scales(tmp_path):
    put(tmp_path, 'plan.json', {'stage1_updates': 1800, 'seconds_per_arm': 3711})
    put(tmp_path, 'data/manifest.json', {'cases': [
        {'index': 0, 'family': 'abc_130k', 'episode': 0, 'anchor': 70},
        {'index': 1, 'family': 'abc_130k', 'episode': 0, 'anchor': 80}]})
    put(tmp_path, 'stage2/joint/status.json', {'budget_seconds': 3700, 'step': 7, 'status': 'running'})
    p = tmp_path / 'stage2/joint/updates.jsonl'
    p.write_text('\n'.join(json.dumps({'step': n, 'sample': 0, 'elapsed_seconds': n * 2,
        'losses': {'total': 1, 'action': .01, 'video': .99, 'fk_position': 0, 'fk_rotation': 0, 'fk_valid_count': 0}})
        for n in (1, 2, 3)) + '\n{"step":')
    s = module().build_report(tmp_path)
    arm = s['training']['joint']
    assert arm['budget_seconds'] == 3700
    assert s['training']['none']['budget_seconds'] == 3711
    assert arm['completed_updates'] == 7
    assert arm['coverage']['unique_drawn_cases'] == 1
    assert arm['coverage']['cases'][0]['draw_count'] == 3
    assert set(arm['losses']) == {'total', 'action', 'video', 'fk_position', 'fk_rotation'}
    assert s['abc_frame70']['training']['joint'][0]['draw_count'] == 3
    assert 'time budget' in (tmp_path / 'report/index.html').read_text()


def test_training_proof_does_not_imply_six_validated_videos(tmp_path):
    put(tmp_path, 'completion.json', {'status': 'complete'})
    put(tmp_path, 'fullspan/report/coverage.json', {'completed_videos': 6})
    put(tmp_path, 'fullspan/report/validation.json', {'status': 'complete', 'verified_videos': 5})
    assert not module().build_report(tmp_path)['overall_complete']
    put(tmp_path, 'fullspan/report/validation.json', {'status': 'complete', 'verified_videos': 6})
    assert not module().build_report(tmp_path)['overall_complete']  # No video artifacts.


def test_physical_error_and_jumps_respect_distinct_masks():
    pred = np.zeros((3, 2, 3)); gt = pred.copy()
    pred[:, 0, 0] = [1, 3, 99]
    valid = [[True, False], [True, False], [False, False]]
    result = module().trajectory_metrics({'world_xyz': pred.tolist(), 'gt_world_xyz': gt.tolist(),
        'metric_valid': valid, 'valid': valid, 'metric_gt_valid': valid})
    assert result['world_error_m']['mean'] == 2
    assert result['world_error_m']['count'] == 2
    assert result['pred_adjacent_jump_m']['mean'] == 2
    assert result['gt_adjacent_jump_m']['mean'] == 0
    result = module().trajectory_metrics({'world_xyz': pred.tolist(), 'gt_world_xyz': gt.tolist(),
        'metric_valid': np.zeros((3, 2), bool).tolist(), 'valid': valid, 'metric_gt_valid': valid})
    assert result['world_error_m']['mean'] is None


def test_only_actual_final_evaluation_gets_final_label(tmp_path):
    put(tmp_path, 'stage2/joint/status.json', {'status': 'complete', 'step': 11})
    for step in (0, 7):
        put(tmp_path, f'stage2/joint/eval_{step:08}/metrics.json', {'step': step,
            'cases': [{'name': 'case_000', 'family': 'abc_130k', 'metrics': {
                'eef_delta_translation_m_mae': {'value': .02, 'valid_count': 3}}}]})
    s = module().build_report(tmp_path)
    assert s['evaluations']['joint'][-1]['phase'] == 'intermediate'
    put(tmp_path, 'stage2/joint/eval_00000011/metrics.json', {'step': 11, 'cases': []})
    assert module().build_report(tmp_path)['evaluations']['joint'][-1]['phase'] == 'final'


def complete_video_run(tmp_path):
    put(tmp_path, 'supervisor.json', {'status': 'complete'})
    put(tmp_path, 'completion.json', {'status': 'complete'})
    sequences = [{'sequence_id': str(i)} for i in range(3)]
    put(tmp_path, 'fullspan/data/manifest.json', {'sequences': sequences})
    videos, checked = [], []
    for seq in sequences:
        for mode in ('joint', 'none'):
            video = f'videos/{seq["sequence_id"]}_{mode}.mp4'
            p = tmp_path / 'fullspan/report' / video
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'test fixture; codec validation is delegated')
            videos.append(dict(sequence_id=seq['sequence_id'], mode=mode, kind='direct', video=video, sha256='verified'))
            checked.append(dict(key=seq['sequence_id']+'/'+mode, video=video, status='passed',
                verified_dependency_hashes={'report/'+video: 'verified'}))
    put(tmp_path, 'fullspan/report/coverage.json', {'completed_videos': 6, 'videos': videos})
    put(tmp_path, 'fullspan/report/validation.json', {'status': 'complete', 'verified_videos': 6, 'videos': checked})
    return checked


def test_complete_requires_current_six_video_validation_records(tmp_path):
    checked = complete_video_run(tmp_path)
    assert module().build_report(tmp_path)['overall_complete']
    checked[0]['verified_dependency_hashes'] = {'report/'+checked[0]['video']: 'stale'}
    put(tmp_path, 'fullspan/report/validation.json', {'status': 'complete', 'verified_videos': 6, 'videos': checked})
    assert not module().build_report(tmp_path)['overall_complete']


def test_selection_and_population_available_before_samples(tmp_path):
    put(tmp_path, 'selection.json', {'sequences': [{'family': 'abc', 'episode': 0, 'start': 0, 'end_exclusive': 255, 'prompt': 'test', 'fps': 30}]})
    put(tmp_path, 'plan.json', {'stage1_population_samples': 147, 'stage1_sample_fps': 3})
    put(tmp_path, 'data/preflight.json', {'cases': [{'family': 'abc_130k', 'anchor': 70, 'trainable': True, 'old': {'clipped_fraction': .5}, 'new': {'clipped_fraction': .1}}]})
    s = module().build_report(tmp_path)
    assert s['sequences'][0]['family'] == 'abc'
    assert s['training']['stage1']['coverage']['available_training_cases'] == 147
    assert s['abc_frame70']['clipping'][0]['new']['clipped_fraction'] == .1
    assert '0.5' in (tmp_path / 'report/index.html').read_text()


def test_live_stage1_status_and_fk_counts_are_not_loss_metrics(tmp_path):
    p = tmp_path / 'stage1/progress.jsonl'; p.parent.mkdir()
    p.write_text(json.dumps({'event': 'update', 'step': 3, 'wall_time': 20, 'loss': .1, 'fm_loss': .1, 'tcp_loss': 0, 'valid_tcp_points': 0, 'family_counts': {'abc_130k': 1}})+'\n')
    q = tmp_path / 'stage2/joint/updates.jsonl'; q.parent.mkdir(parents=True)
    q.write_text(json.dumps({'step': 1, 'losses': {'fk_position': 0, 'fk_valid_count': 0}})+'\n')
    s = module().build_report(tmp_path)
    assert s['training']['stage1']['status'] == 'running'
    assert s['training']['joint']['supervision_counts']['fk_valid_count']['sum'] == 0
    assert 'fk_valid_count' not in s['training']['joint']['losses']


def test_failed_supervisor_prevents_completion_after_six_validated_videos(tmp_path):
    complete_video_run(tmp_path)
    put(tmp_path, 'supervisor.json', {'status': 'failed', 'error': 'projection coverage failed'})
    result = module().build_report(tmp_path)
    assert result['training_complete']
    assert result['verified_videos'] == 6
    assert not result['overall_complete']
    assert result['status'] == 'failed'
    assert result['supervisor_error'] == 'projection coverage failed'
    page = (tmp_path / 'report/index.html').read_text()
    assert 'projection coverage failed' in page
    assert 'http-equiv="refresh"' in page


def test_present_delivery_check_must_pass(tmp_path):
    complete_video_run(tmp_path)
    put(tmp_path, 'report/delivery_check.json', {'all_passed': False, 'errors': ['missing video']})
    assert not module().build_report(tmp_path)['overall_complete']
    put(tmp_path, 'report/delivery_check.json', {'all_passed': True, 'errors': []})
    assert module().build_report(tmp_path)['overall_complete']
    (tmp_path / 'report/delivery_check.json').write_text('{"all_passed":')
    assert not module().build_report(tmp_path)['overall_complete']
