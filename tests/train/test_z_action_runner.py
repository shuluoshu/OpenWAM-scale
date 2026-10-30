import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('runner',ROOT/'scripts/compare_z_action_stage2.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_matched_order_is_complete_reproducible_and_changes_epoch():
    a=runner.shuffled_order(24,42,0)
    assert sorted(a)==list(range(24))
    assert a==runner.shuffled_order(24,42,0)
    assert a!=runner.shuffled_order(24,42,1)


def test_index_retains_all_old_snapshots_and_explicit_missing_video(tmp_path):
    from openwam.train.z_action_report import atomic_json,render
    for step in (0,500):
        atomic_json(tmp_path/'none'/f'eval_{step:08d}'/'metrics.json',dict(step=step,elapsed_seconds=step,cases=[dict(name='case_000',family='robot',metrics={'action_mae':dict(value=.2,valid_count=20)})]))
    first=(tmp_path/'none/eval_00000000/metrics.json').read_bytes()
    render(tmp_path)
    text=(tmp_path/'index.html').read_text()
    assert 'update 0,' in text and 'update 500,' in text
    assert 'not applicable (action-only)' in text
    assert 'action_curves.svg' in text
    assert (tmp_path/'none/eval_00000000/metrics.json').read_bytes()==first


def test_family_balancing_does_not_drop_families_or_cases():
    samples=[{'_eval':{'family':name}} for name in ['a','b','b','b']]
    order=runner.balanced_order(samples,42,0)
    assert sum(samples[i]['_eval']['family']=='a' for i in order)==3
    assert sum(samples[i]['_eval']['family']=='b' for i in order)==3
    assert set(order)==set(range(4))


def test_preload_validation_rejects_changed_normalization(tmp_path):
    import argparse
    import pytest
    from omegaconf import OmegaConf
    source=tmp_path/'selection.yaml';source.write_text('repos: []')
    registry=tmp_path/'registry.json';registry.write_text('{}')
    goal=tmp_path/'goal.safetensors';goal.write_bytes(b'goal')
    baseline=tmp_path/'baseline';baseline.mkdir()
    manifest=dict(source_yaml=str(source),source_sha256=runner.sha(source),normalization_registry=str(registry),normalization_sha256=runner.sha(registry),action_horizon=32,video_stride=4,video_frames=9)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    cfg=OmegaConf.create(dict(dataloader=dict(source_yaml=str(source),normalization_registry=str(registry)),goal=dict(checkpoint=str(goal)),training=dict(finetune_ckpt_path=str(baseline))))
    args=argparse.Namespace(samples=str(tmp_path/'samples.pt'))
    assert runner.validate_inputs(cfg,args)==manifest
    registry.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='identity mismatch'):runner.validate_inputs(cfg,args)


def test_rerun_preserves_completed_status_before_reading_inputs(tmp_path,monkeypatch):
    import pytest
    for arm in ('joint',None):
        marker=tmp_path/arm/'status.json' if arm else tmp_path/'supervisor.json'
        marker.parent.mkdir(exist_ok=True)
        marker.write_text('{"status":"complete","evidence":"keep"}')
        before=marker.read_bytes()
        argv=['runner','--config','missing-config','--samples','missing-samples','--output',str(tmp_path)]
        if arm:argv+=['--arm',arm]
        monkeypatch.setattr(runner.sys,'argv',argv)
        with pytest.raises(FileExistsError):runner.main()
        assert marker.read_bytes()==before
