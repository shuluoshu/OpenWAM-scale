import json

from scripts.experiments.report_stage12_timed import stage2_sources, stage1_sample_path


def test_recovery_plan_does_not_replace_original_until_runtime_status_exists(tmp_path):
    recovery=tmp_path/'none_recovery';recovery.mkdir()
    (recovery/'recovery.json').write_text(json.dumps(dict(source_step=250)))
    sources,active,meta=stage2_sources(tmp_path)
    assert not active and sources['none']==tmp_path/'stage2/none'
    (recovery/'none').mkdir();(recovery/'none/status.json').write_text('{"status":"setup"}')
    sources,active,meta=stage2_sources(tmp_path)
    assert active and sources['none']==recovery/'none'
    assert sources['joint']==tmp_path/'stage2/joint'
    assert meta['source_step']==250


def test_stage1_uses_protocol_cache_and_preserves_legacy_record_fallback(tmp_path):
    record=dict(observed_image_path='/old/cache/raw_cache/case/observed.png',source_index=4,anchor_index=1)
    assert str(stage1_sample_path(tmp_path,record))=='/old/cache/samples/04_1.pt'
    (tmp_path/'stage1').mkdir()
    (tmp_path/'stage1/protocol.json').write_text(json.dumps(dict(smoke_data='/new/rgb_v2')))
    assert str(stage1_sample_path(tmp_path,record))=='/new/rgb_v2/samples/04_1.pt'
