import json
from pathlib import Path


def test_stage1_eta_and_incomplete_json(tmp_path):
    from scripts.pai.monitor_stage12 import read_progress, stage_summary
    p=tmp_path/'progress.jsonl'
    p.write_text('\n'.join(json.dumps(dict(event='update',step=i,cursor=32*i,total_samples=3200,elapsed_s=2*i,loss=1/i)) for i in range(1,11))+'\n{"event":')
    rows=read_progress(p)
    s=stage_summary(rows,total_hint=100,stage='stage1',now=1000)
    assert s['step']==10 and s['total']==100 and s['remaining_s']==180
    assert s['eta_unix']==1180 and s['latest']['loss']==.1


def test_eta_does_not_cross_resume_or_invent_unobserved_stage2():
    from scripts.pai.monitor_stage12 import stage_summary
    rows=[dict(step=20,elapsed_s=100,loss=1),dict(step=21,elapsed_s=1,loss=.8)]
    s=stage_summary(rows,total_hint=100,stage='stage1',now=1000)
    assert s['remaining_s'] is None
    assert stage_summary([],total_hint=100000,stage='stage2',now=1000)['eta_unix'] is None


def test_queue_report_and_job_scoping(tmp_path):
    from scripts.pai.monitor_stage12 import collect, message_text
    job=dict(JobId='dlc_test',DisplayName='my-training',WorkspaceId='1140257',Status='Queuing')
    s=collect(job,tmp_path,now=1000)
    text=message_text(s)
    assert 'my-training' in text and 'dlc_test' in text and '排队' in text
    assert s['eta_unix'] is None
    assert s['stage1']['rows']==[] and s['stage2']['rows']==[]


def test_stale_progress_has_no_eta(tmp_path):
    from scripts.pai.monitor_stage12 import collect
    p=tmp_path/'stage1';p.mkdir()
    (p/'progress.jsonl').write_text('\n'.join(json.dumps(dict(event='update',step=i,total_samples=3200,cursor=32*i,elapsed_s=i,wall_time=i,loss=1)) for i in range(1,10)))
    (tmp_path/'pipeline_state.json').write_text('{"phase":"stage1"}')
    s=collect(dict(JobId='dlc_test',Status='Running'),tmp_path,now=100000)
    assert s['stage1']['stale'] and s['stage1']['eta_unix'] is None


def test_preparation_gate_checks_failure_and_heartbeat(tmp_path):
    import pytest
    from scripts.pai.preparation_gate import gate_status
    marker=tmp_path/'prepared.json'
    assert not gate_status({'status':'running','updated_at':100},marker,101)
    with pytest.raises(RuntimeError,match='failed'):
        gate_status({'status':'failed'},marker,101)
    with pytest.raises(RuntimeError,match='stale'):
        gate_status({'status':'running','updated_at':100},marker,1100)
    marker.write_text('{}')
    assert gate_status({'status':'ready','updated_at':100},marker,1100)


def test_loss_plot_ignores_nonfinite_and_null_metrics():
    from scripts.pai.monitor_stage12 import loss_points
    rows=[{'step':1,'loss':.5},{'step':2,'loss':None},{'step':3,'loss':float('nan')},{'step':4,'loss':.25}]
    assert loss_points(rows,'loss')==[(1,.5),(4,.25)]
