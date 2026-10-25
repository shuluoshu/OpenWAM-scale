import json

from scripts.pai.monitor_stage12 import collect, message_text


def test_queued_prepared_job_reports_known_population_before_first_loss(tmp_path):
    (tmp_path / 'stage1').mkdir()
    (tmp_path / 'stage1/population.json').write_text(json.dumps({
        'complete': True, 'total_samples': 321,
    }))
    summary = collect({'JobId': 'example', 'Status': 'Queuing'}, tmp_path, now=100)
    assert summary['stage1']['step'] == 0
    assert summary['stage1']['total'] == 11
    assert summary['stage1']['latest'] == {}
    assert summary['stage1']['eta_unix'] is None
    assert 'Stage1：0/11 步' in message_text(summary)
