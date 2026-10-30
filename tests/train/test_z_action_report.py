from openwam.train.z_action_report import render,atomic_json,_metric


def test_empty_missing_and_malformed_snapshots_are_not_conclusions(tmp_path):
    atomic_json(tmp_path/'joint/eval_00000000/metrics.json',{'cases':[]})
    broken=tmp_path/'none/eval_00000500/metrics.json';broken.parent.mkdir(parents=True);broken.write_text('{')
    render(tmp_path)
    page=(tmp_path/'index.html').read_text()
    assert '实验尚未全部完成' in page
    assert '<h2>实测结论</h2>' not in page
    assert 'eval_00000500/metrics.json' in page


def test_native_aggregate_zero_is_valid_and_masked_is_absent():
    assert _metric({'aggregate':{'action_mae':{'value':0.,'valid_count':8}}})==0
    assert _metric({'aggregate':{'action_mae':{'value':0.,'valid_count':0}}}) is None


def test_complete_conclusion_requires_valid_matched_update_and_images(tmp_path):
    for mode,value in [('joint',.4),('none',.2)]:
        atomic_json(tmp_path/mode/'status.json',{'status':'complete','step':500})
        metric={'value':value,'sum':value*5,'valid_count':5}
        folder=tmp_path/mode/'eval_00000500'
        atomic_json(folder/'metrics.json',{'step':500,'elapsed_seconds':100,'aggregate':{'action_mae':metric},'cases':[{'name':'case_000','family':'A','granularity':'task','metrics':{'action_mae':metric}}]})
        (folder/'case_000').mkdir();(folder/'case_000/current.png').touch()
        if mode=='joint':(folder/'case_000/video_08.png').touch()
    render(tmp_path)
    page=(tmp_path/'index.html').read_text()
    assert '<h2>实测结论</h2>' in page and '仅动作误差较低' in page
    assert 'joint/eval_00000500/case_000/video_08.png' in page
    assert '不适用 / not applicable (action-only)' in page
    assert 'action loss' in page


def test_actual_mask_coverage_and_early_child_run_link(tmp_path):
    atomic_json(tmp_path/'data_v4/manifest.json',{'cases':[{'index':0,'family':'human','robot_type':'human','granularity':'task','valid_action_elements':32,'valid_action_dims':[9],'mask_reasons':['missing_tcp_action_mapping']}]})
    atomic_json(tmp_path/'smoke/joint/status.json',{'status':'setup'})
    render(tmp_path)
    text=(tmp_path/'index.html').read_text()
    assert 'smoke/index.html' in text
    assert 'missing_tcp_action_mapping' in text and '[9]' in text
    assert '仅有夹爪监督' in text
    assert '按有效动作元素数加权' in text
