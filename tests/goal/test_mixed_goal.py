import copy
from collections import Counter
import pytest


def rows():
    return [dict(id='tA',family='A',granularity='task'),dict(id='sA0',family='A',granularity='subtask'),dict(id='sA1',family='A',granularity='subtask'),dict(id='tB',family='B',granularity='task')]


def test_balanced_mixture_does_not_dilute_tasks():
    from openwam.dataloader.mixed_goal import balanced_schedule
    rr=rows(); seq=balanced_schedule(rr,120,42)
    counts=Counter((rr[i]['family'],rr[i]['granularity']) for i in seq)
    assert counts=={('A','task'):30,('A','subtask'):30,('B','task'):60}
    assert Counter(seq)[1]==Counter(seq)[2]==15
    assert seq==balanced_schedule(rr,120,42)
    with pytest.raises(ValueError,match='task'):
        balanced_schedule(rr[1:3],10,42)


def test_validate_task_and_subtask_bounds():
    from openwam.dataloader.mixed_goal import validate_mixed_rows
    a=dict(id='t',family='A',source_repo='/tmp/source',source_episode=0,source_episode_length=10,
           anchor_source_frame=0,goal_source_frame=9,instruction_start=0,instruction_end_exclusive=10,
           granularity='task',label_kind='episode_end',success_verified=False,instruction='whole task',split='diagnostic')
    b={**a,'id':'s','granularity':'subtask','label_kind':'subtask_end','instruction':'part',
       'instruction_start':2,'anchor_source_frame':2,'instruction_end_exclusive':6,'goal_source_frame':5}
    validate_mixed_rows([a,b])
    bad=copy.deepcopy(b);bad['goal_source_frame']=9
    with pytest.raises(ValueError,match='goal'):validate_mixed_rows([a,bad])
    bad={**b,'source_repo':'/tmp/other'}
    with pytest.raises(ValueError,match='task pair'):validate_mixed_rows([a,bad])
    with pytest.raises(ValueError,match='duplicate'):validate_mixed_rows([a,b,b])


def test_history_does_not_mix_same_family_task_and_subtask_errors(tmp_path):
    import json
    from openwam.train.goal_history import render_goal_history
    d=tmp_path/'eval_step_000500';d.mkdir()
    cards=[dict(id=f'case_{i:04d}',family='A',granularity=level,instruction=level+' instruction',metric_key=level,valid_points=[True,True]) for i,level in enumerate(['task','subtask'])]
    (d/'metrics.json').write_text(json.dumps({'review':[{'source':'task','image_mae_uint8':2.},{'source':'subtask','image_mae_uint8':8.}]}))
    metrics={f'source:{level}':{'left':{'pixel_error':value,'count':1},'right':{'pixel_error':value,'count':1}} for level,value in [('task',3.),('subtask',9.)]}
    render_goal_history(tmp_path,dict(evaluations=[dict(step=500,directory=d.name,point_metrics=metrics)]),cards)
    page=(tmp_path/'timeline.html').read_text();a,b=page.split('<article')[1:]
    assert '图像 MAE：2.00 / 255' in a and 'TCP 平均误差：3.00 px' in a
    assert '图像 MAE：8.00 / 255' in b and 'TCP 平均误差：9.00 px' in b
    assert 'task instruction' in a and 'subtask instruction' in b


def test_group_metrics_weight_points_and_exclude_missing_gt():
    from openwam.train.mixed_metrics import aggregate_mixed_metrics
    rows=[dict(family='A',granularity='task',image_mae_uint8=2.,tcp_pixel_errors=[2.,None]),
          dict(family='A',granularity='task',image_mae_uint8=4.,tcp_pixel_errors=[4.,6.]),
          dict(family='A',granularity='subtask',image_mae_uint8=9.,tcp_pixel_errors=[None,None])]
    result=aggregate_mixed_metrics(rows)
    assert result['A / task']['image_mae_uint8']==3.
    assert result['A / task']['tcp_pixel_error']==4.
    assert result['A / task']['valid_points']==3
    assert result['A / subtask']['tcp_pixel_error'] is None


def test_mixed_training_uses_balanced_schedule_and_unique_case_reports(tmp_path):
    import json
    from openwam.train.goal_patch import make_smoke_data,train_patch
    expert,bb,rr=make_smoke_data()
    records=[];batches=[]
    for r in rows():
        record={**rr[0],**r,'metric_key':r['id']}
        b=copy.deepcopy(bb[0]);b.update(sources=[r['id']],family=r['family'],granularity=r['granularity'],instruction=r['id'])
        records.append(record);batches.append(b)
    result=train_patch(expert,batches=batches,records=records,output_dir=tmp_path/'run',
        steps=4,eval_every=2,sampling_steps=1,learning_rate=.001,
        precomputed_text_fingerprint='a'*64,training_metadata={'sampling_policy':'family_level_balanced'})
    assert result['completed_steps']==4
    assert [e['step'] for e in result['evaluations']]==[0,2,4]
    report=json.loads((tmp_path/'run/eval_step_000004/metrics.json').read_text())
    assert len(report['review'])==4
    assert report['group_metrics']['A / subtask']['samples']==2
    assert all('source:'+r['id'] in report['point_metrics'] for r in records)
    events=[json.loads(x) for x in (tmp_path/'run/progress.jsonl').read_text().splitlines()]
    events=[e for e in events if e['event']=='train_step']
    assert Counter((e['family'],e['granularity']) for e in events)=={('A','task'):1,('A','subtask'):1,('B','task'):2}
