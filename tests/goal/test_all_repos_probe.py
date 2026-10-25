import numpy as np
import pytest


def test_probe_uses_subtask_goal_and_valid_three_fps_anchor():
    from scripts.validate_goal_all_repos import choose_pair
    meta=dict(length=100,info={'fps':30},source_valid=np.ones(100,bool),
              tasks=[dict(start=0,end_exclusive=100,text='task')],
              subtasks=[dict(start=10,end_exclusive=40,text='subtask')])
    meta['source_valid'][20]=False
    level,segment,anchor,goal=choose_pair(meta,[(0,11,90)],3.)
    assert (level,segment['text'],anchor,goal)==('subtask','subtask',30,39)


def test_probe_task_only_and_no_legal_anchor():
    from scripts.validate_goal_all_repos import choose_pair
    meta=dict(length=10,info={'fps':30},source_valid=np.ones(10,bool),
              tasks=[dict(start=0,end_exclusive=10,text='task')],subtasks=[])
    assert choose_pair(meta,[(0,0,10)],3.)[0]=='task'
    with pytest.raises(ValueError,match='no valid'):
        choose_pair(meta,[(0,1,10)],3.)
