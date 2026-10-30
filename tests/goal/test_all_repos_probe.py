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


def test_prepare_tries_later_episode_only_for_no_pair_and_owns_cache():
    from collections import OrderedDict
    from scripts.validate_goal_all_repos import prepare_one
    class Data:
        _episodes=OrderedDict(sentinel=True)
        def _episode(self,repo,episode):
            assert 'sentinel' not in self._episodes
            return dict(length=10,info={'fps':30},source_valid=np.ones(10,bool),
                        tasks=[dict(start=0,end_exclusive=10,text='task')],subtasks=[])
    repo=dict(repo='fixture',path='/fixture',family='fixture',ranges=[(0,1,10),(1,0,10)])
    result=prepare_one(Data(),0,repo,3.)
    assert result['status']=='prepared' and result['episode']==1 and result['no_pair_episodes']==[0]
    assert Data._episodes=={'sentinel':True}
    class Broken(Data):
        def _episode(self,repo,episode):raise ValueError('bad annotation')
    result=prepare_one(Broken(),0,repo,3.)
    assert result['status']=='failed' and result['episode']==0 and 'bad annotation' in result['error']


def test_replay_shards_cover_exact_family_without_duplicates_or_identity_drift():
    from scripts.validate_goal_all_repos import select_replay_entries
    identity=dict(source_sha256='hash',config={'rate':3},population_version=3,repositories=5,sample_fps=3.)
    entries=[dict(repo_index=i,repo=f'family/{i}',path=f'/cpfs/{i}',family='mecka' if i<4 else 'other') for i in range(5)]
    repos=[dict(repo=r['repo'],path=r['path'],family=r['family']) for r in entries]
    groups=[select_replay_entries(entries,identity,identity,repos,['mecka'],i,3) for i in range(3)]
    assert sorted(r['repo_index'] for group in groups for r in group)==[0,1,2,3]
    with pytest.raises(ValueError,match='identity'):
        select_replay_entries(entries,dict(identity,source_sha256='changed'),identity,repos,['mecka'],0,1)
    with pytest.raises(ValueError,match='coverage'):
        select_replay_entries(entries[:-1],identity,identity,repos,['mecka'],0,1)
    with pytest.raises(ValueError,match='family'):
        select_replay_entries(entries,identity,identity,repos,['missing'],0,1)
