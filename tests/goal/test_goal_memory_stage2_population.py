import json
import types
import numpy as np
import pytest
import yaml
from test_rfc_goal_data import source
from openwam.dataloader.rfc_goal import RFCGoalDataset
from openwam.dataloader.curated_stage2 import CuratedStage2Dataset


def population(tmp_path):
    config,root=source(tmp_path)
    value=yaml.safe_load(config.read_text())
    value['selection']['segments_by_repo']['mecka/fixture']=[[0,0,10]]
    config.write_text(yaml.safe_dump(value))
    dataset=RFCGoalDataset(config,tmp_path/'reader',epoch_size=10)
    repo=dataset.reader.repos[0];m=dataset.reader._episode(repo,0)
    native=object.__new__(CuratedStage2Dataset)
    metadata=dict(length=m['length'],tasks=m['tasks'],subtasks=m['subtasks'],
                  columns={'canonical.frame_valid':np.ones(10)},bad=np.zeros(10,dtype=bool))
    native._episode=types.MethodType(lambda self,repo,episode:metadata,native)
    return config,dataset,native,metadata


def test_stage2_population_intersection_preserves_ratio_and_exact_endpoint(tmp_path):
    config,dataset,native,m=population(tmp_path)
    before=dataset.population_identity
    dataset.restrict_stage2(native)
    assert dataset.population_identity!=before
    for epoch in range(3):
        dataset.set_epoch(epoch);samples=[dataset.selection_at(i) for i in range(10)]
        assert sum(s['level']=='subtask' for s in samples)==7
        for item in samples:
            option=item['stage2_option']
            assert option['start']<=item['anchor']<option['end_exclusive']
            assert item['goal']==option['instruction']['end_exclusive']-1
            assert option['instruction']['text']==item['segment']['text']
    fresh=RFCGoalDataset(config,tmp_path/'reader2',epoch_size=10)
    fresh.restrict_stage2(native)
    assert fresh.population_identity==dataset.population_identity
    def identity(s):return (s['record']['anchor_source_frame'],s['granularity'],s['instruction'])
    assert [identity(fresh.sample(i)) for i in range(8,13)]==[identity(dataset.sample(i)) for i in range(8,13)]


def test_stage2_missing_grain_or_changed_instruction_rejects_without_redraw(tmp_path):
    config,dataset,native,m=population(tmp_path)
    m['subtasks']=[]
    with pytest.raises(ValueError,match='subtask'):
        dataset.restrict_stage2(native)
    assert dataset.population_identity  # Failed restriction must not install partial pools.
