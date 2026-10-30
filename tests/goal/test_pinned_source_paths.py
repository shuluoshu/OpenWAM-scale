import json
from pathlib import Path

import yaml
import pytest


def test_alias_file_preserves_prepared_identity_when_source_link_changes(tmp_path):
    from openwam.dataloader.curated_goal import parse_curated_yaml
    from openwam.dataloader.goal_epoch import _identity
    from types import SimpleNamespace
    root=tmp_path/'data';root.mkdir()
    original=tmp_path/'original';original.mkdir()
    replacement=tmp_path/'replacement';replacement.mkdir()
    (root/'mecka').symlink_to(original,target_is_directory=True)
    source=tmp_path/'source.yaml'
    source.write_text(yaml.safe_dump(dict(data_root_dir=str(root),selection=dict(
        segments_by_repo={'mecka/unit':[[0,0,5]]},repo_weights={'mecka/unit':1.}))))
    before=parse_curated_yaml(source)
    expected=_identity(SimpleNamespace(selection=before,repos=before['repos']))
    (root/'mecka').unlink();(root/'mecka').symlink_to(replacement,target_is_directory=True)
    aliases=tmp_path/'aliases.json';aliases.write_text(json.dumps({'mecka/unit':str(original/'unit')}))
    after=parse_curated_yaml(source,path_aliases=str(aliases))
    assert before['repos']==after['repos']
    assert _identity(SimpleNamespace(selection=after,repos=after['repos']))==expected


def test_stage2_and_handoff_use_stage1_pinned_paths(tmp_path):
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.train.stage12 import stage2_path_overrides
    source=tmp_path/'source.yaml'
    source.write_text(yaml.safe_dump(dict(data_root_dir=str(tmp_path),selection=dict(
        segments_by_repo={'mecka/unit':[[0,0,5]]},repo_weights={'mecka/unit':1.}))))
    aliases=tmp_path/'aliases.json';aliases.write_text(json.dumps({'mecka/unit':str(tmp_path/'original')}))
    stage1={'dataset':{'path_aliases':str(aliases)}}
    stage2={'dataloader':{'source_yaml':str(source),'metadata_only':True}}
    overrides=stage2_path_overrides(stage1,stage2)
    from hydra.core.override_parser.overrides_parser import OverridesParser
    parsed=OverridesParser.create().parse_overrides(overrides)
    assert parsed[0].key_or_group=='dataloader.path_aliases'
    assert parsed[0].value()==str(aliases)
    dataset=CuratedStage2Dataset.from_config(stage2['dataloader'])
    assert dataset.repos[0]['path']==str(tmp_path/'original')
    assert stage2_path_overrides({'dataset':{}},{'dataloader':{}})==[]


@pytest.mark.parametrize('override',['++dataloader.path_aliases=/replacement.json',
    '+dataloader.path_aliases.mecka=/replacement','~dataloader.path_aliases'])
def test_cannot_override_checked_source_paths(override):
    from openwam.train.stage12 import validate_stage2_overrides
    with pytest.raises(ValueError,match='shared source'):
        validate_stage2_overrides([override])


def test_relative_alias_file_uses_repository_root(tmp_path):
    from openwam.train.goal_curated import load_config,ROOT
    preset=ROOT/'configs/goal/stage1_query_history_epoch_3fps.yaml'
    config=tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'base_config':str(preset),'dataset':{'path_aliases':'outputs/pinned.json'}}))
    assert load_config(config)['dataset']['path_aliases']==str(ROOT/'outputs/pinned.json')
