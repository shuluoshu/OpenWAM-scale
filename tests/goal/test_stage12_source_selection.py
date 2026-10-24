"""Reproducible, explicitly authorized Stage12 source derivation."""
import hashlib
import json
import pytest
import yaml
from scripts.data.prepare_stage12_selection import prepare_selection, EXCLUDED_REPO


def test_derivation_preserves_other_ranges_and_original_source(tmp_path):
    source=tmp_path/'original.yaml';output=tmp_path/'derived/source.yaml'
    original=dict(data_root_dir='/data',selection=dict(segments_by_repo={
        'mecka/a':[[0,1,4],[2,3,9]],EXCLUDED_REPO:[[0,0,3],[1,0,4]]},
        repo_weights={'mecka/a':.8,EXCLUDED_REPO:.2,'unused':0}),curation={'selected_hours':3000})
    source.write_text(yaml.safe_dump(original));raw=source.read_bytes()
    summary=prepare_selection(source,output)
    derived=yaml.safe_load(output.read_text())
    assert source.read_bytes()==raw
    assert derived['selection']['segments_by_repo']=={'mecka/a':[[0,1,4],[2,3,9]]}
    assert derived['selection']['repo_weights']=={'mecka/a':1.,'unused':0.}
    assert summary['selected_repositories']==1 and summary['selected_segments']==2 and summary['selected_frames']==9
    assert summary['exclusions'][0]['ranges']==[[0,0,3],[1,0,4]]
    assert summary['original_source_yaml_sha256']==hashlib.sha256(raw).hexdigest()
    assert summary['derived_source_yaml_sha256']==hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads((output.parent/'exclusions.json').read_text())==summary
    assert derived['curation']['stage12_derived_selection']['selected_frames']==9


def test_refuses_overwriting_source_and_unselected_exclusion(tmp_path):
    source=tmp_path/'original.yaml';source.write_text('selection: {segments_by_repo: {}, repo_weights: {}}')
    with pytest.raises(ValueError,match='overwrite'):prepare_selection(source,source)
    with pytest.raises(ValueError,match='not selected'):prepare_selection(source,tmp_path/'derived.yaml')
