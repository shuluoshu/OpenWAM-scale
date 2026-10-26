"""Offline shuffle audit binds the actual YAML and compact population."""
import pytest


def test_production_audit_reads_real_index_schema_and_verifies_all_ids(tmp_path):
    from test_curated_goal import fixture
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    from openwam.dataloader.goal_epoch import build_epoch_index
    from scripts.data.audit_epoch_order import audit
    source, _ = fixture(tmp_path, annotations=True)
    dataset = CuratedGoalDataset(source, tmp_path/'cache')
    index = tmp_path/'population.sqlite'
    population = build_epoch_index(dataset, index)
    result = audit(index, source, batches=10, world=3, verify_full=True)
    assert result['full_population_exactly_once']
    assert result['prefix_unique_samples'] == population['total_samples']
    assert result['population_granularity_counts'] == {'task': 2, 'subtask': 2}
    assert result['shuffled_granularity_counts'] == result['population_granularity_counts']
    assert sum(result['shuffled_family_counts'].values()) == 4
    source.write_text(source.read_text()+'\n# changed\n')
    with pytest.raises(ValueError, match='identity'):
        audit(index, source)
