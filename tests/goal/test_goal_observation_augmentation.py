"""Exercise source decode -> augmented current image -> actual query forward."""
import random
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from openwam.dataloader.curated_goal import CuratedGoalDataset
from test_multiview_curated import native_fixture


AUG = dict(enabled=True, color_jitter=dict(p=1, brightness=.2, contrast=.2, saturation=.2),
           patch_mask=dict(p=1, area_range=[.05,.2], aspect_range=[.5,2.]),
           view_dropout=dict(p=1, keep_one_view=True))


def datasets(tmp_path, wrists=('left','right')):
    source, _ = native_fixture(tmp_path, wrists=wrists)
    selection = yaml.safe_load(source.read_text())
    selection['selection']['segments_by_repo']['LightWheel/fixture'] = [[0,0,7]]
    source.write_text(yaml.safe_dump(selection))
    kwargs = dict(source_yaml=source, observation_mode='multiview', history={'enabled':True}, dtype=torch.float32)
    clean = CuratedGoalDataset(**kwargs, cache_dir=tmp_path/'clean')
    augmented = CuratedGoalDataset(**kwargs, cache_dir=tmp_path/'augmented', observation_augmentation=AUG)
    return clean, augmented


def materialize_at_six(dataset, draw=3):
    repo = dataset.repos[0]; metadata = dataset._episode(repo,0)
    dataset._history_draw_id = draw
    return dataset._materialize(repo,0,metadata,metadata['tasks'][0],6,7,'task')


def test_current_only_perturbation_preserves_labels_history_and_rng(tmp_path):
    clean, augmented = datasets(tmp_path)
    cr, cb = materialize_at_six(clean)
    random.seed(27); np.random.seed(13); torch.manual_seed(19)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    ar, ab = materialize_at_six(augmented)
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1],before[1][1])
    assert torch.equal(torch.get_rng_state(),before[2])
    assert not torch.equal(cb['observed'],ab['observed'])
    assert ar['instruction'] == cr['instruction']
    for field in fields(cb['targets']):
        torch.testing.assert_close(getattr(cb['targets'],field.name),getattr(ab['targets'],field.name),rtol=0,atol=0)
    for key in ('history_images','history_ages','history_valid','observed_view_valid'):
        torch.testing.assert_close(cb[key],ab[key],rtol=0,atol=0)
    assert ab['history_valid'].any(), 'test must cover actual past frames'
    trace = ar['provenance']['observation_augmentation']
    assert trace['color_jitter'] and trace['patch'] is not None and trace['dropped_view'] is not None
    # Immutable sample identity, independent of execution order or ambient RNG.
    random.seed(987)
    replay = materialize_at_six(augmented)[1]
    torch.testing.assert_close(ab['observed'],replay['observed'],rtol=0,atol=0)
    assert not torch.equal(ab['observed'],materialize_at_six(augmented,draw=4)[1]['observed'])
    # Exported asset is exactly what becomes the model condition.
    from openwam.train.goal_overfit import prepare_overfit_images
    ar, ab = materialize_at_six(augmented)
    torch.testing.assert_close(ab['observed'],prepare_overfit_images([ar],device='cpu',dtype=torch.float32)[0]['observed'],rtol=0,atol=0)


def test_missing_cameras_and_clean_evaluation(tmp_path):
    clean, augmented = datasets(tmp_path, wrists=('left',))
    ar, ab = materialize_at_six(augmented)
    assert ab['observed_view_valid'].tolist() == [[True,True,False]]
    assert (ab['observed'][...,256:,160:] == -1).all()
    trace = ar['provenance']['observation_augmentation']
    assert trace['dropped_view'] in (0,1) and trace['patch']['view'] in (0,1)
    for (cr,cb),(ar,ab) in zip(clean.evaluation_samples(),augmented.evaluation_samples()):
        torch.testing.assert_close(cb['observed'],ab['observed'],rtol=0,atol=0)
        assert 'observation_augmentation' not in ar['provenance']


def test_augmentation_disabled_compatibility_and_strict_scope(tmp_path):
    source,_ = native_fixture(tmp_path)
    kwargs=dict(source_yaml=source,cache_dir=tmp_path/'cache',observation_mode='multiview')
    clean=CuratedGoalDataset(**kwargs).sample(0)[1]
    disabled=CuratedGoalDataset(**kwargs,observation_augmentation={**AUG,'enabled':False}).sample(0)[1]
    torch.testing.assert_close(clean['observed'],disabled['observed'],rtol=0,atol=0)
    with pytest.raises(ValueError,match='state'):
        CuratedGoalDataset(**kwargs,observation_augmentation={'state_dropout':{'p':.1}})
    with pytest.raises(ValueError,match='multiview'):
        CuratedGoalDataset(source,tmp_path/'head',observation_augmentation=AUG)


def test_augmented_input_reaches_query_loss_and_backward(tmp_path):
    from test_query_goal import make_query
    clean, augmented = datasets(tmp_path)
    _, batch = augmented.sample(0)
    torch.set_num_threads(2); model=make_query().train_stage_a()
    context=torch.zeros(1,3,16)
    out=model(batch['observed'],context,batch['targets'],observed_view_valid=batch['observed_view_valid'])
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    assert model.encoder.image_queries.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.codec.vae.parameters())


@pytest.mark.parametrize('preset',['stage1_query_history_sample.yaml','stage1_query_history_3000h.yaml','stage1_query_history_epoch_3fps.yaml'])
def test_production_presets_enable_current_observation_augmentation(preset):
    from openwam.train.goal_curated import load_config
    root=Path(__file__).resolve().parents[2]
    config=load_config(root/'configs/goal'/preset)
    aug=config['dataset']['observation_augmentation']
    assert aug['enabled'] and aug['patch_mask']['p']==.5
    assert aug['view_dropout']=={'p':.1,'keep_one_view':True}
    assert aug['color_jitter']['p']==.8
