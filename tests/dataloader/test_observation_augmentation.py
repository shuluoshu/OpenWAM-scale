import random

import numpy as np
from PIL import Image
import pytest
import torch

from openwam.dataloader.transforms.observation import ObservationAugmentation
from openwam.train.utils.observation_cache import prepare_observation


def sample(views=(True, True, True)):
    image = np.zeros((96, 80, 3), np.uint8)
    image[:64] = [80, 120, 160] if views[0] else 0
    image[64:, :40] = [170, 100, 60] if views[1] else 0
    image[64:, 40:] = [50, 180, 120] if views[2] else 0
    missing = Image.new('L', (80, 96), 0)
    boxes = [(0, 0, 80, 64), (0, 64, 40, 96), (40, 64, 80, 96)]
    for exists, box in zip(views, boxes):
        if not exists: missing.paste(255, box)
    return dict(video=[Image.fromarray(image.copy()) for _ in range(3)],
                first_frame_image=[Image.fromarray(image.copy())], proprio=torch.ones(1, 112),
                proprio_mask=torch.ones(1, 112, dtype=torch.bool),
                action=torch.full((32, 112), .3), action_mask=torch.ones(32,112,dtype=torch.bool),
                fk_supervision={'q_anchor': torch.arange(14)}, _curated_index=7), [missing]*3


def test_single_patch_is_inside_one_view_and_consistent_over_time():
    s, missing = sample()
    random.seed(3)
    out = ObservationAugmentation({'patch_mask': {'p': 1, 'area_range': [.1,.2]}}).apply(s, missing)
    meta = out['_observation_augmentation']
    patch = meta['patch']
    x0,y0,x1,y1 = patch['box']
    bx0,by0,bx1,by1 = meta['view_boxes'][patch['view']]
    assert bx0 <= x0 < x1 <= bx1 and by0 <= y0 < y1 <= by1
    before = np.array(s['video'][0]);after = np.array(out['video'][0])
    changed = np.any(before != after,axis=-1)
    assert changed.sum() == (x1-x0)*(y1-y0)
    assert all(np.array_equal(after,np.array(im)) for im in out['video'])
    assert np.array_equal(np.array(out['first_frame_image'][0]),after)
    assert not np.array_equal(before,after)


def test_patch_size_varies_and_seed_replays():
    s,m = sample();a=ObservationAugmentation({'patch_mask':{'p':1}})
    random.seed(42);first=a.apply(s,m)['_observation_augmentation']['patch']
    random.seed(42);assert a.apply(s,m)['_observation_augmentation']['patch']==first
    sizes=set()
    for _ in range(20):
        box=a.apply(s,m)['_observation_augmentation']['patch']['box']
        sizes.add((box[2]-box[0],box[3]-box[1]))
    assert len(sizes)>1


def test_drop_one_existing_view_and_choose_patch_from_survivors():
    s,m=sample((True,False,True));a=ObservationAugmentation({'view_dropout':{'p':1},'patch_mask':{'p':1}})
    out=a.apply(s,m);meta=out['_observation_augmentation']
    dropped=meta['dropped_view'];assert dropped in (0,2)
    assert meta['patch']['view'] in (0,2) and meta['patch']['view'] != dropped
    x0,y0,x1,y1=meta['view_boxes'][dropped]
    assert not np.array(out['video'][0])[y0:y1,x0:x1].any()
    assert not np.array(out['video'][0])[64:,:40].any()
    one,m=sample((True,False,False));out=a.apply(one,m)
    assert out['_observation_augmentation']['dropped_view'] is None
    assert out['_observation_augmentation']['patch']['view']==0


def test_color_jitter_preserves_missing_black_and_is_clip_consistent():
    s,m=sample((True,False,False));random.seed(5)
    out=ObservationAugmentation({'color_jitter':{'p':1}}).apply(s,m)
    assert out['_observation_augmentation']['image_changed']
    assert not np.array(out['video'][0])[64:].any()
    assert all(np.array_equal(np.array(out['video'][0]),np.array(im)) for im in out['video'])
    assert not np.array_equal(np.array(out['video'][0]),np.array(s['video'][0]))


def test_state_dropout_keeps_labels_and_fk_and_does_not_mutate_input():
    s,m=sample();out=ObservationAugmentation({'state_dropout':{'p':1}}).apply(s,m)
    assert not out['proprio'].any() and not out['proprio_mask'].any()
    assert s['proprio'].all() and s['proprio_mask'].all()
    for key in ['action','action_mask']:assert torch.equal(out[key],s[key])
    assert out['fk_supervision'] is s['fk_supervision']
    assert not out['_observation_augmentation']['image_changed']


@pytest.mark.parametrize('cfg', [None, {'enabled':False,'state_dropout':{'p':1}}, {'state_dropout':{'p':0}}])
def test_disabled_transform_is_identity(cfg):
    s,m=sample();a=ObservationAugmentation(cfg)
    before=random.getstate();assert a.apply(s,m) is s;assert random.getstate()==before


def test_eval_is_identity_and_preserves_random_state():
    s,m=sample();a=ObservationAugmentation({'color_jitter':{'p':1},'state_dropout':{'p':1}}).eval()
    before=random.getstate();assert a.apply(s,m) is s;assert random.getstate()==before


@pytest.mark.parametrize('cfg', [{'state_dropout':{'p':-1}}, {'patch_mask':{'area_range':[.2,.1]}},
                                {'patch_mask':{'area_range':[0,.1]}}, {'view_dropout':{'p':2}},
                                {'typo':{}}, {'patch_mask':{'probablity':.5}}])
def test_invalid_config_is_rejected(cfg):
    with pytest.raises(ValueError):ObservationAugmentation(cfg)


def test_cache_never_stores_dropped_state_and_state_drop_does_not_reencode_video():
    s,m=sample();drop=ObservationAugmentation({'state_dropout':{'p':1}}).apply(s,m)
    calls=[];cache={}
    def prepare(batch):
        q=batch[0];calls.append(q)
        return {'proprio':q['proprio'].clone(),'proprio_mask':q['proprio_mask'].clone(),
                'actions':q['action'],'input_latents':torch.ones(1)}
    out=prepare_observation(drop,prepare,cache)
    assert not out['proprio'].any() and not out['proprio_mask'].any()
    assert cache[7]['proprio'].all() and cache[7]['proprio_mask'].all()
    clean=prepare_observation(s,prepare,cache)
    assert clean['proprio'].all() and len(calls)==1
    again=prepare_observation(drop,prepare,cache)
    assert not again['proprio'].any() and len(calls)==1


def test_changed_images_bypass_clean_cache_without_polluting_it():
    s,m=sample();cache={};calls=[]
    def prepare(batch):
        calls.append(batch[0]);return {'pixels':np.array(batch[0]['video'][0]),'proprio':batch[0]['proprio']}
    clean=prepare_observation(s,prepare,cache)
    augmented=ObservationAugmentation({'patch_mask':{'p':1}}).apply(s,m)
    out=prepare_observation(augmented,prepare,cache)
    assert len(calls)==2 and not np.array_equal(clean['pixels'],out['pixels'])
    assert np.array_equal(cache[7]['pixels'],clean['pixels'])


def test_cached_fixture_decoder_returns_structural_missing_masks():
    from types import SimpleNamespace
    from openwam.dataloader.curated_fixture import _cached_video
    s,_=sample((True,False,True))
    reader=SimpleNamespace(_curated_frames=np.stack([np.array(x) for x in s['video']]),
                           _num_video_frames=3,_curated_view_available=(True,False,True))
    frames,masks=_cached_video(reader,None,0,0,[0,1,2],0,return_missing_masks=True)
    assert len(frames)==len(masks)==3
    assert np.array(masks[0])[64:,:40].all()
    assert not np.array(masks[0])[:64].any()


def test_curated_evaluation_context_restores_modes_even_on_exception():
    from types import SimpleNamespace
    from openwam.dataloader.curated_fixture import CuratedFixtureDataset
    from openwam.dataloader.transforms.video import VideoColorJitter
    dataset=CuratedFixtureDataset.__new__(CuratedFixtureDataset)
    aug=ObservationAugmentation({'state_dropout':{'p':1}})
    jitter=VideoColorJitter()
    dataset.readers=[SimpleNamespace(observation_augmentation=aug,_color_jitter=jitter)]
    with pytest.raises(RuntimeError):
        with dataset.unaugmented():
            assert not aug.training and not jitter.training
            raise RuntimeError('evaluation interrupted')
    assert aug.training and jitter.training
