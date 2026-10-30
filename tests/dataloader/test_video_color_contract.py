from pathlib import Path

import av
import numpy as np
import pytest

from openwam.dataloader.utils.video_io import decode_video_frames
from scripts.data.prepare_goal_labels import decode_goal
from scripts.data.prepare_subtask_goals import decode_frames


def write_clip(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264rgb', rate=30)
        stream.width = 32; stream.height = 24; stream.pix_fmt = 'rgb24'
        stream.options = {'crf': '0'}
        for _ in range(4):
            frame = av.VideoFrame.from_ndarray(np.tile(rgb, (24, 32, 1)).astype(np.uint8), format='rgb24')
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)


@pytest.mark.parametrize('decoder', ['native', 'current_history', 'goal'])
@pytest.mark.parametrize('repo', ['robomind_v1/benchmark1_0_release/h5_agilex_3rgb/task', 'robomind_v2/task/agilex__source_native_joint'])
def test_agilex_recorded_channel_order_is_corrected_at_every_decode_boundary(tmp_path, decoder, repo):
    path = tmp_path/'pretrain_datasetsV0_1'/repo/'videos/front/file-000.mp4'
    # Actual upstream failure: red RGB recorder bytes went through a JPEG BGR
    # decoder assumption then were encoded as RGB video, storing blue instead.
    write_clip(path, [20, 60, 220])
    if decoder == 'native': image = decode_video_frames(str(path), [1], 24, 32)[0]
    elif decoder == 'current_history': image = decode_frames(path, 0, 30, [1])[0][1]
    else: image = decode_goal(path, 0, 1)[0]
    assert np.asarray(image)[10, 10].tolist() == [220, 60, 20]


def test_unaffected_franka_video_remains_rgb(tmp_path):
    path = tmp_path/'pretrain_datasetsV0_1/robomind_v1/benchmark1_0_release/h5_franka_3rgb/task/videos/front/file-000.mp4'
    write_clip(path, [220, 60, 20])
    assert np.asarray(decode_frames(path, 0, 30, [1])[0][1])[10, 10].tolist() == [220, 60, 20]


def test_fix_is_idempotent_and_does_not_match_unversioned_copies():
    from PIL import Image
    from openwam.dataloader.utils.video_color import canonical_video_rgb, video_color_policy, RGB
    path = '/data/pretrain_datasetsV0_1/robomind_v1/h5_agilex_3rgb/task/videos/f.mp4'
    fixed = canonical_video_rgb(Image.new('RGB', (4, 4), (20, 60, 220)), path)
    assert canonical_video_rgb(fixed, path).getpixel((0, 0)) == (220, 60, 20)
    assert video_color_policy('/corrected/robomind_v1/h5_agilex_3rgb/f.mp4') == RGB


def test_training_rejects_stale_or_partially_corrected_cache():
    import torch
    from types import SimpleNamespace
    from openwam.dataloader.utils import video_color as module
    validate = getattr(module, 'validate_sample_color', None)
    stamp = getattr(module, 'stamp_sample_color', None)
    assert callable(validate) and callable(stamp), 'training needs a versioned color cache contract'
    record = {'source_repo': '/data/pretrain_datasetsV0_1/robomind_v1/h5_agilex_3rgb/task'}
    batch = dict(observed=torch.randn(1,3,1,4,4), targets=SimpleNamespace(image=torch.randn(1,3,1,4,4)), history_images=torch.randn(1,2,3,1,4,4))
    with pytest.raises(ValueError, match='stale RGB cache'): validate(record, batch)
    stamp(record, batch);validate(record, batch)
    batch['history_images'] = batch['history_images'][:,:, [2,1,0]].clone()
    with pytest.raises(ValueError, match='history_images'): validate(record, batch)
    stamp(record, batch)
    del batch['history_images']
    with pytest.raises(ValueError, match='fields'): validate(record, batch)


def test_stage2_cached_images_require_complete_color_contract():
    import torch
    from PIL import Image
    from openwam.dataloader.utils import video_color as module
    stamp=getattr(module,'stamp_stage2_color',None);validate=getattr(module,'validate_stage2_color',None)
    assert callable(stamp) and callable(validate), 'Stage2 must reject old fixed sample caches'
    sample={'trace':{'source_repo':'/data/pretrain_datasetsV0_1/robomind_v2/task/agilex__source_native_joint'},
            'video':[Image.new('RGB',(4,4),'red')], 'first_frame_image':[Image.new('RGB',(4,4),'red')],
            'goal_observed_multiview':torch.zeros(3,1,6,4),'history_images':torch.zeros(10,3,1,4,4)}
    with pytest.raises(ValueError,match='stale RGB cache'):validate(sample)
    stamp(sample);validate(sample)
    sample['video'][0]=Image.new('RGB',(4,4),'blue')
    with pytest.raises(ValueError,match='video'):validate(sample)
