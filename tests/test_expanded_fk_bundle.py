import json

import numpy as np
import pytest

from scripts.data.video_fk import load_fk_tracks


def bundle(tmp_path):
    folder = tmp_path/'fk_tracks'/'new_robot'
    folder.mkdir(parents=True)
    arrays = dict(frame_indices=np.array([30, 31]), fk_joint=np.ones((2, 14)),
                  fk_joint_valid=np.ones((2, 14), bool))
    for pose, mask in [('fk_eef_world', 'fk_eef_valid'), ('fk_tcp_world', 'fk_tcp_valid'),
                       ('fk_local_pose', 'fk_local_valid'), ('label_local_pose', 'label_local_valid')]:
        arrays[pose] = np.tile(np.eye(4), (2, 2, 1, 1))
        arrays[mask] = np.ones((2, 2), bool)
    arrays['fk_tcp_valid'][0, 1] = False
    arrays['fk_joint_valid'][:, [6, 13]] = False
    np.savez(folder/'arrays.npz', **arrays)
    (folder/'metadata.json').write_text(json.dumps(dict(
        source_metadata=dict(dataset_dir='/source/a', episode_index=2),
        local_frame='arm base', limitations=['diagnostic world installation'])))
    return folder


def test_expanded_fk_bundle_enforces_masks_and_keeps_provenance(tmp_path):
    bundle(tmp_path)
    arrays, meta = load_fk_tracks('new_robot', np.array([30, 31]), {},
                                 dict(dataset_dir='/source/a', episode_index=2), tmp_path)
    assert np.isnan(arrays['fk_tcp_world'][0, 1]).all()
    assert np.isnan(arrays['fk_joint'][:, [6, 13]]).all()
    assert meta['local_frame'] == 'arm base'
    assert 'diagnostic world installation' in meta['limitations']


@pytest.mark.parametrize('change,match', [
    ({'episode_index': 3}, 'episode'), ({'dataset_dir': '/source/b'}, 'dataset'),
    ({'frames': [31, 30]}, 'frame'),
])
def test_expanded_fk_bundle_rejects_wrong_source_identity(tmp_path, change, match):
    bundle(tmp_path)
    meta = dict(dataset_dir='/source/a', episode_index=2)
    frames = np.array(change.pop('frames', [30, 31]))
    meta.update(change)
    with pytest.raises(ValueError, match=match):
        load_fk_tracks('new_robot', frames, {}, meta, tmp_path)
