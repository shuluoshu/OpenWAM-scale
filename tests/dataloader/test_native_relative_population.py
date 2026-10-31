"""Batched scalar statistics must reproduce the native training reader exactly."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
from openwam.dataloader.utils.gripper import ALIGNED_COMMAND, ALIGNED_VALID
from openwam.dataloader.utils.mora_relative import MoraHeadTCPRepresentation, _rot6d


def metadata(*, reference=16, eef_width=20, aligned=True, dirty=False, head=True):
    n = 48
    t = np.arange(n, dtype=np.float32)
    rotations = Rotation.from_euler('xyz', np.column_stack((t * .07, t * -.03, t * .11))).as_matrix()
    eef = np.zeros((n, eef_width), np.float32)
    for start in range(0, eef_width, 10):
        eef[:, start:start+3] = np.column_stack((t*t*.001 + start, t*.02, -t*.04))
        eef[:, start+3:start+9] = _rot6d(rotations)
        eef[:, start+9] = .2 + t*.003
    camera = np.tile(np.eye(4), (n, 1, 1))
    camera[:, :3, :3] = rotations
    camera[:, :3, 3] = np.column_stack((t*.1, -t*.03, t*.07))
    base = camera.copy()
    if reference == 7:
        base = np.column_stack((base[:, :3, 3], Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]))
    else:
        base = base.reshape(n, 16)
    cols = {'observation.states.eef': eef, 'joint': np.column_stack([t*.01*(j+1) for j in range(14)]),
            'cmd': np.column_stack((.1+t*.005, .8-t*.004)), 'base': base,
            'cal': camera.reshape(n, 16), 'tcp': np.column_stack((t*.02, t*.01, -t*.03,
                Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]))}
    features = {k: {'shape': [v.shape[1]]} for k, v in cols.items()}
    transform = np.eye(4); transform[:3, 3] = [.1, -.2, .3]
    rep = MoraHeadTCPRepresentation({'features': features}, joint_state_key='joint',
        joint_indices={'left': [0, 1, 2, 3, 4, 5, 6], 'right': [7, 8, 9, 10, 11, 12, 13]},
        tcp_from_eef={'left': transform}, tcp_pose_keys={'right': 'tcp'},
        head_world_to_camera_key='cal', head_camera_available=head,
        tcp_reference_to_world_key='base', gripper_command_sources={
            'left': {'key': 'cmd', 'index': 0, 'row_offset': -1},
            'right': {'key': 'cmd', 'index': 1, 'row_offset': 2}})
    if dirty:
        cols['canonical.frame_valid'] = np.ones((n, 1)); cols['canonical.frame_valid'][3] = 0
        cols['training.valid'] = np.ones(n); cols['training.valid'][7] = np.inf
        cols['observation.states.eef.mask'] = np.ones((n, eef_width))
        cols['observation.states.eef.mask'][6, 2] = np.nan
        cols['observation.states.eef.mask'][8, 9] = 0
        cols['actions.eef.mask'] = np.ones((n, eef_width)); cols['actions.eef.mask'][10, 4] = 0
        cols['actions.eef.mask'][11, 9] = np.inf
        cols['actions.joint.mask'] = np.ones((n, 14)); cols['actions.joint.mask'][13, 4] = 0
        cols['observation.states.hand.valid'] = np.ones((n, 2)); cols['observation.states.hand.valid'][14, 1] = .5
        cols['observation.states.hand.confidence'] = np.ones((n, 2)); cols['observation.states.hand.confidence'][15] = [.8, .799]
        cols['observation.states.eef.confidence'] = np.ones((n, 4)); cols['observation.states.eef.confidence'][16, 2:] = [.799, np.nan]
        cols['source.camera.world_to_camera_valid'] = np.ones(n); cols['source.camera.world_to_camera_valid'][17] = 0
        cols['observation.states.camera.valid'] = np.ones((n, 2)); cols['observation.states.camera.valid'][18, 1] = np.nan
        cols['joint'][19, 5] = np.nan; cols['tcp'][20, 3:] = 0
        cols['base'][21] = np.nan; cols['cal'][22] = 0; cols['cal'][23] = np.inf
        cols['cal'][24, 0] += .1; cols['cmd'][25] = np.nan
        cols['observation.states.eef'][26, 3:9] = 0
    if aligned:
        cols[ALIGNED_COMMAND], cols[ALIGNED_VALID] = rep.gripper_commands.align(cols, n)
    return dict(representation=rep, columns=cols, length=n, head_calibration_eligible=True)


def assert_native(m, anchors, ends):
    from openwam.dataloader.native_relative_population import RelativeEpisodeEncoder
    result = RelativeEpisodeEncoder(m).encode(anchors, ends)
    action, mask, state, sm = result
    assert action.shape == mask.shape == (len(anchors), 32, 52)
    assert state.shape == sm.shape == (len(anchors), 52)
    assert action.dtype == state.dtype == np.float32
    assert mask.dtype == sm.dtype == bool
    for i, (anchor, end) in enumerate(zip(anchors, ends)):
        _, expected, em, es, esm, _ = CuratedStage2Dataset.raw_window(None, m, {'last_source': int(end)-1}, int(anchor))
        count = len(expected)
        np.testing.assert_array_equal(mask[i, :count], em)
        np.testing.assert_array_equal(sm[i], esm[0])
        np.testing.assert_array_equal(action[i, :count], expected)
        np.testing.assert_array_equal(state[i], es[0])
        assert not mask[i, count:].any() and not action[i, count:].any()
    return result


@pytest.mark.parametrize('reference', [7, 16])
@pytest.mark.parametrize('eef_width', [10, 20])
@pytest.mark.parametrize('dirty', [False, True])
@pytest.mark.parametrize('aligned', [False, True])
def test_native_equivalence_every_anchor(reference, eef_width, dirty, aligned):
    m = metadata(reference=reference, eef_width=eef_width, dirty=dirty, aligned=aligned)
    anchors = np.arange(m['length']-1)
    ends = np.minimum(anchors+2+anchors % 35, m['length'])
    assert_native(m, anchors, ends)


@pytest.mark.parametrize('missing', ['camera', 'disabled', 'eligibility', 'reference'])
def test_native_missing_geometry(missing):
    m = metadata(head=missing != 'disabled')
    if missing == 'camera': del m['columns']['cal']
    if missing == 'reference': del m['columns']['base']
    if missing == 'eligibility': m['head_calibration_eligible'] = False
    assert_native(m, [0, 4, 45], [48, 9, 48])


def test_second_target_uses_fixed_anchor_and_gripper_uses_previous_row():
    m = metadata()
    action, mask, state, _ = assert_native(m, [0], [48])
    assert mask[0, 1, 20]
    assert action[0, 1, 20] == pytest.approx(2*action[0, 0, 20])
    np.testing.assert_array_equal(action[0, 1, [9, 19]], m['columns'][ALIGNED_COMMAND][1])
    assert state[0, 9] == pytest.approx(.2)


@pytest.mark.parametrize('anchors,ends', [([-1], [4]), ([0], [49]), ([47], [48]),
    ([3], [3]), ([0, 1], [4]), ([[0]], [[4]]), ([.5], [4]), ([True], [4])])
def test_reject_invalid_bounds(anchors, ends):
    from openwam.dataloader.native_relative_population import RelativeEpisodeEncoder
    with pytest.raises(ValueError): RelativeEpisodeEncoder(metadata()).encode(anchors, ends)


def test_empty_batch():
    assert_native(metadata(), [], [])


def test_loaded_scalar_episode(tmp_path):
    from tests.dataloader.test_curated_stage2 import make_native_source
    dataset = CuratedStage2Dataset(make_native_source(tmp_path))
    m = dataset._episode(dataset.repos[0], 0)
    assert_native(m, [0, 5, 20, 22], [24, 9, 24, 24])


def test_translating_tcp_targets_reference_anchor_not_previous_future():
    m = metadata()
    # Dyadic coordinates keep this independent expected-value check exact.
    m['representation'].tcp_from_eef['left'][:3, 3] = [.125, -.25, .5]
    columns = m['columns']
    n = m['length']
    columns['base'][:] = np.eye(4).ravel()
    columns['cal'][:] = np.eye(4).ravel()
    columns['observation.states.eef'][:, 3:9] = [1, 0, 0, 0, 1, 0]
    columns['observation.states.eef'][:, :3] = np.arange(n)[:, None] * [.125, -.25, .5]
    action, mask, _, _ = assert_native(m, [0, 4], [48, 48])
    for batch in range(2):
        np.testing.assert_array_equal(action[batch, 0, :3], [.125, -.25, .5])
        np.testing.assert_array_equal(action[batch, 1, :3], [.25, -.5, 1.])
        np.testing.assert_array_equal(action[batch, 1, 34:37], [.25, -.5, 1.])
        assert mask[batch, :2, :9].all()
