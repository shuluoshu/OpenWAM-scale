import numpy as np
import pytest

from scripts.data.urdf_fk_audit import URDFModel, pose7_to_matrix, pose_errors


@pytest.fixture
def robot(tmp_path):
    path = tmp_path / 'toy.urdf'
    path.write_text('''<robot name="toy"><link name="base"/><link name="arm"/><link name="slider"/><link name="tip"/>
<joint name="yaw" type="revolute"><parent link="base"/><child link="arm"/><origin xyz="1 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/></joint>
<joint name="slide" type="prismatic"><parent link="arm"/><child link="slider"/><origin xyz="1 0 0"/><axis xyz="1 0 0"/></joint>
<joint name="tool" type="fixed"><parent link="slider"/><child link="tip"/><origin xyz="0 0 0.2" rpy="0 0 1.5707963267948966"/></joint></robot>''')
    return URDFModel(path)


def test_fk_composes_origin_motion_and_fixed_tool(robot):
    t = robot.fk({'yaw': np.array([0, np.pi/2]), 'slide': np.array([.5, .5])}, 'tip')
    np.testing.assert_allclose(t[:, :3, 3], [[2.5, 0, .2], [1, 1.5, .2]], atol=1e-9)
    np.testing.assert_allclose(t[1, :3, :3], np.diag([-1, -1, 1]), atol=1e-9)


def test_subchain_does_not_require_upstream_joints(robot):
    t = robot.fk({'slide': np.array([.5])}, 'tip', base_link='arm')
    np.testing.assert_allclose(t[0, :3, 3], [1.5, 0, .2])


def test_missing_joints_and_unknown_links_fail(robot):
    with pytest.raises(ValueError, match='yaw'):
        robot.fk({'slide': np.array([.5])}, 'tip')
    with pytest.raises(ValueError, match='link'):
        robot.fk({'yaw': np.array([0])}, 'absent')


def test_errors_use_geodesic_rotation_and_millimeters():
    a = pose7_to_matrix(np.array([[0,0,0,1,0,0,0]]))
    b = pose7_to_matrix(np.array([[.003,.004,0,0,0,0,1]]))
    pos, angle = pose_errors(a, b)
    np.testing.assert_allclose(pos, [5])
    np.testing.assert_allclose(angle, [180])
