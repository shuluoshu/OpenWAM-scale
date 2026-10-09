import numpy as np
import pytest
import torch

from openwam.kinematics import compile_chain
from openwam.model.fk_supervision import fk_gt_loss, torch_fk


@pytest.fixture
def chain(tmp_path):
    path = tmp_path / "arm.urdf"
    path.write_text("""<robot name="arm"><link name="base"/><link name="arm"/><link name="tip"/>
    <joint name="q" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/></joint>
    <joint name="tool" type="fixed"><parent link="arm"/><child link="tip"/><origin xyz="1 0 0"/></joint></robot>""")
    return compile_chain(str(path), "base", "tip", ["q"])


def payload(chain, future=(0.5, 1.0), anchor=0.2):
    angles = np.array(future) - anchor
    relative = np.tile(np.eye(4), (len(angles), 1, 1))
    for i, a in enumerate(angles):
        c, s = np.cos(a), np.sin(a)
        relative[i, :3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        relative[i, :3, 3] = [c - 1, s, 0]
    return {
        "arms": [
            {
                "chain": chain,
                "slots": [80],
                "q_anchor": [anchor],
                "scale": [2.0],
                "offset": [0.1],
                "gt_relative": relative,
                "valid": np.ones(len(angles), bool),
            }
        ]
    }


def test_analytic_fk_and_gradcheck(chain):
    q = torch.tensor([[0.0], [np.pi / 2]], dtype=torch.float64, requires_grad=True)
    pose = torch_fk(q, chain)
    torch.testing.assert_close(
        pose[:, :3, 3], torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], dtype=torch.float64), atol=1e-10, rtol=0
    )
    assert torch.autograd.gradcheck(lambda v: torch_fk(v, chain), (q,))


def test_gt_loss_perfect_perturbed_and_no_pose_branch_gradient(chain):
    data = payload(chain)
    clean = torch.zeros((1, 2, 112), requires_grad=True)
    with torch.no_grad():
        clean[0, :, 80] = torch.tensor([0.1, 0.35])  # delta .3/.8, affine inversion
    pos, rot, count = fk_gt_loss(clean, [data])
    assert count == 2 and pos.item() < 1e-12 and rot.item() < 1e-12
    with torch.no_grad():
        clean[0, 1, 80] += 0.1
    pos, rot, count = fk_gt_loss(clean, [data])
    assert pos > 0 and rot > 0
    (pos + rot).backward()
    assert clean.grad[0, 1, 80].abs() > 0
    assert clean.grad[:, :, :80].count_nonzero() == 0 and clean.grad[:, :, 81:].count_nonzero() == 0


def test_missing_and_masked_entries_do_not_dilute_or_poison(chain):
    data = payload(chain)
    data["arms"][0]["valid"][1] = False
    data["arms"][0]["gt_relative"][1] = np.nan
    clean = torch.zeros((2, 2, 112), requires_grad=True)
    pos, rot, count = fk_gt_loss(clean, [data, None])
    assert torch.isfinite(pos + rot) and count == 1
    data["arms"][0]["valid"][:] = False
    pos, rot, count = fk_gt_loss(clean, [data, None])
    assert count == 0 and (pos + rot).item() == 0
    (pos + rot).backward()
    assert clean.grad.count_nonzero() == 0


def test_independent_gt_is_used(chain):
    data = payload(chain)
    clean = torch.zeros((1, 2, 112))
    original = fk_gt_loss(clean, [data])[0]
    data["arms"][0]["gt_relative"][:, :3, 3] += 1
    assert fk_gt_loss(clean, [data])[0] > original


def test_unknown_joint_fails(tmp_path):
    path = tmp_path / "bad.urdf"
    path.write_text(
        '<robot><link name="b"/><link name="t"/><joint name="missing" type="revolute"><parent link="b"/><child link="t"/></joint></robot>'
    )
    with pytest.raises(ValueError, match="joint"):
        compile_chain(str(path), "b", "t", [])


def test_prismatic_with_rotated_parent_and_fixed_endpoint(tmp_path):
    path = tmp_path / "slide.urdf"
    path.write_text(
        '<robot><link name="b"/><link name="t"/><joint name="q" type="prismatic"><parent link="b"/><child link="t"/><origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/><axis xyz="1 0 0"/></joint></robot>'
    )
    tool = np.eye(4)
    tool[0, 3] = 0.5
    compiled = compile_chain(path, "b", "t", ["q"], tool)
    q = torch.tensor([[0.2]], requires_grad=True)
    transform = torch_fk(q, compiled)
    torch.testing.assert_close(transform[0, :3, 3], torch.tensor([1.0, 2.7, 3.0]))
    transform[0, 1, 3].backward()
    torch.testing.assert_close(q.grad, torch.ones_like(q))


def test_action_padding_disables_auxiliary_even_if_payload_valid(chain):
    clean = torch.zeros((1, 2, 112), requires_grad=True)
    padding = torch.zeros_like(clean, dtype=torch.bool)
    padding[:, :, 80] = True
    pos, rot, count = fk_gt_loss(clean, [payload(chain)], padding)
    assert count == 0 and (pos + rot).item() == 0
