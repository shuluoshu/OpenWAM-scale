import numpy as np
import pytest

from openwam.dataloader.utils.fk_supervision import FKSupervisionBuilder
from openwam.dataloader.utils.mora_relative import MoraRelativeRepresentation


def fixture(tmp_path):
    urdf = tmp_path / "arm.urdf"
    urdf.write_text(
        '<robot><link name="b"/><link name="t"/><joint name="j" type="revolute"><parent link="b"/><child link="t"/><axis xyz="0 0 1"/></joint></robot>'
    )
    info = {"features": {"joint": {"shape": [1], "names": ["left_joint_0"]}, "tcp_gt": {"shape": [7]}}}
    rep = MoraRelativeRepresentation(info, state_key="missing", joint_state_key="joint")
    cfg = {
        "arms": {
            "left": {
                "urdf_path": str(urdf),
                "base_link": "b",
                "target_link": "t",
                "joint_names": ["j"],
                "endpoint_transform": np.eye(4).tolist(),
                "gt_pose_key": "tcp_gt",
                "gt_frame": "urdf_base",
                "endpoint": "tcp",
            }
        }
    }
    return FKSupervisionBuilder(cfg, rep, info), rep


def test_payload_uses_native_gt_anchor_and_masks_clipping_and_padding(tmp_path):
    builder, rep = fixture(tmp_path)
    cols = {
        "joint": [[0.2], [0.4], [2.0]],
        "tcp_gt": [[1, 0, 0, 1, 0, 0, 0], [1, 0.1, 0, 1, 0, 0, 0], [1, 0.2, 0, 1, 0, 0, 0]],
    }
    action, mask, state, state_mask = rep.encode_window(cols)
    stats = {"q01": np.full(34, -1.0), "q99": np.ones(34)}
    item = builder.build(cols, action, mask, state, state_mask, stats, "quantile", 4)
    arm = item["arms"][0]
    np.testing.assert_array_equal(arm["valid"], [True, False, False, False])
    np.testing.assert_allclose(arm["q_anchor"], [0.2])
    np.testing.assert_allclose(arm["gt_relative"][0, :3, 3], [0, 0.1, 0])
    assert arm["normalization_clipped_count"] == 1
    assert arm["slots"] == [80]


def test_bad_reference_pose_masks_entire_arm(tmp_path):
    builder, rep = fixture(tmp_path)
    cols = {"joint": [[0.2], [0.4]], "tcp_gt": [[np.nan, 0, 0, 1, 0, 0, 0], [1, 0.1, 0, 1, 0, 0, 0]]}
    action, mask, state, state_mask = rep.encode_window(cols)
    item = builder.build(cols, action, mask, state, state_mask, {}, None, 3)
    assert not item["arms"][0]["valid"].any()


def test_undeclared_frame_rejected(tmp_path):
    builder, rep = fixture(tmp_path)
    cfg = builder.config
    cfg["arms"]["left"]["gt_frame"] = "world"
    with pytest.raises(ValueError, match="urdf_base"):
        FKSupervisionBuilder(cfg, rep, {"features": {"tcp_gt": {"shape": [7]}}})


def test_reader_emits_payload_and_tail_mask(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    from openwam.dataloader.bases.lerobot_v3_reader import LeRobotV3Reader
    from openwam.dataloader.registry import build_dataset
    from tests.dataloader.test_mora_relative_reader import make_relative_bucket

    cfg = make_relative_bucket(tmp_path)
    bucket = Path(cfg["dataset_dir"])
    info_path = bucket / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["tcp_gt"] = {"shape": [7]}
    info_path.write_text(json.dumps(info))
    data_path = bucket / "data/chunk-000/file-000.parquet"
    table = pq.read_table(data_path)
    table = table.append_column("tcp_gt", pa.array([[0, 0, 0, 1, 0, 0, 0]] * len(table)))
    pq.write_table(table, data_path)
    path = tmp_path / "six.urdf"
    path.write_text(
        "<robot>"
        + "".join(f'<link name="b{i}"/>' for i in range(7))
        + "".join(
            f'<joint name="j{i}" type="revolute"><parent link="b{i}"/><child link="b{i + 1}"/><axis xyz="0 0 1"/></joint>'
            for i in range(6)
        )
        + "</robot>"
    )
    cfg["fk_supervision"] = {
        "arms": {
            "left": {
                "urdf_path": str(path),
                "base_link": "b0",
                "target_link": "b6",
                "joint_names": [f"j{i}" for i in range(6)],
                "gt_pose_key": "tcp_gt",
                "gt_frame": "urdf_base",
                "endpoint": "tcp",
                "endpoint_transform": np.eye(4).tolist(),
            }
        }
    }
    monkeypatch.setattr(
        LeRobotV3Reader, "_decode_window_video", lambda self, *a, **kw: [Image.new("RGB", (320, 384))] * 9
    )
    ds = build_dataset(cfg)
    sample = ds[0]
    assert len(sample["fk_supervision"]["arms"]) == 1
    assert sample["fk_supervision"]["arms"][0]["q_anchor"].shape == (6,)
    tail = ds[38]
    assert not tail["fk_supervision"]["arms"][0]["valid"][1:].any()
