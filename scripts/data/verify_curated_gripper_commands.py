"""Validate every accepted window against source state/cmd and FK GT on CPU."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openwam.model.fk_supervision import fk_gt_loss
from openwam.dataloader.curated_fixture import CuratedFixtureDataset


def verify(root):
    root = Path(root).resolve()
    dataset = CuratedFixtureDataset(root / "data_manifest.json")
    checks = []
    for row, reader in zip(dataset.rows, dataset.readers):
        reader._return_trace = True
        frame = pq.read_table(root / "data" / row["id"] / "dataset/data/chunk-000/file-000.parquet").to_pandas()
        rep = reader.representation
        commands = np.asarray(frame["audit.gripper_command"].tolist(), np.float32)
        eef = np.asarray(frame[rep.state_key].tolist(), np.float32) if rep.state_key in frame else None
        joints = np.asarray(frame[rep.joint_state_key].tolist(), np.float32) if rep.joint_state_key in frame else None
        _, _, bad, _ = reader._episode_grippers(0)
        decode = reader._decode_window_video
        fk_records = []
        for j in range(len(reader)):
            original = int(reader._valid_source_indices[j])
            end = min(original + 33, len(frame))
            n = end - original - 1
            assert not bad[original:end].any()
            # Verify actual cached images at each endpoint; other samples only
            # skip RGB allocation, retaining the full production label/FK path.
            visual = j in (0, len(reader)-1)
            reader._decode_window_video = decode if visual else lambda *a, **kw: [None] * 9
            sample = reader[j]
            trace = sample["trace"]
            assert trace["window_offset"] == original
            assert trace["action_target_frame_indices"][:n] == list(range(original+1, end))
            expected = np.nan_to_num(commands[original:end-1], nan=0)
            np.testing.assert_allclose(sample["action"][:n, [9, 43]], expected, rtol=0, atol=0)
            np.testing.assert_array_equal(sample["action_mask"][:n, [9, 43]], np.isfinite(commands[original:end-1]))
            assert not sample["action_mask"][n:].any()
            for side, grip in enumerate((9, 19)):
                if eef is not None and eef.shape[1] > grip and np.isfinite(eef[original, grip]):
                    np.testing.assert_allclose(sample["proprio"][0, (9, 43)[side]], np.clip(eef[original, grip], 0, 1), rtol=0, atol=0)
                indices = rep.joint_indices[("left", "right")[side]]
                if indices:
                    expected_j = joints[original+1:end, indices] - joints[original, indices]
                    block = trace["raw_action_joint14"][:n, side*7:side*7+len(indices)]
                    valid = np.isfinite(expected_j)
                    np.testing.assert_allclose(block[valid], expected_j[valid], rtol=0, atol=0)
            if visual:
                np.testing.assert_array_equal(np.asarray(sample["video"][0]), reader._curated_frames[original])
                last = original + min(32, (end-original-1)//4*4)
                np.testing.assert_array_equal(np.asarray(sample["video"][-1]), reader._curated_frames[last])
            if "fk_supervision" in sample:
                p, r, count = fk_gt_loss(sample["action"][None], [sample["fk_supervision"]], ~sample["action_mask"][None])
                assert p.item() < 1e-8 and r.item() < 1e-8, (row["id"], original, p, r)
                fk_records.append((p.item(), r.item(), count))
        reader._decode_window_video = decode
        summary = dict(id=row["id"], windows=len(reader), image_anchors_checked=2,
                       missing_cmd_sides=[s for s in ("left", "right") if s not in rep.gripper_commands.sources],
                       fk_windows_checked=len(fk_records),
                       fk_max_position_m2=max((x[0] for x in fk_records), default=None),
                       fk_max_rotation=max((x[1] for x in fk_records), default=None))
        checks.append(summary)
        print("PASS", row["id"], len(reader), flush=True)
    report = dict(all_passed=True, windows=len(dataset), clips=len(dataset.readers), skipped=dataset.skipped, checks=checks)
    (root / "validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    result = verify(parser.parse_args().root)
    print({k: result[k] for k in ("all_passed", "windows", "clips", "skipped")})
