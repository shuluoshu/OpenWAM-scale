"""Compare rebuilt targets to source Parquet and check production loss gradients.

CPU only: this does not claim to run a pretrained model forward/backward.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.parquet as pq
import torch

from openwam.dataloader.curated_fixture import CuratedFixtureDataset
from openwam.model.architectures.base import BaseWAMArchitecture


def verify():
    old_root = ROOT / "outputs/curated_gripper_cmd_20261013"
    old = CuratedFixtureDataset(old_root / "data_manifest.json")
    new = CuratedFixtureDataset(OUT / "data_manifest.json")
    assert [r["id"] for r in new.rows] == [r["id"] for r in old.rows]
    reports = []
    slots = [9, 43]
    other = [i for i in range(112) if i not in slots]
    scheduler = SimpleNamespace(training_weight=lambda ids: torch.ones_like(ids, dtype=torch.float32))
    for row, before, after in zip(new.rows, old.readers, new.readers):
        human = row["family"] in {"LightWheel", "egodex", "egoverse", "mecka"}
        np.testing.assert_array_equal(before._valid_source_indices, after._valid_source_indices)
        source_error = None
        if human:
            info = json.loads((Path(row["source"]) / "meta/info.json").read_text())
            valid_key = "action.eef_target_valid"
            cols = ["frame_index", "actions.eef"] + ([valid_key] if valid_key in info["features"] else [])
            table = pq.read_table(row["source_parquet"], columns=cols,
                filters=[("episode_index", "=", row["episode"]), ("frame_index", ">=", row["start"]),
                         ("frame_index", "<", row["end"])]).sort_by([("frame_index", "ascending")])
            target = np.asarray(table["actions.eef"].to_pylist(), np.float32)[:, [9, 19]]
            valid = np.isfinite(target)
            if valid_key in table.column_names:
                valid &= np.asarray(table[valid_key].to_pylist()).reshape(-1, 1) == 1
            target = np.clip(target, 0, 1)
            prepared, prepared_valid, _, _ = after._episode_grippers(0)
            np.testing.assert_array_equal(prepared_valid, valid)
            np.testing.assert_array_equal(prepared[valid], target[valid])
            source_error = float(np.max(np.abs(prepared[valid] - target[valid])))
            assert all(s.get("kind") == "human_openness_target" for s in after.representation.gripper_commands.sources.values())
        else:
            for a, b in zip(before._episode_grippers(0)[:2], after._episode_grippers(0)[:2]):
                np.testing.assert_array_equal(a, b)
        # Cached RGB is checked separately by the standard all-window verifier.
        before._decode_window_video = after._decode_window_video = lambda *a, **kw: [None] * 9
        cells = 0
        gradient = None
        for j in range(len(after)):
            a, b = before[j], after[j]
            for key in ("proprio", "proprio_mask", "video_mask"):
                np.testing.assert_array_equal(a[key], b[key])
            for key in ("action", "action_mask"):
                np.testing.assert_array_equal(a[key][:, other], b[key][:, other])
                if not human:
                    np.testing.assert_array_equal(a[key], b[key])
            cells += int(b["action_mask"][:, slots].sum())
            if human:
                assert not a["action_mask"][:, slots].any()
                assert b["action_mask"][:, slots].any()
                assert not b["action_mask"][:, other].any()
                if j == 0:
                    target = b["action"].float().unsqueeze(0)
                    mask = b["action_mask"].bool().unsqueeze(0)
                    prediction = (target + .25).detach().requires_grad_(True)
                    loss = BaseWAMArchitecture._compute_action_loss(None, prediction, target,
                        torch.zeros(1, dtype=torch.long), scheduler, {"action_is_pad": ~mask}, "cpu")
                    loss.backward()
                    assert torch.isfinite(loss) and loss > 0
                    assert (prediction.grad[mask] != 0).all()
                    assert (prediction.grad[~mask] == 0).all()
                    gradient = dict(production_action_loss=float(loss), valid_gripper_cells=int(mask.sum()),
                        nonzero_gradient_cells=int(torch.count_nonzero(prediction.grad)),
                        first_state=b["proprio"][0, slots].tolist(), first_target=b["action"][0, slots].tolist(),
                        scope="production masked loss on CPU with synthetic predictions; no pretrained model forward")
        report = dict(id=row["id"], human=human, windows=len(after), gripper_gt_cells=cells,
            source_max_abs_error=source_error, unchanged_window_indices=True,
            unchanged_inputs_and_non_gripper_targets=True, loss_gradient_check=gradient)
        reports.append(report)
        print("PASS", row["id"], len(after), "gripper cells", cells, flush=True)
    report = dict(all_passed=True, old_root=str(old_root), windows=len(new), clips=len(new.rows),
        restored_human_clips=sum(r["human"] for r in reports),
        restored_human_gripper_cells=sum(r["gripper_gt_cells"] for r in reports if r["human"]),
        runtime_restarted=False, checks=reports)
    (OUT / "human_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k != "checks"}, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    verify()
