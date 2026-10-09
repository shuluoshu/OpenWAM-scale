"""Bounded real-data FK-loss check without loading a foundation model."""

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from openwam.dataloader.registry import build_dataset
from openwam.model.fk_supervision import fk_gt_loss


def main(device="cpu"):
    config = OmegaConf.load("configs/dataloader/canonical_eef_interndata_smoke.yaml")
    config.max_windows = None
    config.return_trace = True
    dataset = build_dataset(config)
    split = int(dataset._cum_n_starts[1])
    results = []
    for index in [0, 7, split - 1, split + 7, len(dataset) - 1]:
        sample = dataset[index]
        payload = sample["fk_supervision"]
        clean = sample["action"].unsqueeze(0).to(device).requires_grad_()
        padding = ~sample["action_mask"].unsqueeze(0).to(device)
        position, rotation, count = fk_gt_loss(clean, [payload], padding)
        assert position.item() < 1e-10 and rotation.item() < 1e-10, (index, position.item(), rotation.item())
        gt_position, gt_rotation = position.item(), rotation.item()
        clipped = sum(a["normalization_clipped_count"] for a in payload["arms"])
        # Perturb the physical predicted joints by 0.05 rad, not by a scale-dependent normalized amount.
        with torch.no_grad():
            for arm in payload["arms"]:
                clean[0, :, arm["slots"]] += torch.as_tensor(0.05 / arm["scale"], device=device)
        position, rotation, count = fk_gt_loss(clean, [payload], padding)
        (position + rotation).backward()
        gradient = clean.grad[..., 80:94]
        assert torch.isfinite(gradient).all()
        if count:
            assert position.item() > gt_position and rotation.item() > gt_rotation and gradient.abs().sum() > 0
        if index in (split - 1, len(dataset) - 1):
            assert all(not arm["valid"][1:].any() for arm in payload["arms"])
        results.append(
            dict(
                index=index,
                episode=sample["trace"]["episode_index"],
                anchor=sample["trace"]["window_offset"],
                valid_arm_steps=count,
                clipped_arm_steps=clipped,
                gt_position_m2=gt_position,
                gt_rotation_chordal=gt_rotation,
                perturbed_position_m2=position.item(),
                perturbed_rotation_chordal=rotation.item(),
                gradient_abs_sum=gradient.abs().sum().item(),
            )
        )
        print(json.dumps(results[-1]), flush=True)
    agi = build_dataset(OmegaConf.load("configs/dataloader/canonical_eef_agibotworld_smoke.yaml"))
    assert agi._fk_builder is None
    output = Path("outputs/fk_training_loss")
    output.mkdir(parents=True, exist_ok=True)
    evidence = dict(
        device=device,
        torch=torch.__version__,
        horizon=32,
        urdf=payload["arms"][0]["chain"]["urdf_path"],
        urdf_sha256=payload["arms"][0]["chain"]["urdf_sha256"],
        agibot_auxiliary_enabled=False,
        samples=results,
    )
    (output / f"verification_{device}.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    main(parser.parse_args().device)
