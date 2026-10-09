# Relative joint → FK → GT EEF loss implementation plan

> **For agentic workers:** Execute inline with superpowers:executing-plans and test-driven-development; preserve existing uncommitted work.

**Goal:** Add differentiable FK auxiliary supervision against ground-truth relative endpoint poses, as selected by the user (option 1), retaining the 32-action horizon.

**Architecture:** Recover clean normalized actions with x0_hat = x_sigma - sigma*v_hat; invert per-source joint normalization, then q_future=q_anchor+delta_q. Torch URDF FK yields endpoint transforms; inv(T_fk_anchor) @ T_fk_future is compared to independently read GT relative poses in the same declared URDF base. Per-source configuration declares the arm chain, ordered joints, endpoint transform, and GT pose column. GT is never derived from GT joint FK. Missing profiles/chains/labels are masked, never replaced by zero joints.

**Tech Stack:** Existing PyTorch, NumPy, SciPy, XML parser, Hydra; /root/miniconda3/envs/openwam/bin/python. No new dependencies.

## Global constraints

- Add position squared-distance (m²) and rotation half squared Frobenius distance (dimensionless, locally rad²); separate weights initially 1.0 each, not tuned.
- GT FK comparison uses current endpoint axes in the declared URDF arm-base reference, excluding base motion from joint-only supervision. InternData native arm-base TCP labels permit this without estimated world mounting. Keep existing world-relative action targets unchanged.
- InternData uses native TCP with arm-end local +Z0.22m supported by source labels. No online fitting. AgiBot auxiliary supervision stays disabled because its missing torso makes temporal base-relative EEF targets unverified.
- Masks combine GT pose, anchor/target joints, horizon padding and normalization representability. Quantile-clipped target joints are excluded from FK loss to avoid conflicting with the clipped FM target.
- FK runs float32 outside autocast; loss averages valid arm/time endpoints per sample and then batch (unsupported samples contribute zero). No extra diffusion-time weight for physical FK losses. Never backpropagate through GT.
- Keep video/action losses and default-zero auxiliary weights backward compatible. Preserve current files/config horizon; no git commit/push or full model training required.

## Tasks

- [x] Add tests for analytical revolute/prismatic/fixed FK, required joints, perfect/perturbed predicted joints, gradient flow, independent GT, masks, and current-anchor-relative semantics. Run red.
- [x] Implement openwam/kinematics.py (serializable validated arm chain) and openwam/model/fk_supervision.py (Torch FK and masked GT loss); run tests green.
- [x] Add reader payload builder in openwam/dataloader/utils/fk_supervision.py. Read declared native GT pose7 columns, pad masks, preserve raw anchor joints and per-source affine normalization. Add reader/padding/clipping tests.
- [x] Route optional payload through prepare_inputs, consume outside forward, recover clean joint predictions in base/IDM compute_loss, add two weighted terms and valid-count diagnostics. Verify actual compute_loss and trainer/metric plumbing, backward compatibility.
- [x] Enable InternData profile and current two-source smoke with explicit model/GT provenance; configure separate lambda_fk_position/rotation and log metrics. Keep AgiBot masked.
- [x] Verify actual source windows (including episode tails), compare GT FK residuals, perturb joints and backpropagate on CPU/CUDA if available without foundation model loading. Run relevant existing tests/lint. Update usage/session docs with coverage, formulas, results and limitations.
