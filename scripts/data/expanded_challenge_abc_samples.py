"""Read-only 180-frame Challenge v2 / ABC release geometry and media audit.

Current release anchor rectifications are inverted explicitly. No fitted base
transform is promoted to independent world calibration. Source release is read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.data.urdf_fk_audit import URDFModel, eef20_to_matrix, load_episode, pose_errors
from scripts.data.video_projection_samples import decode_sequence
from scripts.data.wrist_projection_samples import camera_validity

ROOT = Path(__file__).resolve().parents[2]
RELEASE = Path("/mnt/data/cpfs/b5big/pretrain_data_release")
ASSETS = Path("/mnt/data/cpfs/b5big/pretrain_urdfs")
FACTORY = Path("/mnt/workspace/mark/opensource_data_factory")
SOURCES = {
    "robochallenge_v2_aloha": (
        "RoboChallenge v2 · ALOHA",
        RELEASE / "pretrain_datasetsV0_2/robochallenge_v2/pack_the_items/dual_arm__parallel_gripper",
    ),
    "robochallenge_v2_dosw1": (
        "RoboChallenge v2 · DOS-W1",
        RELEASE / "pretrain_datasetsV0_2/robochallenge_v2/fold_the_clothes/dual_arm__parallel_gripper",
    ),
    "abc_130k_yam": ("ABC-130K · YAM", RELEASE / "pretrain_datasetsV0_3/abc_130k/train/clear_the_kitchen_counter"),
}
SIDES = ("left", "right")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            data, indent=2, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)
        )
        + "\n"
    )


def jsonl_episode(path, episode=0):
    with path.open() as f:
        for line in f:
            x = json.loads(line)
            if x.get("episode_index") == episode:
                return x
    raise ValueError(f"episode {episode} missing: {path}")


def tx(xyz):
    t = np.eye(4)
    t[:3, 3] = xyz
    return t


def summarize_pose(label, predicted):
    d, a = pose_errors(label, predicted)
    return {
        "translation_mm": {
            k: float(f(d)) for k, f in [("median", np.median), ("p95", lambda x: np.percentile(x, 95)), ("max", np.max)]
        },
        "rotation_deg": {
            k: float(f(a)) for k, f in [("median", np.median), ("p95", lambda x: np.percentile(x, 95)), ("max", np.max)]
        },
    }


def mjcf_link6_fk(path, q):
    """Evaluate the six explicit hinge bodies; geometry/dynamics are not involved."""
    xml = ET.parse(path).getroot()
    body = xml.find("worldbody/body")
    t = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    chain = []

    def fixed(node):
        v = np.fromstring(node.get("quat", "1 0 0 0"), sep=" ")
        out = tx(np.fromstring(node.get("pos", "0 0 0"), sep=" "))
        out[:3, :3] = Rotation.from_quat(v[[1, 2, 3, 0]]).as_matrix()
        return out

    t = t @ fixed(body)
    for i in range(6):
        body = body.find(f"body[@name='link_{i + 1}']")
        if body is None:
            raise ValueError("MJCF chain changed")
        j = body.find("joint")
        assert j.get("name") == f"joint{i + 1}"
        axis = np.fromstring(j.get("axis", "0 0 1"), sep=" ")
        axis /= np.linalg.norm(axis)
        p = np.fromstring(j.get("pos", "0 0 0"), sep=" ")
        motion = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        motion[:, :3, :3] = Rotation.from_rotvec(q[:, i, None] * axis).as_matrix()
        t = t @ fixed(body) @ tx(p) @ motion @ tx(-p)
        chain.append({"body": dict(body.attrib), "joint": dict(j.attrib), "effective_axis": axis.tolist()})
    return t, chain


def aloha_mesh_diagnostics(model):
    """Inspect actual collision vertices; a finger-frame origin is not a TCP."""
    result = []
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    for number in (7, 8):
        path = model.path.parent / f"piper_meshes/link{number}.STL"
        payload = path.read_bytes()
        count = int.from_bytes(payload[80:84], "little")
        if len(payload) != 84 + count * 50:
            raise ValueError("unexpected nonbinary STL")
        vertices = np.frombuffer(payload, offset=84, dtype=dtype, count=count)["vertices"].reshape(-1, 3)
        transform = model.fk({f"left/joint{number}": [0]}, f"left/link{number}", "left/link6")[0]
        points = vertices @ transform[:3, :3].T + transform[:3, 3]
        result.append(
            dict(
                mesh=str(path),
                sha256=hashlib.sha256(payload).hexdigest(),
                triangles=count,
                joint_position_m=0,
                flange_frame_bounds_m=[points.min(0), points.max(0)],
            )
        )
    return {
        "collision_meshes": result,
        "decision": "No semantic contact-pad annotation. Left/right zero-opening collision meshes occupy different axial intervals around z=135.03 mm, so their origins/bounds cannot establish an installed physical contact TCP. No scalar is promoted to TCP.",
    }


def station_geometry():
    # Reuse the audited contact-pad geometry construction, without its FK backend.
    sys.path.insert(0, str(FACTORY))
    from data_curation.fk_pipeline.abc_130k.audit_abc_130k import canonical_geometry

    return canonical_geometry(ASSETS / "abc_130k/put_bottles/put_bottle.xml")


def geometry(name, info, window):
    n = len(window)
    eef = eef20_to_matrix(np.stack(window["observation.states.eef"]))
    q = np.stack(window["observation.state"]).astype(float)
    meta = info["starvla_metadata"]
    audit = {"release_rectification_history": meta.get("rectification_history", [])}
    tcp = np.full_like(eef, np.nan)
    fk = np.full_like(eef, np.nan)
    fk_tcp = np.full_like(eef, np.nan)
    fk_arm = np.full_like(eef, np.nan)
    wrist_mounts = []
    limitations = []
    world_valid = np.zeros((n, 2), bool)
    is_abc = name == "abc_130k_yam"
    is_aloha = name.endswith("aloha")
    if is_abc:
        single = ASSETS / "abc_130k/i2rt_yam/yam.xml"
        station = ASSETS / "abc_130k/put_bottles/put_bottle.xml"
        canonical, mounts = station_geometry()
        B = np.eye(4)
        B[:3, :3] = np.eye(3)[:, [2, 0, 1]]
        history = next(
            x for x in meta["rectification_history"] if x.get("schema_version") == "eef_tcp_to_wrist_0804_v1"
        )
        assert any(
            x.get("schema_version") == "abc_130k_eef_axis_x_forward_z_up_v1" for x in meta["rectification_history"]
        )
        fixed = []
        for i, side in enumerate(SIDES):
            source_wrist = np.array(history["transforms"][i]["T_source_wrist_row_major"]).reshape(4, 4)
            current_wrist_tcp = np.linalg.inv(B) @ np.linalg.inv(source_wrist)
            tcp[:, i] = eef[:, i] @ current_wrist_tcp
            local, chain = mjcf_link6_fk(single, q[:, i * 7 : i * 7 + 6])
            # Single-arm grasp_site shares link6 origin axes; station defines canonical pad frame.
            single_site = tx([0, 0, 0.1347])
            single_site[:3, :3] = Rotation.from_quat([0, 0, -1, 1]).as_matrix()
            base = tx([0.2525, 0.31 if i == 0 else -0.31, 0.76])
            world_link6 = base @ local
            fk_tcp[:, i] = world_link6 @ single_site @ canonical[side]
            fk[:, i] = fk_tcp[:, i] @ source_wrist @ B
            fk_arm[:, i] = np.linalg.inv(base) @ fk[:, i]
            np.testing.assert_allclose(single_site @ canonical[side], np.linalg.inv(source_wrist), atol=1e-9)
            wrist_mounts.append(current_wrist_tcp @ mounts[side])
            fixed.append(current_wrist_tcp)
        world_valid[:] = True
        audit.update(
            model_path=str(single),
            model_sha256=hashlib.sha256(single.read_bytes()).hexdigest(),
            station_path=str(station),
            station_sha256=hashlib.sha256(station.read_bytes()).hexdigest(),
            mjcf_joint_chain=chain,
            release_wrist_from_old_wrist=B,
            release_eef_from_tcp=fixed,
            contact_geometry="Station MJCF mean of left/right fingertip sphere_collision centers; explicitly checked against inverse release transform",
            T_tcp_camera_opencv=mounts,
        )
        status = "mjcf_contact_tcp_restored"
        endpoint = "Current axis-rectified link6 wrist -> inverse 0826 basis -> inverse recorded 0804 contact-to-wrist transform; contact center independently recovered from station MJCF."
        limitations += [
            "ABC model-frame arm bases are declared station geometry, not an independent installation survey. D405 wrist pose is derived from EEF + MJCF mount; agreement is a consistency check."
        ]
    else:
        urdf = ASSETS / (
            "robochallenge_v1/robochallenge_aloha/interndata_split_aloha_piper100/split_aloha_mid_360_with_piper.urdf"
            if is_aloha
            else "dos-w1/dos-w1.urdf"
        )
        model = URDFModel(urdf)
        calib = json.loads((ASSETS / "dos-w1/default_camera_parameters.json").read_text())
        chain_evidence = []
        fixed = []
        if is_aloha:
            heldout_path = FACTORY / "data_curation/outputs/robochallenge_aloha_dynamic_base_heldout.json"
            heldout = json.loads(heldout_path.read_text())["reports"][0]
            audit["anchor_convention_source"] = str(heldout_path)
        else:
            history = next(
                x for x in meta["rectification_history"] if x.get("schema_version") == "eef_tcp_to_wrist_0804_v1"
            )
        for i, side in enumerate(SIDES):
            base = f"{side}/arm_base" if is_aloha else f"{side}_base_link"
            flange = f"{side}/link6" if is_aloha else f"{side}_eef_base_link"
            vals = {(f"{side}/joint{j + 1}" if is_aloha else f"{side}_joint{j + 1}"): q[:, i * 7 + j] for j in range(6)}
            raw_fk = model.fk(vals, flange, base)
            if is_aloha:
                h = heldout["arms"][side]["best_validation"]["transform"]
                delta = tx(h["xyz"])
                v = np.array(h["quaternion_wxyz"])
                delta[:3, :3] = Rotation.from_quat(v[[1, 2, 3, 0]]).as_matrix()
                # The heldout profile already consumes canonical (axis-remapped) EEF.
                fk_arm[:, i] = raw_fk @ delta
                camera = np.linalg.inv(np.array(calib["calibration"]["aloha"][f"cam_{side}_wrist"]))
                wrist_mounts.append(np.linalg.inv(delta) @ camera)
                fixed.append(None)
            else:
                source_wrist = np.array(history["transforms"][i]["T_source_wrist_row_major"]).reshape(4, 4)
                current_wrist_tcp = np.linalg.inv(source_wrist)
                tcp[:, i] = eef[:, i] @ current_wrist_tcp
                # Exact URDF flange convention plus recorded TCP restoration; no fitted global transform.
                fk_arm[:, i] = raw_fk
                end = model.fk({}, f"{side}_end_link", flange)[0]
                camera = np.array(calib["calibration"]["dos_w1"][f"cam_{side}_wrist"])
                wrist_mounts.append(end @ tx([0, 0, 0.02]) @ camera)
                fixed.append(current_wrist_tcp)
            if i == 0:
                fk[:, i] = fk_arm[:, i]
                world_valid[:, i] = True
                if not is_aloha:
                    fk_tcp[:, i] = fk[:, i] @ current_wrist_tcp
            chain_evidence.append(
                {
                    "side": side,
                    "base": base,
                    "flange": flange,
                    "chain": [ET.tostring(j, encoding="unicode") for j in model.chain(flange, base)],
                }
            )
        audit.update(
            model_path=str(urdf),
            model_sha256=model.sha256,
            urdf_chains=chain_evidence,
            release_eef_from_tcp=fixed,
            source_camera_candidate=meta.get("camera_candidate_backfill"),
            eef_from_camera=wrist_mounts,
            right_world_base_status="Unavailable: Stage1 per-episode camera-to-both-arm-base calibration absent from mounted release; no GT-fitted base is used.",
        )
        if is_aloha:
            status = "unresolved_contact_tcp"
            endpoint = "Current EEF is source link6 pose; no verified contact TCP. URDF finger origins and an incompatible RobotWin config were investigated and excluded as contact TCP."
            audit["cad_contact_inspection"] = aloha_mesh_diagnostics(model)
            audit["tcp_recovery_attempts"] = [
                {
                    "asset": "matched Piper100 URDF",
                    "finding": "No tcp/tool/grasp fixed link. joint7 and joint8 origins share local z=0.13503 m, but these are prismatic finger frame origins, not contact pad definitions.",
                    "candidate_offset_link6_m": [0, 0, 0.13503],
                    "decision": "retain origin diagnostic only; not a verified contact TCP",
                },
                {
                    "asset": str(ASSETS / "robochallenge_v1/robochallenge_aloha/robotwin_aloha_agilex/config.yml"),
                    "finding": "gripper_bias=0.12 belongs to arx5_description_isaac.urdf / fl_joint* model, not matched Piper100 URDF",
                    "decision": "rejected for this embodiment; do not transplant scalar offset",
                },
            ]
            limitations += [
                "ALOHA TCP remains masked: matching URDF has finger origins but no calibrated/contact-defined tool frame. Its local anchor convention comes from an existing held-out dataset fit and is identified as such."
            ]
        else:
            status = "source_tcp_restored"
            endpoint = "Inverse recorded 0804 TCP-to-URDF-eef_base_link transform restores the source gripper contact center. The 8.9 mm end_link correction is prior held-out data evidence, not a pure URDF constant."
        limitations += [
            "Challenge dataset world is left_arm_base. Right arm world FK is masked because per-episode independent inter-arm base calibration is unavailable. Both arms retain initial-EEF-local motion comparison.",
            "Wrist extrinsics are explicitly candidate mounts. Reconstructed as source EEF @ fixed mount; this avoids the known wrong fixed right-base candidate but is not independent hand-eye calibration.",
        ]
    local = np.linalg.inv(fk_arm[0])[None] @ fk_arm
    label_local = np.linalg.inv(eef[0])[None] @ eef
    audit["local_fk_vs_label"] = {s: summarize_pose(label_local[:, i], local[:, i]) for i, s in enumerate(SIDES)}
    audit["world_fk_vs_label"] = {
        s: summarize_pose(eef[:, i], fk[:, i]) for i, s in enumerate(SIDES) if world_valid[:, i].all()
    }
    audit["tcp_world_fk_vs_restored_label"] = {
        s: summarize_pose(tcp[:, i], fk_tcp[:, i]) for i, s in enumerate(SIDES) if np.isfinite(fk_tcp[:, i]).all()
    }
    return dict(
        eef=eef,
        tcp=tcp,
        fk=fk,
        fk_tcp=fk_tcp,
        fk_local=local,
        label_local=label_local,
        joint=q,
        world_valid=world_valid,
        wrist_mounts=wrist_mounts,
        audit=audit,
        tcp_status=status,
        endpoint=endpoint,
        limitations=limitations,
    )


def extract(name, output, anchor=30, count=180, decode=True):
    title, dataset = SOURCES[name]
    root = output / name
    root.mkdir(parents=True, exist_ok=True)
    profile = root / "source.yaml"
    profile.write_text(f"dataset_dir: {dataset}\n")
    info = json.loads((dataset / "meta/info.json").read_text())
    keys = [
        "observation.state",
        "observation.states.eef",
        "observation.states.camera.intrinsic",
        "observation.states.camera.world_to_camera",
    ]
    for side in SIDES:
        keys += [f"observation.states.cameras.cam_{side}_wrist.{x}" for x in ["intrinsic", "world_to_camera"]]
    _, _, episode, frame = load_episode(profile, episode_index=0, columns=keys)
    window = frame[(frame.frame_index >= anchor) & (frame.frame_index < anchor + count)]
    indices = window.frame_index.to_numpy(int)
    np.testing.assert_array_equal(indices, np.arange(anchor, anchor + count))
    n = len(indices)
    g = geometry(name, info, window)
    is_abc = name == "abc_130k_yam"
    family = "abc_130k" if is_abc else "robochallenge_v2"
    base = dict(
        name=title,
        dataset_family=name,
        dataset_dir=str(dataset),
        episode=0,
        episode_index=0,
        frame_indices=indices.tolist(),
        model="pinhole",
        tcp_status=g["tcp_status"],
        endpoint=g["endpoint"],
        limitations=" ".join(g["limitations"]),
        clip_label=f"{dataset.parent.name}/{dataset.name} · episode 0 · frames {anchor}–{anchor + count - 1}",
        origin_diagnostics=[],
        pose_frame="abc_yam_mjcf_model_frame" if is_abc else "left_arm_base",
        pose_units="meter",
    )
    if is_abc:
        eligibility = jsonl_episode(dataset / "meta/camera_geometry_eligibility.jsonl")
        broadcast = jsonl_episode(dataset / "meta/geometry_broadcast_per_episode.jsonl")
        raw_intrinsics = jsonl_episode(dataset / "meta/camera_intrinsics_raw_per_episode.jsonl")
        base.update(
            camera_geometry_eligibility=eligibility,
            camera_broadcast_metadata=broadcast,
            dataset_resolved_path=str(dataset.resolve()),
            release_alias_note="V0_3/abc_130k is a symlink to the same V0_2 release",
        )
        g["audit"]["episode_camera_geometry_eligibility"] = eligibility
        g["audit"]["episode_broadcast_metadata"] = broadcast
        conversion_path = dataset / "meta/source_stage2_conversion_report.json"
        conversion = json.loads(conversion_path.read_text())
        probe = next(
            x for x in conversion["eef_probe_records"] if Path(x["episode_dir"]).name == broadcast["episode_id"]
        )
        g["audit"]["eef_provenance"] = {
            "source": str(conversion_path),
            "episode_probe": probe,
            "interpretation": "source_pose_available"
            if probe["eef_probe"]["state_pose_available"]
            else "joint_fk_fallback_not_independent",
        }
    wrist_summary = dict(name=title, dataset_dir=str(dataset), episode_index=0, views={})
    for view in ["head", *SIDES]:
        camera_key = "observation.images.cam_high" if view == "head" else f"observation.images.cam_{view}_wrist"
        height, width = info["features"][camera_key]["shape"][:2]
        prefix = "observation.states.camera" if view == "head" else f"observation.states.cameras.cam_{view}_wrist"
        k = np.stack(window[prefix + ".intrinsic"]).reshape(n, 3, 3)
        stored_camera = np.stack(window[prefix + ".world_to_camera"]).reshape(n, 4, 4)
        camera = stored_camera.copy()
        meta = dict(
            base,
            image_width=width,
            image_height=height,
            nativewh=[width, height],
            camera_key=camera_key,
            intrinsic_key=prefix + ".intrinsic",
            extrinsic_key=prefix + ".world_to_camera",
            video_path=str(
                dataset
                / info["video_path"].format(
                    video_key=camera_key,
                    chunk_index=int(episode[f"videos/{camera_key}/chunk_index"]),
                    file_index=int(episode[f"videos/{camera_key}/file_index"]),
                )
            ),
            video_episode_from_timestamp=float(episode.get(f"videos/{camera_key}/from_timestamp", 0)),
        )
        distortion = np.empty((n, 0))
        if is_abc:
            raw = raw_intrinsics["camera_intrinsics"][camera_key]
            meta.update(
                source_distortion_model=raw.get("distortion_model"),
                source_distortion_coefficients=raw.get("D"),
                lens_note="Pinhole preview uses released K. Inverse-Brown-Conrady coefficients are retained in metadata, not misapplied as forward Brown distortion. Pixel-level distortion accuracy is not claimed.",
            )
        if view != "head":
            side_index = SIDES.index(view)
            # Derive each moving optical camera from its own current EEF, same TCP in all views.
            camera = np.linalg.inv(g["eef"][:, side_index] @ g["wrist_mounts"][side_index])
            meta.update(
                side=view,
                status="mjcf_derived_preview" if is_abc else "cad_derived_preview",
                video_available=True,
                reason="MJCF D405 fixed mount; source-pose dependent consistency chain."
                if is_abc
                else "Source EEF plus documented fixed candidate mount; corrected old right-arm fixed-base problem, still not physical calibration.",
                calibration_status="derived_candidate",
                eef_from_camera=g["wrist_mounts"][side_index].tolist(),
                extrinsic_key="derived: inverse(source_eef @ fixed_eef_from_camera)",
            )
            g["audit"].setdefault("wrist_rederived_vs_stored", {})[view] = summarize_pose(
                np.linalg.inv(stored_camera), np.linalg.inv(camera)
            )
        valid = camera_validity(k, camera)
        if (
            is_abc
            and not eligibility[
                "strict_head_camera_eef_eligible" if view == "head" else "strict_wrist_camera_eef_eligible"
            ]
        ):
            valid[:] = False
        arrays = dict(frame_indices=indices, K=k, world_to_camera=camera, camera_valid=valid, distortion=distortion)
        if view == "head":
            target = root / "video_samples" / name
            arrays.update(
                eef_world=g["eef"],
                tcp_world=g["tcp"],
                eef_valid=np.isfinite(g["eef"]).all((-1, -2)),
                tcp_valid=np.isfinite(g["tcp"]).all((-1, -2)),
            )
        else:
            target = root / "wrist_samples" / name / view
            meta.update(projection_available=bool(valid.any()), valid_camera_frames=int(valid.sum()))
        target.mkdir(parents=True, exist_ok=True)
        if decode:
            meta.update(decode_sequence(meta, target))
        np.savez_compressed(target / "arrays.npz", **arrays)
        write_json(target / "metadata.json", meta)
        if view != "head":
            wrist_summary["views"][view] = meta
    write_json(root / "wrist_samples" / name / "summary.json", wrist_summary)
    joint = g["joint"].copy()
    joint_valid = np.ones(joint.shape, bool)
    joint_valid[:, [6, 13]] = False
    joint[:, [6, 13]] = np.nan
    tracks = dict(
        frame_indices=indices,
        fk_joint=joint,
        fk_joint_valid=joint_valid,
        fk_eef_world=g["fk"],
        fk_eef_valid=g["world_valid"],
        fk_tcp_world=g["fk_tcp"],
        fk_tcp_valid=np.isfinite(g["fk_tcp"]).all((-1, -2)),
        fk_local_pose=g["fk_local"],
        fk_local_valid=np.ones((n, 2), bool),
        label_local_pose=g["label_local"],
        label_local_valid=np.ones((n, 2), bool),
    )
    target = root / "fk_tracks" / name
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "arrays.npz", **tracks)
    write_json(
        target / "metadata.json",
        dict(
            status="model_world_consistency" if is_abc else "left_world_and_bilateral_local",
            local_frame="each arm first released EEF frame",
            world_frame_status="declared MJCF model frame"
            if is_abc
            else "left arm valid; right unavailable independent episode mount",
            limitations=g["limitations"],
            source_metadata=dict(dataset_dir=str(dataset), episode_index=0),
            tcp_status=g["tcp_status"],
            independent_world_calibration=False,
            model_path=g["audit"]["model_path"],
            model_sha256=g["audit"]["model_sha256"],
            source_eef_key="observation.states.eef",
            audit="../../geometry_audit.json",
        ),
    )
    write_json(root / "geometry_audit.json", g["audit"])
    write_json(
        root / "entry.json",
        dict(
            id=name,
            title=title,
            family=name,
            root=f"expanded_sources/{name}",
            source_group=family,
            reason="180 consecutive source head and wrist frames; release-aware TCP restoration and independent joint model evaluation with explicit geometry limits.",
        ),
    )
    print(
        json.dumps(
            {
                "id": name,
                "frames": n,
                "tcp_status": g["tcp_status"],
                "audit": g["audit"]["world_fk_vs_label"],
                "local": g["audit"]["local_fk_vs_label"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return root


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/fk_consistency/expanded_sources")
    p.add_argument("--names", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    p.add_argument("--no-decode", action="store_true")
    p.add_argument("--anchor", type=int, default=30)
    p.add_argument("--count", type=int, default=180)
    a = p.parse_args()
    for name in a.names:
        extract(name, a.output, a.anchor, a.count, not a.no_decode)
