"""Reproduce named-link FK and pad-CAD evidence for the visual TCP preview."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.data.fk_audit_agibot import AXIS_REMAP, URDF_DIR
from scripts.data.tcp_endpoint_policy import AGIBOT_VISUAL_TCP_X_M
from scripts.data.urdf_fk_audit import URDFModel


def derive_evidence():
    remap = np.eye(4)
    remap[:3, :3] = AXIS_REMAP
    evidence = dict(
        formula='T_flange_target = inv(T_base_flange) @ T_base_target; T_canonical_target = inv(A4) @ T_flange_target',
        canonical_axis_remap=remap.tolist(),
        library_check='urdfpy link_fk returns base-relative link poses (official API documentation). urdfpy is not installed here; executed the repository URDFModel FK, comparing direct-chain and base-relative formulas.',
        api_reference='https://urdfpy.readthedocs.io/en/latest/generated/urdfpy.URDF.html#urdfpy.URDF.link_fk',
        cfg_warning='Empty cfg is a default joint configuration, not the recorded gripper opening. A fixed flange-to-target chain is configuration-independent; a finger-pad chain is not.',
        models={},
    )
    for filename in ['g1_120s_agibot_beta_audit.urdf', 'G1_120s.urdf']:
        model = URDFModel(URDF_DIR/filename)
        result = dict(sha256=model.sha256, sides={})
        zero = {name: 0. for name, node in model.joints.items()
                if node.get('type') != 'fixed' and node.find('mimic') is None}
        rng = np.random.default_rng(42)
        for side in 'lr':
            flange, center = f'arm_{side}_end_link', f'gripper_{side}_center_link'
            assert all(j.get('type') == 'fixed' for j in model.chain(center, flange))
            direct = model.fk({}, center, base_link=flange)[0]
            errors = []
            for draw in range(5):
                cfg = zero if draw == 0 else {name: rng.uniform(-.3, .3) for name in zero}
                from_base = np.linalg.inv(model.fk(cfg, flange)[0]) @ model.fk(cfg, center)[0]
                errors.append(float(np.max(np.abs(direct-from_base))))
            pads = []
            # Audit URDF retains unavailable package:// STL visual references.
            # The manifest's complete visual URDF supplies the archived meshes.
            for finger in (['outer', 'inner'] if filename == 'G1_120s.urdf' else []):
                link = f'gripper_{side}_{finger}_link5'
                visual = model.xml.find(f"link[@name='{link}']/visual")
                origin = visual.find('origin')
                visual_t = np.eye(4)
                if origin is not None:
                    visual_t[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                    visual_t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
                mesh = visual.find('geometry/mesh')
                vertices = np.asarray([list(map(float, line.split()[1:4]))
                                       for line in (URDF_DIR/mesh.get('filename')).read_text().splitlines()
                                       if line.startswith('v ')])
                vertices *= np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                pose = np.linalg.inv(remap) @ model.fk(zero, link, base_link=flange)[0] @ visual_t
                points = vertices @ pose[:3, :3].T + pose[:3, 3]
                pads.append(dict(link=link, min_m=points.min(0).tolist(), max_m=points.max(0).tolist(),
                                 bbox_center_m=((points.min(0)+points.max(0))/2).tolist()))
            result['sides'][side] = dict(flange=flange, named_center=center,
                                        flange_to_named_center=direct.tolist(),
                                        canonical_to_named_center=(np.linalg.inv(remap)@direct).tolist(),
                                        formula_vs_direct_max_error=max(errors),
                                        zero_joint_pad_geometry=pads,
                                        zero_joint_pad_midpoint_m=np.mean([p['bbox_center_m'] for p in pads], axis=0).tolist() if pads else None)
        evidence['models'][filename] = result
    g2 = URDFModel(URDF_DIR/'agibot26_G2.urdf')
    evidence['g2_check'] = dict(sha256=g2.sha256,
                               gripper_links=[link for link in sorted(g2.links) if 'gripper' in link],
                               reason='This is a different robot model and lacks the G1 gripper-center/pad links; do not substitute it for the source A2D/Beta wrist audit model.')
    transform = np.eye(4)
    transform[0, 3] = AGIBOT_VISUAL_TCP_X_M
    evidence['visual_definition'] = dict(
        status='CAD-informed manual visual preview; not independent physical calibration',
        canonical_flange_to_virtual_tcp=transform.tolist(),
        rotation='Identity by convention, retaining canonical wrist axes',
        source_raw_flange_to_virtual_tcp=(remap@transform).tolist(),
        reviewed_tasks=[362, 365, 410], reviewed_episode=0, reviewed_frames=[216, 280, 330, 420],
        compared_local_x_m=[.100, .128, .207056, .229306, .257306, .280],
        fine_tuning_local_x_m=[.160, .190, .200, .210, .245], previous_local_x_m=.245,
        selected_local_x_m=AGIBOT_VISUAL_TCP_X_M,
        note='Head-only 240/245mm choices did not align wrist images. Multi-view inspection selected a shorter 200mm preview, but left/right wrist landmark fits disagree (roughly 192-199 versus 155-167mm). Camera extrinsics were rigidified from source annotations, not independently calibrated. Do not interpret 200mm as recovered physical length or change the point per camera. See multiview_offset_audit.json.',
    )
    return evidence


if __name__ == '__main__':
    result = derive_evidence()
    output = Path('outputs/fk_consistency/tcp_origin_audit/agibotworld/visual_tcp_geometry.json')
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    for name, model in result['models'].items():
        print(name, {side: value['zero_joint_pad_midpoint_m'] for side, value in model['sides'].items()})
    print(output)
