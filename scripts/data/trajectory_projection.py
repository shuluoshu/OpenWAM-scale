"""Explicit calibrated camera models for trajectory reports (NumPy only).

Camera coordinates use positive Z forward, X right and Y down. Projection
validity means finite, in front of the camera and inside the model's domain;
it does not imply image bounds, lack of occlusion, or calibration accuracy.
"""

import numpy as np


def _points_array(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError("points must have shape (..., 3)")
    return points


def project_points(points_camera, K, model="pinhole", distortion=None):
    """Return ``(pixels[..., 2], valid[...])``; invalid pixels are NaN.

    ``K`` is a finite 3x3 intrinsic matrix with positive focal lengths,
    optional K[0,1] skew, and last row [0,0,1]. Supported model parameters:

    * pinhole: no distortion (None or empty).
    * brown_conrady: OpenCV [k1, k2, p1, p2, optional k3].
    * fisheye: OpenCV [k1, k2, k3, k4] theta polynomial.
    * double_sphere: [xi, alpha], alpha in [0,1], using equations 40-45
      of https://arxiv.org/html/1807.08957v2#S3. Additionally requires
      positive Z for this report, even where the lens can see behind itself.

    Parameters are never inferred from a source name or padded with invented
    calibration. The optional Brown k3 is zero in the four-coefficient model.
    No image clipping or arbitrary near plane is applied.
    """
    points = _points_array(points_camera)
    intrinsic = np.asarray(K, dtype=np.float64)
    if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or intrinsic[1, 0] != 0
            or not np.array_equal(intrinsic[2], [0, 0, 1])):
        raise ValueError("K must be a finite standard 3x3 intrinsic matrix with positive focal lengths")
    sizes = {"pinhole": (0,), "brown_conrady": (4, 5), "fisheye": (4,), "double_sphere": (2,)}
    if model not in sizes:
        raise ValueError(f"unsupported camera model: {model!r}")
    coefficients = np.asarray([] if distortion is None else distortion, dtype=np.float64)
    if (coefficients.ndim != 1 or coefficients.size not in sizes[model]
            or not np.isfinite(coefficients).all()):
        raise ValueError(f"{model} requires finite distortion with coefficient count {sizes[model]}")
    if model == "double_sphere":
        xi, alpha = coefficients
        if not 0 <= alpha <= 1:
            raise ValueError("double_sphere alpha must be in [0, 1]")
        w1 = alpha / (1 - alpha) if alpha <= .5 else (1 - alpha) / alpha
        # hypot form avoids cancellation at xi ~= -w1.
        domain_norm = np.hypot(xi + w1, np.sqrt(1 - w1 * w1))
        if domain_norm == 0:
            raise ValueError("double_sphere calibration has a singular projection domain")
        w2 = (xi + w1) / domain_norm

    flat = points.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1) & (flat[:, 2] > 0)
    indices = np.flatnonzero(valid)
    pixels = np.full((len(flat), 2), np.nan)
    # Only process candidate points. Overflow of extreme finite data also
    # becomes an invalid projection, never a clipped or fabricated pixel.
    with np.errstate(all="ignore"):
        x, y, z = flat[indices].T
        domain = np.ones(len(indices), dtype=bool)
        if model == "double_sphere":
            scale = np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z)])
            x, y, z = x / scale, y / scale, z / scale
            d1 = np.hypot(np.hypot(x, y), z)
            shifted_z = xi * d1 + z
            d2 = np.hypot(np.hypot(x, y), shifted_z)
            denominator = alpha * d2 + (1 - alpha) * shifted_z
            domain = (z > -w2 * d1) & (denominator > 0)
            normalized = np.column_stack((x / denominator, y / denominator))
        elif model == "fisheye":
            radius = np.hypot(x, y)
            theta = np.arctan2(radius, z)
            t2 = theta * theta
            k1, k2, k3, k4 = coefficients
            theta_d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
            factor = np.divide(theta_d, radius, out=np.zeros_like(radius), where=radius != 0)
            normalized = np.column_stack((x * factor, y * factor))
        else:
            xn, yn = x / z, y / z
            if model == "brown_conrady":
                k1, k2, p1, p2 = coefficients[:4]
                k3 = coefficients[4] if len(coefficients) == 5 else 0.
                r2 = xn * xn + yn * yn
                radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
                xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
                yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
                xn, yn = xd, yd
            normalized = np.column_stack((xn, yn))
        projected = normalized @ intrinsic[:2, :2].T + intrinsic[:2, 2]
        accepted = domain & np.isfinite(projected).all(axis=1)
    valid[indices] = accepted
    pixels[indices[accepted]] = projected[accepted]
    return pixels.reshape(points.shape[:-1] + (2,)), valid.reshape(points.shape[:-1])


def transform_points(transform, points):
    """Apply one source-to-target homogeneous transform to ``points[...,3]``.

    For column-vector notation this computes R @ p + t. The caller supplies
    the transform direction explicitly; this helper never inverts a pose.
    """
    matrix = np.asarray(transform, dtype=np.float64)
    points = _points_array(points)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            # Float32 inversion/export can perturb the homogeneous row by a
            # few ULPs (EgoDex records 0.9999997616). Reject projective matrices,
            # not normal serialization roundoff; leave recorded R and t intact.
            or not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-6)):
        raise ValueError("transform must be a finite homogeneous 4x4 matrix")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def project_goal_tcp(tcp_world, world_to_camera, K, source_wh, *, camera_model='pinhole',
                     distortion=None, tcp_valid=None, visibility=None):
    """Project goal-row endpoints with goal-row camera; retain independent masks.

    Pixel centers follow PIL resize to the 320x256 head tile. Out-of-frame UV
    is zero only as a safe masked payload, never clipped into supervision.
    """
    width, height = source_wh
    if width <= 0 or height <= 0:
        raise ValueError('source image size must be positive')
    xyz = transform_points(world_to_camera, tcp_world)
    xy, valid = project_points(xyz, K, model=camera_model, distortion=distortion)
    if tcp_valid is not None:
        valid &= np.asarray(tcp_valid, dtype=bool)
    scaled = (xy + .5) * [320. / width, 256. / height] - .5
    inside = (valid & (xy[..., 0] >= 0) & (xy[..., 0] <= width-1)
              & (xy[..., 1] >= 0) & (xy[..., 1] <= height-1)
              & (scaled[..., 0] >= 0) & (scaled[..., 0] <= 319)
              & (scaled[..., 1] >= 0) & (scaled[..., 1] <= 255))
    uv = np.where(inside[..., None], scaled / [319., 255.], 0.).astype(np.float32)
    visibility = np.full(valid.shape, -1, np.int8) if visibility is None else np.asarray(visibility, np.int8)
    if visibility.shape != valid.shape or not np.isin(visibility, [-1,0,1]).all():
        raise ValueError('visibility must match points and contain only -1,0,1')
    return dict(uv=uv, projection_valid=valid, in_frame=inside, visibility=visibility, xyz_camera=xyz)


def goal_source_tcp(family, row, *, tcp_from_eef=None):
    """Explicit source endpoint adapters using the selected TCP conventions.

    Tracking confidence >=.8 is this adapter's policy, not calibration proof.
    Output ordering is always left, right. Missing one hand preserves the other.
    """
    tcp = np.full((2,3), np.nan); valid = np.zeros(2, bool)
    provenance = dict(definition_id='missing_verified_tcp', used_fields=[],
                      transforms=[], missing_reasons=[], tcp_world=tcp, tcp_valid=valid)
    humans = {'LightWheel':('observation.states.hand.position',21,4,8,0),
              'mecka':('observation.states.hand.position',21,4,8,0),
              'egoverse':('observation.states.joints',21,0,1,5),
              'egodex':('observation.states.finger.tip.position',5,4,0,None)}
    try:
        if family in humans:
            key,n,thumb,index,wrist = humans[family]
            raw=row[key]
            hands=raw if len(raw)==2 else np.asarray(raw,float).reshape(2,n,3)
            bases=None
            if wrist is None:
                raw_base=row['observation.states.hand.position']
                bases=raw_base if len(raw_base)==2 else np.asarray(raw_base,float).reshape(2,3)
            for i,side in enumerate(('left','right')):
                try:
                    hand=np.asarray(hands[i],float)
                    if hand.size!=n*3:raise ValueError(f'hand must contain {n} xyz points')
                    hand=hand.reshape(n,3)
                    base=hand[wrist] if wrist is not None else np.asarray(bases[i],float)
                    if base.shape!=(3,):raise ValueError('wrist must have shape (3,)')
                    tcp[i]=(hand[thumb]+hand[index])/2
                    valid[i]=(np.isfinite(hand[[thumb,index]]).all() and np.isfinite(base).all()
                              and np.linalg.norm(tcp[i]-base)>1e-6)
                except (KeyError, ValueError, TypeError) as exc:
                    provenance['missing_reasons'].append(f'{side}: missing_or_invalid_tcp: {exc}')
            provenance.update(definition_id='human_thumb_index_midpoint_v1',used_fields=[key],
                              transforms=['world thumb/index fingertip midpoint; left then right'],
                              tracking_policy='finite fingertips, nondegenerate wrist-to-pinch, hand.valid>0.5, confidence>=0.8 when present')
            if wrist is None: provenance['used_fields'].append('observation.states.hand.position')
        elif family == 'interndata_a1':
            from scipy.spatial.transform import Rotation
            def pose(value):
                value=np.asarray(value,float)
                if value.shape!=(7,) or not np.isfinite(value).all():
                    raise ValueError('pose must be seven finite xyz+wxyz values')
                out=np.eye(4)
                out[:3,:3]=Rotation.from_quat(value[[4,5,6,3]]).as_matrix();out[:3,3]=value[:3]
                return out
            # A malformed shared reference invalidates both; hand poses do not.
            reference=pose(row['states.robot_to_env_pose'])
            for side,name in enumerate(('left','right')):
                key=f'states.{name}_tcp_to_robot_pose'
                try:
                    tcp[side]=(reference@pose(row[key]))[:3,3]
                    valid[side]=np.isfinite(tcp[side]).all()
                except (KeyError, ValueError, TypeError) as exc:
                    provenance['missing_reasons'].append(f'{name}: missing_or_invalid_tcp: {exc}')
            provenance.update(definition_id='interndata_native_tcp_env_v1',
                              used_fields=['states.robot_to_env_pose','states.left_tcp_to_robot_pose','states.right_tcp_to_robot_pose'],
                              transforms=['T_env_robot @ T_robot_TCP; source xyz+wxyz'])
        elif family in ('morphirobot0930', 'Genrobot') or tcp_from_eef:
            raw=row['observation.states.eef']
            # Keep separately recorded/ragged hands separate until validation.
            values=raw if len(raw)==2 else np.asarray(raw,float).reshape(2,10)
            raw_mask=row.get('observation.states.eef.mask')
            masks=(None if raw_mask is None else raw_mask if len(raw_mask)==2
                   else np.asarray(raw_mask,float).reshape(2,10))
            for i,side in enumerate(('left','right')):
                try:
                    value=np.asarray(values[i],float)
                    if value.shape!=(10,) or not np.isfinite(value[:9]).all():
                        raise ValueError('EEF pose must contain ten values with finite xyz+rotation')
                    a=value[3:6].copy();b=value[6:9].copy()
                    if np.linalg.norm(a)<=1e-7:raise ValueError('degenerate EEF rotation')
                    a/=np.linalg.norm(a);b-=(a*b).sum()*a
                    if np.linalg.norm(b)<=1e-7:raise ValueError('degenerate EEF rotation')
                    b/=np.linalg.norm(b);rotation=np.stack((a,b,np.cross(a,b)),axis=-1)
                    offset=np.array([.23116,0,0])
                    if family=='Genrobot' and not tcp_from_eef:
                        # Same finger-device convention as the existing multiview
                        # renderer; this is not a human fingertip landmark.
                        offset=np.array([.1015,-.0003,-.0398])
                    elif family!='morphirobot0930':
                        fixed=np.asarray(tcp_from_eef[side],float)
                        if (fixed.shape!=(4,4) or not np.isfinite(fixed).all()
                                or not np.array_equal(fixed[3],[0,0,0,1])):
                            raise ValueError('invalid recorded TCP transform')
                        offset=fixed[:3,3]
                    tcp[i]=value[:3]+rotation@offset
                    valid[i]=np.isfinite(tcp[i]).all()
                    if masks is not None:
                        mask=np.asarray(masks[i],float)
                        if mask.shape!=(10,):raise ValueError('invalid EEF mask')
                        valid[i] &= (mask[:9]>.5).all()
                    if family=='Genrobot':
                        key=f'raw.robot{i+1}.finger.eef_pose.alignment_valid'
                        if key in row:
                            alignment=np.asarray(row[key],float).reshape(-1)
                            if alignment.size!=1 or not np.isfinite(alignment).all():
                                raise ValueError('invalid device pose alignment flag')
                            valid[i] &= alignment[0]>.5
                            provenance['used_fields'].append(key)
                except (KeyError, ValueError, TypeError) as exc:
                    valid[i]=False
                    provenance['missing_reasons'].append(f'{side}: missing_or_invalid_tcp: {exc}')
            if family == 'morphirobot0930':
                provenance.update(definition_id='morphi_arm08_z_0.23116_released_eef_x_v1',
                                  transforms=['p_EEF + R_EEF @ [0.23116,0,0]; arm08 original +Z'],
                                  evidence=['morphi_eef_axes.py','deploy_mcap_analysis/references/urdf_tcp.md'])
            elif family=='Genrobot' and not tcp_from_eef:
                provenance.update(definition_id='genrobot_finger_device_tcp_offset_v1',
                                  transforms=['p_TCP = p_EEF + R_EEF @ [0.1015,-0.0003,-0.0398] m'],
                                  evidence=['outputs/tcp_multiview_6s_20261017/build.py:source_endpoints',
                                            'outputs/fk_consistency/genrobot_wrist_audit/audit.json'],
                                  endpoint_semantics='selected finger-device TCP convention; not human thumb/index midpoint',
                                  independent_physical_calibration_verified=False)
            else:
                provenance.update(definition_id='recorded_contact_tcp_restored_v1',
                                  transforms=['T_reference_EEF @ T_EEF_TCP'],fixed_transforms=tcp_from_eef)
            provenance['used_fields'].insert(0,'observation.states.eef')
            if masks is not None:provenance['used_fields'].append('observation.states.eef.mask')
        else:
            provenance['missing_reasons'].append('No reviewed source TCP chain; report candidates excluded')
        for key,threshold in [('observation.states.hand.valid',.5),('observation.states.hand.confidence',.8)]:
            if family in humans and key in row:
                good=np.asarray(row[key],float).reshape(2)
                valid &= good>=threshold if 'confidence' in key else good>threshold
                provenance['used_fields'].append(key)
        for key in ('canonical.frame_valid','source.frame_validity'):
            if key in row:
                frame_valid=np.asarray(row[key]).reshape(-1)
                if not frame_valid.size:raise ValueError(f'{key}: empty shared frame validity')
                valid &= bool(frame_valid[0]>.5)
        valid &= np.isfinite(tcp).all(-1)
    except (KeyError, ValueError, TypeError) as exc:
        valid[:]=False;provenance['missing_reasons'].append(f'missing_or_invalid_tcp: {exc}')
    tcp[~valid]=np.nan
    provenance['missing_reasons'].extend(f'{side}: missing/invalid endpoint or tracking' for side,good in zip(('left','right'),valid) if not good)
    return provenance
