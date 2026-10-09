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
            or not np.array_equal(matrix[3], [0, 0, 0, 1])):
        raise ValueError("transform must be a finite homogeneous 4x4 matrix")
    return points @ matrix[:3, :3].T + matrix[:3, 3]
