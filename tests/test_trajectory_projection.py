"""Independent numerical references for the report's camera projections."""

import cv2
import numpy as np
import pytest

from scripts.data.trajectory_projection import project_points, transform_points

K = np.array([[430., 0., 320.], [0., 415., 240.], [0., 0., 1.]])
POINTS = np.array([[0., 0., 2.], [.4, -.3, 1.2], [-.8, .6, .7], [2., 1., .2]])
MODELS = [("pinhole", None), ("brown_conrady", [.1, -.02, .003, -.004, .001]),
          ("fisheye", [.1, -.02, .003, -.004]), ("double_sphere", [.4, .6])]


@pytest.mark.parametrize("model,distortion", MODELS[:2] + [("brown_conrady", [.1, -.02, .003, -.004])])
def test_perspective_models_match_opencv(model, distortion):
    expected, _ = cv2.projectPoints(POINTS, np.zeros(3), np.zeros(3), K,
                                    None if distortion is None else np.array(distortion))
    pixels, valid = project_points(POINTS, K, model, distortion)
    np.testing.assert_allclose(pixels, expected[:, 0], rtol=1e-12, atol=1e-10)
    assert valid.all()


def test_fisheye_matches_opencv_including_optical_axis():
    distortion = np.array([.1, -.02, .003, -.004])
    expected, _ = cv2.fisheye.projectPoints(POINTS[:, None], np.zeros(3), np.zeros(3), K, distortion)
    pixels, valid = project_points(POINTS, K, "fisheye", distortion)
    np.testing.assert_allclose(pixels, expected[:, 0], rtol=1e-12, atol=1e-10)
    assert valid.all()


@pytest.mark.parametrize("alpha", [0., .2, .5, .8, 1.])
def test_double_sphere_roundtrips_independent_analytic_unprojection(alpha):
    # Inverse from Usenko et al., equations 46-50; does not repeat projection.
    xi = .4
    pixels, valid = project_points(POINTS, K, "double_sphere", [xi, alpha])
    m = (pixels - K[:2, 2]) / np.diag(K)[:2]
    r2 = np.sum(m * m, axis=-1)
    mz = (1 - alpha * alpha * r2) / (alpha * np.sqrt(1 - (2 * alpha - 1) * r2) + 1 - alpha)
    scale = (mz * xi + np.sqrt(mz * mz + (1 - xi * xi) * r2)) / (mz * mz + r2)
    rays = scale[:, None] * np.column_stack((m, mz)) - [0, 0, xi]
    expected = POINTS / np.linalg.norm(POINTS, axis=-1, keepdims=True)
    np.testing.assert_allclose(rays, expected, atol=1e-12)
    assert valid.all()


def test_double_sphere_masks_model_domain_even_with_positive_depth_and_denominator():
    # xi=-1/2, alpha=1 yields z/d1 > 1/sqrt(5), stricter than z > 0.
    points = np.array([[1., 0., .1], [0., 0., 1.], [1., 0., 1.]])
    pixels, valid = project_points(points, K, "double_sphere", [-.5, 1.])
    np.testing.assert_array_equal(valid, [False, True, True])
    assert np.isnan(pixels[0]).all()
    np.testing.assert_allclose(pixels[1], [320, 240])


@pytest.mark.parametrize("model,distortion", MODELS)
def test_invalid_points_are_nan_without_numeric_warnings(model, distortion):
    points = np.array([[0, 0, 0], [1, 0, -1], [np.nan, 0, 1],
                       [0, np.inf, 1], [0, 0, np.inf], [0, 0, 1.]])
    with np.errstate(all="raise"):
        pixels, valid = project_points(points, K, model, distortion)
    np.testing.assert_array_equal(valid, [False] * 5 + [True])
    assert np.isnan(pixels[:5]).all()
    np.testing.assert_allclose(pixels[5], [320, 240])


@pytest.mark.parametrize("model,distortion", MODELS)
def test_preserves_leading_dimensions_and_empty_inputs(model, distortion):
    pixels, valid = project_points(POINTS.reshape(2, 2, 3), K, model, distortion)
    assert pixels.shape == (2, 2, 2)
    assert valid.shape == (2, 2)
    empty, mask = project_points(np.empty((0, 3)), K, model, distortion)
    assert empty.shape == (0, 2) and mask.shape == (0,)


def test_pinhole_skew_and_no_image_bounds_or_arbitrary_depth_cutoff():
    intrinsic = K.copy()
    intrinsic[0, 1] = 7
    points = np.array([[4., 2., 1.], [1e-100, 2e-100, 1e-100]])
    homogeneous = points @ intrinsic.T
    pixels, valid = project_points(points, intrinsic)
    np.testing.assert_allclose(pixels, homogeneous[:, :2] / homogeneous[:, 2:])
    assert valid.all()


@pytest.mark.parametrize("model,distortion", [
    ("unknown", None), ("brown_conrady", None), ("brown_conrady", [0, 0]),
    ("fisheye", None), ("fisheye", [0] * 5), ("double_sphere", None),
    ("double_sphere", [0, -.1]), ("double_sphere", [0, 1.1]),
    ("double_sphere", [-1, .5]), ("fisheye", [0, 0, np.nan, 0]),
    ("pinhole", [.1]),
])
def test_rejects_missing_unsupported_or_nonfinite_calibration(model, distortion):
    with pytest.raises(ValueError):
        project_points(POINTS, K, model, distortion)


def test_rejects_malformed_intrinsics_and_points():
    for intrinsic in [np.eye(2), np.zeros((3, 3)), K * np.nan,
                      np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]])]:
        with pytest.raises(ValueError):
            project_points(POINTS, intrinsic)
    with pytest.raises(ValueError):
        project_points(np.zeros((3, 2)), K)


def test_transform_applies_rotation_then_translation_and_preserves_shape():
    transform = np.array([[0, -1, 0, 4], [1, 0, 0, 5], [0, 0, 1, 6], [0, 0, 0, 1.]])
    points = np.array([[[1, 2, 3], [4, 5, 6]]])
    np.testing.assert_allclose(transform_points(transform, points), [[[2, 6, 9], [-1, 9, 12]]])
    np.testing.assert_allclose(transform_points(transform, [1, 2, 3]), [2, 6, 9])


def test_transform_rejects_malformed_matrix_or_points():
    for transform in [np.eye(3), np.zeros((4, 4)), np.eye(4) * np.nan]:
        with pytest.raises(ValueError):
            transform_points(transform, POINTS)
    with pytest.raises(ValueError):
        transform_points(np.eye(4), [1, 2])


def test_transform_accepts_float32_homogeneous_roundoff_but_not_projective_rows():
    transform = np.eye(4)
    transform[3, 3] = 0.9999997615814209  # actual EgoDex terminal-row export
    np.testing.assert_allclose(transform_points(transform, POINTS), POINTS)
    for row in ([0, 0, 0, .999], [1e-4, 0, 0, 1]):
        transform[3] = row
        with pytest.raises(ValueError):
            transform_points(transform, POINTS)
