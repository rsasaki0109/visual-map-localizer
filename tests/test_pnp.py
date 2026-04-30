"""Unit tests for the OpenCV-based PnP path.

These do not depend on hloc / pycolmap availability — they construct synthetic
2D-3D correspondences from a known pose and verify the recovered pose.
"""
from __future__ import annotations
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pycolmap = pytest.importorskip("pycolmap")

from visual_map_localizer.localization.pnp import (  # noqa: E402
    pnp_opencv,
    estimate_pose,
    _qvec_to_rotmat,
    _rotmat_to_qvec,
)


def _make_pinhole_camera(width=640, height=480, fx=525.0, fy=525.0):
    cx, cy = width / 2.0, height / 2.0
    return pycolmap.Camera(
        model="PINHOLE",
        width=width, height=height,
        params=[fx, fy, cx, cy],
    )


def _make_synthetic_correspondences(camera, R, t, n=100, seed=0):
    rng = np.random.default_rng(seed)
    # Random 3D points in front of the camera.
    pts3d = rng.uniform(-2, 2, size=(n, 3))
    pts3d[:, 2] += 6.0  # push away from origin so they project safely

    # Bring them into the world frame using the inverse of the camera pose.
    Rcw = R
    tcw = t.reshape(3, 1)
    pts_world = (Rcw.T @ (pts3d.T - tcw)).T

    # Project from the world frame.
    fx, fy, cx, cy = camera.params
    pts_cam = (Rcw @ pts_world.T + tcw).T
    proj = (pts_cam[:, :2] / pts_cam[:, 2:3])
    pts2d = proj * np.array([fx, fy]) + np.array([cx, cy])
    return pts2d, pts_world


def test_pnp_opencv_recovers_pose_within_tolerance():
    camera = _make_pinhole_camera()
    # Ground-truth pose: small rotation + translation
    angle = np.deg2rad(5.0)
    R_gt = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])
    t_gt = np.array([0.2, -0.1, 0.05])

    pts2d, pts3d = _make_synthetic_correspondences(camera, R_gt, t_gt, n=120)
    res = pnp_opencv(pts2d, pts3d, camera, max_error_px=2.0)
    assert res.success
    assert res.num_inliers >= 50

    rot_err = np.rad2deg(np.arccos(np.clip((np.trace(R_gt.T @ res.R) - 1) / 2, -1, 1)))
    t_err = float(np.linalg.norm(res.t - t_gt))
    assert rot_err < 0.5, f"rotation error too large: {rot_err}"
    assert t_err < 0.05, f"translation error too large: {t_err}"
    assert res.reproj_error < 1.5


def test_estimate_pose_falls_back_to_opencv_when_pycolmap_disabled():
    camera = _make_pinhole_camera()
    R_gt = np.eye(3)
    t_gt = np.array([0.0, 0.0, 0.0])
    pts2d, pts3d = _make_synthetic_correspondences(camera, R_gt, t_gt, n=80)
    # Force the OpenCV path:
    res = estimate_pose(pts2d, pts3d, camera, prefer="opencv", min_inliers=10)
    assert res.success
    assert res.method == "opencv"


def test_quaternion_roundtrip():
    rng = np.random.default_rng(123)
    for _ in range(10):
        v = rng.normal(size=4)
        v /= np.linalg.norm(v)
        # Ensure a unique (canonical) quaternion sign so the equality check is well-defined.
        if v[0] < 0:
            v = -v
        R = _qvec_to_rotmat(v)
        q2 = _rotmat_to_qvec(R)
        if q2[0] < 0:
            q2 = -q2
        np.testing.assert_allclose(v, q2, atol=1e-6)
