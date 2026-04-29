"""Unit tests for the pure helpers inside vps_node.

These do not spin a ROS context, so they are safe to run as plain pytest.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the package importable when running tests directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("rclpy")
pytest.importorskip("sensor_msgs")
pytest.importorskip("pycolmap")

from visual_map_localizer_ros.vps_node import (  # noqa: E402
    _quat_from_rotmat,
    _pycolmap_camera_from_camera_info,
)


def test_quat_from_rotmat_identity():
    q = _quat_from_rotmat(np.eye(3))
    # Either (0, 0, 0, 1) or (0, 0, 0, -1) — canonical form has w >= 0.
    if q[3] < 0:
        q = -q
    np.testing.assert_allclose(q, [0, 0, 0, 1], atol=1e-12)


def test_quat_from_rotmat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(10):
        # Random unit quaternion (xyzw) -> rotation matrix -> quaternion
        v = rng.normal(size=4)
        v /= np.linalg.norm(v)
        x, y, z, w = v
        # Build R from xyzw (ROS convention)
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        q = _quat_from_rotmat(R)
        # Quaternions are equivalent up to sign — normalize both to w >= 0.
        if q[3] < 0:
            q = -q
        if v[3] < 0:
            v = -v
        np.testing.assert_allclose(q, v, atol=1e-9)


def test_pycolmap_camera_from_camera_info_pinhole():
    from sensor_msgs.msg import CameraInfo

    msg = CameraInfo()
    msg.width = 640
    msg.height = 480
    msg.k = [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]
    msg.distortion_model = ""
    msg.d = []
    cam = _pycolmap_camera_from_camera_info(msg)
    assert cam.width == 640
    assert cam.height == 480
    assert cam.model_name == "PINHOLE"
    np.testing.assert_allclose(list(cam.params), [600, 600, 320, 240])


def test_pycolmap_camera_from_camera_info_plumb_bob():
    from sensor_msgs.msg import CameraInfo

    msg = CameraInfo()
    msg.width = 1280
    msg.height = 720
    msg.k = [800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0]
    msg.distortion_model = "plumb_bob"
    msg.d = [-0.1, 0.05, 0.001, 0.002, 0.0]
    cam = _pycolmap_camera_from_camera_info(msg)
    assert cam.width == 1280
    assert cam.height == 720
    # Last term k3 == 0 → OPENCV (8 params)
    assert cam.model_name == "OPENCV"
    np.testing.assert_allclose(
        list(cam.params),
        [800, 800, 640, 360, -0.1, 0.05, 0.001, 0.002],
    )


def test_pycolmap_camera_from_camera_info_plumb_bob_with_k3():
    from sensor_msgs.msg import CameraInfo

    msg = CameraInfo()
    msg.width = 1280
    msg.height = 720
    msg.k = [800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0]
    msg.distortion_model = "plumb_bob"
    msg.d = [-0.1, 0.05, 0.001, 0.002, 0.01]
    cam = _pycolmap_camera_from_camera_info(msg)
    assert cam.model_name == "FULL_OPENCV"
    assert len(cam.params) == 12
