"""Unit tests for `PoseGate` (no rclpy needed)."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

# Make the ament-style package importable when pytest is run from the
# package root or repo root without `colcon build`.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from visual_map_localizer_ros.pose_gate import PoseGate  # noqa: E402


def Rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_first_pose_always_accepted():
    g = PoseGate(max_linear_velocity_mps=10.0, max_angular_velocity_dps=60.0)
    ok, why = g.accept(np.eye(3), np.zeros(3), stamp_sec=0.0)
    assert ok and why == "first"


def test_small_motion_accepted():
    g = PoseGate(10.0, 60.0)
    g.accept(np.eye(3), np.zeros(3), 0.0)
    ok, why = g.accept(Rz(1.0), np.array([0.5, 0.0, 0.0]), stamp_sec=1.0)
    # 0.5 m / 1 s = 0.5 m/s, 1 deg / s — well within limits
    assert ok and why == "ok"


def test_large_translation_rejected():
    g = PoseGate(max_linear_velocity_mps=10.0, max_angular_velocity_dps=60.0)
    g.accept(np.eye(3), np.zeros(3), 0.0)
    # 100 m in 1 s = 100 m/s, way above 10 m/s
    ok, why = g.accept(np.eye(3), np.array([100.0, 0.0, 0.0]), 1.0)
    assert not ok and why.startswith("linear-velocity=")


def test_large_rotation_rejected():
    g = PoseGate(max_linear_velocity_mps=10.0, max_angular_velocity_dps=60.0)
    g.accept(np.eye(3), np.zeros(3), 0.0)
    # 180 deg in 1 s = 180 deg/s, above 60
    ok, why = g.accept(Rz(180.0), np.zeros(3), 1.0)
    assert not ok and why.startswith("angular-velocity=")


def test_reject_does_not_corrupt_state():
    """A rejected pose must not advance the gate's internal anchor."""
    g = PoseGate(max_linear_velocity_mps=10.0, max_angular_velocity_dps=60.0)
    g.accept(np.eye(3), np.zeros(3), 0.0)
    # Reject one
    g.accept(np.eye(3), np.array([100.0, 0.0, 0.0]), 1.0)
    # The next plausible pose should still be accepted relative to (0, 0, 0)
    ok, why = g.accept(np.eye(3), np.array([1.0, 0.0, 0.0]), 2.0)
    assert ok and why == "ok"


def test_state_timeout_resets():
    g = PoseGate(10.0, 60.0, state_timeout_sec=5.0)
    g.accept(np.eye(3), np.zeros(3), 0.0)
    # 100 s gap — would be > 5 m/s with old anchor; gate should reset and accept
    ok, why = g.accept(np.eye(3), np.array([100.0, 0.0, 0.0]), 100.0)
    assert ok and why == "state-timeout-reset"


def test_non_monotonic_stamp_re_anchors():
    g = PoseGate(10.0, 60.0)
    g.accept(np.eye(3), np.zeros(3), 5.0)
    ok, why = g.accept(np.eye(3), np.array([100.0, 0.0, 0.0]), 4.0)
    # Sender ordering looks broken; we accept and re-anchor rather than mask it
    assert ok and why == "non-monotonic-stamp"


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        PoseGate(max_linear_velocity_mps=0.0, max_angular_velocity_dps=60.0)
    with pytest.raises(ValueError):
        PoseGate(max_linear_velocity_mps=10.0, max_angular_velocity_dps=-1.0)
