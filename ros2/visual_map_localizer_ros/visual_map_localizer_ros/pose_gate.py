"""Velocity-based outlier gate for VPS poses.

The VPS localizer occasionally returns a confidently-localized but spatially
implausible pose — typically when matching latches onto a repeated facade
or a different floor of the same building. Per-frame quality metrics
(inliers, reproj_error) cannot always tell these apart from good poses,
so we add a second filter: any new pose whose implied linear or angular
velocity (relative to the last accepted pose) exceeds physical limits
is rejected.

This module is intentionally ROS-agnostic so it can be unit-tested
without rclpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class _State:
    R_wc: np.ndarray   # 3x3 world-from-camera rotation
    C_w: np.ndarray    # camera centre in world (3,)
    stamp_sec: float


class PoseGate:
    """Reject pose updates that imply unrealistic motion.

    Parameters
    ----------
    max_linear_velocity_mps:
        Implied linear velocity (m/s) over which a new pose is rejected.
        Set to a value comfortably above the platform's true top speed.
    max_angular_velocity_dps:
        Implied angular velocity (deg/s) over which a new pose is rejected.
    state_timeout_sec:
        If more than this many seconds have passed since the last accepted
        pose, the gate forgets the previous state and treats the new pose
        as a fresh "first" sample. Use this to prevent a stale state from
        permanently blocking re-localization after a long gap.
    """

    def __init__(
        self,
        max_linear_velocity_mps: float,
        max_angular_velocity_dps: float,
        state_timeout_sec: float = 30.0,
    ) -> None:
        if max_linear_velocity_mps <= 0:
            raise ValueError("max_linear_velocity_mps must be > 0")
        if max_angular_velocity_dps <= 0:
            raise ValueError("max_angular_velocity_dps must be > 0")
        self.max_lv = float(max_linear_velocity_mps)
        self.max_av = float(max_angular_velocity_dps)
        self.state_timeout_sec = float(state_timeout_sec)
        self._last: Optional[_State] = None

    def reset(self) -> None:
        self._last = None

    def accept(
        self,
        R_wc: np.ndarray,
        C_w: np.ndarray,
        stamp_sec: float,
    ) -> Tuple[bool, str]:
        """Decide whether to accept a candidate pose.

        Returns (accepted, reason). When accepted=True, internal state is
        updated to the new pose; when accepted=False, internal state is
        unchanged so a subsequent plausible pose can still be accepted
        relative to the last good one.
        """
        R_wc = np.asarray(R_wc, dtype=float)
        C_w = np.asarray(C_w, dtype=float).reshape(3)
        if R_wc.shape != (3, 3):
            raise ValueError(f"R_wc must be 3x3, got {R_wc.shape}")

        if self._last is None:
            self._last = _State(R_wc=R_wc, C_w=C_w, stamp_sec=float(stamp_sec))
            return True, "first"

        dt = float(stamp_sec) - self._last.stamp_sec
        if dt > self.state_timeout_sec:
            # A long gap (e.g. node was paused) — drop history, accept fresh.
            self._last = _State(R_wc=R_wc, C_w=C_w, stamp_sec=float(stamp_sec))
            return True, "state-timeout-reset"
        if dt <= 0.0:
            # Out-of-order or duplicate stamp; accept and re-anchor rather
            # than silently masking sender bugs.
            self._last = _State(R_wc=R_wc, C_w=C_w, stamp_sec=float(stamp_sec))
            return True, "non-monotonic-stamp"

        d_trans = float(np.linalg.norm(C_w - self._last.C_w))
        v = d_trans / dt
        if v > self.max_lv:
            return False, f"linear-velocity={v:.2f}>{self.max_lv}"

        # Angle between rotations: arccos((trace(R_a R_b^T) - 1) / 2)
        cos = (np.trace(R_wc @ self._last.R_wc.T) - 1.0) / 2.0
        cos = float(np.clip(cos, -1.0, 1.0))
        ang_rad = math.acos(cos)
        omega_dps = math.degrees(ang_rad) / dt
        if omega_dps > self.max_av:
            return False, f"angular-velocity={omega_dps:.1f}>{self.max_av}"

        self._last = _State(R_wc=R_wc, C_w=C_w, stamp_sec=float(stamp_sec))
        return True, "ok"
