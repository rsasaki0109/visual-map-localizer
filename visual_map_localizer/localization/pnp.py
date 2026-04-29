"""PnP + RANSAC pose estimation.

Two implementations are provided:

* `pnp_pycolmap`  — uses pycolmap.absolute_pose_estimation (preferred: handles
  every COLMAP camera model natively, including radial distortion).
* `pnp_opencv`    — solvePnPRansac fallback for environments where pycolmap is
  unavailable or fails. Distortion is honoured for OPENCV / FULL_OPENCV /
  SIMPLE_RADIAL camera models.

Both return a `PnPResult` with the estimated rigid transform (camera-from-world
convention, matching COLMAP).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- types
@dataclass
class PnPResult:
    success: bool
    R: Optional[np.ndarray] = None      # 3x3 rotation, world -> camera
    t: Optional[np.ndarray] = None      # length-3 translation, world -> camera
    qvec: Optional[np.ndarray] = None   # quaternion (w, x, y, z), optional
    inliers: Optional[np.ndarray] = None  # boolean mask over input points
    num_inliers: int = 0
    reproj_error: float = float("nan")
    method: str = ""
    error: Optional[str] = None


# --------------------------------------------------------------- public entry
def estimate_pose(
    points2D: np.ndarray,
    points3D: np.ndarray,
    camera,
    *,
    max_error_px: float = 12.0,
    min_inliers: int = 12,
    prefer: str = "pycolmap",
) -> PnPResult:
    """Estimate a 6DoF pose from 2D-3D correspondences.

    `camera` must be a `pycolmap.Camera` (we use its model + parameters for
    both back-ends).
    """
    points2D = np.asarray(points2D, dtype=np.float64).reshape(-1, 2)
    points3D = np.asarray(points3D, dtype=np.float64).reshape(-1, 3)
    if len(points2D) != len(points3D):
        raise ValueError(
            f"points2D ({len(points2D)}) and points3D ({len(points3D)}) "
            "must have the same length"
        )
    if len(points2D) < 4:
        return PnPResult(success=False, error=f"only {len(points2D)} correspondences (need 4)")

    if prefer == "pycolmap":
        try:
            res = pnp_pycolmap(points2D, points3D, camera, max_error_px=max_error_px)
            if res.success and res.num_inliers >= min_inliers:
                return res
            logger.debug("pycolmap PnP rejected (inliers=%d, err=%s)", res.num_inliers, res.error)
        except Exception as exc:  # pragma: no cover - depends on pycolmap build
            logger.warning("pycolmap PnP raised %s — falling back to OpenCV", exc)

    res = pnp_opencv(points2D, points3D, camera, max_error_px=max_error_px)
    if res.success and res.num_inliers < min_inliers:
        res.success = False
        res.error = f"only {res.num_inliers} inliers (< {min_inliers})"
    return res


# ---------------------------------------------------------------- pycolmap
def pnp_pycolmap(
    points2D: np.ndarray,
    points3D: np.ndarray,
    camera,
    *,
    max_error_px: float = 12.0,
) -> PnPResult:
    """Estimate pose using pycolmap.absolute_pose_estimation.

    The pycolmap API has changed several times; we try the modern signature
    first and fall back to the legacy one.
    """
    import pycolmap

    ret = None
    err: Optional[str] = None

    # Modern API: estimation_options=AbsolutePoseEstimationOptions
    try:
        opts = pycolmap.AbsolutePoseEstimationOptions()
        if hasattr(opts, "ransac"):
            opts.ransac.max_error = float(max_error_px)
        if hasattr(opts, "estimate_focal_length"):
            opts.estimate_focal_length = False
        ret = pycolmap.absolute_pose_estimation(
            points2D, points3D, camera, estimation_options=opts
        )
    except TypeError as exc:
        err = str(exc)

    # Legacy API: positional max_error_px
    if ret is None:
        try:
            ret = pycolmap.absolute_pose_estimation(
                points2D, points3D, camera, float(max_error_px)
            )
        except Exception as exc:
            return PnPResult(success=False, error=f"pycolmap failed: {exc}")

    if not ret or (isinstance(ret, dict) and not ret.get("success", False)):
        return PnPResult(success=False, error=err or "pycolmap returned no pose")

    # Newer pycolmap returns a Rigid3d under 'cam_from_world'; older returns
    # 'qvec' / 'tvec'. Handle both.
    R, t, qvec = _extract_pose(ret)
    inliers = np.asarray(ret.get("inliers", []), dtype=bool)
    num_inliers = int(ret.get("num_inliers", inliers.sum()))

    reproj_err = _reprojection_error(points2D[inliers], points3D[inliers], R, t, camera)
    return PnPResult(
        success=True,
        R=R, t=t, qvec=qvec,
        inliers=inliers,
        num_inliers=num_inliers,
        reproj_error=reproj_err,
        method="pycolmap",
    )


def _extract_pose(ret) -> tuple:
    """Pull (R, t, qvec) out of a pycolmap absolute_pose_estimation return value."""
    if "cam_from_world" in ret:
        rig = ret["cam_from_world"]
        # pycolmap.Rigid3d exposes `rotation` (Rotation3d) and `translation`
        R = np.asarray(rig.rotation.matrix(), dtype=np.float64)
        t = np.asarray(rig.translation, dtype=np.float64).reshape(3)
        qvec = np.asarray(rig.rotation.quat, dtype=np.float64) if hasattr(rig.rotation, "quat") else None
        return R, t, qvec
    # Legacy keys
    qvec = np.asarray(ret["qvec"], dtype=np.float64)
    t = np.asarray(ret["tvec"], dtype=np.float64).reshape(3)
    R = _qvec_to_rotmat(qvec)
    return R, t, qvec


# ------------------------------------------------------------------ opencv
def pnp_opencv(
    points2D: np.ndarray,
    points3D: np.ndarray,
    camera,
    *,
    max_error_px: float = 8.0,
) -> PnPResult:
    K, dist = camera_matrix_and_distortion(camera)
    pts3 = points3D.reshape(-1, 1, 3).astype(np.float64)
    pts2 = points2D.reshape(-1, 1, 2).astype(np.float64)

    ok, rvec, tvec, inlier_idx = cv2.solvePnPRansac(
        pts3, pts2, K, dist,
        iterationsCount=2000,
        reprojectionError=float(max_error_px),
        confidence=0.9999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inlier_idx is None or len(inlier_idx) == 0:
        return PnPResult(success=False, error="OpenCV solvePnPRansac failed", method="opencv")

    inlier_idx = inlier_idx.flatten()
    # Refine pose on inliers (improves the fit a lot in practice)
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            pts3[inlier_idx], pts2[inlier_idx], K, dist, rvec, tvec
        )
    except cv2.error:
        pass

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    inliers_mask = np.zeros(len(points2D), dtype=bool)
    inliers_mask[inlier_idx] = True
    reproj_err = _reprojection_error(
        points2D[inliers_mask], points3D[inliers_mask], R, t, camera
    )
    return PnPResult(
        success=True,
        R=R, t=t, qvec=_rotmat_to_qvec(R),
        inliers=inliers_mask,
        num_inliers=int(inliers_mask.sum()),
        reproj_error=reproj_err,
        method="opencv",
    )


# --------------------------------------------------------------- intrinsics
def camera_matrix_and_distortion(camera) -> tuple:
    """Convert a `pycolmap.Camera` into (K, distortion) usable by OpenCV.

    Distortion is returned as the 5-vector ``[k1, k2, p1, p2, k3]`` expected by
    OpenCV (zeros when the COLMAP model has no distortion terms).
    """
    name = getattr(camera, "model_name", None) or str(getattr(camera, "model", ""))
    name = name.upper()
    p = list(camera.params)

    if name in ("SIMPLE_PINHOLE",):
        f, cx, cy = p[0], p[1], p[2]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
    elif name in ("PINHOLE",):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
    elif name in ("SIMPLE_RADIAL",):
        f, cx, cy, k1 = p[0], p[1], p[2], p[3]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, 0, 0, 0, 0], dtype=np.float64)
    elif name in ("RADIAL",):
        f, cx, cy, k1, k2 = p[0], p[1], p[2], p[3], p[4]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
    elif name in ("OPENCV",):
        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2, 0], dtype=np.float64)
    elif name in ("FULL_OPENCV",):
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = p[:9]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    else:
        # Conservative fallback: assume the first two params are fx/fy.
        fx = p[0]
        fy = p[1] if len(p) > 1 else fx
        cx = camera.width / 2.0
        cy = camera.height / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
    return K, dist


# --------------------------------------------------------------- reprojection
def _reprojection_error(
    pts2d: np.ndarray,
    pts3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    camera,
) -> float:
    if len(pts2d) == 0:
        return float("nan")
    K, dist = camera_matrix_and_distortion(camera)
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    proj, _ = cv2.projectPoints(
        pts3d.reshape(-1, 1, 3).astype(np.float64),
        rvec, t.astype(np.float64).reshape(3, 1), K, dist,
    )
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - pts2d, axis=1)
    return float(np.mean(err))


# --------------------------------------------------------------- quaternions
def _qvec_to_rotmat(qvec: Sequence[float]) -> np.ndarray:
    """COLMAP convention: q = (w, x, y, z)."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Inverse of `_qvec_to_rotmat` (uses the standard Shepperd method)."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)
