"""End-to-end single-image localization pipeline.

Glues together:
    1. global descriptor extraction for the query
    2. retrieval of Top-K candidate db images
    3. local feature extraction for the query
    4. SuperPoint / LightGlue matching against retrieved images
    5. building 2D-3D correspondences from db keypoints + COLMAP point3Ds
    6. PnP + RANSAC pose estimation (pycolmap, OpenCV fallback)
"""
from __future__ import annotations
import logging
import shutil
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import (
    DB_IMAGE_LIST_FILE,
    FEATURES_FILE,
    GLOBAL_DESC_FILE,
    META_FILE,
    LocalizeConfig,
    SFM_DIRNAME,
)
from ..io.camera import infer_camera, build_camera
from ..io.colmap_map import ColmapMap
from ..io.output import LocalizationResult

logger = logging.getLogger(__name__)


class VisualMapLocalizer:
    """Object-oriented entry point for localization.

    Loads a previously built map (`build-map` output) and offers `.localize()`
    on a single query image. The class is intentionally cheap to construct so
    long-running services can keep one alive.
    """

    def __init__(self, map_dir: str | Path, config: Optional[LocalizeConfig] = None):
        self.map_dir = Path(map_dir).resolve()
        self.config = config or LocalizeConfig()
        self._validate_map_dir()
        # Auto-align local_feature / global_descriptor / matcher with the values
        # used at build-map time, but let an explicit user-provided config win.
        self._apply_meta_defaults()
        self.map = ColmapMap.load(self.map_dir / SFM_DIRNAME)
        self.db_image_list = self._load_db_image_list()
        logger.info(
            "Loaded map: %d images, %d points3D (features=%s, gdesc=%s, matcher=%s)",
            self.map.num_images, self.map.num_points,
            self.config.local_feature, self.config.global_descriptor, self.config.matcher,
        )

    # ------------------------------------------------------------------ paths
    def _validate_map_dir(self) -> None:
        for name in (SFM_DIRNAME, FEATURES_FILE, GLOBAL_DESC_FILE, DB_IMAGE_LIST_FILE):
            p = self.map_dir / name
            if not p.exists():
                raise FileNotFoundError(f"missing in map dir: {p}")

    @property
    def features_path(self) -> Path:
        return self.map_dir / FEATURES_FILE

    @property
    def global_desc_path(self) -> Path:
        return self.map_dir / GLOBAL_DESC_FILE

    def _load_db_image_list(self) -> List[str]:
        text = (self.map_dir / DB_IMAGE_LIST_FILE).read_text(encoding="utf-8")
        return [ln.strip() for ln in text.splitlines() if ln.strip()]

    def _apply_meta_defaults(self) -> None:
        """If the map has a manifest, copy build-time configs that the user did
        not override on the localize side. We compare against a fresh
        `LocalizeConfig()` to detect "user supplied a non-default" — if the
        caller already changed a field, we leave it alone."""
        meta_path = self.map_dir / META_FILE
        if not meta_path.exists():
            return
        try:
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - corrupted manifest
            return
        defaults = LocalizeConfig()
        for field_name in ("local_feature", "global_descriptor", "matcher"):
            current = getattr(self.config, field_name)
            if current == getattr(defaults, field_name) and field_name in meta:
                setattr(self.config, field_name, meta[field_name])

    # --------------------------------------------------------------- localize
    def localize(
        self,
        query_image,
        *,
        work_dir: Optional[Path] = None,
        keep_work_dir: bool = False,
        camera=None,
        name: Optional[str] = None,
    ) -> LocalizationResult:
        """Localize a single image.

        Parameters
        ----------
        query_image : str | Path | np.ndarray
            Either a path on disk, or an in-memory image (HxWx3 uint8, RGB or
            BGR — both work because hloc converts internally).
        camera : pycolmap.Camera | None
            Override for the query camera. Required when `query_image` is a
            numpy array; optional for path inputs (EXIF-inferred otherwise).
        name : str | None
            Logical filename to use inside the hloc cache. Defaults to the
            input path's basename or, for arrays, an auto-generated name.
        """
        from hloc import (  # noqa: WPS433 — heavy import, kept lazy
            extract_features,
            match_features,
            pairs_from_retrieval,
        )
        from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

        t0 = time.perf_counter()
        timing: dict = {}

        is_array = _is_image_array(query_image)
        if is_array:
            if camera is None:
                return LocalizationResult.failure(
                    "camera must be provided when query_image is a numpy array",
                    query=name,
                )
            stem = name or f"live_{int(t0 * 1000)}.png"
            query_path_for_log = stem
        else:
            query_image = Path(query_image).resolve()
            if not query_image.exists():
                return LocalizationResult.failure(
                    f"query image not found: {query_image}", query=str(query_image)
                )
            query_path_for_log = str(query_image)

        # All hloc helpers operate on (image_dir, image_name) so we stage the
        # query into a temp folder that contains a single file. For ndarray
        # inputs we serialize once to PNG inside the same staging folder.
        work_dir = Path(work_dir or self.map_dir / "_query_tmp").resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        query_dir = work_dir / "query_images"
        query_dir.mkdir(parents=True, exist_ok=True)
        if is_array:
            query_name = f"query/{stem}"
            staged = query_dir / "query" / stem
            staged.parent.mkdir(parents=True, exist_ok=True)
            _write_image_array(query_image, staged)
            input_image_path = staged
        else:
            query_name = f"query/{query_image.name}"
            staged = query_dir / "query" / query_image.name
            staged.parent.mkdir(parents=True, exist_ok=True)
            if not staged.exists():
                staged.write_bytes(query_image.read_bytes())
            input_image_path = query_image

        try:
            # When the query came from an in-memory ndarray, the H5 may
            # already contain stale features for the same `query_name` from
            # a previous run (vps_node restart, replay, etc.) — we must
            # always re-extract to avoid silently reusing the wrong image.
            overwrite_q = is_array
            # ---------------------- 1. global descriptor for the query --------
            t = time.perf_counter()
            extract_features.main(
                extract_features.confs[self.config.global_descriptor],
                query_dir,
                self.map_dir,
                image_list=[query_name],
                feature_path=self.global_desc_path,
                overwrite=overwrite_q,
            )
            timing["global_descriptor"] = time.perf_counter() - t

            # ---------------------- 2. retrieval pairs ------------------------
            t = time.perf_counter()
            loc_pairs = work_dir / "pairs-loc.txt"
            pairs_from_retrieval.main(
                self.global_desc_path,
                loc_pairs,
                num_matched=self.config.top_k,
                query_list=[query_name],
                db_list=self.db_image_list,
            )
            timing["retrieval"] = time.perf_counter() - t
            retrieved = _read_retrieval_neighbors(loc_pairs, query_name)
            if not retrieved:
                return LocalizationResult.failure(
                    "retrieval returned no candidate db images",
                    query=query_path_for_log, timing=timing,
                )

            # ---------------------- 3. local features for the query ----------
            t = time.perf_counter()
            extract_features.main(
                extract_features.confs[self.config.local_feature],
                query_dir,
                self.map_dir,
                image_list=[query_name],
                feature_path=self.features_path,
                overwrite=overwrite_q,
            )
            timing["local_features"] = time.perf_counter() - t

            # ---------------------- 4. matching ------------------------------
            t = time.perf_counter()
            from ..matching.matcher import resolve_matcher_conf
            # hloc requires that `features` and `matches` are both passed as
            # Paths *or* both as conf-name strings — never mixed.
            matches_path = work_dir / "matches-loc.h5"
            match_features.main(
                resolve_matcher_conf(self.config.matcher),
                loc_pairs,
                features=self.features_path,
                matches=matches_path,
                overwrite=False,
            )
            timing["matching"] = time.perf_counter() - t

            # ---------------------- 5. localize via QueryLocalizer ------------
            t = time.perf_counter()
            if camera is None:
                camera = self._build_query_camera(input_image_path)
            db_ids = self.map.db_ids_for_names(retrieved)

            ransac_conf = {"estimation": {"ransac": {"max_error": self.config.ransac_max_error_px}}}
            localizer = QueryLocalizer(self.map.reconstruction, ransac_conf)
            ret, log = pose_from_cluster(
                localizer, query_name, camera, db_ids,
                self.features_path, matches_path,
            )
            timing["pnp"] = time.perf_counter() - t

            # Modern hloc returns `cam_from_world` on success and omits it on
            # failure (older versions used a `success` boolean).
            success = (
                ret.get("success", False)
                or "cam_from_world" in ret
                or "qvec" in ret
            )
            num_inliers = int(ret.get("num_inliers", 0))
            if not success or num_inliers < self.config.ransac_min_inliers:
                return LocalizationResult.failure(
                    f"PnP rejected ({num_inliers} inliers < {self.config.ransac_min_inliers})"
                    if success else "PnP failed",
                    retrieval=retrieved,
                    num_matches=int(log.get("num_matches", 0)) if isinstance(log, dict) else 0,
                    query=query_path_for_log, timing=timing,
                )

            R, t_vec, qvec = _extract_pose_from_hloc(ret)
            inliers = num_inliers
            num_matches = int(log.get("num_matches", 0)) if isinstance(log, dict) else 0
            reproj_err = _reprojection_error_for_inliers(ret, log, camera, self.map)

            timing["total"] = time.perf_counter() - t0
            return LocalizationResult.from_pose(
                R, t_vec,
                inliers=inliers,
                reproj_error=reproj_err,
                num_matches=num_matches,
                retrieval=retrieved,
                query=query_path_for_log,
                timing=timing,
                qvec=qvec,
            )
        finally:
            if not keep_work_dir and work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)

    # ----------------------------------------------------- query intrinsics
    def _build_query_camera(self, query_image: Path):
        cfg = self.config
        if cfg.camera_model and cfg.camera_params and cfg.image_size:
            w, h = cfg.image_size
            return build_camera(cfg.camera_model, w, h, cfg.camera_params)
        return infer_camera(query_image)


# ---------------------------------------------------------------- helpers
def _is_image_array(obj) -> bool:
    return isinstance(obj, np.ndarray) and obj.ndim in (2, 3)


def _write_image_array(arr: np.ndarray, dst: Path) -> None:
    """Persist a HxW or HxWx3 uint8 image to PNG.

    Expects RGB layout for 3-channel inputs (which is what ROS, PIL and
    matplotlib all use). cv2.imwrite needs BGR so we convert before writing.
    Floating-point inputs are clipped to ``[0, 255]`` and downcast to uint8.
    """
    import cv2

    if arr.ndim == 3 and arr.shape[2] not in (1, 3):
        raise ValueError(f"unexpected image array shape: {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        bgr = arr
    ok = cv2.imwrite(str(dst), bgr)
    if not ok or not dst.exists():
        raise IOError(
            f"cv2.imwrite failed for {dst} "
            f"(shape={arr.shape}, dtype={arr.dtype}, dir_exists={dst.parent.exists()})"
        )


def _read_retrieval_neighbors(pairs_path: Path, query_name: str) -> List[str]:
    out: List[str] = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            a, b = parts
            if a == query_name:
                out.append(b)
            elif b == query_name:
                out.append(a)
    return out


def _extract_pose_from_hloc(ret: dict):
    """Convert an hloc QueryLocalizer result into (R, t, qvec).

    The output `qvec` always follows the COLMAP convention ``[w, x, y, z]``;
    pycolmap's `Rotation3d.quat` returns ``[x, y, z, w]`` (xyzw) so we re-order
    when needed.
    """
    if "cam_from_world" in ret:
        rig = ret["cam_from_world"]
        R = np.asarray(rig.rotation.matrix(), dtype=np.float64)
        t = np.asarray(rig.translation, dtype=np.float64).reshape(3)
        if hasattr(rig.rotation, "quat"):
            xyzw = np.asarray(rig.rotation.quat, dtype=np.float64)
            qvec = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
        else:
            qvec = None
        return R, t, qvec
    # Legacy hloc: top-level qvec/tvec already follow COLMAP wxyz convention.
    qvec = np.asarray(ret["qvec"], dtype=np.float64)
    t = np.asarray(ret["tvec"], dtype=np.float64).reshape(3)
    from .pnp import _qvec_to_rotmat
    R = _qvec_to_rotmat(qvec)
    return R, t, qvec


def _count_total_matches(log) -> int:
    if not isinstance(log, dict):
        return 0
    db_log = log.get("db", {})
    total = 0
    for entry in db_log.values():
        m = entry.get("matches")
        if m is not None:
            total += int(np.asarray(m).shape[0])
    return total


def _reprojection_error_for_inliers(ret: dict, log, camera, colmap_map: ColmapMap) -> float:
    """Best-effort mean reprojection error on PnP inliers.

    hloc surfaces this directly only in some versions, so we recompute from
    the 2D-3D correspondences embedded in `log`. Returns NaN when unavailable.
    """
    try:
        if not isinstance(log, dict):
            return float("nan")
        keypoints_q = log.get("keypoints_query")
        # Modern hloc puts the 3D point coordinates directly into the log;
        # older versions only have point IDs that need a reconstruction lookup.
        points3D_xyz = log.get("points3D_xyz")
        points3D_ids = log.get("points3D_ids")
        # `inlier_mask` in modern hloc; `inliers` in legacy.
        inliers_mask = ret.get("inlier_mask", ret.get("inliers"))
        if keypoints_q is None or inliers_mask is None:
            return float("nan")
        keypoints_q = np.asarray(keypoints_q)
        inliers_mask = np.asarray(inliers_mask, dtype=bool)
        if inliers_mask.sum() == 0:
            return float("nan")
        pts2d = keypoints_q[inliers_mask]
        if points3D_xyz is not None:
            pts3d = np.asarray(points3D_xyz)[inliers_mask]
        elif points3D_ids is not None:
            pts3d = np.array([
                colmap_map.reconstruction.points3D[int(pid)].xyz
                for pid in np.asarray(points3D_ids)[inliers_mask]
                if int(pid) in colmap_map.reconstruction.points3D
            ])
        else:
            return float("nan")
        if len(pts3d) != len(pts2d) or len(pts3d) == 0:
            return float("nan")
        from .pnp import _reprojection_error
        if "cam_from_world" in ret:
            rig = ret["cam_from_world"]
            R = np.asarray(rig.rotation.matrix(), dtype=np.float64)
            t = np.asarray(rig.translation, dtype=np.float64).reshape(3)
        else:
            from .pnp import _qvec_to_rotmat
            R = _qvec_to_rotmat(np.asarray(ret["qvec"]))
            t = np.asarray(ret["tvec"]).reshape(3)
        return _reprojection_error(pts2d, pts3d, R, t, camera)
    except Exception:  # pragma: no cover - diagnostic only
        return float("nan")
