"""Map building pipeline (SfM via hloc).

Pipeline steps:
    1. Enumerate db images under `image_dir`.
    2. Extract local features (SuperPoint) into ``features.h5``.
    3. Generate image pairs:
         * exhaustive (default for <= 50 images)
         * retrieval-based via NetVLAD (when num_covisible_pairs is set)
    4. Match features (LightGlue) into ``matches-sfm.h5``.
    5. Run COLMAP SfM via ``hloc.reconstruction.main`` -> ``sfm/`` dir.
    6. Extract NetVLAD global descriptors for the localization step.
    7. Persist a ``map_meta.json`` manifest + ``db_images.txt`` index.

The output layout matches what `VisualMapLocalizer` expects.
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from ..config import (
    DB_IMAGE_LIST_FILE,
    FEATURES_FILE,
    GLOBAL_DESC_FILE,
    META_FILE,
    SFM_DIRNAME,
    SFM_PAIRS_FILE,
    SFM_MATCHES_FILE,
    MappingConfig,
)

logger = logging.getLogger(__name__)


def build_map(
    image_dir: str | Path,
    output_dir: str | Path,
    *,
    config: Optional[MappingConfig] = None,
    image_list: Optional[List[str]] = None,
    overwrite: bool = False,
) -> Path:
    """Build a localizer-ready map from a folder of images.

    Returns the resolved `output_dir` containing the SfM reconstruction,
    feature / descriptor files and a manifest.
    """
    cfg = config or MappingConfig()
    image_dir = Path(image_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not image_dir.exists():
        raise FileNotFoundError(f"image dir not found: {image_dir}")

    # ------------------------------------------------------------ enumerate
    if image_list is None:
        image_list = _list_images(image_dir, cfg.image_extensions)
    if not image_list:
        raise RuntimeError(f"no images found under {image_dir}")
    logger.info("Found %d images under %s", len(image_list), image_dir)

    # Persist DB image list early so partial builds are still inspectable.
    (output_dir / DB_IMAGE_LIST_FILE).write_text(
        "\n".join(image_list) + "\n", encoding="utf-8"
    )

    timing: dict = {}

    # ------------------------------------------------------------ features
    from hloc import (  # noqa: WPS433 — heavy import, kept lazy
        extract_features, match_features,
        pairs_from_exhaustive, pairs_from_retrieval,
        reconstruction,
    )

    t = time.perf_counter()
    features_path = extract_features.main(
        extract_features.confs[cfg.local_feature],
        image_dir, output_dir,
        image_list=image_list,
        feature_path=output_dir / FEATURES_FILE,
        overwrite=overwrite,
    )
    timing["local_features"] = time.perf_counter() - t
    logger.info("local features -> %s (%.1fs)", features_path, timing["local_features"])

    # ------------------------------------------------------------ pairs
    sfm_pairs = output_dir / SFM_PAIRS_FILE
    use_retrieval_pairs = (
        cfg.num_covisible_pairs is not None and cfg.num_covisible_pairs > 0
    )

    t = time.perf_counter()
    if use_retrieval_pairs:
        gdesc_path = extract_features.main(
            extract_features.confs[cfg.global_descriptor],
            image_dir, output_dir,
            image_list=image_list,
            feature_path=output_dir / GLOBAL_DESC_FILE,
            overwrite=overwrite,
        )
        pairs_from_retrieval.main(
            gdesc_path, sfm_pairs,
            num_matched=cfg.num_covisible_pairs,
        )
    else:
        pairs_from_exhaustive.main(sfm_pairs, image_list=image_list)
    timing["pairs"] = time.perf_counter() - t
    logger.info("pairs -> %s (%.1fs)", sfm_pairs, timing["pairs"])

    # ------------------------------------------------------------ matching
    from ..matching.matcher import resolve_matcher_conf

    t = time.perf_counter()
    matches_path = match_features.main(
        resolve_matcher_conf(cfg.matcher),
        sfm_pairs,
        features=features_path,
        export_dir=output_dir,
        matches=output_dir / SFM_MATCHES_FILE,
        overwrite=overwrite,
    )
    timing["matching"] = time.perf_counter() - t
    logger.info("matches -> %s (%.1fs)", matches_path, timing["matching"])

    # ------------------------------------------------------------ SfM
    sfm_dir = output_dir / SFM_DIRNAME
    sfm_dir.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    model = reconstruction.main(
        sfm_dir, image_dir, sfm_pairs, features_path, matches_path,
        image_list=image_list,
    )
    timing["sfm"] = time.perf_counter() - t
    if model is None:
        raise RuntimeError(
            "SfM failed — COLMAP could not reconstruct any model. "
            "Try more / better-overlapped images, or pass --num-covisible-pairs."
        )
    n_imgs = model.num_images() if hasattr(model, "num_images") else "?"
    n_pts = model.num_points3D() if hasattr(model, "num_points3D") else "?"
    logger.info(
        "SfM reconstruction: %s images, %s points3D (%.1fs)",
        n_imgs, n_pts, timing["sfm"],
    )

    # ------------------------------------------------------------ retrieval descriptors
    t = time.perf_counter()
    extract_features.main(
        extract_features.confs[cfg.global_descriptor],
        image_dir, output_dir,
        image_list=image_list,
        feature_path=output_dir / GLOBAL_DESC_FILE,
        overwrite=overwrite,
    )
    timing["global_descriptors"] = time.perf_counter() - t
    logger.info("global descriptors ready (%.1fs)", timing["global_descriptors"])

    # ------------------------------------------------------------ manifest
    meta = {
        "version": 1,
        "image_dir": str(image_dir),
        "num_db_images": len(image_list),
        "local_feature": cfg.local_feature,
        "global_descriptor": cfg.global_descriptor,
        "matcher": cfg.matcher,
        "num_covisible_pairs": cfg.num_covisible_pairs,
        "timing_seconds": timing,
        "files": {
            "sfm_dir": SFM_DIRNAME,
            "features": FEATURES_FILE,
            "global_descriptors": GLOBAL_DESC_FILE,
            "db_image_list": DB_IMAGE_LIST_FILE,
            "sfm_pairs": SFM_PAIRS_FILE,
            "sfm_matches": SFM_MATCHES_FILE,
        },
    }
    (output_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("map manifest -> %s", output_dir / META_FILE)
    return output_dir


# --------------------------------------------------------------- helpers
def _list_images(root: Path, extensions: tuple) -> List[str]:
    """Recursively list image files under `root` as paths *relative* to root.

    The relative path is what hloc / COLMAP use as the image identifier,
    so this function defines the canonical naming scheme.
    """
    files: list = []
    ext_set = {e.lower() for e in extensions}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in ext_set:
            files.append(str(p.relative_to(root)).replace("\\", "/"))
    return files
