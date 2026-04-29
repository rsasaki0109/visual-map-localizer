"""SuperPoint / DISK local feature extraction (hloc wrapper)."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional


AVAILABLE_LOCAL = (
    "superpoint_aachen",
    "superpoint_max",
    "superpoint_inloc",
    "disk",
    "r2d2",
    "sift",
)


def extract_local_features(
    image_dir: Path,
    out_dir: Path,
    *,
    image_list: Optional[List[str]] = None,
    conf_name: str = "superpoint_aachen",
    feature_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """Extract local features for the given images and append to an HDF5 file.

    If `feature_path` is provided, query features are appended to the existing
    database features file in-place — this is what hloc's localization pipeline
    expects when the same `features.h5` must contain both DB and query keys.
    """
    from hloc import extract_features  # noqa: WPS433

    confs = extract_features.confs
    if conf_name not in confs:
        raise ValueError(
            f"unknown local feature '{conf_name}'. available: {sorted(confs)}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return extract_features.main(
        confs[conf_name],
        Path(image_dir),
        out_dir,
        image_list=image_list,
        feature_path=feature_path,
        overwrite=overwrite,
    )
