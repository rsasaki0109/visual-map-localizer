"""Default configuration values shared across modules."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# Sub-directory layout inside a built map directory ------------------------
SFM_DIRNAME = "sfm"                  # COLMAP sparse reconstruction
FEATURES_FILE = "features.h5"        # local features (SuperPoint)
GLOBAL_DESC_FILE = "global_descriptors.h5"  # global descriptors (NetVLAD)
SFM_PAIRS_FILE = "pairs-sfm.txt"
LOC_PAIRS_FILE = "pairs-loc.txt"
SFM_MATCHES_FILE = "matches-sfm.h5"
LOC_MATCHES_FILE = "matches-loc.h5"
DB_IMAGE_LIST_FILE = "db_images.txt"
META_FILE = "map_meta.json"


@dataclass
class MappingConfig:
    """Configuration for the build-map pipeline."""

    local_feature: str = "superpoint_aachen"
    global_descriptor: str = "netvlad"
    matcher: str = "superpoint+lightglue"
    # If None -> exhaustive pairs (small datasets); otherwise retrieval pairs
    num_covisible_pairs: Optional[int] = None
    camera_mode: str = "AUTO"  # 'AUTO', 'SINGLE', 'PER_IMAGE', 'PER_FOLDER'
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


@dataclass
class LocalizeConfig:
    """Configuration for the localize pipeline."""

    local_feature: str = "superpoint_aachen"
    global_descriptor: str = "netvlad"
    matcher: str = "superpoint+lightglue"
    top_k: int = 10
    ransac_max_error_px: float = 12.0
    ransac_min_inliers: int = 12
    use_pycolmap_pnp: bool = True  # fall back to OpenCV when False or unavailable
    # Optional intrinsics override (otherwise inferred from EXIF when possible).
    camera_model: Optional[str] = None       # e.g. "PINHOLE", "SIMPLE_RADIAL"
    camera_params: Optional[list] = None     # model-specific params
    image_size: Optional[tuple] = None       # (width, height)
