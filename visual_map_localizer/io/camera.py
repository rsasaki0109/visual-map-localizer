"""Helpers for building / inferring a `pycolmap.Camera` for a query image."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Tuple, TYPE_CHECKING

from PIL import Image, ExifTags
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pycolmap


_EXIF_FOCAL = "FocalLength"
_EXIF_FPLANE_X_RES = "FocalPlaneXResolution"
_EXIF_FPLANE_RESUNIT = "FocalPlaneResolutionUnit"
_DEFAULT_FOV_DEG = 60.0  # last-resort fallback when nothing else is known


def build_camera(
    model: str,
    width: int,
    height: int,
    params: Sequence[float],
) -> "pycolmap.Camera":
    """Construct a `pycolmap.Camera` from explicit intrinsics."""
    import pycolmap  # local import keeps the package light-weight to import
    return pycolmap.Camera(
        model=model,
        width=int(width),
        height=int(height),
        params=list(params),
    )


def infer_camera(image_path: Path) -> "pycolmap.Camera":
    """Best-effort camera inference for a query image.

    Strategy (in order):
        1. Use `pycolmap.infer_camera_from_image` (reads EXIF / sensor DB).
        2. Parse EXIF focal length manually.
        3. Fall back to a 60-degree FoV pinhole approximation.
    """
    image_path = Path(image_path)
    try:
        import pycolmap
        cam = pycolmap.infer_camera_from_image(str(image_path))
        if cam is not None:
            return cam
    except Exception:  # pragma: no cover - depends on pycolmap build
        pass

    width, height = _image_size(image_path)
    focal = _focal_from_exif(image_path, width)
    if focal is None:
        focal = _focal_from_fov(width, _DEFAULT_FOV_DEG)
    return build_camera(
        model="SIMPLE_PINHOLE",
        width=width,
        height=height,
        params=[focal, width / 2.0, height / 2.0],
    )


# --------------------------------------------------------------------- EXIF
def _image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (w, h)


def _focal_from_exif(path: Path, width: int) -> Optional[float]:
    try:
        with Image.open(path) as im:
            exif = im.getexif()
        if not exif:
            return None
        tag_lut = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    except Exception:
        return None

    focal_mm = tag_lut.get(_EXIF_FOCAL)
    if focal_mm is None:
        return None
    focal_mm = float(focal_mm)

    # If a focal-plane resolution is present, derive pixel focal directly.
    fplane_x = tag_lut.get(_EXIF_FPLANE_X_RES)
    fplane_unit = tag_lut.get(_EXIF_FPLANE_RESUNIT)  # 2=inch, 3=cm
    if fplane_x and fplane_unit in (2, 3):
        # pixels per (mm) on the sensor:
        pixels_per_mm = float(fplane_x) / (25.4 if fplane_unit == 2 else 10.0)
        return focal_mm * pixels_per_mm

    # Otherwise, assume a 35-mm-equivalent sensor diagonal of 36 mm.
    return focal_mm * width / 36.0


def _focal_from_fov(width: int, fov_deg: float) -> float:
    return 0.5 * width / np.tan(0.5 * np.deg2rad(fov_deg))
