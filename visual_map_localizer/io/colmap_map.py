"""Wrapper around a COLMAP sparse reconstruction loaded via pycolmap."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pycolmap


@dataclass
class ColmapMap:
    """Lightweight handle around `pycolmap.Reconstruction`.

    The raw reconstruction is exposed as `.reconstruction` for callers that
    need its full API; common accessors are wrapped for ergonomics.
    """

    reconstruction: pycolmap.Reconstruction
    model_path: Path

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, model_path: str | Path) -> "ColmapMap":
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"COLMAP model not found: {model_path}")
        rec = pycolmap.Reconstruction(str(model_path))
        return cls(reconstruction=rec, model_path=model_path)

    # ---------------------------------------------------------------- summary
    @property
    def num_images(self) -> int:
        return self.reconstruction.num_images()

    @property
    def num_points(self) -> int:
        return self.reconstruction.num_points3D()

    @property
    def num_cameras(self) -> int:
        return self.reconstruction.num_cameras()

    def image_names(self) -> List[str]:
        return [img.name for img in self.reconstruction.images.values()]

    def name_to_id(self) -> Dict[str, int]:
        return {img.name: image_id for image_id, img in self.reconstruction.images.items()}

    # ------------------------------------------------------------------ cameras
    def get_camera_for_image(self, image_id: int) -> pycolmap.Camera:
        image = self.reconstruction.images[image_id]
        return self.reconstruction.cameras[image.camera_id]

    def find_image_by_name(self, name: str) -> pycolmap.Image:
        for image in self.reconstruction.images.values():
            if image.name == name:
                return image
        raise KeyError(f"Image '{name}' not found in reconstruction ({self.num_images} images)")

    # ------------------------------------------------------------------ helpers
    def db_ids_for_names(self, names: List[str]) -> List[int]:
        lut = self.name_to_id()
        missing = [n for n in names if n not in lut]
        if missing:
            raise KeyError(
                f"{len(missing)} retrieval candidates are not in the reconstruction "
                f"(first missing: {missing[0]!r})"
            )
        return [lut[n] for n in names]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ColmapMap(images={self.num_images}, points3D={self.num_points}, "
            f"cameras={self.num_cameras}, path={self.model_path})"
        )
