"""I/O helpers (COLMAP map, camera intrinsics, JSON results).

`colmap_map` and `camera` import pycolmap, which is heavy. They are imported
lazily so `from visual_map_localizer.io.output import ...` stays cheap.
"""
from __future__ import annotations
import importlib
from typing import TYPE_CHECKING

from .output import LocalizationResult, dump_json

_LAZY = {
    "ColmapMap": "visual_map_localizer.io.colmap_map",
    "build_camera": "visual_map_localizer.io.camera",
    "infer_camera": "visual_map_localizer.io.camera",
}

__all__ = [
    "ColmapMap",
    "LocalizationResult",
    "dump_json",
    "build_camera",
    "infer_camera",
]


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target)
    value = getattr(module, name)
    globals()[name] = value
    return value


if TYPE_CHECKING:  # pragma: no cover
    from .colmap_map import ColmapMap  # noqa: F401
    from .camera import build_camera, infer_camera  # noqa: F401
