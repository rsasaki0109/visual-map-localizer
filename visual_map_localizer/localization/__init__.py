"""Pose estimation pipeline."""
from __future__ import annotations
import importlib
from typing import TYPE_CHECKING

from .pnp import (
    PnPResult,
    pnp_pycolmap,
    pnp_opencv,
    estimate_pose,
)

_LAZY = {
    "VisualMapLocalizer": "visual_map_localizer.localization.pipeline",
}

__all__ = [
    "PnPResult",
    "pnp_pycolmap",
    "pnp_opencv",
    "estimate_pose",
    "VisualMapLocalizer",
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
    from .pipeline import VisualMapLocalizer  # noqa: F401
