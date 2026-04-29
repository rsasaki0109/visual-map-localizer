"""visual-map-localizer: single-image 6DoF Visual Positioning System.

Heavy submodules (`localization.pipeline`, `io.colmap_map`) pull in pycolmap
and torch, which can take seconds and may not be available in unit-testing
environments. They are exposed via PEP 562 lazy attributes so simply
importing the package stays cheap.
"""
from __future__ import annotations
import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# Re-exposed names. The implementations live in submodules and are imported
# on first attribute access.
_LAZY = {
    "LocalizationResult": "visual_map_localizer.io.output",
    "VisualMapLocalizer": "visual_map_localizer.localization.pipeline",
}

__all__ = ["LocalizationResult", "VisualMapLocalizer", "__version__"]


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target)
    value = getattr(module, name)
    globals()[name] = value
    return value


if TYPE_CHECKING:  # pragma: no cover - for static type checkers only
    from .io.output import LocalizationResult  # noqa: F401
    from .localization.pipeline import VisualMapLocalizer  # noqa: F401
