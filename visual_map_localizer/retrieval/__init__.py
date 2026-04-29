"""Image retrieval (global descriptors)."""
from .global_features import (
    extract_global_descriptors,
    pairs_from_retrieval,
    AVAILABLE_GLOBAL,
)

__all__ = [
    "extract_global_descriptors",
    "pairs_from_retrieval",
    "AVAILABLE_GLOBAL",
]
