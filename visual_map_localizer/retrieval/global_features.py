"""Global image descriptor extraction + retrieval pair generation.

Thin wrapper around the corresponding `hloc` modules so the rest of the
codebase does not need to know how hloc is structured.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional


# Lazy import of hloc — it is heavy and not strictly needed for unit tests
def _load_hloc():
    from hloc import extract_features, pairs_from_retrieval as p2  # noqa: WPS433
    return extract_features, p2


def _global_confs():
    from hloc import extract_features  # noqa: WPS433
    return extract_features.confs


# Re-exposed for the CLI / tests --------------------------------------------
AVAILABLE_GLOBAL = ("netvlad", "openibl", "eigenplaces", "dir")


def extract_global_descriptors(
    image_dir: Path,
    out_dir: Path,
    *,
    image_list: Optional[List[str]] = None,
    conf_name: str = "netvlad",
    feature_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """Extract global descriptors for `image_list` (defaults to every image
    under `image_dir`) and append them to a single HDF5 file.

    Returns the path of the descriptor file.
    """
    extract_features, _ = _load_hloc()
    confs = _global_confs()
    if conf_name not in confs:
        raise ValueError(
            f"unknown global descriptor '{conf_name}'. "
            f"available: {sorted(confs)}"
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


def pairs_from_retrieval(
    descriptor_path: Path,
    output_path: Path,
    *,
    num_matched: int,
    query_list: Optional[List[str]] = None,
    db_list: Optional[List[str]] = None,
) -> Path:
    """Build (query, db) image pairs ranked by global-descriptor similarity."""
    _, p2 = _load_hloc()
    p2.main(
        Path(descriptor_path),
        Path(output_path),
        num_matched=num_matched,
        query_list=query_list,
        db_list=db_list,
    )
    return Path(output_path)
