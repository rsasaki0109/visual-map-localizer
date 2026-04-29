"""SuperGlue / LightGlue / nearest-neighbour matching (hloc wrapper)."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional


AVAILABLE_MATCHERS = (
    "superpoint+lightglue",     # current default
    "disk+lightglue",
    "superglue",
    "superglue-fast",
    "NN-superpoint",
    "NN-mutual",
)


def resolve_matcher_conf(name: str):
    """Look up an hloc matcher config by name with backwards-compatible aliases."""
    from hloc import match_features  # noqa: WPS433

    confs = match_features.confs
    if name in confs:
        return confs[name]
    fallback_aliases = {
        "superpoint+lightglue": ("lightglue",),
        "disk+lightglue": ("disk-lightglue", "lightglue-disk"),
    }
    for alias in fallback_aliases.get(name, ()):
        if alias in confs:
            return confs[alias]
    raise ValueError(
        f"unknown matcher '{name}'. available: {sorted(confs)}"
    )


# Kept for backwards-compatibility with internal callers.
_resolve_matcher_conf = resolve_matcher_conf


def match_features(
    pairs_path: Path,
    feature_path: Path,
    out_dir: Path,
    *,
    matches_path: Optional[Path] = None,
    conf_name: str = "superpoint+lightglue",
    overwrite: bool = False,
) -> Path:
    """Run pairwise matching for (image_a, image_b) pairs from `pairs_path`.

    Both database features and query features are expected to live inside
    the same `feature_path` HDF5 file (this is the convention hloc uses).

    Parameters
    ----------
    pairs_path:    text file with one ``image_a image_b`` pair per line
    feature_path:  features.h5 (must contain entries for *all* names in pairs)
    out_dir:       directory where matches.h5 is written, if `matches_path`
                   is not given explicitly
    matches_path:  optional override for the matches output file
    """
    from hloc import match_features as mf  # noqa: WPS433

    conf = resolve_matcher_conf(conf_name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return mf.main(
        conf,
        Path(pairs_path),
        features=Path(feature_path),
        export_dir=out_dir,
        matches=matches_path,
        overwrite=overwrite,
    )


def pairs_from_exhaustive(
    pairs_out: Path,
    *,
    image_list: List[str],
) -> Path:
    """All-pairs combinations for small datasets (<~50 images)."""
    from hloc import pairs_from_exhaustive as pe  # noqa: WPS433

    pe.main(Path(pairs_out), image_list=image_list)
    return Path(pairs_out)
