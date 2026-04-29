"""`visual-map-localizer` command-line entry point."""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

import click

from .. import __version__


# --------------------------------------------------------------- logging
def _configure_logging(verbose: int) -> None:
    level = logging.WARNING if verbose <= 0 else (
        logging.INFO if verbose == 1 else logging.DEBUG
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # hloc and pycolmap are quite chatty by default — keep them aligned.
    logging.getLogger("hloc").setLevel(level)


# ----------------------------------------------------------------- root
@click.group(help="Single-image 6DoF Visual Positioning System.")
@click.version_option(__version__, prog_name="visual-map-localizer")
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v, -vv).")
@click.pass_context
def cli(ctx: click.Context, verbose: int) -> None:
    _configure_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# --------------------------------------------------------------- build-map
@cli.command("build-map")
@click.option(
    "--images", "image_dir",
    required=True, type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory of input images (recursively scanned).",
)
@click.option(
    "--output", "output_dir",
    required=True, type=click.Path(file_okay=False, path_type=Path),
    help="Map output directory.",
)
@click.option(
    "--local-feature", default="superpoint_aachen", show_default=True,
    help="hloc local feature config (e.g. superpoint_aachen, superpoint_max, disk).",
)
@click.option(
    "--global-descriptor", default="netvlad", show_default=True,
    help="hloc global descriptor config (e.g. netvlad, openibl, eigenplaces).",
)
@click.option(
    "--matcher", default="superpoint+lightglue", show_default=True,
    help="hloc matcher config (e.g. superpoint+lightglue, superglue).",
)
@click.option(
    "--num-covisible-pairs", "num_covisible_pairs",
    type=int, default=None,
    help="If set, build retrieval-based pairs with this Top-K. Otherwise exhaustive pairs.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing intermediate files.")
def build_map_cmd(
    image_dir: Path,
    output_dir: Path,
    local_feature: str,
    global_descriptor: str,
    matcher: str,
    num_covisible_pairs: int | None,
    overwrite: bool,
) -> None:
    """Build a map (SfM reconstruction + descriptors) from a folder of images."""
    from ..config import MappingConfig
    from ..mapping import build_map

    cfg = MappingConfig(
        local_feature=local_feature,
        global_descriptor=global_descriptor,
        matcher=matcher,
        num_covisible_pairs=num_covisible_pairs,
    )
    out = build_map(image_dir, output_dir, config=cfg, overwrite=overwrite)
    click.echo(f"Map ready: {out}")


# ----------------------------------------------------------------- localize
@cli.command("localize")
@click.option(
    "--map", "map_dir",
    required=True, type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Map directory produced by `build-map`.",
)
@click.option(
    "--query", "query_image",
    required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Query image path.",
)
@click.option(
    "--top-k", "top_k", default=10, show_default=True, type=int,
    help="Number of retrieval candidates.",
)
@click.option(
    "--ransac-max-error", "ransac_max_error", default=12.0, show_default=True, type=float,
    help="RANSAC reprojection threshold in pixels.",
)
@click.option(
    "--min-inliers", "min_inliers", default=12, show_default=True, type=int,
    help="Minimum inliers for a localization to be considered successful.",
)
@click.option(
    "--matcher", default="superpoint+lightglue", show_default=True,
    help="hloc matcher config (must match the map's matcher for best results).",
)
@click.option(
    "--camera-model", default=None,
    help="Optional COLMAP camera model name (e.g. PINHOLE). Inferred from EXIF when omitted.",
)
@click.option(
    "--camera-params", default=None,
    help="Comma-separated COLMAP camera parameters matching --camera-model.",
)
@click.option(
    "--image-size", default=None,
    help="WxH override for the query image (used with --camera-model).",
)
@click.option(
    "--output", "json_out", default=None, type=click.Path(dir_okay=False, path_type=Path),
    help="Optional path to write the JSON result. Defaults to stdout.",
)
@click.option(
    "--keep-work-dir", is_flag=True,
    help="Keep the temporary working directory created during localization.",
)
def localize_cmd(
    map_dir: Path,
    query_image: Path,
    top_k: int,
    ransac_max_error: float,
    min_inliers: int,
    matcher: str,
    camera_model: str | None,
    camera_params: str | None,
    image_size: str | None,
    json_out: Path | None,
    keep_work_dir: bool,
) -> None:
    """Localize a single query image against a previously built map."""
    from ..config import LocalizeConfig
    from ..localization.pipeline import VisualMapLocalizer
    from ..io.output import dump_json

    cfg = LocalizeConfig(
        matcher=matcher,
        top_k=top_k,
        ransac_max_error_px=ransac_max_error,
        ransac_min_inliers=min_inliers,
        camera_model=camera_model,
        camera_params=_parse_floats(camera_params) if camera_params else None,
        image_size=_parse_size(image_size) if image_size else None,
    )
    localizer = VisualMapLocalizer(map_dir, config=cfg)
    result = localizer.localize(query_image, keep_work_dir=keep_work_dir)
    if json_out:
        dump_json(result, json_out)
    click.echo(result.to_json())
    sys.exit(0 if result.success else 2)


# ----------------------------------------------------------------- inspect
@cli.command("inspect")
@click.option(
    "--map", "map_dir",
    required=True, type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def inspect_cmd(map_dir: Path) -> None:
    """Print a summary of a built map (image / point counts, files)."""
    from ..config import META_FILE, SFM_DIRNAME
    from ..io.colmap_map import ColmapMap

    meta_path = map_dir / META_FILE
    summary: dict = {"map_dir": str(map_dir)}
    if meta_path.exists():
        summary["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    sfm_dir = map_dir / SFM_DIRNAME
    if sfm_dir.exists():
        m = ColmapMap.load(sfm_dir)
        summary["reconstruction"] = {
            "num_images": m.num_images,
            "num_points3D": m.num_points,
            "num_cameras": m.num_cameras,
        }
    click.echo(json.dumps(summary, indent=2, ensure_ascii=False))


# ---------------------------------------------------------- option parsers
def _parse_floats(text: str):
    return [float(x) for x in text.replace(" ", "").split(",") if x]


def _parse_size(text: str):
    parts = text.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise click.BadParameter(f"--image-size must be WxH or W,H, got {text!r}")
    return (int(parts[0]), int(parts[1]))


# --------------------------------------------------------------- main
def main() -> None:
    cli(prog_name="visual-map-localizer")


if __name__ == "__main__":  # pragma: no cover
    main()
