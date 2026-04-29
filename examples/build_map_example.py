"""Programmatic map building.

Usage:
    python examples/build_map_example.py --images ./my_images --output ./map
"""
from __future__ import annotations
import argparse
from pathlib import Path

from visual_map_localizer.config import MappingConfig
from visual_map_localizer.mapping import build_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--num-covisible-pairs", type=int, default=None,
        help="Top-K retrieval-based pairs. Defaults to exhaustive pairs.",
    )
    args = parser.parse_args()

    cfg = MappingConfig(num_covisible_pairs=args.num_covisible_pairs)
    out = build_map(args.images, args.output, config=cfg)
    print(f"Map ready: {out}")


if __name__ == "__main__":
    main()
