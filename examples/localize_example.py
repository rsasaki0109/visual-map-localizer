"""Programmatic single-image localization.

Usage:
    python examples/localize_example.py --map ./map --query ./query.jpg
"""
from __future__ import annotations
import argparse
from pathlib import Path

from visual_map_localizer import VisualMapLocalizer
from visual_map_localizer.config import LocalizeConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    localizer = VisualMapLocalizer(args.map, config=LocalizeConfig(top_k=args.top_k))
    result = localizer.localize(args.query)
    print(result.to_json())


if __name__ == "__main__":
    main()
