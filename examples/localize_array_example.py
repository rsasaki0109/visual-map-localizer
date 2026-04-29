"""Localize an in-memory numpy image (no temp file at the call site).

Usage:
    python examples/localize_array_example.py --map ./map --query ./query.jpg

This example loads the query off disk just to mimic a streaming source —
the point is that `localize()` is given a `np.ndarray`, not a file path.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import pycolmap

from visual_map_localizer import VisualMapLocalizer
from visual_map_localizer.config import LocalizeConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--map",   type=Path, required=True)
    p.add_argument("--query", type=Path, required=True)
    p.add_argument(
        "--camera", default="2559.68,1536,1152,-0.0204997",
        help="SIMPLE_RADIAL params (f, cx, cy, k1)",
    )
    args = p.parse_args()

    rgb = np.asarray(Image.open(args.query).convert("RGB"))
    h, w, _ = rgb.shape
    f, cx, cy, k1 = (float(x) for x in args.camera.split(","))
    cam = pycolmap.Camera(model="SIMPLE_RADIAL", width=w, height=h,
                          params=[f, cx, cy, k1])

    localizer = VisualMapLocalizer(args.map, config=LocalizeConfig(top_k=10))
    res = localizer.localize(rgb, camera=cam, name="live.png")
    print(res.to_json())


if __name__ == "__main__":
    main()
