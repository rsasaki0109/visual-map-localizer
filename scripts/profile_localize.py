"""Persistent-process latency benchmark.

Loads a built map ONCE, then localizes every image in a query directory and
prints the per-step timing breakdown.

Usage::

    python scripts/profile_localize.py \\
        --map      /path/to/map \\
        --queries  /path/to/query_images \\
        [--use-ndarray] [--camera fx,fy,cx,cy] [--camera-model PINHOLE]
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

T0 = time.perf_counter()

import numpy as np
from PIL import Image

from visual_map_localizer import VisualMapLocalizer
from visual_map_localizer.config import LocalizeConfig

import_done = time.perf_counter()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map",     required=True, type=Path)
    p.add_argument("--queries", required=True, type=Path)
    p.add_argument("--top-k",   type=int, default=10)
    p.add_argument("--use-ndarray", action="store_true",
                   help="Pass images as np.ndarray (RGB) instead of file paths.")
    p.add_argument("--camera-model", default=None,
                   help="COLMAP camera model (required with --use-ndarray).")
    p.add_argument("--camera", default=None,
                   help="Comma-separated COLMAP camera params.")
    p.add_argument("--ext", default=".JPG",
                   help="Image extension to search for (default .JPG).")
    args = p.parse_args()

    print(f"[boot] python+imports : {import_done - T0:5.2f}s")

    localizer = VisualMapLocalizer(args.map, config=LocalizeConfig(top_k=args.top_k))
    init_done = time.perf_counter()
    print(f"[boot] map load       : {init_done - import_done:5.2f}s")

    queries = sorted(args.queries.glob(f"*{args.ext}"))
    if not queries:
        raise SystemExit(f"no {args.ext} files in {args.queries}")

    cam = None
    if args.use_ndarray:
        if args.camera_model is None or args.camera is None:
            raise SystemExit("--use-ndarray requires --camera-model and --camera")
        import pycolmap
        sample = np.asarray(Image.open(queries[0]).convert("RGB"))
        cam = pycolmap.Camera(
            model=args.camera_model,
            width=sample.shape[1], height=sample.shape[0],
            params=[float(x) for x in args.camera.split(",")],
        )

    print(f"\n{'#':>3} {'query':<22} {'gd':>5} {'rt':>5} {'lf':>5} {'mt':>5} "
          f"{'pp':>5} {'inl':>5} {'TOTAL':>6}")
    print("-" * 75)
    totals = []
    for i, q in enumerate(queries, 1):
        if args.use_ndarray:
            arg = np.asarray(Image.open(q).convert("RGB"))
        else:
            arg = q
        t = time.perf_counter()
        res = localizer.localize(arg, camera=cam, name=q.stem + ".png")
        elapsed = time.perf_counter() - t
        totals.append(elapsed)
        if not res.success:
            print(f"{i:>3} {q.stem:<22} FAILED ({res.error})")
            continue
        tm = res.timing
        print(f"{i:>3} {q.stem:<22} "
              f"{tm.get('global_descriptor', 0):>5.2f} "
              f"{tm.get('retrieval', 0):>5.2f} "
              f"{tm.get('local_features', 0):>5.2f} "
              f"{tm.get('matching', 0):>5.2f} "
              f"{tm.get('pnp', 0):>5.2f} "
              f"{res.inliers:>5d} "
              f"{elapsed:>6.2f}")
    print("-" * 75)
    if totals:
        print(f"first call (warmup): {totals[0]:5.2f}s")
        if len(totals) > 1:
            tail = totals[1:]
            print(f"steady-state       : mean={sum(tail) / len(tail):5.2f}s, "
                  f"min={min(tail):5.2f}s, max={max(tail):5.2f}s")
            print(f"throughput (excl. warmup): {len(tail) / sum(tail):.2f} Hz")


if __name__ == "__main__":
    main()
