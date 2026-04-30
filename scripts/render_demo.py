"""Render a side-by-side demo GIF: query image + localized camera pose in 3D.

For each successfully localized south-building query, draws:

    +--------------------------+---------------------------------+
    |                          |   3D point cloud (gray)         |
    |     query JPG            |   db cameras (blue dots)        |
    |                          |   localized query cam (green)   |
    +--------------------------+---------------------------------+

Install the matplotlib + imageio extras before running:

    pip install -e ".[demo]"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def camera_pyramid(C: np.ndarray, R_wc: np.ndarray, scale: float = 0.6) -> np.ndarray:
    """Return 3xN line segments tracing a small camera frustum at world pose."""
    a = scale
    # Pyramid vertices in camera frame (origin at apex, +Z forward)
    apex = np.array([0.0, 0.0, 0.0])
    near = a
    half_w, half_h = a * 0.7, a * 0.5
    fr_tl = np.array([-half_w, -half_h, near])
    fr_tr = np.array([+half_w, -half_h, near])
    fr_br = np.array([+half_w, +half_h, near])
    fr_bl = np.array([-half_w, +half_h, near])

    edges = [
        (apex, fr_tl), (apex, fr_tr), (apex, fr_br), (apex, fr_bl),
        (fr_tl, fr_tr), (fr_tr, fr_br), (fr_br, fr_bl), (fr_bl, fr_tl),
    ]
    segs = []
    for a_pt, b_pt in edges:
        a_w = R_wc @ a_pt + C
        b_w = R_wc @ b_pt + C
        segs.append([a_w, b_w])
    return np.asarray(segs)  # (E, 2, 3)


def render_frame(
    pts3d: np.ndarray,
    db_centers: np.ndarray,
    C_query: np.ndarray,
    R_wc_query: np.ndarray,
    query_img: np.ndarray,
    title: str,
    inliers: int | None = None,
    reproj_error: float | None = None,
    fig_size: tuple[int, int] = (1000, 500),
    dpi: int = 90,
    bbox_min: np.ndarray | None = None,
    bbox_max: np.ndarray | None = None,
) -> np.ndarray:
    fig = plt.figure(figsize=(fig_size[0] / dpi, fig_size[1] / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")

    # --- left: query image
    ax_im = fig.add_subplot(1, 2, 1)
    ax_im.imshow(query_img)
    ax_im.set_title(f"query: {title}", fontsize=12, fontweight="bold")
    ax_im.axis("off")
    if inliers is not None:
        overlay = f"inliers: {inliers}"
        if reproj_error is not None:
            overlay += f"\nreproj err: {reproj_error:.2f} px"
        ax_im.text(
            0.02, 0.98, overlay,
            transform=ax_im.transAxes, va="top", ha="left",
            fontsize=11, color="white",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#10b981", edgecolor="none", alpha=0.9),
        )

    # --- right: 3D scene
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    ax_3d.scatter(
        pts3d[:, 0], pts3d[:, 1], pts3d[:, 2],
        s=1.6, c="#6b7280", alpha=0.8, linewidths=0,
    )
    ax_3d.scatter(
        db_centers[:, 0], db_centers[:, 1], db_centers[:, 2],
        s=22, c="#3b82f6", alpha=0.95, label="db cameras",
        edgecolors="white", linewidths=0.4,
    )
    ax_3d.scatter(
        [C_query[0]], [C_query[1]], [C_query[2]],
        s=220, c="#10b981", marker="o", edgecolors="black",
        linewidths=1.5, label="localized query", zorder=10,
    )
    pyr = camera_pyramid(C_query, R_wc_query, scale=0.85)
    pyr_lines = Line3DCollection(pyr, colors="#10b981", linewidths=2.4)
    ax_3d.add_collection3d(pyr_lines)

    if bbox_min is not None and bbox_max is not None:
        ax_3d.set_xlim(bbox_min[0], bbox_max[0])
        ax_3d.set_ylim(bbox_min[1], bbox_max[1])
        ax_3d.set_zlim(bbox_min[2], bbox_max[2])
    ax_3d.view_init(elev=22, azim=-65)
    # Hide tick labels — they don't add information for a demo plot
    ax_3d.set_xticklabels([])
    ax_3d.set_yticklabels([])
    ax_3d.set_zticklabels([])
    ax_3d.set_title("south-building map + localized camera",
                    fontsize=12, fontweight="bold")
    ax_3d.legend(loc="upper right", fontsize=10)

    # COLMAP world axes are arbitrary; flatten the look so the building reads.
    ax_3d.set_box_aspect([1.0, 0.55, 1.0])

    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-dir", type=Path, default=Path("/tmp/vml-public/map"))
    ap.add_argument("--results-dir", type=Path,
                    default=Path("/tmp/vml-public/results"))
    ap.add_argument("--query-dir", type=Path,
                    default=Path("/tmp/vml-public/query_images"))
    ap.add_argument("--out", type=Path,
                    default=Path("docs/assets/demo.gif"))
    ap.add_argument("--frame-duration", type=float, default=1.2,
                    help="seconds per frame in the GIF")
    ap.add_argument("--max-points", type=int, default=8000)
    ap.add_argument("--query-thumb-width", type=int, default=900)
    args = ap.parse_args()

    sfm_dir = args.map_dir / "sfm"
    if not sfm_dir.exists():
        print(f"SfM dir not found: {sfm_dir}", file=sys.stderr)
        return 1
    rec = pycolmap.Reconstruction(str(sfm_dir))

    # Subsampled point cloud + bbox
    xyz = np.array([p.xyz for p in rec.points3D.values()])
    if len(xyz) > args.max_points:
        rng = np.random.default_rng(0)
        xyz = xyz[rng.choice(len(xyz), args.max_points, replace=False)]
    bbox_min = xyz.min(0)
    bbox_max = xyz.max(0)

    # DB camera centres in world frame (pycolmap >= 0.6 API)
    db_centers = np.asarray([
        np.asarray(img.projection_center()) for img in rec.images.values()
    ])

    # Slightly expand bbox so query cam stays inside the plot frame
    pad = 0.05 * (bbox_max - bbox_min)
    bbox_min = bbox_min - pad
    bbox_max = bbox_max + pad

    result_files = sorted(args.results_dir.glob("*.json"))
    if not result_files:
        print(f"no result JSONs under {args.results_dir}", file=sys.stderr)
        return 1

    frames: list[np.ndarray] = []
    for jf in result_files:
        data = json.loads(jf.read_text())
        if not data.get("success"):
            continue
        R_cw = np.asarray(data["pose"]["R"])
        t_cw = np.asarray(data["pose"]["t"])
        C = -R_cw.T @ t_cw
        R_wc = R_cw.T

        img_path = args.query_dir / f"{jf.stem}.JPG"
        if not img_path.exists():
            print(f"(skip {jf.stem}: query image missing)", file=sys.stderr)
            continue
        query = Image.open(img_path).convert("RGB")
        if query.width > args.query_thumb_width:
            scale = args.query_thumb_width / query.width
            query = query.resize(
                (args.query_thumb_width, int(query.height * scale)),
                Image.LANCZOS,
            )
        query_arr = np.asarray(query)

        frame = render_frame(
            xyz, db_centers, C, R_wc, query_arr,
            title=img_path.name,
            inliers=data.get("inliers"),
            reproj_error=data.get("reproj_error"),
            bbox_min=bbox_min, bbox_max=bbox_max,
        )
        frames.append(frame)
        print(f"  rendered {jf.stem}  ({len(frames)}/{len(result_files)})")

    if not frames:
        print("no frames rendered", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(args.frame_duration * 1000)
    imageio.mimsave(args.out, frames, duration=duration_ms, loop=0)
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
