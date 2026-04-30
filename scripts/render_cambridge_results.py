"""Render evaluation summaries for Cambridge Landmarks scenes.

For each scene this writes a single PNG containing:

    +-------------------------+-------------------------+
    | top-down trajectory     | rotation / translation  |
    | (GT vs estimated test   | error histograms        |
    |  + train cameras)       |                         |
    +-------------------------+-------------------------+

These figures are intended for the README and need only matplotlib +
Pillow + pycolmap (i.e. the `[demo]` extras).

Run:
    python scripts/render_cambridge_results.py shopfacade
    python scripts/render_cambridge_results.py oldhospital
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluate_cambridge import (  # noqa: E402
    parse_nvm, parse_split_file, quat_wxyz_to_rotmat, umeyama_robust,
)

SCENES = {
    "shopfacade": {
        "scene": "/tmp/vml-public/cambridge/ShopFacade",
        "work": "/tmp/vml-public/cambridge/work",
        "title": "Cambridge ShopFacade",
        "out": "docs/assets/cambridge_shopfacade.png",
    },
    "oldhospital": {
        "scene": "/tmp/vml-public/cambridge/OldHospital",
        "work": "/tmp/vml-public/cambridge/work_oh",
        "title": "Cambridge Old Hospital",
        "out": "docs/assets/cambridge_oldhospital.png",
    },
}


def _principal_axes_xy(points: np.ndarray) -> np.ndarray:
    """Return rotation that maps `points` so the longest axis lies along X.

    Cambridge scenes are roughly building-shaped (long+narrow), and the
    raw NVM frame is arbitrarily oriented; rotating to principal axes
    makes top-down plots actually look like a building front rather than
    a tilted mess.
    """
    centred = points - points.mean(0)
    cov = (centred[:, :2].T @ centred[:, :2]) / max(len(centred), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    R2 = eigvecs[:, order]
    if np.linalg.det(R2) < 0:
        R2[:, 1] *= -1
    R3 = np.eye(3)
    R3[:2, :2] = R2.T
    return R3


def render(scene_dir: Path, work_dir: Path, title: str, out_path: Path) -> None:
    gt = parse_nvm(scene_dir / "reconstruction.nvm")
    rec = pycolmap.Reconstruction(str(work_dir / "map" / "sfm"))
    sfm_centres = {img.name: np.asarray(img.projection_center())
                   for img in rec.images.values()}
    train_names = parse_split_file(scene_dir / "dataset_train.txt")
    pairs = [n for n in train_names if n in sfm_centres and n in gt]
    src = np.asarray([sfm_centres[n] for n in pairs])
    dst = np.asarray([gt[n].Cw for n in pairs])
    s, R, t, _ = umeyama_robust(src, dst)

    train_centres_nvm = (s * R @ src.T).T + t
    gt_train_centres = dst

    rot_errs, trans_errs = [], []
    gt_test_C, est_test_C = [], []
    for jf in sorted((work_dir / "results").glob("*.json")):
        flat = jf.stem + ".png"
        g = gt.get(flat)
        if g is None:
            continue
        d = json.loads(jf.read_text())
        if not d["success"]:
            continue
        R_cw = np.asarray(d["pose"]["R"])
        t_cw = np.asarray(d["pose"]["t"])
        C_sfm = -R_cw.T @ t_cw
        C_nvm = s * R @ C_sfm + t
        R_cw_gt = quat_wxyz_to_rotmat(g.qcw)
        cos = np.clip((np.trace((R @ R_cw.T) @ R_cw_gt) - 1) / 2, -1.0, 1.0)
        rot_errs.append(float(np.rad2deg(np.arccos(cos))))
        trans_errs.append(float(np.linalg.norm(C_nvm - g.Cw)))
        gt_test_C.append(g.Cw)
        est_test_C.append(C_nvm)

    gt_test_C = np.asarray(gt_test_C)
    est_test_C = np.asarray(est_test_C)

    # Rotate everything to principal axes so a top-down plot is interpretable.
    all_train = np.concatenate([train_centres_nvm, gt_train_centres], axis=0)
    R_pa = _principal_axes_xy(all_train)

    def rot(x):
        return (R_pa @ x.T).T

    rec_train = rot(train_centres_nvm)
    gt_train = rot(gt_train_centres)
    gt_test = rot(gt_test_C)
    est_test = rot(est_test_C)

    fig = plt.figure(figsize=(13, 5.4), dpi=110)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0])

    # ----- left: top-down trajectory
    ax = fig.add_subplot(gs[0])
    ax.scatter(gt_train[:, 0], gt_train[:, 1], s=14, color="#9ca3af",
               alpha=0.65, label=f"train cameras ({len(gt_train)})", zorder=1)
    # Draw error vectors first so they sit under the markers
    for a, b in zip(gt_test, est_test):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#f87171",
                alpha=0.55, linewidth=0.8, zorder=2)
    ax.scatter(gt_test[:, 0], gt_test[:, 1], s=42, color="#10b981",
               alpha=0.95, edgecolor="black", linewidth=0.4,
               label=f"GT test ({len(gt_test)})", zorder=3)
    ax.scatter(est_test[:, 0], est_test[:, 1], s=24, color="#f59e0b",
               alpha=0.95, edgecolor="black", linewidth=0.3,
               label=f"estimated test ({len(est_test)})", zorder=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"{title} — top-down (NVM frame)", fontsize=12, fontweight="bold")
    ax.set_xlabel("X (m, scene principal axis)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # ----- right: histograms
    ax2 = fig.add_subplot(gs[1])
    rot_arr = np.asarray(rot_errs)
    tr_arr = np.asarray(trans_errs)
    ax3 = ax2.twiny()
    ax2.hist(rot_arr, bins=20, color="#3b82f6",
             alpha=0.7, label=f"rotation err (deg) — median {np.median(rot_arr):.2f}")
    ax3.hist(tr_arr, bins=20, color="#f59e0b",
             alpha=0.55, label=f"translation err (m) — median {np.median(tr_arr):.2f}")
    ax2.set_xlabel("rotation error (deg)", color="#3b82f6")
    ax3.set_xlabel("translation error (m)", color="#f59e0b")
    ax2.set_ylabel("# test images")
    ax2.tick_params(axis="x", colors="#3b82f6")
    ax3.tick_params(axis="x", colors="#f59e0b")
    ax2.set_title(f"{title} — error distribution ({len(rot_arr)} test imgs)",
                  fontsize=12, fontweight="bold")
    # Combined legend
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax2.legend(h2 + h3, l2 + l3, loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path}  ({size_kb:.0f} KB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*",
                    choices=list(SCENES.keys()))
    args = ap.parse_args()
    scenes = args.scenes or list(SCENES.keys())
    for key in scenes:
        cfg = SCENES[key]
        render(Path(cfg["scene"]), Path(cfg["work"]), cfg["title"],
               Path(cfg["out"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
