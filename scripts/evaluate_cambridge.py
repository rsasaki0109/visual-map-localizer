"""End-to-end evaluation on a Cambridge Landmarks scene (e.g. ShopFacade).

Stages:
  1. Parse the scene's reconstruction.nvm for per-image GT pose + intrinsics.
  2. Stage train/test images as flat symlink trees (build-map expects a flat
     directory; the scene uses seqN/frameXXXXX.png subpaths).
  3. Caller runs `visual-map-localizer build-map` on the train tree,
     then `localize` on every test image.
  4. This script reads back the localizer JSONs, performs Sim(3) alignment
     between our SfM frame and the NVM frame using the *train* poses, then
     reports rotation / translation error of every test pose against GT.

Run as:
    python scripts/evaluate_cambridge.py prepare \\
        --scene /tmp/vml-public/cambridge/ShopFacade \\
        --work /tmp/vml-public/cambridge/work

    visual-map-localizer build-map \\
        --images /tmp/vml-public/cambridge/work/train_images \\
        --output /tmp/vml-public/cambridge/work/map \\
        --local-feature disk --matcher disk+lightglue \\
        --num-covisible-pairs 20

    python scripts/evaluate_cambridge.py localize-all \\
        --scene /tmp/vml-public/cambridge/ShopFacade \\
        --work /tmp/vml-public/cambridge/work

    python scripts/evaluate_cambridge.py score \\
        --scene /tmp/vml-public/cambridge/ShopFacade \\
        --work /tmp/vml-public/cambridge/work
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Allow `python3 scripts/evaluate_cambridge.py` to find the package without
# needing an editable install (handy when running on a stock interpreter).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------
# NVM:                   q (camera-to-world, wxyz), C (camera centre in world)
# our localizer JSON:    R, t  (camera-from-world; pos = -R^T t = camera centre)
# Cambridge dataset_*.txt has the same convention as NVM.

@dataclass
class GtRecord:
    name: str          # flat name, e.g. "seq2_frame00001.png"
    focal: float
    k1: float
    qcw: np.ndarray    # NVM stores world-to-camera quaternion (wxyz)
    Cw: np.ndarray     # camera centre in world


def parse_nvm(nvm_path: Path) -> Dict[str, GtRecord]:
    text = nvm_path.read_text().splitlines()
    if not text or not text[0].startswith("NVM_V3"):
        raise ValueError(f"unexpected NVM header: {text[0]!r}")
    # Skip "NVM_V3", blank, count
    n = int(text[2].strip())
    out: Dict[str, GtRecord] = {}
    for line in text[3:3 + n]:
        parts = line.split()
        if len(parts) < 11:
            continue
        name = parts[0]
        focal = float(parts[1])
        qw, qx, qy, qz = (float(parts[i]) for i in range(2, 6))
        cx, cy, cz = (float(parts[i]) for i in range(6, 9))
        k1 = float(parts[9])
        # Cambridge stores .jpg in NVM but .png on disk.
        png_name = Path(name).with_suffix(".png").as_posix()
        flat = png_name.replace("/", "_")
        out[flat] = GtRecord(
            name=flat, focal=focal, k1=k1,
            qcw=np.array([qw, qx, qy, qz]),
            Cw=np.array([cx, cy, cz]),
        )
    return out


def parse_split_file(path: Path) -> List[str]:
    """Return list of flat image names from dataset_train/test.txt."""
    out: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("ImageFile") or line.startswith("Visual"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        out.append(parts[0].replace("/", "_"))
    return out


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Estimate Sim(3) (s, R, t) such that dst ~ s * R @ src + t."""
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    sigma_s = ((src - mu_s) ** 2).sum() / len(src)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    s = float((D * np.diag(S)).sum() / sigma_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


def umeyama_robust(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    n_iters: int = 5,
    inlier_factor: float = 5.0,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """IRLS-style Sim(3): drop train pairs whose residual exceeds
    `inlier_factor * median(residuals)` and refit, repeating up to
    `n_iters` times. Useful when our SfM happens to misregister a
    handful of train images vs. the reference NVM and these outliers
    bias the LS alignment.

    Returns (s, R, t, inlier_mask) where inlier_mask is bool over `src`.
    """
    s, R, t = umeyama(src, dst)
    aligned = (s * R @ src.T).T + t
    resid = np.linalg.norm(aligned - dst, axis=1)
    keep = np.ones(len(src), dtype=bool)
    for _ in range(n_iters):
        med = float(np.median(resid[keep]))
        new_keep = resid <= max(med * inlier_factor, 1e-9)
        if int(new_keep.sum()) == int(keep.sum()) or new_keep.sum() < 4:
            keep = new_keep
            break
        keep = new_keep
        s, R, t = umeyama(src[keep], dst[keep])
        aligned = (s * R @ src.T).T + t
        resid = np.linalg.norm(aligned - dst, axis=1)
    return s, R, t, keep


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> int:
    scene = Path(args.scene)
    work = Path(args.work)

    train_names = parse_split_file(scene / "dataset_train.txt")
    test_names = parse_split_file(scene / "dataset_test.txt")
    print(f"[prepare] train: {len(train_names)}  test: {len(test_names)}")

    train_dir = work / "train_images"
    test_dir = work / "test_images"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    def stage(names: List[str], target: Path) -> int:
        ok = 0
        for flat in names:
            # flat = "seq2_frame00001.png" -> on-disk "seq2/frame00001.png"
            seq, frame = flat.split("_", 1)
            src = scene / seq / frame
            if not src.exists():
                print(f"  miss: {src}", file=sys.stderr)
                continue
            os.symlink(src.resolve(), target / flat)
            ok += 1
        return ok

    n_train = stage(train_names, train_dir)
    n_test = stage(test_names, test_dir)
    print(f"[prepare] staged {n_train} train + {n_test} test symlinks under {work}")

    # Persist intrinsics so later stages don't re-read the NVM.
    gt = parse_nvm(scene / "reconstruction.nvm")
    matched = [name for name in train_names + test_names if name in gt]
    focals = [gt[n].focal for n in matched]
    k1s = [gt[n].k1 for n in matched]
    intr = {
        "width": 1920, "height": 1080,
        "fx": float(np.median(focals)), "fy": float(np.median(focals)),
        "cx": 1920 / 2, "cy": 1080 / 2,
        "k1": float(np.median(k1s)),
    }
    (work / "intrinsics.json").write_text(json.dumps(intr, indent=2))
    print(f"[prepare] intrinsics -> {work / 'intrinsics.json'}: {intr}")
    return 0


def cmd_localize_all(args: argparse.Namespace) -> int:
    """Localize every test image through a single persistent VisualMapLocalizer.

    Subprocess-per-call (CLI) would pay the ~4 s warm-up on each of the 103
    test images. Doing it in-process collapses to ~1.4 s steady-state.
    """
    work = Path(args.work)
    intr = json.loads((work / "intrinsics.json").read_text())

    test_dir = work / "test_images"
    map_dir = work / "map"
    results_dir = work / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    import time
    import numpy as np
    import pycolmap
    from PIL import Image
    from visual_map_localizer import VisualMapLocalizer
    from visual_map_localizer.config import LocalizeConfig

    localizer = VisualMapLocalizer(map_dir, config=LocalizeConfig(top_k=10))
    camera = pycolmap.Camera(
        model="SIMPLE_RADIAL",
        width=intr["width"], height=intr["height"],
        params=[intr["fx"], intr["cx"], intr["cy"], intr["k1"]],
    )

    test_imgs = sorted(test_dir.glob("*.png"))
    print(f"[localize-all] {len(test_imgs)} test images")
    n_succ = 0
    times = []
    for i, p in enumerate(test_imgs, 1):
        out_json = results_dir / f"{p.stem}.json"
        if out_json.exists() and not args.overwrite:
            continue
        rgb = np.asarray(Image.open(p).convert("RGB"))
        t0 = time.perf_counter()
        result = localizer.localize(rgb, camera=camera, name=p.name)
        dt = time.perf_counter() - t0
        times.append(dt)
        out_json.write_text(result.to_json())
        if result.success:
            n_succ += 1
        flag = "ok" if result.success else "FAIL"
        print(f"  [{i:3}/{len(test_imgs)}] {p.stem}  {flag}  "
              f"inliers={result.inliers}  {dt:.2f}s")
    if times:
        med = sorted(times)[len(times) // 2]
        print(f"[localize-all] {n_succ}/{len(test_imgs)} success  "
              f"median latency {med:.2f}s")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    scene = Path(args.scene)
    work = Path(args.work)
    sfm_dir = work / "map" / "sfm"
    results_dir = work / "results"

    import pycolmap
    rec = pycolmap.Reconstruction(str(sfm_dir))
    sfm_centres = {img.name: np.asarray(img.projection_center())
                   for img in rec.images.values()}

    gt = parse_nvm(scene / "reconstruction.nvm")
    train_names = parse_split_file(scene / "dataset_train.txt")

    # Sim(3): SfM -> NVM, using only successfully reconstructed train images.
    src, dst = [], []
    for n in train_names:
        if n in sfm_centres and n in gt:
            src.append(sfm_centres[n])
            dst.append(gt[n].Cw)
    src = np.asarray(src)
    dst = np.asarray(dst)
    if len(src) < 4:
        print("[score] not enough train alignment points", file=sys.stderr)
        return 1
    if args.robust_sim3:
        s, R, t, mask = umeyama_robust(src, dst)
        n_in = int(mask.sum())
        print(f"[score] robust Sim(3) on {n_in}/{len(src)} train imgs "
              f"(IRLS dropped {len(src) - n_in} alignment outliers): scale={s:.4f}")
    else:
        s, R, t = umeyama(src, dst)
        print(f"[score] LS Sim(3) on {len(src)} train imgs: scale={s:.4f}")
    scene_extent = float(np.linalg.norm(dst.max(0) - dst.min(0)))
    print(f"[score] scene extent={scene_extent:.2f}")

    rot_errs: List[float] = []
    trans_errs: List[float] = []
    n_succ = 0
    n_total = 0
    misses: List[str] = []

    for jf in sorted(results_dir.glob("*.json")):
        n_total += 1
        flat_name = jf.stem + ".png"
        if flat_name not in gt:
            misses.append(f"no GT: {flat_name}")
            continue
        gt_rec = gt[flat_name]
        data = json.loads(jf.read_text())
        if not data.get("success"):
            misses.append(f"failed: {flat_name}")
            continue
        n_succ += 1

        # Localizer pose is camera-from-world in SfM frame.
        R_cw = np.asarray(data["pose"]["R"])
        t_cw = np.asarray(data["pose"]["t"])
        C_sfm = -R_cw.T @ t_cw

        # Lift to NVM frame
        C_nvm_est = s * R @ C_sfm + t
        R_wc_sfm = R_cw.T
        R_wc_nvm = R @ R_wc_sfm

        # NVM stores world-to-camera, so transpose to get camera-to-world.
        R_cw_gt = quat_wxyz_to_rotmat(gt_rec.qcw)
        R_wc_gt = R_cw_gt.T
        C_gt = gt_rec.Cw

        cos = np.clip((np.trace(R_wc_nvm @ R_wc_gt.T) - 1) / 2, -1.0, 1.0)
        rot_errs.append(float(np.rad2deg(np.arccos(cos))))
        trans_errs.append(float(np.linalg.norm(C_nvm_est - C_gt)))

    print(f"[score] localized {n_succ}/{n_total} test images")
    if rot_errs:
        rot = np.asarray(rot_errs)
        tr = np.asarray(trans_errs)
        print(f"  rot err  median={np.median(rot):.3f}  mean={rot.mean():.3f}  "
              f"max={rot.max():.3f} deg")
        print(f"  trans    median={np.median(tr):.3f}  mean={tr.mean():.3f}  "
              f"max={tr.max():.3f} m")
        print(f"  trans/scene  median={np.median(tr)/scene_extent*100:.3f}%  "
              f"max={tr.max()/scene_extent*100:.3f}% (scene={scene_extent:.2f})")
    if misses:
        print(f"[score] {len(misses)} miss(es): {misses[:5]} ...")
    out = work / "score.json"
    out.write_text(json.dumps({
        "n_total": n_total, "n_success": n_succ,
        "scene_extent": scene_extent,
        "sim3_scale": s,
        "rot_err_median": float(np.median(rot_errs)) if rot_errs else None,
        "rot_err_max": float(max(rot_errs)) if rot_errs else None,
        "trans_err_median": float(np.median(trans_errs)) if trans_errs else None,
        "trans_err_max": float(max(trans_errs)) if trans_errs else None,
    }, indent=2))
    print(f"[score] wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--scene", required=True)
    p.add_argument("--work", required=True)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("localize-all")
    p.add_argument("--scene", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_localize_all)

    p = sub.add_parser("score")
    p.add_argument("--scene", required=True)
    p.add_argument("--work", required=True)
    p.add_argument(
        "--robust-sim3",
        dest="robust_sim3", action="store_true", default=True,
        help="Iteratively drop train alignment outliers (default).",
    )
    p.add_argument(
        "--no-robust-sim3",
        dest="robust_sim3", action="store_false",
        help="Use plain least-squares Sim(3) (legacy behaviour).",
    )
    p.set_defaults(func=cmd_score)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
