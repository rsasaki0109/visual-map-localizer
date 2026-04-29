"""End-to-end evaluation against the COLMAP `south-building` dataset.

Steps performed by this script:

1. Reads the reference SfM bundled with the dataset (sparse/cameras.txt etc.).
2. Reads the local SfM produced by `visual-map-localizer build-map`.
3. Solves Sim(3) alignment between the two reconstructions on shared images.
4. Reads localized query poses (JSON files written by `... localize --output`).
5. Lifts each query pose into the reference frame and reports rotation /
   translation errors against the reference.

Usage (assumes layout from README's "公開データ検証" section):

    python scripts/evaluate_south_building.py \\
        --dataset    /path/to/south-building \\
        --map-dir    /path/to/built_map \\
        --results-dir /path/to/localize_jsons
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------- IO
def read_reference_poses(images_txt: Path) -> dict:
    """Parse a COLMAP ``images.txt`` -> {name: (R, t, C)} (cam<-world)."""
    out: dict = {}
    with open(images_txt) as f:
        lines = [ln.rstrip() for ln in f if not ln.startswith('#') and ln.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        R = qvec_to_rotmat(np.array([qw, qx, qy, qz]))
        t = np.array([tx, ty, tz])
        out[name] = (R, t, -R.T @ t)
        i += 2  # skip the POINTS2D line
    return out


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def read_local_poses(sfm_dir: Path) -> dict:
    import pycolmap
    rec = pycolmap.Reconstruction(str(sfm_dir))
    out: dict = {}
    for img in rec.images.values():
        rig = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        R = np.asarray(rig.rotation.matrix(), dtype=np.float64)
        t = np.asarray(rig.translation, dtype=np.float64).reshape(3)
        out[img.name] = (R, t, -R.T @ t)
    return out


# ----------------------------------------------------- Sim(3) (Umeyama)
def umeyama(src: np.ndarray, dst: np.ndarray, with_scaling: bool = True):
    """Estimate s, R, t such that ``dst ~= s * R @ src + t`` (least squares)."""
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((src_c ** 2).sum() / n) if with_scaling else 1.0
    t = mu_dst - s * R @ mu_src
    return s, R, t


# --------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',     required=True, type=Path,
                   help='south-building/ directory (containing sparse/, images/)')
    p.add_argument('--map-dir',     required=True, type=Path,
                   help='build-map output directory')
    p.add_argument('--results-dir', required=True, type=Path,
                   help='Directory of localize JSON outputs (one per query)')
    args = p.parse_args()

    ref = read_reference_poses(args.dataset / 'sparse' / 'images.txt')
    loc = read_local_poses(args.map_dir / 'sfm')
    print(f"reference poses: {len(ref)}")
    print(f"local SfM poses: {len(loc)}")

    shared_names = sorted(set(ref) & set(loc))
    Cs_loc = np.stack([loc[n][2] for n in shared_names])
    Cs_ref = np.stack([ref[n][2] for n in shared_names])
    s, Rsim, tsim = umeyama(Cs_loc, Cs_ref)
    Cs_loc_in_ref = (s * Rsim @ Cs_loc.T).T + tsim
    align_residual = np.linalg.norm(Cs_loc_in_ref - Cs_ref, axis=1)
    scene_extent = float(np.linalg.norm(Cs_ref.max(0) - Cs_ref.min(0)))
    print(f"shared (alignment): {len(shared_names)}")
    print(f"alignment scale s = {s:.4f}")
    print(f"alignment residual (camera centers): "
          f"mean={align_residual.mean():.4f}, max={align_residual.max():.4f}")
    print(f"scene extent (reference frame): {scene_extent:.2f}")

    print(f"\n{'query':<22} {'inliers':>8} {'rot_err_deg':>12} "
          f"{'trans_err':>10} {'trans_pct':>10}")
    print('-' * 70)
    rot_errs, trans_errs = [], []
    for jpath in sorted(args.results_dir.glob('*.json')):
        with open(jpath) as f:
            d = json.load(f)
        if not d['success']:
            print(f"{jpath.stem:<22} (FAIL: {d.get('error','')})")
            continue
        R_loc = np.asarray(d['pose']['R'])
        t_loc = np.asarray(d['pose']['t'])
        C_loc = -R_loc.T @ t_loc
        C_ref_est = s * Rsim @ C_loc + tsim
        R_ref_est = R_loc @ Rsim.T

        # Match this estimate against the reference using the file name.
        # The dataset uses uppercase .JPG; tolerate both.
        for cand in (jpath.stem + '.JPG', jpath.stem + '.jpg', jpath.stem):
            if cand in ref:
                R_gt, _, C_gt = ref[cand]
                break
        else:
            print(f"{jpath.stem:<22} (no GT)")
            continue
        cos = np.clip((np.trace(R_ref_est @ R_gt.T) - 1) / 2, -1.0, 1.0)
        rot_err_deg = float(np.rad2deg(np.arccos(cos)))
        trans_err = float(np.linalg.norm(C_ref_est - C_gt))
        rot_errs.append(rot_err_deg)
        trans_errs.append(trans_err)
        print(f"{jpath.stem:<22} {d['inliers']:>8d} "
              f"{rot_err_deg:>12.3f} {trans_err:>10.4f} "
              f"{trans_err/scene_extent*100:>9.3f}%")

    if rot_errs:
        print(f"\nrot error (deg)  median={np.median(rot_errs):.3f}, "
              f"mean={np.mean(rot_errs):.3f}, max={np.max(rot_errs):.3f}")
        print(f"trans error      median={np.median(trans_errs):.4f}, "
              f"mean={np.mean(trans_errs):.4f}, max={np.max(trans_errs):.4f}")
        print(f"trans / scene    median="
              f"{np.median(trans_errs)/scene_extent*100:.3f}%, "
              f"max={np.max(trans_errs)/scene_extent*100:.3f}%")


if __name__ == '__main__':
    main()
