#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def parse_projection_matrix(path: str) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split()]
            if vals:
                rows.append(vals)
    P = np.asarray(rows, dtype=np.float64)
    if P.shape != (3, 4):
        raise RuntimeError(f"Expected 3x4 projection matrix in {path}, got {P.shape}")
    return P


def decompose_projection_matrix(P: np.ndarray):
    K, R, t_h, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    K = K / max(float(K[2, 2]), 1e-12)
    C = (t_h[:3] / t_h[3]).reshape(3)
    R = R.astype(np.float64)
    C = C.astype(np.float64)
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
    return K.astype(np.float64), R, C


def write_pose_txt_c2w(path: str, R_w2c: np.ndarray, C_w: np.ndarray):
    with open(path, "w", encoding="utf-8") as f:
        for R, C in zip(R_w2c, C_w):
            P = np.hstack([R.T, C.reshape(3, 1)])
            f.write(" ".join(f"{v:.9e}" for v in P.reshape(-1)) + "\n")


def write_pose_txt_w2c(path: str, R_w2c: np.ndarray, C_w: np.ndarray):
    with open(path, "w", encoding="utf-8") as f:
        for R, C in zip(R_w2c, C_w):
            t = -R @ C.reshape(3, 1)
            P = np.hstack([R, t])
            f.write(" ".join(f"{v:.9e}" for v in P.reshape(-1)) + "\n")


def list_scene_images(scene_dir: str, light_id: int, use_rectified: bool):
    prefix = "rect" if use_rectified else "clean"
    pattern = f"{prefix}_*_{light_id}_r5000.png"
    files = sorted(Path(scene_dir).glob(pattern))
    if not files:
        raise RuntimeError(f"No DTU images matched pattern {pattern} under {scene_dir}")
    return files


def extract_view_index(path: Path) -> int:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise RuntimeError(f"Unexpected DTU image filename: {path.name}")
    return int(parts[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_id", required=True, help="e.g. scan1, scan6")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--raw_root",
        default="data/raw/DTU/SampleSet/MVS Data",
        help="DTU MVS Data root containing Calibration / Cleaned / Rectified",
    )
    ap.add_argument(
        "--light_id",
        type=int,
        default=3,
        help="DTU lighting index. 3 is the diffuse setting commonly used in papers.",
    )
    ap.add_argument(
        "--use_rectified",
        action="store_true",
        help="Use Rectified images. Recommended because downstream assumes undistorted intrinsics.",
    )
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    calib_candidates = [
        raw_root / "Calibration" / "cal18",
        raw_root / "SampleSet" / "MVS Data" / "Calibration" / "cal18",
        raw_root.parent / "SampleSet" / "MVS Data" / "Calibration" / "cal18",
        raw_root.parent / "MVS Data" / "Calibration" / "cal18",
    ]
    calib_dir = next((p for p in calib_candidates if p.is_dir()), None)
    if calib_dir is None:
        raise RuntimeError(f"Missing DTU calibration dir. Tried: {calib_candidates}")

    scene_kind = "Rectified" if args.use_rectified else "Cleaned"
    scene_candidates = [
        raw_root / scene_kind / args.scan_id,
        raw_root.parent / scene_kind / args.scan_id,
        raw_root.parent / "SampleSet" / "MVS Data" / scene_kind / args.scan_id,
    ]
    scene_dir = next((p for p in scene_candidates if p.is_dir()), None)
    if scene_dir is None:
        raise RuntimeError(f"Missing DTU scene dir. Tried: {scene_candidates}")

    img_paths = list_scene_images(str(scene_dir), args.light_id, args.use_rectified)

    K_list = []
    R_list = []
    C_list = []
    ordered_img_paths = []
    used_view_ids = []
    for img_path in img_paths:
        view_idx = extract_view_index(img_path)
        pos_path = calib_dir / f"pos_{view_idx:03d}.txt"
        if not pos_path.is_file():
            raise RuntimeError(f"Missing DTU projection file for {img_path.name}: {pos_path}")
        P = parse_projection_matrix(str(pos_path))
        K, R_w2c, C_w = decompose_projection_matrix(P)
        K_list.append(K)
        R_list.append(R_w2c)
        C_list.append(C_w)
        ordered_img_paths.append(str(img_path.resolve()))
        used_view_ids.append(view_idx)

    Ks = np.stack(K_list, axis=0).astype(np.float64)
    R_abs_w2c = np.stack(R_list, axis=0).astype(np.float64)
    R_abs_c2w = np.transpose(R_abs_w2c, (0, 2, 1)).astype(np.float64)
    C_all = np.stack(C_list, axis=0).astype(np.float64)

    os.makedirs(args.out_dir, exist_ok=True)
    if np.allclose(Ks, Ks[0:1], atol=1e-8):
        np.save(os.path.join(args.out_dir, "K.npy"), Ks[0])
    np.save(os.path.join(args.out_dir, "Ks.npy"), Ks)
    np.save(os.path.join(args.out_dir, "image_K_idx.npy"), np.arange(Ks.shape[0], dtype=np.int64))
    np.save(os.path.join(args.out_dir, "dtu_view_ids.npy"), np.asarray(used_view_ids, dtype=np.int64))
    np.save(os.path.join(args.out_dir, "R_abs_gt_w2c.npy"), R_abs_w2c)
    np.save(os.path.join(args.out_dir, "R_abs_gt_c2w.npy"), R_abs_c2w)
    np.save(os.path.join(args.out_dir, "gt_centers.npy"), C_all)

    write_pose_txt_w2c(os.path.join(args.out_dir, "gt_poses_w2c.txt"), R_abs_w2c, C_all)
    write_pose_txt_c2w(os.path.join(args.out_dir, "gt_poses_c2w.txt"), R_abs_w2c, C_all)

    with open(os.path.join(args.out_dir, "image_list.txt"), "w", encoding="utf-8") as f:
        for p in ordered_img_paths:
            f.write(p + "\n")

    print("[OK] prepared DTU scene")
    print("scan_id  :", args.scan_id)
    print("out_dir  :", args.out_dir)
    print("frames   :", len(ordered_img_paths))
    print("light_id :", args.light_id)
    print("variant  :", "Rectified" if args.use_rectified else "Cleaned")
    print("scene_dir :", scene_dir)
    print("calib_dir :", calib_dir)


if __name__ == "__main__":
    main()
