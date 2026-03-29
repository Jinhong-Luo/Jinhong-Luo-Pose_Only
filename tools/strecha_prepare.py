#!/usr/bin/env python3
import os
import glob
import argparse
from pathlib import Path
import numpy as np


def list_images(scene_dir):
    candidates = [
        os.path.join(scene_dir, "images"),
        os.path.join(scene_dir, "urd"),
        scene_dir,
    ]
    exts = ("*.png", "*.jpg", "*.jpeg", "*.ppm", "*.bmp", "*.tif", "*.tiff")
    files = []
    for d in candidates:
        if not os.path.isdir(d):
            continue
        for e in exts:
            files += glob.glob(os.path.join(d, e))
    files = sorted(set(files))
    if not files:
        raise RuntimeError(f"No images found under {scene_dir}")
    return files


def list_camera_files(scene_dir):
    candidates = [
        os.path.join(scene_dir, "gt_dense_cameras"),
        os.path.join(scene_dir, "cameras"),
        scene_dir,
    ]
    cams = []
    for d in candidates:
        if os.path.isdir(d):
            cams += glob.glob(os.path.join(d, "*.camera"))
    cams = sorted(set(cams))
    if not cams:
        raise RuntimeError(f"No .camera files found under {scene_dir}")
    return cams


def canonical_stem(path_or_name: str) -> str:
    """
    Normalize image / camera filename stem so that:
      0000.png         -> 0000
      0000.jpg         -> 0000
      0000.png.camera  -> 0000
      0000.jpg.camera  -> 0000
      0000.camera      -> 0000
    """
    name = Path(path_or_name).name

    while name.endswith(".camera"):
        name = name[:-len(".camera")]

    for ext in [".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".tif", ".tiff"]:
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break

    return name


def parse_camera_file(path):
    nums = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            vals = [float(x) for x in ln.split()]
            nums.append(vals)

    rows3 = [r for r in nums if len(r) == 3]
    if len(rows3) < 8:
        raise RuntimeError(f"Unexpected .camera format: {path}")

    K = np.array(rows3[0:3], dtype=np.float64)

    # Strecha .camera files contain an extra "0 0 0" row between K and R.
    # The actual layout is:
    #   K(3 rows), dummy row, R(3 rows), C(1 row), image_size(1 row)
    # Older parsing accidentally treated the dummy row as part of R.
    start_R = 3
    if np.allclose(rows3[3], 0.0, atol=1e-12):
        start_R = 4

    if len(rows3) < start_R + 4:
        raise RuntimeError(f"Unexpected .camera format after K block: {path}")

    R = np.array(rows3[start_R:start_R + 3], dtype=np.float64)
    C = np.array(rows3[start_R + 3], dtype=np.float64).reshape(3)

    if not np.allclose(R @ R.T, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(R), 1.0, atol=1e-4):
        raise RuntimeError(f"Parsed non-rotation matrix from camera file: {path}")
    return K, R, C


def write_pose_txt_c2w(path, R_w2c, C_w):
    with open(path, "w") as f:
        for R, C in zip(R_w2c, C_w):
            P = np.hstack([R.T, C.reshape(3, 1)])
            f.write(" ".join(f"{v:.9e}" for v in P.reshape(-1)) + "\n")


def write_pose_txt_w2c(path, R_w2c, C_w):
    with open(path, "w") as f:
        for R, C in zip(R_w2c, C_w):
            t = -R @ C.reshape(3, 1)
            P = np.hstack([R, t])
            f.write(" ".join(f"{v:.9e}" for v in P.reshape(-1)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    imgs = list_images(args.scene_dir)
    cams = list_camera_files(args.scene_dir)

    cam_map = {canonical_stem(p): p for p in cams}

    img_paths = []
    R_list = []
    C_list = []
    K_list = []

    missed = []
    for img in imgs:
        stem = canonical_stem(img)
        if stem not in cam_map:
            missed.append(Path(img).name)
            continue
        K, R, C = parse_camera_file(cam_map[stem])
        img_paths.append(img)
        K_list.append(K)
        R_list.append(R)
        C_list.append(C)

    if len(img_paths) == 0:
        print("[DEBUG] first 10 images:", [Path(p).name for p in imgs[:10]])
        print("[DEBUG] first 10 cameras:", [Path(p).name for p in cams[:10]])
        raise RuntimeError("No matched image-camera pairs found.")

    if missed:
        print(f"[WARN] unmatched images: {len(missed)} / {len(imgs)}")
        print("[WARN] first 10 unmatched:", missed[:10])

    K0 = K_list[0]
    for Ki in K_list[1:]:
        if not np.allclose(Ki, K0, atol=1e-8):
            raise RuntimeError("Found varying K across cameras; current pipeline assumes fixed K.")

    # Strecha .camera files store camera orientation in a convention that aligns
    # with c2w when compared against OpenCV recoverPose outputs. Save both forms
    # explicitly to avoid downstream convention confusion.
    R_abs_c2w = np.stack(R_list, axis=0).astype(np.float64)
    R_abs_w2c = np.transpose(R_abs_c2w, (0, 2, 1))
    C_all = np.stack(C_list, axis=0).astype(np.float64)   # camera center in world

    np.save(os.path.join(args.out_dir, "K.npy"), K0)
    np.save(os.path.join(args.out_dir, "R_abs_gt_c2w.npy"), R_abs_c2w)
    np.save(os.path.join(args.out_dir, "R_abs_gt_w2c.npy"), R_abs_w2c)
    np.save(os.path.join(args.out_dir, "gt_centers.npy"), C_all)

    write_pose_txt_c2w(os.path.join(args.out_dir, "gt_poses_c2w.txt"), R_abs_w2c, C_all)
    write_pose_txt_w2c(os.path.join(args.out_dir, "gt_poses_w2c.txt"), R_abs_w2c, C_all)

    with open(os.path.join(args.out_dir, "image_list.txt"), "w") as f:
        for p in img_paths:
            f.write(p + "\n")

    print("[OK] prepared scene")
    print("scene_dir:", args.scene_dir)
    print("out_dir  :", args.out_dir)
    print("frames   :", len(img_paths))


if __name__ == "__main__":
    main()

# python tools/strecha_prepare.py \
#   --scene_dir data/raw/strecha/fountain-P11 \
#   --out_dir data/prepared/strecha/fountain-P11
