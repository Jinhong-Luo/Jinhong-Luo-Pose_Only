#!/usr/bin/env python3
import argparse
import os

import numpy as np

import _bootstrap  # noqa: F401
from calib_utils import list_images_sorted


def parse_cameras_txt(path):
    blocks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(x) for x in parts[4:]], dtype=np.float64)
            blocks.append({
                "camera_id": cam_id,
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            })
    return blocks


def camera_block_to_K(block):
    if block["model"] != "PINHOLE":
        raise RuntimeError(f"Only ETH3D PINHOLE cameras are supported, got: {block['model']}")
    if block["params"].shape[0] != 4:
        raise RuntimeError(f"Expected 4 PINHOLE params, got {block['params'].shape[0]}")

    fx, fy, cx, cy = block["params"].tolist()
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def quat_to_rotmat(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    nq = np.linalg.norm(q)
    if nq <= 0:
        raise RuntimeError("Invalid quaternion with zero norm.")
    q /= nq
    qw, qx, qy, qz = q.tolist()
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def parse_images_txt(path):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 10:
            raise RuntimeError(f"Unexpected ETH3D images.txt line: {line}")

        image_id = int(parts[0])
        qw, qx, qy, qz = [float(x) for x in parts[1:5]]
        tx, ty, tz = [float(x) for x in parts[5:8]]
        camera_id = int(parts[8])
        name = parts[9]
        basename = os.path.basename(name)
        records[basename] = {
            "image_id": image_id,
            "camera_id": camera_id,
            "name": name,
            "R_w2c": quat_to_rotmat(qw, qx, qy, qz),
            "t_w2c": np.array([tx, ty, tz], dtype=np.float64),
        }

        if i < len(lines):
            i += 1
    return records


def write_pose_txt_c2w(path, R_w2c, C_w):
    with open(path, "w", encoding="utf-8") as f:
        for R, C in zip(R_w2c, C_w):
            P = np.hstack([R.T, C.reshape(3, 1)])
            f.write(" ".join(f"{v:.9e}" for v in P.reshape(-1)) + "\n")


def write_pose_txt_w2c(path, R_w2c, C_w):
    with open(path, "w", encoding="utf-8") as f:
        for R, C in zip(R_w2c, C_w):
            t = -R @ C.reshape(3, 1)
            P = np.hstack([R, t])
            f.write(" ".join(f"{v:.9e}" for v in P.reshape(-1)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    scene_dir = args.scene_dir
    out_dir = args.out_dir
    calib_dir = os.path.join(scene_dir, "dslr_calibration_undistorted")
    cameras_txt = os.path.join(calib_dir, "cameras.txt")
    images_txt = os.path.join(calib_dir, "images.txt")
    img_dir = os.path.join(scene_dir, "images", "dslr_images_undistorted")

    if not os.path.isfile(cameras_txt):
        raise RuntimeError(f"Missing ETH3D cameras.txt: {cameras_txt}")
    if not os.path.isfile(images_txt):
        raise RuntimeError(f"Missing ETH3D images.txt: {images_txt}")
    if not os.path.isdir(img_dir):
        raise RuntimeError(f"ETH3D image directory does not exist: {img_dir}")

    img_paths = list_images_sorted(img_dir)

    camera_blocks = parse_cameras_txt(cameras_txt)
    if len(camera_blocks) == 0:
        raise RuntimeError(f"No valid camera blocks found in {cameras_txt}")
    cameras_by_id = {}
    camera_ids = []
    Ks = []
    for block in camera_blocks:
        cam_id = int(block["camera_id"])
        if cam_id in cameras_by_id:
            raise RuntimeError(f"Duplicate camera_id={cam_id} in {cameras_txt}")
        cameras_by_id[cam_id] = block
        camera_ids.append(cam_id)
        Ks.append(camera_block_to_K(block))
    camera_ids = np.asarray(camera_ids, dtype=np.int64)
    Ks = np.stack(Ks, axis=0).astype(np.float64)

    image_records = parse_images_txt(images_txt)
    if not image_records:
        raise RuntimeError(f"No valid image poses found in {images_txt}")

    sorted_names = [os.path.basename(p) for p in img_paths]
    missing = [name for name in sorted_names if name not in image_records]
    if missing:
        raise RuntimeError(
            f"Could not match {len(missing)} images from {img_dir} to poses in {images_txt}. "
            f"First missing names: {missing[:10]}"
        )

    extra = [name for name in image_records.keys() if name not in set(sorted_names)]
    if extra:
        print(f"[WARN] images.txt contains {len(extra)} entries without local files. First 10: {extra[:10]}")

    R_abs_w2c = []
    R_abs_c2w = []
    C_all = []
    ordered_img_paths = []
    image_camera_ids = []
    image_K_idx = []
    for img_path in img_paths:
        basename = os.path.basename(img_path)
        rec = image_records[basename]
        cam_id = int(rec["camera_id"])
        if cam_id not in cameras_by_id:
            raise RuntimeError(f"Image {basename} references unknown camera_id={cam_id}")
        R_w2c = rec["R_w2c"]
        t_w2c = rec["t_w2c"]
        C = -R_w2c.T @ t_w2c
        R_abs_w2c.append(R_w2c)
        R_abs_c2w.append(R_w2c.T)
        C_all.append(C)
        ordered_img_paths.append(img_path)
        image_camera_ids.append(cam_id)
        image_K_idx.append(int(np.where(camera_ids == cam_id)[0][0]))

    R_abs_w2c = np.stack(R_abs_w2c, axis=0).astype(np.float64)
    R_abs_c2w = np.stack(R_abs_c2w, axis=0).astype(np.float64)
    C_all = np.stack(C_all, axis=0).astype(np.float64)

    os.makedirs(out_dir, exist_ok=True)
    if Ks.shape[0] == 1:
        np.save(os.path.join(out_dir, "K.npy"), Ks[0])
    np.save(os.path.join(out_dir, "Ks.npy"), Ks)
    np.save(os.path.join(out_dir, "camera_ids.npy"), camera_ids)
    np.save(os.path.join(out_dir, "image_camera_ids.npy"), np.asarray(image_camera_ids, dtype=np.int64))
    np.save(os.path.join(out_dir, "image_K_idx.npy"), np.asarray(image_K_idx, dtype=np.int64))
    np.save(os.path.join(out_dir, "R_abs_gt_w2c.npy"), R_abs_w2c)
    np.save(os.path.join(out_dir, "R_abs_gt_c2w.npy"), R_abs_c2w)
    np.save(os.path.join(out_dir, "gt_centers.npy"), C_all)

    write_pose_txt_w2c(os.path.join(out_dir, "gt_poses_w2c.txt"), R_abs_w2c, C_all)
    write_pose_txt_c2w(os.path.join(out_dir, "gt_poses_c2w.txt"), R_abs_w2c, C_all)

    with open(os.path.join(out_dir, "image_list.txt"), "w", encoding="utf-8") as f:
        for path in ordered_img_paths:
            f.write(path + "\n")

    print("[OK] prepared ETH3D scene")
    print("scene_dir:", scene_dir)
    print("out_dir  :", out_dir)
    print("frames   :", len(ordered_img_paths))
    print("cameras  :", Ks.shape[0])


if __name__ == "__main__":
    main()
