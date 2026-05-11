#!/usr/bin/env python3
import argparse
import os
import struct
from pathlib import Path

import numpy as np


def read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected EOF while reading COLMAP binary model.")
    return struct.unpack(endian + fmt, data)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def read_images_binary(path: str):
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = props[8]

            name_bytes = bytearray()
            while True:
                ch = fid.read(1)
                if ch == b"\x00":
                    break
                if ch == b"":
                    raise EOFError("Unexpected EOF while reading image name.")
                name_bytes.extend(ch)
            name = name_bytes.decode("utf-8")

            num_points2d = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2d * 24, 1)

            images[image_id] = {
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }
    return images


def read_image_order(image_list_path: str | None, images: dict) -> list[str]:
    if image_list_path and os.path.isfile(image_list_path):
        ordered = []
        with open(image_list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ordered.append(os.path.basename(line.replace("\\", "/")))
        return ordered
    return sorted((item["name"] for item in images.values()))


def build_pose_row_c2w(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    r_w2c = qvec_to_rotmat(qvec)
    c = -r_w2c.T @ tvec
    t_c2w = c
    return np.hstack([r_w2c.T, t_c2w.reshape(3, 1)])


def main():
    ap = argparse.ArgumentParser(description="Export COLMAP images.bin poses to project-style poses_c2w.txt.")
    ap.add_argument("--images_bin", required=True, help="COLMAP sparse/images.bin")
    ap.add_argument("--image_list", default=None, help="Optional project image_list.txt to align row order")
    ap.add_argument("--out_txt", required=True, help="Output 3x4-per-line poses_c2w.txt")
    ap.add_argument("--out_names", default=None, help="Optional output ordered image names txt")
    ap.add_argument("--out_rabs_w2c_npy", default=None, help="Optional output stacked w2c rotations, shape (N,3,3)")
    args = ap.parse_args()

    images = read_images_binary(args.images_bin)
    by_name = {item["name"]: item for item in images.values()}
    ordered_names = read_image_order(args.image_list, images)

    rows = []
    rabs_w2c = []
    used_names = []
    for name in ordered_names:
        if name not in by_name:
            continue
        item = by_name[name]
        pose = build_pose_row_c2w(item["qvec"], item["tvec"])
        rows.append(pose.reshape(-1))
        rabs_w2c.append(qvec_to_rotmat(item["qvec"]))
        used_names.append(name)

    if not rows:
        raise RuntimeError("No matching images found between images.bin and image list/order.")

    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(" ".join(f"{x:.9e}" for x in row) + "\n")
    print("saved:", out_path)
    print("num_poses:", len(rows))

    if args.out_names:
        names_path = Path(args.out_names)
        names_path.parent.mkdir(parents=True, exist_ok=True)
        with open(names_path, "w", encoding="utf-8") as f:
            for name in used_names:
                f.write(name + "\n")
        print("saved:", names_path)

    if args.out_rabs_w2c_npy:
        rabs_path = Path(args.out_rabs_w2c_npy)
        rabs_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(rabs_path, np.stack(rabs_w2c, axis=0))
        print("saved:", rabs_path)


if __name__ == "__main__":
    main()
