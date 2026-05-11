#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def export_scan(scan_dir: Path, out_root: Path):
    scan_name = scan_dir.name
    out_dir = out_root / scan_name
    out_dir.mkdir(parents=True, exist_ok=True)

    Ks = np.load(scan_dir / "Ks.npy")
    image_K_idx = np.load(scan_dir / "image_K_idx.npy")
    view_ids = np.load(scan_dir / "dtu_view_ids.npy")
    R_w2c = np.load(scan_dir / "R_abs_gt_w2c.npy")
    R_c2w = np.load(scan_dir / "R_abs_gt_c2w.npy")
    centers = np.load(scan_dir / "gt_centers.npy")
    with open(scan_dir / "image_list.txt", "r", encoding="utf-8") as f:
        image_list = [line.strip() for line in f if line.strip()]

    np.save(out_dir / "Ks.npy", Ks)
    np.save(out_dir / "image_K_idx.npy", image_K_idx)
    np.save(out_dir / "dtu_view_ids.npy", view_ids)
    np.save(out_dir / "R_abs_gt_w2c.npy", R_w2c)
    np.save(out_dir / "R_abs_gt_c2w.npy", R_c2w)
    np.save(out_dir / "gt_centers.npy", centers)

    for name in ["gt_poses_c2w.txt", "gt_poses_w2c.txt", "image_list.txt"]:
        (out_dir / name).write_text((scan_dir / name).read_text(encoding="utf-8"), encoding="utf-8")

    with open(out_dir / "camera_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx", "view_id", "image_name", "image_K_idx",
            "center_x", "center_y", "center_z"
        ])
        for idx, (view_id, image_path, k_idx, C) in enumerate(zip(view_ids, image_list, image_K_idx, centers)):
            writer.writerow([
                idx, int(view_id), Path(image_path).name, int(k_idx),
                float(C[0]), float(C[1]), float(C[2]),
            ])

    k_summary = []
    for idx, K in enumerate(Ks):
        k_summary.append({
            "frame_idx": idx,
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        })
    with open(out_dir / "K_summary.json", "w", encoding="utf-8") as f:
        json.dump(k_summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / "metadata_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "scan": scan_name,
            "frame_count": int(len(image_list)),
            "view_id_min": int(np.min(view_ids)),
            "view_id_max": int(np.max(view_ids)),
            "image_list_example": [Path(p).name for p in image_list[:5]],
            "output_files": [
                "Ks.npy",
                "image_K_idx.npy",
                "dtu_view_ids.npy",
                "R_abs_gt_w2c.npy",
                "R_abs_gt_c2w.npy",
                "gt_centers.npy",
                "gt_poses_c2w.txt",
                "gt_poses_w2c.txt",
                "image_list.txt",
                "camera_summary.csv",
                "K_summary.json",
            ],
        }, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared_root", default="data/prepared/DTU")
    ap.add_argument("--out_root", default="data/raw/DTU/metadata")
    ap.add_argument("--scans", default=None, help="comma-separated scan names, e.g. scan24,scan37")
    args = ap.parse_args()

    prepared_root = Path(args.prepared_root)
    out_root = Path(args.out_root)
    if args.scans:
        scans = [prepared_root / s.strip() for s in args.scans.split(",") if s.strip()]
    else:
        scans = sorted([p for p in prepared_root.iterdir() if p.is_dir()])

    for scan_dir in scans:
        if not scan_dir.is_dir():
            raise RuntimeError(f"Missing prepared scan dir: {scan_dir}")
        export_scan(scan_dir, out_root)
        print(f"exported: {out_root / scan_dir.name}")


if __name__ == "__main__":
    main()
