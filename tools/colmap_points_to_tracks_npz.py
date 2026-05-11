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


def read_images_binary_with_points(path: str):
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
            xys = np.empty((num_points2d, 2), dtype=np.float64)
            point3D_ids = np.empty((num_points2d,), dtype=np.int64)
            for i in range(num_points2d):
                x, y, p3d = read_next_bytes(fid, 24, "ddq")
                xys[i, 0] = x
                xys[i, 1] = y
                point3D_ids[i] = p3d

            images[image_id] = {
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
                "xys": xys,
                "point3D_ids": point3D_ids,
            }
    return images


def _normalize_name(text: str) -> str:
    return str(text).replace("\\", "/").strip()


def read_image_order(image_list_path: str | None, images: dict):
    if image_list_path and os.path.isfile(image_list_path):
        ordered = []
        with open(image_list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ordered.append(_normalize_name(line))
        return ordered
    return sorted(_normalize_name(item["name"]) for item in images.values())


def resolve_ordered_images(ordered_names: list[str], images: dict):
    by_exact = {_normalize_name(item["name"]): item for item in images.values()}
    by_basename = {os.path.basename(_normalize_name(item["name"])): item for item in images.values()}
    resolved = []
    for name in ordered_names:
        key = _normalize_name(name)
        item = by_exact.get(key)
        if item is None:
            item = by_basename.get(os.path.basename(key))
        resolved.append(item)
    return resolved


def export_tracks_from_colmap_images(
    images: dict,
    ordered_names: list[str],
    out_dir: str,
    min_track_len: int = 2,
):
    resolved_images = resolve_ordered_images(ordered_names, images)

    track_len = {}
    for item in images.values():
        for p3d in item["point3D_ids"]:
            p3d = int(p3d)
            if p3d < 0:
                continue
            track_len[p3d] = track_len.get(p3d, 0) + 1

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    used_frames = 0
    total_obs = 0
    total_tracks = 0
    frame_counts = []

    for fidx, item in enumerate(resolved_images):
        if item is None:
            track_ids = np.zeros((0,), dtype=np.int64)
            xy = np.zeros((0, 2), dtype=np.float32)
        else:
            point_ids = item["point3D_ids"]
            mask = np.array(
                [(int(p) >= 0 and track_len.get(int(p), 0) >= min_track_len) for p in point_ids],
                dtype=bool,
            )
            track_ids = point_ids[mask].astype(np.int64, copy=False)
            xy = item["xys"][mask].astype(np.float32, copy=False)
            if track_ids.size > 0:
                order = np.argsort(track_ids)
                track_ids = track_ids[order]
                xy = xy[order]
                used_frames += 1
                total_obs += int(track_ids.size)
                total_tracks += len(set(track_ids.tolist()))
        np.savez_compressed(out_dir_path / f"{fidx:06d}.npz", track_ids=track_ids, xy=xy)
        frame_counts.append(int(track_ids.size))

    stats = {
        "ordered_frames": len(ordered_names),
        "registered_images_in_colmap": len(images),
        "used_frames_nonempty": used_frames,
        "min_track_len": int(min_track_len),
        "unique_colmap_tracks_kept": len([k for k, v in track_len.items() if v >= min_track_len]),
        "frame_obs_min": int(min(frame_counts)) if frame_counts else 0,
        "frame_obs_median": float(np.median(frame_counts)) if frame_counts else 0.0,
        "frame_obs_p90": float(np.quantile(frame_counts, 0.9)) if frame_counts else 0.0,
        "frame_obs_max": int(max(frame_counts)) if frame_counts else 0,
        "total_obs_written": total_obs,
        "avg_unique_track_refs_per_nonempty_frame": (total_tracks / used_frames) if used_frames > 0 else 0.0,
    }
    return stats


def main():
    ap = argparse.ArgumentParser(description="Export COLMAP sparse observations to per-frame tracks/*.npz.")
    ap.add_argument("--images_bin", required=True, help="COLMAP sparse/images.bin")
    ap.add_argument("--image_list", default=None, help="Optional image_list.txt to enforce frame order")
    ap.add_argument("--out_dir", required=True, help="Directory for per-frame npz files")
    ap.add_argument("--min_track_len", type=int, default=2, help="Only keep COLMAP 3D points observed in at least this many images")
    ap.add_argument("--stats_json", default=None, help="Optional summary json")
    args = ap.parse_args()

    images = read_images_binary_with_points(args.images_bin)
    ordered_names = read_image_order(args.image_list, images)
    stats = export_tracks_from_colmap_images(
        images=images,
        ordered_names=ordered_names,
        out_dir=args.out_dir,
        min_track_len=args.min_track_len,
    )
    out_dir = Path(args.out_dir)

    if args.stats_json:
        import json

        payload = {
            "images_bin": str(Path(args.images_bin).resolve()),
            "image_list": str(Path(args.image_list).resolve()) if args.image_list else None,
            "out_dir": str(out_dir.resolve()),
        }
        payload.update(stats)
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("saved:", stats_path)

    print("saved tracks to:", out_dir)
    print("registered images:", stats["registered_images_in_colmap"])
    print("ordered frames:", stats["ordered_frames"])
    print("min_track_len:", stats["min_track_len"])
    print("frame obs median:", stats["frame_obs_median"])


if __name__ == "__main__":
    main()
