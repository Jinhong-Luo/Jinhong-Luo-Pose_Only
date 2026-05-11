#!/usr/bin/env python3
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap

import _bootstrap  # noqa: F401
from calib_utils import (
    format_kitti_layout_help,
    load_intrinsics,
    read_custom_Ks,
    resolve_kitti_scene_inputs,
)
from colmap_points_to_tracks_npz import (
    export_tracks_from_colmap_images,
    read_image_order,
    read_images_binary_with_points,
    resolve_ordered_images,
)
from degeneracy_utils import dump_json, safe_ratio, summarize_array
from frontend_cache import (
    build_lightglue_frontend,
    load_or_extract_feature,
    read_gray_u8,
    resolve_device,
)
from quality_utils import (
    compute_qpair,
    infer_qpair_config,
    load_quality_config,
    make_quality_config_record,
    resolve_quality_section,
    save_quality_config_record,
)


def parse_deltas(text):
    vals = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(int(item))
    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        raise ValueError("Empty --deltas")
    return vals


def normalize_name(text: str) -> str:
    return str(text).replace("\\", "/").strip()


def read_image_paths(args):
    if args.dataset == "kitti":
        resolved = resolve_kitti_scene_inputs(
            kitti_calib=args.kitti_calib,
            image_glob=args.image_glob,
        )
        if resolved["image_glob"] != args.image_glob or resolved["kitti_calib"] != args.kitti_calib:
            print(f"[KITTI] auto-resolved scene {resolved['scene_id']} to:")
            print(f"  calib     : {resolved['kitti_calib']}")
            print(f"  image_glob: {resolved['image_glob']}")
            args.image_glob = resolved["image_glob"]
            args.kitti_calib = resolved["kitti_calib"]

    if args.image_list:
        paths = []
        with open(args.image_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    paths.append(os.path.abspath(line))
    else:
        if not args.image_glob:
            raise ValueError("Provide either --image_list or --image_glob")
        paths = sorted(glob.glob(args.image_glob))
        if not paths:
            if args.dataset == "kitti":
                scene_id = resolved.get("scene_id") if "resolved" in locals() else None
                raise FileNotFoundError(
                    f"No KITTI images matched: {args.image_glob}\n"
                    f"{format_kitti_layout_help(scene_id)}"
                )
            raise FileNotFoundError(args.image_glob)
        paths = [os.path.abspath(p) for p in paths]

    if args.max_frames is not None:
        paths = paths[: args.max_frames]
    return paths


def infer_image_root(image_paths, image_list_path=None):
    if image_list_path:
        raw_lines = []
        with open(image_list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_lines.append(line)
        if raw_lines and not os.path.isabs(raw_lines[0]):
            return os.path.abspath(".")
    return os.path.commonpath(image_paths)


def relative_image_names(image_paths, image_root):
    rel = []
    for p in image_paths:
        rel.append(normalize_name(os.path.relpath(p, image_root)))
    return rel


def build_camera_for_K(width, height, K):
    K = np.asarray(K, dtype=np.float64)
    return pycolmap.Camera(
        model="PINHOLE",
        width=int(width),
        height=int(height),
        params=[float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
    )


def write_database_schema(db_path, image_paths, image_names, Ks_per_frame):
    if os.path.exists(db_path):
        os.remove(db_path)
    db = pycolmap.Database.open(db_path)
    camera_cache = {}
    image_ids = []
    image_id_by_name = {}

    for img_path, img_name, K in zip(image_paths, image_names, Ks_per_frame):
        img_u8 = read_gray_u8(img_path)
        h, w = img_u8.shape[:2]
        key = tuple(np.asarray(K, dtype=np.float64).reshape(-1).round(12).tolist()) + (int(w), int(h))
        camera_id = camera_cache.get(key)
        if camera_id is None:
            camera = build_camera_for_K(w, h, K)
            camera_id = int(db.write_camera(camera))
            camera_cache[key] = camera_id
        image = pycolmap.Image(name=img_name, camera_id=camera_id)
        image_id = int(db.write_image(image))
        image_ids.append(image_id)
        image_id_by_name[img_name] = image_id

    db.close()
    return image_ids, image_id_by_name


def summarize_matching(
    *,
    image_paths,
    image_names,
    image_ids,
    args,
    Ks_per_frame,
    out_dir,
):
    import torch

    device = resolve_device(args.device)
    print("[Device]", device)
    extractor, matcher = build_lightglue_frontend(
        max_kpts=args.max_kpts,
        filter_th=args.filter_th,
        device=device,
    )

    feats_cache = []
    kpts_all = []
    for image_path in image_paths:
        feats, kpts = load_or_extract_feature(
            image_path,
            extractor=extractor,
            device=device,
            max_kpts=args.max_kpts,
            cache_dir=args.feature_cache_dir,
            image_loader=read_gray_u8,
        )
        feats_cache.append(feats)
        kpts_all.append(kpts.astype(np.float32))

    db_path = str(Path(out_dir) / "database.db")
    db = pycolmap.Database.open(db_path)
    for image_id, kpts in zip(image_ids, kpts_all):
        db.write_keypoints(int(image_id), np.asarray(kpts, dtype=np.float32))

    deltas = parse_deltas(args.deltas)
    pair_records = []
    pair_lines = []
    gap_attempts = {int(d): 0 for d in deltas}

    for i in range(len(image_paths)):
        feats0 = feats_cache[i]
        for delta in deltas:
            j = i + delta
            if j >= len(image_paths):
                continue
            gap_attempts[int(delta)] += 1
            feats1 = feats_cache[j]
            with torch.no_grad():
                out = matcher({"image0": feats0, "image1": feats1})

            m0 = out["matches0"][0].detach().cpu().numpy().astype(np.int64)
            valid = m0 >= 0
            nmatch_raw = int(valid.sum())

            if args.mutual and ("matches1" in out):
                m1 = out["matches1"][0].detach().cpu().numpy().astype(np.int64)
                idx0_all = np.arange(m0.shape[0], dtype=np.int64)
                ok = valid.copy()
                ok[valid] &= (m1[m0[valid]] == idx0_all[valid])
                valid &= ok

            scores = None
            if "matching_scores0" in out:
                scores = out["matching_scores0"][0].detach().cpu().numpy()
            if args.min_score > 0 and scores is not None:
                valid &= scores >= args.min_score

            idx0 = np.where(valid)[0].astype(np.uint32)
            idx1 = m0[idx0].astype(np.uint32)
            matches = np.stack([idx0, idx1], axis=1) if idx0.size > 0 else np.zeros((0, 2), dtype=np.uint32)
            db.write_matches(int(image_ids[i]), int(image_ids[j]), matches)
            pair_lines.append(f"{image_names[i]} {image_names[j]}")

            score_mean = np.nan
            score_med = np.nan
            if scores is not None and idx0.size > 0:
                pair_scores = scores[idx0]
                score_mean = float(np.mean(pair_scores))
                score_med = float(np.median(pair_scores))

            pair_records.append({
                "i": int(i),
                "j": int(j),
                "delta": int(delta),
                "image_id1": int(image_ids[i]),
                "image_id2": int(image_ids[j]),
                "image_name1": image_names[i],
                "image_name2": image_names[j],
                "nmatches": int(matches.shape[0]),
                "nmatch_raw": int(nmatch_raw),
                "score_mean": float(score_mean) if np.isfinite(score_mean) else np.nan,
                "score_med": float(score_med) if np.isfinite(score_med) else np.nan,
            })

    db.close()
    pairs_path = Path(out_dir) / "pairs.txt"
    pairs_path.write_text("\n".join(pair_lines) + ("\n" if pair_lines else ""), encoding="utf-8")
    return db_path, str(pairs_path), pair_records, gap_attempts


def run_pycolmap_reconstruction(db_path, pairs_path, image_root, out_dir):
    pycolmap.verify_matches(db_path, pairs_path)
    sparse_root = Path(out_dir) / "sparse"
    sparse_root.mkdir(parents=True, exist_ok=True)
    reconstructions = pycolmap.incremental_mapping(db_path, image_root, str(sparse_root))
    if not reconstructions:
        raise RuntimeError("pycolmap.incremental_mapping produced no reconstruction.")
    best_idx, best_rec = max(reconstructions.items(), key=lambda item: item[1].num_reg_images())
    best_dir = sparse_root / "0"
    best_rec.write(str(best_dir))
    return str(best_dir), int(best_idx), best_rec


def decode_pair_id(pair_id):
    pair_id = int(pair_id)
    max_image_id = 2147483647
    image_id2 = pair_id % max_image_id
    image_id1 = (pair_id - image_id2) // max_image_id
    return int(image_id1), int(image_id2)


def enrich_pair_records(db_path, pair_records, quality_config_path=None, auto_quality_refs=False):
    db = pycolmap.Database.open(db_path)
    pair_to_inliers = {}
    try:
        pair_ids, geometries = db.read_two_view_geometries()
        for pair_id, tvg in zip(pair_ids, geometries):
            key = decode_pair_id(pair_id)
            pair_to_inliers[key] = int(np.asarray(tvg.inlier_matches).shape[0])
    except Exception:
        pair_to_inliers = {}
    quality_payload = load_quality_config(quality_config_path)
    qpair_config_resolved = resolve_quality_section(quality_payload, "qpair", {})
    qpair_config = dict(qpair_config_resolved)

    raw_score_values = []
    ninliers_list = []
    inlier_ratio_list = []
    for rec in pair_records:
        ninliers = int(pair_to_inliers.get((rec["image_id1"], rec["image_id2"]), 0))
        rec["ninliers"] = ninliers
        rec["inlier_ratio"] = safe_ratio(ninliers, max(rec["nmatches"], 1))
        ninliers_list.append(ninliers)
        inlier_ratio_list.append(rec["inlier_ratio"])
        vals = [rec["score_mean"], rec["score_med"]]
        vals = [float(v) for v in vals if np.isfinite(v)]
        if vals:
            raw_score_values.append(float(np.mean(vals)))

    if auto_quality_refs and pair_records:
        qpair_config = infer_qpair_config(
            ninliers=ninliers_list,
            inlier_ratio=inlier_ratio_list,
            score_values=raw_score_values,
            base_config=qpair_config,
        )

    quality_record = make_quality_config_record(
        stage="build_lightglue_pycolmap_tracks_npz",
        section="qpair",
        raw_payload=quality_payload,
        resolved_section=qpair_config_resolved,
        effective_section=qpair_config,
        mode="weighted_sum",
        auto_quality_refs=auto_quality_refs,
        quality_config_path=quality_config_path,
    )

    for rec in pair_records:
        rec["q_pair"] = float(
            compute_qpair(
                ninliers=rec["ninliers"],
                nmatches=max(rec["nmatches"], 1),
                inlier_ratio=rec["inlier_ratio"],
                score_mean=rec["score_mean"],
                score_med=rec["score_med"],
                delta=rec["delta"],
                method=0,
                mode="weighted_sum",
                config=qpair_config,
            )
        )
    db.close()
    return pair_records, qpair_config, quality_record


def build_track_quality_sidecars(model_dir, ordered_names, pair_records, tracks_dir, min_track_len):
    images_bin = str(Path(model_dir) / "images.bin")
    images = read_images_binary_with_points(images_bin)
    resolved_images = resolve_ordered_images(ordered_names, images)

    track_len = defaultdict(int)
    track_to_frames = defaultdict(list)
    for fidx, item in enumerate(resolved_images):
        if item is None:
            continue
        for p3d in item["point3D_ids"]:
            p3d = int(p3d)
            if p3d < 0:
                continue
            track_len[p3d] += 1
            track_to_frames[p3d].append(fidx)

    kept_track_ids = sorted([tid for tid, length in track_len.items() if length >= min_track_len])
    pair_q_map = {}
    edge_i = []
    edge_j = []
    edge_delta = []
    edge_q = []
    edge_ninliers = []
    edge_inlier_ratio = []
    edge_score_mean = []
    edge_score_med = []
    for rec in pair_records:
        key = (int(rec["i"]), int(rec["j"]))
        pair_q_map[key] = float(rec["q_pair"])
        edge_i.append(int(rec["i"]))
        edge_j.append(int(rec["j"]))
        edge_delta.append(int(rec["delta"]))
        edge_q.append(float(rec["q_pair"]))
        edge_ninliers.append(int(rec["ninliers"]))
        edge_inlier_ratio.append(float(rec["inlier_ratio"]))
        edge_score_mean.append(float(rec["score_mean"]))
        edge_score_med.append(float(rec["score_med"]))

    pair_q_mean = np.ones((len(kept_track_ids),), dtype=np.float32)
    pair_q_min = np.ones((len(kept_track_ids),), dtype=np.float32)
    pair_q_count = np.zeros((len(kept_track_ids),), dtype=np.int32)
    lengths = np.zeros((len(kept_track_ids),), dtype=np.int32)

    for idx, tid in enumerate(kept_track_ids):
        frames = sorted(set(track_to_frames[int(tid)]))
        lengths[idx] = int(track_len[int(tid)])
        vals = []
        for a in range(len(frames)):
            for b in range(a + 1, len(frames)):
                q = pair_q_map.get((int(frames[a]), int(frames[b])))
                if q is None:
                    q = pair_q_map.get((int(frames[b]), int(frames[a])))
                if q is not None:
                    vals.append(float(q))
        if vals:
            vals_arr = np.asarray(vals, dtype=np.float64)
            pair_q_mean[idx] = float(np.mean(vals_arr))
            pair_q_min[idx] = float(np.min(vals_arr))
            pair_q_count[idx] = int(vals_arr.size)

    np.savez_compressed(
        Path(tracks_dir) / "track_quality_summary.npz",
        track_ids=np.asarray(kept_track_ids, dtype=np.int64),
        track_len=lengths,
        pair_q_mean=pair_q_mean,
        pair_q_min=pair_q_min,
        pair_q_count=pair_q_count,
    )
    np.savez_compressed(
        Path(tracks_dir) / "pair_quality_edges.npz",
        i=np.asarray(edge_i, dtype=np.int32),
        j=np.asarray(edge_j, dtype=np.int32),
        delta=np.asarray(edge_delta, dtype=np.int16),
        q_pair=np.asarray(edge_q, dtype=np.float32),
        ninliers=np.asarray(edge_ninliers, dtype=np.int32),
        inlier_ratio=np.asarray(edge_inlier_ratio, dtype=np.float32),
        score_mean=np.asarray(edge_score_mean, dtype=np.float32),
        score_med=np.asarray(edge_score_med, dtype=np.float32),
    )
    dump_json(
        str(Path(tracks_dir) / "track_build_quality_stats.json"),
        {
            "pair_stats": {
                "q_pair": summarize_array(edge_q),
                "ninliers": summarize_array(edge_ninliers),
                "inlier_ratio": summarize_array(edge_inlier_ratio),
            },
            "track_stats": {
                "count": int(len(kept_track_ids)),
                "track_length": summarize_array(lengths),
                "pair_q_mean": summarize_array(pair_q_mean),
                "pair_q_min": summarize_array(pair_q_min),
                "pair_q_count": summarize_array(pair_q_count),
            },
        },
    )


def main():
    ap = argparse.ArgumentParser(
        description="Build LightGlue matches, reconstruct with pycolmap, and export tracks/*.npz compatible with Pose_Only."
    )
    ap.add_argument("--image_list", default=None, help="Optional image_list.txt defining frame order")
    ap.add_argument("--image_glob", default=None, help="Fallback image glob if --image_list is not provided")
    ap.add_argument("--out_dir", required=True, help="Output root directory")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--deltas", type=str, default="1,2,3,5")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--max_kpts", type=int, default=2048)
    ap.add_argument("--filter_th", type=float, default=0.1)
    ap.add_argument("--mutual", action="store_true")
    ap.add_argument("--min_score", type=float, default=0.0)
    ap.add_argument("--dataset", choices=["kitti", "euroc", "custom", "none"], default="none")
    ap.add_argument("--K_npy", type=str, default=None)
    ap.add_argument("--Ks_npy", type=str, default=None)
    ap.add_argument("--image_K_idx_npy", type=str, default=None)
    ap.add_argument("--kitti_calib", type=str, default=None)
    ap.add_argument("--kitti_cam", type=str, default="P0")
    ap.add_argument("--euroc_yaml", type=str, default=None)
    ap.add_argument("--feature_cache_dir", type=str, default=None)
    ap.add_argument("--min_track_len", type=int, default=2)
    ap.add_argument("--image_root", type=str, default=None, help="Root directory passed to pycolmap; defaults to repo root for image_list-relative paths or common path otherwise")
    ap.add_argument("--quality_config", type=str, default=None)
    ap.add_argument("--auto_quality_refs", action="store_true")
    args = ap.parse_args()

    image_paths = read_image_paths(args)
    image_root = os.path.abspath(args.image_root) if args.image_root else infer_image_root(image_paths, args.image_list)
    image_names = relative_image_names(image_paths, image_root)
    print("[Tracks][pycolmap] frames:", len(image_paths))
    print("[Tracks][pycolmap] image_root:", image_root)

    if args.dataset == "custom" and args.Ks_npy is not None:
        Ks, image_K_idx = read_custom_Ks(args.Ks_npy, args.image_K_idx_npy)
        if image_K_idx is None:
            raise ValueError("--Ks_npy requires --image_K_idx_npy")
        if image_K_idx.shape[0] < len(image_paths):
            raise ValueError("image_K_idx has fewer entries than the requested frames")
        Ks_per_frame = [Ks[int(idx)] for idx in image_K_idx[: len(image_paths)]]
    else:
        K, _, _ = load_intrinsics(
            dataset=args.dataset,
            kitti_calib=args.kitti_calib,
            kitti_cam=args.kitti_cam,
            euroc_yaml=args.euroc_yaml,
            K_npy=args.K_npy,
        )
        if K is None:
            raise ValueError("This bridge currently requires known intrinsics (custom/KITTI/EuRoC).")
        Ks_per_frame = [K for _ in image_paths]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_ids, _ = write_database_schema(
        db_path=str(out_dir / "database.db"),
        image_paths=image_paths,
        image_names=image_names,
        Ks_per_frame=Ks_per_frame,
    )
    db_path, pairs_path, pair_records, gap_attempts = summarize_matching(
        image_paths=image_paths,
        image_names=image_names,
        image_ids=image_ids,
        args=args,
        Ks_per_frame=Ks_per_frame,
        out_dir=str(out_dir),
    )
    model_dir, best_idx, best_rec = run_pycolmap_reconstruction(
        db_path=db_path,
        pairs_path=pairs_path,
        image_root=image_root,
        out_dir=str(out_dir),
    )
    pair_records, qpair_config, quality_record = enrich_pair_records(
        db_path=db_path,
        pair_records=pair_records,
        quality_config_path=args.quality_config,
        auto_quality_refs=args.auto_quality_refs,
    )
    save_quality_config_record(str(out_dir / "quality_config_used.json"), quality_record)

    tracks_dir = out_dir / "tracks"
    export_stats = export_tracks_from_colmap_images(
        images=read_images_binary_with_points(str(Path(model_dir) / "images.bin")),
        ordered_names=image_names,
        out_dir=str(tracks_dir),
        min_track_len=args.min_track_len,
    )
    build_track_quality_sidecars(
        model_dir=model_dir,
        ordered_names=image_names,
        pair_records=pair_records,
        tracks_dir=str(tracks_dir),
        min_track_len=args.min_track_len,
    )

    gap_success = defaultdict(int)
    for rec in pair_records:
        if rec["ninliers"] > 0:
            gap_success[int(rec["delta"])] += 1

    dump_json(
        str(out_dir / "pycolmap_track_build_stats.json"),
        {
            "config": vars(args),
            "image_root": image_root,
            "image_names_sample": image_names[: min(5, len(image_names))],
            "database_path": db_path,
            "pairs_path": pairs_path,
            "best_model_index": int(best_idx),
            "best_model_num_reg_images": int(best_rec.num_reg_images()),
            "best_model_num_points3D": int(best_rec.num_points3D()),
            "pair_stats": {
                "q_pair": summarize_array([rec["q_pair"] for rec in pair_records]),
                "ninliers": summarize_array([rec["ninliers"] for rec in pair_records]),
                "inlier_ratio": summarize_array([rec["inlier_ratio"] for rec in pair_records]),
            },
            "gap_stats": {
                str(int(delta)): {
                    "attempts": int(gap_attempts.get(int(delta), 0)),
                    "successful_pairs": int(gap_success.get(int(delta), 0)),
                    "success_rate": safe_ratio(gap_success.get(int(delta), 0), max(gap_attempts.get(int(delta), 0), 1)),
                }
                for delta in parse_deltas(args.deltas)
            },
            "qpair_config": qpair_config,
            "quality_config_used": quality_record,
            "track_export": export_stats,
        },
    )

    print("[Tracks][pycolmap] done. out_dir =", out_dir)
    print("[Tracks][pycolmap] best_model_num_reg_images =", int(best_rec.num_reg_images()))
    print("[Tracks][pycolmap] best_model_num_points3D =", int(best_rec.num_points3D()))
    print("[Tracks][pycolmap] tracks_dir =", tracks_dir)


if __name__ == "__main__":
    main()
