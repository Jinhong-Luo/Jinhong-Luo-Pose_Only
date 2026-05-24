#!/usr/bin/env python3
# build_lightglue_tracks_npz_v2.py
# Only build multi-frame tracks (per-frame npz: track_ids + xy). No Rij output.

import os, glob, argparse
from collections import defaultdict
import numpy as np
import _bootstrap  # noqa: F401
from calib_utils import (
    format_kitti_layout_help,
    load_intrinsics,
    read_custom_Ks,
    resolve_kitti_scene_inputs,
)
from degeneracy_utils import dump_json, safe_ratio, summarize_array
from frontend_cache import build_lightglue_frontend, load_or_extract_feature, read_gray_u8, resolve_device
from pair_match_cache import extract_pair_matches, load_pair_match_cache, save_pair_match_cache
from quality_utils import (
    compute_qpair,
    infer_qpair_config,
    load_quality_config,
    make_quality_config_record,
    resolve_quality_section,
    save_quality_config_record,
)

def save_frame_npz(out_dir, fidx, kp2tid, kpts_xy):
    # Save only keypoints that exist in kp2tid mapping.
    if len(kp2tid) == 0:
        track_ids = np.zeros((0,), np.int64)
        xy = np.zeros((0, 2), np.float32)
    else:
        kp_idx = np.array(list(kp2tid.keys()), np.int64)
        tids = np.array([kp2tid[int(i)] for i in kp_idx], np.int64)
        xy = kpts_xy[kp_idx].astype(np.float32)

        # stable ordering
        order = np.argsort(tids)
        track_ids = tids[order]
        xy = xy[order]

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{fidx:06d}.npz"),
                        track_ids=track_ids, xy=xy)


class TrackDSU:
    def __init__(self, global_frames):
        self.parent = np.arange(len(global_frames), dtype=np.int64)
        self.size = np.ones(len(global_frames), dtype=np.int32)
        self.frames = [{int(f)} for f in global_frames]

    def find(self, x):
        p = self.parent[x]
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        while x != p:
            nxt = self.parent[x]
            self.parent[x] = p
            x = nxt
        return p

    def union_if_compatible(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return True
        if not self.frames[ra].isdisjoint(self.frames[rb]):
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.frames[ra].update(self.frames[rb])
        self.frames[rb] = set()
        return True
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

# def read_kitti_K(calib_txt, cam="P0"):
#     with open(calib_txt, "r") as f:
#         for line in f:
#             if line.startswith(cam + ":"):
#                 vals = [float(x) for x in line.split()[1:]]
#                 P = np.array(vals, np.float64).reshape(3, 4)
#                 return P[:, :3].copy()
#     raise RuntimeError(f"Cannot find {cam}: in {calib_txt}")

# def read_euroc_cam_yaml(sensor_yaml):
#     import yaml
#     with open(sensor_yaml, "r") as f:
#         y = yaml.safe_load(f)
#     fx, fy, cx, cy = y["intrinsics"]
#     K = np.array([[fx, 0, cx],
#                   [0, fy, cy],
#                   [0,  0,  1]], dtype=np.float64)
#     dist = np.array(y.get("distortion_coefficients", []), dtype=np.float64).reshape(-1)
#     model = y.get("distortion_model", "radial-tangential")
#     return K, dist, model

def undistort_to_pixel(pts_px, K, dist, model):
    """
    input : (N,2) distorted pixel points
    output: (N,2) undistorted pixel points (consistent with K)
    """
    import cv2
    pts = np.asarray(pts_px, np.float32).reshape(-1, 1, 2)

    if model in ["radial-tangential", "plumb_bob", "radtan"]:
        out = cv2.undistortPoints(pts, K, dist, P=K)
        return out.reshape(-1, 2).astype(np.float64)

    if "fisheye" in str(model).lower():
        out = cv2.fisheye.undistortPoints(pts, K, dist, P=K)
        return out.reshape(-1, 2).astype(np.float64)

    out = cv2.undistortPoints(pts, K, dist, P=K)
    return out.reshape(-1, 2).astype(np.float64)


def pixel_to_normalized(pts_px, K):
    pts_px = np.asarray(pts_px, np.float64).reshape(-1, 2)
    pts_h = np.concatenate([pts_px, np.ones((pts_px.shape[0], 1), np.float64)], axis=1)
    Kinv = np.linalg.inv(np.asarray(K, np.float64))
    pts_n = (Kinv @ pts_h.T).T
    pts_n /= np.maximum(pts_n[:, 2:3], 1e-12)
    return pts_n[:, :2]


def estimate_essential_inliers_normalized(pts0_n, pts1_n, min_inliers):
    import cv2

    if len(pts0_n) < 8:
        return np.zeros((0,), dtype=bool)

    K_I = np.eye(3, dtype=np.float64)
    E, mask = cv2.findEssentialMat(
        np.asarray(pts0_n, np.float64),
        np.asarray(pts1_n, np.float64),
        K_I,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1e-3,
    )
    if E is None or mask is None:
        return np.zeros((len(pts0_n),), dtype=bool)

    inliers = mask.reshape(-1).astype(bool)
    if int(inliers.sum()) < min_inliers:
        return np.zeros((len(pts0_n),), dtype=bool)
    return inliers

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--image_glob", required=True, help="e.g. .../image_0/*.png")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--deltas", type=str, default="1",
                    help="frame gaps used to build tracks, e.g. 1 or 1,2,3,5")

    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--max_kpts", type=int, default=2048)
    ap.add_argument("--filter_th", type=float, default=0.1, help="LightGlue filter_threshold")

    # match filtering
    ap.add_argument("--mutual", action="store_true", help="mutual consistency check")
    ap.add_argument("--min_score", type=float, default=0.0, help="LightGlue matching score threshold")

    # geometry filter (optional, strongly recommended on KITTI)
    ap.add_argument("--use_ransac", action="store_true")
    ap.add_argument("--ransac_thresh", type=float, default=1.0, help="Essential RANSAC threshold (px)")
    ap.add_argument("--min_inliers", type=int, default=50)

    # calibration / undistort (needed for EuRoC or for ransac-K)
    ap.add_argument("--dataset", choices=["kitti", "euroc", "custom", "none"], default="none")
    ap.add_argument("--K_npy", type=str, default=None, help="custom dataset intrinsics .npy")
    ap.add_argument("--Ks_npy", type=str, default=None, help="custom dataset per-camera intrinsics .npy, shape (M,3,3)")
    ap.add_argument("--image_K_idx_npy", type=str, default=None, help="per-frame intrinsics index .npy, shape (N,)")
    ap.add_argument("--kitti_calib", type=str, default=None, help="KITTI calib.txt")
    ap.add_argument("--kitti_cam", type=str, default="P0", help="P0/P2 ...")
    ap.add_argument("--euroc_yaml", type=str, default=None, help="EuRoC cam sensor.yaml")
    ap.add_argument("--undistort", action="store_true", help="undistort keypoints before saving xy (EuRoC)")

    # optional: seed all keypoints with unique tid (bigger output, can help later post-process)
    ap.add_argument("--seed_all", action="store_true",
                    help="assign a track id to every keypoint each frame (larger files)")
    ap.add_argument("--qpair_mode", choices=["weighted_sum", "product"], default="weighted_sum")
    ap.add_argument("--qpair_threshold", type=float, default=0.0,
                    help="optional pair-quality threshold for track-building stats/reporting only; matching/union remains unchanged")
    ap.add_argument("--dump_quality_stats", action="store_true",
                    help="save pair/track quality sidecar files for later LiGT/PA weighting")
    ap.add_argument("--feature_cache_dir", type=str, default=None,
                    help="optional directory for reusable SuperPoint feature cache")
    ap.add_argument("--pair_match_cache_dir", type=str, default=None,
                    help="optional directory for reusable pair raw match cache")
    ap.add_argument("--quality_config", type=str, default=None,
                    help="optional JSON config with a qpair section")
    ap.add_argument("--auto_quality_refs", action="store_true",
                    help="fit q_pair reference scales from current pair statistics")

    args = ap.parse_args()

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

    imgs = sorted(glob.glob(args.image_glob))
    if not imgs:
        if args.dataset == "kitti":
            scene_id = resolved.get("scene_id") if "resolved" in locals() else None
            raise FileNotFoundError(
                f"No KITTI images matched: {args.image_glob}\n"
                f"{format_kitti_layout_help(scene_id)}"
            )
        raise FileNotFoundError(args.image_glob)
    if args.max_frames is not None:
        imgs = imgs[:args.max_frames]
    N = len(imgs)
    deltas = parse_deltas(args.deltas)
    print("[Tracks] frames:", N)
    print("[Tracks] deltas:", deltas)

    # load intrinsics if needed
    Ks = None
    image_K_idx = None
    if args.dataset == "custom" and args.Ks_npy is not None:
        Ks, image_K_idx = read_custom_Ks(args.Ks_npy, args.image_K_idx_npy)
        if image_K_idx is None:
            raise ValueError("--Ks_npy requires --image_K_idx_npy")
        if image_K_idx.shape[0] < N:
            raise ValueError(f"image_K_idx has {image_K_idx.shape[0]} entries but needs at least {N}")
        image_K_idx = image_K_idx[:N]
        K = None
        dist = None
        dist_model = None
    else:
        K, dist, dist_model = load_intrinsics(
            dataset=args.dataset,
            kitti_calib=args.kitti_calib,
            kitti_cam=args.kitti_cam,
            euroc_yaml=args.euroc_yaml,
            K_npy=args.K_npy,
        )

    if K is not None:
        print("[K]\n", K)
    elif Ks is not None:
        print(f"[Ks] count={Ks.shape[0]}")
    if dist is not None:
        print("[DIST]", dist, "model=", dist_model)

    if args.use_ransac and K is None and Ks is None:
        raise ValueError("--use_ransac needs intrinsics K (set --dataset kitti/euroc/custom and calib args)")

    # LightGlue
    import torch
    dev = resolve_device(args.device)
    print("[Device]", dev)

    extractor, matcher = build_lightglue_frontend(
        max_kpts=args.max_kpts,
        filter_th=args.filter_th,
        device=dev,
    )

    feats_cache = []
    kpts_save_all = []
    n_kpts_per_frame = []
    for i, img_path in enumerate(imgs):
        feats, kpts = load_or_extract_feature(
            img_path,
            extractor=extractor,
            device=dev,
            max_kpts=args.max_kpts,
            cache_dir=args.feature_cache_dir,
            image_loader=read_gray_u8,
        )
        if args.undistort:
            if K is None or dist is None:
                raise ValueError("--undistort requires calibration (use --dataset euroc with --euroc_yaml)")
            kpts_save = undistort_to_pixel(kpts, K, dist, dist_model)
        else:
            kpts_save = kpts.astype(np.float64)
        feats_cache.append(feats)
        kpts_save_all.append(kpts_save)
        n_kpts_per_frame.append(kpts_save.shape[0])

    offsets = np.zeros(N + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(n_kpts_per_frame, dtype=np.int64)
    global_frames = np.concatenate([
        np.full(n_kpts_per_frame[i], i, dtype=np.int32) for i in range(N)
    ], axis=0) if offsets[-1] > 0 else np.zeros((0,), dtype=np.int32)
    dsu = TrackDSU(global_frames)
    pair_records = []
    gap_attempts = {int(d): 0 for d in deltas}
    gap_success = {int(d): 0 for d in deltas}

    for i in range(N):
        feats0 = feats_cache[i]
        kpts0_save = kpts_save_all[i]
        for d in deltas:
            j = i + d
            if j >= N:
                continue
            gap_attempts[int(d)] += 1
            feats1 = feats_cache[j]
            kpts1_save = kpts_save_all[j]
            kpts0_raw = feats_cache[i]["keypoints"][0].detach().cpu().numpy().astype(np.float64)
            kpts1_raw = feats_cache[j]["keypoints"][0].detach().cpu().numpy().astype(np.float64)
            cached_pair = load_pair_match_cache(args.pair_match_cache_dir, i, j)
            if cached_pair is not None:
                matches = cached_pair["matches"]
                pair_scores_full = cached_pair["pair_scores"]
                pts0_cached = cached_pair.get("pts0")
                pts1_cached = cached_pair.get("pts1")
            else:
                with torch.no_grad():
                    out = matcher({"image0": feats0, "image1": feats1})
                matches, pair_scores_full = extract_pair_matches(out, mutual=args.mutual)
                pts0_cached = kpts0_raw[matches[:, 0]].astype(np.float32, copy=False) if matches.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
                pts1_cached = kpts1_raw[matches[:, 1]].astype(np.float32, copy=False) if matches.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
                if args.pair_match_cache_dir:
                    save_pair_match_cache(args.pair_match_cache_dir, i, j, matches, pair_scores_full, pts0_cached, pts1_cached)

            nmatch_raw = int(matches.shape[0])
            score_mean = np.nan
            score_med = np.nan

            if nmatch_raw > 0:
                idx0 = matches[:, 0].astype(np.int64, copy=False)
                idx1 = matches[:, 1].astype(np.int64, copy=False)
            else:
                idx0 = np.zeros((0,), dtype=np.int64)
                idx1 = np.zeros((0,), dtype=np.int64)

            pair_scores = pair_scores_full
            pts0_pair = pts0_cached if pts0_cached is not None else None
            pts1_pair = pts1_cached if pts1_cached is not None else None
            if args.min_score > 0 and pair_scores is not None and idx0.size > 0:
                keep = pair_scores >= args.min_score
                idx0 = idx0[keep]
                idx1 = idx1[keep]
                pair_scores = pair_scores[keep]
                if pts0_pair is not None:
                    pts0_pair = pts0_pair[keep]
                if pts1_pair is not None:
                    pts1_pair = pts1_pair[keep]

            if pair_scores is not None and idx0.size > 0:
                score_mean = float(np.mean(pair_scores))
                score_med = float(np.median(pair_scores))

            if args.use_ransac:
                if len(idx0) >= 8:
                    if pts0_pair is not None and pts1_pair is not None and not args.undistort:
                        pts0 = pts0_pair.astype(np.float64, copy=False)
                        pts1 = pts1_pair.astype(np.float64, copy=False)
                    else:
                        pts0 = kpts0_save[idx0].astype(np.float64)
                        pts1 = kpts1_save[idx1].astype(np.float64)
                    if Ks is not None:
                        K0 = Ks[int(image_K_idx[i])]
                        K1 = Ks[int(image_K_idx[j])]
                        pts0_n = pixel_to_normalized(pts0, K0)
                        pts1_n = pixel_to_normalized(pts1, K1)
                        inl = estimate_essential_inliers_normalized(pts0_n, pts1_n, args.min_inliers)
                    else:
                        import cv2
                        E, mask = cv2.findEssentialMat(
                            pts0, pts1, K,
                            method=cv2.RANSAC, prob=0.999, threshold=args.ransac_thresh
                        )
                        if mask is None:
                            inl = np.zeros((len(idx0),), dtype=bool)
                        else:
                            inl = mask.reshape(-1).astype(bool)
                            if int(inl.sum()) < args.min_inliers:
                                inl = np.zeros((len(idx0),), dtype=bool)
                    if int(inl.sum()) >= args.min_inliers:
                        idx0 = idx0[inl]
                        idx1 = idx1[inl]
                    else:
                        idx0 = idx0[:0]
                        idx1 = idx1[:0]

            ninliers = int(len(idx0))
            inlier_ratio = safe_ratio(ninliers, max(nmatch_raw, 1))
            merged = 0
            conflicts = 0
            ga_list = []
            for a, b in zip(idx0.tolist(), idx1.tolist()):
                ga = int(offsets[i] + a)
                gb = int(offsets[j] + b)
                if dsu.union_if_compatible(ga, gb):
                    merged += 1
                    ga_list.append(ga)
                else:
                    conflicts += 1
            if ninliers > 0:
                gap_success[int(d)] += 1
            pair_records.append({
                "i": int(i),
                "j": int(j),
                "delta": int(d),
                "nmatches": int(max(nmatch_raw, 1)),
                "ninliers": int(ninliers),
                "inlier_ratio": float(inlier_ratio),
                "score_mean": float(score_mean) if np.isfinite(score_mean) else np.nan,
                "score_med": float(score_med) if np.isfinite(score_med) else np.nan,
                "ga_list": np.asarray(ga_list, dtype=np.int64),
                "merged": int(merged),
                "conflicts": int(conflicts),
            })

    quality_payload = load_quality_config(args.quality_config)
    qpair_config_resolved = resolve_quality_section(quality_payload, "qpair", {})
    qpair_config = dict(qpair_config_resolved)
    if args.auto_quality_refs and pair_records:
        score_values = []
        for rec in pair_records:
            vals = [rec["score_mean"], rec["score_med"]]
            vals = [float(v) for v in vals if np.isfinite(v)]
            if vals:
                score_values.append(float(np.mean(vals)))
        qpair_config = infer_qpair_config(
            ninliers=[rec["ninliers"] for rec in pair_records],
            inlier_ratio=[rec["inlier_ratio"] for rec in pair_records],
            score_values=score_values,
            base_config=qpair_config,
        )
    quality_record = make_quality_config_record(
        stage="build_lightglue_tracks_npz",
        section="qpair",
        raw_payload=quality_payload,
        resolved_section=qpair_config_resolved,
        effective_section=qpair_config,
        mode=args.qpair_mode,
        auto_quality_refs=args.auto_quality_refs,
        quality_config_path=args.quality_config,
        extra={
            "output_path": os.path.join(args.out_dir, "quality_config_used.json"),
            "decision_parameters": {
                "qpair_threshold": float(args.qpair_threshold),
            },
        },
    )

    pair_stats = []
    for rec in pair_records:
        q_pair = compute_qpair(
            ninliers=rec["ninliers"],
            nmatches=max(rec["nmatches"], 1),
            inlier_ratio=rec["inlier_ratio"],
            score_mean=rec["score_mean"],
            score_med=rec["score_med"],
            delta=rec["delta"],
            method=0,
            mode=args.qpair_mode,
            config=qpair_config,
        )
        rec["q_pair"] = float(q_pair)
        pair_stats.append((
            rec["i"],
            rec["j"],
            rec["delta"],
            rec["ninliers"],
            rec["merged"],
            rec["conflicts"],
            float(q_pair),
        ))

    root_to_tid = {}
    next_tid = 0
    kp_maps = [dict() for _ in range(N)]
    for fidx in range(N):
        for kp_idx in range(n_kpts_per_frame[fidx]):
            gid = int(offsets[fidx] + kp_idx)
            root = dsu.find(gid)
            comp_size = int(dsu.size[dsu.find(root)])
            if comp_size < 2 and not args.seed_all:
                continue
            if root not in root_to_tid:
                root_to_tid[root] = next_tid
                next_tid += 1
            kp_maps[fidx][kp_idx] = root_to_tid[root]

    for fidx in range(N):
        save_frame_npz(args.out_dir, fidx, kp_maps[fidx], kpts_save_all[fidx])

    track_lengths = np.zeros((next_tid,), dtype=np.int32)
    for fidx in range(N):
        for tid in kp_maps[fidx].values():
            track_lengths[int(tid)] += 1

    track_pair_qs = defaultdict(list)
    edge_i = []
    edge_j = []
    edge_delta = []
    edge_q = []
    edge_ninliers = []
    edge_inlier_ratio = []
    edge_score_mean = []
    edge_score_med = []
    for rec in pair_records:
        tids = set()
        for ga in rec["ga_list"].tolist():
            root = dsu.find(int(ga))
            tid = root_to_tid.get(root, None)
            if tid is not None:
                tids.add(int(tid))
        for tid in tids:
            track_pair_qs[tid].append(float(rec["q_pair"]))
        edge_i.append(int(rec["i"]))
        edge_j.append(int(rec["j"]))
        edge_delta.append(int(rec["delta"]))
        edge_q.append(float(rec["q_pair"]))
        edge_ninliers.append(int(rec["ninliers"]))
        edge_inlier_ratio.append(float(rec["inlier_ratio"]))
        edge_score_mean.append(float(rec["score_mean"]))
        edge_score_med.append(float(rec["score_med"]))

    track_ids = np.arange(next_tid, dtype=np.int64)
    track_pair_q_mean = np.ones((next_tid,), dtype=np.float32)
    track_pair_q_min = np.ones((next_tid,), dtype=np.float32)
    track_pair_q_count = np.zeros((next_tid,), dtype=np.int32)
    for tid in range(next_tid):
        vals = np.asarray(track_pair_qs.get(int(tid), []), dtype=np.float64)
        if vals.size == 0:
            continue
        track_pair_q_mean[tid] = float(np.mean(vals))
        track_pair_q_min[tid] = float(np.min(vals))
        track_pair_q_count[tid] = int(vals.size)

    if pair_stats:
        kept = [x[3] for x in pair_stats]
        merged = [x[4] for x in pair_stats]
        conflicts = [x[5] for x in pair_stats]
        qvals = [x[6] for x in pair_stats]
        print(f"[Tracks] pair count={len(pair_stats)}, kept matches median={np.median(kept):.1f}, merged median={np.median(merged):.1f}, conflicts total={int(np.sum(conflicts))}")
        print(f"[Tracks] q_pair median={np.median(qvals):.3f}, p10={np.quantile(qvals, 0.1):.3f}, p90={np.quantile(qvals, 0.9):.3f}")
    if next_tid > 0:
        print(f"[Tracks] tracks={next_tid}, mean_len={np.mean(track_lengths):.2f}, median_len={np.median(track_lengths):.2f}")
    for d in deltas:
        succ = gap_success.get(int(d), 0)
        att = gap_attempts.get(int(d), 0)
        print(f"[Tracks][gap={int(d)}] attempts={att}, successful_pairs={succ}, success_rate={safe_ratio(succ, max(att, 1)):.3f}")

    if args.dump_quality_stats:
        os.makedirs(args.out_dir, exist_ok=True)
        save_quality_config_record(os.path.join(args.out_dir, "quality_config_used.json"), quality_record)
        np.savez_compressed(
            os.path.join(args.out_dir, "track_quality_summary.npz"),
            track_ids=track_ids,
            track_len=track_lengths.astype(np.int32),
            pair_q_mean=track_pair_q_mean,
            pair_q_min=track_pair_q_min,
            pair_q_count=track_pair_q_count,
        )
        np.savez_compressed(
            os.path.join(args.out_dir, "pair_quality_edges.npz"),
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
            os.path.join(args.out_dir, "track_build_quality_stats.json"),
            {
                "pair_stats": {
                    "q_pair": summarize_array(edge_q),
                    "ninliers": summarize_array(edge_ninliers),
                    "inlier_ratio": summarize_array(edge_inlier_ratio),
                },
                "track_stats": {
                    "count": int(next_tid),
                    "track_length": summarize_array(track_lengths),
                    "pair_q_mean": summarize_array(track_pair_q_mean),
                    "pair_q_min": summarize_array(track_pair_q_min),
                    "pair_q_count": summarize_array(track_pair_q_count),
                },
                "gap_stats": {
                    str(int(d)): {
                        "attempts": int(gap_attempts.get(int(d), 0)),
                        "successful_pairs": int(gap_success.get(int(d), 0)),
                        "success_rate": safe_ratio(gap_success.get(int(d), 0), max(gap_attempts.get(int(d), 0), 1)),
                    }
                    for d in deltas
                },
                "qpair_config": qpair_config,
                "quality_config_used": quality_record,
                "config": vars(args),
            },
        )
    else:
        save_quality_config_record(os.path.join(args.out_dir, "quality_config_used.json"), quality_record)
    print("[Tracks] done. out_dir =", args.out_dir)

if __name__ == "__main__":
    main()


# python tools/build_lightglue_tracks_npz.py \
#   --dataset custom \
#   --K_npy data/prepared/strecha/fountain-P11/K.npy \
#   --image_glob "data/raw/strecha/fountain-P11/images/*.jpg" \
#   --out_dir runs/strecha/fountain-P11/tracks_npz \
#   --device cpu \
#   --max_kpts 2048 \
#   --filter_th 0.1 \
#   --mutual \
#   --min_score 0.2 \
#   --use_ransac \
#   --ransac_thresh 1.0 \
#   --min_inliers 50
