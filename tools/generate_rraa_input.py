#!/usr/bin/env python3
import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import torch

import _bootstrap  # noqa: F401
from calib_utils import (
    format_kitti_layout_help,
    load_intrinsics,
    list_images_sorted,
    read_custom_Ks,
    resolve_kitti_scene_inputs,
)
from degeneracy_utils import dump_json, safe_ratio, summarize_array
from frontend_cache import build_lightglue_frontend, load_or_extract_feature, read_gray_u8, resolve_device
from lightglue.utils import rbd
from pair_match_cache import extract_pair_matches, load_pair_match_cache, save_pair_match_cache
from quality_utils import (
    compute_qpair,
    infer_qpair_config,
    load_quality_config,
    make_quality_config_record,
    resolve_quality_section,
    save_quality_config_record,
)


# -----------------------------
# Utils
# -----------------------------
def to_numpy_cpu(x):
    """
    Robustly convert torch tensor / numpy array / list to CPU numpy array.
    Works for CUDA tensors.
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def project_to_so3(Rm: np.ndarray) -> np.ndarray:
    """Project a matrix to the nearest SO(3)."""
    U, _, Vt = np.linalg.svd(Rm)
    Rproj = (U @ Vt).astype(np.float32)
    if np.linalg.det(Rproj) < 0:
        U[:, -1] *= -1
        Rproj = (U @ Vt).astype(np.float32)
    return Rproj


def parse_min_inliers_map(s: str):
    """
    Example:
      "1:10,2:20,3:30,5:40,8:60,13:60,default:60"
    """
    mp = {}
    default = None
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split(":")
        k = k.strip()
        v = int(v.strip())
        if k == "default":
            default = v
        else:
            mp[int(k)] = v
    return mp, default


def min_inliers_for_delta(d: int, mp: dict, default: int):
    if d in mp:
        return mp[d]
    return default


# -----------------------------
# Distortion / normalized coords
# -----------------------------
def undistort_to_normalized(pts_px, K, dist, model):
    pts_px = np.asarray(pts_px, np.float32).reshape(-1, 1, 2)
    if model in ["radial-tangential", "plumb_bob", "radtan"]:
        pts_n = cv2.undistortPoints(pts_px, K, dist)
        return pts_n.reshape(-1, 2).astype(np.float64)
    if "fisheye" in str(model).lower():
        pts_n = cv2.fisheye.undistortPoints(pts_px, K, dist)
        return pts_n.reshape(-1, 2).astype(np.float64)
    pts_n = cv2.undistortPoints(pts_px, K, dist)
    return pts_n.reshape(-1, 2).astype(np.float64)


def pixel_to_normalized(pts_px, K):
    pts_px = np.asarray(pts_px, np.float64).reshape(-1, 2)
    pts_h = np.concatenate([pts_px, np.ones((pts_px.shape[0], 1), np.float64)], axis=1)
    Kinv = np.linalg.inv(np.asarray(K, np.float64))
    pts_n = (Kinv @ pts_h.T).T
    pts_n /= np.maximum(pts_n[:, 2:3], 1e-12)
    return pts_n[:, :2].astype(np.float64)


# -----------------------------
# Geometry: Essential + optional Homography fallback
# method_flag: 0=Essential, 1=Homography
# -----------------------------
def estimate_Rij_from_matches_pixel(
    pts0,
    pts1,
    K,
    ransac_thresh_px=2.0,
    min_inliers=30,
    use_h_fallback=True,
    h_reproj_px=3.0,
    min_h_inliers=30,
):
    if len(pts0) < 8:
        return None, 0, 0, -1

    pts0 = np.asarray(pts0, np.float32)
    pts1 = np.asarray(pts1, np.float32)
    K = np.asarray(K, np.float64)
    nmatches = int(len(pts0))

    # 1) Essential
    E, inlier_mask = cv2.findEssentialMat(
        pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh_px
    )

    best_R = None
    best_inl = -1

    if E is not None:
        # Multi-solution E
        if E.ndim == 2 and E.shape[0] == 3 and E.shape[1] % 3 == 0:
            for kk in range(E.shape[1] // 3):
                Ei = E[:, 3 * kk:3 * (kk + 1)]
                inliers, Rm, _, _ = cv2.recoverPose(Ei, pts0, pts1, K, mask=inlier_mask)
                if Rm is not None and int(inliers) > best_inl:
                    best_R = Rm
                    best_inl = int(inliers)
        else:
            inliers, Rm, _, _ = cv2.recoverPose(E, pts0, pts1, K, mask=inlier_mask)
            if Rm is not None:
                best_R = Rm
                best_inl = int(inliers)

        if best_R is not None and best_inl >= min_inliers:
            return project_to_so3(best_R), best_inl, nmatches, 0

    # 2) Homography fallback
    if use_h_fallback:
        H, maskH = cv2.findHomography(
            pts0, pts1, cv2.RANSAC, ransacReprojThreshold=h_reproj_px
        )
        if H is None or maskH is None:
            return None, 0, nmatches, -1

        hinl = int(maskH.sum())
        if hinl < min_h_inliers:
            return None, 0, nmatches, -1

        R_approx = np.linalg.inv(K) @ H @ K
        return project_to_so3(R_approx), hinl, nmatches, 1

    return None, 0, nmatches, -1


def estimate_Rij_from_matches_normalized(
    pts0_n, pts1_n, ransac_thresh_norm=1e-3, min_inliers=30
):
    """
    normalized coords (undistortPoints output), using K=I.
    """
    if len(pts0_n) < 8:
        return None, 0, 0, -1

    pts0_n = np.asarray(pts0_n, np.float64)
    pts1_n = np.asarray(pts1_n, np.float64)
    K_I = np.eye(3, dtype=np.float64)
    nmatches = int(len(pts0_n))

    E, inlier_mask = cv2.findEssentialMat(
        pts0_n, pts1_n, K_I, method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh_norm
    )
    if E is None:
        return None, 0, nmatches, -1

    best_R = None
    best_inl = -1

    if E.ndim == 2 and E.shape[0] == 3 and E.shape[1] % 3 == 0 and E.shape[1] > 3:
        for kk in range(E.shape[1] // 3):
            Ei = E[:, 3 * kk:3 * (kk + 1)]
            inliers, Rm, _, _ = cv2.recoverPose(Ei, pts0_n, pts1_n, K_I, mask=inlier_mask)
            if Rm is not None and int(inliers) > best_inl:
                best_R = Rm
                best_inl = int(inliers)
    else:
        inliers, Rm, _, _ = cv2.recoverPose(E, pts0_n, pts1_n, K_I, mask=inlier_mask)
        if Rm is not None:
            best_R = Rm
            best_inl = int(inliers)

    if best_R is None or best_inl < min_inliers:
        return None, 0, nmatches, -1

    return project_to_so3(best_R), best_inl, nmatches, 0


# -----------------------------
# Main
# -----------------------------
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["kitti", "euroc", "custom"], required=True)
    parser.add_argument("--img_dir", default=None)
    parser.add_argument("--image_list", default=None,
                        help="optional text file with one image path per line; if set, overrides --img_dir")
    parser.add_argument("--out_npz", required=True)

    # device
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Feature extraction/matching device. Geometry still runs on CPU.",
    )

    # custom
    parser.add_argument("--K_npy", default=None, help="custom dataset intrinsics .npy")
    parser.add_argument("--Ks_npy", default=None, help="custom dataset per-camera intrinsics .npy, shape (M,3,3)")
    parser.add_argument("--image_K_idx_npy", default=None, help="per-frame intrinsics index .npy, shape (N,)")

    # KITTI
    parser.add_argument("--kitti_calib", default=None)
    parser.add_argument("--kitti_cam", default="P0")

    # EuRoC
    parser.add_argument("--euroc_yaml", default=None)
    parser.add_argument("--euroc_undistort", action="store_true")

    # Pair strategy
    parser.add_argument("--deltas", default="1,2,5")
    parser.add_argument("--max_pairs", type=int, default=-1)

    # LightGlue / SuperPoint
    parser.add_argument("--max_kpts", type=int, default=2048)
    parser.add_argument("--filter_th", type=float, default=0.1)
    parser.add_argument(
        "--min_match_score",
        type=float,
        default=0.0,
        help="Optional: drop matches with per-match score < this (if scores available).",
    )

    # Geometry
    parser.add_argument("--ransac_px", type=float, default=2.0)
    parser.add_argument("--ransac_norm", type=float, default=1e-3)

    # Essential / Homography thresholds (pixel branch)
    parser.add_argument(
        "--min_inliers_map",
        type=str,
        default="1:10,2:20,3:30,4:30,5:40,8:60,13:60,default:60",
        help='Per-delta min inliers map, e.g. "1:10,2:20,5:40,default:60"',
    )
    parser.add_argument("--use_h_fallback", action="store_true", help="Enable homography fallback.")
    parser.add_argument("--h_reproj_px", type=float, default=3.0)
    parser.add_argument("--min_h_inliers", type=int, default=30)

    # Graph
    parser.add_argument("--add_reverse", action="store_true")

    # Saving extras
    parser.add_argument("--save_meta", action="store_true", help="Save per-edge metadata arrays into npz.")
    parser.add_argument("--save_names", action="store_true", help="Save image basenames and K.")
    parser.add_argument("--qpair_mode", choices=["weighted_sum", "product"], default="weighted_sum")
    parser.add_argument("--qpair_threshold", type=float, default=0.0,
                        help="Optional threshold for reporting and downstream RRAA edge prefiltering.")
    parser.add_argument("--dump_quality_stats", action="store_true",
                        help="Save pair-quality and gap-statistics JSON next to output npz.")
    parser.add_argument("--feature_cache_dir", type=str, default=None,
                        help="optional directory for reusable SuperPoint feature cache")
    parser.add_argument("--pair_match_cache_dir", type=str, default=None,
                        help="optional directory for reusable pair raw match cache")
    parser.add_argument("--quality_config", type=str, default=None,
                        help="optional JSON config with a qpair section")
    parser.add_argument("--auto_quality_refs", action="store_true",
                        help="fit q_pair reference scales from current pair statistics")

    args = parser.parse_args()

    # expand paths safely
    if args.img_dir is not None:
        args.img_dir = os.path.abspath(os.path.expanduser(args.img_dir))
    if args.image_list is not None:
        args.image_list = os.path.abspath(os.path.expanduser(args.image_list))
    args.out_npz = os.path.abspath(os.path.expanduser(args.out_npz))
    if args.K_npy is not None:
        args.K_npy = os.path.abspath(os.path.expanduser(args.K_npy))
    if args.kitti_calib is not None:
        args.kitti_calib = os.path.abspath(os.path.expanduser(args.kitti_calib))
    if args.euroc_yaml is not None:
        args.euroc_yaml = os.path.abspath(os.path.expanduser(args.euroc_yaml))

    if args.image_list is None and args.img_dir is None:
        raise ValueError("Either --img_dir or --image_list must be provided")

    if args.dataset == "kitti":
        if args.image_list is not None:
            raise ValueError("--image_list is not supported for dataset=kitti")
        resolved = resolve_kitti_scene_inputs(
            kitti_calib=args.kitti_calib,
            img_dir=args.img_dir,
        )
        if resolved["img_dir"] != args.img_dir or resolved["kitti_calib"] != args.kitti_calib:
            print(f"[KITTI] auto-resolved scene {resolved['scene_id']} to:")
            print(f"  calib  : {resolved['kitti_calib']}")
            print(f"  img_dir: {resolved['img_dir']}")
            args.img_dir = resolved["img_dir"]
            args.kitti_calib = resolved["kitti_calib"]

    # choose device
    device = resolve_device(args.device, strict_cuda=True)

    print("[Device]", device)

    try:
        if args.image_list is not None:
            with open(args.image_list, "r", encoding="utf-8") as f:
                image_paths = [line.strip() for line in f if line.strip()]
            if not image_paths:
                raise RuntimeError(f"No valid image paths found in {args.image_list}")
        else:
            image_paths = list_images_sorted(args.img_dir)
    except RuntimeError as exc:
        if args.dataset == "kitti":
            scene_id = resolved.get("scene_id") if "resolved" in locals() else None
            raise RuntimeError(f"{exc}\n{format_kitti_layout_help(scene_id)}") from exc
        raise
    N = len(image_paths)
    deltas = [int(x) for x in args.deltas.split(",") if x.strip()]
    if not deltas:
        raise RuntimeError("Empty --deltas")
    max_delta = max(deltas)

    # parse min_inliers map
    mp, default_inl = parse_min_inliers_map(args.min_inliers_map)
    if default_inl is None:
        default_inl = 60

    # load calibration
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
    print("[K]\n", K)
    if Ks is not None:
        print(f"[Ks] count={Ks.shape[0]}")

    # LightGlue / SuperPoint
    extractor, matcher = build_lightglue_frontend(
        max_kpts=args.max_kpts,
        filter_th=args.filter_th,
        device=device,
    )

    edges = []
    Rij_list = []

    # meta lists
    ninl_list = []
    nmatch_list = []
    delta_list = []
    method_list = []   # 0=Essential, 1=Homography
    score_med_list = []
    score_mean_list = []

    feat_cache = {}

    def get_feats(idx: int):
        f = feat_cache.get(idx, None)
        if f is None:
            f, _ = load_or_extract_feature(
                image_paths[idx],
                extractor=extractor,
                device=device,
                max_kpts=args.max_kpts,
                cache_dir=args.feature_cache_dir,
                image_loader=read_gray_u8,
            )
            feat_cache[idx] = f
        return f

    pair_count = 0
    gap_attempts = {int(d): 0 for d in deltas}
    gap_success = {int(d): 0 for d in deltas}

    for i in tqdm(range(N), desc="frames"):
        feats0 = get_feats(i)

        for d in deltas:
            j = i + d
            if j >= N:
                continue
            gap_attempts[int(d)] += 1

            feats1 = get_feats(j)

            cached_pair = load_pair_match_cache(args.pair_match_cache_dir, i, j)
            if cached_pair is not None:
                matches = cached_pair["matches"]
                pair_scores = cached_pair["pair_scores"]
                pts0 = cached_pair.get("pts0")
                pts1 = cached_pair.get("pts1")
            else:
                out = matcher({"image0": feats0, "image1": feats1})
                matches, pair_scores = extract_pair_matches(out, mutual=False)
                kpts0 = to_numpy_cpu(rbd(feats0)["keypoints"])
                kpts1 = to_numpy_cpu(rbd(feats1)["keypoints"])
                pts0 = kpts0[matches[:, 0]] if matches.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
                pts1 = kpts1[matches[:, 1]] if matches.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
                if args.pair_match_cache_dir:
                    save_pair_match_cache(args.pair_match_cache_dir, i, j, matches, pair_scores, pts0, pts1)

            if matches.shape[0] < 8:
                continue

            # optional score-based prefilter
            if pair_scores is not None and args.min_match_score > 0:
                keep = pair_scores >= args.min_match_score
                matches = matches[keep]
                if matches.shape[0] < 8:
                    continue
                pair_scores = pair_scores[keep]
                if pts0 is not None:
                    pts0 = pts0[keep]
                if pts1 is not None:
                    pts1 = pts1[keep]

            if pts0 is None or pts1 is None:
                kpts0 = to_numpy_cpu(rbd(feats0)["keypoints"])
                kpts1 = to_numpy_cpu(rbd(feats1)["keypoints"])
                pts0 = kpts0[matches[:, 0]]
                pts1 = kpts1[matches[:, 1]]

            min_inl = min_inliers_for_delta(d, mp, default_inl)

            if Ks is not None:
                K0 = Ks[int(image_K_idx[i])]
                K1 = Ks[int(image_K_idx[j])]
                pts0_n = pixel_to_normalized(pts0, K0)
                pts1_n = pixel_to_normalized(pts1, K1)
                R_ij, ninl, nmatches, method = estimate_Rij_from_matches_normalized(
                    pts0_n,
                    pts1_n,
                    ransac_thresh_norm=args.ransac_norm,
                    min_inliers=min_inl,
                )
            elif args.dataset == "euroc" and args.euroc_undistort:
                pts0_n = undistort_to_normalized(pts0, K, dist, dist_model)
                pts1_n = undistort_to_normalized(pts1, K, dist, dist_model)
                R_ij, ninl, nmatches, method = estimate_Rij_from_matches_normalized(
                    pts0_n,
                    pts1_n,
                    ransac_thresh_norm=args.ransac_norm,
                    min_inliers=min_inl,
                )
            else:
                R_ij, ninl, nmatches, method = estimate_Rij_from_matches_pixel(
                    pts0,
                    pts1,
                    K,
                    ransac_thresh_px=args.ransac_px,
                    min_inliers=min_inl,
                    use_h_fallback=args.use_h_fallback,
                    h_reproj_px=args.h_reproj_px,
                    min_h_inliers=args.min_h_inliers,
                )

            if R_ij is None:
                continue

            inlier_ratio = float(ninl) / max(float(nmatches), 1.0)
            gap_success[int(d)] += 1

            # save directed edge (i -> j)
            edges.append((i, j))
            Rij_list.append(R_ij)

            if args.save_meta:
                ninl_list.append(ninl)
                nmatch_list.append(nmatches)
                delta_list.append(d)
                method_list.append(method)
                if pair_scores is None or len(pair_scores) == 0:
                    score_med_list.append(np.nan)
                    score_mean_list.append(np.nan)
                else:
                    score_med_list.append(float(np.median(pair_scores)))
                    score_mean_list.append(float(np.mean(pair_scores)))

            if args.add_reverse:
                edges.append((j, i))
                Rij_list.append(R_ij.T)

                if args.save_meta:
                    ninl_list.append(ninl)
                    nmatch_list.append(nmatches)
                    delta_list.append(d)
                    method_list.append(method)
                    if pair_scores is None or len(pair_scores) == 0:
                        score_med_list.append(np.nan)
                        score_mean_list.append(np.nan)
                    else:
                        score_med_list.append(float(np.median(pair_scores)))
                        score_mean_list.append(float(np.mean(pair_scores)))

            pair_count += 1
            if args.max_pairs > 0 and pair_count >= args.max_pairs:
                break

        if args.max_pairs > 0 and pair_count >= args.max_pairs:
            break

        # sliding-window cache cleanup
        lo, hi = i, i + max_delta
        for k in list(feat_cache.keys()):
            if k < lo or k > hi:
                del feat_cache[k]

    if len(edges) == 0:
        raise RuntimeError("No valid edges produced. Try lowering thresholds.")

    edges = np.array(edges, dtype=np.int32)
    Rij = np.stack(Rij_list, axis=0).astype(np.float32)

    save_kwargs = dict(N=int(N), edges=edges, Rij=Rij)
    quality_payload = load_quality_config(args.quality_config)
    qpair_config_resolved = resolve_quality_section(quality_payload, "qpair", {})
    qpair_config = dict(qpair_config_resolved)

    if args.save_meta:
        ninl = np.array(ninl_list, dtype=np.int32)
        nm = np.array(nmatch_list, dtype=np.int32)
        delta_arr = np.array(delta_list, dtype=np.int16)
        method_arr = np.array(method_list, dtype=np.int8)
        inlier_ratio = ninl.astype(np.float32) / np.maximum(nm.astype(np.float32), 1.0)
        score_med = np.array(score_med_list, dtype=np.float32)
        score_mean = np.array(score_mean_list, dtype=np.float32)
        if args.auto_quality_refs:
            score_values = []
            for s_mean, s_med in zip(score_mean.tolist(), score_med.tolist()):
                vals = [v for v in [s_mean, s_med] if np.isfinite(v)]
                if vals:
                    score_values.append(float(np.mean(vals)))
            qpair_config = infer_qpair_config(
                ninliers=ninl,
                inlier_ratio=inlier_ratio,
                score_values=score_values,
                base_config=qpair_config,
            )
        q_pair_meta = np.array(
            [
                compute_qpair(
                    ninliers=int(ninl[k]),
                    nmatches=int(max(nm[k], 1)),
                    inlier_ratio=float(inlier_ratio[k]),
                    score_mean=float(score_mean[k]),
                    score_med=float(score_med[k]),
                    delta=int(delta_arr[k]),
                    method=int(method_arr[k]),
                    mode=args.qpair_mode,
                    config=qpair_config,
                )
                for k in range(len(ninl))
            ],
            dtype=np.float32,
        )

        save_kwargs.update(
            ninliers=ninl,
            nmatches=nm,
            inlier_ratio=inlier_ratio,
            delta=delta_arr,
            method=method_arr,
            score_med=score_med,
            score_mean=score_mean,
            q_pair=q_pair_meta,
        )

    quality_record = make_quality_config_record(
        stage="generate_rraa_input",
        section="qpair",
        raw_payload=quality_payload,
        resolved_section=qpair_config_resolved,
        effective_section=qpair_config,
        mode=args.qpair_mode,
        auto_quality_refs=args.auto_quality_refs,
        quality_config_path=args.quality_config,
        extra={
            "output_path": os.path.splitext(args.out_npz)[0] + "_quality_config_used.json",
            "decision_parameters": {
                "qpair_threshold": float(args.qpair_threshold),
            },
        },
    )

    if args.save_names:
        names = np.array([os.path.basename(p) for p in image_paths], dtype="S")
        save_kwargs.update(img_names=names)
        save_kwargs.update(names=names)
        if K is not None:
            save_kwargs.update(K=K.astype(np.float64))
        if Ks is not None:
            save_kwargs.update(Ks=Ks.astype(np.float64), image_K_idx=image_K_idx.astype(np.int64))

    save_kwargs.update(config=str(vars(args)))

    out_dir = os.path.dirname(args.out_npz)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out_npz, **save_kwargs)

    print(f"[OK] saved {args.out_npz}")
    print(f"     N={N}, M={len(edges)}")
    if args.save_meta:
        qp = save_kwargs["q_pair"]
        print("     meta saved: ninliers/nmatches/inlier_ratio/delta/method/score_*/q_pair")
        print(f"     q_pair median={np.median(qp):.3f}, p10={np.quantile(qp, 0.1):.3f}, p90={np.quantile(qp, 0.9):.3f}, above_thr={int(np.sum(qp >= args.qpair_threshold))}/{len(qp)}")
    if args.save_names:
        print("     names saved: img_names (+K if available)")
    for d in deltas:
        att = gap_attempts.get(int(d), 0)
        succ = gap_success.get(int(d), 0)
        print(f"     gap={int(d)} attempts={att}, success={succ}, success_rate={safe_ratio(succ, max(att, 1)):.3f}")
    if args.dump_quality_stats:
        save_quality_config_record(os.path.splitext(args.out_npz)[0] + "_quality_config_used.json", quality_record)
        dump_json(
            os.path.splitext(args.out_npz)[0] + "_quality_stats.json",
            {
                "q_pair": summarize_array(save_kwargs.get("q_pair", [])),
                "ninliers": summarize_array(save_kwargs.get("ninliers", [])),
                "inlier_ratio": summarize_array(save_kwargs.get("inlier_ratio", [])),
                "gap_stats": {
                    str(int(d)): {
                        "attempts": int(gap_attempts.get(int(d), 0)),
                        "success": int(gap_success.get(int(d), 0)),
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
        save_quality_config_record(os.path.splitext(args.out_npz)[0] + "_quality_config_used.json", quality_record)


if __name__ == "__main__":
    main()

# python tools/generate_rraa_input.py \
#   --dataset custom \
#   --K_npy data/prepared/strecha/fountain-P11/K.npy \
#   --img_dir data/raw/strecha/fountain-P11/images \
#   --out_npz runs/strecha/fountain-P11/rraa_input/rraa_input_fountain-P11.npz \
#   --device cpu \
#   --deltas 1,2,3,5 \
#   --max_kpts 2048 \
#   --filter_th 0.1 \
#   --ransac_px 2.0 \
#   --add_reverse \
#   --use_h_fallback \
#   --save_meta \
#   --save_names
