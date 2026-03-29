#!/usr/bin/env python3
import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import torch

import _bootstrap  # noqa: F401
from calib_utils import load_intrinsics, list_images_sorted
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd


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


def get_pair_scores(matches_dict: dict, matches: np.ndarray):
    """
    Try to extract a per-match confidence score aligned with matches (M,2).
    Returns: scores (M,) or None
    """
    if matches is None or len(matches) == 0:
        return None

    # case 1: matching_scores0 is per-kpt score on image0: (K0,)
    if "matching_scores0" in matches_dict:
        s0 = to_numpy_cpu(matches_dict["matching_scores0"]).reshape(-1)
        return s0[matches[:, 0]]

    # case 2: scores already per-match: (M,)
    for key in ["scores", "matching_scores", "mscores"]:
        if key in matches_dict:
            s = to_numpy_cpu(matches_dict[key]).reshape(-1)
            if s.shape[0] == matches.shape[0]:
                return s

    return None


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
    parser.add_argument("--img_dir", required=True)
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
        default="1:10,2:20,3:30,5:40,8:60,13:60,default:60",
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

    args = parser.parse_args()

    # expand paths safely
    args.img_dir = os.path.abspath(os.path.expanduser(args.img_dir))
    args.out_npz = os.path.abspath(os.path.expanduser(args.out_npz))
    if args.K_npy is not None:
        args.K_npy = os.path.abspath(os.path.expanduser(args.K_npy))
    if args.kitti_calib is not None:
        args.kitti_calib = os.path.abspath(os.path.expanduser(args.kitti_calib))
    if args.euroc_yaml is not None:
        args.euroc_yaml = os.path.abspath(os.path.expanduser(args.euroc_yaml))

    # choose device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        device = "cuda"
    else:
        device = "cpu"

    print("[Device]", device)

    image_paths = list_images_sorted(args.img_dir)
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
    K, dist, dist_model = load_intrinsics(
        dataset=args.dataset,
        kitti_calib=args.kitti_calib,
        kitti_cam=args.kitti_cam,
        euroc_yaml=args.euroc_yaml,
        K_npy=args.K_npy,
    )
    print("[K]\n", K)

    # LightGlue / SuperPoint
    extractor = SuperPoint(max_num_keypoints=args.max_kpts).eval().to(device)
    matcher = LightGlue(features="superpoint", filter_threshold=args.filter_th).eval().to(device)

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
            img = load_image(image_paths[idx]).to(device)
            f = extractor.extract(img)
            feat_cache[idx] = f
        return f

    pair_count = 0

    for i in tqdm(range(N), desc="frames"):
        feats0 = get_feats(i)

        for d in deltas:
            j = i + d
            if j >= N:
                continue

            feats1 = get_feats(j)

            out = matcher({"image0": feats0, "image1": feats1})
            feats0r, feats1r, outr = [rbd(x) for x in [feats0, feats1, out]]

            matches_t = outr.get("matches", None)
            if matches_t is None:
                continue

            matches = to_numpy_cpu(matches_t).astype(np.int64)
            if matches.shape[0] < 8:
                continue

            # optional score-based prefilter
            pair_scores = get_pair_scores(outr, matches)
            if pair_scores is not None and args.min_match_score > 0:
                keep = pair_scores >= args.min_match_score
                matches = matches[keep]
                if matches.shape[0] < 8:
                    continue
                pair_scores = pair_scores[keep]

            kpts0 = to_numpy_cpu(feats0r["keypoints"])
            kpts1 = to_numpy_cpu(feats1r["keypoints"])

            pts0 = kpts0[matches[:, 0]]
            pts1 = kpts1[matches[:, 1]]

            min_inl = min_inliers_for_delta(d, mp, default_inl)

            if args.dataset == "euroc" and args.euroc_undistort:
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

    if args.save_meta:
        ninl = np.array(ninl_list, dtype=np.int32)
        nm = np.array(nmatch_list, dtype=np.int32)
        delta_arr = np.array(delta_list, dtype=np.int16)
        method_arr = np.array(method_list, dtype=np.int8)
        inlier_ratio = ninl.astype(np.float32) / np.maximum(nm.astype(np.float32), 1.0)
        score_med = np.array(score_med_list, dtype=np.float32)
        score_mean = np.array(score_mean_list, dtype=np.float32)

        save_kwargs.update(
            ninliers=ninl,
            nmatches=nm,
            inlier_ratio=inlier_ratio,
            delta=delta_arr,
            method=method_arr,
            score_med=score_med,
            score_mean=score_mean,
        )

    if args.save_names:
        names = np.array([os.path.basename(p) for p in image_paths], dtype="S")
        save_kwargs.update(img_names=names)
        if K is not None:
            save_kwargs.update(K=K.astype(np.float64))

    save_kwargs.update(config=str(vars(args)))

    out_dir = os.path.dirname(args.out_npz)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out_npz, **save_kwargs)

    print(f"[OK] saved {args.out_npz}")
    print(f"     N={N}, M={len(edges)}")
    if args.save_meta:
        print("     meta saved: ninliers/nmatches/inlier_ratio/delta/method/score_*")
    if args.save_names:
        print("     names saved: img_names (+K if available)")


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
