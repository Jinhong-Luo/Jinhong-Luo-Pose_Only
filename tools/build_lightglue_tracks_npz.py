#!/usr/bin/env python3
# build_lightglue_tracks_npz_v2.py
# Only build multi-frame tracks (per-frame npz: track_ids + xy). No Rij output.

import os, glob, argparse
import numpy as np
import _bootstrap  # noqa: F401
from calib_utils import load_intrinsics

def read_gray_u8(path):
    from PIL import Image
    img = Image.open(path).convert("L")
    return np.array(img)

def to_torch_image_u8(img_u8, device):
    import torch
    t = torch.from_numpy(img_u8).to(device=device)
    t = t.float() / 255.0
    return t[None, None]  # (1,1,H,W)

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


def resolve_device(device_arg):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] --device cuda requested but CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(device_arg)


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
    ap.add_argument("--kitti_calib", type=str, default=None, help="KITTI calib.txt")
    ap.add_argument("--kitti_cam", type=str, default="P0", help="P0/P2 ...")
    ap.add_argument("--euroc_yaml", type=str, default=None, help="EuRoC cam sensor.yaml")
    ap.add_argument("--undistort", action="store_true", help="undistort keypoints before saving xy (EuRoC)")

    # optional: seed all keypoints with unique tid (bigger output, can help later post-process)
    ap.add_argument("--seed_all", action="store_true",
                    help="assign a track id to every keypoint each frame (larger files)")

    args = ap.parse_args()

    imgs = sorted(glob.glob(args.image_glob))
    if not imgs:
        raise FileNotFoundError(args.image_glob)
    if args.max_frames is not None:
        imgs = imgs[:args.max_frames]
    N = len(imgs)
    deltas = parse_deltas(args.deltas)
    print("[Tracks] frames:", N)
    print("[Tracks] deltas:", deltas)

    # load intrinsics if needed
    K, dist, dist_model = load_intrinsics(
        dataset=args.dataset,
        kitti_calib=args.kitti_calib,
        kitti_cam=args.kitti_cam,
        euroc_yaml=args.euroc_yaml,
        K_npy=args.K_npy,
    )

    if K is not None:
        print("[K]\n", K)
    if dist is not None:
        print("[DIST]", dist, "model=", dist_model)

    if args.use_ransac and K is None:
        raise ValueError("--use_ransac needs intrinsics K (set --dataset kitti/euroc/custom and calib args)")

    # LightGlue
    import torch
    from lightglue import SuperPoint, LightGlue

    dev = resolve_device(args.device)
    print("[Device]", dev)

    extractor = SuperPoint(max_num_keypoints=args.max_kpts).eval().to(dev)
    matcher = LightGlue(features="superpoint", filter_threshold=args.filter_th).eval().to(dev)

    feats_cache = []
    kpts_save_all = []
    n_kpts_per_frame = []
    for i, img_path in enumerate(imgs):
        img = read_gray_u8(img_path)
        t = to_torch_image_u8(img, dev)
        with torch.no_grad():
            feats = extractor.extract(t)
        kpts = feats["keypoints"][0].detach().cpu().numpy()
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
    pair_stats = []

    for i in range(N):
        feats0 = feats_cache[i]
        kpts0_save = kpts_save_all[i]
        for d in deltas:
            j = i + d
            if j >= N:
                continue
            feats1 = feats_cache[j]
            kpts1_save = kpts_save_all[j]
            with torch.no_grad():
                out = matcher({"image0": feats0, "image1": feats1})

            m0 = out["matches0"][0].detach().cpu().numpy().astype(np.int64)
            valid = (m0 >= 0)

            if args.mutual and ("matches1" in out):
                m1 = out["matches1"][0].detach().cpu().numpy().astype(np.int64)
                idx0_all = np.arange(m0.shape[0], dtype=np.int64)
                jj = m0.copy()
                ok = valid.copy()
                ok[valid] &= (m1[jj[valid]] == idx0_all[valid])
                valid &= ok

            if args.min_score > 0 and ("matching_scores0" in out):
                s0 = out["matching_scores0"][0].detach().cpu().numpy()
                valid &= (s0 >= args.min_score)

            idx0 = np.where(valid)[0]
            idx1 = m0[idx0]

            if args.use_ransac:
                import cv2
                if len(idx0) >= 8:
                    pts0 = kpts0_save[idx0].astype(np.float64)
                    pts1 = kpts1_save[idx1].astype(np.float64)
                    E, mask = cv2.findEssentialMat(
                        pts0, pts1, K,
                        method=cv2.RANSAC, prob=0.999, threshold=args.ransac_thresh
                    )
                    if mask is None:
                        idx0 = idx0[:0]
                        idx1 = idx1[:0]
                    else:
                        inl = mask.reshape(-1).astype(bool)
                        if int(inl.sum()) >= args.min_inliers:
                            idx0 = idx0[inl]
                            idx1 = idx1[inl]
                        else:
                            idx0 = idx0[:0]
                            idx1 = idx1[:0]
                else:
                    idx0 = idx0[:0]
                    idx1 = idx1[:0]

            merged = 0
            conflicts = 0
            for a, b in zip(idx0.tolist(), idx1.tolist()):
                ga = int(offsets[i] + a)
                gb = int(offsets[j] + b)
                if dsu.union_if_compatible(ga, gb):
                    merged += 1
                else:
                    conflicts += 1
            pair_stats.append((i, j, d, len(idx0), merged, conflicts))

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

    if pair_stats:
        kept = [x[3] for x in pair_stats]
        merged = [x[4] for x in pair_stats]
        conflicts = [x[5] for x in pair_stats]
        print(f"[Tracks] pair count={len(pair_stats)}, kept matches median={np.median(kept):.1f}, merged median={np.median(merged):.1f}, conflicts total={int(np.sum(conflicts))}")
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
