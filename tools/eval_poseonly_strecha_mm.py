#!/usr/bin/env python3
import argparse
import json
import numpy as np


def read_poses_txt_12(path: str) -> np.ndarray:
    mats = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            vals = [float(x) for x in ln.split()]
            if len(vals) != 12:
                raise ValueError("Each line must have 12 floats (3x4).")
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
            mats.append(T)
    return np.stack(mats, axis=0)


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Rt = R.T
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rt
    out[:3, 3] = -(Rt @ t)
    return out


def ensure_c2w(Ts: np.ndarray, pose_type: str) -> np.ndarray:
    pose_type = pose_type.lower()
    if pose_type == "c2w":
        return Ts
    if pose_type == "w2c":
        out = np.empty_like(Ts)
        for i in range(Ts.shape[0]):
            out[i] = invert_T(Ts[i])
        return out
    raise ValueError("--est_type must be c2w or w2c")


def umeyama_alignment(X: np.ndarray, Y: np.ndarray, with_scale: bool = True):
    N = X.shape[1]
    muX = X.mean(axis=1, keepdims=True)
    muY = Y.mean(axis=1, keepdims=True)
    Xc = X - muX
    Yc = Y - muY

    Sigma = (Yc @ Xc.T) / N
    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    if with_scale:
        varX = (Xc * Xc).sum() / N
        s = float((D * np.diag(S)).sum() / max(varX, 1e-12))
    else:
        s = 1.0

    t = (muY - s * (R @ muX)).reshape(3)
    return s, R, t


def summarize(vals: np.ndarray):
    return {
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
        "rmse": float(np.sqrt(np.mean(vals ** 2))),
        "p90": float(np.quantile(vals, 0.9)),
        "max": float(np.max(vals)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est_poses", required=True, help="estimated poses txt")
    ap.add_argument("--est_type", choices=["c2w", "w2c"], default="c2w")
    ap.add_argument("--gt_centers_npy", required=True, help="GT camera centers, shape (N,3)")
    ap.add_argument("--no_scale", action="store_true", help="use SE(3) instead of Sim(3)")
    ap.add_argument(
        "--gt_unit_to_mm",
        type=float,
        default=None,
        help="Convert GT raw unit to mm. Example: 1000 if GT is in meters; 1 if already mm."
    )
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    T_est = read_poses_txt_12(args.est_poses)
    T_est = ensure_c2w(T_est, args.est_type)
    C_est = T_est[:, :3, 3]

    C_gt = np.load(args.gt_centers_npy).astype(np.float64)

    N = min(C_est.shape[0], C_gt.shape[0])
    C_est = C_est[:N]
    C_gt = C_gt[:N]

    X = C_est.T
    Y = C_gt.T
    s, R, t = umeyama_alignment(X, Y, with_scale=(not args.no_scale))
    C_aligned = (s * (R @ X) + t.reshape(3, 1)).T

    err_raw = np.linalg.norm(C_aligned - C_gt, axis=1)
    raw_stats = summarize(err_raw)

    print("==== PoseOnly Translation Accuracy (Strecha) ====")
    print("N:", int(N))
    print("use_scale:", (not args.no_scale))
    print("scale:", float(s))
    print(f"raw median = {raw_stats['median']:.6f}")
    print(f"raw mean   = {raw_stats['mean']:.6f}")
    print(f"raw rmse   = {raw_stats['rmse']:.6f}")
    print(f"raw p90    = {raw_stats['p90']:.6f}")
    print(f"raw max    = {raw_stats['max']:.6f}")

    report = {
        "N": int(N),
        "use_scale": (not args.no_scale),
        "scale": float(s),
        "raw": raw_stats,
    }

    if args.gt_unit_to_mm is not None:
        err_mm = err_raw * args.gt_unit_to_mm
        mm_stats = summarize(err_mm)
        print(f"median_mm  = {mm_stats['median']:.6f}")
        print(f"mean_mm    = {mm_stats['mean']:.6f}")
        print(f"rmse_mm    = {mm_stats['rmse']:.6f}")
        print(f"p90_mm     = {mm_stats['p90']:.6f}")
        print(f"max_mm     = {mm_stats['max']:.6f}")
        report["mm"] = mm_stats
        report["gt_unit_to_mm"] = float(args.gt_unit_to_mm)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print("saved:", args.out_json)


if __name__ == "__main__":
    main()
    
# python tools/eval_poseonly_strecha_mm.py \
#   --est_poses runs/strecha/fountain-P11/poseonly_gt/poses_c2w.txt \
#   --est_type c2w \
#   --gt_centers_npy data/prepared/strecha/fountain-P11/gt_centers.npy \
#   --gt_unit_to_mm 1000


# python tools/eval_poseonly_strecha_mm.py \
#   --est_poses runs/strecha/fountain-P11/poseonly_rraa/poses_c2w.txt \
#   --est_type c2w \
#   --gt_centers_npy data/prepared/strecha/fountain-P11/gt_centers.npy \
#   --gt_unit_to_mm 1000
