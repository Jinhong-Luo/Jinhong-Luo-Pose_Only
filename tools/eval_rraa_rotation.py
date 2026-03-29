#!/usr/bin/env python3
import argparse
import json
import numpy as np


def rot_angle_deg(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R, axis1=-2, axis2=-1)
    c = (tr - 1.0) * 0.5
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def project_to_so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def align_right(R_gt: np.ndarray, R_est: np.ndarray) -> np.ndarray:
    """
    Solve min_Q sum ||R_gt - R_est Q||_F^2
    """
    M = np.zeros((3, 3), dtype=np.float64)
    for i in range(R_gt.shape[0]):
        M += R_est[i].T @ R_gt[i]
    Q = project_to_so3(M)
    return R_est @ Q[None, :, :], Q


def align_left(R_gt: np.ndarray, R_est: np.ndarray) -> np.ndarray:
    """
    Solve min_Q sum ||R_gt - Q R_est||_F^2
    """
    M = np.zeros((3, 3), dtype=np.float64)
    for i in range(R_gt.shape[0]):
        M += R_gt[i] @ R_est[i].T
    Q = project_to_so3(M)
    return Q[None, :, :] @ R_est, Q


def eval_one(name: str, R_gt: np.ndarray, R_est: np.ndarray, side: str):
    if side == "right":
        R_aligned, Q = align_right(R_gt, R_est)
    elif side == "left":
        R_aligned, Q = align_left(R_gt, R_est)
    else:
        raise ValueError(side)

    R_err = R_gt @ np.transpose(R_aligned, (0, 2, 1))
    ang = rot_angle_deg(R_err)

    return {
        "name": name,
        "gauge": side,
        "median_deg": float(np.median(ang)),
        "mean_deg": float(np.mean(ang)),
        "p90_deg": float(np.quantile(ang, 0.9)),
        "max_deg": float(np.max(ang)),
        "Q": Q,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est_npy", required=True, help="estimated global rotations (N,3,3)")
    ap.add_argument("--gt_npy", required=True, help="GT/reference global rotations (N,3,3)")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    R_est = np.load(args.est_npy).astype(np.float64)
    R_gt = np.load(args.gt_npy).astype(np.float64)

    N = min(R_est.shape[0], R_gt.shape[0])
    R_est = R_est[:N]
    R_gt = R_gt[:N]

    candidates = [
        ("est", R_est),
        ("est_T", np.transpose(R_est, (0, 2, 1))),
    ]

    reports = []
    for name, Rest in candidates:
        reports.append(eval_one(name, R_gt, Rest, "right"))
        reports.append(eval_one(name, R_gt, Rest, "left"))

    reports = sorted(reports, key=lambda x: x["median_deg"])
    best = reports[0]

    print("==== RRAA Rotation Error (multi-convention check) ====")
    print("N:", N)
    for r in reports:
        print(
            f"[{r['name']:>5s} | {r['gauge']:>5s}] "
            f"median={r['median_deg']:.6f} deg, "
            f"mean={r['mean_deg']:.6f} deg, "
            f"p90={r['p90_deg']:.6f} deg, "
            f"max={r['max_deg']:.6f} deg"
        )

    print("\nBest convention:")
    print(
        f"{best['name']} + {best['gauge']} gauge  ->  "
        f"median={best['median_deg']:.6f} deg"
    )

    if args.out_json:
        out = []
        for r in reports:
            rr = dict(r)
            rr["Q"] = rr["Q"].tolist()
            out.append(rr)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print("saved:", args.out_json)


if __name__ == "__main__":
    main()
    
# python tools/eval_rraa_rotation.py \
#   --est_npy runs/strecha/fountain-P11/rraa_output/R_abs_fountain-P11.npy \
#   --gt_npy  data/prepared/strecha/fountain-P11/R_abs_gt_w2c.npy
