#!/usr/bin/env python3
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "runs" / "paper_v2" / "dtu_diagnostics"
CALIB_DIR = REPO_ROOT / "data" / "raw" / "DTU" / "SampleSet" / "MVS Data" / "Calibration" / "cal18"
SCANS = ["scan1", "scan6"]


def load_image_list(scan: str):
    path = REPO_ROOT / "data" / "prepared" / "DTU" / scan / "image_list.txt"
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def extract_view_id(path_text: str) -> int:
    name = Path(path_text).name
    m = re.match(r"rect_(\d{3})_(\d|max)_", name)
    if not m:
        raise ValueError(f"Unexpected DTU image name: {name}")
    return int(m.group(1))


def check_correspondence(scan: str):
    image_paths = load_image_list(scan)
    prepared = REPO_ROOT / "data" / "prepared" / "DTU" / scan
    view_ids = np.load(prepared / "dtu_view_ids.npy").astype(int)
    image_k_idx = np.load(prepared / "image_K_idx.npy").astype(int)
    Ks = np.load(prepared / "Ks.npy")
    centers = np.load(prepared / "gt_centers.npy")
    R = np.load(prepared / "R_abs_gt_w2c.npy")

    rows = []
    missing_images = []
    missing_pos = []
    mismatched_view_ids = []
    for idx, image_path in enumerate(image_paths):
        view_id = extract_view_id(image_path)
        pos_path = CALIB_DIR / f"pos_{view_id:03d}.txt"
        if not Path(image_path).is_file():
            missing_images.append(image_path)
        if not pos_path.is_file():
            missing_pos.append(str(pos_path))
        if int(view_ids[idx]) != int(view_id):
            mismatched_view_ids.append({
                "frame_idx": idx,
                "image_view_id": int(view_id),
                "prepared_view_id": int(view_ids[idx]),
            })
        rows.append({
            "frame_idx": idx,
            "image_name": Path(image_path).name,
            "view_id": int(view_id),
            "pos_file": str(pos_path.relative_to(REPO_ROOT)),
            "image_K_idx": int(image_k_idx[idx]),
            "center_x": float(centers[idx, 0]),
            "center_y": float(centers[idx, 1]),
            "center_z": float(centers[idx, 2]),
        })

    return {
        "scan": scan,
        "frame_count": len(image_paths),
        "view_id_min": int(min(extract_view_id(p) for p in image_paths)),
        "view_id_max": int(max(extract_view_id(p) for p in image_paths)),
        "view_ids_are_1_to_N": [extract_view_id(p) for p in image_paths] == list(range(1, len(image_paths) + 1)),
        "missing_images": missing_images,
        "missing_pos": missing_pos,
        "mismatched_view_ids": mismatched_view_ids,
        "Ks_shape": list(Ks.shape),
        "R_shape": list(R.shape),
        "centers_shape": list(centers.shape),
        "rows": rows,
    }


def load_pair_edges(scan: str):
    path = REPO_ROOT / "runs" / "DTU" / scan / "tracks" / "pair_quality_edges.npz"
    data = np.load(path)
    return {
        "i": data["i"].astype(int),
        "j": data["j"].astype(int),
        "q_pair": data["q_pair"].astype(float),
        "ninliers": data["ninliers"].astype(float),
        "inlier_ratio": data["inlier_ratio"].astype(float),
    }


def plot_camera_centers():
    fig = plt.figure(figsize=(11, 5))
    for ax_idx, scan in enumerate(SCANS, start=1):
        centers = np.load(REPO_ROOT / "data" / "prepared" / "DTU" / scan / "gt_centers.npy")
        ax = fig.add_subplot(1, 2, ax_idx, projection="3d")
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], "-o", ms=3, lw=1.2)
        ax.scatter(centers[0, 0], centers[0, 1], centers[0, 2], c="tab:green", s=50, label="start")
        ax.scatter(centers[-1, 0], centers[-1, 1], centers[-1, 2], c="tab:red", s=50, label="end")
        for k in [0, 12, 24, 36, 48]:
            if k < centers.shape[0]:
                ax.text(centers[k, 0], centers[k, 1], centers[k, 2], str(k + 1), fontsize=7)
        ax.set_title(f"DTU {scan}: GT camera centers")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_camera_centers_scan1_vs_scan6.png", dpi=220)
    fig.savefig(OUT_DIR / "01_camera_centers_scan1_vs_scan6.pdf")
    plt.close(fig)


def plot_pair_graphs():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    summaries = {}
    for col, scan in enumerate(SCANS):
        edges = load_pair_edges(scan)
        n = 49
        qmat = np.full((n, n), np.nan, dtype=float)
        deg = np.zeros(n, dtype=int)
        weighted_deg = np.zeros(n, dtype=float)
        for i, j, q in zip(edges["i"], edges["j"], edges["q_pair"]):
            qmat[i, j] = q
            qmat[j, i] = q
            deg[i] += 1
            deg[j] += 1
            weighted_deg[i] += q
            weighted_deg[j] += q
        im = axes[0, col].imshow(qmat, vmin=0.0, vmax=1.0, cmap="viridis")
        axes[0, col].set_title(f"{scan}: pair q heatmap")
        axes[0, col].set_xlabel("frame")
        axes[0, col].set_ylabel("frame")
        fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        x = np.arange(1, n + 1)
        axes[1, col].bar(x, deg, color="tab:blue", alpha=0.75, label="degree")
        axes[1, col].plot(x, weighted_deg, color="tab:orange", lw=1.4, label="weighted degree")
        axes[1, col].set_title(f"{scan}: graph degree")
        axes[1, col].set_xlabel("frame")
        axes[1, col].set_ylabel("degree / weighted degree")
        axes[1, col].legend(fontsize=8)

        summaries[scan] = {
            "pair_count": int(len(edges["i"])),
            "q_pair_median": float(np.median(edges["q_pair"])) if len(edges["q_pair"]) else None,
            "q_pair_p10": float(np.quantile(edges["q_pair"], 0.1)) if len(edges["q_pair"]) else None,
            "q_pair_p90": float(np.quantile(edges["q_pair"], 0.9)) if len(edges["q_pair"]) else None,
            "degree_min": int(deg.min()),
            "degree_median": float(np.median(deg)),
            "degree_max": int(deg.max()),
            "zero_degree_frames": [int(i + 1) for i, v in enumerate(deg) if v == 0],
        }
    fig.savefig(OUT_DIR / "02_pair_graph_scan1_vs_scan6.png", dpi=220)
    fig.savefig(OUT_DIR / "02_pair_graph_scan1_vs_scan6.pdf")
    plt.close(fig)
    return summaries


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    correspondence = {scan: check_correspondence(scan) for scan in SCANS}
    pair_graph = plot_pair_graphs()
    plot_camera_centers()

    for scan, payload in correspondence.items():
        rows = payload.pop("rows")
        csv_path = OUT_DIR / f"{scan}_image_pose_correspondence.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            cols = list(rows[0].keys()) if rows else []
            f.write(",".join(cols) + "\n")
            for row in rows:
                f.write(",".join(str(row[c]) for c in cols) + "\n")
        payload["correspondence_csv"] = str(csv_path.relative_to(REPO_ROOT))

    summary = {
        "correspondence": correspondence,
        "pair_graph": pair_graph,
        "outputs": [
            "runs/paper_v2/dtu_diagnostics/01_camera_centers_scan1_vs_scan6.png",
            "runs/paper_v2/dtu_diagnostics/02_pair_graph_scan1_vs_scan6.png",
        ],
    }
    with open(OUT_DIR / "dtu_scan_diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
