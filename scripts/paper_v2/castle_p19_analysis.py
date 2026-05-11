#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from tools.Pose_Only_patched_v3_fixed import build_tracks_obs, choose_base_pair


SCENE_ID = "Castle-P19"
TRACK_DIR = REPO_ROOT / "runs" / "strecha" / SCENE_ID / "tracks"
PREPARED_DIR = REPO_ROOT / "data" / "prepared" / "strecha" / SCENE_ID
OUTPUT_ROOT = REPO_ROOT / "runs" / "paper_v2" / "castle_p19_analysis"
VARIANT_ROOT = OUTPUT_ROOT / "threshold_ablation"
QUALITY_CONFIG = REPO_ROOT / "configs" / "quality_config.template.json"
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

K_NPY = PREPARED_DIR / "K.npy"
R_ABS_GTS = PREPARED_DIR / "R_abs_gt_w2c.npy"
GT_POSES = PREPARED_DIR / "gt_poses_c2w.txt"
GT_CENTERS = PREPARED_DIR / "gt_centers.npy"

VARIANTS = [
    {"name": "baseline", "u_min": 1e-3, "g_min": 1e-3, "qtrack_threshold": 0.0},
    {"name": "ug_1e-2", "u_min": 1e-2, "g_min": 1e-2, "qtrack_threshold": 0.0},
    {"name": "qtrack_055", "u_min": 1e-3, "g_min": 1e-3, "qtrack_threshold": 0.55},
    {"name": "qtrack_060", "u_min": 1e-3, "g_min": 1e-3, "qtrack_threshold": 0.60},
    {"name": "qtrack_065", "u_min": 1e-3, "g_min": 1e-3, "qtrack_threshold": 0.65},
    {"name": "ug_1e-2_qtrack_060", "u_min": 1e-2, "g_min": 1e-2, "qtrack_threshold": 0.60},
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, stem: str) -> None:
    ensure_dir(OUTPUT_ROOT)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_ROOT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_frame_tracks() -> tuple[list[Path], list[np.ndarray], list[np.ndarray]]:
    npz_paths = sorted([p for p in TRACK_DIR.glob("*.npz") if p.stem.isdigit()])
    ids_list: list[np.ndarray] = []
    xy_list: list[np.ndarray] = []
    for path in npz_paths:
        data = np.load(path)
        ids_list.append(data["track_ids"].astype(np.int64))
        xy_list.append(data["xy"].astype(np.float64))
    return npz_paths, ids_list, xy_list


def compute_global_bounds(xy_list: list[np.ndarray], K: np.ndarray) -> tuple[float, float]:
    max_x = max(float(xy[:, 0].max()) for xy in xy_list if len(xy) > 0)
    max_y = max(float(xy[:, 1].max()) for xy in xy_list if len(xy) > 0)
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    width = max(max_x, 2.0 * cx)
    height = max(max_y, 2.0 * cy)
    return width, height


def plot_frame_track_counts(npz_paths: list[Path], ids_list: list[np.ndarray]) -> None:
    counts = [len(ids) for ids in ids_list]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(counts))
    ax.bar(x, counts, color="#2b6cb0", alpha=0.9)
    ax.axhline(np.median(counts), color="#d94841", linestyle="--", linewidth=1.5, label=f"median={np.median(counts):.0f}")
    ax.set_xticks(x)
    ax.set_xticklabels([p.stem for p in npz_paths], rotation=0)
    ax.set_ylabel("track count")
    ax.set_xlabel("frame")
    ax.set_title("Castle-P19: Tracks Participating per Frame")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_fig(fig, "01_frame_track_counts")


def occupancy_ratio(xy: np.ndarray, width: float, height: float, nx: int = 8, ny: int = 6) -> float:
    if len(xy) == 0:
        return 0.0
    xbins = np.linspace(0.0, width, nx + 1)
    ybins = np.linspace(0.0, height, ny + 1)
    hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xbins, ybins])
    return float(np.count_nonzero(hist) / hist.size)


def plot_spatial_coverage(npz_paths: list[Path], xy_list: list[np.ndarray], width: float, height: float) -> None:
    ratios = [occupancy_ratio(xy, width, height) for xy in xy_list]
    counts = [len(xy) for xy in xy_list]
    fig, ax1 = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(xy_list))
    ax1.bar(x, ratios, color="#6a994e", alpha=0.88, label="grid occupancy")
    ax1.set_ylabel("occupancy ratio (8x6 grid)")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.stem for p in npz_paths], rotation=0)
    ax1.set_xlabel("frame")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, counts, color="#e36414", marker="o", linewidth=1.8, label="track count")
    ax2.set_ylabel("track count")
    ax1.set_title("Castle-P19: Per-Frame Spatial Coverage and Track Count")
    lines, labels = [], []
    for ax in [ax1, ax2]:
        ll, lb = ax.get_legend_handles_labels()
        lines.extend(ll)
        labels.extend(lb)
    ax1.legend(lines, labels, frameon=False, loc="upper right")
    save_fig(fig, "02_frame_spatial_coverage")


def plot_feature_montage(npz_paths: list[Path], xy_list: list[np.ndarray], width: float, height: float) -> None:
    n = len(xy_list)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 2.8 * nrows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, path, xy in zip(axes, npz_paths, xy_list):
        if len(xy) > 0:
            ax.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.45, color="#0f4c5c")
        ax.set_title(path.stem, fontsize=9)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.grid(alpha=0.12)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Castle-P19: Per-Frame Feature Spatial Distribution", y=1.02)
    save_fig(fig, "03_frame_feature_scatter_montage")


def plot_pair_quality_heatmap(n_frames: int) -> None:
    edge_npz = np.load(TRACK_DIR / "pair_quality_edges.npz")
    qmat = np.full((n_frames, n_frames), np.nan, dtype=np.float64)
    for i, j, q in zip(edge_npz["i"], edge_npz["j"], edge_npz["q_pair"]):
        qmat[int(i), int(j)] = float(q)
        qmat[int(j), int(i)] = float(q)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(qmat, cmap="viridis", vmin=np.nanmin(qmat), vmax=np.nanmax(qmat))
    ax.set_title("Castle-P19: Pair Quality Heatmap")
    ax.set_xlabel("frame j")
    ax.set_ylabel("frame i")
    fig.colorbar(im, ax=ax, shrink=0.88, label="q_pair")
    save_fig(fig, "04_pair_quality_heatmap")


def plot_base_pair_usage(K: np.ndarray, R_abs: np.ndarray) -> None:
    tracks_obs, npz_files = build_tracks_obs(str(TRACK_DIR), K=K)
    n_frames = len(npz_files)
    usage = np.zeros((n_frames, n_frames), dtype=np.int32)
    delta_counter: Counter[int] = Counter()
    per_frame_usage = np.zeros((n_frames,), dtype=np.int32)

    for obs in tracks_obs.values():
        if len(obs) < 5:
            continue
        obs = sorted(obs, key=lambda item: item[0])
        frames = [int(f) for f, _ in obs]
        Xs = [np.asarray(X, np.float64).reshape(3,) for _, X in obs]
        best, _ = choose_base_pair(
            frames,
            Xs,
            R_abs,
            max_candidates=80,
            full_search_len=50,
            min_gap=0,
            max_gap=0,
            rng=np.random.default_rng(0),
        )
        if best is None:
            continue
        p_idx, q_idx = best
        fi, fj = frames[p_idx], frames[q_idx]
        if fi > fj:
            fi, fj = fj, fi
        usage[fi, fj] += 1
        usage[fj, fi] += 1
        per_frame_usage[fi] += 1
        per_frame_usage[fj] += 1
        delta_counter[int(abs(fj - fi))] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    im = axes[0].imshow(usage, cmap="magma")
    axes[0].set_title("Castle-P19: Selected Base-Pair Usage")
    axes[0].set_xlabel("frame j")
    axes[0].set_ylabel("frame i")
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="track count")

    deltas = sorted(delta_counter.keys())
    vals = [delta_counter[d] for d in deltas]
    axes[1].bar(deltas, vals, color="#bc4749", alpha=0.9)
    axes[1].set_title("Castle-P19: Base-Pair Frame Gap Distribution")
    axes[1].set_xlabel("|i - j|")
    axes[1].set_ylabel("selected track count")
    axes[1].grid(axis="y", alpha=0.25)
    save_fig(fig, "05_base_pair_usage")

    with (OUTPUT_ROOT / "base_pair_usage_per_frame.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "base_pair_selected_count"])
        for idx, count in enumerate(per_frame_usage.tolist()):
            writer.writerow([idx, count])


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_threshold_ablation() -> list[dict]:
    ensure_dir(VARIANT_ROOT)
    summary_rows: list[dict] = []

    for variant in VARIANTS:
        name = variant["name"]
        out_dir = VARIANT_ROOT / name / "pose_only"
        ensure_dir(out_dir)

        pose_cmd = [
            str(PYTHON_EXE),
            "tools\\Pose_Only_patched_v3_fixed.py",
            "--track_npz_dir", str(TRACK_DIR),
            "--r_abs_npy", str(R_ABS_GTS),
            "--dataset", "custom",
            "--K_npy", str(K_NPY),
            "--gt_pose_txt", str(GT_POSES),
            "--out_dir", str(out_dir),
            "--min_track_len", "5",
            "--max_tracks", "20000",
            "--u_min", str(variant["u_min"]),
            "--g_min", str(variant["g_min"]),
            "--base_pair_candidates", "80",
            "--base_pair_full_search_len", "50",
            "--irls_iters", "2",
            "--qtrack_mode", "weighted_sum",
            "--qtrack_threshold", str(variant["qtrack_threshold"]),
            "--quality_config", str(QUALITY_CONFIG),
            "--auto_quality_refs",
            "--dump_quality_stats",
            "--dump_degeneracy_stats",
            "--enable_quality_weighting",
            "--ligt_use_qtrack_weight",
        ]
        run(pose_cmd)

        eval_json = out_dir / "eval_translation.json"
        run([
            str(PYTHON_EXE),
            "tools\\eval_poseonly_strecha_mm.py",
            "--est_poses", str(out_dir / "poses_c2w.txt"),
            "--est_type", "c2w",
            "--gt_centers_npy", str(GT_CENTERS),
            "--gt_unit_to_mm", "1000",
            "--out_json", str(eval_json),
        ])

        deg = json.loads((out_dir / "ligt_degeneracy_stats.json").read_text(encoding="utf-8"))
        eva = json.loads(eval_json.read_text(encoding="utf-8"))
        summary_rows.append({
            "variant": name,
            "u_min": variant["u_min"],
            "g_min": variant["g_min"],
            "qtrack_threshold": variant["qtrack_threshold"],
            "tracks_kept": deg["tracks_kept"],
            "equations_kept": deg["equations_kept"],
            "g_reject_ratio": deg["g_reject_ratio"],
            "qtrack_reject_ratio": deg["qtrack_reject_ratio"],
            "kept_track_ratio": deg["kept_track_ratio"],
            "translation_mm_median": eva["mm"]["median"],
            "translation_mm_p90": eva["mm"]["p90"],
            "translation_mm_max": eva["mm"]["max"],
        })

    return summary_rows


def write_summary_csv(rows: list[dict]) -> None:
    ensure_dir(OUTPUT_ROOT)
    path = OUTPUT_ROOT / "threshold_ablation_summary.csv"
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_threshold_ablation(rows: list[dict]) -> None:
    names = [row["variant"] for row in rows]
    vals = [row["translation_mm_median"] for row in rows]
    kept = [row["tracks_kept"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.6), sharex=True)
    x = np.arange(len(rows))
    axes[0].bar(x, vals, color="#4361ee", alpha=0.9)
    axes[0].set_ylabel("translation_mm_median (mm)")
    axes[0].set_title("Castle-P19: Threshold Ablation")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, kept, color="#2a9d8f", alpha=0.9)
    axes[1].set_ylabel("tracks_kept")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    save_fig(fig, "06_threshold_ablation")


def main() -> None:
    ensure_dir(OUTPUT_ROOT)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 9})

    K = np.load(K_NPY).astype(np.float64)
    R_abs = np.load(R_ABS_GTS).astype(np.float64)
    npz_paths, ids_list, xy_list = load_frame_tracks()
    width, height = compute_global_bounds(xy_list, K)

    plot_frame_track_counts(npz_paths, ids_list)
    plot_spatial_coverage(npz_paths, xy_list, width, height)
    plot_feature_montage(npz_paths, xy_list, width, height)
    plot_pair_quality_heatmap(len(npz_paths))
    plot_base_pair_usage(K, R_abs)

    rows = run_threshold_ablation()
    write_summary_csv(rows)
    plot_threshold_ablation(rows)

    print(f"saved Castle-P19 analysis to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
