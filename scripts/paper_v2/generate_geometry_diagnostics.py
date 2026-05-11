#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from tools.Pose_Only_patched_v3_fixed import build_tracks_obs, choose_base_pair, compute_u, skew


SCENES_CONFIG = REPO_ROOT / "configs" / "paper_v2" / "scenes_strecha6_eth3d4_original.json"
OUT_DIR = REPO_ROOT / "runs" / "paper_v2" / "geometry_diagnostics"
MIN_TRACK_LEN = 5
BASE_PAIR_CANDIDATES = 80
BASE_PAIR_FULL_SEARCH_LEN = 50
REPRESENTATIVE_SCENES = ["fountain-P11", "Castle-P19", "office", "terrace"]
SCENE_SHORT = {
    "fountain-P11": "fountain",
    "entry-P10": "entry",
    "Herz-Jesus-P8": "Herz-P8",
    "Herz-Jesus-P25": "Herz-P25",
    "Castle-P19": "Castle-P19",
    "Castle-P30": "Castle-P30",
    "courtyard": "courtyard",
    "delivery_area": "delivery",
    "office": "office",
    "terrace": "terrace",
}
DATASET_COLORS = {"strecha": "#2b6cb0", "ETH3D": "#d94841"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig: plt.Figure, stem: str) -> None:
    ensure_dir(OUT_DIR)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_intrinsics(prepared_dir: Path):
    k_path = prepared_dir / "K.npy"
    ks_path = prepared_dir / "Ks.npy"
    idx_path = prepared_dir / "image_K_idx.npy"
    if k_path.exists():
        return np.load(k_path).astype(np.float64), None, None
    if ks_path.exists() and idx_path.exists():
        return None, np.load(ks_path).astype(np.float64), np.load(idx_path).astype(np.int64)
    raise FileNotFoundError(f"Missing intrinsics under {prepared_dir}")


def resolve_track_dir(scene_block: dict) -> Path:
    scene_id = scene_block["scene_id"]
    candidates = [
        REPO_ROOT / scene_block["scene_root"] / "tracks",
        REPO_ROOT / "runs" / "ETH3D" / scene_id / "tracks",
        REPO_ROOT / "runs" / "ETH3D_repaired" / scene_id / "tracks",
        REPO_ROOT / "runs" / "strecha" / scene_id / "tracks",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing track dir for {scene_id}: {candidates}")


def load_track_support_map(track_dir: Path) -> dict[int, float]:
    path = track_dir / "track_quality_summary.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    tids = data["track_ids"].astype(np.int64)
    counts = data["pair_q_count"].astype(np.float64) if "pair_q_count" in data.files else np.zeros_like(tids, dtype=np.float64)
    return {int(tid): float(cnt) for tid, cnt in zip(tids.tolist(), counts.tolist())}


def compute_scene_metrics(scene_block: dict) -> tuple[list[dict], dict]:
    dataset = scene_block["dataset"]
    scene = scene_block["scene_id"]
    prepared_dir = REPO_ROOT / scene_block["prepared_scene_dir"]
    scene_root = REPO_ROOT / scene_block["scene_root"]
    track_dir = resolve_track_dir(scene_block)
    r_abs = np.load(prepared_dir / "R_abs_gt_w2c.npy").astype(np.float64)
    gt_centers = np.load(prepared_dir / "gt_centers.npy").astype(np.float64)
    K, Ks, image_k_idx = load_intrinsics(prepared_dir)
    tracks_obs, npz_files = build_tracks_obs(str(track_dir), K=K, Ks=Ks, image_K_idx=image_k_idx)
    track_support = load_track_support_map(track_dir)

    rows: list[dict] = []
    for tid, obs in tracks_obs.items():
        if len(obs) < MIN_TRACK_LEN:
            continue
        obs = sorted(obs, key=lambda x: x[0])
        frames = [int(f) for f, _ in obs]
        Xs = [np.asarray(X, np.float64).reshape(3,) for _, X in obs]
        best, best_u = choose_base_pair(
            frames,
            Xs,
            r_abs,
            max_candidates=BASE_PAIR_CANDIDATES,
            full_search_len=BASE_PAIR_FULL_SEARCH_LEN,
            min_gap=0,
            max_gap=0,
            rng=np.random.default_rng(0),
        )
        if best is None:
            continue
        p_idx, q_idx = best
        f_xi, f_h = frames[p_idx], frames[q_idx]
        X_xi, X_h = Xs[p_idx], Xs[q_idx]
        u_xh = compute_u(r_abs[f_xi], r_abs[f_h], X_xi, X_h)

        local_g = []
        for k in range(len(frames)):
            f_i = frames[k]
            if f_i == f_xi or f_i == f_h:
                continue
            X_i = Xs[k]
            R_xi_i = r_abs[f_i] @ r_abs[f_xi].T
            v = R_xi_i @ X_xi
            g = float(np.linalg.norm(skew(X_i) @ v))
            local_g.append(g)

        if not local_g:
            continue

        rows.append(
            {
                "dataset": dataset,
                "scene": scene,
                "track_id": int(tid),
                "track_len": int(len(obs)),
                "frame_span": int(max(frames) - min(frames)),
                "distinct_pair_supports": float(track_support.get(int(tid), 0.0)),
                "u_xh": float(u_xh),
                "g_median": float(np.median(local_g)),
                "g_mean": float(np.mean(local_g)),
                "g_min": float(np.min(local_g)),
            }
        )

    summary = {
        "dataset": dataset,
        "scene": scene,
        "n_frames": int(len(npz_files)),
        "n_tracks_total": int(len(tracks_obs)),
        "n_tracks_used": int(len(rows)),
        "u_median": float(np.median([r["u_xh"] for r in rows])) if rows else float("nan"),
        "g_median": float(np.median([r["g_median"] for r in rows])) if rows else float("nan"),
        "frame_span_median": float(np.median([r["frame_span"] for r in rows])) if rows else float("nan"),
        "distinct_pair_supports_median": float(np.median([r["distinct_pair_supports"] for r in rows])) if rows else float("nan"),
        "camera_centers_path": str(prepared_dir / "gt_centers.npy"),
        "gt_centers": gt_centers,
    }
    return rows, summary


def boxplot_by_scene(ax, rows: list[dict], metric_key: str, title: str, ylabel: str) -> None:
    scenes = [s["scene_id"] for s in load_json(SCENES_CONFIG)["scenes"]]
    vals = []
    labels = []
    colors = []
    for scene in scenes:
        sub = [r[metric_key] for r in rows if r["scene"] == scene and np.isfinite(r[metric_key])]
        if not sub:
            continue
        vals.append(sub)
        labels.append(SCENE_SHORT.get(scene, scene))
        dataset = next(r["dataset"] for r in rows if r["scene"] == scene)
        colors.append(DATASET_COLORS[dataset])
    bp = ax.boxplot(vals, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(labels, rotation=15)
    ax.grid(axis="y", alpha=0.2)


def plot_u_distribution(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    strecha_rows = [r for r in rows if r["dataset"] == "strecha"]
    eth_rows = [r for r in rows if r["dataset"] == "ETH3D"]
    boxplot_by_scene(axes[0], strecha_rows, "u_xh", "Strecha: base-pair u_xh distribution", "u_xh")
    boxplot_by_scene(axes[1], eth_rows, "u_xh", "ETH3D: base-pair u_xh distribution", "u_xh")
    save_fig(fig, "01_u_distribution_by_dataset")


def plot_g_distribution(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    strecha_rows = [r for r in rows if r["dataset"] == "strecha"]
    eth_rows = [r for r in rows if r["dataset"] == "ETH3D"]
    boxplot_by_scene(axes[0], strecha_rows, "g_median", "Strecha: per-track g median distribution", "g median")
    boxplot_by_scene(axes[1], eth_rows, "g_median", "ETH3D: per-track g median distribution", "g median")
    save_fig(fig, "02_g_distribution_by_dataset")


def plot_track_span_support(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    boxplot_by_scene(axes[0], rows, "frame_span", "Track frame span by scene", "frame span")
    boxplot_by_scene(axes[1], rows, "distinct_pair_supports", "Track distinct pair supports by scene", "distinct pair supports")
    save_fig(fig, "03_track_span_and_support")


def plot_camera_centers(scene_summaries: list[dict]) -> None:
    picked = [s for s in scene_summaries if s["scene"] in REPRESENTATIVE_SCENES]
    fig = plt.figure(figsize=(12.5, 9.5))
    for idx, summary in enumerate(picked, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        centers = np.asarray(summary["gt_centers"], dtype=np.float64)
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color=DATASET_COLORS[summary["dataset"]], linewidth=1.4)
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=np.arange(len(centers)), cmap="viridis", s=18)
        ax.set_title(f"{summary['scene']} ({summary['dataset']})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.suptitle("Representative GT camera-center layouts", y=0.98)
    save_fig(fig, "04_camera_centers_representatives")


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ensure_dir(OUT_DIR)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 9})

    scenes_doc = load_json(SCENES_CONFIG)
    all_rows: list[dict] = []
    scene_summaries: list[dict] = []
    summary_json = []

    for scene_block in scenes_doc["scenes"]:
        rows, summary = compute_scene_metrics(scene_block)
        all_rows.extend(rows)
        scene_summaries.append(summary)
        summary_json.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in summary.items()})
        print(f"[diag] scene={summary['scene']} dataset={summary['dataset']} tracks_used={summary['n_tracks_used']}")

    write_csv(
        all_rows,
        OUT_DIR / "geometry_track_metrics.csv",
        ["dataset", "scene", "track_id", "track_len", "frame_span", "distinct_pair_supports", "u_xh", "g_median", "g_mean", "g_min"],
    )
    write_csv(
        [
            {
                "dataset": s["dataset"],
                "scene": s["scene"],
                "n_frames": s["n_frames"],
                "n_tracks_total": s["n_tracks_total"],
                "n_tracks_used": s["n_tracks_used"],
                "u_median": s["u_median"],
                "g_median": s["g_median"],
                "frame_span_median": s["frame_span_median"],
                "distinct_pair_supports_median": s["distinct_pair_supports_median"],
            }
            for s in scene_summaries
        ],
        OUT_DIR / "geometry_scene_summary.csv",
        ["dataset", "scene", "n_frames", "n_tracks_total", "n_tracks_used", "u_median", "g_median", "frame_span_median", "distinct_pair_supports_median"],
    )
    (OUT_DIR / "geometry_scene_summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_u_distribution(all_rows)
    plot_g_distribution(all_rows)
    plot_track_span_support(all_rows)
    plot_camera_centers(scene_summaries)
    print(f"[diag] saved geometry diagnostics to: {OUT_DIR}")


if __name__ == "__main__":
    main()
