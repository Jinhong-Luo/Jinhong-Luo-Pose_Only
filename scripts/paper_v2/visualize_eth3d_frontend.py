#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "runs" / "paper_v2" / "eth3d_frontend_viz"
DEFAULT_SCENES = ["courtyard", "delivery_area", "office", "terrace", "electro", "facade"]


def maybe_scene_root(scene: str) -> Path | None:
    candidates = [
        REPO_ROOT / "runs" / "ETH3D" / scene,
        REPO_ROOT / "runs" / "ETH3D_repaired" / scene,
    ]
    for path in candidates:
        if (path / "tracks" / "pair_quality_edges.npz").exists():
            return path
    return None


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def frame_npz_paths(track_dir: Path) -> list[Path]:
    return sorted([p for p in track_dir.glob("*.npz") if p.stem.isdigit()])


def compute_components(n_frames: int, edges_i: np.ndarray, edges_j: np.ndarray, edge_mask: np.ndarray) -> list[list[int]]:
    graph = [[] for _ in range(n_frames)]
    for i, j, keep in zip(edges_i.tolist(), edges_j.tolist(), edge_mask.tolist()):
        if not keep:
            continue
        graph[i].append(j)
        graph[j].append(i)

    seen = [False] * n_frames
    comps: list[list[int]] = []
    for start in range(n_frames):
        if seen[start]:
            continue
        q: deque[int] = deque([start])
        seen[start] = True
        comp: list[int] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in graph[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    q.append(nxt)
        comps.append(sorted(comp))
    comps.sort(key=len, reverse=True)
    return comps


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_scene_report(scene: str, scene_root: Path, out_root: Path) -> None:
    track_dir = scene_root / "tracks"
    edge_npz = np.load(track_dir / "pair_quality_edges.npz")
    stats = load_json(track_dir / "track_build_quality_stats.json")
    npz_paths = frame_npz_paths(track_dir)
    n_frames = len(npz_paths)

    i = edge_npz["i"].astype(np.int64)
    j = edge_npz["j"].astype(np.int64)
    q_pair = edge_npz["q_pair"].astype(np.float64)
    ninliers = edge_npz["ninliers"].astype(np.int64)
    inlier_ratio = edge_npz["inlier_ratio"].astype(np.float64)
    delta = edge_npz["delta"].astype(np.int64)

    pair_exists = ninliers > 0
    comps = compute_components(n_frames, i, j, pair_exists)
    largest_ratio = len(comps[0]) / max(n_frames, 1) if comps else 0.0

    qmat = np.full((n_frames, n_frames), np.nan, dtype=np.float64)
    nmat = np.full((n_frames, n_frames), np.nan, dtype=np.float64)
    for ii, jj, qq, nn in zip(i.tolist(), j.tolist(), q_pair.tolist(), ninliers.tolist()):
        qmat[ii, jj] = qq
        qmat[jj, ii] = qq
        nmat[ii, jj] = nn
        nmat[jj, ii] = nn

    degree = np.zeros((n_frames,), dtype=np.int64)
    weighted_degree = np.zeros((n_frames,), dtype=np.float64)
    for ii, jj, nn, qq in zip(i.tolist(), j.tolist(), ninliers.tolist(), q_pair.tolist()):
        if nn > 0:
            degree[ii] += 1
            degree[jj] += 1
            weighted_degree[ii] += qq
            weighted_degree[jj] += qq

    obs_counts = []
    for path in npz_paths:
        data = np.load(path)
        obs_counts.append(int(len(data["track_ids"])))
    obs_counts = np.asarray(obs_counts, dtype=np.int64)

    out_dir = out_root / scene

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    valid = pair_exists
    sc = ax.scatter(
        i[valid],
        j[valid],
        c=q_pair[valid],
        s=np.clip(ninliers[valid] * 1.4, 18, 180),
        cmap="viridis",
        alpha=0.88,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.scatter(i[~valid], j[~valid], c="#c7c7c7", s=16, alpha=0.55, label="ninliers = 0")
    ax.set_title(f"{scene}: Verified Pair Graph Scatter")
    ax.set_xlabel("frame i")
    ax.set_ylabel("frame j")
    ax.grid(alpha=0.18)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.86)
    cbar.set_label("q_pair")
    ax.legend(frameon=False, loc="upper right")
    save_fig(fig, out_dir, "01_pair_scatter")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    im0 = axes[0].imshow(qmat, cmap="viridis")
    axes[0].set_title(f"{scene}: q_pair heatmap")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("frame")
    fig.colorbar(im0, ax=axes[0], shrink=0.84, label="q_pair")

    show_nmat = np.where(np.isnan(nmat), np.nan, np.log10(np.maximum(nmat, 1.0)))
    im1 = axes[1].imshow(show_nmat, cmap="magma")
    axes[1].set_title(f"{scene}: ninliers heatmap (log10)")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("frame")
    fig.colorbar(im1, ax=axes[1], shrink=0.84, label="log10(max(ninliers,1))")
    save_fig(fig, out_dir, "02_pair_heatmaps")

    fig, ax1 = plt.subplots(figsize=(12.8, 5.0))
    x = np.arange(n_frames)
    ax1.bar(x, degree, color="#2b6cb0", alpha=0.88, label="positive-pair degree")
    ax1.axhline(np.median(degree), color="#d94841", linestyle="--", linewidth=1.5, label=f"degree median={np.median(degree):.1f}")
    ax1.set_xlabel("frame")
    ax1.set_ylabel("verified neighbor count")
    ax1.grid(axis="y", alpha=0.22)
    ax2 = ax1.twinx()
    ax2.plot(x, obs_counts, color="#e36414", marker="o", linewidth=1.6, label="track observations")
    ax2.set_ylabel("frame observations")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")
    ax1.set_title(f"{scene}: Per-frame connectivity and observation count")
    save_fig(fig, out_dir, "03_frame_degree_and_observations")

    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    ax.bar(x, weighted_degree, color="#6a994e", alpha=0.88)
    low = np.where(degree <= 1)[0]
    if len(low) > 0:
        ax.scatter(low, weighted_degree[low], color="#d94841", s=45, zorder=3, label="degree <= 1")
    ax.set_xlabel("frame")
    ax.set_ylabel("sum q_pair over verified neighbors")
    ax.set_title(f"{scene}: Weighted frame connectivity")
    ax.grid(axis="y", alpha=0.22)
    if len(low) > 0:
        ax.legend(frameon=False)
    save_fig(fig, out_dir, "04_weighted_degree")

    summary_path = out_dir / "frontend_graph_summary.json"
    summary = {
        "scene": scene,
        "scene_root": str(scene_root),
        "n_frames": n_frames,
        "pair_count": int(len(i)),
        "positive_pair_count": int(pair_exists.sum()),
        "positive_pair_ratio": float(pair_exists.mean()) if len(pair_exists) else 0.0,
        "components": comps,
        "component_count": len(comps),
        "largest_component_ratio": largest_ratio,
        "degree_median": float(np.median(degree)) if len(degree) else 0.0,
        "degree_min": int(degree.min()) if len(degree) else 0,
        "degree_max": int(degree.max()) if len(degree) else 0,
        "low_degree_frames": [int(v) for v in np.where(degree <= 1)[0].tolist()],
        "zero_degree_frames": [int(v) for v in np.where(degree == 0)[0].tolist()],
        "track_build_gap_stats": stats.get("gap_stats", {}),
        "pair_stats": stats.get("pair_stats", {}),
        "track_stats": stats.get("track_stats", {}),
        "delta_summary": {
            "attempted_unique": sorted({int(v) for v in delta.tolist()}),
            "positive_counts": {str(int(d)): int(np.sum((delta == d) & pair_exists)) for d in sorted(set(delta.tolist()))},
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ETH3D frontend pair graph and matching health.")
    parser.add_argument("--scenes", type=str, default=",".join(DEFAULT_SCENES), help="Comma-separated ETH3D scenes.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    missing = []
    for scene in scenes:
        root = maybe_scene_root(scene)
        if root is None:
            missing.append(scene)
            continue
        print(f"[viz] scene={scene} scene_root={root}")
        make_scene_report(scene, root, out_root)

    if missing:
        print(f"[viz] skipped scenes with no frontend tracks: {missing}")
    print(f"[viz] saved ETH3D frontend visualizations to: {out_root}")


if __name__ == "__main__":
    main()
