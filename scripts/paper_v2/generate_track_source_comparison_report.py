#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "runs" / "paper_v2" / "track_source_comparison_report"

ORIGINAL_CSV = REPO_ROOT / "runs" / "paper_v2" / "eval_rerun_original_tracks" / "paper_summary_main.csv"
COLMAP_TRACKS_CSV = REPO_ROOT / "runs" / "paper_v2" / "eval_rerun_colmap_tracks" / "paper_summary_main.csv"
COLMAP_STRECHA_CSV = REPO_ROOT / "runs" / "paper_v2" / "colmap_compare_strecha6" / "colmap_strecha6_summary.csv"
COLMAP_ETH3D_CSV = REPO_ROOT / "runs" / "paper_v2" / "colmap_compare_eth3d5" / "colmap_eth3d5_summary.csv"

STRECHA_SCENES = ["fountain-P11", "entry-P10", "Herz-Jesus-P8", "Herz-Jesus-P25", "Castle-P19", "Castle-P30"]
ETH3D_SCENES = ["courtyard", "delivery_area", "office", "terrace"]

METHOD_ORDER = [
    "orig_gt_ligt",
    "orig_gt_ligt_pa",
    "orig_rraa_ligt",
    "orig_rraa_ligt_pa",
    "coltrk_gt_ligt",
    "coltrk_gt_ligt_pa",
    "coltrk_rraa_ligt",
    "coltrk_rraa_ligt_pa",
    "colmap",
]

METHOD_LABELS = {
    "orig_gt_ligt": "Orig GT+LiGT",
    "orig_gt_ligt_pa": "Orig GT+LiGT+PA",
    "orig_rraa_ligt": "Orig RRAA+LiGT",
    "orig_rraa_ligt_pa": "Orig RRAA+LiGT+PA",
    "coltrk_gt_ligt": "COL-trk GT+LiGT",
    "coltrk_gt_ligt_pa": "COL-trk GT+LiGT+PA",
    "coltrk_rraa_ligt": "COL-trk RRAA+LiGT",
    "coltrk_rraa_ligt_pa": "COL-trk RRAA+LiGT+PA",
    "colmap": "COLMAP",
}

METHOD_COLORS = {
    "orig_gt_ligt": "#8d99ae",
    "orig_gt_ligt_pa": "#a68a64",
    "orig_rraa_ligt": "#d94841",
    "orig_rraa_ligt_pa": "#9c3f97",
    "coltrk_gt_ligt": "#4c78a8",
    "coltrk_gt_ligt_pa": "#2a9d8f",
    "coltrk_rraa_ligt": "#f28e2b",
    "coltrk_rraa_ligt_pa": "#76b7b2",
    "colmap": "#e9c46a",
}

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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def load_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = dict(row)
            out["_rotation_median_deg"] = maybe_float(row.get("rotation_median_deg"))
            out["_translation_mm_median"] = maybe_float(row.get("translation_mm_median"))
            out["use_pa"] = str(row.get("use_pa", "")).strip() == "True"
            rows.append(out)
    return rows


def select_one(rows: list[dict], **conds: str) -> dict | None:
    for row in rows:
        if all(row.get(k) == v for k, v in conds.items()):
            return row
    return None


def merged_rows(original_rows: list[dict], coltrk_rows: list[dict], colmap_rows: list[dict], scenes: list[str]) -> list[dict]:
    out: list[dict] = []
    for scene in scenes:
        for exp in ["gt_ligt", "gt_ligt_pa", "rraa_ligt", "rraa_ligt_pa"]:
            row = select_one(original_rows, scene=scene, experiment_group=exp)
            out.append({
                "scene": scene,
                "method": f"orig_{exp}",
                "rotation_median_deg": float("nan") if row is None else row["_rotation_median_deg"],
                "translation_mm_median": float("nan") if row is None else row["_translation_mm_median"],
                "status": None if row is None else row.get("status"),
            })
        for exp in ["gt_ligt", "gt_ligt_pa", "rraa_ligt", "rraa_ligt_pa"]:
            row = select_one(coltrk_rows, scene=scene, experiment_group=exp)
            out.append({
                "scene": scene,
                "method": f"coltrk_{exp}",
                "rotation_median_deg": float("nan") if row is None else row["_rotation_median_deg"],
                "translation_mm_median": float("nan") if row is None else row["_translation_mm_median"],
                "status": None if row is None else row.get("status"),
            })
        crow = select_one(colmap_rows, scene=scene)
        out.append({
            "scene": scene,
            "method": "colmap",
            "rotation_median_deg": float("nan") if crow is None else crow["_rotation_median_deg"],
            "translation_mm_median": float("nan") if crow is None else crow["_translation_mm_median"],
            "status": "ok" if crow is not None else None,
        })
    return out


def write_table(rows: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene", "method", "method_label", "rotation_median_deg", "translation_mm_median", "status"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "scene": row["scene"],
                "method": row["method"],
                "method_label": METHOD_LABELS[row["method"]],
                "rotation_median_deg": row["rotation_median_deg"],
                "translation_mm_median": row["translation_mm_median"],
                "status": row["status"],
            })


def save_fig(fig: plt.Figure, stem: str) -> None:
    ensure_dir(OUT_DIR)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric(rows: list[dict], scenes: list[str], metric_key: str, ylabel: str, title: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 6.4))
    x = np.arange(len(scenes))
    width = 0.085
    offsets = (np.arange(len(METHOD_ORDER)) - (len(METHOD_ORDER) - 1) / 2.0) * width

    for idx, method in enumerate(METHOD_ORDER):
        vals = []
        for scene in scenes:
            row = next((r for r in rows if r["scene"] == scene and r["method"] == method), None)
            vals.append(float("nan") if row is None else row[metric_key])
        ax.bar(x + offsets[idx], vals, width=width * 0.95, color=METHOD_COLORS[method], label=METHOD_LABELS[method])

    ax.set_xticks(x)
    ax.set_xticklabels([SCENE_SHORT[s] for s in scenes], rotation=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    save_fig(fig, stem)


def write_notes() -> None:
    text = """# Track Source Comparison Report

This folder compares:

- Original tracks + 4 backend groups
- COLMAP tracks + 4 backend groups
- COLMAP final pipeline baseline

Datasets:
- Strecha6
- ETH3D4 (`courtyard`, `delivery_area`, `office`, `terrace`)
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dir(OUT_DIR)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 9})

    original_rows = load_csv(ORIGINAL_CSV)
    coltrk_rows = load_csv(COLMAP_TRACKS_CSV)
    colmap_strecha_rows = load_csv(COLMAP_STRECHA_CSV)
    colmap_eth3d_rows = [row for row in load_csv(COLMAP_ETH3D_CSV) if row.get("scene") in ETH3D_SCENES]

    strecha_rows = merged_rows(original_rows, coltrk_rows, colmap_strecha_rows, STRECHA_SCENES)
    eth3d_rows = merged_rows(original_rows, coltrk_rows, colmap_eth3d_rows, ETH3D_SCENES)

    write_table(strecha_rows, OUT_DIR / "strecha6_track_source_compare_table.csv")
    write_table(eth3d_rows, OUT_DIR / "eth3d4_track_source_compare_table.csv")

    plot_metric(
        strecha_rows,
        STRECHA_SCENES,
        "translation_mm_median",
        "translation_mm_median (mm)",
        "Strecha6: 8 Backend Experiments + COLMAP",
        "01_strecha6_translation_compare",
    )
    plot_metric(
        strecha_rows,
        STRECHA_SCENES,
        "rotation_median_deg",
        "rotation_median_deg",
        "Strecha6: Rotation Comparison",
        "02_strecha6_rotation_compare",
    )
    plot_metric(
        eth3d_rows,
        ETH3D_SCENES,
        "translation_mm_median",
        "translation_mm_median (mm)",
        "ETH3D4: 8 Backend Experiments + COLMAP",
        "03_eth3d4_translation_compare",
    )
    plot_metric(
        eth3d_rows,
        ETH3D_SCENES,
        "rotation_median_deg",
        "rotation_median_deg",
        "ETH3D4: Rotation Comparison",
        "04_eth3d4_rotation_compare",
    )
    write_notes()
    print(f"saved track-source comparison report to: {OUT_DIR}")


if __name__ == "__main__":
    main()
