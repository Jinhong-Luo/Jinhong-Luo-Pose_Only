#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = REPO_ROOT / "runs" / "paper_v2" / "teacher_report_strecha6"

STRECHA6_CSV = REPO_ROOT / "runs" / "paper_v2" / "eval_strecha6" / "paper_summary_main.csv"
ABLATION_CSV = REPO_ROOT / "runs" / "paper_v2" / "eval_ablation" / "paper_summary_main.csv"
COLMAP_CSV = REPO_ROOT / "runs" / "paper_v2" / "colmap_compare_strecha6" / "colmap_strecha6_summary.csv"

SCENE_ORDER = ["fountain-P11", "entry-P10", "Herz-Jesus-P8", "Herz-Jesus-P25", "Castle-P19", "Castle-P30"]
ABLATION_SCENES = ["fountain-P11", "entry-P10", "Herz-Jesus-P8"]
SCENE_SHORT = {
    "fountain-P11": "fountain",
    "entry-P10": "entry",
    "Herz-Jesus-P8": "Herz-P8",
    "Herz-Jesus-P25": "Herz-P25",
    "Castle-P19": "Castle-P19",
    "Castle-P30": "Castle-P30",
}
COMPARE_GROUPS = ["gt_ligt", "gt_ligt_pa", "rraa_ligt", "rraa_ligt_pa", "COLMAP"]
COMPARE_LABELS = {
    "gt_ligt": "GT-LiGT",
    "gt_ligt_pa": "GT-LiGT-PA",
    "rraa_ligt": "Ours-LiGT",
    "rraa_ligt_pa": "Ours-LiGT-PA",
    "COLMAP": "COLMAP",
}
COLORS = {
    "gt_ligt": "#6c757d",
    "gt_ligt_pa": "#9c6644",
    "rraa_ligt": "#d94841",
    "rraa_ligt_pa": "#2b6cb0",
    "COLMAP": "#f4a261",
    "quality_off": "#8d99ae",
    "quality_on": "#2a9d8f",
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
            rows.append(out)
    return rows


def select_one(rows: list[dict], **conds: str) -> dict | None:
    for row in rows:
        if all(row.get(k) == v for k, v in conds.items()):
            return row
    return None


def save_fig(fig: plt.Figure, stem: str) -> None:
    ensure_dir(OUTDIR)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_compare_table(strecha_rows: list[dict], colmap_rows: list[dict]) -> None:
    path = OUTDIR / "strecha6_compare_table.csv"
    fieldnames = ["scene", "method", "rotation_median_deg", "translation_mm_median"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scene in SCENE_ORDER:
            for group in ["gt_ligt", "gt_ligt_pa", "rraa_ligt", "rraa_ligt_pa"]:
                row = select_one(strecha_rows, scene=scene, experiment_group=group)
                writer.writerow({
                    "scene": scene,
                    "method": COMPARE_LABELS[group],
                    "rotation_median_deg": row["_rotation_median_deg"],
                    "translation_mm_median": row["_translation_mm_median"],
                })
            crow = select_one(colmap_rows, scene=scene)
            writer.writerow({
                "scene": scene,
                "method": "COLMAP",
                "rotation_median_deg": crow["_rotation_median_deg"],
                "translation_mm_median": crow["_translation_mm_median"],
            })


def plot_compare_metric(strecha_rows: list[dict], colmap_rows: list[dict], metric_key: str, ylabel: str, stem: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    x = np.arange(len(SCENE_ORDER))
    width = 0.16
    offsets = (np.arange(len(COMPARE_GROUPS)) - (len(COMPARE_GROUPS) - 1) / 2.0) * width

    for offset, group in zip(offsets, COMPARE_GROUPS):
        vals = []
        for scene in SCENE_ORDER:
            if group == "COLMAP":
                row = select_one(colmap_rows, scene=scene)
            else:
                row = select_one(strecha_rows, scene=scene, experiment_group=group)
            vals.append(float("nan") if row is None else row[metric_key])
        ax.bar(x + offset, vals, width=width * 0.95, color=COLORS[group], label=COMPARE_LABELS[group])

    ax.set_xticks(x)
    ax.set_xticklabels([SCENE_SHORT[s] for s in SCENE_ORDER], rotation=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    save_fig(fig, stem)


def plot_compare_scatter(strecha_rows: list[dict], colmap_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    marker_map = {
        "GT-LiGT": "o",
        "GT-LiGT-PA": "^",
        "Ours-LiGT": "s",
        "Ours-LiGT-PA": "D",
        "COLMAP": "P",
    }
    for group in ["gt_ligt", "gt_ligt_pa", "rraa_ligt", "rraa_ligt_pa"]:
        label = COMPARE_LABELS[group]
        pts = [select_one(strecha_rows, scene=scene, experiment_group=group) for scene in SCENE_ORDER]
        ax.scatter(
            [p["_rotation_median_deg"] for p in pts],
            [p["_translation_mm_median"] for p in pts],
            s=85,
            marker=marker_map[label],
            color=COLORS[group],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.88,
            label=label,
        )
        if group in ["rraa_ligt_pa", "COLMAP"]:
            for scene, p in zip(SCENE_ORDER, pts):
                ax.annotate(SCENE_SHORT[scene], (p["_rotation_median_deg"], p["_translation_mm_median"]), textcoords="offset points", xytext=(4, 4), fontsize=8)

    pts = [select_one(colmap_rows, scene=scene) for scene in SCENE_ORDER]
    ax.scatter(
        [p["_rotation_median_deg"] for p in pts],
        [p["_translation_mm_median"] for p in pts],
        s=80,
        marker=marker_map["COLMAP"],
        color=COLORS["COLMAP"],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.88,
        label="COLMAP",
    )
    ax.set_xlabel("rotation_median_deg")
    ax.set_ylabel("translation_mm_median (mm)")
    ax.set_title("Strecha6: Rotation-to-Translation Comparison")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_fig(fig, "03_compare_scatter_rotation_vs_translation")


def plot_pa_ablation(ablation_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    x = np.arange(len(ABLATION_SCENES))
    width = 0.32
    off_vals, on_vals = [], []
    for scene in ABLATION_SCENES:
        off = select_one(ablation_rows, scene=scene, experiment_group="ablation_pa_off")
        on = select_one(ablation_rows, scene=scene, experiment_group="ablation_pa_on")
        off_vals.append(off["_translation_mm_median"])
        on_vals.append(on["_translation_mm_median"])
    ax.bar(x - width / 2, off_vals, width=width, color=COLORS["rraa_ligt"], label="PA off")
    ax.bar(x + width / 2, on_vals, width=width, color=COLORS["rraa_ligt_pa"], label="PA on")
    for idx, (a, b) in enumerate(zip(off_vals, on_vals)):
        ax.text(x[idx] + width / 2, b, f"{a-b:+.1f}", fontsize=8, ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENE_SHORT[s] for s in ABLATION_SCENES])
    ax.set_ylabel("translation_mm_median (mm)")
    ax.set_title("Ablation: Effect of PA")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_fig(fig, "04_ablation_pa")


def plot_qtrack_mode_ablation(ablation_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    mode_groups = [
        ("ablation_qtrack_weighted_sum", "weighted_sum"),
        ("ablation_qtrack_product", "product"),
        ("ablation_qtrack_log_additive", "log_additive"),
    ]
    x = np.arange(len(mode_groups))
    width = 0.22
    scene_colors = ["#0f4c5c", "#e36414", "#5f0f40"]
    for idx, scene in enumerate(ABLATION_SCENES):
        vals = []
        for group, _ in mode_groups:
            row = select_one(ablation_rows, scene=scene, experiment_group=group)
            vals.append(row["_translation_mm_median"])
        ax.bar(x + (idx - 1) * width, vals, width=width * 0.95, color=scene_colors[idx], label=SCENE_SHORT[scene])
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in mode_groups], rotation=10)
    ax.set_ylabel("translation_mm_median (mm)")
    ax.set_title("Ablation: Track-Quality Fusion Mode")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_fig(fig, "05_ablation_qtrack_mode")


def plot_quality_on_off_ablation(ablation_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    x = np.arange(len(ABLATION_SCENES))
    width = 0.32
    off_vals, on_vals = [], []
    for scene in ABLATION_SCENES:
        off = select_one(ablation_rows, scene=scene, experiment_group="ablation_rraa_quality_off")
        on = select_one(ablation_rows, scene=scene, experiment_group="ablation_rraa_quality_on")
        off_vals.append(off["_translation_mm_median"])
        on_vals.append(on["_translation_mm_median"])
    ax.bar(x - width / 2, off_vals, width=width, color=COLORS["quality_off"], label="quality off")
    ax.bar(x + width / 2, on_vals, width=width, color=COLORS["quality_on"], label="quality on")
    for idx, (a, b) in enumerate(zip(off_vals, on_vals)):
        ax.text(x[idx] + width / 2, b, f"{a-b:+.1f}", fontsize=8, ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENE_SHORT[s] for s in ABLATION_SCENES])
    ax.set_ylabel("translation_mm_median (mm)")
    ax.set_title("Ablation: Quality Weighting On vs Off")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_fig(fig, "06_ablation_quality_on_off")


def write_notes() -> None:
    text = """# Strecha6 Teacher Report

This folder contains only the corrected report assets requested for the advisor update.

## Compare experiments

- Ours-LiGT = RRAA + LiGT
- Ours-LiGT-PA = RRAA + LiGT + PA
- COLMAP

Metrics currently plotted:
- rotation error
- translation error

Runtime / memory are intentionally not included yet.

## Ablations currently included

- PA on/off
- track-quality fusion mode (`weighted_sum`, `product`, `log_additive`)
- quality weighting on/off

Note:
- The current `quality on/off` ablation is a joint switch on the quality-aware weighting pipeline.
- If a strict pair-quality-only vs track-quality-only decomposition is needed, additional runs are still required.
- Compare labels used in this folder:
  - `GT-LiGT = GT + LiGT`
  - `GT-LiGT-PA = GT + LiGT + PA`
  - `Ours-LiGT = RRAA + LiGT`
  - `Ours-LiGT-PA = RRAA + LiGT + PA`
  - `COLMAP`
"""
    (OUTDIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dir(OUTDIR)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 9})

    strecha_rows = load_csv(STRECHA6_CSV)
    ablation_rows = load_csv(ABLATION_CSV)
    colmap_rows = load_csv(COLMAP_CSV)

    write_compare_table(strecha_rows, colmap_rows)
    plot_compare_metric(
        strecha_rows,
        colmap_rows,
        "_translation_mm_median",
        "translation_mm_median (mm)",
        "01_compare_translation",
        "Strecha6 Comparison: Translation Error",
    )
    plot_compare_metric(
        strecha_rows,
        colmap_rows,
        "_rotation_median_deg",
        "rotation_median_deg",
        "02_compare_rotation",
        "Strecha6 Comparison: Rotation Error",
    )
    plot_compare_scatter(strecha_rows, colmap_rows)
    plot_pa_ablation(ablation_rows)
    plot_qtrack_mode_ablation(ablation_rows)
    plot_quality_on_off_ablation(ablation_rows)
    write_notes()
    print(f"saved teacher-report assets to: {OUTDIR}")


if __name__ == "__main__":
    main()
