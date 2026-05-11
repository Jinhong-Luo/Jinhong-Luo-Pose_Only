#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = REPO_ROOT / "runs" / "paper_v2" / "progress_report"

STRECHA6_CSV = REPO_ROOT / "runs" / "paper_v2" / "eval_strecha6" / "paper_summary_main.csv"
MAIN_CSV = REPO_ROOT / "runs" / "paper_v2" / "eval_main" / "paper_summary_main.csv"
COLMAP_DIR = REPO_ROOT / "runs" / "paper_v2" / "colmap_compare" / "fountain-P11"
CASTLE_P19_ANALYSIS = REPO_ROOT / "runs" / "paper_v2" / "castle_p19_analysis"

SCENE_ORDER = ["fountain-P11", "entry-P10", "Herz-Jesus-P8", "Herz-Jesus-P25", "Castle-P19", "Castle-P30"]
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
    "electro": "electro",
}
OURS_GROUPS = ["rraa_ligt", "rraa_ligt_pa"]
MAIN_GROUP_ORDER = ["gt_ligt", "gt_ligt_pa", "rraa_ligt", "rraa_ligt_pa"]
GROUP_LABELS = {
    "gt_ligt": "GT + LiGT",
    "gt_ligt_pa": "GT + LiGT + PA",
    "rraa_ligt": "Ours-LiGT",
    "rraa_ligt_pa": "Ours-LiGT-PA",
}
COLORS = {
    "rraa_ligt": "#d94841",
    "rraa_ligt_pa": "#2b6cb0",
    "COLMAP": "#f4a261",
    "GT": "#2b6cb0",
    "RRAA": "#d94841",
}
STATUS_COLORS = {"ok": "#2a9d8f", "partial": "#e9c46a", "failed": "#e76f51"}
PA_STATUS_COLORS = {"accepted": "#2a9d8f", "rejected": "#e76f51", "skipped": "#e9c46a"}


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


def load_table(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = dict(row)
            out["use_pa"] = str(row.get("use_pa", "")).strip() == "True"
            out["_rotation_median_deg"] = maybe_float(row.get("rotation_median_deg"))
            out["_translation_mm_median"] = maybe_float(row.get("translation_mm_median"))
            out["_translation_mm_p90"] = maybe_float(row.get("translation_mm_p90"))
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


def load_colmap_metrics() -> dict:
    rot = json.loads((COLMAP_DIR / "eval_rotation_colmap.json").read_text(encoding="utf-8"))
    trans = json.loads((COLMAP_DIR / "eval_translation_colmap.json").read_text(encoding="utf-8"))
    if isinstance(rot, list):
        rot_best = min(rot, key=lambda item: float(item.get("median_deg", float("inf"))))
    else:
        rot_best = rot
    return {
        "rotation_median_deg": float(rot_best.get("best_median_deg", rot_best.get("median_deg", np.nan))),
        "translation_mm_median": float(trans["mm"]["median"]),
        "translation_mm_rmse": float(trans["mm"]["rmse"]),
    }


def write_comparison_table(strecha_rows: list[dict], colmap: dict) -> None:
    ensure_dir(OUTDIR)
    path = OUTDIR / "comparison_table_strecha6.csv"
    fieldnames = [
        "scene",
        "ours_ligt_rotation_deg",
        "ours_ligt_translation_mm",
        "ours_ligt_pa_rotation_deg",
        "ours_ligt_pa_translation_mm",
        "colmap_rotation_deg",
        "colmap_translation_mm",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scene in SCENE_ORDER:
            r1 = select_one(strecha_rows, scene=scene, experiment_group="rraa_ligt")
            r2 = select_one(strecha_rows, scene=scene, experiment_group="rraa_ligt_pa")
            writer.writerow({
                "scene": scene,
                "ours_ligt_rotation_deg": "" if r1 is None else r1["_rotation_median_deg"],
                "ours_ligt_translation_mm": "" if r1 is None else r1["_translation_mm_median"],
                "ours_ligt_pa_rotation_deg": "" if r2 is None else r2["_rotation_median_deg"],
                "ours_ligt_pa_translation_mm": "" if r2 is None else r2["_translation_mm_median"],
                "colmap_rotation_deg": colmap["rotation_median_deg"] if scene == "fountain-P11" else "",
                "colmap_translation_mm": colmap["translation_mm_median"] if scene == "fountain-P11" else "",
            })


def plot_strecha_translation_comparison(strecha_rows: list[dict], colmap: dict) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    x = np.arange(len(SCENE_ORDER))
    width = 0.26

    ours = []
    ours_pa = []
    for scene in SCENE_ORDER:
        r1 = select_one(strecha_rows, scene=scene, experiment_group="rraa_ligt")
        r2 = select_one(strecha_rows, scene=scene, experiment_group="rraa_ligt_pa")
        ours.append(float("nan") if r1 is None else r1["_translation_mm_median"])
        ours_pa.append(float("nan") if r2 is None else r2["_translation_mm_median"])

    ax.bar(x - width / 2, ours, width=width, color=COLORS["rraa_ligt"], label="Ours-LiGT")
    ax.bar(x + width / 2, ours_pa, width=width, color=COLORS["rraa_ligt_pa"], label="Ours-LiGT-PA")
    ax.scatter(
        [x[0]],
        [colmap["translation_mm_median"]],
        marker="*",
        s=230,
        color=COLORS["COLMAP"],
        edgecolor="black",
        linewidth=0.6,
        label="COLMAP (fountain-P11 only)",
        zorder=5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([SCENE_SHORT[s] for s in SCENE_ORDER], rotation=10)
    ax.set_ylabel("translation_mm_median (mm)")
    ax.set_title("Progress Report: Strecha6 Translation Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    save_fig(fig, "01_strecha6_translation_comparison")


def plot_fountain_three_way(strecha_rows: list[dict], colmap: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
    methods = ["Ours-LiGT", "Ours-LiGT-PA", "COLMAP"]
    trans_vals = [
        select_one(strecha_rows, scene="fountain-P11", experiment_group="rraa_ligt")["_translation_mm_median"],
        select_one(strecha_rows, scene="fountain-P11", experiment_group="rraa_ligt_pa")["_translation_mm_median"],
        colmap["translation_mm_median"],
    ]
    rot_vals = [
        select_one(strecha_rows, scene="fountain-P11", experiment_group="rraa_ligt")["_rotation_median_deg"],
        select_one(strecha_rows, scene="fountain-P11", experiment_group="rraa_ligt_pa")["_rotation_median_deg"],
        colmap["rotation_median_deg"],
    ]
    colors = [COLORS["rraa_ligt"], COLORS["rraa_ligt_pa"], COLORS["COLMAP"]]
    axes[0].bar(methods, trans_vals, color=colors)
    axes[0].set_title("fountain-P11 Translation")
    axes[0].set_ylabel("median translation error (mm)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(methods, rot_vals, color=colors)
    axes[1].set_title("fountain-P11 Rotation")
    axes[1].set_ylabel("median rotation error (deg)")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Three-Way Comparison on the Available COLMAP Case", y=1.03)
    save_fig(fig, "02_fountain_three_way_comparison")


def plot_main_success_heatmap(main_rows: list[dict]) -> None:
    scenes = ["fountain-P11", "entry-P10", "Herz-Jesus-P8", "courtyard", "delivery_area", "office", "terrace", "electro"]
    status_value = {"failed": 0.0, "partial": 0.5, "ok": 1.0}
    arr = np.full((len(scenes), len(MAIN_GROUP_ORDER)), np.nan)
    labels = [["missing" for _ in MAIN_GROUP_ORDER] for _ in scenes]
    for i, scene in enumerate(scenes):
        for j, group in enumerate(MAIN_GROUP_ORDER):
            row = select_one(main_rows, scene=scene, experiment_group=group)
            if row is None:
                continue
            status = str(row.get("status", "missing"))
            arr[i, j] = status_value.get(status, np.nan)
            labels[i][j] = status

    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap([STATUS_COLORS["failed"], STATUS_COLORS["partial"], STATUS_COLORS["ok"]])
    norm = BoundaryNorm([-0.1, 0.25, 0.75, 1.1], cmap.N)

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(MAIN_GROUP_ORDER)))
    ax.set_xticklabels([GROUP_LABELS[g] for g in MAIN_GROUP_ORDER], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(scenes)))
    ax.set_yticklabels([SCENE_SHORT.get(scene, scene) for scene in scenes])
    ax.set_title("Main8 Success / Failure Overview")
    for i in range(len(scenes)):
        for j in range(len(MAIN_GROUP_ORDER)):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8)
    save_fig(fig, "03_main8_success_heatmap")


def plot_pa_status_summary(main_rows: list[dict], strecha_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    datasets = [("Main8", main_rows), ("Strecha6", strecha_rows)]
    cats = ["accepted", "rejected", "skipped"]
    sources = ["GT", "RRAA"]

    for ax, (title, rows) in zip(axes, datasets):
        counts = {src: {cat: 0 for cat in cats} for src in sources}
        for row in rows:
            if not row.get("use_pa"):
                continue
            if row.get("status") != "ok":
                continue
            src = row.get("rotation_source")
            raw = str(row.get("pa_status", "")).strip()
            if raw == "":
                raw = "skipped"
            if src in counts and raw in counts[src]:
                counts[src][raw] += 1

        x = np.arange(len(sources))
        bottom = np.zeros(len(sources))
        for cat in cats:
            vals = np.array([counts[src][cat] for src in sources], dtype=float)
            ax.bar(x, vals, bottom=bottom, color=PA_STATUS_COLORS[cat], label=cat)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(sources)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("count")
    axes[0].legend(frameon=False)
    fig.suptitle("PA Accepted / Rejected / Skipped", y=1.03)
    save_fig(fig, "04_pa_status_summary")


def write_failure_summary(main_rows: list[dict]) -> None:
    rows_out: list[dict] = []
    for row in main_rows:
        scene = row["scene"]
        group = row["experiment_group"]
        status = row["status"]
        note = ""
        if status == "failed" and row.get("rotation_source") == "RRAA":
            note = "graph disconnected / RRAA could not initialize"
        elif scene == "office" and status == "ok" and row.get("experiment_group") == "gt_ligt":
            note = "very low-quality front-end support (track_len_median=2, kept tracks=25)"
        elif scene == "courtyard" and status == "ok" and row.get("experiment_group") == "rraa_ligt":
            note = "large rotation error propagated to large translation error"
        elif scene == "terrace" and row.get("pa_status") == "rejected":
            note = "PA rejected: no reprojection improvement"
        elif scene == "delivery_area" and row.get("experiment_group") == "gt_ligt_pa" and row.get("pa_status") == "skipped":
            note = "PA skipped due to exception"
        if note:
            rows_out.append({
                "dataset": row["dataset"],
                "scene": scene,
                "experiment_group": group,
                "status": status,
                "rotation_median_deg": row.get("_rotation_median_deg"),
                "translation_mm_median": row.get("_translation_mm_median"),
                "pa_status": row.get("pa_status"),
                "failure_mode_or_case_note": note,
            })

    path = OUTDIR / "failure_case_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)


def write_progress_markdown(colmap: dict) -> None:
    text = f"""# Progress Report Assets

This folder contains the current report-ready assets for advisor updates.

## Main comparison

- `01_strecha6_translation_comparison.*`: Strecha6 translation median comparison for `Ours-LiGT` and `Ours-LiGT-PA`, with the currently available COLMAP point on `fountain-P11`.
- `02_fountain_three_way_comparison.*`: three-way comparison on the currently available COLMAP case.
- `comparison_table_strecha6.csv`: compact numeric table for direct quoting in slides.

Current available COLMAP comparison:
- `fountain-P11`: rotation median `{colmap["rotation_median_deg"]:.3f} deg`, translation median `{colmap["translation_mm_median"]:.2f} mm`

## Robustness / failure analysis

- `03_main8_success_heatmap.*`: scene-level success/failure overview on the main8 benchmark.
- `04_pa_status_summary.*`: accepted / rejected / skipped distribution for PA.
- `failure_case_summary.csv`: concise failure-mode / case-study summary.

## Detailed case study

The Castle-P19 diagnostic set is under:
- `../castle_p19_analysis`

Most useful files there:
- `01_frame_track_counts.*`
- `02_frame_spatial_coverage.*`
- `03_frame_feature_scatter_montage.*`
- `04_pair_quality_heatmap.*`
- `05_base_pair_usage.*`
- `06_threshold_ablation.*`

## Notes

- The current direct COLMAP comparison is only available for `fountain-P11`.
- Main comparison names:
  - `Ours-LiGT = RRAA + LiGT`
  - `Ours-LiGT-PA = RRAA + LiGT + PA`
"""
    (OUTDIR / "PROGRESS_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dir(OUTDIR)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 9})

    strecha_rows = load_table(STRECHA6_CSV)
    main_rows = load_table(MAIN_CSV)
    colmap = load_colmap_metrics()

    write_comparison_table(strecha_rows, colmap)
    plot_strecha_translation_comparison(strecha_rows, colmap)
    plot_fountain_three_way(strecha_rows, colmap)
    plot_main_success_heatmap(main_rows)
    plot_pa_status_summary(main_rows, strecha_rows)
    write_failure_summary(main_rows)
    write_progress_markdown(colmap)
    print(f"saved progress report assets to: {OUTDIR}")


if __name__ == "__main__":
    main()
