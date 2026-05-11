#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MAIN_GROUP_ORDER = ["gt_ligt", "gt_ligt_pa", "rraa_ligt", "rraa_ligt_pa"]
MAIN_SCENE_ORDER = [
    "fountain-P11",
    "entry-P10",
    "Herz-Jesus-P8",
    "courtyard",
    "delivery_area",
    "office",
    "terrace",
    "electro",
]
STRECHA_SCENES = ["fountain-P11", "entry-P10", "Herz-Jesus-P8"]
STRECHA6_SCENE_ORDER = [
    "fountain-P11",
    "entry-P10",
    "Herz-Jesus-P8",
    "Herz-Jesus-P25",
    "Castle-P19",
    "Castle-P30",
]
SCENE_SHORT = {
    "fountain-P11": "fountain",
    "entry-P10": "entry",
    "Herz-Jesus-P8": "Herz",
    "Herz-Jesus-P25": "Herz-P25",
    "Castle-P19": "Castle-P19",
    "Castle-P30": "Castle-P30",
    "courtyard": "courtyard",
    "delivery_area": "delivery",
    "office": "office",
    "terrace": "terrace",
    "electro": "electro",
}
GROUP_LABELS = {
    "gt_ligt": "GT + LiGT",
    "gt_ligt_pa": "GT + LiGT + PA",
    "rraa_ligt": "RRAA + LiGT",
    "rraa_ligt_pa": "RRAA + LiGT + PA",
}
SOURCE_COLORS = {"GT": "#2b6cb0", "RRAA": "#d94841"}
SCENE_COLORS = [
    "#0f4c5c",
    "#e36414",
    "#5f0f40",
    "#6a994e",
    "#8d99ae",
    "#bc4749",
    "#4361ee",
    "#ffb703",
]
PA_MARKERS = {False: "o", True: "^"}
STATUS_COLORS = {"accepted": "#2a9d8f", "rejected": "#e76f51", "skipped": "#e9c46a"}
HEATMAP_COLORS = {"ok": "#2a9d8f", "partial": "#e9c46a", "failed": "#e76f51"}


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
            out["_delta_translation_mm_median"] = maybe_float(row.get("delta_translation_mm_median"))
            out["_u_min"] = maybe_float(row.get("u_min"))
            out["_g_min"] = maybe_float(row.get("g_min"))
            rows.append(out)
    return rows


def select_one(rows: list[dict], **conds: str) -> dict | None:
    for row in rows:
        ok = True
        for key, val in conds.items():
            if row.get(key) != val:
                ok = False
                break
        if ok:
            return row
    return None


def filter_rows(rows: list[dict], fn) -> list[dict]:
    return [row for row in rows if fn(row)]


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_main_grouped_bars(rows: list[dict], outdir: Path, scene_order: list[str], stem_prefix: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.5))
    plot_rows = filter_rows(rows, lambda r: r.get("status") == "ok" and r.get("experiment_group") in MAIN_GROUP_ORDER)
    x = np.arange(len(MAIN_GROUP_ORDER))
    width = 0.72 / max(len(scene_order), 1)
    offsets = (np.arange(len(scene_order)) - (len(scene_order) - 1) / 2.0) * width

    for idx, scene in enumerate(scene_order):
        y = []
        for group in MAIN_GROUP_ORDER:
            row = select_one(plot_rows, scene=scene, experiment_group=group)
            y.append(float("nan") if row is None else row["_translation_mm_median"])
        ax.bar(
            x + offsets[idx],
            y,
            width=width * 0.95,
            color=SCENE_COLORS[idx % len(SCENE_COLORS)],
            label=SCENE_SHORT.get(scene, scene),
            alpha=0.95,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS[g] for g in MAIN_GROUP_ORDER], rotation=10)
    ax.set_ylabel("translation_mm_median (mm)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save_fig(fig, outdir, f"{stem_prefix}01_main_grouped_bar_translation_mm_median")


def plot_rotation_translation_scatter(rows: list[dict], outdir: Path, scene_order: list[str], stem_prefix: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    plot_rows = filter_rows(
        rows,
        lambda r: r.get("status") == "ok"
        and not np.isnan(r["_rotation_median_deg"])
        and not np.isnan(r["_translation_mm_median"]),
    )
    plot_rows = [r for r in plot_rows if r.get("scene") in scene_order]

    for rotation_source in ["GT", "RRAA"]:
        for use_pa in [False, True]:
            sub = [r for r in plot_rows if r.get("rotation_source") == rotation_source and r.get("use_pa") == use_pa]
            if not sub:
                continue
            ax.scatter(
                [r["_rotation_median_deg"] for r in sub],
                [r["_translation_mm_median"] for r in sub],
                s=80,
                color=SOURCE_COLORS[rotation_source],
                marker=PA_MARKERS[use_pa],
                edgecolor="black",
                linewidth=0.6,
                alpha=0.88,
                label=f"{rotation_source}, PA {'on' if use_pa else 'off'}",
            )
            for row in sub:
                ax.annotate(
                    SCENE_SHORT.get(row["scene"], row["scene"]),
                    (row["_rotation_median_deg"], row["_translation_mm_median"]),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=8,
                    alpha=0.82,
                )

    ax.set_xlabel("rotation_median_deg")
    ax.set_ylabel("translation_mm_median (mm)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_fig(fig, outdir, f"{stem_prefix}02_rotation_to_translation_scatter")


def plot_pa_gain_dumbbell(rows: list[dict], outdir: Path, scene_order: list[str], stem_prefix: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    base_map = {"GT": "gt_ligt", "RRAA": "rraa_ligt"}
    pa_map = {"GT": "gt_ligt_pa", "RRAA": "rraa_ligt_pa"}

    for ax, rotation_source in zip(axes, ["GT", "RRAA"]):
        pairs = []
        for scene in scene_order:
            base = select_one(rows, scene=scene, experiment_group=base_map[rotation_source])
            pa = select_one(rows, scene=scene, experiment_group=pa_map[rotation_source])
            if base is None or pa is None:
                continue
            if np.isnan(base["_translation_mm_median"]) or np.isnan(pa["_translation_mm_median"]):
                continue
            pairs.append((scene, base["_translation_mm_median"], pa["_translation_mm_median"]))

        for i, (scene, baseline, pa_val) in enumerate(pairs):
            color = SOURCE_COLORS[rotation_source]
            ax.plot([baseline, pa_val], [i, i], color=color, linewidth=2.2, alpha=0.7)
            ax.scatter(baseline, i, color="#6c757d", s=55, zorder=3, label="baseline" if i == 0 else None)
            ax.scatter(pa_val, i, color=color, s=70, zorder=3, label="PA" if i == 0 else None)
            ax.text(max(baseline, pa_val) + 15, i, f"{baseline - pa_val:+.1f} mm", va="center", fontsize=8)

        ax.set_yticks(np.arange(len(pairs)))
        ax.set_yticklabels([SCENE_SHORT.get(scene, scene) for scene, _, _ in pairs])
        ax.set_xlabel("translation_mm_median (mm)")
        ax.set_title(f"PA Gain: {rotation_source} Rotation")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(frameon=False, loc="lower right")

    fig.suptitle(title, y=1.03)
    save_fig(fig, outdir, f"{stem_prefix}03_pa_gain_dumbbell")


def plot_pa_status_distribution(rows: list[dict], outdir: Path, scene_order: list[str], stem_prefix: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.6))
    pa_rows = filter_rows(rows, lambda r: r.get("use_pa") is True and r.get("status") == "ok")
    pa_rows = [r for r in pa_rows if r.get("scene") in scene_order]
    categories = ["accepted", "rejected", "skipped"]
    sources = ["GT", "RRAA"]
    counts = {src: {cat: 0 for cat in categories} for src in sources}

    for row in pa_rows:
        src = row.get("rotation_source", "GT")
        status = str(row.get("pa_status", "")).strip() or "skipped"
        if src in counts and status in counts[src]:
            counts[src][status] += 1

    x = np.arange(len(sources))
    bottom = np.zeros(len(sources))
    for cat in categories:
        vals = np.array([counts[src][cat] for src in sources], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=STATUS_COLORS[cat], label=cat, width=0.6)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_fig(fig, outdir, f"{stem_prefix}04_pa_status_stacked_bar")


def plot_availability_heatmap(rows: list[dict], outdir: Path, scene_order: list[str], stem_prefix: str, title: str) -> None:
    status_map = {"failed": 0.0, "partial": 0.5, "ok": 1.0}
    arr = np.full((len(scene_order), len(MAIN_GROUP_ORDER)), np.nan)
    labels: list[list[str]] = [["missing" for _ in MAIN_GROUP_ORDER] for _ in scene_order]

    for i, scene in enumerate(scene_order):
        for j, group in enumerate(MAIN_GROUP_ORDER):
            row = select_one(rows, scene=scene, experiment_group=group)
            if row is None:
                continue
            status = str(row.get("status", "missing"))
            arr[i, j] = status_map.get(status, np.nan)
            labels[i][j] = status

    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap([HEATMAP_COLORS["failed"], HEATMAP_COLORS["partial"], HEATMAP_COLORS["ok"]])
    norm = BoundaryNorm([-0.1, 0.25, 0.75, 1.1], cmap.N)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(MAIN_GROUP_ORDER)))
    ax.set_xticklabels([GROUP_LABELS[g] for g in MAIN_GROUP_ORDER], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(scene_order)))
    ax.set_yticklabels([SCENE_SHORT.get(scene, scene) for scene in scene_order])
    ax.set_title(title)

    for i in range(len(scene_order)):
        for j in range(len(MAIN_GROUP_ORDER)):
            ax.text(j, i, labels[i][j], ha="center", va="center", color="black", fontsize=8)

    save_fig(fig, outdir, f"{stem_prefix}05_failure_availability_heatmap")


def plot_strecha6_scene_lines(rows: list[dict], outdir: Path, stem_prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    plot_rows = filter_rows(rows, lambda r: r.get("status") == "ok")
    x = np.arange(len(STRECHA6_SCENE_ORDER))

    for idx, group in enumerate(MAIN_GROUP_ORDER):
        y = []
        for scene in STRECHA6_SCENE_ORDER:
            row = select_one(plot_rows, scene=scene, experiment_group=group)
            y.append(float("nan") if row is None else row["_translation_mm_median"])
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            markersize=6.5,
            color=SCENE_COLORS[idx],
            label=GROUP_LABELS[group],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SCENE_SHORT.get(scene, scene) for scene in STRECHA6_SCENE_ORDER], rotation=10)
    ax.set_ylabel("translation_mm_median (mm)")
    ax.set_title("Strecha6: Translation Median Across Scenes")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    save_fig(fig, outdir, f"{stem_prefix}06_strecha6_scene_line_translation_mm_median")


def plot_ablation_point_plot(rows: list[dict], outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_rows = filter_rows(rows, lambda r: r.get("status") == "ok")

    qtrack_order = [
        "ablation_qtrack_weighted_sum",
        "ablation_qtrack_product",
        "ablation_qtrack_log_additive",
    ]
    qtrack_labels = ["weighted_sum", "product", "log_additive"]
    deg_order = ["ablation_deg_1e-4", "ablation_deg_1e-3", "ablation_deg_1e-2"]
    deg_labels = ["1e-4", "1e-3", "1e-2"]

    for idx, scene in enumerate(STRECHA_SCENES):
        y = []
        for group in qtrack_order:
            row = select_one(plot_rows, scene=scene, experiment_group=group)
            y.append(float("nan") if row is None else row["_translation_mm_median"])
        axes[0].plot(np.arange(len(qtrack_order)), y, marker="o", linewidth=2, color=SCENE_COLORS[idx], label=SCENE_SHORT[scene])

    axes[0].set_xticks(np.arange(len(qtrack_order)))
    axes[0].set_xticklabels(qtrack_labels, rotation=10)
    axes[0].set_ylabel("translation_mm_median (mm)")
    axes[0].set_title("Ablation: qtrack_mode")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    for idx, scene in enumerate(STRECHA_SCENES):
        y = []
        for group in deg_order:
            row = select_one(plot_rows, scene=scene, experiment_group=group)
            y.append(float("nan") if row is None else row["_translation_mm_median"])
        axes[1].plot(np.arange(len(deg_order)), y, marker="o", linewidth=2, color=SCENE_COLORS[idx], label=SCENE_SHORT[scene])

    axes[1].set_xticks(np.arange(len(deg_order)))
    axes[1].set_xticklabels(deg_labels)
    axes[1].set_ylabel("translation_mm_median (mm)")
    axes[1].set_title("Ablation: u_min / g_min")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    save_fig(fig, outdir, "06_ablation_point_plot")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-v2 figures from eval CSVs.")
    parser.add_argument("--main_csv", default="runs/paper_v2/eval_main/paper_summary_main.csv")
    parser.add_argument("--ablation_csv", default="runs/paper_v2/eval_ablation/paper_summary_main.csv")
    parser.add_argument("--output_dir", default="runs/paper_v2/figures")
    parser.add_argument("--mode", choices=["main8", "strecha6"], default="main8")
    args = parser.parse_args()

    main_rows = load_table(Path(args.main_csv))
    outdir = Path(args.output_dir)
    ensure_outdir(outdir)

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 9})

    if args.mode == "strecha6":
        scene_order = STRECHA6_SCENE_ORDER
        stem_prefix = "strecha6_"
        plot_main_grouped_bars(
            main_rows,
            outdir,
            scene_order,
            stem_prefix,
            "Strecha6 Main Comparison: Translation Median by Experiment Group",
        )
        plot_rotation_translation_scatter(
            main_rows,
            outdir,
            scene_order,
            stem_prefix,
            "Strecha6 Rotation-to-Translation Scatter",
        )
        plot_pa_gain_dumbbell(
            main_rows,
            outdir,
            scene_order,
            stem_prefix,
            "Strecha6 Paired Baseline vs PA Translation Error",
        )
        plot_pa_status_distribution(
            main_rows,
            outdir,
            scene_order,
            stem_prefix,
            "Strecha6 PA Status Distribution by Rotation Source",
        )
        plot_availability_heatmap(
            main_rows,
            outdir,
            scene_order,
            stem_prefix,
            "Strecha6 Failure / Availability Heatmap",
        )
        plot_strecha6_scene_lines(main_rows, outdir, stem_prefix)
    else:
        ablation_rows = load_table(Path(args.ablation_csv))
        plot_main_grouped_bars(
            main_rows,
            outdir,
            MAIN_SCENE_ORDER,
            "",
            "Main Comparison: Translation Median by Experiment Group",
        )
        plot_rotation_translation_scatter(
            main_rows,
            outdir,
            MAIN_SCENE_ORDER,
            "",
            "Rotation-to-Translation Scatter",
        )
        plot_pa_gain_dumbbell(
            main_rows,
            outdir,
            MAIN_SCENE_ORDER,
            "",
            "Paired Baseline vs PA Translation Error",
        )
        plot_pa_status_distribution(
            main_rows,
            outdir,
            MAIN_SCENE_ORDER,
            "",
            "PA Status Distribution by Rotation Source",
        )
        plot_availability_heatmap(
            main_rows,
            outdir,
            MAIN_SCENE_ORDER,
            "",
            "Failure / Availability Heatmap",
        )
        plot_ablation_point_plot(ablation_rows, outdir)

    print(f"saved figures to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
