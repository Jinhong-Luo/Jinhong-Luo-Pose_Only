#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "runs" / "paper_v2" / "teacher_progress_2026-04-21"


def load_json(rel_path: str):
    with open(REPO_ROOT / rel_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(rel_path: str):
    with open(REPO_ROOT / rel_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value, default=None):
    if value in (None, "", "null"):
        return default
    return float(value)


def canonical_method_key(label: str) -> str:
    mapping = {
        "GT+LiGT": "gt_ligt",
        "GT+LiGT+PA": "gt_ligt_pa",
        "RRAA+LiGT": "rraa_ligt",
        "RRAA+LiGT+PA": "rraa_ligt_pa",
        "COLMAP": "colmap",
    }
    return mapping.get(label, label)


def save_phase_timeline():
    phase25 = load_json(
        "runs/paper_v2/optuna_frontend_qpair_phase2_5_vs_colmap_paircache_12scenes/best_result.json"
    )
    phase26c = load_json(
        "runs/paper_v2/optuna_frontend_qpair_phase2_6c_vs_colmap_paircache_12scenes/best_result.json"
    )
    robust = load_json("runs/paper_v2/recommended_top3_validation/robust_config.json")

    labels = ["Phase-2.5 best", "Phase-2.6c best", "Top-3 validated best"]
    scores = [
        float(phase25["best_summary"]["score"]),
        float(phase26c["best_summary"]["score"]),
        float(robust["validation_summary"]["score"]),
    ]
    colors = ["#2D6A4F", "#B23A48", "#F4A261"]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(labels, scores, color=colors, width=0.62)
    ax.set_ylabel("Validation score (lower is better)")
    ax.set_title("Frontend Search Progress: phase-2.5 to recommendation validation")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score, f"{score:.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.text(0.02, 0.95,
            "Score = mean ratio + std + worst (+ fail/skip/reject penalties)\n"
            "Phase-2.5 remains the strongest global configuration so far.",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="#F8F9FA", ec="#CED4DA"))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_phase_score_timeline.png", dpi=180)
    plt.close(fig)


def summarize_scan106():
    rows = load_csv(
        "runs/paper_v2/optuna_frontend_qpair_phase2_5_vs_colmap_paircache_12scenes/DTU_scan106_trial_analysis.csv"
    )
    grouped = {}
    for row in rows:
        key = row["deltas_rraa"]
        mm = as_float(row["scene_translation_mm_median"])
        if mm is None:
            continue
        grouped.setdefault(key, []).append(mm)
    summary = {}
    for key, vals in grouped.items():
        summary[key] = median(vals)
    return summary


def summarize_scan114():
    rows_ok = load_csv(
        "runs/paper_v2/optuna_frontend_qpair_phase2_5_vs_colmap_paircache_12scenes/DTU_scan114_trial_analysis.csv"
    )
    rows_fail = load_csv(
        "runs/paper_v2/optuna_frontend_qpair_phase2_6b_vs_colmap_paircache_12scenes/DTU_scan114_trial_analysis.csv"
    )
    ok_mm = [
        as_float(row["scene_translation_mm_median"])
        for row in rows_ok
        if row["deltas_rraa"] == "1,2,3,5,8" and row["scene_status"] == "ok"
    ]
    fail_count = sum(
        1 for row in rows_fail
        if row["deltas_rraa"] == "1,2,3,5" and str(row["scene_status"]).lower() == "failed"
    )
    return {
        "1,2,3,5,8_mm_median": median(ok_mm) if ok_mm else None,
        "1,2,3,5_failed_count": fail_count,
    }


def save_gap_conflict_chart():
    s106 = summarize_scan106()
    s114 = summarize_scan114()

    scenes = ["DTU scan106", "DTU scan114"]
    short_gap = [s106.get("1,2,3,5"), None]
    long_gap = [s106.get("1,2,3,5,8"), s114.get("1,2,3,5,8_mm_median")]

    x = range(len(scenes))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars1 = ax.bar([i - width / 2 for i in x], [v or 0 for v in short_gap], width,
                   label="deltas_rraa = 1,2,3,5", color="#4C78A8")
    bars2 = ax.bar([i + width / 2 for i in x], [v or 0 for v in long_gap], width,
                   label="deltas_rraa = 1,2,3,5,8", color="#E07A5F")

    ax.set_xticks(list(x), scenes)
    ax.set_ylabel("Scene translation median (mm)")
    ax.set_title("RRAA gap choice creates a real scene-level conflict")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, val in zip(bars1, short_gap):
        if val is None:
            ax.text(bar.get_x() + bar.get_width() / 2, 6, "failed",
                    ha="center", va="bottom", fontsize=10, color="#B23A48")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=10)
    for bar, val in zip(bars2, long_gap):
        if val is None:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}",
                ha="center", va="bottom", fontsize=10)

    ax.text(0.02, 0.96,
            f"scan114 with 1,2,3,5: failed in {s114['1,2,3,5_failed_count']} analyzed trials.\n"
            "This is why fallback became necessary.",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="#F8F9FA", ec="#CED4DA"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_scan106_scan114_gap_conflict.png", dpi=180)
    plt.close(fig)


def save_dtu_blocker_chart():
    rows = load_csv("runs/paper_v2/recommended_top3_validation/validation_results.csv")
    if not rows:
        raise RuntimeError("validation_results.csv is empty")
    best = min(rows, key=lambda row: as_float(row["score"], 1e18))
    scenes = ["DTU_scan1", "DTU_scan40", "DTU_scan69", "DTU_scan97", "DTU_scan106", "DTU_scan114"]
    ratios = [as_float(best[f"{scene}__translation_vs_colmap_ratio"], 0.0) for scene in scenes]
    colors = ["#B23A48" if scene in {"DTU_scan1", "DTU_scan40", "DTU_scan69"} else "#2A9D8F" for scene in scenes]
    labels = [scene.replace("DTU_", "") for scene in scenes]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.bar(labels, ratios, color=colors)
    ax.set_ylabel("translation_vs_colmap_ratio")
    ax.set_title("After fallback, the main blockers moved to DTU scan1/40/69")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, ratio, f"{ratio:.1f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_dtu_blockers_after_fallback.png", dpi=180)
    plt.close(fig)

    with open(OUT_DIR / "03_dtu_blockers_after_fallback.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "translation_vs_colmap_ratio"])
        for scene, ratio in zip(labels, ratios):
            writer.writerow([scene, ratio])


def save_teacher_three_way_compare():
    rows = load_csv("runs/paper_v2/final_benchmark_report/table_ai_analysis_long.csv")
    selected_methods = ["rraa_ligt_pa", "gt_ligt_pa", "colmap"]
    method_label = {
        "rraa_ligt_pa": "Best ours\n(RRAA+LiGT+PA)",
        "gt_ligt_pa": "Upper bound\n(GT+LiGT+PA)",
        "colmap": "COLMAP",
    }
    method_color = {
        "rraa_ligt_pa": "#C1121F",
        "gt_ligt_pa": "#6D28D9",
        "colmap": "#0F766E",
    }

    for dataset, out_name, title in [
        ("strecha", "03_strecha6_bestours_upper_colmap.png", "Strecha6: Best ours vs upper bound vs COLMAP"),
        ("DTU", "04_dtu6_bestours_upper_colmap.png", "DTU6: Best ours vs upper bound vs COLMAP"),
    ]:
        subset = [r for r in rows if r["dataset"] == dataset]
        scenes = sorted({r["scene"] for r in subset})
        by_scene_method = {}
        for row in subset:
            scene = row["scene"]
            method = canonical_method_key(row["method"])
            by_scene_method[(scene, method)] = as_float(row["translation_mm_median"])

        fig, ax = plt.subplots(figsize=(10.2, 4.8))
        x = list(range(len(scenes)))
        width = 0.24
        offsets = {
            "rraa_ligt_pa": -width,
            "gt_ligt_pa": 0.0,
            "colmap": width,
        }
        for method in selected_methods:
            vals = [by_scene_method.get((scene, method), 0.0) for scene in scenes]
            bars = ax.bar(
                [i + offsets[method] for i in x],
                vals,
                width=width,
                label=method_label[method],
                color=method_color[method],
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90 if val > 50 else 0,
                )
        ax.set_xticks(x, scenes, rotation=20, ha="right")
        ax.set_ylabel("Translation median (mm)")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(ncols=3, fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_DIR / out_name, dpi=180)
        plt.close(fig)

        if dataset == "strecha":
            legacy = OUT_DIR / "03_strecha6_best_ours_vs_colmap.png"
        else:
            legacy = OUT_DIR / "04_dtu6_best_ours_vs_colmap.png"
        legacy.write_bytes((OUT_DIR / out_name).read_bytes())


def save_recommended_assets_readme():
    text = """# Teacher Progress Assets

## Recommended order for tomorrow's progress report

1. `01_strecha6_translation_by_scene.png`
   - Show the main story on the stable dataset.
   - Easy message: GT+LiGT is strongest, RRAA+LiGT drops, PA can pull RRAA back.

2. `03_strecha6_best_ours_vs_colmap.png`
   - This version is intentionally redefined for the teacher report:
   - `best ours = RRAA+LiGT+PA`, `upper bound = GT+LiGT+PA`, plus `COLMAP`.

3. `04_dtu6_best_ours_vs_colmap.png`
   - Same three-way comparison on DTU.
   - Useful transition to the tuning story.

4. `05_runtime_by_method_dataset.png` and `06_memory_by_method_dataset.png`
   - Use for the efficiency section.
   - Explain fair comparison carefully: backend/core vs COLMAP mapper is the cleaner wording.

5. `01_phase_score_timeline.png`
   - New progress summary figure.
   - Key message: phase-2.5 is still the strongest global baseline; later work mainly identified structural conflicts.

6. `02_scan106_scan114_gap_conflict.png`
   - New diagnosis figure.
   - Key message: scan106 prefers shorter RRAA gaps, scan114 needs gap=8; single global `deltas_rraa` is insufficient.

7. `03_dtu_blockers_after_fallback.png`
   - New diagnosis figure.
   - Key message: fallback fixed scan106/114, but the blocker moved to DTU scan1/40/69.

## Suggested table files

- `table_best_ours_vs_colmap.csv`
  - Good for one-slide headline numbers.
- `table_runtime_memory_aggregate.csv`
  - Good for efficiency summary.
- `03_dtu_blockers_after_fallback.csv`
  - Good if the teacher wants exact numbers for the current blocker scenes.

## One-sentence status summary

The pipeline, evaluation, and Optuna/TPE-to-recommender workflow are all working end-to-end; the current challenge is no longer running experiments, but finding a frontend policy that stays strong on DTU without sacrificing the gains already obtained on Strecha and on scan106/114.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def copy_existing_assets():
    existing = [
        "runs/paper_v2/final_benchmark_report/01_strecha6_translation_by_scene.png",
        "runs/paper_v2/final_benchmark_report/05_runtime_by_method_dataset.png",
        "runs/paper_v2/final_benchmark_report/06_memory_by_method_dataset.png",
        "runs/paper_v2/final_benchmark_report/table_best_ours_vs_colmap.csv",
        "runs/paper_v2/final_benchmark_report/table_runtime_memory_aggregate.csv",
    ]
    for rel in existing:
        src = REPO_ROOT / rel
        dst = OUT_DIR / src.name
        dst.write_bytes(src.read_bytes())


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    copy_existing_assets()
    save_teacher_three_way_compare()
    save_phase_timeline()
    save_gap_conflict_chart()
    save_dtu_blocker_chart()
    save_recommended_assets_readme()
    print(f"saved assets to: {OUT_DIR}")


if __name__ == "__main__":
    main()
