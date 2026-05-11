#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

OURS_STRECHA_CSV = REPO_ROOT / "runs" / "paper_v2" / "eval_strecha6" / "paper_summary_main.csv"
OURS_DTU_CSV = REPO_ROOT / "runs" / "paper_v2" / "eval_dtu6_final_selected" / "paper_summary_main.csv"

COLMAP_ROOT = REPO_ROOT / "Colmap_runs"
COLMAP_RUNTIME_CSV = COLMAP_ROOT / "colmap_runtime_memory_summary.csv"
OUT_DIR = REPO_ROOT / "runs" / "paper_v2" / "final_benchmark_report"
COLMAP_EVAL_DIR = OUT_DIR / "colmap_eval_cache"

METHOD_ORDER = ["gt_ligt", "gt_ligt_pa", "rraa_ligt", "rraa_ligt_pa"]
METHOD_LABELS = {
    "gt_ligt": "GT+LiGT",
    "gt_ligt_pa": "GT+LiGT+PA",
    "rraa_ligt": "RRAA+LiGT",
    "rraa_ligt_pa": "RRAA+LiGT+PA",
    "colmap": "COLMAP",
}
METHOD_COLORS = {
    "gt_ligt": "#2563eb",
    "gt_ligt_pa": "#7c3aed",
    "rraa_ligt": "#f97316",
    "rraa_ligt_pa": "#dc2626",
    "colmap": "#059669",
}

STRECHA_SCENES = [
    {
        "dataset": "strecha",
        "scene": "fountain-P11",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "strecha" / "fountain-P11",
        "image_list": REPO_ROOT / "data" / "prepared" / "strecha" / "fountain-P11" / "image_list.txt",
        "images_bin": COLMAP_ROOT / "strecha" / "fountain-P11" / "sparse" / "0" / "images.bin",
        "gt_unit_to_mm": 1000.0,
    },
    {
        "dataset": "strecha",
        "scene": "entry-P10",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "strecha" / "entry-P10",
        "image_list": REPO_ROOT / "data" / "prepared" / "strecha" / "entry-P10" / "image_list.txt",
        "images_bin": COLMAP_ROOT / "strecha" / "entry-P10" / "sparse" / "0" / "images.bin",
        "gt_unit_to_mm": 1000.0,
    },
    {
        "dataset": "strecha",
        "scene": "Herz-Jesus-P8",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "strecha" / "Herz-Jesus-P8",
        "image_list": REPO_ROOT / "data" / "prepared" / "strecha" / "Herz-Jesus-P8" / "image_list.txt",
        "images_bin": COLMAP_ROOT / "strecha" / "Herz-Jesus-P8" / "sparse" / "0" / "images.bin",
        "gt_unit_to_mm": 1000.0,
    },
    {
        "dataset": "strecha",
        "scene": "Castle-P19",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "strecha" / "Castle-P19",
        "image_list": REPO_ROOT / "data" / "prepared" / "strecha" / "Castle-P19" / "image_list.txt",
        "images_bin": COLMAP_ROOT / "strecha" / "castle-P19" / "sparse" / "0" / "images.bin",
        "gt_unit_to_mm": 1000.0,
    },
    {
        "dataset": "strecha",
        "scene": "Castle-P30",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "strecha" / "Castle-P30-first29",
        "image_list": REPO_ROOT / "data" / "prepared" / "strecha" / "Castle-P30-first29" / "image_list.txt",
        "images_bin": COLMAP_ROOT / "strecha" / "castle-P30" / "sparse" / "0" / "images.bin",
        "gt_unit_to_mm": 1000.0,
    },
    {
        "dataset": "strecha",
        "scene": "Herz-Jesus-P25",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "strecha" / "Herz-Jesus-P25",
        "image_list": REPO_ROOT / "data" / "prepared" / "strecha" / "Herz-Jesus-P25" / "image_list.txt",
        "images_bin": COLMAP_ROOT / "strecha" / "Herz-Jesus-P25" / "sparse" / "0" / "images.bin",
        "gt_unit_to_mm": 1000.0,
    },
]

DTU_SCENES = [
    {
        "dataset": "DTU",
        "scene": "scan1",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "DTU" / "scan1",
        "image_list": REPO_ROOT / "data" / "prepared" / "DTU" / "scan1" / "image_list.txt",
        "images_bin": COLMAP_ROOT / "DTU" / "scan1" / "sparse" / "0" / "images.bin",
        "gt_unit_to_mm": 1.0,
    },
    {
        "dataset": "DTU",
        "scene": "scan40",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "DTU" / "scan40",
        "image_list": REPO_ROOT / "data" / "prepared" / "DTU" / "scan40" / "image_list.txt",
        "images_bin": COLMAP_ROOT / "DTU" / "scan40" / "sparse" / "0" / "images.bin",
        "gt_unit_to_mm": 1.0,
    },
    {
        "dataset": "DTU",
        "scene": "scan69",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "DTU" / "scan69",
        "image_list": REPO_ROOT / "data" / "prepared" / "DTU" / "scan69" / "image_list.txt",
        "images_bin": COLMAP_ROOT / "DTU" / "scan69" / "sparse" / "0" / "images.bin",
        "gt_unit_to_mm": 1.0,
    },
    {
        "dataset": "DTU",
        "scene": "scan97",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "DTU" / "scan97",
        "image_list": REPO_ROOT / "data" / "prepared" / "DTU" / "scan97" / "image_list.txt",
        "images_bin": COLMAP_ROOT / "DTU" / "scan97" / "sparse" / "0" / "images.bin",
        "gt_unit_to_mm": 1.0,
    },
    {
        "dataset": "DTU",
        "scene": "scan106",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "DTU" / "scan106",
        "image_list": REPO_ROOT / "data" / "prepared" / "DTU" / "scan106" / "image_list.txt",
        "images_bin": COLMAP_ROOT / "DTU" / "scan106" / "sparse" / "0" / "images.bin",
        "gt_unit_to_mm": 1.0,
    },
    {
        "dataset": "DTU",
        "scene": "scan114",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "DTU" / "scan114",
        "image_list": REPO_ROOT / "data" / "prepared" / "DTU" / "scan114" / "image_list.txt",
        "images_bin": COLMAP_ROOT / "DTU" / "scan114" / "sparse" / "0" / "images.bin",
        "gt_unit_to_mm": 1.0,
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: str | float | int | None) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def fmt_num(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def normalize_runtime_scene(dataset: str, scene: str) -> str:
    if dataset.lower() == "strecha":
        mapping = {
            "castle-p19": "Castle-P19",
            "castle-p30": "Castle-P30",
            "entry-p10": "entry-P10",
            "fountain-p11": "fountain-P11",
            "herz-jesus-p25": "Herz-Jesus-P25",
            "herz-jesus-p8": "Herz-Jesus-P8",
        }
        return mapping.get(scene.lower(), scene)
    return scene


def load_ours_rows() -> list[dict]:
    rows = []
    for path in [OURS_STRECHA_CSV, OURS_DTU_CSV]:
        for row in read_csv(path):
            if row.get("status") != "ok":
                continue
            if row.get("experiment_group") not in METHOD_ORDER:
                continue
            rows.append(row)
    return rows


def export_colmap_eval(scene_cfg: dict) -> dict:
    dataset = scene_cfg["dataset"]
    scene = scene_cfg["scene"]
    out_dir = COLMAP_EVAL_DIR / dataset / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    poses_txt = out_dir / "poses_c2w_colmap.txt"
    names_txt = out_dir / "image_names_used.txt"
    rabs_npy = out_dir / "R_abs_colmap_w2c.npy"
    eval_rot = out_dir / "eval_rotation_colmap.json"
    eval_trans = out_dir / "eval_translation_colmap.json"
    summary_json = out_dir / "summary.json"

    if not summary_json.exists():
        run([
            str(PYTHON),
            "tools\\colmap_export_poses.py",
            "--images_bin",
            str(scene_cfg["images_bin"]),
            "--image_list",
            str(scene_cfg["image_list"]),
            "--out_txt",
            str(poses_txt),
            "--out_names",
            str(names_txt),
            "--out_rabs_w2c_npy",
            str(rabs_npy),
        ])
        run([
            str(PYTHON),
            "tools\\eval_poseonly_strecha_mm.py",
            "--est_poses",
            str(poses_txt),
            "--est_type",
            "c2w",
            "--gt_centers_npy",
            str(scene_cfg["prepared_dir"] / "gt_centers.npy"),
            "--gt_unit_to_mm",
            str(scene_cfg["gt_unit_to_mm"]),
            "--out_json",
            str(eval_trans),
        ])
        run([
            str(PYTHON),
            "tools\\eval_rraa_rotation.py",
            "--est_npy",
            str(rabs_npy),
            "--gt_npy",
            str(scene_cfg["prepared_dir"] / "R_abs_gt_w2c.npy"),
            "--out_json",
            str(eval_rot),
        ])

        trans = json.loads(eval_trans.read_text(encoding="utf-8"))
        rot = json.loads(eval_rot.read_text(encoding="utf-8"))
        rot_best = min(float(item["median_deg"]) for item in rot)
        summary = {
            "dataset": dataset,
            "scene": scene,
            "rotation_median_deg": rot_best,
            "translation_mm_median": float(trans["mm"]["median"]),
            "translation_mm_p90": float(trans["mm"]["p90"]),
            "translation_mm_rmse": float(trans["mm"]["rmse"]),
        }
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return json.loads(summary_json.read_text(encoding="utf-8"))


def load_colmap_runtime_rows() -> list[dict]:
    rows = []
    for row in read_csv(COLMAP_RUNTIME_CSV):
        if row.get("Stage") != "TOTAL":
            continue
        dataset = row["Dataset"]
        scene = normalize_runtime_scene(dataset, row["Scene"])
        rows.append(
            {
                "dataset": dataset,
                "scene": scene,
                "runtime_total_sec": to_float(row["TimeSeconds"]),
                "peak_memory_mb": to_float(row["PeakObservedWorkingSetGB"]) * 1024.0,
            }
        )
    return rows


def build_main_table(rows: list[dict], dataset_name: str) -> list[dict]:
    out = []
    for row in rows:
        if row["dataset"] != dataset_name:
            continue
        out.append(
            {
                "dataset": row["dataset"],
                "scene": row["scene"],
                "experiment_group": row["experiment_group"],
                "rotation_median_deg": fmt_num(to_float(row["rotation_median_deg"]), 3),
                "translation_mm_median": fmt_num(to_float(row["translation_mm_median"]), 3),
                "translation_mm_p90": fmt_num(to_float(row["translation_mm_p90"]), 3),
                "runtime_total_sec": fmt_num(to_float(row["runtime_total_sec"]), 3),
                "peak_memory_mb": fmt_num(to_float(row["peak_memory_mb"]), 1),
                "runtime_rraa_sec": fmt_num(to_float(row["runtime_rraa_sec"]), 3),
                "runtime_pose_only_sec": fmt_num(to_float(row["runtime_pose_only_sec"]), 3),
                "pa_status": row.get("pa_status", ""),
            }
        )
    return out


def best_ours_by_scene(rows: list[dict]) -> dict[tuple[str, str], dict]:
    best = {}
    for row in rows:
        key = (row["dataset"], row["scene"])
        val = to_float(row["translation_mm_median"])
        if key not in best or val < to_float(best[key]["translation_mm_median"]):
            best[key] = row
    return best


def aggregate_runtime_memory(ours_rows: list[dict], colmap_rows: list[dict]) -> list[dict]:
    out = []
    for dataset in ["strecha", "DTU"]:
        for method in METHOD_ORDER:
            selected = [r for r in ours_rows if r["dataset"] == dataset and r["experiment_group"] == method]
            if not selected:
                continue
            out.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "mean_runtime_total_sec": fmt_num(statistics.mean(to_float(r["runtime_total_sec"]) for r in selected), 3),
                    "mean_peak_memory_mb": fmt_num(statistics.mean(to_float(r["peak_memory_mb"]) for r in selected), 1),
                    "mean_translation_mm_median": fmt_num(statistics.mean(to_float(r["translation_mm_median"]) for r in selected), 3),
                }
            )

    for dataset in ["strecha", "DTU"]:
        selected = [r for r in colmap_rows if r["dataset"] == dataset]
        if not selected:
            continue
        out.append(
            {
                "dataset": dataset,
                "method": "colmap",
                "method_label": METHOD_LABELS["colmap"],
                "mean_runtime_total_sec": fmt_num(statistics.mean(r["runtime_total_sec"] for r in selected), 3),
                "mean_peak_memory_mb": fmt_num(statistics.mean(r["peak_memory_mb"] for r in selected), 1),
                "mean_translation_mm_median": fmt_num(statistics.mean(r["translation_mm_median"] for r in selected), 3),
            }
        )
    return out


def save_grouped_scene_bars(rows: list[dict], dataset: str, scenes: list[str], out_path: Path, title: str) -> None:
    width = 0.18
    x = list(range(len(scenes)))
    plt.figure(figsize=(12, 4.8))
    for idx, method in enumerate(METHOD_ORDER):
        vals = []
        for scene in scenes:
            match = next((r for r in rows if r["dataset"] == dataset and r["scene"] == scene and r["experiment_group"] == method), None)
            vals.append(to_float(match["translation_mm_median"]) if match else float("nan"))
        xpos = [v + (idx - 1.5) * width for v in x]
        plt.bar(
            xpos,
            vals,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.8,
        )
    plt.yscale("log")
    plt.ylabel("translation_mm_median (log scale)")
    plt.title(title)
    plt.xticks(x, scenes, rotation=20, ha="right")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def save_best_vs_colmap(best_rows: list[dict], out_path: Path, title: str) -> None:
    scenes = [r["scene"] for r in best_rows]
    colmap_vals = [r["colmap_translation_mm_median"] for r in best_rows]
    ours_vals = [r["best_ours_translation_mm_median"] for r in best_rows]
    width = 0.36
    x = list(range(len(scenes)))
    plt.figure(figsize=(10.5, 4.8))
    plt.bar(
        [v - width / 2 for v in x],
        colmap_vals,
        width=width,
        label="COLMAP",
        color=METHOD_COLORS["colmap"],
        edgecolor="black",
        linewidth=0.8,
    )
    plt.bar(
        [v + width / 2 for v in x],
        ours_vals,
        width=width,
        label="Best Ours",
        color="#111827",
        edgecolor="black",
        linewidth=0.8,
    )
    plt.yscale("log")
    plt.ylabel("translation_mm_median (log scale)")
    plt.title(title)
    plt.xticks(x, scenes, rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def save_runtime_memory_bars(rows: list[dict], value_key: str, ylabel: str, out_path: Path, title: str) -> None:
    ordered_methods = METHOD_ORDER + ["colmap"]
    datasets = ["strecha", "DTU"]
    width = 0.36
    x = list(range(len(ordered_methods)))
    plt.figure(figsize=(10.8, 4.8))
    for d_idx, dataset in enumerate(datasets):
        vals = []
        for method in ordered_methods:
            match = next((r for r in rows if r["dataset"] == dataset and r["method"] == method), None)
            vals.append(to_float(match[value_key]) if match else float("nan"))
        xpos = [v + (d_idx - 0.5) * width for v in x]
        color = "#2563eb" if dataset == "strecha" else "#f97316"
        plt.bar(xpos, vals, width=width, label=dataset, color=color, edgecolor="black", linewidth=0.8)
    plt.yscale("log")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, [METHOD_LABELS[m] for m in ordered_methods], rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def write_report_md(
    ours_rows: list[dict],
    colmap_eval_rows: list[dict],
    compare_rows: list[dict],
    out_path: Path,
) -> None:
    best = best_ours_by_scene(ours_rows)
    lines = [
        "# Final Benchmark Report",
        "",
        "Generated from:",
        f"- `{OURS_STRECHA_CSV}`",
        f"- `{OURS_DTU_CSV}`",
        f"- `{COLMAP_RUNTIME_CSV}`",
        "",
        "## Headline",
        "",
        f"- Strecha scenes: {sum(1 for r in compare_rows if r['dataset'] == 'strecha')}",
        f"- DTU scenes: {sum(1 for r in compare_rows if r['dataset'] == 'DTU')}",
        "",
        "## Best Ours Per Scene",
        "",
    ]
    for key in sorted(best.keys()):
        row = best[key]
        lines.append(
            f"- {row['dataset']} / {row['scene']}: {row['experiment_group']} "
            f"with {to_float(row['translation_mm_median']):.3f} mm"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `table_strecha6_main.csv`",
            "- `table_dtu6_final_main.csv`",
            "- `table_colmap_strecha6.csv`",
            "- `table_colmap_dtu6_final.csv`",
            "- `table_best_ours_vs_colmap.csv`",
            "- `table_runtime_memory_aggregate.csv`",
            "- `01_strecha6_translation_by_scene.png/pdf`",
            "- `02_dtu6_translation_by_scene.png/pdf`",
            "- `03_strecha6_best_ours_vs_colmap.png/pdf`",
            "- `04_dtu6_best_ours_vs_colmap.png/pdf`",
            "- `05_runtime_by_method_dataset.png/pdf`",
            "- `06_memory_by_method_dataset.png/pdf`",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COLMAP_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    ours_rows = load_ours_rows()
    colmap_runtime_rows = load_colmap_runtime_rows()

    colmap_eval_rows = []
    for scene_cfg in STRECHA_SCENES + DTU_SCENES:
        summary = export_colmap_eval(scene_cfg)
        runtime_match = next(
            (
                r
                for r in colmap_runtime_rows
                if r["dataset"] == scene_cfg["dataset"] and r["scene"] == scene_cfg["scene"]
            ),
            None,
        )
        if runtime_match:
            summary["runtime_total_sec"] = runtime_match["runtime_total_sec"]
            summary["peak_memory_mb"] = runtime_match["peak_memory_mb"]
        else:
            summary["runtime_total_sec"] = float("nan")
            summary["peak_memory_mb"] = float("nan")
        colmap_eval_rows.append(summary)

    strecha_table = build_main_table(ours_rows, "strecha")
    dtu_table = build_main_table(ours_rows, "DTU")
    write_csv(
        OUT_DIR / "table_strecha6_main.csv",
        strecha_table,
        list(strecha_table[0].keys()),
    )
    write_csv(
        OUT_DIR / "table_dtu6_final_main.csv",
        dtu_table,
        list(dtu_table[0].keys()),
    )

    colmap_strecha = [r for r in colmap_eval_rows if r["dataset"] == "strecha"]
    colmap_dtu = [r for r in colmap_eval_rows if r["dataset"] == "DTU"]
    write_csv(
        OUT_DIR / "table_colmap_strecha6.csv",
        [
            {
                "dataset": r["dataset"],
                "scene": r["scene"],
                "rotation_median_deg": fmt_num(r["rotation_median_deg"], 3),
                "translation_mm_median": fmt_num(r["translation_mm_median"], 3),
                "translation_mm_p90": fmt_num(r["translation_mm_p90"], 3),
                "runtime_total_sec": fmt_num(r["runtime_total_sec"], 3),
                "peak_memory_mb": fmt_num(r["peak_memory_mb"], 1),
            }
            for r in colmap_strecha
        ],
        ["dataset", "scene", "rotation_median_deg", "translation_mm_median", "translation_mm_p90", "runtime_total_sec", "peak_memory_mb"],
    )
    write_csv(
        OUT_DIR / "table_colmap_dtu6_final.csv",
        [
            {
                "dataset": r["dataset"],
                "scene": r["scene"],
                "rotation_median_deg": fmt_num(r["rotation_median_deg"], 3),
                "translation_mm_median": fmt_num(r["translation_mm_median"], 3),
                "translation_mm_p90": fmt_num(r["translation_mm_p90"], 3),
                "runtime_total_sec": fmt_num(r["runtime_total_sec"], 3),
                "peak_memory_mb": fmt_num(r["peak_memory_mb"], 1),
            }
            for r in colmap_dtu
        ],
        ["dataset", "scene", "rotation_median_deg", "translation_mm_median", "translation_mm_p90", "runtime_total_sec", "peak_memory_mb"],
    )

    best_ours = best_ours_by_scene(ours_rows)
    compare_rows = []
    for colmap_row in colmap_eval_rows:
        key = (colmap_row["dataset"], colmap_row["scene"])
        ours_row = best_ours.get(key)
        compare_rows.append(
            {
                "dataset": colmap_row["dataset"],
                "scene": colmap_row["scene"],
                "best_ours_experiment_group": ours_row["experiment_group"] if ours_row else "",
                "best_ours_translation_mm_median": fmt_num(to_float(ours_row["translation_mm_median"]) if ours_row else float("nan"), 3),
                "colmap_translation_mm_median": fmt_num(colmap_row["translation_mm_median"], 3),
                "ours_minus_colmap_mm": fmt_num(
                    (to_float(ours_row["translation_mm_median"]) - colmap_row["translation_mm_median"]) if ours_row else float("nan"),
                    3,
                ),
                "best_ours_runtime_total_sec": fmt_num(to_float(ours_row["runtime_total_sec"]) if ours_row else float("nan"), 3),
                "colmap_runtime_total_sec": fmt_num(colmap_row["runtime_total_sec"], 3),
                "best_ours_peak_memory_mb": fmt_num(to_float(ours_row["peak_memory_mb"]) if ours_row else float("nan"), 1),
                "colmap_peak_memory_mb": fmt_num(colmap_row["peak_memory_mb"], 1),
            }
        )
    write_csv(OUT_DIR / "table_best_ours_vs_colmap.csv", compare_rows, list(compare_rows[0].keys()))

    aggregate_rows = aggregate_runtime_memory(ours_rows, colmap_eval_rows)
    write_csv(OUT_DIR / "table_runtime_memory_aggregate.csv", aggregate_rows, list(aggregate_rows[0].keys()))

    save_grouped_scene_bars(
        ours_rows,
        "strecha",
        [cfg["scene"] for cfg in STRECHA_SCENES],
        OUT_DIR / "01_strecha6_translation_by_scene.png",
        "Strecha6: Translation Median by Scene and Method",
    )
    save_grouped_scene_bars(
        ours_rows,
        "DTU",
        [cfg["scene"] for cfg in DTU_SCENES],
        OUT_DIR / "02_dtu6_translation_by_scene.png",
        "DTU Final Selected: Translation Median by Scene and Method",
    )
    save_best_vs_colmap(
        [
            {
                "scene": row["scene"],
                "colmap_translation_mm_median": to_float(row["colmap_translation_mm_median"]),
                "best_ours_translation_mm_median": to_float(row["best_ours_translation_mm_median"]),
            }
            for row in compare_rows
            if row["dataset"] == "strecha"
        ],
        OUT_DIR / "03_strecha6_best_ours_vs_colmap.png",
        "Strecha6: Best Ours vs COLMAP",
    )
    save_best_vs_colmap(
        [
            {
                "scene": row["scene"],
                "colmap_translation_mm_median": to_float(row["colmap_translation_mm_median"]),
                "best_ours_translation_mm_median": to_float(row["best_ours_translation_mm_median"]),
            }
            for row in compare_rows
            if row["dataset"] == "DTU"
        ],
        OUT_DIR / "04_dtu6_best_ours_vs_colmap.png",
        "DTU Final Selected: Best Ours vs COLMAP",
    )
    save_runtime_memory_bars(
        aggregate_rows,
        "mean_runtime_total_sec",
        "mean runtime_total_sec (log scale)",
        OUT_DIR / "05_runtime_by_method_dataset.png",
        "Average Runtime by Dataset and Method",
    )
    save_runtime_memory_bars(
        aggregate_rows,
        "mean_peak_memory_mb",
        "mean peak_memory_mb (log scale)",
        OUT_DIR / "06_memory_by_method_dataset.png",
        "Average Peak Memory by Dataset and Method",
    )

    write_report_md(ours_rows, colmap_eval_rows, compare_rows, OUT_DIR / "report_summary.md")
    print(f"saved final benchmark report to: {OUT_DIR}")


if __name__ == "__main__":
    main()
