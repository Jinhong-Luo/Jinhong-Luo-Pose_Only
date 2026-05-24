#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "runs" / "paper_v2" / "paper_ablation_assets_2026-05-02" / "external_compare"

OURS_CSV = OUT_DIR / "m3_m4_scene_metrics_for_external_compare.csv"
COLMAP_CSV = OUT_DIR / "colmap_all_pose_runtime_memory.csv"

STRECHA_ORDER = [
    "Herz-Jesus-P8",
    "Herz-Jesus-P25",
    "Fountain-P11",
    "Entry-P10",
    "Castle-P19",
    "Castle-P30",
]

DTU_ORDER = [
    "scan1",
    "scan40",
    "scan69",
    "scan97",
    "scan106",
    "scan114",
]

METHOD_ORDER = ["BaselineB", "M3", "M4", "COLMAP"]
WIDE_METHODS = [
    ("BaselineB", "Baseline"),
    ("M3", "Ours"),
    ("M4", "Ours + PA"),
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})


def try_write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    try:
        write_csv(path, rows, fieldnames)
    except PermissionError:
        print(f"skip locked file: {path}")


def try_write_md(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    try:
        write_md(path, rows, fieldnames)
    except PermissionError:
        print(f"skip locked file: {path}")


def write_md(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    fieldnames = list(fieldnames)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(key)) for key in fieldnames) + " |\n")


def normalize_strecha_scene(value: str) -> str:
    text = value.replace("strecha_", "").replace("_", "-")
    text = text.replace("fountain-P11", "Fountain-P11")
    text = text.replace("entry-P10", "Entry-P10")
    text = text.replace("Herz-Jesus-P8", "Herz-Jesus-P8")
    text = text.replace("Herz-Jesus-P25", "Herz-Jesus-P25")
    text = text.replace("Castle-P19", "Castle-P19")
    text = text.replace("Castle-P30-first29", "Castle-P30")
    text = text.replace("castle-P19", "Castle-P19")
    text = text.replace("castle-P30", "Castle-P30")
    return text


def normalize_dtu_scene(value: str) -> str:
    return value.replace("DTU_", "")


def sorted_rows(rows: List[Dict[str, Any]], scene_order: List[str]) -> List[Dict[str, Any]]:
    scene_rank = {scene: idx for idx, scene in enumerate(scene_order)}
    method_rank = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            scene_rank.get(str(row.get("scene", "")), 10_000),
            method_rank.get(str(row.get("method", "")), 10_000),
        ),
    )


def metric_wide_table(
    rows: List[Dict[str, Any]],
    scene_order: List[str],
    metric: str,
    *,
    include_colmap: bool = False,
) -> List[Dict[str, Any]]:
    methods = list(WIDE_METHODS)
    if include_colmap:
        methods.append(("COLMAP", "COLMAP"))
    lookup = {(row["scene"], row["method"]): row for row in rows}
    out: List[Dict[str, Any]] = []
    for scene in scene_order:
        record: Dict[str, Any] = {"Dataset": scene}
        for method, label in methods:
            row = lookup.get((scene, method), {})
            record[label] = row.get(metric)
        out.append(record)
    return out


def write_metric_tables(
    *,
    prefix: str,
    rows: List[Dict[str, Any]],
    scene_order: List[str],
    metrics: List[tuple[str, str]],
) -> None:
    for metric, title in metrics:
        table = metric_wide_table(rows, scene_order, metric, include_colmap=True)
        fields = ["Dataset", "Baseline", "Ours", "Ours + PA", "COLMAP"]
        try_write_csv(OUT_DIR / f"{prefix}_{metric}.csv", table, fields)
        try_write_md(OUT_DIR / f"{prefix}_{metric}.md", table, fields)

    combined_md = OUT_DIR / f"{prefix}_metric_tables.md"
    try:
        with combined_md.open("w", encoding="utf-8") as f:
            for metric, title in metrics:
                table = metric_wide_table(rows, scene_order, metric, include_colmap=True)
                fields = ["Dataset", "Baseline", "Ours", "Ours + PA", "COLMAP"]
                f.write(f"## {title}\n\n")
                f.write("| " + " | ".join(fields) + " |\n")
                f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
                for row in table:
                    f.write("| " + " | ".join(fmt(row.get(key)) for key in fields) + " |\n")
                f.write("\n")
    except PermissionError:
        print(f"skip locked file: {combined_md}")


def build_tables() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ours = read_csv(OURS_CSV)
    colmap = read_csv(COLMAP_CSV)
    strecha_rows: List[Dict[str, Any]] = []
    dtu_rows: List[Dict[str, Any]] = []

    for row in ours:
        method = row.get("method", "")
        group = row.get("scene_group", "")
        if group == "strecha":
            scene = normalize_strecha_scene(row.get("scene_name", ""))
            if scene in STRECHA_ORDER:
                strecha_rows.append(
                    {
                        "scene": scene,
                        "method": method,
                        "camera_position_mean_error_mm": row.get("camera_position_mean_error_mm"),
                        "rotation_mean_deg": row.get("rotation_mean_error_deg"),
                        "time_sec": row.get("runtime_total_sec"),
                        "peak_memory_mb": row.get("peak_memory_mb"),
                    }
                )
        elif group == "DTU":
            scene = normalize_dtu_scene(row.get("scene_name", ""))
            if scene in DTU_ORDER:
                dtu_rows.append(
                    {
                        "scene": scene,
                        "method": method,
                        "camera_position_ate_rmse_mm": row.get("camera_position_ate_rmse_mm"),
                        "rotation_ate_rmse_deg": row.get("rotation_ate_rmse_deg"),
                        "time_sec": row.get("runtime_total_sec"),
                        "peak_memory_mb": row.get("peak_memory_mb"),
                    }
                )

    for row in colmap:
        dataset = row.get("dataset", "")
        if dataset == "strecha":
            scene = normalize_strecha_scene(row.get("scene", ""))
            if scene in STRECHA_ORDER:
                strecha_rows.append(
                    {
                        "scene": scene,
                        "method": "COLMAP",
                        "camera_position_mean_error_mm": row.get("translation_mean_mm"),
                        "rotation_mean_deg": row.get("rotation_mean_deg"),
                        "time_sec": row.get("mapper_time_sec"),
                        "peak_memory_mb": float(row.get("mapper_peak_memory_gb", 0.0)) * 1024.0,
                    }
                )
        elif dataset == "DTU":
            scene = row.get("scene", "")
            if scene in DTU_ORDER:
                dtu_rows.append(
                    {
                        "scene": scene,
                        "method": "COLMAP",
                        "camera_position_ate_rmse_mm": row.get("translation_ate_rmse_mm"),
                        "rotation_ate_rmse_deg": row.get("rotation_ate_rmse_deg"),
                        "time_sec": row.get("mapper_time_sec"),
                        "peak_memory_mb": float(row.get("mapper_peak_memory_gb", 0.0)) * 1024.0,
                    }
                )

    return sorted_rows(strecha_rows, STRECHA_ORDER), sorted_rows(dtu_rows, DTU_ORDER)


def main() -> None:
    strecha_rows, dtu_rows = build_tables()
    strecha_fields = [
        "scene",
        "method",
        "camera_position_mean_error_mm",
        "rotation_mean_deg",
        "time_sec",
        "peak_memory_mb",
    ]
    dtu_fields = [
        "scene",
        "method",
        "camera_position_ate_rmse_mm",
        "rotation_ate_rmse_deg",
        "time_sec",
        "peak_memory_mb",
    ]

    try_write_csv(OUT_DIR / "selected_comparison_strecha.csv", strecha_rows, strecha_fields)
    try_write_md(OUT_DIR / "selected_comparison_strecha.md", strecha_rows, strecha_fields)
    try_write_csv(OUT_DIR / "selected_comparison_dtu.csv", dtu_rows, dtu_fields)
    try_write_md(OUT_DIR / "selected_comparison_dtu.md", dtu_rows, dtu_fields)

    write_metric_tables(
        prefix="selected_strecha",
        rows=strecha_rows,
        scene_order=STRECHA_ORDER,
        metrics=[
            ("camera_position_mean_error_mm", "Camera Position Mean Error (mm) on Strecha Dataset"),
            ("rotation_mean_deg", "Rotation Mean Error (deg) on Strecha Dataset"),
            ("time_sec", "Time (sec) on Strecha Dataset"),
            ("peak_memory_mb", "Peak Memory (MB) on Strecha Dataset"),
        ],
    )
    write_metric_tables(
        prefix="selected_dtu",
        rows=dtu_rows,
        scene_order=DTU_ORDER,
        metrics=[
            ("camera_position_ate_rmse_mm", "Camera Position ATE RMSE (mm) on DTU Dataset"),
            ("rotation_ate_rmse_deg", "Rotation ATE RMSE (deg) on DTU Dataset"),
            ("time_sec", "Time (sec) on DTU Dataset"),
            ("peak_memory_mb", "Peak Memory (MB) on DTU Dataset"),
        ],
    )

    print(OUT_DIR / "selected_comparison_strecha.csv")
    print(OUT_DIR / "selected_comparison_dtu.csv")
    print(OUT_DIR / "selected_strecha_metric_tables.md")
    print(OUT_DIR / "selected_dtu_metric_tables.md")


if __name__ == "__main__":
    main()
