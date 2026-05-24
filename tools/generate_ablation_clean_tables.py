#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from validation_search import load_structured_config, resolve_path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def quantile(values: List[float], q: float) -> Optional[float]:
    clean = sorted(v for v in values if v is not None and not math.isnan(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return statistics.mean(clean) if clean else None


def std_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return statistics.pstdev(clean) if clean else None


def max_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return max(clean) if clean else None


def median_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def run_docs_by_id(matrix: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(run["id"]): run for run in matrix.get("runs", [])}


def run_output_root(matrix: Dict[str, Any], config_dir: Path, run_doc: Dict[str, Any]) -> Path:
    base = Path(resolve_path(matrix["output_root_base"], str(config_dir))).resolve()
    return base / str(run_doc["output_subdir"])


def candidate_validation_path(run_root: Path) -> Path:
    return run_root / "trials" / "trial_0000" / "validation_results.json"


def read_nested_float(payload: Dict[str, Any], *keys: str) -> Optional[float]:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return as_float(cur)


def load_scene_extra(scene: Dict[str, Any]) -> Dict[str, Optional[float]]:
    summary_path_raw = scene.get("summary_json")
    if not summary_path_raw:
        return {}
    summary_path = Path(str(summary_path_raw))
    if not summary_path.exists():
        return {}
    summary = load_json(summary_path)
    scene_dir = summary_path.parent
    rraa_stats_path = scene_dir / "rraa_output" / "R_abs_stats.json"
    degeneracy_path = scene_dir / "pose_only" / "ligt_degeneracy_stats.json"

    rraa_stats = load_json(rraa_stats_path) if rraa_stats_path.exists() else {}
    degeneracy = load_json(degeneracy_path) if degeneracy_path.exists() else {}

    graph = rraa_stats.get("graph_before_filter") if isinstance(rraa_stats, dict) else {}
    residual_stats = degeneracy.get("ligt_residual_final_stats") if isinstance(degeneracy, dict) else {}

    stage_elapsed = [
        as_float(stage.get("elapsed_sec"))
        for stage in scene.get("stages", [])
        if isinstance(stage, dict) and stage.get("status") != "reused"
    ]
    stage_peaks = [
        as_float(stage.get("peak_working_set_mb"))
        for stage in scene.get("stages", [])
        if isinstance(stage, dict) and stage.get("status") != "reused"
    ]
    runtime_total_sec = as_float(summary.get("runtime_total_sec"))
    if runtime_total_sec is None and any(v is not None for v in stage_elapsed):
        runtime_total_sec = sum(v for v in stage_elapsed if v is not None)
    peak_memory_mb = as_float(summary.get("peak_memory_mb"))
    if peak_memory_mb is None and any(v is not None for v in stage_peaks):
        peak_memory_mb = max(v for v in stage_peaks if v is not None)

    return {
        "runtime_total_sec": runtime_total_sec,
        "peak_memory_mb": peak_memory_mb,
        "mean_edge_count": as_float(rraa_stats.get("input_edges") if isinstance(rraa_stats, dict) else None),
        "largest_component_ratio": as_float(graph.get("largest_component_ratio") if isinstance(graph, dict) else summary.get("rraa_graph_largest_component_ratio")),
        "equations_kept": as_float(degeneracy.get("equations_kept") if isinstance(degeneracy, dict) else summary.get("ligt_equations_kept")),
        "tracks_kept": as_float(degeneracy.get("tracks_kept") if isinstance(degeneracy, dict) else summary.get("ligt_tracks_kept")),
        "ligt_residual_median": read_nested_float({"x": residual_stats}, "x", "median"),
        "ligt_residual_p90": read_nested_float({"x": residual_stats}, "x", "q90"),
        "ligt_residual_p99": read_nested_float({"x": residual_stats}, "x", "q99"),
    }


def summarize_run(matrix: Dict[str, Any], config_dir: Path, run_doc: Dict[str, Any], label: str, title: str = "") -> Dict[str, Any]:
    run_root = run_output_root(matrix, config_dir, run_doc)
    validation_path = candidate_validation_path(run_root)
    params = dict(matrix.get("defaults") or {})
    params.update(run_doc.get("params") or {})

    row: Dict[str, Any] = {
        "label": label,
        "title": title or run_doc.get("description") or run_doc["id"],
        "run_id": run_doc["id"],
        "status": "missing" if not validation_path.exists() else "ok",
        "deltas_tracks": params.get("deltas_tracks"),
        "deltas_rraa": params.get("deltas_rraa"),
        "min_score": params.get("min_score"),
        "min_inliers_tracks": params.get("min_inliers_tracks"),
        "min_inliers_map": params.get("min_inliers_map"),
        "irls_iters": params.get("irls_iters"),
        "irls_huber_k": params.get("irls_huber_k"),
        "source": str(run_root),
    }
    if not validation_path.exists():
        return row

    validation = load_json(validation_path)
    candidate = (validation.get("candidates") or [{}])[0]
    scenes = candidate.get("scenes") or []
    best_summary = validation.get("best_summary") or candidate.get("summary") or {}

    trans_ratio = [as_float(s.get("translation_vs_colmap_ratio")) for s in scenes if s.get("status") == "ok"]
    trans_mm = [as_float(s.get("translation_mm_median")) for s in scenes if s.get("status") == "ok"]
    trans_mm_mean = [as_float(s.get("translation_mm_mean")) for s in scenes if s.get("status") == "ok"]
    rot_median = [as_float(s.get("rotation_median_deg")) for s in scenes if s.get("status") == "ok"]
    rot_mean = [as_float(s.get("rotation_mean_deg")) for s in scenes if s.get("status") == "ok"]
    extras = [load_scene_extra(s) for s in scenes if s.get("status") == "ok"]

    for prefix, values in [
        ("trans_ratio", trans_ratio),
        ("trans_mm_median", trans_mm),
        ("trans_mm_mean", trans_mm_mean),
        ("rot_median_deg", rot_median),
        ("rot_mean_deg", rot_mean),
    ]:
        clean = [v for v in values if v is not None]
        row[f"{prefix}_mean"] = mean_or_none(clean)
        row[f"{prefix}_median"] = median_or_none(clean)
        row[f"{prefix}_std"] = std_or_none(clean)
        row[f"{prefix}_worst"] = max_or_none(clean)
        row[f"{prefix}_p90"] = quantile(clean, 0.90)
        row[f"{prefix}_p95"] = quantile(clean, 0.95)
        row[f"{prefix}_p99"] = quantile(clean, 0.99)

    for key in [
        "runtime_total_sec",
        "peak_memory_mb",
        "mean_edge_count",
        "largest_component_ratio",
        "equations_kept",
        "tracks_kept",
        "ligt_residual_median",
        "ligt_residual_p90",
        "ligt_residual_p99",
    ]:
        values = [extra.get(key) for extra in extras]
        row[f"{key}_mean"] = mean_or_none(values)
        if key == "largest_component_ratio":
            row[f"{key}_min"] = min([v for v in values if v is not None], default=None)

    row.update({
        "score": as_float(best_summary.get("score")),
        "failure_count": as_float(best_summary.get("failure_count")),
        "failure_rate": as_float(best_summary.get("failure_rate")),
        "skip_count": as_float(best_summary.get("skip_count")),
        "scene_count": as_float(best_summary.get("scene_count")),
        "valid_scene_count": as_float(best_summary.get("valid_scene_count")),
    })
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key), digits=6) for key in fieldnames})


def write_md(path: Path, title: str, rows: List[Dict[str, Any]]) -> None:
    compact_cols = [
        "label",
        "title",
        "trans_ratio_mean",
        "trans_ratio_median",
        "trans_ratio_std",
        "trans_ratio_worst",
        "rot_median_deg_mean",
        "rot_median_deg_median",
        "rot_median_deg_std",
        "rot_median_deg_worst",
        "failure_count",
        "mean_edge_count_mean",
        "largest_component_ratio_mean",
        "equations_kept_mean",
        "ligt_residual_median_mean",
        "score",
    ]
    header = "| " + " | ".join(compact_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(compact_cols)) + " |"
    lines = [f"# {title}", "", header, sep]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col), digits=3) for col in compact_cols) + " |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_table(matrix: Dict[str, Any], config_dir: Path, table_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    runs = run_docs_by_id(matrix)
    rows: List[Dict[str, Any]] = []
    for item in table_items:
        run_id = str(item["source_run_id"])
        run_doc = runs.get(run_id)
        if run_doc is None:
            rows.append({"label": item.get("label", run_id), "title": item.get("title", ""), "run_id": run_id, "status": "unknown_run"})
            continue
        rows.append(summarize_run(matrix, config_dir, run_doc, str(item.get("label", run_id)), str(item.get("title", ""))))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate clean paper ablation tables with extended metrics.")
    ap.add_argument("--config", default="configs/paper_v2/experiments_ablation_clean_main.json")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    matrix = load_structured_config(args.config)
    config_dir = Path(matrix["_config_dir"]).resolve()
    output_root = Path(resolve_path(matrix["output_root_base"], str(config_dir))).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else output_root / "tables"

    logical_tables = matrix.get("logical_tables") or {}
    all_outputs: Dict[str, List[Dict[str, Any]]] = {}
    for table_name, items in logical_tables.items():
        rows = build_table(matrix, config_dir, list(items))
        all_outputs[table_name] = rows
        write_csv(out_dir / f"table_{table_name}.csv", rows)
        write_md(out_dir / f"table_{table_name}.md", table_name, rows)

    readme_lines = [
        "# Clean Main Ablation Tables",
        "",
        "Generated tables include extended metrics for selective paper reporting.",
        "",
        "Primary translation columns use `translation_vs_colmap_ratio` for consistency with the staged-search objective.",
        "`trans_mm_*` columns are also exported in CSV files.",
        "",
        "LiGT residual columns are populated only for runs generated after residual dumping was added to `ligt_degeneracy_stats.json`.",
        "",
        "Tables:",
    ]
    for table_name in all_outputs:
        readme_lines.append(f"- `table_{table_name}.csv` / `table_{table_name}.md`")
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
