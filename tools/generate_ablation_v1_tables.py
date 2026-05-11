#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path
    try:
        with target.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return target
    except PermissionError:
        target = path.with_name(f"{path.stem}_updated{path.suffix}")
        with target.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"warning: {path.name} is locked; wrote {target.name} instead.")
        return target


def save_md(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path
    try:
        target.write_text(text, encoding="utf-8")
        return target
    except PermissionError:
        target = path.with_name(f"{path.stem}_updated{path.suffix}")
        target.write_text(text, encoding="utf-8")
        print(f"warning: {path.name} is locked; wrote {target.name} instead.")
        return target


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value in (None, "", "null"):
        return default
    try:
        return float(value)
    except Exception:
        return default


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def compute_stats(values: Iterable[float]) -> Dict[str, float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v)) and not math.isinf(float(v))]
    if not vals:
        return {
            "mean": float("nan"),
            "rmse": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    sorted_vals = sorted(vals)

    def percentile(q: float) -> float:
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        idx = (len(sorted_vals) - 1) * q
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return sorted_vals[lo]
        frac = idx - lo
        return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

    return {
        "mean": mean(vals),
        "rmse": math.sqrt(sum(v * v for v in vals) / len(vals)),
        "median": median(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "p90": percentile(0.90),
        "p95": percentile(0.95),
    }


def scene_ref_key(scene_key: str) -> str:
    if scene_key.startswith("strecha_"):
        tail = scene_key[len("strecha_"):].replace("_", "-")
        tail = tail.replace("Castle-P30-first29", "Castle-P30-first29")
        tail = tail.replace("Herz-Jesus", "Herz-Jesus")
        return f"strecha::{tail}"
    if scene_key.startswith("DTU_"):
        tail = scene_key[len("DTU_"):]
        return f"DTU::{tail}"
    raise KeyError(scene_key)


def load_colmap_refs(path: Path) -> Dict[str, float]:
    doc = load_json(path)
    refs = {}
    for key, payload in doc.items():
        refs[key] = float(payload["translation_mm_median"])
    return refs


def summarize_run_group(run_root: Path, group_name: str, refs: Dict[str, float], label: str) -> Dict[str, Any]:
    group_dir = run_root / group_name
    per_scene = []
    for scene_dir in sorted(group_dir.iterdir()) if group_dir.exists() else []:
        summary_path = scene_dir / "experiment_summary.json"
        if not summary_path.exists():
            continue
        try:
            ref_key = scene_ref_key(scene_dir.name)
        except KeyError:
            continue
        payload = load_json(summary_path)
        trans_mm = as_float(payload.get("translation_eval_mm_median"))
        rot_deg = as_float(payload.get("rotation_eval_median_deg"))
        runtime_sec = as_float(payload.get("runtime_total_sec"))
        peak_mem_mb = as_float(payload.get("peak_memory_mb"))
        ref = refs.get(ref_key)
        per_scene.append(
            {
                "scene_key": scene_dir.name,
                "translation_mm_median": trans_mm,
                "rotation_median_deg": rot_deg,
                "runtime_total_sec": runtime_sec,
                "peak_memory_mb": peak_mem_mb,
                "pa_status": payload.get("pa_status"),
            }
        )

    valid_trans = [row["translation_mm_median"] for row in per_scene if row["translation_mm_median"] is not None]
    valid_rot = [row["rotation_median_deg"] for row in per_scene if row["rotation_median_deg"] is not None]
    valid_runtime = [row["runtime_total_sec"] for row in per_scene if row["runtime_total_sec"] is not None]
    valid_mem = [row["peak_memory_mb"] for row in per_scene if row["peak_memory_mb"] is not None]
    trans_stats = compute_stats(valid_trans)
    rot_stats = compute_stats(valid_rot)
    runtime_stats = compute_stats(valid_runtime)
    mem_stats = compute_stats(valid_mem)
    return {
        "label": label,
        "source": str(group_dir),
        "translation_mm_median_mean": trans_stats["mean"],
        "translation_mm_median_rmse": trans_stats["rmse"],
        "translation_mm_median_median": trans_stats["median"],
        "translation_mm_median_std": trans_stats["std"],
        "translation_mm_median_p90": trans_stats["p90"],
        "translation_mm_median_p95": trans_stats["p95"],
        "rotation_deg_median_mean": rot_stats["mean"],
        "rotation_deg_median_rmse": rot_stats["rmse"],
        "rotation_deg_median_median": rot_stats["median"],
        "rotation_deg_median_std": rot_stats["std"],
        "rotation_deg_median_p90": rot_stats["p90"],
        "rotation_deg_median_p95": rot_stats["p95"],
        "runtime_total_sec_mean": runtime_stats["mean"],
        "runtime_total_sec_rmse": runtime_stats["rmse"],
        "runtime_total_sec_median": runtime_stats["median"],
        "runtime_total_sec_std": runtime_stats["std"],
        "runtime_total_sec_p90": runtime_stats["p90"],
        "runtime_total_sec_p95": runtime_stats["p95"],
        "peak_memory_mb_mean": mem_stats["mean"],
        "peak_memory_mb_rmse": mem_stats["rmse"],
        "peak_memory_mb_median": mem_stats["median"],
        "peak_memory_mb_std": mem_stats["std"],
        "peak_memory_mb_p90": mem_stats["p90"],
        "peak_memory_mb_p95": mem_stats["p95"],
    }


def summarize_validation_best(results: Dict[str, Any], label: str, source: str) -> Dict[str, Any]:
    summary = results.get("validation_summary", results.get("best_summary", {}))
    scenes = results.get("scene_breakdown", [])
    valid_trans = [as_float(row.get("translation_mm_median")) for row in scenes if as_float(row.get("translation_mm_median")) is not None]
    valid_rot = [as_float(row.get("rotation_median_deg")) for row in scenes if as_float(row.get("rotation_median_deg")) is not None]
    valid_runtime = [as_float(row.get("runtime_total_sec")) for row in scenes if as_float(row.get("runtime_total_sec")) is not None]
    valid_mem = [as_float(row.get("peak_memory_mb")) for row in scenes if as_float(row.get("peak_memory_mb")) is not None]
    return {
        "label": label,
        "source": source,
        "translation_mm_median_mean": mean(valid_trans) if valid_trans else float("nan"),
        "rotation_deg_median_mean": mean(valid_rot) if valid_rot else as_float(summary.get("mean_rotation_median_deg"), float("nan")),
        "runtime_total_sec_mean": mean(valid_runtime) if valid_runtime else float("nan"),
        "peak_memory_mb_mean": mean(valid_mem) if valid_mem else float("nan"),
    }


def summarize_candidate_from_validation_csv(csv_path: Path, recommended_rank: int, label: str) -> Dict[str, Any]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if recommended_rank < 1 or recommended_rank > len(rows):
        raise ValueError(f"recommended_rank out of range for {csv_path}: {recommended_rank}")
    row = rows[recommended_rank - 1]
    return {
        "label": label,
        "source": str(csv_path),
        "translation_mm_median_mean": float("nan"),
        "rotation_deg_median_mean": as_float(row.get("mean_rotation_median_deg"), float("nan")),
        "runtime_total_sec_mean": float("nan"),
        "peak_memory_mb_mean": float("nan"),
    }


def summarize_candidate_run_dir(candidate_root: Path, label: str) -> Dict[str, Any]:
    valid_trans = []
    valid_rot = []
    valid_runtime = []
    valid_mem = []
    for scene_dir in sorted(candidate_root.iterdir()) if candidate_root.exists() else []:
        summary_path = scene_dir / "experiment_summary.json"
        if not summary_path.exists():
            continue
        payload = load_json(summary_path)
        trans = as_float(payload.get("translation_eval_mm_median"))
        rot = as_float(payload.get("rotation_eval_median_deg"))
        runtime = as_float(payload.get("runtime_total_sec"))
        mem = as_float(payload.get("peak_memory_mb"))
        if trans is not None:
            valid_trans.append(trans)
        if rot is not None:
            valid_rot.append(rot)
        if runtime is not None:
            valid_runtime.append(runtime)
        if mem is not None:
            valid_mem.append(mem)
    return {
        "label": label,
        "source": str(candidate_root),
        "translation_mm_median_mean": mean(valid_trans) if valid_trans else float("nan"),
        "rotation_deg_median_mean": mean(valid_rot) if valid_rot else float("nan"),
        "runtime_total_sec_mean": mean(valid_runtime) if valid_runtime else float("nan"),
        "peak_memory_mb_mean": mean(valid_mem) if valid_mem else float("nan"),
    }


def write_table(path: Path, rows: List[Dict[str, Any]], *, extended: bool = False) -> None:
    if extended:
        fieldnames = [
            "label",
            "translation_mm_median_mean",
            "translation_mm_median_rmse",
            "translation_mm_median_median",
            "translation_mm_median_std",
            "translation_mm_median_p90",
            "translation_mm_median_p95",
            "rotation_deg_median_mean",
            "rotation_deg_median_rmse",
            "rotation_deg_median_median",
            "rotation_deg_median_std",
            "rotation_deg_median_p90",
            "rotation_deg_median_p95",
            "runtime_total_sec_mean",
            "runtime_total_sec_rmse",
            "runtime_total_sec_median",
            "runtime_total_sec_std",
            "runtime_total_sec_p90",
            "runtime_total_sec_p95",
            "peak_memory_mb_mean",
            "peak_memory_mb_rmse",
            "peak_memory_mb_median",
            "peak_memory_mb_std",
            "peak_memory_mb_p90",
            "peak_memory_mb_p95",
            "source",
        ]
    else:
        fieldnames = [
            "label",
            "translation_mm_median_mean",
            "rotation_deg_median_mean",
            "runtime_total_sec_mean",
            "peak_memory_mb_mean",
            "source",
        ]
    normalized = []
    for row in rows:
        item = {
                "label": row.get("label"),
                "translation_mm_median_mean": fmt(row.get("translation_mm_median_mean")),
                "source": row.get("source", ""),
            }
        if extended:
            item.update(
                {
                    "translation_mm_median_rmse": fmt(row.get("translation_mm_median_rmse")),
                    "translation_mm_median_median": fmt(row.get("translation_mm_median_median")),
                    "translation_mm_median_std": fmt(row.get("translation_mm_median_std")),
                    "translation_mm_median_p90": fmt(row.get("translation_mm_median_p90")),
                    "translation_mm_median_p95": fmt(row.get("translation_mm_median_p95")),
                    "rotation_deg_median_mean": fmt(row.get("rotation_deg_median_mean")),
                    "rotation_deg_median_rmse": fmt(row.get("rotation_deg_median_rmse")),
                    "rotation_deg_median_median": fmt(row.get("rotation_deg_median_median")),
                    "rotation_deg_median_std": fmt(row.get("rotation_deg_median_std")),
                    "rotation_deg_median_p90": fmt(row.get("rotation_deg_median_p90")),
                    "rotation_deg_median_p95": fmt(row.get("rotation_deg_median_p95")),
                    "runtime_total_sec_mean": fmt(row.get("runtime_total_sec_mean")),
                    "runtime_total_sec_rmse": fmt(row.get("runtime_total_sec_rmse")),
                    "runtime_total_sec_median": fmt(row.get("runtime_total_sec_median")),
                    "runtime_total_sec_std": fmt(row.get("runtime_total_sec_std")),
                    "runtime_total_sec_p90": fmt(row.get("runtime_total_sec_p90")),
                    "runtime_total_sec_p95": fmt(row.get("runtime_total_sec_p95")),
                    "peak_memory_mb_mean": fmt(row.get("peak_memory_mb_mean")),
                    "peak_memory_mb_rmse": fmt(row.get("peak_memory_mb_rmse")),
                    "peak_memory_mb_median": fmt(row.get("peak_memory_mb_median")),
                    "peak_memory_mb_std": fmt(row.get("peak_memory_mb_std")),
                    "peak_memory_mb_p90": fmt(row.get("peak_memory_mb_p90")),
                    "peak_memory_mb_p95": fmt(row.get("peak_memory_mb_p95")),
                }
            )
        else:
            item.update(
                {
                    "rotation_deg_median_mean": fmt(row.get("rotation_deg_median_mean")),
                    "runtime_total_sec_mean": fmt(row.get("runtime_total_sec_mean")),
                    "peak_memory_mb_mean": fmt(row.get("peak_memory_mb_mean")),
                }
            )
        normalized.append(item)
    save_csv(path, normalized, fieldnames)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate first-version ablation tables.")
    ap.add_argument("--out_dir", default="runs/paper_v2/ablation_v1/tables")
    args = ap.parse_args()

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    refs = load_colmap_refs(REPO_ROOT / "runs/paper_v2/final_benchmark_report/colmap_translation_references_12scenes.json")

    quality_rows = [
        summarize_run_group(REPO_ROOT / "runs/paper_v2/ablation_v1/quality", "Q0_no_quality", refs, "Q0"),
        summarize_run_group(REPO_ROOT / "runs/paper_v2/ablation_v1/quality", "Q1_qpair_only", refs, "Q1"),
        summarize_run_group(REPO_ROOT / "runs/paper_v2/ablation_v1/quality", "Q2_qtrack_only", refs, "Q2"),
        summarize_run_group(REPO_ROOT / "runs/paper_v2/ablation_v1/quality", "Q3_full_quality", refs, "Q3"),
    ]
    write_table(out_dir / "table_Q_quality_ablation.csv", quality_rows, extended=True)

    degeneracy_rows = [
        summarize_run_group(REPO_ROOT / "runs/paper_v2/ablation_v1/degeneracy", "D0_no_degeneracy", refs, "D0"),
        summarize_run_group(REPO_ROOT / "runs/paper_v2/ablation_v1/degeneracy", "D1_geom_filter_only", refs, "D1"),
        summarize_run_group(REPO_ROOT / "runs/paper_v2/ablation_v1/degeneracy", "D2_basepair_only", refs, "D2"),
        summarize_run_group(REPO_ROOT / "runs/paper_v2/ablation_v1/degeneracy", "D3_robust_only", refs, "D3"),
        summarize_run_group(REPO_ROOT / "runs/paper_v2/ablation_v1/degeneracy", "D4_full_degeneracy", refs, "D4"),
    ]
    write_table(out_dir / "table_D_degeneracy_ablation.csv", degeneracy_rows, extended=True)

    auto_rows = [
        summarize_run_group(REPO_ROOT / "runs/paper_v2/main", "rraa_ligt", refs, "A0_manual_baseline"),
        summarize_run_group(REPO_ROOT / "runs/paper_v2/main_phase2_5_frontend", "rraa_ligt", refs, "A1_optuna_best"),
        summarize_candidate_run_dir(
            REPO_ROOT / "runs/paper_v2/optuna_frontend_qpair_phase2_6c_vs_colmap_paircache_12scenes/trials/trial_0000/candidates/candidate_000",
            "A2_fallback_policy",
        ),
        summarize_candidate_run_dir(
            REPO_ROOT / "runs/paper_v2/recommended_top3_validation/candidate_rank_01/candidates/candidate_000",
            "A3_recommender_top1",
        ),
        summarize_candidate_run_dir(
            REPO_ROOT / "runs/paper_v2/recommended_top3_validation/candidate_rank_03/candidates/candidate_000",
            "A4_final_protocol",
        ),
    ]
    write_table(out_dir / "table_A_auto_strategy_ablation.csv", auto_rows)

    stable_pa_candidate_root = REPO_ROOT / "runs/paper_v2/ablation_v1/stable_protocol_with_pa/candidates/candidate_000"
    system_rows = [
        next(row for row in quality_rows if row["label"] == "Q0") | {"label": "S0_base"},
        next(row for row in quality_rows if row["label"] == "Q3") | {"label": "S1_quality"},
        next(row for row in degeneracy_rows if row["label"] == "D4") | {"label": "S2_degeneracy"},
        next(row for row in auto_rows if row["label"] == "A4_final_protocol") | {"label": "S3_strategy"},
    ]
    if stable_pa_candidate_root.exists():
        system_rows.append(summarize_candidate_run_dir(stable_pa_candidate_root, "S4_refinement"))
    write_table(out_dir / "table_S_system_accumulation.csv", system_rows)

    md_lines = [
        "# Ablation V1 Tables",
        "",
        "Generated files:",
        "- `table_Q_quality_ablation.csv`",
        "- `table_D_degeneracy_ablation.csv`",
        "- `table_A_auto_strategy_ablation.csv`",
        "- `table_S_system_accumulation.csv`",
        "",
        "Interpretation notes:",
        "- All tables now keep only four metrics: translation error, rotation error, runtime, and peak memory.",
        "- `A` and `S` tables read metrics directly from concrete candidate run directories instead of unstable robust summary JSONs.",
    ]
    save_md(out_dir / "README.md", "\n".join(md_lines) + "\n")
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
