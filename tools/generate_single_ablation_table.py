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


def as_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except Exception:
        return None


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
        return {key: float("nan") for key in ("mean", "rmse", "median", "std", "p90", "p95")}
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


def summarize_group(run_root: Path, group_name: str, label: str) -> Dict[str, Any]:
    group_dir = run_root / group_name
    trans, rot, runtime, mem = [], [], [], []
    for scene_dir in sorted(group_dir.iterdir()) if group_dir.exists() else []:
        summary_path = scene_dir / "experiment_summary.json"
        if not summary_path.exists():
            continue
        payload = load_json(summary_path)
        t = as_float(payload.get("translation_eval_mm_median"))
        r = as_float(payload.get("rotation_eval_median_deg"))
        rt = as_float(payload.get("runtime_total_sec"))
        pm = as_float(payload.get("peak_memory_mb"))
        if t is not None:
            trans.append(t)
        if r is not None:
            rot.append(r)
        if rt is not None:
            runtime.append(rt)
        if pm is not None:
            mem.append(pm)

    trans_stats = compute_stats(trans)
    rot_stats = compute_stats(rot)
    runtime_stats = compute_stats(runtime)
    mem_stats = compute_stats(mem)
    row = {"label": label, "source": str(group_dir)}
    for prefix, stats in (
        ("translation_mm_median", trans_stats),
        ("rotation_deg_median", rot_stats),
        ("runtime_total_sec", runtime_stats),
        ("peak_memory_mb", mem_stats),
    ):
        for suffix, value in stats.items():
            row[f"{prefix}_{suffix}"] = value
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    normalized = []
    for row in rows:
        normalized.append({key: fmt(row.get(key)) for key in fieldnames})
        normalized[-1]["label"] = row["label"]
        normalized[-1]["source"] = row["source"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a single ablation CSV from experiment_summary files.")
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--groups", required=True, help="Comma-separated group names, e.g. D0_no_degeneracy,D1_geom_filter_only")
    ap.add_argument("--labels", required=True, help="Comma-separated row labels, e.g. D0,D1")
    args = ap.parse_args()

    groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    if len(groups) != len(labels):
        raise ValueError("groups and labels must have the same length")

    run_root = (REPO_ROOT / args.run_root).resolve()
    rows = [summarize_group(run_root, group, label) for group, label in zip(groups, labels)]
    write_csv((REPO_ROOT / args.out_csv).resolve(), rows)
    print(f"saved: {Path(args.out_csv)}")


if __name__ == "__main__":
    main()
