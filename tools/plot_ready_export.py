#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Any, Dict, List


NUMERIC_FIELDS = [
    "rotation_median_deg",
    "rotation_mean_deg",
    "rotation_p90_deg",
    "rotation_max_deg",
    "translation_raw_median",
    "translation_raw_mean",
    "translation_raw_rmse",
    "translation_raw_p90",
    "translation_raw_max",
    "translation_mm_median",
    "translation_mm_mean",
    "translation_mm_rmse",
    "translation_mm_p90",
    "translation_mm_max",
    "rraa_kept_ratio",
    "rraa_graph_component_count",
    "rraa_graph_largest_component_ratio",
    "rraa_anchor_degree",
    "rraa_anchor_component_ratio",
    "rraa_anchor_sensitivity_median_deg",
    "rraa_anchor_sensitivity_p90_deg",
    "tracks_count",
    "track_len_median",
    "q_pair_median",
    "q_track_count",
    "q_track_median",
    "q_track_p10",
    "q_track_p90",
    "ligt_tracks_kept",
    "ligt_equations_kept",
    "ligt_g_reject_ratio",
    "qtrack_reject_ratio",
    "kept_track_ratio",
    "pa_final_rmse",
    "pa_accept_iters",
    "delta_translation_mm_median",
    "delta_translation_mm_p90",
]


def load_rows(path: str) -> List[Dict[str, Any]]:
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8-sig") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return payload["rows"]
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Unsupported JSON structure: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> Any:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metric_group(name: str) -> str:
    if name.startswith("rotation_"):
        return "rotation"
    if name.startswith("translation_") or name.startswith("delta_translation_"):
        return "translation"
    if name.startswith("rraa_"):
        return "rraa_graph"
    if name.startswith("pa_"):
        return "pa"
    return "quality"


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-export paper summary wide table into plot-ready long tables.")
    ap.add_argument("--input", required=True, help="paper_summary_main.csv or paper_summary_main.json")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    rows = load_rows(args.input)
    long_rows: List[Dict[str, Any]] = []
    gain_rows: List[Dict[str, Any]] = []
    base_fields = [
        "dataset",
        "scene",
        "scene_slug",
        "experiment_group",
        "rotation_source",
        "use_pa",
        "qtrack_mode",
        "enable_quality_weighting",
        "rraa_use_qpair_weight",
        "u_min",
        "g_min",
        "threshold_group",
        "baseline_experiment_group",
        "status",
    ]

    for row in rows:
        normalized = dict(row)
        for key in NUMERIC_FIELDS:
            normalized[key] = to_float(normalized.get(key))
        for metric_name in NUMERIC_FIELDS:
            value = normalized.get(metric_name)
            if value is None:
                continue
            item = {k: normalized.get(k) for k in base_fields}
            item["metric_name"] = metric_name
            item["metric_value"] = value
            item["metric_group"] = metric_group(metric_name)
            long_rows.append(item)
        if normalized.get("delta_translation_mm_median") is not None or normalized.get("delta_translation_mm_p90") is not None:
            gain_rows.append(
                {
                    "dataset": normalized.get("dataset"),
                    "scene": normalized.get("scene"),
                    "experiment_group": normalized.get("experiment_group"),
                    "rotation_source": normalized.get("rotation_source"),
                    "baseline_experiment_group": normalized.get("baseline_experiment_group"),
                    "delta_translation_mm_median": normalized.get("delta_translation_mm_median"),
                    "delta_translation_mm_p90": normalized.get("delta_translation_mm_p90"),
                }
            )

    write_csv(
        os.path.join(args.output_dir, "paper_summary_long.csv"),
        long_rows,
        base_fields + ["metric_name", "metric_value", "metric_group"],
    )
    write_csv(
        os.path.join(args.output_dir, "paper_pa_gain.csv"),
        gain_rows,
        [
            "dataset",
            "scene",
            "experiment_group",
            "rotation_source",
            "baseline_experiment_group",
            "delta_translation_mm_median",
            "delta_translation_mm_p90",
        ],
    )
    print("saved:", os.path.join(args.output_dir, "paper_summary_long.csv"))
    print("saved:", os.path.join(args.output_dir, "paper_pa_gain.csv"))


if __name__ == "__main__":
    main()
