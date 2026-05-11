#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_scene_features(scene: Dict[str, Any]) -> Dict[str, Any]:
    fallback_tags = [str(item.get("tag")) for item in scene.get("fallbacks", []) if isinstance(item, dict) and item.get("tag")]
    features = {
        "scene_name": scene.get("scene_name"),
        "scene_group": scene.get("scene_group"),
        "scene_id": scene.get("scene_id"),
        "scene_status": scene.get("status"),
        "scene_translation_vs_colmap_ratio": scene.get("translation_vs_colmap_ratio"),
        "scene_translation_vs_colmap_gap": scene.get("translation_vs_colmap_gap"),
        "scene_translation_mm_median": scene.get("translation_mm_median"),
        "scene_translation_mm_mean": scene.get("translation_mm_mean"),
        "scene_rotation_median_deg": scene.get("rotation_median_deg"),
        "scene_rraa_kept_ratio": scene.get("rraa_kept_ratio"),
        "scene_rraa_reject_ratio": scene.get("rraa_reject_ratio"),
        "scene_qtrack_reject_ratio": scene.get("qtrack_reject_ratio"),
        "scene_g_reject_ratio": scene.get("g_reject_ratio"),
        "scene_primary_metric_value": scene.get("primary_metric_value"),
        "scene_fallback_used": scene.get("fallback_used"),
        "scene_fallback_count": len(scene.get("fallbacks", []) or []),
        "scene_fallback_tags": ",".join(sorted(fallback_tags)),
    }
    return features


def infer_policy_features(results: Dict[str, Any]) -> Dict[str, Any]:
    base_cfg = results.get("protocol") or {}
    if not isinstance(base_cfg, dict):
        base_cfg = {}
    fallback_specs = results.get("candidate_stage_fallbacks")
    if fallback_specs is None:
        config_path = results.get("config_path")
        if config_path:
            config_file = Path(str(config_path))
            if config_file.exists():
                config_payload = load_json(config_file)
                fallback_specs = ((config_payload.get("base_validation_config") or {}).get("candidate_stage_fallbacks"))
    if fallback_specs is None:
        fallback_specs = (results.get("base_validation_config_snapshot") or {}).get("candidate_stage_fallbacks")
    if fallback_specs is None:
        fallback_specs = base_cfg.get("candidate_stage_fallbacks")
    fallback_specs = fallback_specs or []

    enabled = bool(fallback_specs)
    fallback_spec = fallback_specs[0] if enabled and isinstance(fallback_specs[0], dict) else {}
    override_params = fallback_spec.get("override_params") if isinstance(fallback_spec, dict) else {}
    override_params = override_params if isinstance(override_params, dict) else {}

    return {
        "policy_rraa_fallback_enabled": int(enabled),
        "policy_rraa_fallback_trigger_stage": fallback_spec.get("trigger_stage") if enabled else "none",
        "policy_rraa_fallback_rerun_from_stage": fallback_spec.get("rerun_from_stage") if enabled else "none",
        "policy_rraa_fallback_deltas_rraa": override_params.get("deltas_rraa") if enabled else "none",
        "policy_effective_deltas_rraa_policy": (
            f"base_then_fallback:{override_params.get('deltas_rraa')}" if enabled else "base_only"
        ),
    }


def infer_fallback_diagnostics(scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
    fallback_scenes = [scene for scene in scenes if scene.get("fallback_used")]
    scene_count = max(len(scenes), 1)
    by_name = {str(scene.get("scene_name")): scene for scene in scenes}
    return {
        "diag_fallback_used_count": len(fallback_scenes),
        "diag_fallback_used_ratio": float(len(fallback_scenes) / scene_count),
        "diag_fallback_scene_names": ",".join(sorted(str(scene.get("scene_name")) for scene in fallback_scenes)),
        "diag_scan114_fallback_used": int(bool(by_name.get("DTU_scan114", {}).get("fallback_used"))),
        "diag_scan106_fallback_used": int(bool(by_name.get("DTU_scan106", {}).get("fallback_used"))),
    }


def collect_rows(study_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    trials_csv = study_root / "trials_summary.csv"
    rows_trial: List[Dict[str, Any]] = []
    rows_scene: List[Dict[str, Any]] = []
    with trials_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            trial_number = int(row["trial_number"])
            trial_dir = study_root / "trials" / f"trial_{trial_number:04d}"
            results_json = trial_dir / "validation_results.json"
            if not results_json.exists():
                continue
            results = load_json(results_json)
            candidates = results.get("candidates", [])
            if not candidates:
                continue
            candidate = candidates[0]
            summary = candidate.get("summary", {})
            params = candidate.get("params", {})
            scenes = candidate.get("scenes", [])
            base = {
                "study_name": study_root.name,
                "study_root": str(study_root),
                "trial_number": trial_number,
                "score": row.get("score"),
                "state": row.get("state"),
                "mean_primary_metric": row.get("mean_primary_metric"),
                "std_primary_metric": row.get("std_primary_metric"),
                "worst_primary_metric": row.get("worst_primary_metric"),
                "failure_rate": row.get("failure_rate"),
                "skip_rate": row.get("skip_rate"),
                "mean_rotation_median_deg": row.get("mean_rotation_median_deg"),
                "mean_reject_ratio": row.get("mean_reject_ratio"),
                "results_json": row.get("results_json"),
            }
            base.update({f"param_{k}": v for k, v in params.items()})
            base.update(infer_policy_features(results))
            base.update(infer_fallback_diagnostics(scenes))
            base.update({f"summary_{k}": v for k, v in summary.items()})
            rows_trial.append(base)

            for scene in scenes:
                scene_row = dict(base)
                scene_row.update(infer_scene_features(scene))
                rows_scene.append(scene_row)
    return rows_trial, rows_scene


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge multiple Optuna study outputs into training tables.")
    ap.add_argument("--study_root", action="append", required=True, help="Study root path. Pass multiple times.")
    ap.add_argument("--out_trial_csv", required=True, help="Merged trial-level CSV.")
    ap.add_argument("--out_scene_csv", required=True, help="Merged scene-level CSV.")
    args = ap.parse_args()

    merged_trials: List[Dict[str, Any]] = []
    merged_scenes: List[Dict[str, Any]] = []
    for study_root_text in args.study_root:
        study_root = Path(study_root_text).resolve()
        trial_rows, scene_rows = collect_rows(study_root)
        merged_trials.extend(trial_rows)
        merged_scenes.extend(scene_rows)

    if not merged_trials:
        raise SystemExit("No trials found across provided study roots.")

    write_csv(Path(args.out_trial_csv).resolve(), merged_trials)
    write_csv(Path(args.out_scene_csv).resolve(), merged_scenes)
    print("saved:", Path(args.out_trial_csv).resolve())
    print("saved:", Path(args.out_scene_csv).resolve())


if __name__ == "__main__":
    main()
