#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from validation_search import build_robust_config, dump_json, load_structured_config, resolve_path, run_search, write_results_csv

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "optuna is not installed in the current environment. "
        "Install it first, for example:\n"
        r"  .\.venv\Scripts\python.exe -m pip install optuna"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]


def infer_image_glob(dataset: str, scene_id: str, prepared_scene_dir: str) -> str:
    dataset_norm = str(dataset).lower()
    if dataset_norm == "strecha":
        return f"data/raw/strecha/{scene_id}/images/*.jpg"
    if dataset_norm == "dtu":
        return f"data/raw/DTU/Rectified/{scene_id}/rect_*_3_r5000.png"
    if dataset_norm == "eth3d":
        return f"data/raw/ETH3D/{scene_id}/images/*.jpg"
    return str(Path(prepared_scene_dir) / "images" / "*")


def build_validation_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    dataset = str(scene.get("dataset") or scene.get("group") or "")
    scene_id = str(scene.get("scene_id") or "")
    scene_name = str(scene.get("scene_key") or scene_id)
    group = "strecha" if dataset.lower() == "strecha" else dataset
    prepared_scene_dir = str(scene.get("prepared_scene_dir") or "")
    return {
        "name": scene_name,
        "group": group,
        "scene_id": scene_id,
        "image_glob": infer_image_glob(dataset, scene_id, prepared_scene_dir),
        "scene_root": scene.get("scene_root"),
        "prepared_scene_dir": prepared_scene_dir,
        "gt_unit_to_mm": scene.get("gt_unit_to_mm"),
        "metrics": {
            "rotation_json": "{candidate_root}\\rraa_output\\eval_rotation.json",
            "translation_json": "{candidate_root}\\pose_only\\eval_translation.json",
            "summary_json": "{candidate_root}\\experiment_summary.json",
            "ligt_json": "{candidate_root}\\pose_only\\quality_stats.json",
            "rraa_stats_json": "{candidate_root}\\rraa_output\\R_abs_stats.json",
        },
    }


def resolve_validation_scenes(base_validation: Dict[str, Any], config_dir: Path) -> List[Dict[str, Any]]:
    scenes = base_validation.get("scenes")
    if isinstance(scenes, list):
        return scenes
    if isinstance(scenes, str) and scenes.strip():
        scenes_path = Path(resolve_path(scenes, str(config_dir)))
        scene_doc = load_structured_config(str(scenes_path))
        raw_scenes = scene_doc.get("scenes")
        if not isinstance(raw_scenes, list) or not raw_scenes:
            raise ValueError(f"Scene preset must contain a non-empty scenes list: {scenes_path}")
        return [build_validation_scene(scene) for scene in raw_scenes]
    raise ValueError("base_validation_config.scenes must be a non-empty list or a path to a scene preset JSON/YAML")


def suggest_value(trial: "optuna.trial.Trial", name: str, spec: Dict[str, Any]) -> Any:
    kind = spec.get("type", "categorical")
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if kind == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        log = bool(spec.get("log", False))
        return trial.suggest_int(name, low, high, step=step, log=log)
    if kind == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        step = spec.get("step")
        log = bool(spec.get("log", False))
        if step is None:
            return trial.suggest_float(name, low, high, log=log)
        return trial.suggest_float(name, low, high, step=float(step), log=log)
    raise ValueError(f"Unsupported search space type for {name}: {kind}")


def build_single_candidate_config(base_config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(base_config)
    cfg["search_space"] = {k: [v] for k, v in params.items()}
    return cfg


def round_to_step(value: float, step: int) -> int:
    if step <= 1:
        return int(round(value))
    return int(round(float(value) / float(step)) * step)


def build_log_linear_min_inliers_map(params: Dict[str, Any], spec: Dict[str, Any]) -> str:
    import math

    gaps = [int(x) for x in spec.get("gaps", [1, 2, 3, 4, 5, 8, 13])]
    base = float(params[str(spec.get("base_param", "map_base"))])
    slope = float(params[str(spec.get("slope_param", "map_slope"))])
    cap = float(params[str(spec.get("cap_param", "map_cap"))])
    floor = float(params.get(str(spec.get("floor_param", "map_floor")), spec.get("floor", 8)))
    round_to = int(spec.get("round_to", 4))
    default_gap = int(spec.get("default_gap", max(gaps)))

    values: Dict[int, int] = {}
    prev = 0
    for gap in gaps:
        raw = base + slope * math.log2(max(float(gap), 1.0))
        val = round_to_step(min(max(raw, floor), cap), round_to)
        val = max(val, prev)
        values[gap] = int(val)
        prev = int(val)
    default = values.get(default_gap, values[max(gaps)])
    return ",".join([f"{gap}:{values[gap]}" for gap in gaps] + [f"default:{default}"])


def build_power_law_min_inliers_map(params: Dict[str, Any], spec: Dict[str, Any]) -> str:
    gaps = [int(x) for x in spec.get("gaps", [1, 2, 3, 4, 5, 8, 13])]
    t0 = float(params[str(spec.get("t0_param", "map_t0"))])
    alpha = float(params[str(spec.get("alpha_param", "map_alpha"))])
    t_max = float(params[str(spec.get("tmax_param", "map_tmax"))])
    t_min = float(params.get(str(spec.get("tmin_param", "map_tmin")), spec.get("t_min", 8)))
    round_to = int(spec.get("round_to", 6))
    default_gap = int(spec.get("default_gap", max(gaps)))

    values: Dict[int, int] = {}
    prev = 0
    for gap in gaps:
        raw = t0 * (float(gap) ** alpha)
        val = round_to_step(min(max(raw, t_min), t_max), round_to)
        val = max(val, prev)
        values[gap] = int(val)
        prev = int(val)
    default = values.get(default_gap, values[max(gaps)])
    return ",".join([f"{gap}:{values[gap]}" for gap in gaps] + [f"default:{default}"])


def apply_derived_params(params: Dict[str, Any], derived_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    min_inliers_spec = derived_cfg.get("min_inliers_map")
    if isinstance(min_inliers_spec, dict):
        kind = str(min_inliers_spec.get("type", "log_linear"))
        if kind == "log_linear":
            out["min_inliers_map"] = build_log_linear_min_inliers_map(out, min_inliers_spec)
        elif kind == "power_law":
            out["min_inliers_map"] = build_power_law_min_inliers_map(out, min_inliers_spec)
        else:
            raise ValueError(f"Unsupported derived min_inliers_map type: {kind}")
    return out


def objective_factory(
    study_root: Path,
    base_config: Dict[str, Any],
    search_space: Dict[str, Any],
    reuse_mode: str,
    derived_cfg: Dict[str, Any] | None = None,
):
    def objective(trial: "optuna.trial.Trial") -> float:
        raw_params = {name: suggest_value(trial, name, spec) for name, spec in search_space.items()}
        params = apply_derived_params(raw_params, derived_cfg or {})
        config = build_single_candidate_config(base_config, params)
        trial_root = study_root / "trials" / f"trial_{trial.number:04d}"
        trial_root.mkdir(parents=True, exist_ok=True)

        args = SimpleNamespace(
            output_root=str(trial_root),
            reuse_mode=reuse_mode,
            criterion=None,
        )
        results = run_search(config, args)
        results_path = trial_root / "validation_results.json"
        results["results_json"] = str(results_path)
        dump_json(str(results_path), results)
        write_results_csv(str(trial_root / "validation_results.csv"), results["candidates"])
        dump_json(str(trial_root / "robust_config.json"), build_robust_config(results, results["criterion"]))
        best_summary = results.get("best_summary", {})
        score = float(best_summary.get("score", float("inf")))

        trial.set_user_attr("params", params)
        trial.set_user_attr("raw_params", raw_params)
        trial.set_user_attr("best_summary", best_summary)
        trial.set_user_attr("results_json", str(results_path))
        return score

    return objective


def export_trials_csv(study: "optuna.study.Study", out_csv: Path) -> None:
    rows = []
    for trial in study.trials:
        row = {
            "trial_number": trial.number,
            "state": str(trial.state),
            "score": trial.value,
        }
        exported_params = dict(trial.user_attrs.get("params", {}) or trial.params)
        for key, value in sorted(exported_params.items()):
            row[key] = value
        summary = trial.user_attrs.get("best_summary", {}) or {}
        for key in [
            "mean_primary_metric",
            "std_primary_metric",
            "worst_primary_metric",
            "failure_rate",
            "skip_rate",
            "mean_rotation_median_deg",
            "mean_reject_ratio",
        ]:
            row[key] = summary.get(key)
        row["results_json"] = trial.user_attrs.get("results_json")
        rows.append(row)

    fieldnames = sorted({k for row in rows for k in row.keys()})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal Optuna + TPE wrapper around tools/validation_search.py")
    ap.add_argument("--config", required=True, help="Optuna search config JSON/YAML.")
    ap.add_argument("--study_name", default=None)
    ap.add_argument("--output_root", default=None)
    ap.add_argument("--n_trials", type=int, default=None)
    ap.add_argument("--reuse_mode", choices=["reuse", "resume", "full"], default="reuse")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_structured_config(args.config)
    config_dir = Path(cfg["_config_dir"]).resolve()
    config_path = Path(cfg["_config_path"]).resolve()
    output_root = Path(args.output_root or cfg.get("output_root") or "runs/optuna_validation_search")
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    n_trials = int(args.n_trials or cfg.get("n_trials") or 20)
    study_name = args.study_name or cfg.get("study_name") or "optuna_validation_search"
    sampler_cfg = cfg.get("sampler", {})
    seed = args.seed if args.seed is not None else sampler_cfg.get("seed")
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        multivariate=bool(sampler_cfg.get("multivariate", True)),
        group=bool(sampler_cfg.get("group", True)),
    )

    base_validation = cfg.get("base_validation_config")
    if not isinstance(base_validation, dict):
        raise ValueError("Config must contain a dict field: base_validation_config")
    base_validation = deepcopy(base_validation)
    base_validation["_config_dir"] = str(config_dir)
    base_validation["_config_path"] = str(config_path)
    base_validation["scenes"] = resolve_validation_scenes(base_validation, config_dir)
    search_space = cfg.get("optuna_search_space")
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("Config must contain a non-empty dict field: optuna_search_space")

    metadata = {
        "config_path": str(config_path),
        "study_name": study_name,
        "output_root": str(output_root),
        "n_trials": n_trials,
        "reuse_mode": args.reuse_mode,
        "search_space": search_space,
        "derived_params": cfg.get("derived_params", {}),
        "base_validation_config_snapshot": cfg.get("base_validation_config"),
    }
    dump_json(str(output_root / "study_metadata.json"), metadata)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(
        objective_factory(output_root, base_validation, search_space, args.reuse_mode, cfg.get("derived_params", {})),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = {
        "study_name": study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "best_summary": study.best_trial.user_attrs.get("best_summary", {}),
        "best_results_json": study.best_trial.user_attrs.get("results_json"),
    }
    dump_json(str(output_root / "best_result.json"), best)
    export_trials_csv(study, output_root / "trials_summary.csv")
    print("saved:", output_root / "study_metadata.json")
    print("saved:", output_root / "best_result.json")
    print("saved:", output_root / "trials_summary.csv")


if __name__ == "__main__":
    main()
