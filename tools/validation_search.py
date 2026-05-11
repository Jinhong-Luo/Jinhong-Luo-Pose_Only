#!/usr/bin/env python3
import argparse
import csv
import ctypes
import itertools
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_structured_config(path: str) -> Dict[str, Any]:
    path = os.path.abspath(path)
    suffix = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8-sig") as f:
        if suffix in (".yml", ".yaml"):
            import yaml

            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dict: {path}")
    data["_config_path"] = path
    data["_config_dir"] = os.path.dirname(path)
    return data


def dump_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json_if_exists(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    norm = path.replace("/", os.sep).replace("\\", os.sep)
    if norm.startswith(f".{os.sep}") or norm.startswith(f"..{os.sep}") or norm in (".", ".."):
        return os.path.abspath(os.path.join(base_dir, path))
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_value(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**context)
    if isinstance(value, list):
        return [render_value(v, context) for v in value]
    if isinstance(value, dict):
        return {k: render_value(v, context) for k, v in value.items()}
    return value


def outputs_exist(paths: List[str]) -> bool:
    return bool(paths) and all(os.path.exists(p) for p in paths)


def get_peak_working_set_mb(proc: subprocess.Popen) -> Optional[float]:
    if os.name != "nt":
        return None
    try:
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(int(proc._handle)),
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return None
        return float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        return None


def run_stage(
    stage: Dict[str, Any],
    context: Dict[str, Any],
    *,
    default_cwd: str,
    reuse_mode: str,
) -> Dict[str, Any]:
    stage_name = stage.get("name", "unnamed_stage")
    rendered = render_value(stage, context)
    cmd = rendered.get("cmd")
    if not isinstance(cmd, list) or not cmd:
        raise ValueError(f"Stage '{stage_name}' must provide a non-empty cmd list.")

    reuse_outputs = [resolve_path(p, default_cwd) for p in ensure_list(rendered.get("reuse_outputs"))]
    cwd = resolve_path(rendered.get("cwd"), default_cwd) or default_cwd
    should_skip = reuse_mode != "full" and outputs_exist(reuse_outputs) and not rendered.get("always_run", False)
    if should_skip:
        return {
            "name": stage_name,
            "status": "reused",
            "cmd": [stringify(x) for x in cmd],
            "cwd": cwd,
            "reuse_outputs": reuse_outputs,
        }

    print(f"[validation_search] stage={stage_name}")
    t0 = time.perf_counter()
    proc = subprocess.Popen([stringify(x) for x in cmd], cwd=cwd)
    peak_memory_mb = None
    while True:
        polled = proc.poll()
        current_peak = get_peak_working_set_mb(proc)
        if current_peak is not None:
            peak_memory_mb = max(peak_memory_mb or 0.0, current_peak)
        if polled is not None:
            break
        time.sleep(0.2)
    elapsed_sec = time.perf_counter() - t0
    status = "ok" if proc.returncode == 0 else "failed"
    return {
        "name": stage_name,
        "status": status,
        "returncode": int(proc.returncode),
        "elapsed_sec": float(elapsed_sec),
        "peak_working_set_mb": peak_memory_mb,
        "cmd": [stringify(x) for x in cmd],
        "cwd": cwd,
        "reuse_outputs": reuse_outputs,
    }


def find_stage_index(stages: List[Dict[str, Any]], stage_name: str) -> Optional[int]:
    for idx, stage in enumerate(ensure_list(stages)):
        if stage.get("name") == stage_name:
            return idx
    return None


def pick_stage_fallback(
    fallbacks: List[Dict[str, Any]],
    *,
    failed_stage_name: str,
    context: Dict[str, Any],
    used_tags: set,
) -> Optional[Dict[str, Any]]:
    for fallback in ensure_list(fallbacks):
        if not isinstance(fallback, dict):
            continue
        if fallback.get("trigger_stage") != failed_stage_name:
            continue
        tag = str(fallback.get("tag") or f"{failed_stage_name}_fallback")
        if tag in used_tags:
            continue
        when_context = fallback.get("when_context", {})
        if isinstance(when_context, dict):
            matched = True
            for key, expected in when_context.items():
                if context.get(key) != expected:
                    matched = False
                    break
            if not matched:
                continue
        selected = deepcopy(fallback)
        selected["tag"] = tag
        return selected
    return None


def make_candidate_records(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not search_space:
        return [{"candidate_id": "candidate_000", "params": {}}]
    keys = sorted(search_space.keys())
    values = [ensure_list(search_space[k]) for k in keys]
    records = []
    for idx, combo in enumerate(itertools.product(*values)):
        params = {k: combo[i] for i, k in enumerate(keys)}
        records.append(
            {
                "candidate_id": f"candidate_{idx:03d}",
                "params": params,
            }
        )
    return records


def extract_rotation_metrics(payload: Optional[Any]) -> Dict[str, Optional[float]]:
    if payload is None:
        return {
            "rotation_median_deg": None,
            "rotation_mean_deg": None,
            "rotation_p90_deg": None,
            "rotation_max_deg": None,
        }
    best = None
    if isinstance(payload, list):
        valid = [item for item in payload if isinstance(item, dict) and "median_deg" in item]
        if valid:
            best = min(valid, key=lambda item: float(item.get("median_deg", float("inf"))))
    elif isinstance(payload, dict):
        best = payload if "median_deg" in payload else None
    if not best:
        return {
            "rotation_median_deg": None,
            "rotation_mean_deg": None,
            "rotation_p90_deg": None,
            "rotation_max_deg": None,
        }
    return {
        "rotation_median_deg": float(best.get("median_deg")) if best.get("median_deg") is not None else None,
        "rotation_mean_deg": float(best.get("mean_deg")) if best.get("mean_deg") is not None else None,
        "rotation_p90_deg": float(best.get("p90_deg")) if best.get("p90_deg") is not None else None,
        "rotation_max_deg": float(best.get("max_deg")) if best.get("max_deg") is not None else None,
    }


def extract_translation_metrics(payload: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    raw = payload.get("raw", {}) if isinstance(payload, dict) else {}
    mm = payload.get("mm", {}) if isinstance(payload, dict) else {}
    return {
        "translation_raw_median": float(raw.get("median")) if raw.get("median") is not None else None,
        "translation_raw_mean": float(raw.get("mean")) if raw.get("mean") is not None else None,
        "translation_raw_p90": float(raw.get("p90")) if raw.get("p90") is not None else None,
        "translation_mm_median": float(mm.get("median")) if mm.get("median") is not None else None,
        "translation_mm_mean": float(mm.get("mean")) if mm.get("mean") is not None else None,
        "translation_mm_p90": float(mm.get("p90")) if mm.get("p90") is not None else None,
    }


def first_metric(record: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return float(value)
    return None


def scene_metric_value(record: Dict[str, Any], primary_metric: str) -> Optional[float]:
    if primary_metric == "auto_translation":
        return first_metric(record, ["translation_mm_median", "translation_raw_median"])
    if primary_metric == "auto_translation_mean":
        return first_metric(record, ["translation_mm_mean", "translation_raw_mean"])
    if primary_metric == "translation_vs_colmap_ratio":
        ours = first_metric(record, ["translation_mm_median", "translation_raw_median"])
        ref = record.get("colmap_translation_mm_median")
        if ours is None or ref is None:
            return None
        ref = float(ref)
        if ref <= 0.0:
            return None
        return float(ours / ref)
    if primary_metric == "translation_vs_colmap_gap":
        ours = first_metric(record, ["translation_mm_median", "translation_raw_median"])
        ref = record.get("colmap_translation_mm_median")
        if ours is None or ref is None:
            return None
        return float(ours - float(ref))
    value = record.get(primary_metric)
    return float(value) if value is not None else None


def compute_reject_ratio(record: Dict[str, Any]) -> float:
    candidates = []
    for key in ["rraa_reject_ratio", "qtrack_reject_ratio", "g_reject_ratio"]:
        value = record.get(key)
        if value is not None:
            candidates.append(float(value))
    if not candidates:
        return 0.0
    return max(candidates)


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def stddev(values: List[float]) -> Optional[float]:
    if not values:
        return None
    mu = mean(values)
    assert mu is not None
    return float((sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5)


def select_score(summary: Dict[str, Any], scoring_cfg: Dict[str, Any], criterion: str) -> float:
    mean_metric = float(summary.get("mean_primary_metric", float("inf")))
    std_metric = float(summary.get("std_primary_metric", 0.0) or 0.0)
    worst_metric = float(summary.get("worst_primary_metric", mean_metric))
    mean_rotation = float(summary.get("mean_rotation_median_deg", 0.0) or 0.0)
    failure_rate = float(summary.get("failure_rate", 0.0) or 0.0)
    skip_rate = float(summary.get("skip_rate", 0.0) or 0.0)
    reject_rate = float(summary.get("mean_reject_ratio", 0.0) or 0.0)

    lambda_fail = float(scoring_cfg.get("lambda_fail", 1000.0))
    lambda_skip = float(scoring_cfg.get("lambda_skip", 100.0))
    lambda_var = float(scoring_cfg.get("lambda_var", 0.0))
    lambda_worst = float(scoring_cfg.get("lambda_worst", 0.0))
    lambda_rot = float(scoring_cfg.get("lambda_rot", 0.0))
    lambda_reject = float(scoring_cfg.get("lambda_reject", 0.0))

    if criterion == "mean_only":
        return mean_metric + lambda_rot * mean_rotation
    if criterion == "mean_plus_std":
        return mean_metric + lambda_var * std_metric + lambda_rot * mean_rotation
    if criterion == "mean_plus_worst":
        return mean_metric + lambda_worst * worst_metric + lambda_rot * mean_rotation
    if criterion == "mean_plus_fail_penalty_plus_std":
        return (
            mean_metric
            + lambda_fail * failure_rate
            + lambda_skip * skip_rate
            + lambda_var * std_metric
            + lambda_worst * worst_metric
            + lambda_rot * mean_rotation
            + lambda_reject * reject_rate
        )
    raise ValueError(f"Unknown criterion: {criterion}")


def aggregate_candidate(candidate: Dict[str, Any], scoring_cfg: Dict[str, Any], criterion: str) -> Dict[str, Any]:
    primary_metric = scoring_cfg.get("primary_metric", "auto_translation")
    scene_records = candidate.get("scenes", [])
    metric_values = []
    rotation_values = []
    reject_values = []
    failure_count = 0
    skip_count = 0
    for record in scene_records:
        metric = scene_metric_value(record, primary_metric)
        record["primary_metric_value"] = metric
        if metric is None or record.get("status") != "ok":
            failure_count += 1
        else:
            metric_values.append(metric)
        rot = record.get("rotation_median_deg")
        if rot is not None:
            rotation_values.append(float(rot))
        reject_values.append(compute_reject_ratio(record))
        if str(record.get("pa_status", "")).lower() == "skipped" or str(record.get("status", "")).lower() == "skipped":
            skip_count += 1

    scene_count = max(len(scene_records), 1)
    summary = {
        "primary_metric": primary_metric,
        "mean_primary_metric": mean(metric_values) if metric_values else float("inf"),
        "std_primary_metric": stddev(metric_values) if metric_values else float("inf"),
        "worst_primary_metric": max(metric_values) if metric_values else float("inf"),
        "mean_rotation_median_deg": mean(rotation_values),
        "mean_reject_ratio": mean(reject_values) or 0.0,
        "scene_count": len(scene_records),
        "valid_scene_count": len(metric_values),
        "failure_count": failure_count,
        "skip_count": skip_count,
        "failure_rate": float(failure_count / scene_count),
        "skip_rate": float(skip_count / scene_count),
    }
    summary["score"] = select_score(summary, scoring_cfg, criterion)
    candidate["summary"] = summary
    return candidate


def compare_candidates(best: Dict[str, Any], runner_up: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not runner_up:
        return {"message": "Only one candidate was available."}
    best_summary = best.get("summary", {})
    other_summary = runner_up.get("summary", {})
    return {
        "runner_up_candidate_id": runner_up.get("candidate_id"),
        "score_gap": float(other_summary.get("score", float("inf")) - best_summary.get("score", float("inf"))),
        "mean_metric_gap": float(other_summary.get("mean_primary_metric", float("inf")) - best_summary.get("mean_primary_metric", float("inf"))),
        "worst_metric_gap": float(other_summary.get("worst_primary_metric", float("inf")) - best_summary.get("worst_primary_metric", float("inf"))),
        "failure_rate_gap": float(other_summary.get("failure_rate", 0.0) - best_summary.get("failure_rate", 0.0)),
    }


def write_results_csv(path: str, candidates: List[Dict[str, Any]]) -> None:
    scene_names = sorted({scene["scene_name"] for candidate in candidates for scene in candidate.get("scenes", [])})
    fieldnames = [
        "candidate_id",
        "score",
        "mean_primary_metric",
        "std_primary_metric",
        "worst_primary_metric",
        "failure_rate",
        "skip_rate",
        "mean_rotation_median_deg",
        "mean_reject_ratio",
        "params_json",
    ]
    for scene_name in scene_names:
        fieldnames.extend(
            [
                f"{scene_name}__status",
                f"{scene_name}__primary_metric",
                f"{scene_name}__rotation_median_deg",
                f"{scene_name}__translation_mm_median",
                f"{scene_name}__translation_raw_median",
                f"{scene_name}__colmap_translation_mm_median",
                f"{scene_name}__translation_vs_colmap_ratio",
                f"{scene_name}__translation_vs_colmap_gap",
                f"{scene_name}__pa_status",
                f"{scene_name}__pa_reason_code",
            ]
        )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            row = {
                "candidate_id": candidate.get("candidate_id"),
                "score": candidate.get("summary", {}).get("score"),
                "mean_primary_metric": candidate.get("summary", {}).get("mean_primary_metric"),
                "std_primary_metric": candidate.get("summary", {}).get("std_primary_metric"),
                "worst_primary_metric": candidate.get("summary", {}).get("worst_primary_metric"),
                "failure_rate": candidate.get("summary", {}).get("failure_rate"),
                "skip_rate": candidate.get("summary", {}).get("skip_rate"),
                "mean_rotation_median_deg": candidate.get("summary", {}).get("mean_rotation_median_deg"),
                "mean_reject_ratio": candidate.get("summary", {}).get("mean_reject_ratio"),
                "params_json": json.dumps(candidate.get("params", {}), ensure_ascii=False, sort_keys=True),
            }
            scenes_by_name = {scene["scene_name"]: scene for scene in candidate.get("scenes", [])}
            for scene_name in scene_names:
                scene = scenes_by_name.get(scene_name, {})
                row[f"{scene_name}__status"] = scene.get("status")
                row[f"{scene_name}__primary_metric"] = scene.get("primary_metric_value")
                row[f"{scene_name}__rotation_median_deg"] = scene.get("rotation_median_deg")
                row[f"{scene_name}__translation_mm_median"] = scene.get("translation_mm_median")
                row[f"{scene_name}__translation_raw_median"] = scene.get("translation_raw_median")
                row[f"{scene_name}__colmap_translation_mm_median"] = scene.get("colmap_translation_mm_median")
                row[f"{scene_name}__translation_vs_colmap_ratio"] = scene.get("translation_vs_colmap_ratio")
                row[f"{scene_name}__translation_vs_colmap_gap"] = scene.get("translation_vs_colmap_gap")
                row[f"{scene_name}__pa_status"] = scene.get("pa_status")
                row[f"{scene_name}__pa_reason_code"] = scene.get("pa_reason_code")
            writer.writerow(row)


def extract_scene_record(
    scene_cfg: Dict[str, Any],
    context: Dict[str, Any],
    colmap_refs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics_cfg = deepcopy(scene_cfg.get("metrics", {}))
    rendered_metrics = render_value(metrics_cfg, context)
    rotation_payload = read_json_if_exists(resolve_path(rendered_metrics.get("rotation_json"), REPO_ROOT))
    translation_payload = read_json_if_exists(resolve_path(rendered_metrics.get("translation_json"), REPO_ROOT))
    summary_payload = read_json_if_exists(resolve_path(rendered_metrics.get("summary_json"), REPO_ROOT))
    pa_payload = read_json_if_exists(resolve_path(rendered_metrics.get("pa_json"), REPO_ROOT))
    ligt_payload = read_json_if_exists(resolve_path(rendered_metrics.get("ligt_json"), REPO_ROOT))
    rraa_stats_payload = read_json_if_exists(resolve_path(rendered_metrics.get("rraa_stats_json"), REPO_ROOT))

    record = {
        "scene_name": scene_cfg["name"],
        "scene_group": scene_cfg.get("group"),
        "scene_id": scene_cfg.get("scene_id"),
        "summary_json": resolve_path(rendered_metrics.get("summary_json"), REPO_ROOT),
        "status": "ok",
        "pa_status": pa_payload.get("status") if isinstance(pa_payload, dict) else None,
        "pa_reason_code": pa_payload.get("reason_code") if isinstance(pa_payload, dict) else None,
        "pa_failure_stage": pa_payload.get("failure_stage") if isinstance(pa_payload, dict) else None,
        "rraa_kept_ratio": None,
        "rraa_reject_ratio": None,
        "qtrack_reject_ratio": None,
        "g_reject_ratio": None,
        "kept_track_ratio": None,
        "colmap_translation_mm_median": None,
        "translation_vs_colmap_ratio": None,
        "translation_vs_colmap_gap": None,
    }
    record.update(extract_rotation_metrics(rotation_payload))
    record.update(extract_translation_metrics(translation_payload))

    if isinstance(summary_payload, dict):
        if summary_payload.get("rraa_kept_ratio") is not None:
            record["rraa_kept_ratio"] = float(summary_payload["rraa_kept_ratio"])
            record["rraa_reject_ratio"] = 1.0 - float(summary_payload["rraa_kept_ratio"])
        if summary_payload.get("ligt_g_reject_ratio") is not None:
            record["g_reject_ratio"] = float(summary_payload["ligt_g_reject_ratio"])

    if isinstance(rraa_stats_payload, dict) and rraa_stats_payload.get("kept_ratio") is not None:
        record["rraa_kept_ratio"] = float(rraa_stats_payload["kept_ratio"])
        record["rraa_reject_ratio"] = 1.0 - float(rraa_stats_payload["kept_ratio"])

    if isinstance(ligt_payload, dict):
        if ligt_payload.get("qtrack_reject_ratio") is not None:
            record["qtrack_reject_ratio"] = float(ligt_payload["qtrack_reject_ratio"])
        if ligt_payload.get("g_reject_ratio") is not None:
            record["g_reject_ratio"] = float(ligt_payload["g_reject_ratio"])
        if ligt_payload.get("kept_track_ratio") is not None:
            record["kept_track_ratio"] = float(ligt_payload["kept_track_ratio"])

    if isinstance(colmap_refs, dict):
        scene_key_candidates = [
            f"{scene_cfg.get('group')}::{scene_cfg.get('scene_id')}",
            f"{scene_cfg.get('group')}::{scene_cfg.get('name')}",
            str(scene_cfg.get("scene_id") or ""),
            str(scene_cfg.get("name") or ""),
        ]
        ref_payload = None
        for key in scene_key_candidates:
            if key and key in colmap_refs:
                ref_payload = colmap_refs[key]
                break
        if isinstance(ref_payload, dict):
            ref_value = ref_payload.get("translation_mm_median")
        else:
            ref_value = ref_payload
        if ref_value is not None:
            record["colmap_translation_mm_median"] = float(ref_value)

    ours_translation = first_metric(record, ["translation_mm_median", "translation_raw_median"])
    ref_translation = record.get("colmap_translation_mm_median")
    if ours_translation is not None and ref_translation is not None:
        ref_translation = float(ref_translation)
        if ref_translation > 0.0:
            record["translation_vs_colmap_ratio"] = float(ours_translation / ref_translation)
        record["translation_vs_colmap_gap"] = float(ours_translation - ref_translation)

    if scene_metric_value(record, "auto_translation") is None and record.get("rotation_median_deg") is None:
        record["status"] = "failed"

    return record


def run_search(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config_dir = config["_config_dir"]
    output_root = os.path.abspath(os.path.join(REPO_ROOT, args.output_root or config.get("output_root") or "runs/validation_search"))
    os.makedirs(output_root, exist_ok=True)

    scoring_cfg = deepcopy(config.get("scoring", {}))
    criterion = args.criterion or config.get("selection_criterion") or scoring_cfg.get("criterion") or "mean_plus_fail_penalty_plus_std"
    quality_config_path = resolve_path(config.get("quality_config"), config_dir)
    colmap_ref_path = resolve_path(config.get("colmap_reference_json"), config_dir)
    colmap_refs = read_json_if_exists(colmap_ref_path)
    search_space = deepcopy(config.get("search_space", {}))
    candidates = make_candidate_records(search_space)
    results = {
        "config_path": config["_config_path"],
        "output_root": output_root,
        "criterion": criterion,
        "scoring": scoring_cfg,
        "protocol": deepcopy(config.get("protocol", {})),
        "colmap_reference_json": colmap_ref_path,
        "search_space": search_space,
        "candidates": [],
    }
    prepare_stage_cache: Dict[str, Dict[str, Any]] = {}

    for candidate in candidates:
        candidate = deepcopy(candidate)
        candidate_root = os.path.join(output_root, "candidates", candidate["candidate_id"])
        os.makedirs(candidate_root, exist_ok=True)
        candidate["candidate_root"] = candidate_root
        candidate["scenes"] = []
        dump_json(os.path.join(candidate_root, "candidate_params.json"), candidate["params"])

        for scene_cfg in config.get("scenes", []):
            scene_name = scene_cfg["name"]
            scene_root = resolve_path(scene_cfg.get("scene_root", "."), config_dir)
            prepare_root = resolve_path(scene_cfg.get("prepare_root", scene_root), config_dir)
            candidate_scene_root = os.path.join(candidate_root, scene_name)
            os.makedirs(candidate_scene_root, exist_ok=True)

            context = {
                "python": sys.executable,
                "repo_root": REPO_ROOT,
                "config_dir": config_dir,
                "quality_config": quality_config_path or "",
                "output_root": output_root,
                "scene_root": scene_root,
                "prepare_root": prepare_root,
                "candidate_root": candidate_scene_root,
                "candidate_id": candidate["candidate_id"],
                "scene_name": scene_name,
                "scene_group": scene_cfg.get("group", ""),
            }
            context.update({k: v for k, v in scene_cfg.items() if isinstance(v, (str, int, float, bool))})
            context.update(candidate["params"])
            fallback_policies = ensure_list(
                scene_cfg.get("candidate_stage_fallbacks", config.get("candidate_stage_fallbacks"))
            )

            prepare_stages = scene_cfg.get("prepare_stages")
            if not prepare_stages:
                prepare_stages = config.get("default_prepare_stages") or scene_cfg.get("stages", {}).get("prepare")
            candidate_stages = scene_cfg.get("candidate_stages")
            if not candidate_stages:
                candidate_stages = config.get("default_candidate_stages") or scene_cfg.get("stages", {}).get("candidate")

            scene_result = {
                "scene_name": scene_name,
                "scene_group": scene_cfg.get("group"),
                "status": "ok",
                "stages": [],
                "fallbacks": [],
            }

            for stage in ensure_list(prepare_stages):
                rendered_stage = render_value(stage, context)
                cache_key = json.dumps(
                    {
                        "scene": scene_name,
                        "stage": rendered_stage.get("name", "unnamed_stage"),
                        "cmd": rendered_stage.get("cmd"),
                        "reuse_outputs": rendered_stage.get("reuse_outputs"),
                        "cwd": rendered_stage.get("cwd"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if cache_key in prepare_stage_cache:
                    cached = deepcopy(prepare_stage_cache[cache_key])
                    cached["status"] = "reused_prepare_cache" if cached.get("status") == "ok" else cached.get("status")
                    stage_result = cached
                else:
                    stage_result = run_stage(stage, context, default_cwd=REPO_ROOT, reuse_mode=args.reuse_mode)
                    prepare_stage_cache[cache_key] = deepcopy(stage_result)
                scene_result["stages"].append(stage_result)
                if stage_result["status"] == "failed":
                    scene_result["status"] = "failed"
                    break

            if scene_result["status"] == "ok":
                active_context = deepcopy(context)
                stage_list = ensure_list(candidate_stages)
                used_fallback_tags = set()
                fallback_force_full = False
                stage_idx = 0
                while stage_idx < len(stage_list):
                    stage = stage_list[stage_idx]
                    stage_result = run_stage(
                        stage,
                        active_context,
                        default_cwd=REPO_ROOT,
                        reuse_mode="full" if fallback_force_full else args.reuse_mode,
                    )
                    scene_result["stages"].append(stage_result)
                    if stage_result["status"] != "failed":
                        stage_idx += 1
                        continue

                    fallback = pick_stage_fallback(
                        fallback_policies,
                        failed_stage_name=stage.get("name", ""),
                        context=active_context,
                        used_tags=used_fallback_tags,
                    )
                    if not fallback:
                        scene_result["status"] = "failed"
                        break

                    rerun_from = fallback.get("rerun_from_stage") or stage.get("name")
                    rerun_idx = find_stage_index(stage_list, str(rerun_from))
                    if rerun_idx is None:
                        scene_result["status"] = "failed"
                        scene_result["stages"].append(
                            {
                                "name": f"fallback::{fallback['tag']}",
                                "status": "failed",
                                "reason": f"rerun_from_stage_not_found:{rerun_from}",
                                "override_params": deepcopy(fallback.get('override_params', {})),
                            }
                        )
                        break

                    used_fallback_tags.add(fallback["tag"])
                    active_context.update(deepcopy(fallback.get("override_params", {})))
                    active_context["fallback_tag"] = fallback["tag"]
                    scene_result["fallbacks"].append(
                        {
                            "tag": fallback["tag"],
                            "trigger_stage": stage.get("name"),
                            "rerun_from_stage": rerun_from,
                            "override_params": deepcopy(fallback.get("override_params", {})),
                        }
                    )
                    scene_result["stages"].append(
                        {
                            "name": f"fallback::{fallback['tag']}",
                            "status": "triggered",
                            "trigger_stage": stage.get("name"),
                            "rerun_from_stage": rerun_from,
                            "override_params": deepcopy(fallback.get("override_params", {})),
                        }
                    )
                    fallback_force_full = True
                    stage_idx = rerun_idx
                context = active_context

            metrics_record = extract_scene_record(scene_cfg, context, colmap_refs)
            if scene_result["status"] == "failed":
                metrics_record["status"] = "failed"
            metrics_record["stages"] = scene_result["stages"]
            metrics_record["fallbacks"] = scene_result["fallbacks"]
            metrics_record["fallback_used"] = bool(scene_result["fallbacks"])
            candidate["scenes"].append(metrics_record)

        aggregate_candidate(candidate, scoring_cfg, criterion)
        results["candidates"].append(candidate)

    results["candidates"].sort(key=lambda item: float(item.get("summary", {}).get("score", float("inf"))))
    if results["candidates"]:
        best = results["candidates"][0]
        runner_up = results["candidates"][1] if len(results["candidates"]) > 1 else None
        results["best_candidate_id"] = best["candidate_id"]
        results["best_params"] = best["params"]
        results["best_summary"] = best["summary"]
        results["best_vs_runner_up"] = compare_candidates(best, runner_up)
    return results


def build_robust_config(results: Dict[str, Any], criterion: str) -> Dict[str, Any]:
    candidates = list(results.get("candidates", []))
    if not candidates:
        raise RuntimeError("No candidates available for robust selection.")
    candidates = sorted(candidates, key=lambda item: float(item.get("summary", {}).get("score", float("inf"))))
    best = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    return {
        "selection_criterion": criterion,
        "selected_candidate_id": best.get("candidate_id"),
        "selected_params": best.get("params", {}),
        "validation_summary": best.get("summary", {}),
        "scene_breakdown": best.get("scenes", []),
        "protocol": deepcopy(results.get("protocol", {})),
        "comparison_to_runner_up": compare_candidates(best, runner_up),
        "source_validation_results": results.get("results_json"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Small-scale validation search with cross-scene robust selection.")
    ap.add_argument("--config", type=str, default=None, help="JSON/YAML validation search config.")
    ap.add_argument("--results_json", type=str, default=None, help="Reuse existing validation_results.json for select-only mode.")
    ap.add_argument("--output_root", type=str, default=None)
    ap.add_argument("--reuse_mode", choices=["reuse", "resume", "full"], default="reuse")
    ap.add_argument("--criterion", choices=["mean_only", "mean_plus_std", "mean_plus_worst", "mean_plus_fail_penalty_plus_std"], default=None)
    ap.add_argument("--select_only", action="store_true", help="Skip running commands and only produce robust_config.json from results.")
    args = ap.parse_args()

    if args.select_only:
        if not args.results_json:
            raise ValueError("--select_only requires --results_json")
        results = read_json_if_exists(args.results_json)
        if not isinstance(results, dict):
            raise RuntimeError(f"Could not read results JSON: {args.results_json}")
        results["results_json"] = os.path.abspath(args.results_json)
        criterion = args.criterion or results.get("criterion") or "mean_plus_fail_penalty_plus_std"
        robust = build_robust_config(results, criterion)
        out_path = os.path.join(os.path.dirname(os.path.abspath(args.results_json)), "robust_config.json")
        dump_json(out_path, robust)
        print("saved:", out_path)
        return

    if not args.config:
        raise ValueError("Search mode requires --config")

    config = load_structured_config(args.config)
    results = run_search(config, args)
    output_root = results["output_root"]
    results_path = os.path.join(output_root, "validation_results.json")
    results["results_json"] = results_path
    dump_json(results_path, results)
    write_results_csv(os.path.join(output_root, "validation_results.csv"), results["candidates"])
    robust = build_robust_config(results, results["criterion"])
    dump_json(os.path.join(output_root, "robust_config.json"), robust)
    print("saved:", results_path)
    print("saved:", os.path.join(output_root, "validation_results.csv"))
    print("saved:", os.path.join(output_root, "robust_config.json"))


if __name__ == "__main__":
    main()
