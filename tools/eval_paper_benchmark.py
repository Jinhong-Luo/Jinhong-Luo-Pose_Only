#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CORE_NUMERIC_FIELDS = [
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
    "runtime_total_sec",
    "peak_memory_mb",
    "runtime_rraa_sec",
    "runtime_pose_only_sec",
]

MAIN_COLUMNS = [
    "dataset",
    "scene",
    "scene_slug",
    "scene_tags",
    "experiment_group",
    "experiment_tags",
    "rotation_source",
    "use_pa",
    "qtrack_mode",
    "enable_quality_weighting",
    "rraa_use_qpair_weight",
    "u_min",
    "g_min",
    "threshold_group",
    "run_dir",
    "scene_run_root",
    "prepared_scene_dir",
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
    "rraa_is_connected",
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
    "pa_status",
    "pa_reason_code",
    "pa_failure_stage",
    "pa_final_rmse",
    "pa_accept_iters",
    "baseline_experiment_group",
    "delta_translation_mm_median",
    "delta_translation_mm_p90",
    "runtime_total_sec",
    "peak_memory_mb",
    "runtime_rraa_sec",
    "runtime_pose_only_sec",
    "status",
    "missing_files",
    "generated_files",
    "notes",
]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def dump_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, ensure_ascii=False)


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (list, dict)):
                    out[key] = json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True)
                else:
                    out[key] = value
            writer.writerow(out)


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def safe_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return None


def render_value(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**context)
    if isinstance(value, list):
        return [render_value(v, context) for v in value]
    if isinstance(value, dict):
        return {k: render_value(v, context) for k, v in value.items()}
    return value


def resolve_path(path: Optional[str], base_dir: str = REPO_ROOT) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return os.path.abspath(path)
    norm = path.replace("/", os.sep).replace("\\", os.sep)
    if norm.startswith(f".{os.sep}") or norm.startswith(f"..{os.sep}") or norm in {".", ".."}:
        return os.path.abspath(os.path.join(base_dir, path))
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def resolve_candidates(values: Sequence[str], base_dir: str = REPO_ROOT) -> List[str]:
    out = []
    for value in values:
        path = resolve_path(value, base_dir=base_dir)
        if path and path not in out:
            out.append(path)
    return out


def first_existing(paths: Sequence[str]) -> Optional[str]:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def add_missing(missing: List[str], label: str, path: Optional[str]) -> None:
    if not path:
        missing.append(f"{label}:<unresolved>")
    elif not os.path.exists(path):
        missing.append(f"{label}:{path}")


def load_json_if_exists(path: Optional[str]) -> Any:
    if not path or not os.path.exists(path):
        return None
    return load_json(path)


def load_npz_if_exists(path: Optional[str]) -> Optional[np.lib.npyio.NpzFile]:
    if not path or not os.path.exists(path):
        return None
    return np.load(path)


def choose_best_rotation_report(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, list):
        valid = [item for item in payload if isinstance(item, dict) and item.get("median_deg") is not None]
        if not valid:
            return None
        return min(valid, key=lambda item: float(item["median_deg"]))
    if isinstance(payload, dict) and payload.get("median_deg") is not None:
        return payload
    return None


def summarize_anchor_sensitivity(payload: Any) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(payload, dict):
        return None, None
    comparisons = payload.get("comparisons", [])
    medians = []
    p90s = []
    for item in comparisons:
        if not isinstance(item, dict):
            continue
        m = safe_float(item.get("median_deg"))
        p = safe_float(item.get("p90_deg"))
        if m is not None:
            medians.append(m)
        if p is not None:
            p90s.append(p)
    median_value = float(np.median(np.asarray(medians, dtype=np.float64))) if medians else None
    p90_value = float(np.quantile(np.asarray(p90s, dtype=np.float64), 0.9)) if p90s else None
    return median_value, p90_value


def load_manifest(path: str) -> Dict[str, Any]:
    manifest = load_json(path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a dict: {path}")
    manifest["_manifest_path"] = os.path.abspath(path)
    manifest["_manifest_dir"] = os.path.dirname(os.path.abspath(path))
    return manifest


def scene_slug(dataset: str, scene: str) -> str:
    text = f"{dataset}_{scene}"
    return (
        text.replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
    )


def select_scene_entries(manifest: Dict[str, Any], scene_tags: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    wanted_tags = set(scene_tags or manifest.get("default_scene_tags") or ["main"])
    entries: List[Dict[str, Any]] = []
    for dataset_block in manifest.get("datasets", []):
        dataset = dataset_block["name"]
        for scene_block in dataset_block.get("scenes", []):
            tags = set(scene_block.get("tags", []))
            if wanted_tags and not (tags & wanted_tags):
                continue
            item = deepcopy(scene_block)
            item["dataset"] = dataset
            item["scene"] = scene_block["scene"]
            item["scene_slug"] = scene_block.get("scene_slug") or scene_slug(dataset, scene_block["scene"])
            entries.append(item)
    return entries


def select_experiments(manifest: Dict[str, Any], experiment_groups: Optional[Sequence[str]], experiment_tags: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    group_filter = set(experiment_groups or [])
    tag_filter = set(experiment_tags or manifest.get("default_experiment_tags") or ["main"])
    entries: List[Dict[str, Any]] = []
    for exp in manifest.get("experiment_groups", []):
        name = exp["name"]
        tags = set(exp.get("tags", []))
        if group_filter and name not in group_filter:
            continue
        if tag_filter and not (tags & tag_filter):
            continue
        entries.append(deepcopy(exp))
    return entries


def default_path_specs(use_pa: bool) -> Dict[str, List[str]]:
    translation_eval_candidates = [
        "pose_only/eval_translation.json",
        "pose_only/eval_translation_pa.json" if use_pa else "pose_only/eval_translation_ligt.json",
        "poseonly_qd/eval_translation_pa.json" if use_pa else "poseonly_qd/eval_translation_ligt.json",
        "poseonly_qd/eval_translation.json",
        "pose_only/eval_translation_pa.json",
        "pose_only/eval_translation_ligt.json",
    ]
    pose_candidates = [
        "pose_only/poses_c2w_pa.txt" if use_pa else "pose_only/poses_c2w_ligt.txt",
        "pose_only/poses_c2w.txt",
        "poseonly_qd/poses_c2w_pa.txt" if use_pa else "poseonly_qd/poses_c2w_ligt.txt",
        "poseonly_qd/poses_c2w.txt",
        "pose_only/poses_c2w_pa.txt",
        "pose_only/poses_c2w_ligt.txt",
        "pose_only/poses_w2c_pa.txt" if use_pa else "pose_only/poses_w2c_ligt.txt",
        "pose_only/poses_w2c.txt",
        "poseonly_qd/poses_w2c_pa.txt" if use_pa else "poseonly_qd/poses_w2c_ligt.txt",
        "poseonly_qd/poses_w2c.txt",
    ]
    return {
        "rotation_eval_json": [
            "rraa_output/eval_rotation.json",
            "rraa_output_qd/eval_rotation_qw.json",
            "rraa_output_qd/eval_rotation_baseline.json",
        ],
        "rraa_stats_json": [
            "rraa_output/R_abs_stats.json",
            "rraa_output_qd/R_abs_qw_stats.json",
            "rraa_output_qd/R_abs_stats.json",
        ],
        "r_abs_npy": [
            "rraa_output/R_abs.npy",
            "rraa_output_qd/R_abs.npy",
            "rraa_output_qd/R_abs_qw.npy",
        ],
        "translation_eval_json": translation_eval_candidates,
        "poses_txt": pose_candidates,
        "quality_stats_json": [
            "pose_only/quality_stats.json",
            "poseonly_qd/quality_stats.json",
        ],
        "ligt_degeneracy_json": [
            "pose_only/ligt_degeneracy_stats.json",
            "poseonly_qd/ligt_degeneracy_stats.json",
        ],
        "pa_degeneracy_json": [
            "pose_only/pa_degeneracy_stats.json",
            "poseonly_qd/pa_degeneracy_stats.json",
        ],
        "qtrack_npz": [
            "pose_only/track_quality_scores.npz",
            "poseonly_qd/track_quality_scores.npz",
        ],
        "summary_json": [
            "experiment_summary.json"
        ],
    }


def resolve_scene_experiment_paths(
    scene: Dict[str, Any],
    exp: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    context = {
        "repo_root": REPO_ROOT,
        "manifest_dir": manifest["_manifest_dir"],
        "dataset": scene["dataset"],
        "scene": scene["scene"],
        "scene_slug": scene["scene_slug"],
        "experiment_group": exp["name"],
    }

    prepared_scene_dir = resolve_path(
        render_value(scene.get("prepared_scene_dir"), context),
        base_dir=manifest["_manifest_dir"],
    )
    context["prepared_scene_dir"] = prepared_scene_dir or ""

    scene_root_templates = ensure_list(scene.get("scene_run_root_candidates") or scene.get("scene_run_root"))
    if not scene_root_templates:
        scene_root_templates = [f"runs/{scene['dataset']}/{scene['scene']}"]
    rendered_scene_roots = render_value(scene_root_templates, context)
    scene_run_root_candidates = resolve_candidates(rendered_scene_roots, base_dir=manifest["_manifest_dir"])
    scene_run_root = first_existing(scene_run_root_candidates) or (scene_run_root_candidates[0] if scene_run_root_candidates else None)
    context["scene_run_root"] = scene_run_root or ""

    run_dir_templates = ensure_list(exp.get("run_dir_candidates") or exp.get("run_dir"))
    if not run_dir_templates:
        run_dir_templates = [f"runs/paper_benchmark/{exp['name']}/{scene['dataset']}/{scene['scene']}"]
    rendered_run_dirs = render_value(run_dir_templates, context)
    run_dir_candidates = resolve_candidates(rendered_run_dirs, base_dir=manifest["_manifest_dir"])
    run_dir = first_existing(run_dir_candidates) or (run_dir_candidates[0] if run_dir_candidates else None)
    context["run_dir"] = run_dir or ""

    specs = default_path_specs(bool(exp.get("use_pa", False)))
    specs.update(exp.get("relative_paths", {}))

    resolved: Dict[str, Any] = {
        "prepared_scene_dir": prepared_scene_dir,
        "scene_run_root": scene_run_root,
        "run_dir": run_dir,
        "run_dir_candidates": run_dir_candidates,
        "scene_run_root_candidates": scene_run_root_candidates,
    }

    for key, rel_list in specs.items():
        rel_candidates = ensure_list(render_value(rel_list, context))
        abs_candidates = []
        for rel in rel_candidates:
            if not rel:
                continue
            if os.path.isabs(rel):
                abs_candidates.append(os.path.abspath(rel))
            elif run_dir:
                abs_candidates.append(os.path.abspath(os.path.join(run_dir, rel)))
        resolved[f"{key}_candidates"] = abs_candidates
        resolved[key] = first_existing(abs_candidates) or (abs_candidates[0] if abs_candidates else None)

    shared_rel = scene.get("shared_relative_paths", {})
    shared_defaults = {
        "track_stats_json": ["tracks/track_build_quality_stats.json", "tracks_npz_qd/track_build_quality_stats.json"],
        "track_quality_summary_npz": ["tracks/track_quality_summary.npz", "tracks_npz_qd/track_quality_summary.npz"],
        "pair_quality_edges_npz": ["tracks/pair_quality_edges.npz", "tracks_npz_qd/pair_quality_edges.npz"],
    }
    shared_defaults.update(shared_rel)
    for key, rel_list in shared_defaults.items():
        rel_candidates = ensure_list(render_value(rel_list, context))
        abs_candidates = []
        for rel in rel_candidates:
            if not rel:
                continue
            if os.path.isabs(rel):
                abs_candidates.append(os.path.abspath(rel))
            elif scene_run_root:
                abs_candidates.append(os.path.abspath(os.path.join(scene_run_root, rel)))
        resolved[f"{key}_candidates"] = abs_candidates
        resolved[key] = first_existing(abs_candidates) or (abs_candidates[0] if abs_candidates else None)

    gt_paths = scene.get("gt_paths", {})
    gt_defaults = {
        "gt_rot_npy": os.path.join("{prepared_scene_dir}", "R_abs_gt_w2c.npy"),
        "gt_centers_npy": os.path.join("{prepared_scene_dir}", "gt_centers.npy"),
        "gt_pose_txt": os.path.join("{prepared_scene_dir}", "gt_poses_c2w.txt"),
    }
    gt_defaults.update(gt_paths)
    for key, template in gt_defaults.items():
        rendered = render_value(template, context)
        resolved[key] = resolve_path(rendered, base_dir=manifest["_manifest_dir"])

    return resolved


def run_command(cmd: Sequence[str], *, cwd: str, dry_run: bool) -> Tuple[bool, str]:
    printable = " ".join(cmd)
    if dry_run:
        return True, f"dry_run:{printable}"
    try:
        subprocess.run(list(cmd), cwd=cwd, check=True)
        return True, printable
    except subprocess.CalledProcessError as exc:
        return False, f"{printable} (returncode={exc.returncode})"


def ensure_rotation_eval(paths: Dict[str, Any], *, dry_run: bool, force_eval: bool) -> Tuple[Optional[str], List[str], List[str]]:
    notes: List[str] = []
    generated: List[str] = []
    out_path = paths.get("rotation_eval_json")
    if out_path and os.path.exists(out_path) and not force_eval:
        return out_path, generated, notes
    est_npy = paths.get("r_abs_npy")
    gt_npy = paths.get("gt_rot_npy")
    if not est_npy or not os.path.exists(est_npy):
        notes.append("rotation_eval_skipped:missing_est_npy")
        return out_path, generated, notes
    if not gt_npy or not os.path.exists(gt_npy):
        notes.append("rotation_eval_skipped:missing_gt_rot_npy")
        return out_path, generated, notes
    out_path = out_path or os.path.join(os.path.dirname(est_npy), "eval_rotation.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ok, message = run_command(
        [
            sys.executable,
            os.path.join("tools", "eval_rraa_rotation.py"),
            "--est_npy",
            est_npy,
            "--gt_npy",
            gt_npy,
            "--out_json",
            out_path,
        ],
        cwd=REPO_ROOT,
        dry_run=dry_run,
    )
    if ok:
        generated.append(out_path)
        notes.append(f"generated_rotation_eval:{message}")
    else:
        notes.append(f"rotation_eval_failed:{message}")
    return out_path, generated, notes


def detect_pose_type(path: str) -> str:
    name = os.path.basename(path).lower()
    if "w2c" in name:
        return "w2c"
    return "c2w"


def ensure_translation_eval(
    paths: Dict[str, Any],
    *,
    gt_unit_to_mm: Optional[float],
    dry_run: bool,
    force_eval: bool,
) -> Tuple[Optional[str], List[str], List[str]]:
    notes: List[str] = []
    generated: List[str] = []
    out_path = paths.get("translation_eval_json")
    if out_path and os.path.exists(out_path) and not force_eval:
        return out_path, generated, notes
    pose_path = first_existing(ensure_list(paths.get("poses_txt_candidates", []))) or paths.get("poses_txt")
    gt_centers = paths.get("gt_centers_npy")
    if not pose_path or not os.path.exists(pose_path):
        notes.append("translation_eval_skipped:missing_pose_txt")
        return out_path, generated, notes
    if not gt_centers or not os.path.exists(gt_centers):
        notes.append("translation_eval_skipped:missing_gt_centers")
        return out_path, generated, notes
    out_path = out_path or os.path.join(os.path.dirname(pose_path), "eval_translation.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join("tools", "eval_poseonly_strecha_mm.py"),
        "--est_poses",
        pose_path,
        "--est_type",
        detect_pose_type(pose_path),
        "--gt_centers_npy",
        gt_centers,
        "--out_json",
        out_path,
    ]
    if gt_unit_to_mm is not None:
        cmd.extend(["--gt_unit_to_mm", str(gt_unit_to_mm)])
    ok, message = run_command(cmd, cwd=REPO_ROOT, dry_run=dry_run)
    if ok:
        generated.append(out_path)
        notes.append(f"generated_translation_eval:{message}")
    else:
        notes.append(f"translation_eval_failed:{message}")
    return out_path, generated, notes


def ensure_experiment_summary(paths: Dict[str, Any], *, dry_run: bool, force_eval: bool) -> Tuple[Optional[str], List[str], List[str]]:
    notes: List[str] = []
    generated: List[str] = []
    out_path = paths.get("summary_json")
    run_dir = paths.get("run_dir")
    if out_path and os.path.exists(out_path) and not force_eval:
        return out_path, generated, notes
    if not run_dir or not os.path.exists(run_dir):
        notes.append("summary_skipped:missing_run_dir")
        return out_path, generated, notes
    out_path = out_path or os.path.join(run_dir, "experiment_summary.json")
    cmd = [
        sys.executable,
        os.path.join("tools", "experiment_summary.py"),
        "--run_dir",
        run_dir,
        "--out_json",
        out_path,
    ]
    optional_args = {
        "--track_stats_json": paths.get("track_stats_json"),
        "--rraa_stats_json": paths.get("rraa_stats_json"),
        "--ligt_quality_json": paths.get("quality_stats_json"),
        "--ligt_degeneracy_json": paths.get("ligt_degeneracy_json"),
        "--pa_degeneracy_json": paths.get("pa_degeneracy_json"),
        "--qtrack_npz": paths.get("qtrack_npz"),
        "--rraa_eval_json": paths.get("rotation_eval_json"),
        "--pose_eval_json": paths.get("translation_eval_json"),
    }
    for flag, value in optional_args.items():
        if value:
            cmd.extend([flag, value])
    ok, message = run_command(cmd, cwd=REPO_ROOT, dry_run=dry_run)
    if ok:
        generated.append(out_path)
        notes.append(f"generated_summary:{message}")
    else:
        notes.append(f"summary_failed:{message}")
    return out_path, generated, notes


def extract_from_payloads(
    *,
    scene: Dict[str, Any],
    exp: Dict[str, Any],
    paths: Dict[str, Any],
    track_stats: Any,
    quality_stats: Any,
    ligt_stats: Any,
    pa_stats: Any,
    rraa_stats: Any,
    rotation_eval: Any,
    translation_eval: Any,
    summary_payload: Any,
    qtrack_npz: Optional[np.lib.npyio.NpzFile],
) -> Dict[str, Any]:
    q_track_summary = None
    if isinstance(qtrack_npz, np.lib.npyio.NpzFile) and "q_track" in qtrack_npz.files:
        q = qtrack_npz["q_track"].astype(np.float64).reshape(-1)
        q = q[np.isfinite(q)]
        if q.size:
            q_track_summary = {
                "count": int(q.size),
                "median": float(np.median(q)),
                "p10": float(np.quantile(q, 0.1)),
                "p90": float(np.quantile(q, 0.9)),
            }

    best_rot = choose_best_rotation_report(rotation_eval)
    anchor_med, anchor_p90 = summarize_anchor_sensitivity(
        rraa_stats.get("anchor_sensitivity") if isinstance(rraa_stats, dict) else None
    )
    quality_cfg = quality_stats.get("config", {}) if isinstance(quality_stats, dict) else {}
    quality_used = quality_stats.get("quality_config_used", {}) if isinstance(quality_stats, dict) else {}

    row = {
        "dataset": scene["dataset"],
        "scene": scene["scene"],
        "scene_slug": scene["scene_slug"],
        "scene_tags": scene.get("tags", []),
        "experiment_group": exp["name"],
        "experiment_tags": exp.get("tags", []),
        "rotation_source": exp.get("rotation_source"),
        "use_pa": bool(exp.get("use_pa", False)),
        "qtrack_mode": exp.get("qtrack_mode")
        or (ligt_stats.get("qtrack_mode") if isinstance(ligt_stats, dict) else None)
        or quality_cfg.get("qtrack_mode"),
        "enable_quality_weighting": (
            safe_bool(exp.get("enable_quality_weighting"))
            if exp.get("enable_quality_weighting") is not None
            else safe_bool(quality_cfg.get("enable_quality_weighting"))
        ),
        "rraa_use_qpair_weight": (
            safe_bool(exp.get("rraa_use_qpair_weight"))
            if exp.get("rraa_use_qpair_weight") is not None
            else safe_bool(
                (rraa_stats.get("quality_config_used", {}) if isinstance(rraa_stats, dict) else {})
                .get("effective_section", {})
                .get("rraa_use_qpair_weight")
            )
        ),
        "u_min": safe_float(exp.get("u_min")) or safe_float(quality_used.get("decision_parameters", {}).get("u_min")) or safe_float(quality_cfg.get("u_min")),
        "g_min": safe_float(exp.get("g_min")) or safe_float(quality_used.get("decision_parameters", {}).get("g_min")) or safe_float(quality_cfg.get("g_min")),
        "run_dir": paths.get("run_dir"),
        "scene_run_root": paths.get("scene_run_root"),
        "prepared_scene_dir": paths.get("prepared_scene_dir"),
        "rotation_median_deg": safe_float(best_rot.get("median_deg") if best_rot else None),
        "rotation_mean_deg": safe_float(best_rot.get("mean_deg") if best_rot else None),
        "rotation_p90_deg": safe_float(best_rot.get("p90_deg") if best_rot else None),
        "rotation_max_deg": safe_float(best_rot.get("max_deg") if best_rot else None),
        "translation_raw_median": safe_float((translation_eval or {}).get("raw", {}).get("median") if isinstance(translation_eval, dict) else None),
        "translation_raw_mean": safe_float((translation_eval or {}).get("raw", {}).get("mean") if isinstance(translation_eval, dict) else None),
        "translation_raw_rmse": safe_float((translation_eval or {}).get("raw", {}).get("rmse") if isinstance(translation_eval, dict) else None),
        "translation_raw_p90": safe_float((translation_eval or {}).get("raw", {}).get("p90") if isinstance(translation_eval, dict) else None),
        "translation_raw_max": safe_float((translation_eval or {}).get("raw", {}).get("max") if isinstance(translation_eval, dict) else None),
        "translation_mm_median": safe_float((translation_eval or {}).get("mm", {}).get("median") if isinstance(translation_eval, dict) else None),
        "translation_mm_mean": safe_float((translation_eval or {}).get("mm", {}).get("mean") if isinstance(translation_eval, dict) else None),
        "translation_mm_rmse": safe_float((translation_eval or {}).get("mm", {}).get("rmse") if isinstance(translation_eval, dict) else None),
        "translation_mm_p90": safe_float((translation_eval or {}).get("mm", {}).get("p90") if isinstance(translation_eval, dict) else None),
        "translation_mm_max": safe_float((translation_eval or {}).get("mm", {}).get("max") if isinstance(translation_eval, dict) else None),
        "rraa_kept_ratio": safe_float((rraa_stats or {}).get("kept_ratio") if isinstance(rraa_stats, dict) else None),
        "rraa_graph_component_count": safe_float((rraa_stats or {}).get("graph_after_filter", {}).get("component_count") if isinstance(rraa_stats, dict) else None),
        "rraa_graph_largest_component_ratio": safe_float((rraa_stats or {}).get("graph_after_filter", {}).get("largest_component_ratio") if isinstance(rraa_stats, dict) else None),
        "rraa_is_connected": safe_bool((rraa_stats or {}).get("graph_after_filter", {}).get("is_connected") if isinstance(rraa_stats, dict) else None),
        "rraa_anchor_degree": safe_float((rraa_stats or {}).get("graph_after_filter", {}).get("anchor_degree") if isinstance(rraa_stats, dict) else None),
        "rraa_anchor_component_ratio": safe_float((rraa_stats or {}).get("graph_after_filter", {}).get("anchor_component_ratio") if isinstance(rraa_stats, dict) else None),
        "rraa_anchor_sensitivity_median_deg": anchor_med,
        "rraa_anchor_sensitivity_p90_deg": anchor_p90,
        "tracks_count": safe_float((track_stats or {}).get("track_stats", {}).get("count") if isinstance(track_stats, dict) else None),
        "track_len_median": safe_float((track_stats or {}).get("track_stats", {}).get("track_length", {}).get("median") if isinstance(track_stats, dict) else None),
        "q_pair_median": safe_float((track_stats or {}).get("pair_stats", {}).get("q_pair", {}).get("median") if isinstance(track_stats, dict) else None),
        "q_track_count": safe_float((q_track_summary or {}).get("count") or (quality_stats or {}).get("track_quality", {}).get("count") if isinstance(quality_stats, dict) else None),
        "q_track_median": safe_float((q_track_summary or {}).get("median") or (quality_stats or {}).get("track_quality", {}).get("median") if isinstance(quality_stats, dict) else None),
        "q_track_p10": safe_float((q_track_summary or {}).get("p10") or (quality_stats or {}).get("track_quality", {}).get("q10") if isinstance(quality_stats, dict) else None),
        "q_track_p90": safe_float((q_track_summary or {}).get("p90") or (quality_stats or {}).get("track_quality", {}).get("q90") if isinstance(quality_stats, dict) else None),
        "ligt_tracks_kept": safe_float((ligt_stats or {}).get("tracks_kept") if isinstance(ligt_stats, dict) else None),
        "ligt_equations_kept": safe_float((ligt_stats or {}).get("equations_kept") if isinstance(ligt_stats, dict) else None),
        "ligt_g_reject_ratio": safe_float((ligt_stats or {}).get("g_reject_ratio") if isinstance(ligt_stats, dict) else None),
        "qtrack_reject_ratio": safe_float((ligt_stats or {}).get("qtrack_reject_ratio") if isinstance(ligt_stats, dict) else None),
        "kept_track_ratio": safe_float((ligt_stats or {}).get("kept_track_ratio") if isinstance(ligt_stats, dict) else None),
        "pa_status": (pa_stats or {}).get("status") if isinstance(pa_stats, dict) else None,
        "pa_reason_code": (pa_stats or {}).get("reason_code") if isinstance(pa_stats, dict) else None,
        "pa_failure_stage": (pa_stats or {}).get("failure_stage") if isinstance(pa_stats, dict) else None,
        "pa_final_rmse": safe_float((pa_stats or {}).get("final_rmse") if isinstance(pa_stats, dict) else None),
        "pa_accept_iters": safe_float(len((pa_stats or {}).get("iterations", [])) if isinstance(pa_stats, dict) and isinstance(pa_stats.get("iterations"), list) else None),
        "runtime_total_sec": safe_float(summary_payload.get("runtime_total_sec") if isinstance(summary_payload, dict) else None),
        "peak_memory_mb": safe_float(summary_payload.get("peak_memory_mb") if isinstance(summary_payload, dict) else None),
        "runtime_rraa_sec": safe_float((summary_payload or {}).get("runtime_by_step_sec", {}).get("rraa") if isinstance(summary_payload, dict) and isinstance(summary_payload.get("runtime_by_step_sec"), dict) else None),
        "runtime_pose_only_sec": safe_float((summary_payload or {}).get("runtime_by_step_sec", {}).get("pose_only") if isinstance(summary_payload, dict) and isinstance(summary_payload.get("runtime_by_step_sec"), dict) else None),
        "threshold_group": None,
        "baseline_experiment_group": None,
        "delta_translation_mm_median": None,
        "delta_translation_mm_p90": None,
        "status": "failed",
        "missing_files": [],
        "generated_files": [],
        "notes": [],
    }

    if row["u_min"] is not None and row["g_min"] is not None:
        row["threshold_group"] = f"u={row['u_min']:.0e},g={row['g_min']:.0e}"

    if isinstance(summary_payload, dict):
        row["tracks_count"] = row["tracks_count"] if row["tracks_count"] is not None else safe_float(summary_payload.get("tracks_count"))
        row["track_len_median"] = row["track_len_median"] if row["track_len_median"] is not None else safe_float(summary_payload.get("track_len_median"))
        row["q_pair_median"] = row["q_pair_median"] if row["q_pair_median"] is not None else safe_float(summary_payload.get("q_pair_median"))
        q_track_payload = summary_payload.get("q_track")
        if isinstance(q_track_payload, dict):
            row["q_track_count"] = row["q_track_count"] if row["q_track_count"] is not None else safe_float(q_track_payload.get("count"))
            row["q_track_median"] = row["q_track_median"] if row["q_track_median"] is not None else safe_float(q_track_payload.get("median"))
            row["q_track_p10"] = row["q_track_p10"] if row["q_track_p10"] is not None else safe_float(q_track_payload.get("p10"))
            row["q_track_p90"] = row["q_track_p90"] if row["q_track_p90"] is not None else safe_float(q_track_payload.get("p90"))

    return row


def determine_status(row: Dict[str, Any], critical_missing: bool) -> str:
    has_rotation = row.get("rotation_median_deg") is not None
    has_translation = row.get("translation_mm_median") is not None or row.get("translation_raw_median") is not None
    if has_rotation and has_translation:
        return "partial" if critical_missing else "ok"
    if has_rotation or has_translation:
        return "partial"
    return "failed"


def compute_pa_deltas(rows: List[Dict[str, Any]]) -> None:
    base_lookup: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("dataset"),
            row.get("scene"),
            row.get("rotation_source"),
            row.get("qtrack_mode"),
            row.get("enable_quality_weighting"),
            row.get("rraa_use_qpair_weight"),
            row.get("u_min"),
            row.get("g_min"),
        )
        if not row.get("use_pa"):
            base_lookup[key] = row
    for row in rows:
        if not row.get("use_pa"):
            continue
        key = (
            row.get("dataset"),
            row.get("scene"),
            row.get("rotation_source"),
            row.get("qtrack_mode"),
            row.get("enable_quality_weighting"),
            row.get("rraa_use_qpair_weight"),
            row.get("u_min"),
            row.get("g_min"),
        )
        base = base_lookup.get(key)
        if not base:
            continue
        row["baseline_experiment_group"] = base.get("experiment_group")
        if row.get("translation_mm_median") is not None and base.get("translation_mm_median") is not None:
            row["delta_translation_mm_median"] = float(base["translation_mm_median"]) - float(row["translation_mm_median"])
        if row.get("translation_mm_p90") is not None and base.get("translation_mm_p90") is not None:
            row["delta_translation_mm_p90"] = float(base["translation_mm_p90"]) - float(row["translation_mm_p90"])


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


def build_long_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    long_rows: List[Dict[str, Any]] = []
    base_keys = [k for k in MAIN_COLUMNS if k not in CORE_NUMERIC_FIELDS]
    for row in rows:
        base = {k: row.get(k) for k in base_keys}
        for metric_name in CORE_NUMERIC_FIELDS:
            value = row.get(metric_name)
            if value is None:
                continue
            item = dict(base)
            item["metric_name"] = metric_name
            item["metric_value"] = value
            item["metric_group"] = metric_group(metric_name)
            long_rows.append(item)
    return long_rows


def aggregate_rows(rows: List[Dict[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        buckets.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for key, items in buckets.items():
        record = {group_keys[i]: key[i] for i in range(len(group_keys))}
        record["row_count"] = len(items)
        rotation_vals = [float(r["rotation_median_deg"]) for r in items if r.get("rotation_median_deg") is not None]
        translation_med = [float(r["translation_mm_median"]) for r in items if r.get("translation_mm_median") is not None]
        translation_p90 = [float(r["translation_mm_p90"]) for r in items if r.get("translation_mm_p90") is not None]
        record["rotation_median_deg"] = float(np.mean(rotation_vals)) if rotation_vals else None
        record["translation_mm_median"] = float(np.mean(translation_med)) if translation_med else None
        record["translation_mm_p90"] = float(np.mean(translation_p90)) if translation_p90 else None
        record["failure_rate"] = float(sum(1 for r in items if r.get("status") == "failed") / max(len(items), 1))
        pa_rows = [r for r in items if r.get("use_pa")]
        record["pa_accept_rate"] = (
            float(sum(1 for r in pa_rows if str(r.get("pa_status", "")).lower() == "accepted") / max(len(pa_rows), 1))
            if pa_rows else None
        )
        out.append(record)
    out.sort(key=lambda item: tuple("" if item.get(k) is None else str(item.get(k)) for k in group_keys))
    return out


def build_pa_status_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[Tuple[Any, ...], int] = {}
    for row in rows:
        key = (
            row.get("experiment_group"),
            row.get("dataset"),
            row.get("scene"),
            row.get("pa_status"),
            row.get("pa_reason_code"),
        )
        counts[key] = counts.get(key, 0) + 1
    out = []
    for key, count in sorted(counts.items()):
        out.append(
            {
                "experiment_group": key[0],
                "dataset": key[1],
                "scene": key[2],
                "pa_status": key[3],
                "pa_reason_code": key[4],
                "count": count,
            }
        )
    return out


def build_missing_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("status") == "ok" and not row.get("missing_files"):
            continue
        out.append(
            {
                "dataset": row.get("dataset"),
                "scene": row.get("scene"),
                "experiment_group": row.get("experiment_group"),
                "status": row.get("status"),
                "run_dir": row.get("run_dir"),
                "missing_files": row.get("missing_files"),
                "generated_files": row.get("generated_files"),
                "notes": row.get("notes"),
            }
        )
    return out


def evaluate_one(
    scene: Dict[str, Any],
    exp: Dict[str, Any],
    manifest: Dict[str, Any],
    *,
    dry_run: bool,
    force_eval: bool,
) -> Dict[str, Any]:
    paths = resolve_scene_experiment_paths(scene, exp, manifest)
    generated_files: List[str] = []
    notes: List[str] = []
    missing_files: List[str] = []

    add_missing(missing_files, "run_dir", paths.get("run_dir"))
    add_missing(missing_files, "prepared_scene_dir", paths.get("prepared_scene_dir"))
    add_missing(missing_files, "scene_run_root", paths.get("scene_run_root"))
    add_missing(missing_files, "gt_rot_npy", paths.get("gt_rot_npy"))
    add_missing(missing_files, "gt_centers_npy", paths.get("gt_centers_npy"))

    rotation_eval_path, gen, msg = ensure_rotation_eval(paths, dry_run=dry_run, force_eval=force_eval)
    generated_files.extend(gen)
    notes.extend(msg)
    if rotation_eval_path:
        paths["rotation_eval_json"] = rotation_eval_path

    translation_eval_path, gen, msg = ensure_translation_eval(
        paths,
        gt_unit_to_mm=safe_float(scene.get("gt_unit_to_mm", manifest.get("defaults", {}).get("gt_unit_to_mm", 1000.0))),
        dry_run=dry_run,
        force_eval=force_eval,
    )
    generated_files.extend(gen)
    notes.extend(msg)
    if translation_eval_path:
        paths["translation_eval_json"] = translation_eval_path

    summary_path, gen, msg = ensure_experiment_summary(paths, dry_run=dry_run, force_eval=force_eval)
    generated_files.extend(gen)
    notes.extend(msg)
    if summary_path:
        paths["summary_json"] = summary_path

    rotation_eval = load_json_if_exists(paths.get("rotation_eval_json"))
    translation_eval = load_json_if_exists(paths.get("translation_eval_json"))
    summary_payload = load_json_if_exists(paths.get("summary_json"))
    quality_stats = load_json_if_exists(paths.get("quality_stats_json"))
    ligt_stats = load_json_if_exists(paths.get("ligt_degeneracy_json"))
    pa_stats = load_json_if_exists(paths.get("pa_degeneracy_json"))
    rraa_stats = load_json_if_exists(paths.get("rraa_stats_json"))
    track_stats = load_json_if_exists(paths.get("track_stats_json"))
    qtrack_npz = load_npz_if_exists(paths.get("qtrack_npz"))

    expected_optional = {
        "track_stats_json": True,
        "quality_stats_json": True,
        "ligt_degeneracy_json": True,
        "qtrack_npz": True,
        "rraa_stats_json": True,
        "rotation_eval_json": True,
        "translation_eval_json": True,
        "summary_json": False,
        "pa_degeneracy_json": bool(exp.get("use_pa", False)),
        "track_quality_summary_npz": False,
        "pair_quality_edges_npz": False,
    }
    for key, expected in expected_optional.items():
        if expected:
            add_missing(missing_files, key, paths.get(key))
        elif key in {"track_quality_summary_npz", "pair_quality_edges_npz"} and paths.get(key) and not os.path.exists(paths.get(key)):
            missing_files.append(f"{key}:{paths.get(key)}")

    row = extract_from_payloads(
        scene=scene,
        exp=exp,
        paths=paths,
        track_stats=track_stats,
        quality_stats=quality_stats,
        ligt_stats=ligt_stats,
        pa_stats=pa_stats,
        rraa_stats=rraa_stats,
        rotation_eval=rotation_eval,
        translation_eval=translation_eval,
        summary_payload=summary_payload,
        qtrack_npz=qtrack_npz,
    )
    row["generated_files"] = sorted(set(generated_files))
    row["notes"] = notes
    row["missing_files"] = missing_files
    critical_missing = (
        row.get("rotation_median_deg") is None
        or (row.get("translation_mm_median") is None and row.get("translation_raw_median") is None)
    )
    row["status"] = determine_status(row, critical_missing=critical_missing)
    return row


def parse_csv_list_arg(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified paper-friendly benchmark aggregation for Pose-only experiments.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--scene_tags", default=None)
    ap.add_argument("--experiment_tags", default=None)
    ap.add_argument("--experiment_groups", default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force_eval", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    defaults = manifest.get("defaults", {})
    if args.output_dir:
        # CLI-provided paths should behave like normal shell paths and resolve
        # relative to the caller's working directory, not the manifest folder.
        output_dir = resolve_path(args.output_dir, base_dir=os.getcwd())
    else:
        output_dir = resolve_path(
            defaults.get("output_dir") or "runs/paper_benchmark_eval",
            base_dir=manifest["_manifest_dir"],
        )
    os.makedirs(output_dir, exist_ok=True)

    scenes = select_scene_entries(manifest, parse_csv_list_arg(args.scene_tags))
    experiments = select_experiments(
        manifest,
        parse_csv_list_arg(args.experiment_groups),
        parse_csv_list_arg(args.experiment_tags),
    )

    rows: List[Dict[str, Any]] = []
    for scene in scenes:
        for exp in experiments:
            print(f"[paper-benchmark] dataset={scene['dataset']} scene={scene['scene']} experiment={exp['name']}")
            rows.append(
                evaluate_one(
                    scene,
                    exp,
                    manifest,
                    dry_run=args.dry_run,
                    force_eval=args.force_eval,
                )
            )

    compute_pa_deltas(rows)
    rows.sort(key=lambda item: (str(item.get("dataset")), str(item.get("scene")), str(item.get("experiment_group"))))

    long_rows = build_long_rows(rows)
    pa_status_rows = build_pa_status_rows(rows)
    missing_rows = build_missing_rows(rows)
    aggregate_scene_group = aggregate_rows(rows, ["experiment_group"])
    aggregate_dataset = aggregate_rows(rows, ["dataset", "experiment_group"])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": manifest["_manifest_path"],
        "output_dir": output_dir,
        "scene_count": len(scenes),
        "experiment_count": len(experiments),
        "rows": rows,
    }

    dump_json(os.path.join(output_dir, "paper_summary_main.json"), payload)
    write_csv(os.path.join(output_dir, "paper_summary_main.csv"), rows, MAIN_COLUMNS)
    write_csv(os.path.join(output_dir, "paper_summary_long.csv"), long_rows, list(MAIN_COLUMNS) + ["metric_name", "metric_value", "metric_group"])
    write_csv(os.path.join(output_dir, "paper_pa_status_counts.csv"), pa_status_rows, ["experiment_group", "dataset", "scene", "pa_status", "pa_reason_code", "count"])
    write_csv(os.path.join(output_dir, "paper_missing_or_failed.csv"), missing_rows, ["dataset", "scene", "experiment_group", "status", "run_dir", "missing_files", "generated_files", "notes"])
    write_csv(os.path.join(output_dir, "paper_aggregate_by_scene_group.csv"), aggregate_scene_group, ["experiment_group", "row_count", "rotation_median_deg", "translation_mm_median", "translation_mm_p90", "failure_rate", "pa_accept_rate"])
    write_csv(os.path.join(output_dir, "paper_aggregate_by_dataset.csv"), aggregate_dataset, ["dataset", "experiment_group", "row_count", "rotation_median_deg", "translation_mm_median", "translation_mm_p90", "failure_rate", "pa_accept_rate"])

    print("saved:", os.path.join(output_dir, "paper_summary_main.json"))
    print("saved:", os.path.join(output_dir, "paper_summary_main.csv"))
    print("saved:", os.path.join(output_dir, "paper_summary_long.csv"))
    print("saved:", os.path.join(output_dir, "paper_pa_status_counts.csv"))
    print("saved:", os.path.join(output_dir, "paper_missing_or_failed.csv"))
    print("saved:", os.path.join(output_dir, "paper_aggregate_by_scene_group.csv"))
    print("saved:", os.path.join(output_dir, "paper_aggregate_by_dataset.csv"))


if __name__ == "__main__":
    main()
