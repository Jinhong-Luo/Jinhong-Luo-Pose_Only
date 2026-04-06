#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict

import numpy as np

from degeneracy_utils import dump_json


def load_json_if_exists(path: str):
    if not path:
        return None
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_npz_if_exists(path: str):
    if not path:
        return None
    if not os.path.exists(path):
        return None
    return np.load(path)


def pick(stats: Dict, *keys, default=None):
    cur = stats
    for key in keys:
        if cur is None or key not in cur:
            return default
        cur = cur[key]
    return cur


def first_existing_path(*paths: str):
    for path in paths:
        if not path:
            continue
        if os.path.exists(path):
            return path
    return None


def main():
    ap = argparse.ArgumentParser(description="Summarize one experiment directory into a compact JSON report.")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--track_stats_json", default=None)
    ap.add_argument("--rraa_stats_json", default=None)
    ap.add_argument("--ligt_quality_json", default=None)
    ap.add_argument("--ligt_degeneracy_json", default=None)
    ap.add_argument("--pa_degeneracy_json", default=None)
    ap.add_argument("--qtrack_npz", default=None)
    ap.add_argument("--rraa_eval_json", default=None)
    ap.add_argument("--pose_eval_json", default=None)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    track_stats = load_json_if_exists(first_existing_path(
        args.track_stats_json,
        os.path.join(run_dir, "tracks_npz_qd", "track_build_quality_stats.json"),
        os.path.join(run_dir, "tracks", "track_build_quality_stats.json"),
    ))
    rraa_stats = load_json_if_exists(first_existing_path(
        args.rraa_stats_json,
        os.path.join(run_dir, "rraa_output_qd", "R_abs_qw_stats.json"),
        os.path.join(run_dir, "rraa_output_qd", "R_abs_stats.json"),
        os.path.join(run_dir, "rraa_output", "R_abs_stats.json"),
    ))
    ligt_quality = load_json_if_exists(first_existing_path(
        args.ligt_quality_json,
        os.path.join(run_dir, "poseonly_qd", "quality_stats.json"),
        os.path.join(run_dir, "pose_only", "quality_stats.json"),
    ))
    ligt_deg = load_json_if_exists(first_existing_path(
        args.ligt_degeneracy_json,
        os.path.join(run_dir, "poseonly_qd", "ligt_degeneracy_stats.json"),
        os.path.join(run_dir, "pose_only", "ligt_degeneracy_stats.json"),
    ))
    pa_deg = load_json_if_exists(first_existing_path(
        args.pa_degeneracy_json,
        os.path.join(run_dir, "poseonly_qd", "pa_degeneracy_stats.json"),
        os.path.join(run_dir, "pose_only", "pa_degeneracy_stats.json"),
    ))
    qtrack_npz = load_npz_if_exists(first_existing_path(
        args.qtrack_npz,
        os.path.join(run_dir, "poseonly_qd", "track_quality_scores.npz"),
        os.path.join(run_dir, "pose_only", "track_quality_scores.npz"),
    ))
    rraa_eval = load_json_if_exists(first_existing_path(
        args.rraa_eval_json,
        os.path.join(run_dir, "rraa_output_qd", "eval_rotation_qw.json"),
        os.path.join(run_dir, "rraa_output_qd", "eval_rotation_baseline.json"),
        os.path.join(run_dir, "rraa_output", "eval_rotation.json"),
    ))
    pose_eval = load_json_if_exists(first_existing_path(
        args.pose_eval_json,
        os.path.join(run_dir, "poseonly_qd", "eval_translation_pa.json"),
        os.path.join(run_dir, "poseonly_qd", "eval_translation_ligt.json"),
        os.path.join(run_dir, "poseonly_baseline", "eval_translation_ligt.json"),
        os.path.join(run_dir, "pose_only", "eval_translation.json"),
    ))

    qtrack_summary = None
    if qtrack_npz is not None and "q_track" in qtrack_npz.files:
        q = qtrack_npz["q_track"].astype(np.float64)
        if q.size:
            qtrack_summary = {
                "count": int(q.size),
                "median": float(np.median(q)),
                "p10": float(np.quantile(q, 0.1)),
                "p90": float(np.quantile(q, 0.9)),
            }

    best_rraa_eval = None
    if isinstance(rraa_eval, list) and rraa_eval:
        best_rraa_eval = min(
            [item for item in rraa_eval if isinstance(item, dict) and "median_deg" in item],
            key=lambda x: float(x["median_deg"]),
            default=None,
        )
    elif isinstance(rraa_eval, dict):
        best_rraa_eval = rraa_eval

    summary = {
        "run_dir": run_dir,
        "tracks_count": pick(track_stats, "track_stats", "count"),
        "track_len_median": pick(track_stats, "track_stats", "track_length", "median"),
        "q_pair_median": pick(track_stats, "pair_stats", "q_pair", "median"),
        "q_track": qtrack_summary or pick(ligt_quality, "track_quality"),
        "ligt_tracks_kept": pick(ligt_deg, "tracks_kept"),
        "ligt_equations_kept": pick(ligt_deg, "equations_kept"),
        "ligt_g_reject_ratio": pick(ligt_deg, "g_reject_ratio"),
        "rraa_kept_ratio": pick(rraa_stats, "kept_ratio"),
        "rraa_graph_component_count": pick(rraa_stats, "graph_after_filter", "component_count"),
        "rraa_graph_largest_component_ratio": pick(rraa_stats, "graph_after_filter", "largest_component_ratio"),
        "pa_status": pick(pa_deg, "status"),
        "pa_reason_code": pick(pa_deg, "reason_code"),
        "pa_failure_stage": pick(pa_deg, "failure_stage"),
        "pa_final_rmse": pick(pa_deg, "final_rmse"),
        "pa_accept_iters": len(pick(pa_deg, "iterations", default=[])),
        "rotation_eval_median_deg": pick(best_rraa_eval, "median_deg"),
        "translation_eval_raw_median": pick(pose_eval, "raw", "median"),
        "translation_eval_mm_median": pick(pose_eval, "mm", "median"),
    }

    out_json = args.out_json or os.path.join(run_dir, "experiment_summary.json")
    dump_json(out_json, summary)
    print("saved:", out_json)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
