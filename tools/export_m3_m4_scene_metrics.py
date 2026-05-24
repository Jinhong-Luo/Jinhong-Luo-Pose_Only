#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from eval_rraa_rotation import eval_one


REPO_ROOT = Path(__file__).resolve().parents[1]


RUNS = [
    (
        "BaselineB",
        "Paper-Base / A3",
        REPO_ROOT
        / "runs"
        / "paper_v2"
        / "ablation_clean_main"
        / "A"
        / "A3_paper_base"
        / "trials"
        / "trial_0000"
        / "validation_results.json",
    ),
    (
        "M3",
        "A4 + IRLS(2), k=1.5",
        REPO_ROOT
        / "runs"
        / "paper_v2"
        / "ablation_clean_main"
        / "I"
        / "I2_gap8_irls2_huber15"
        / "trials"
        / "trial_0000"
        / "validation_results.json",
    ),
    (
        "M4",
        "M3 + translation-only PA",
        REPO_ROOT
        / "runs"
        / "paper_v2"
        / "ablation_clean_main"
        / "M"
        / "M4_gap8_irls2_pa"
        / "trials"
        / "trial_0000"
        / "validation_results.json",
    ),
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def stage_runtime_and_memory(scene: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    elapsed: List[float] = []
    peaks: List[float] = []
    for stage in scene.get("stages") or []:
        if not isinstance(stage, dict) or stage.get("status") == "reused":
            continue
        e = as_float(stage.get("elapsed_sec"))
        p = as_float(stage.get("peak_working_set_mb"))
        if e is not None:
            elapsed.append(e)
        if p is not None:
            peaks.append(p)
    return (sum(elapsed) if elapsed else None, max(peaks) if peaks else None)


def camera_position_std_mm(scene: Dict[str, Any]) -> Optional[float]:
    mean = as_float(scene.get("translation_mm_mean"))
    rmse = as_float(scene.get("translation_mm_rmse"))
    if mean is not None and rmse is not None:
        variance = max(0.0, rmse * rmse - mean * mean)
        return math.sqrt(variance)

    eval_path = scene.get("pose_eval_json")
    if not eval_path:
        eval_path = (
            scene.get("artifacts", {}).get("pose_eval_json")
            if isinstance(scene.get("artifacts"), dict)
            else None
        )
    if not eval_path:
        return None
    path = Path(eval_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return None
    payload = load_json(path)
    mm = payload.get("mm") or {}
    mean = as_float(mm.get("mean"))
    rmse = as_float(mm.get("rmse"))
    if mean is None or rmse is None:
        return None
    variance = max(0.0, rmse * rmse - mean * mean)
    return math.sqrt(variance)


def camera_position_std_mm_from_summary(scene: Dict[str, Any]) -> Optional[float]:
    value = camera_position_std_mm(scene)
    if value is not None:
        return value

    summary_json = scene.get("summary_json")
    if not summary_json:
        return None
    summary_path = Path(summary_json)
    eval_path = summary_path.parent / "pose_only" / "eval_translation.json"
    if not eval_path.exists():
        return None
    payload = load_json(eval_path)
    mm = payload.get("mm") or {}
    mean = as_float(mm.get("mean"))
    rmse = as_float(mm.get("rmse"))
    if mean is None or rmse is None:
        return None
    return math.sqrt(max(0.0, rmse * rmse - mean * mean))


def camera_position_ate_rmse_mm(scene: Dict[str, Any]) -> Optional[float]:
    value = as_float(scene.get("translation_mm_rmse"))
    if value is not None:
        return value

    summary_json = scene.get("summary_json")
    if not summary_json:
        return None
    summary_path = Path(summary_json)
    eval_path = summary_path.parent / "pose_only" / "eval_translation.json"
    if not eval_path.exists():
        return None
    payload = load_json(eval_path)
    return as_float((payload.get("mm") or {}).get("rmse"))


def arg_after(cmd: List[Any], flag: str) -> Optional[str]:
    for idx, item in enumerate(cmd):
        if str(item) == flag and idx + 1 < len(cmd):
            return str(cmd[idx + 1])
    return None


def stage_cmd(scene: Dict[str, Any], stage_name: str) -> Optional[List[Any]]:
    for stage in scene.get("stages") or []:
        if isinstance(stage, dict) and stage.get("name") == stage_name:
            cmd = stage.get("cmd")
            if isinstance(cmd, list):
                return cmd
    return None


def rotation_error_stats(scene: Dict[str, Any]) -> Dict[str, Optional[float]]:
    cmd = stage_cmd(scene, "eval_rotation")
    if not cmd:
        return {"rotation_ate_rmse_deg": None, "rotation_std_error_deg": None}
    est_npy = arg_after(cmd, "--est_npy")
    gt_npy = arg_after(cmd, "--gt_npy")
    if not est_npy or not gt_npy:
        return {"rotation_ate_rmse_deg": None, "rotation_std_error_deg": None}

    est_path = Path(est_npy)
    gt_path = Path(gt_npy)
    if not est_path.is_absolute():
        est_path = REPO_ROOT / est_path
    if not gt_path.is_absolute():
        gt_path = REPO_ROOT / gt_path
    if not est_path.exists() or not gt_path.exists():
        return {"rotation_ate_rmse_deg": None, "rotation_std_error_deg": None}

    R_est = np.load(est_path).astype(np.float64)
    R_gt = np.load(gt_path).astype(np.float64)
    n = min(R_est.shape[0], R_gt.shape[0])
    R_est = R_est[:n]
    R_gt = R_gt[:n]

    reports = []
    for name, rotations in [
        ("est", R_est),
        ("est_T", np.transpose(R_est, (0, 2, 1))),
    ]:
        reports.append(eval_one(name, R_gt, rotations, "right"))
        reports.append(eval_one(name, R_gt, rotations, "left"))
    best = min(reports, key=lambda item: float(item["median_deg"]))
    mean = as_float(best.get("mean_deg"))
    # The original evaluator does not return per-camera errors, but it does
    # expose the aligned rotations through the best gauge. Recompute them here.
    if best["name"] == "est_T":
        rotations = np.transpose(R_est, (0, 2, 1))
    else:
        rotations = R_est
    aligned = eval_one(best["name"], R_gt, rotations, best["gauge"])
    q = aligned["Q"]
    if best["gauge"] == "right":
        R_aligned = rotations @ q[None, :, :]
    else:
        R_aligned = q[None, :, :] @ rotations
    R_err = R_gt @ np.transpose(R_aligned, (0, 2, 1))
    traces = np.trace(R_err, axis1=-2, axis2=-1)
    angles = np.degrees(np.arccos(np.clip((traces - 1.0) * 0.5, -1.0, 1.0)))
    rmse = float(np.sqrt(np.mean(angles * angles)))
    if mean is None:
        std = float(np.std(angles))
    else:
        std = float(np.sqrt(max(0.0, float(np.mean(angles * angles)) - mean * mean)))
    return {"rotation_ate_rmse_deg": rmse, "rotation_std_error_deg": std}


def rows_for_run(label: str, protocol: str, validation_path: Path) -> List[Dict[str, Any]]:
    payload = load_json(validation_path)
    candidate = (payload.get("candidates") or [{}])[0]
    rows: List[Dict[str, Any]] = []
    for scene in candidate.get("scenes") or []:
        runtime_sec, peak_memory_mb = stage_runtime_and_memory(scene)
        rot_stats = rotation_error_stats(scene)
        rows.append(
            {
                "method": label,
                "protocol": protocol,
                "scene_name": scene.get("scene_name"),
                "scene_group": scene.get("scene_group"),
                "scene_id": scene.get("scene_id"),
                "status": scene.get("status"),
                "camera_position_mean_error_mm": as_float(scene.get("translation_mm_mean")),
                "camera_position_ate_rmse_mm": camera_position_ate_rmse_mm(scene),
                "camera_position_std_error_mm": camera_position_std_mm_from_summary(scene),
                "camera_position_median_error_mm": as_float(scene.get("translation_mm_median")),
                "camera_position_p90_error_mm": as_float(scene.get("translation_mm_p90")),
                "camera_position_vs_colmap_ratio": as_float(scene.get("translation_vs_colmap_ratio")),
                "rotation_mean_error_deg": as_float(scene.get("rotation_mean_deg")),
                "rotation_ate_rmse_deg": rot_stats["rotation_ate_rmse_deg"],
                "rotation_std_error_deg": rot_stats["rotation_std_error_deg"],
                "rotation_median_error_deg": as_float(scene.get("rotation_median_deg")),
                "rotation_p90_error_deg": as_float(scene.get("rotation_p90_deg")),
                "rotation_max_error_deg": as_float(scene.get("rotation_max_deg")),
                "runtime_total_sec": runtime_sec,
                "peak_memory_mb": peak_memory_mb,
            }
        )
    return rows


def main() -> None:
    out_dir = REPO_ROOT / "runs" / "paper_v2" / "paper_ablation_assets_2026-05-02" / "external_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "m3_m4_scene_metrics_for_external_compare.csv"

    rows: List[Dict[str, Any]] = []
    for label, protocol, path in RUNS:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(rows_for_run(label, protocol, path))

    fieldnames = [
        "method",
        "protocol",
        "scene_name",
        "scene_group",
        "scene_id",
        "status",
        "camera_position_mean_error_mm",
        "camera_position_ate_rmse_mm",
        "camera_position_std_error_mm",
        "camera_position_median_error_mm",
        "camera_position_p90_error_mm",
        "camera_position_vs_colmap_ratio",
        "rotation_mean_error_deg",
        "rotation_ate_rmse_deg",
        "rotation_std_error_deg",
        "rotation_median_error_deg",
        "rotation_p90_error_deg",
        "rotation_max_error_deg",
        "runtime_total_sec",
        "peak_memory_mb",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# BaselineB/M3/M4 Scene Metrics for External Comparison",
                "",
                "File:",
                "- `m3_m4_scene_metrics_for_external_compare.csv`",
                "",
                "Rows are per scene and per method.",
                "",
                "Definitions:",
                "- `camera_position_mean_error_mm`: per-camera mean position error in millimeters from `eval_translation.json`.",
                "- `camera_position_ate_rmse_mm`: ATE RMSE translation error in millimeters from `eval_translation.json`; use this for DTU pose-estimation ATE RMSE comparisons.",
                "- `camera_position_std_error_mm`: per-camera position error standard deviation in millimeters, computed as `sqrt(rmse^2 - mean^2)` from `eval_translation.json`.",
                "- `camera_position_median_error_mm`: per-camera median position error in millimeters.",
                "- `camera_position_p90_error_mm`: per-camera p90 position error in millimeters.",
                "- `rotation_*_error_deg`: RRAA absolute rotation error in degrees.",
                "- `rotation_ate_rmse_deg`: RMSE of per-camera absolute rotation error in degrees; use this for rotation ATE RMSE comparisons.",
                "- `rotation_std_error_deg`: standard deviation of per-camera RRAA absolute rotation error, recomputed from `R_abs.npy` and GT rotations with the same gauge convention as `eval_rotation.json`.",
                "- `runtime_total_sec`: sum of non-reused pipeline stage wall times for the scene.",
                "- `peak_memory_mb`: maximum observed stage peak working set for the scene.",
                "",
                "Methods:",
                "- `BaselineB`: Paper-Base / A3, tracks=1,2,3, RRAA=1,2,3,5, IRLS=0, no quality/fallback/PA.",
                "- `M3`: A4 + IRLS(2), k=1.5.",
                "- `M4`: M3 + translation-only PA.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(out_csv)


if __name__ == "__main__":
    main()
