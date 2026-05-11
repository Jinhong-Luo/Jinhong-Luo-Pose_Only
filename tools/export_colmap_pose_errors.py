#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from colmap_export_poses import qvec_to_rotmat, read_images_binary
from eval_poseonly_strecha_mm import summarize, umeyama_alignment
from eval_rraa_rotation import eval_one


REPO_ROOT = Path(__file__).resolve().parents[1]
COLMAP_ROOT = REPO_ROOT / "Colmap_runs"
PREPARED_ROOT = REPO_ROOT / "data" / "prepared"
OUT_DIR = REPO_ROOT / "runs" / "paper_v2" / "paper_ablation_assets_2026-05-02" / "external_compare"


SCENE_MAP = {
    ("strecha", "fountain-P11"): ("strecha", "fountain-P11", 1000.0),
    ("strecha", "entry-P10"): ("strecha", "entry-P10", 1000.0),
    ("strecha", "Herz-Jesus-P8"): ("strecha", "Herz-Jesus-P8", 1000.0),
    ("strecha", "castle-P19"): ("strecha", "Castle-P19", 1000.0),
    ("strecha", "castle-P30"): ("strecha", "Castle-P30", 1000.0),
    ("strecha", "Herz-Jesus-P25"): ("strecha", "Herz-Jesus-P25", 1000.0),
}


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def center_from_colmap(item: Dict[str, Any]) -> np.ndarray:
    r_w2c = qvec_to_rotmat(item["qvec"])
    return -r_w2c.T @ item["tvec"]


def prepared_scene(dataset: str, scene: str) -> Tuple[str, str, float]:
    key = (dataset, scene)
    if key in SCENE_MAP:
        return SCENE_MAP[key]
    if dataset == "DTU":
        return "DTU", scene, 1.0
    return dataset, scene, 1.0


def load_image_names(image_list: Path) -> List[str]:
    return [
        Path(line.strip().replace("\\", "/")).name
        for line in image_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rotation_stats(R_gt: np.ndarray, R_est: np.ndarray) -> Dict[str, Any]:
    reports = []
    for name, rotations in [
        ("est", R_est),
        ("est_T", np.transpose(R_est, (0, 2, 1))),
    ]:
        reports.append(eval_one(name, R_gt, rotations, "right"))
        reports.append(eval_one(name, R_gt, rotations, "left"))
    best = min(reports, key=lambda item: float(item["median_deg"]))

    rotations = np.transpose(R_est, (0, 2, 1)) if best["name"] == "est_T" else R_est
    q = best["Q"]
    if best["gauge"] == "right":
        R_aligned = rotations @ q[None, :, :]
    else:
        R_aligned = q[None, :, :] @ rotations
    R_err = R_gt @ np.transpose(R_aligned, (0, 2, 1))
    traces = np.trace(R_err, axis1=-2, axis2=-1)
    angles = np.degrees(np.arccos(np.clip((traces - 1.0) * 0.5, -1.0, 1.0)))

    return {
        "rotation_mean_deg": float(np.mean(angles)),
        "rotation_ate_rmse_deg": float(np.sqrt(np.mean(angles * angles))),
        "rotation_rmse_deg": float(np.sqrt(np.mean(angles * angles))),
        "rotation_std_deg": float(np.std(angles)),
        "rotation_median_deg": float(np.median(angles)),
        "rotation_p90_deg": float(np.quantile(angles, 0.9)),
        "rotation_max_deg": float(np.max(angles)),
        "rotation_convention": f"{best['name']}+{best['gauge']}",
    }


def runtime_memory(scene_root: Path) -> Dict[str, Optional[float]]:
    total_sec = 0.0
    peak_gb: Optional[float] = None
    out: Dict[str, Optional[float]] = {}
    for stage in ["feature_extractor", "exhaustive_matcher", "mapper"]:
        path = scene_root / "logs" / stage / "stage_measure.json"
        if not path.exists():
            out[f"{stage}_time_sec"] = None
            out[f"{stage}_peak_memory_gb"] = None
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        stage_sec = float(payload.get("TimeSeconds") or 0.0)
        stage_peak = payload.get("PeakObservedWorkingSetGB")
        stage_peak = float(stage_peak) if stage_peak is not None else None
        total_sec += stage_sec
        if stage_peak is not None:
            peak_gb = stage_peak if peak_gb is None else max(peak_gb, stage_peak)
        out[f"{stage}_time_sec"] = stage_sec
        out[f"{stage}_peak_memory_gb"] = stage_peak
        out[f"{stage}_peak_memory_mb"] = stage_peak * 1024.0 if stage_peak is not None else None
    out["runtime_total_sec"] = total_sec
    out["peak_memory_gb"] = peak_gb
    out["peak_memory_mb"] = peak_gb * 1024.0 if peak_gb is not None else None
    return out


def evaluate_scene(dataset: str, colmap_scene: str) -> Dict[str, Any]:
    colmap_scene_root = COLMAP_ROOT / dataset / colmap_scene
    images_bin = colmap_scene_root / "sparse" / "0" / "images.bin"
    prep_dataset, prep_scene, gt_unit_to_mm = prepared_scene(dataset, colmap_scene)
    prep_root = PREPARED_ROOT / prep_dataset / prep_scene

    images = read_images_binary(str(images_bin))
    by_name = {item["name"]: item for item in images.values()}
    image_names = load_image_names(prep_root / "image_list.txt")
    gt_centers = np.load(prep_root / "gt_centers.npy").astype(np.float64)
    gt_rotations = np.load(prep_root / "R_abs_gt_w2c.npy").astype(np.float64)

    C_est, C_gt, R_est, R_gt = [], [], [], []
    for idx, name in enumerate(image_names):
        item = by_name.get(name)
        if item is None:
            continue
        C_est.append(center_from_colmap(item))
        C_gt.append(gt_centers[idx])
        R_est.append(qvec_to_rotmat(item["qvec"]))
        R_gt.append(gt_rotations[idx])

    if not C_est:
        raise RuntimeError(f"No registered image matched GT image_list for {dataset}/{colmap_scene}")

    C_est_arr = np.asarray(C_est, dtype=np.float64)
    C_gt_arr = np.asarray(C_gt, dtype=np.float64)
    R_est_arr = np.asarray(R_est, dtype=np.float64)
    R_gt_arr = np.asarray(R_gt, dtype=np.float64)

    scale, R_align, t_align = umeyama_alignment(C_est_arr.T, C_gt_arr.T, with_scale=True)
    C_aligned = (scale * (R_align @ C_est_arr.T) + t_align.reshape(3, 1)).T
    err_mm = np.linalg.norm(C_aligned - C_gt_arr, axis=1) * gt_unit_to_mm
    trans = summarize(err_mm)

    row: Dict[str, Any] = {
        "dataset": dataset,
        "scene": colmap_scene,
        "prepared_scene": prep_scene,
        "registered_images": len(C_est),
        "total_gt_images": len(image_names),
        "translation_ate_rmse_mm": trans["rmse"],
        "translation_mean_mm": trans["mean"],
        "translation_std_mm": float(np.std(err_mm)),
        "translation_median_mm": trans["median"],
        "translation_p90_mm": trans["p90"],
        "translation_max_mm": trans["max"],
        "sim3_scale": scale,
        "gt_unit_to_mm": gt_unit_to_mm,
    }
    row.update(rotation_stats(R_gt_arr, R_est_arr))
    row.update(runtime_memory(colmap_scene_root))
    return row


def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    metrics = [
        "translation_ate_rmse_mm",
        "translation_mean_mm",
        "translation_std_mm",
        "translation_median_mm",
        "translation_p90_mm",
        "rotation_mean_deg",
        "rotation_ate_rmse_deg",
        "rotation_rmse_deg",
        "rotation_std_deg",
        "rotation_median_deg",
        "rotation_p90_deg",
        "runtime_total_sec",
        "peak_memory_mb",
        "mapper_time_sec",
        "mapper_peak_memory_mb",
    ]
    for dataset in sorted({row["dataset"] for row in rows}) + ["ALL"]:
        if dataset == "ALL":
            subset = rows
        else:
            subset = [row for row in rows if row["dataset"] == dataset]
        agg: Dict[str, Any] = {
            "dataset": dataset,
            "scene_count": len(subset),
            "registered_images": sum(int(row["registered_images"]) for row in subset),
            "total_gt_images": sum(int(row["total_gt_images"]) for row in subset),
        }
        for metric in metrics:
            vals = [float(row[metric]) for row in subset if row.get(metric) is not None]
            agg[f"{metric}_scene_mean"] = float(np.mean(vals)) if vals else None
        out.append(agg)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})


def main() -> None:
    rows = []
    for dataset_root in sorted([p for p in COLMAP_ROOT.iterdir() if p.is_dir()]):
        dataset = dataset_root.name
        if dataset not in {"strecha", "DTU"}:
            continue
        for scene_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
            if (scene_dir / "sparse" / "0" / "images.bin").exists():
                rows.append(evaluate_scene(dataset, scene_dir.name))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene_csv = OUT_DIR / "colmap_all_pose_runtime_memory.csv"
    aggregate_csv = OUT_DIR / "colmap_all_aggregate.csv"
    write_csv(scene_csv, rows)
    write_csv(aggregate_csv, aggregate_rows(rows))

    # Keep the previous filenames as aliases for existing paper notes.
    write_csv(OUT_DIR / "colmap_strecha_dtu_pose_runtime_memory.csv", rows)
    write_csv(OUT_DIR / "colmap_strecha_dtu_aggregate.csv", aggregate_rows(rows))

    readme = OUT_DIR / "colmap_all_README.md"
    readme.write_text(
        "\n".join(
            [
                "# COLMAP Pose, Runtime, and Memory",
                "",
                "Files:",
                "- `colmap_all_pose_runtime_memory.csv`: per-scene metrics.",
                "- `colmap_all_aggregate.csv`: scene-mean aggregate by dataset plus `ALL`.",
                "- `colmap_strecha_dtu_pose_runtime_memory.csv` and `colmap_strecha_dtu_aggregate.csv` are compatibility aliases with the same content.",
                "",
                "Pose evaluation:",
                "- Translation is Sim(3)-aligned camera-center error.",
                "- `translation_ate_rmse_mm` is the ATE RMSE translation error in millimeters.",
                "- `translation_mean_mm` is the mean camera-center error in millimeters.",
                "- Strecha GT units are converted with `gt_unit_to_mm=1000`; DTU uses `gt_unit_to_mm=1`.",
                "- Rotation is absolute rotation error after the same gauge-convention check used by `tools/eval_rraa_rotation.py`.",
                "- `rotation_ate_rmse_deg` is the RMSE of per-camera rotation errors in degrees; `rotation_rmse_deg` is kept as a compatibility alias.",
                "- `rotation_rmse_deg` is the RMSE of per-camera rotation errors in degrees.",
                "",
                "Runtime/memory:",
                "- `runtime_total_sec` sums feature extraction, exhaustive matching, and mapper stage times.",
                "- `peak_memory_mb` is the maximum observed working-set peak across those stages.",
                "- For backend/core-only comparison against cached-front-end methods, use `mapper_time_sec` and `mapper_peak_memory_mb`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(scene_csv)
    print(aggregate_csv)


if __name__ == "__main__":
    main()
