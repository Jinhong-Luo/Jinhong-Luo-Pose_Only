#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
QUALITY_CONFIG = REPO / "configs" / "quality_config.template.json"
OUT_ROOT = REPO / "runs" / "paper_v2" / "eth3d_triplet_search"

SCENES = {
    "office": {
        "raw_glob": "data/raw/ETH3D/office/images/dslr_images_undistorted/*.JPG",
        "prepared": REPO / "data" / "prepared" / "ETH3D" / "office",
        "frontend_cache": REPO / "runs" / "ETH3D" / "office" / "frontend_cache",
        "track_mode": "K",
        "baseline_eval": REPO / "runs" / "paper_v2" / "main" / "gt_ligt" / "ETH3D_office" / "pose_only" / "eval_translation.json",
    },
    "courtyard": {
        "raw_glob": "data/raw/ETH3D/courtyard/images/dslr_images_undistorted/*.JPG",
        "prepared": REPO / "data" / "prepared" / "ETH3D" / "courtyard",
        "frontend_cache": REPO / "runs" / "ETH3D_repaired" / "courtyard" / "frontend_cache",
        "track_mode": "Ks",
        "baseline_eval": REPO / "runs" / "paper_v2" / "main" / "gt_ligt" / "ETH3D_courtyard" / "pose_only" / "eval_translation.json",
        "existing_track_dir": REPO / "runs" / "paper_v2" / "courtyard_probe" / "frontend20_d1235813" / "tracks",
        "existing_base_eval": REPO / "runs" / "paper_v2" / "courtyard_probe" / "frontend20_d1235813" / "pose_only" / "eval_translation.json",
    },
    "terrace": {
        "raw_glob": "data/raw/ETH3D/terrace/images/dslr_images_undistorted/*.JPG",
        "prepared": REPO / "data" / "prepared" / "ETH3D" / "terrace",
        "frontend_cache": REPO / "runs" / "ETH3D_repaired" / "terrace" / "frontend_cache",
        "track_mode": "Ks",
        "baseline_eval": REPO / "runs" / "paper_v2" / "main" / "gt_ligt" / "ETH3D_terrace" / "pose_only" / "eval_translation.json",
        "existing_track_dir": REPO / "runs" / "paper_v2" / "terrace_probe" / "frontend20_d1235813" / "tracks",
        "existing_base_eval": REPO / "runs" / "paper_v2" / "terrace_probe" / "frontend20_d1235813" / "pose_only" / "eval_translation.json",
        "existing_gap3_eval": REPO / "runs" / "paper_v2" / "terrace_probe" / "frontend20_d1235813_gap3" / "pose_only" / "eval_translation.json",
    },
}


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO, check=True)


def mm_median(eval_json: Path) -> float:
    obj = json.loads(eval_json.read_text(encoding="utf-8"))
    return float(obj["mm"]["median"])


def build_tracks(scene: str, out_dir: Path) -> None:
    if (out_dir / "track_build_quality_stats.json").exists():
        return
    cfg = SCENES[scene]
    prepared = cfg["prepared"]
    cmd = [
        str(PYTHON),
        str(REPO / "tools" / "build_lightglue_tracks_npz.py"),
        "--image_glob", cfg["raw_glob"],
        "--out_dir", str(out_dir),
        "--deltas", "1,2,3,5,8,13",
        "--device", "auto",
        "--max_kpts", "2048",
        "--filter_th", "0.1",
        "--mutual",
        "--min_score", "0.2",
        "--use_ransac",
        "--ransac_thresh", "1.0",
        "--min_inliers", "20",
        "--dataset", "custom",
        "--qpair_mode", "weighted_sum",
        "--qpair_threshold", "0.0",
        "--dump_quality_stats",
        "--feature_cache_dir", str(cfg["frontend_cache"]),
        "--quality_config", str(QUALITY_CONFIG),
        "--auto_quality_refs",
    ]
    if cfg["track_mode"] == "K":
        cmd += ["--K_npy", str(prepared / "K.npy")]
    else:
        cmd += ["--Ks_npy", str(prepared / "Ks.npy"), "--image_K_idx_npy", str(prepared / "image_K_idx.npy")]
    run(cmd)


def run_pose(scene: str, track_dir: Path, out_dir: Path, *, base_gap: int = 0, qtrack_threshold: float = 0.0) -> None:
    if (out_dir / "eval_translation.json").exists():
        return
    prepared = SCENES[scene]["prepared"]
    cmd = [
        str(PYTHON),
        str(REPO / "tools" / "Pose_Only_patched_v3_fixed.py"),
        "--track_npz_dir", str(track_dir),
        "--r_abs_npy", str(prepared / "R_abs_gt_w2c.npy"),
        "--dataset", "custom",
        "--gt_pose_txt", str(prepared / "gt_poses_c2w.txt"),
        "--out_dir", str(out_dir),
        "--min_track_len", "5",
        "--max_tracks", "20000",
        "--u_min", "0.001",
        "--g_min", "0.001",
        "--base_pair_candidates", "80",
        "--base_pair_full_search_len", "50",
        "--base_pair_min_gap", str(base_gap),
        "--irls_iters", "2",
        "--qtrack_mode", "weighted_sum",
        "--qtrack_threshold", str(qtrack_threshold),
        "--quality_config", str(QUALITY_CONFIG),
        "--auto_quality_refs",
        "--dump_quality_stats",
        "--dump_degeneracy_stats",
        "--enable_quality_weighting",
        "--ligt_use_qtrack_weight",
    ]
    if SCENES[scene]["track_mode"] == "K":
        cmd += ["--K_npy", str(prepared / "K.npy")]
    else:
        cmd += ["--Ks_npy", str(prepared / "Ks.npy"), "--image_K_idx_npy", str(prepared / "image_K_idx.npy")]
    run(cmd)
    run([
        str(PYTHON),
        str(REPO / "tools" / "eval_poseonly_strecha_mm.py"),
        "--est_poses", str(out_dir / "poses_c2w.txt"),
        "--est_type", "c2w",
        "--gt_centers_npy", str(prepared / "gt_centers.npy"),
        "--gt_unit_to_mm", "1000",
        "--out_json", str(out_dir / "eval_translation.json"),
    ])


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []

    for scene, cfg in SCENES.items():
        rows.append({"scene": scene, "config": "baseline", "translation_mm_median": mm_median(cfg["baseline_eval"])})

        if "existing_track_dir" in cfg:
            track_dir = cfg["existing_track_dir"]
        else:
            track_dir = OUT_ROOT / scene / "frontend20_d1235813" / "tracks"
            build_tracks(scene, track_dir)

        if "existing_base_eval" in cfg:
            rows.append({"scene": scene, "config": "relaxed_ext", "translation_mm_median": mm_median(cfg["existing_base_eval"])})
        else:
            out_dir = track_dir.parent / "pose_only"
            run_pose(scene, track_dir, out_dir, base_gap=0, qtrack_threshold=0.0)
            rows.append({"scene": scene, "config": "relaxed_ext", "translation_mm_median": mm_median(out_dir / "eval_translation.json")})

        if "existing_gap3_eval" in cfg:
            rows.append({"scene": scene, "config": "relaxed_ext_gap3", "translation_mm_median": mm_median(cfg["existing_gap3_eval"])})
        else:
            out_dir = track_dir.parent.parent / f"{track_dir.parent.name}_gap3" / "pose_only"
            run_pose(scene, track_dir, out_dir, base_gap=3, qtrack_threshold=0.0)
            rows.append({"scene": scene, "config": "relaxed_ext_gap3", "translation_mm_median": mm_median(out_dir / "eval_translation.json")})

        out_dir = track_dir.parent.parent / f"{track_dir.parent.name}_gap3_q055" / "pose_only"
        run_pose(scene, track_dir, out_dir, base_gap=3, qtrack_threshold=0.55)
        rows.append({"scene": scene, "config": "relaxed_ext_gap3_q055", "translation_mm_median": mm_median(out_dir / "eval_translation.json")})

    out_csv = OUT_ROOT / "summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "config", "translation_mm_median"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
