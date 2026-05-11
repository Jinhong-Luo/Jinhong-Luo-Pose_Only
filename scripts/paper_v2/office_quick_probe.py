#!/usr/bin/env python3
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
OUT_ROOT = REPO / "runs" / "paper_v2" / "office_quick_probe"

IMAGE_GLOB = "data/raw/ETH3D/office/images/dslr_images_undistorted/*.JPG"
FEATURE_CACHE = REPO / "runs" / "ETH3D" / "office" / "frontend_cache"
K_NPY = REPO / "data" / "prepared" / "ETH3D" / "office" / "K.npy"
GT_RABS = REPO / "data" / "prepared" / "ETH3D" / "office" / "R_abs_gt_w2c.npy"
GT_POSE = REPO / "data" / "prepared" / "ETH3D" / "office" / "gt_poses_c2w.txt"
GT_CENTERS = REPO / "data" / "prepared" / "ETH3D" / "office" / "gt_centers.npy"
QUALITY_CONFIG = REPO / "configs" / "quality_config.template.json"


def run(cmd):
    print("\n[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=REPO, check=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_case(name: str, track_dir: Path, pose_dir: Path):
    track_stats = load_json(track_dir / "track_build_quality_stats.json")
    quality_stats = load_json(pose_dir / "quality_stats.json")
    degeneracy = load_json(pose_dir / "ligt_degeneracy_stats.json")
    eval_translation = load_json(pose_dir / "eval_translation.json")
    return {
        "case": name,
        "tracks_count": track_stats["track_stats"]["count"],
        "track_len_median": track_stats["track_stats"]["track_length"]["median"],
        "q_pair_median": track_stats["pair_stats"]["q_pair"]["median"],
        "positive_inlier_pairs": int(
            sum(1 for v in load_npz_pairs(track_dir)["ninliers"] if int(v) > 0)
        ),
        "track_count_for_qtrack": quality_stats["track_count"],
        "qtrack_median": quality_stats["track_quality"]["median"],
        "tracks_total_input": degeneracy["tracks_total_input"],
        "tracks_kept": degeneracy["tracks_kept"],
        "equations_kept": degeneracy["equations_kept"],
        "translation_mm_median": eval_translation["mm"]["median"],
        "translation_mm_p90": eval_translation["mm"]["p90"],
        "translation_mm_max": eval_translation["mm"]["max"],
    }


def load_npz_pairs(track_dir: Path):
    import numpy as np

    d = np.load(track_dir / "pair_quality_edges.npz")
    return {k: d[k] for k in d.files}


def build_tracks(out_dir: Path, min_inliers: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "track_build_quality_stats.json").exists():
        return
    cmd = [
        str(PYTHON),
        str(REPO / "tools" / "build_lightglue_tracks_npz.py"),
        "--image_glob", IMAGE_GLOB,
        "--out_dir", str(out_dir),
        "--deltas", "1,2,3,5",
        "--device", "auto",
        "--max_kpts", "2048",
        "--filter_th", "0.1",
        "--mutual",
        "--min_score", "0.2",
        "--use_ransac",
        "--ransac_thresh", "1.0",
        "--min_inliers", str(min_inliers),
        "--dataset", "custom",
        "--K_npy", str(K_NPY),
        "--qpair_mode", "weighted_sum",
        "--qpair_threshold", "0.0",
        "--dump_quality_stats",
        "--feature_cache_dir", str(FEATURE_CACHE),
        "--quality_config", str(QUALITY_CONFIG),
        "--auto_quality_refs",
    ]
    run(cmd)


def run_pose_only(track_dir: Path, pose_dir: Path, min_track_len: int):
    pose_dir.mkdir(parents=True, exist_ok=True)
    if (pose_dir / "eval_translation.json").exists() and (pose_dir / "quality_stats.json").exists() and (pose_dir / "ligt_degeneracy_stats.json").exists():
        return
    cmd = [
        str(PYTHON),
        str(REPO / "tools" / "Pose_Only_patched_v3_fixed.py"),
        "--track_npz_dir", str(track_dir),
        "--r_abs_npy", str(GT_RABS),
        "--dataset", "custom",
        "--K_npy", str(K_NPY),
        "--gt_pose_txt", str(GT_POSE),
        "--out_dir", str(pose_dir),
        "--min_track_len", str(min_track_len),
        "--max_tracks", "20000",
        "--u_min", "0.001",
        "--g_min", "0.001",
        "--base_pair_candidates", "80",
        "--base_pair_full_search_len", "50",
        "--irls_iters", "2",
        "--qtrack_mode", "weighted_sum",
        "--qtrack_threshold", "0.0",
        "--quality_config", str(QUALITY_CONFIG),
        "--auto_quality_refs",
        "--dump_quality_stats",
        "--dump_degeneracy_stats",
        "--enable_quality_weighting",
        "--ligt_use_qtrack_weight",
    ]
    run(cmd)

    eval_cmd = [
        str(PYTHON),
        str(REPO / "tools" / "eval_poseonly_strecha_mm.py"),
        "--est_poses", str(pose_dir / "poses_c2w.txt"),
        "--est_type", "c2w",
        "--gt_centers_npy", str(GT_CENTERS),
        "--gt_unit_to_mm", "1000",
        "--out_json", str(pose_dir / "eval_translation.json"),
    ]
    run(eval_cmd)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    cases = [
        ("frontend20_track5", 20, 5),
        ("frontend50_track3", 50, 3),
        ("frontend20_track3", 20, 3),
    ]

    baseline = {
        "case": "baseline_frontend50_track5",
        "tracks_count": 1735,
        "track_len_median": 2.0,
        "q_pair_median": 0.24011722087860113,
        "positive_inlier_pairs": 23,
        "track_count_for_qtrack": 25,
        "qtrack_median": 0.4899008883822441,
        "tracks_total_input": 25,
        "tracks_kept": 25,
        "equations_kept": 85,
        "translation_mm_median": 399.50048059464007,
        "translation_mm_p90": 718.7097175270364,
        "translation_mm_max": 1326.1686355427803,
    }
    rows = [baseline]

    for name, min_inliers, min_track_len in cases:
        case_root = OUT_ROOT / name
        track_dir = case_root / "tracks"
        pose_dir = case_root / "pose_only"
        build_tracks(track_dir, min_inliers=min_inliers)
        run_pose_only(track_dir, pose_dir, min_track_len=min_track_len)
        rows.append(summarize_case(name, track_dir, pose_dir))

    out_csv = OUT_ROOT / "summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
