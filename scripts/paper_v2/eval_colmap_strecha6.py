#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
COLMAP_ROOT = REPO_ROOT / "COLMAP"
OUT_ROOT = REPO_ROOT / "runs" / "paper_v2" / "colmap_compare_strecha6"

SCENES = [
    {
        "scene": "fountain-P11",
        "colmap_dir": COLMAP_ROOT / "fountain-P11",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "strecha" / "fountain-P11",
        "image_list": REPO_ROOT / "data" / "prepared" / "strecha" / "fountain-P11" / "image_list.txt",
    },
    {
        "scene": "entry-P10",
        "colmap_dir": COLMAP_ROOT / "entry-P10",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "strecha" / "entry-P10",
        "image_list": REPO_ROOT / "data" / "prepared" / "strecha" / "entry-P10" / "image_list.txt",
    },
    {
        "scene": "Herz-Jesus-P8",
        "colmap_dir": COLMAP_ROOT / "Herz-Jesus-P8",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "strecha" / "Herz-Jesus-P8",
        "image_list": REPO_ROOT / "data" / "prepared" / "strecha" / "Herz-Jesus-P8" / "image_list.txt",
    },
    {
        "scene": "Herz-Jesus-P25",
        "colmap_dir": COLMAP_ROOT / "Herz-Jesus-P25",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "strecha" / "Herz-Jesus-P25",
        "image_list": REPO_ROOT / "data" / "prepared" / "strecha" / "Herz-Jesus-P25" / "image_list.txt",
    },
    {
        "scene": "Castle-P19",
        "colmap_dir": COLMAP_ROOT / "castle-P19",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "strecha" / "Castle-P19",
        "image_list": REPO_ROOT / "data" / "prepared" / "strecha" / "Castle-P19" / "image_list.txt",
    },
    {
        "scene": "Castle-P30",
        "colmap_dir": COLMAP_ROOT / "castle-P30",
        "prepared_dir": REPO_ROOT / "data" / "prepared" / "strecha" / "Castle-P30-first29",
        "image_list": REPO_ROOT / "data" / "prepared" / "strecha" / "Castle-P30-first29" / "image_list.txt",
    },
]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rotation_best_median(eval_json: Path) -> float:
    obj = json.loads(eval_json.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return min(float(item["median_deg"]) for item in obj)
    return float(obj.get("best_median_deg", obj["median_deg"]))


def main() -> None:
    ensure_dir(OUT_ROOT)
    rows = []

    for item in SCENES:
        scene = item["scene"]
        out_dir = OUT_ROOT / scene
        ensure_dir(out_dir)

        poses_txt = out_dir / "poses_c2w_colmap.txt"
        names_txt = out_dir / "image_names_used.txt"
        rabs_npy = out_dir / "R_abs_colmap_w2c.npy"
        eval_rot = out_dir / "eval_rotation_colmap.json"
        eval_trans = out_dir / "eval_translation_colmap.json"

        run([
            str(PYTHON),
            "tools\\colmap_export_poses.py",
            "--images_bin", str(item["colmap_dir"] / "images.bin"),
            "--image_list", str(item["image_list"]),
            "--out_txt", str(poses_txt),
            "--out_names", str(names_txt),
            "--out_rabs_w2c_npy", str(rabs_npy),
        ])
        run([
            str(PYTHON),
            "tools\\eval_poseonly_strecha_mm.py",
            "--est_poses", str(poses_txt),
            "--est_type", "c2w",
            "--gt_centers_npy", str(item["prepared_dir"] / "gt_centers.npy"),
            "--gt_unit_to_mm", "1000",
            "--out_json", str(eval_trans),
        ])
        run([
            str(PYTHON),
            "tools\\eval_rraa_rotation.py",
            "--est_npy", str(rabs_npy),
            "--gt_npy", str(item["prepared_dir"] / "R_abs_gt_w2c.npy"),
            "--out_json", str(eval_rot),
        ])

        trans = json.loads(eval_trans.read_text(encoding="utf-8"))
        rows.append({
            "scene": scene,
            "rotation_median_deg": rotation_best_median(eval_rot),
            "translation_mm_median": float(trans["mm"]["median"]),
            "translation_mm_rmse": float(trans["mm"]["rmse"]),
            "translation_mm_p90": float(trans["mm"]["p90"]),
        })

    csv_path = OUT_ROOT / "colmap_strecha6_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved COLMAP strecha6 summary to: {csv_path}")


if __name__ == "__main__":
    main()
