#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
STRECHA_CSV = REPO_ROOT / "runs" / "paper_v2" / "colmap_compare_strecha6" / "colmap_strecha6_summary.csv"
ETH3D_CSV = REPO_ROOT / "runs" / "paper_v2" / "colmap_compare_eth3d5" / "colmap_eth3d5_summary.csv"
OURS_ETH3D_CSV = REPO_ROOT / "runs" / "paper_v2" / "eval_eth3d5" / "paper_summary_main.csv"
OUT_DIR = REPO_ROOT / "runs" / "paper_v2" / "colmap_report"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(v: str) -> float:
    return float(v) if v not in ("", None) else float("nan")


def save_bar(rows, x_key, y_key, title, ylabel, out_path, color="#3b82f6"):
    scenes = [r[x_key] for r in rows]
    vals = [to_float(r[y_key]) for r in rows]
    plt.figure(figsize=(10, 4.8))
    plt.bar(scenes, vals, color=color, edgecolor="black", linewidth=0.8)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def build_eth3d_best_ours(rows):
    best = {}
    for r in rows:
        if r["dataset"] != "ETH3D" or r["status"] != "ok":
            continue
        scene = r["scene"]
        val = to_float(r["translation_mm_median"])
        if scene not in best or val < best[scene]["translation_mm_median"]:
            best[scene] = {
                "scene": scene,
                "experiment_group": r["experiment_group"],
                "rotation_median_deg": to_float(r["rotation_median_deg"]),
                "translation_mm_median": val,
            }
    return [best[k] for k in sorted(best.keys())]


def save_eth3d_compare(colmap_rows, ours_rows, out_csv, out_png):
    ours_by_scene = {r["scene"]: r for r in ours_rows}
    merged = []
    for c in colmap_rows:
        scene = c["scene"]
        o = ours_by_scene.get(scene)
        merged.append({
            "scene": scene,
            "colmap_translation_mm_median": to_float(c["translation_mm_median"]),
            "ours_best_translation_mm_median": o["translation_mm_median"] if o else float("nan"),
            "ours_best_experiment": o["experiment_group"] if o else "",
        })

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        writer.writeheader()
        writer.writerows(merged)

    scenes = [r["scene"] for r in merged]
    colmap_vals = [r["colmap_translation_mm_median"] for r in merged]
    ours_vals = [r["ours_best_translation_mm_median"] for r in merged]
    x = range(len(scenes))
    width = 0.36
    plt.figure(figsize=(9.8, 4.8))
    plt.bar([i - width / 2 for i in x], colmap_vals, width=width, label="COLMAP", color="#2563eb", edgecolor="black", linewidth=0.8)
    plt.bar([i + width / 2 for i in x], ours_vals, width=width, label="Ours best current", color="#f97316", edgecolor="black", linewidth=0.8)
    plt.title("ETH3D: COLMAP vs Best Current Ours (Translation Median)")
    plt.ylabel("translation_mm_median")
    plt.xticks(list(x), scenes, rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.savefig(out_png.with_suffix(".pdf"))
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    strecha = read_csv(STRECHA_CSV)
    eth3d = read_csv(ETH3D_CSV)
    ours_eth3d = read_csv(OURS_ETH3D_CSV)

    save_bar(
        strecha,
        "scene",
        "translation_mm_median",
        "COLMAP on Strecha6: Translation Median",
        "translation_mm_median",
        OUT_DIR / "01_strecha6_colmap_translation.png",
        color="#2563eb",
    )
    save_bar(
        strecha,
        "scene",
        "rotation_median_deg",
        "COLMAP on Strecha6: Rotation Median",
        "rotation_median_deg",
        OUT_DIR / "02_strecha6_colmap_rotation.png",
        color="#60a5fa",
    )
    save_bar(
        eth3d,
        "scene",
        "translation_mm_median",
        "COLMAP on ETH3D5: Translation Median",
        "translation_mm_median",
        OUT_DIR / "03_eth3d5_colmap_translation.png",
        color="#1d4ed8",
    )
    save_bar(
        eth3d,
        "scene",
        "rotation_median_deg",
        "COLMAP on ETH3D5: Rotation Median",
        "rotation_median_deg",
        OUT_DIR / "04_eth3d5_colmap_rotation.png",
        color="#93c5fd",
    )
    save_eth3d_compare(
        eth3d,
        build_eth3d_best_ours(ours_eth3d),
        OUT_DIR / "eth3d5_colmap_vs_ours_best.csv",
        OUT_DIR / "05_eth3d5_colmap_vs_ours_best_translation.png",
    )


if __name__ == "__main__":
    main()
