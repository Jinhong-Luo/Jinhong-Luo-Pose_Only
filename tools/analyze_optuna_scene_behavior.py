#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize one scene's behavior across Optuna validation trials.")
    ap.add_argument("--study_root", required=True, help="Optuna study output root containing trials_summary.csv and trials/")
    ap.add_argument("--scene_name", required=True, help="Scene name as stored in validation_results.json, e.g. DTU_scan106")
    ap.add_argument("--out_csv", default=None, help="Optional CSV path for the per-trial scene summary.")
    args = ap.parse_args()

    study_root = Path(args.study_root).resolve()
    trials_csv = study_root / "trials_summary.csv"
    if not trials_csv.exists():
        raise SystemExit(f"Missing trials_summary.csv under study root: {study_root}")

    rows: List[Dict[str, Any]] = []
    with trials_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            trial_number = int(row["trial_number"])
            results_json = study_root / "trials" / f"trial_{trial_number:04d}" / "validation_results.json"
            if not results_json.exists():
                continue
            results = load_json(results_json)
            candidates = results.get("candidates", [])
            if not candidates:
                continue
            candidate = candidates[0]
            scene = next((item for item in candidate.get("scenes", []) if item.get("scene_name") == args.scene_name), None)
            if scene is None:
                continue
            rows.append(
                {
                    "trial_number": trial_number,
                    "score": row.get("score"),
                    "worst_primary_metric": row.get("worst_primary_metric"),
                    "deltas_tracks": row.get("deltas_tracks"),
                    "deltas_rraa": row.get("deltas_rraa"),
                    "min_inliers_tracks": row.get("min_inliers_tracks"),
                    "min_score": row.get("min_score"),
                    "qpair_mode_tracks": row.get("qpair_mode_tracks"),
                    "qpair_mode_rraa": row.get("qpair_mode_rraa"),
                    "ransac_px": row.get("ransac_px"),
                    "min_inliers_map": row.get("min_inliers_map"),
                    "scene_primary_metric": scene.get(results.get("scoring", {}).get("primary_metric", "translation_vs_colmap_ratio")),
                    "scene_translation_mm_median": scene.get("translation_mm_median"),
                    "scene_rotation_median_deg": scene.get("rotation_median_deg"),
                    "scene_status": scene.get("status"),
                }
            )

    if not rows:
        raise SystemExit(f"No rows found for scene '{args.scene_name}' in {study_root}")

    def sort_key(item: Dict[str, Any]) -> Tuple[int, float]:
        value = item.get("scene_primary_metric")
        if value in (None, ""):
            return (1, float("inf"))
        return (0, float(value))

    rows.sort(key=sort_key)
    for item in rows[:10]:
        print(json.dumps(item, ensure_ascii=False))
    print("--- worst ---")
    for item in rows[-10:]:
        print(json.dumps(item, ensure_ascii=False))

    if args.out_csv:
        out_csv = Path(args.out_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("saved:", out_csv)


if __name__ == "__main__":
    main()
