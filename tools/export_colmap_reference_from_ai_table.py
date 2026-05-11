#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def normalize_group(dataset: str) -> str:
    if dataset.lower() == "strecha":
        return "strecha"
    return dataset


ALIASES = {
    "strecha::Castle-P30": ["strecha::Castle-P30-first29"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract scene-level COLMAP translation references from AI analysis CSV.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    input_csv = Path(args.input_csv).resolve()
    out_json = Path(args.out_json).resolve()
    refs: dict[str, dict[str, float | str]] = {}

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("method", "")).strip() != "COLMAP":
                continue
            dataset = str(row.get("dataset", "")).strip()
            scene = str(row.get("scene", "")).strip()
            value = row.get("translation_mm_median")
            if not dataset or not scene or value in (None, ""):
                continue
            key = f"{normalize_group(dataset)}::{scene}"
            payload = {
                "dataset": dataset,
                "scene": scene,
                "translation_mm_median": float(value),
            }
            refs[key] = payload
            for alias in ALIASES.get(key, []):
                refs[alias] = dict(payload)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(refs, f, indent=2, ensure_ascii=False)
    print("saved:", out_json)


if __name__ == "__main__":
    main()
