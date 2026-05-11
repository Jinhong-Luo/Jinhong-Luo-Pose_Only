#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List


METRICS = [
    "failure_rate",
    "mean_primary_metric",
    "std_primary_metric",
    "worst_primary_metric",
    "mean_rotation_median_deg",
]


def as_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.inf
    return out if math.isfinite(out) else math.inf


def dominates(a: Dict[str, str], b: Dict[str, str], metrics: List[str]) -> bool:
    av = [as_float(a, k) for k in metrics]
    bv = [as_float(b, k) for k in metrics]
    return all(x <= y for x, y in zip(av, bv)) and any(x < y for x, y in zip(av, bv))


def rank_values(rows: List[Dict[str, str]], key: str) -> Dict[int, int]:
    order = sorted(range(len(rows)), key=lambda i: as_float(rows[i], key))
    ranks: Dict[int, int] = {}
    last_val = None
    last_rank = 0
    for pos, idx in enumerate(order, start=1):
        val = as_float(rows[idx], key)
        if last_val is None or val != last_val:
            last_rank = pos
            last_val = val
        ranks[idx] = last_rank
    return ranks


def simplicity_penalty(row: Dict[str, str]) -> float:
    penalty = 0.0
    if row.get("irls_iters") not in ("", "0", "1"):
        penalty += 0.5
    if row.get("min_score") not in ("", "0", "0.0"):
        penalty += 0.25
    if row.get("min_inliers_map") and "30,2:30" not in row.get("min_inliers_map", ""):
        penalty += 0.5
    if row.get("deltas_rraa") and "13" in row.get("deltas_rraa", ""):
        penalty += 0.5
    return penalty


def main() -> None:
    ap = argparse.ArgumentParser(description="Select robust Pareto candidates from optuna trials_summary.csv.")
    ap.add_argument("--trials_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    with open(args.trials_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("state", "").endswith("COMPLETE")]

    available_metrics = [m for m in METRICS if any(row.get(m, "") != "" for row in rows)]
    pareto = []
    for i, row in enumerate(rows):
        if not any(dominates(other, row, available_metrics) for j, other in enumerate(rows) if i != j):
            pareto.append((i, row))

    ranks_by_metric = {m: rank_values(rows, m) for m in available_metrics}
    selected = []
    for idx, row in pareto:
        robust_rank = sum(ranks_by_metric[m][idx] for m in available_metrics) + simplicity_penalty(row)
        out = dict(row)
        out["pareto"] = "1"
        out["robust_rank_sum"] = f"{robust_rank:.3f}"
        selected.append(out)

    selected.sort(key=lambda row: float(row["robust_rank_sum"]))
    selected = selected[: args.top_k]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in selected for k in row.keys()})
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
