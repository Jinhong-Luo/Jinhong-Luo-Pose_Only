#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "translation_mm_median_mean",
        "rotation_deg_median_mean",
        "runtime_total_sec_mean",
        "peak_memory_mb_mean",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) for k in fieldnames} | {"label": row["label"], "source": row["source"]})


def save_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def summarize_group(group_dir: Path, label: str) -> Dict[str, Any]:
    trans: List[float] = []
    rot: List[float] = []
    runtime: List[float] = []
    mem: List[float] = []
    for scene_dir in sorted(group_dir.iterdir()) if group_dir.exists() else []:
        summary_path = scene_dir / "experiment_summary.json"
        if not summary_path.exists():
            continue
        payload = load_json(summary_path)
        t = as_float(payload.get("translation_eval_mm_median"))
        r = as_float(payload.get("rotation_eval_median_deg"))
        rt = as_float(payload.get("runtime_total_sec"))
        pm = as_float(payload.get("peak_memory_mb"))
        if t is not None:
            trans.append(t)
        if r is not None:
            rot.append(r)
        if rt is not None:
            runtime.append(rt)
        if pm is not None:
            mem.append(pm)
    return {
        "label": label,
        "translation_mm_median_mean": mean(trans) if trans else float("nan"),
        "rotation_deg_median_mean": mean(rot) if rot else float("nan"),
        "runtime_total_sec_mean": mean(runtime) if runtime else float("nan"),
        "peak_memory_mb_mean": mean(mem) if mem else float("nan"),
        "source": str(group_dir),
    }


def summarize_candidate_dir(candidate_dir: Path, label: str) -> Dict[str, Any]:
    return summarize_group(candidate_dir, label)


def main() -> None:
    out_dir = REPO_ROOT / "runs" / "paper_v2" / "ablation_paper_v2" / "tables"
    d_root = REPO_ROOT / "runs" / "paper_v2" / "ablation_paper_v2" / "degeneracy"
    r_root = REPO_ROOT / "runs" / "paper_v2" / "ablation_paper_v2" / "refinement"

    d_rows = [
        summarize_group(d_root / "D0_baseline", "D0"),
        summarize_group(d_root / "D1_geom_basepair_only", "D1"),
        summarize_group(d_root / "D2_robust_only", "D2"),
        summarize_group(d_root / "D3_full_stable_ligt", "D3"),
    ]
    save_csv(out_dir / "table_D_degeneracy_main.csv", d_rows)

    a_rows = [
        summarize_group(REPO_ROOT / "runs" / "paper_v2" / "main" / "rraa_ligt", "A0"),
        summarize_group(REPO_ROOT / "runs" / "paper_v2" / "main_phase2_5_frontend" / "rraa_ligt", "A1"),
        summarize_candidate_dir(REPO_ROOT / "runs" / "paper_v2" / "optuna_frontend_qpair_phase2_6c_vs_colmap_paircache_12scenes" / "trials" / "trial_0000" / "candidates" / "candidate_000", "A2"),
        summarize_candidate_dir(REPO_ROOT / "runs" / "paper_v2" / "recommended_top3_validation" / "candidate_rank_01" / "candidates" / "candidate_000", "A3"),
        summarize_candidate_dir(REPO_ROOT / "runs" / "paper_v2" / "recommended_top3_validation" / "candidate_rank_03" / "candidates" / "candidate_000", "A4"),
    ]
    save_csv(out_dir / "table_A_strategy_main.csv", a_rows)

    r_rows = [
        summarize_group(r_root / "R0_no_refinement", "R0"),
        summarize_group(r_root / "R1_pa_always_accept", "R1"),
        summarize_group(r_root / "R2_pa_conservative_accept", "R2"),
    ]
    save_csv(out_dir / "table_R_refinement_main.csv", r_rows)

    s_rows = [
        dict(d_rows[0], label="S0"),
        dict(d_rows[3], label="S1"),
        dict(a_rows[1], label="S2"),
        dict(r_rows[2], label="S3"),
    ]
    save_csv(out_dir / "table_S_system_main.csv", s_rows)

    save_md(
        out_dir / "README.md",
        "\n".join(
            [
                "# Ablation Paper V2 Tables",
                "",
                "Files:",
                "- `table_D_degeneracy_main.csv`",
                "- `table_A_strategy_main.csv`",
                "- `table_R_refinement_main.csv`",
                "- `table_S_system_main.csv`",
                "",
                "Notes:",
                "- `D` and `R` tables are generated from fresh no-quality runs under `runs/paper_v2/ablation_paper_v2`.",
                "- `A` table reuses existing strategy/protocol runs.",
                "- `S` table summarizes the intended paper storyline as `base -> stable backend -> strategy/frontend -> conservative PA`.",
            ]
        )
        + "\n",
    )
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
