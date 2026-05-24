#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = REPO_ROOT / "runs" / "paper_v2" / "ablation_clean_main" / "tables"
OUT_DIR = TABLE_DIR / "paper_ready"

INPUTS = [
    ("table_A_graph_span.csv", "ablation_A_graph_span_paper"),
    ("table_T_track_span_sensitivity.csv", "ablation_T_track_span_sensitivity_paper"),
    ("table_I_irls.csv", "ablation_I_irls_paper"),
    ("table_M_main_cumulative.csv", "ablation_M_main_cumulative_paper"),
    ("table_S_staged_search_effect.csv", "ablation_S_staged_search_effect_paper"),
]

FIELDS = [
    "ID",
    "Protocol",
    "Med. Rot. err. (deg) ↓",
    "Med. Trans. err. (mm) ↓",
    "Worst Trans. err. (mm) ↓",
    "Time (s) ↓",
    "Memory (MB) ↓",
    "Score ↓",
    "Failure ↓",
]


def as_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def fmt_num(value: Any, digits: int = 2) -> str:
    num = as_float(value)
    if num is None:
        return ""
    return f"{num:.{digits}f}"


def fmt_pm(mean_value: Any, std_value: Any, digits: int = 2) -> str:
    mean_num = as_float(mean_value)
    std_num = as_float(std_value)
    if mean_num is None:
        return ""
    if std_num is None:
        return f"{mean_num:.{digits}f}"
    return f"{mean_num:.{digits}f} ± {std_num:.{digits}f}"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_md(path: Path, rows: List[Dict[str, str]], fields: Iterable[str]) -> None:
    fields = list(fields)
    with path.open("w", encoding="utf-8-sig") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row.get(field, "") for field in fields) + " |\n")


def convert_row(row: Dict[str, str]) -> Dict[str, str]:
    return {
        "ID": row.get("label", ""),
        "Protocol": row.get("title", ""),
        "Med. Rot. err. (deg) ↓": fmt_pm(row.get("rot_median_deg_mean"), row.get("rot_median_deg_std")),
        "Med. Trans. err. (mm) ↓": fmt_pm(row.get("trans_mm_median_mean"), row.get("trans_mm_median_std")),
        "Worst Trans. err. (mm) ↓": fmt_num(row.get("trans_mm_median_worst")),
        "Time (s) ↓": fmt_num(row.get("runtime_total_sec_mean")),
        "Memory (MB) ↓": fmt_num(row.get("peak_memory_mb_mean")),
        "Score ↓": fmt_num(row.get("score")),
        "Failure ↓": fmt_num(row.get("failure_count"), digits=0),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_md = OUT_DIR / "ablation_paper_ready_all.md"
    with combined_md.open("w", encoding="utf-8-sig") as combined:
        for input_name, output_stem in INPUTS:
            input_path = TABLE_DIR / input_name
            if not input_path.exists():
                print(f"skip missing: {input_path}")
                continue
            rows = [convert_row(row) for row in read_csv(input_path)]
            out_csv = OUT_DIR / f"{output_stem}.csv"
            out_md = OUT_DIR / f"{output_stem}.md"
            write_csv(out_csv, rows, FIELDS)
            write_md(out_md, rows, FIELDS)

            title = output_stem.replace("_", " ")
            combined.write(f"## {title}\n\n")
            combined.write("| " + " | ".join(FIELDS) + " |\n")
            combined.write("| " + " | ".join(["---"] * len(FIELDS)) + " |\n")
            for row in rows:
                combined.write("| " + " | ".join(row.get(field, "") for field in FIELDS) + " |\n")
            combined.write("\n")

            print(out_csv)
            print(out_md)
    print(combined_md)


if __name__ == "__main__":
    main()
