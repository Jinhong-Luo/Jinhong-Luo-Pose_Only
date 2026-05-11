#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is not installed. Install it first, for example:\n"
        r"  .\.venv\Scripts\python.exe -m pip install matplotlib"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def maybe_int(value: Any) -> Optional[int]:
    if value in (None, "", "null"):
        return None
    try:
        return int(value)
    except Exception:
        return None


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def load_trial_curve(path: Path, label: str) -> Dict[str, Any]:
    rows = load_csv(path)
    usable: List[Tuple[int, float]] = []
    for row in rows:
        score = maybe_float(row.get("score"))
        trial_number = maybe_int(row.get("trial_number"))
        state = str(row.get("state") or "")
        if score is None or trial_number is None or "COMPLETE" not in state:
            continue
        usable.append((trial_number, score))
    usable.sort(key=lambda item: item[0])
    xs: List[int] = []
    raw_scores: List[float] = []
    best_scores: List[float] = []
    best_so_far = math.inf
    best_trial = None
    for idx, (trial_number, score) in enumerate(usable, start=1):
        xs.append(idx)
        raw_scores.append(score)
        if score < best_so_far:
            best_so_far = score
            best_trial = trial_number
        best_scores.append(best_so_far)
    return {
        "label": label,
        "csv_path": str(path),
        "study_root": str(path.parent),
        "xs": xs,
        "trial_numbers": [trial_number for trial_number, _ in usable],
        "raw_scores": raw_scores,
        "best_scores": best_scores,
        "trial_count": len(xs),
        "best_score": best_so_far if xs else None,
        "best_trial_number": best_trial,
        "first_score": raw_scores[0] if raw_scores else None,
        "last_score": raw_scores[-1] if raw_scores else None,
    }


def ema(values: List[float], alpha: float = 0.3) -> List[float]:
    if not values:
        return []
    out = [float(values[0])]
    for value in values[1:]:
        out.append(alpha * float(value) + (1.0 - alpha) * out[-1])
    return out


def rolling_mean(values: List[float], window: int = 5) -> List[float]:
    if not values:
        return []
    out: List[float] = []
    for idx in range(len(values)):
        lo = max(0, idx - window + 1)
        chunk = values[lo : idx + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def save_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_study_curves(curves: List[Dict[str, Any]], out_path: Path) -> None:
    if not curves:
        return
    fig, axes = plt.subplots(len(curves), 1, figsize=(10, 3.2 * len(curves)), squeeze=False)
    for ax, curve in zip(axes.flat, curves):
        xs = curve["xs"]
        raw_scores = curve["raw_scores"]
        best_scores = curve["best_scores"]
        ema_scores = ema(raw_scores, alpha=0.25)
        smooth_scores = rolling_mean(raw_scores, window=min(5, max(2, len(raw_scores))))
        ax.scatter(xs, raw_scores, s=18, alpha=0.75, color="#7f8c8d", label="Trial Score")
        ax.plot(xs, smooth_scores, linewidth=1.8, color="#16a085", alpha=0.9, label="Rolling Mean")
        ax.plot(xs, ema_scores, linewidth=1.8, color="#8e44ad", alpha=0.9, label="EMA")
        ax.plot(xs, best_scores, linewidth=2.2, color="#d35400", label="Best-So-Far")
        ax.set_title(f'{curve["label"]} ({curve["trial_count"]} trials)')
        ax.set_xlabel("Trial Index")
        ax.set_ylabel("Score")
        ax.grid(alpha=0.25)
        if xs:
            best_x = best_scores.index(min(best_scores)) + 1
            best_y = min(best_scores)
            ax.scatter([best_x], [best_y], s=45, color="#c0392b", zorder=3)
        ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_scene_metric_series(study_root: Path) -> Dict[str, List[Tuple[int, float]]]:
    trials_csv = study_root / "trials_summary.csv"
    rows = load_csv(trials_csv)
    usable: List[Tuple[int, Path]] = []
    for row in rows:
        trial_number = maybe_int(row.get("trial_number"))
        state = str(row.get("state") or "")
        if trial_number is None or "COMPLETE" not in state:
            continue
        results_json = study_root / "trials" / f"trial_{trial_number:04d}" / "validation_results.json"
        if results_json.exists():
            usable.append((trial_number, results_json))
    usable.sort(key=lambda item: item[0])
    series: Dict[str, List[Tuple[int, float]]] = {}
    for plot_x, (_, results_json) in enumerate(usable, start=1):
        payload = load_json(results_json)
        candidates = payload.get("candidates") or []
        if not candidates:
            continue
        scenes = (candidates[0] or {}).get("scenes") or []
        for scene in scenes:
            name = str(scene.get("scene_name") or "")
            value = maybe_float(scene.get("translation_vs_colmap_ratio"))
            if not name or value is None:
                continue
            series.setdefault(name, []).append((plot_x, value))
    return series


def plot_hard_scene_traces(curves: List[Dict[str, Any]], out_dir: Path, top_k: int = 4) -> None:
    for curve in curves:
        study_root = Path(curve["study_root"])
        series = load_scene_metric_series(study_root)
        if not series:
            continue
        ranked = sorted(
            series.items(),
            key=lambda item: sum(value for _, value in item[1]) / max(len(item[1]), 1),
            reverse=True,
        )
        selected = ranked[:top_k]
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for scene_name, values in selected:
            xs = [x for x, _ in values]
            ys = [y for _, y in values]
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label=scene_name)
        ax.set_title(f'{curve["label"]}: Hard-Scene Translation Ratio Traces')
        ax.set_xlabel("Trial Index")
        ax.set_ylabel("translation_vs_colmap_ratio")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f'hard_scene_traces_{slugify(curve["label"])}.png', dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_param_trajectory(study_root: Path, label: str, out_path: Path) -> None:
    rows = load_csv(study_root / "trials_summary.csv")
    usable = [row for row in rows if "COMPLETE" in str(row.get("state") or "")]
    if not usable:
        return
    keys = [
        "deltas_tracks",
        "min_inliers_tracks",
        "qpair_mode_tracks",
        "min_score",
        "min_inliers_map",
        "ransac_px",
    ]
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 2.3 * len(keys)), squeeze=False)
    xs = list(range(1, len(usable) + 1))
    for ax, key in zip(axes.flat, keys, strict=True):
        values = [str(row.get(key) or "") for row in usable]
        uniq = sorted(set(values))
        mapping = {value: idx for idx, value in enumerate(uniq)}
        ys = [mapping[value] for value in values]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, color="#34495e")
        ax.set_yticks(list(mapping.values()))
        ax.set_yticklabels(uniq, fontsize=8)
        ax.set_ylabel(key, fontsize=8)
        ax.grid(alpha=0.2)
    axes[-1][0].set_xlabel("Trial Index")
    fig.suptitle(f"{label}: Parameter Trajectory by Trial", y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_best(curves: List[Dict[str, Any]], out_path: Path) -> None:
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(11, 4.8))
    global_xs: List[int] = []
    global_best: List[float] = []
    offset = 0
    best_so_far = math.inf
    separators: List[int] = []
    labels: List[Tuple[float, str]] = []
    for curve in curves:
        xs = curve["xs"]
        raw_scores = curve["raw_scores"]
        for local_x, score in zip(xs, raw_scores, strict=True):
            gx = offset + local_x
            best_so_far = min(best_so_far, score)
            global_xs.append(gx)
            global_best.append(best_so_far)
        if xs:
            separators.append(offset + len(xs))
            labels.append((offset + max(1, len(xs)) / 2.0, curve["label"]))
            offset += len(xs)
    ax.plot(global_xs, global_best, linewidth=2.5, color="#2980b9")
    for sep in separators[:-1]:
        ax.axvline(sep + 0.5, color="#bdc3c7", linestyle="--", linewidth=1)
    for x_mid, label in labels:
        ax.text(x_mid, max(global_best), label, ha="center", va="bottom", fontsize=9, alpha=0.85)
    ax.set_title("Optuna Cumulative Best-So-Far Across Studies")
    ax.set_xlabel("Cumulative Trial Index")
    ax.set_ylabel("Best Score")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def extract_recommender_rows_from_csv(reco_json: Path, validation_csv: Path) -> List[Dict[str, Any]]:
    reco = load_json(reco_json)
    val_rows = load_csv(validation_csv)
    actual_by_rank: Dict[int, Dict[str, Any]] = {}
    for row in val_rows:
        candidate_id = str(row.get("candidate_id") or "")
        rank = maybe_int(candidate_id.replace("candidate_", "")) if candidate_id.startswith("candidate_") else None
        if rank is None:
            continue
        actual_by_rank[rank + 1] = {
            "actual_score": maybe_float(row.get("score")),
            "actual_mean_primary_metric": maybe_float(row.get("mean_primary_metric")),
            "actual_std_primary_metric": maybe_float(row.get("std_primary_metric")),
            "actual_worst_primary_metric": maybe_float(row.get("worst_primary_metric")),
        }
    rows: List[Dict[str, Any]] = []
    for item in reco.get("recommendations", []):
        rank = int(item["rank"])
        merged = {
            "rank": rank,
            "pred_score_rf": maybe_float(item.get("pred_score_rf")),
            "pred_score_xgb": maybe_float(item.get("pred_score_xgb")),
            "pred_score_mean": maybe_float(item.get("pred_score_mean")),
        }
        merged.update(actual_by_rank.get(rank, {}))
        rows.append(merged)
    return rows


def extract_recommender_rows_from_validation_root(reco_json: Path, validation_root: Path) -> List[Dict[str, Any]]:
    reco = load_json(reco_json)
    actual_by_rank: Dict[int, Dict[str, Any]] = {}
    for rank_dir in sorted(validation_root.glob("candidate_rank_*")):
        if not rank_dir.is_dir():
            continue
        rank = maybe_int(rank_dir.name.replace("candidate_rank_", ""))
        if rank is None:
            continue
        results_json = rank_dir / "validation_results.json"
        if not results_json.exists():
            continue
        payload = load_json(results_json)
        summary = payload.get("best_summary") or {}
        actual_by_rank[rank] = {
            "actual_score": maybe_float(summary.get("score")),
            "actual_mean_primary_metric": maybe_float(summary.get("mean_primary_metric")),
            "actual_std_primary_metric": maybe_float(summary.get("std_primary_metric")),
            "actual_worst_primary_metric": maybe_float(summary.get("worst_primary_metric")),
        }
    rows: List[Dict[str, Any]] = []
    for item in reco.get("recommendations", []):
        rank = int(item["rank"])
        merged = {
            "rank": rank,
            "pred_score_rf": maybe_float(item.get("pred_score_rf")),
            "pred_score_xgb": maybe_float(item.get("pred_score_xgb")),
            "pred_score_mean": maybe_float(item.get("pred_score_mean")),
        }
        merged.update(actual_by_rank.get(rank, {}))
        rows.append(merged)
    return rows


def plot_recommender_validation(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    ranks = [str(row["rank"]) for row in rows]
    pred = [row.get("pred_score_mean") if row.get("pred_score_mean") is not None else float("nan") for row in rows]
    actual = [row.get("actual_score") if row.get("actual_score") is not None else float("nan") for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.36
    xs = list(range(len(rows)))
    ax.bar([x - width / 2 for x in xs], pred, width=width, label="Predicted Score", color="#3498db")
    ax.bar([x + width / 2 for x in xs], actual, width=width, label="Validated Score", color="#e67e22")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Top-{r}" for r in ranks])
    ax.set_ylabel("Score")
    ax.set_title("Recommender Prediction vs Real Validation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_model_quality(model_json: Path, out_path: Path) -> None:
    payload = load_json(model_json)
    reports = payload.get("reports") or []
    if not reports:
        return
    labels = [report["model"] for report in reports]
    r2s = [maybe_float(report.get("r2")) for report in reports]
    maes = [maybe_float(report.get("mae")) for report in reports]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].bar(labels, r2s, color=["#27ae60", "#8e44ad"][: len(labels)])
    axes[0].set_title("Model R²")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, maes, color=["#16a085", "#c0392b"][: len(labels)])
    axes[1].set_title("Model MAE")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def infer_default_label(path: Path) -> str:
    return path.parent.name.replace("optuna_frontend_qpair_", "").replace("_vs_colmap_paircache_12scenes", "")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Optuna + recommender convergence/report plots.")
    ap.add_argument("--study_csv", action="append", default=[], help="One trials_summary.csv path per study.")
    ap.add_argument("--study_label", action="append", default=[], help="Optional label matching each --study_csv.")
    ap.add_argument("--recommendation_json", default=None)
    ap.add_argument("--validation_csv", default=None)
    ap.add_argument("--validation_root", default=None)
    ap.add_argument("--model_json", default=None)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    study_paths = [Path(p).resolve() for p in args.study_csv]
    labels = list(args.study_label)
    while len(labels) < len(study_paths):
        labels.append(infer_default_label(study_paths[len(labels)]))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = [load_trial_curve(path, label) for path, label in zip(study_paths, labels, strict=True)]
    summary_rows = []
    for curve in curves:
        summary_rows.append(
            {
                "label": curve["label"],
                "trial_count": curve["trial_count"],
                "first_score": curve["first_score"],
                "last_score": curve["last_score"],
                "best_score": curve["best_score"],
                "best_trial_number": curve["best_trial_number"],
                "source_csv": curve["csv_path"],
            }
        )
    if summary_rows:
        save_summary_csv(out_dir / "optuna_convergence_summary.csv", summary_rows)
        plot_study_curves(curves, out_dir / "optuna_convergence_by_study.png")
        plot_cumulative_best(curves, out_dir / "optuna_convergence_cumulative.png")
        plot_hard_scene_traces(curves, out_dir)
        for curve in curves:
            plot_param_trajectory(Path(curve["study_root"]), curve["label"], out_dir / f'param_trajectory_{slugify(curve["label"])}.png')

    if args.recommendation_json and args.validation_root:
        reco_rows = extract_recommender_rows_from_validation_root(
            Path(args.recommendation_json).resolve(),
            Path(args.validation_root).resolve(),
        )
        if reco_rows:
            save_summary_csv(out_dir / "recommender_validation_summary.csv", reco_rows)
            plot_recommender_validation(reco_rows, out_dir / "recommender_pred_vs_actual.png")
    elif args.recommendation_json and args.validation_csv:
        reco_rows = extract_recommender_rows_from_csv(
            Path(args.recommendation_json).resolve(),
            Path(args.validation_csv).resolve(),
        )
        if reco_rows:
            save_summary_csv(out_dir / "recommender_validation_summary.csv", reco_rows)
            plot_recommender_validation(reco_rows, out_dir / "recommender_pred_vs_actual.png")

    if args.model_json:
        plot_model_quality(Path(args.model_json).resolve(), out_dir / "recommender_model_quality.png")

    readme_lines = [
        "# Optuna + Recommender Report",
        "",
        "Generated files:",
        "- `optuna_convergence_by_study.png`: each Optuna study's trial score scatter + best-so-far line.",
        "- `optuna_convergence_cumulative.png`: cumulative best-so-far across studies in the given order.",
        "- `recommender_pred_vs_actual.png`: top-K recommender predicted score vs real validated score.",
        "- `recommender_model_quality.png`: RF/XGB model R2 and MAE.",
        "- `optuna_convergence_summary.csv`: numeric summary for each study.",
        "- `recommender_validation_summary.csv`: numeric summary for recommender top-K validation.",
        "",
        "Lower score is better.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
