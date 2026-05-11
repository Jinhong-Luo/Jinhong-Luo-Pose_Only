#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from optuna_validation_search import resolve_validation_scenes
from validation_search import (
    build_robust_config,
    dump_json,
    load_structured_config,
    run_search,
    write_results_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def normalize_candidate_params(recommendation: Dict[str, Any]) -> Dict[str, Any]:
    params = {}
    for key, value in recommendation.items():
        if not key.startswith("param_"):
            continue
        params[key[len("param_"):]] = value
    return params


def main() -> None:
    ap = argparse.ArgumentParser(description="Run explicit validation for top-K recommended frontend candidates.")
    ap.add_argument("--recommendations_json", required=True)
    ap.add_argument("--optuna_config", required=True)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--reuse_mode", choices=["reuse", "resume", "full"], default="reuse")
    args = ap.parse_args()

    rec_doc = load_structured_config(args.recommendations_json)
    recommendations = list(rec_doc.get("recommendations") or [])
    if not recommendations:
        raise SystemExit("No recommendations found in recommendations JSON.")
    recommendations = recommendations[: max(int(args.top_k), 1)]

    cfg = load_structured_config(args.optuna_config)
    config_dir = Path(cfg["_config_dir"]).resolve()
    base_validation = deepcopy(cfg.get("base_validation_config"))
    if not isinstance(base_validation, dict):
        raise ValueError("Optuna config must contain base_validation_config")
    base_validation["_config_dir"] = str(config_dir)
    base_validation["_config_path"] = str(Path(cfg["_config_path"]).resolve())
    base_validation["scenes"] = resolve_validation_scenes(base_validation, config_dir)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate = {
        "config_path": str(Path(cfg["_config_path"]).resolve()),
        "recommendations_json": str(Path(args.recommendations_json).resolve()),
        "output_root": str(output_root),
        "criterion": base_validation.get("selection_criterion"),
        "scoring": deepcopy(base_validation.get("scoring", {})),
        "candidates": [],
    }

    for idx, recommendation in enumerate(recommendations, start=1):
        params = normalize_candidate_params(recommendation)
        candidate_cfg = deepcopy(base_validation)
        candidate_cfg["output_root"] = str(output_root / f"candidate_rank_{idx:02d}")
        candidate_cfg["search_space"] = {key: [value] for key, value in params.items()}
        args_ns = SimpleNamespace(
            output_root=str(output_root / f"candidate_rank_{idx:02d}"),
            reuse_mode=args.reuse_mode,
            criterion=None,
        )
        result = run_search(candidate_cfg, args_ns)
        result_path = output_root / f"candidate_rank_{idx:02d}" / "validation_results.json"
        result["results_json"] = str(result_path)
        dump_json(str(result_path), result)

        candidate = deepcopy((result.get("candidates") or [])[0])
        candidate["recommended_rank"] = recommendation.get("rank")
        candidate["pred_score_mean"] = recommendation.get("pred_score_mean")
        candidate["pred_score_rf"] = recommendation.get("pred_score_rf")
        candidate["pred_score_xgb"] = recommendation.get("pred_score_xgb")
        aggregate["candidates"].append(candidate)

    aggregate["candidates"].sort(key=lambda item: float(item.get("summary", {}).get("score", float("inf"))))
    if aggregate["candidates"]:
        best = aggregate["candidates"][0]
        aggregate["best_candidate_id"] = best.get("candidate_id")
        aggregate["best_summary"] = best.get("summary", {})
        aggregate["best_params"] = best.get("params", {})
    aggregate_path = output_root / "validation_results.json"
    aggregate["results_json"] = str(aggregate_path)
    dump_json(str(aggregate_path), aggregate)
    write_results_csv(str(output_root / "validation_results.csv"), aggregate["candidates"])
    dump_json(
        str(output_root / "robust_config.json"),
        build_robust_config(aggregate, aggregate.get("criterion") or "mean_plus_fail_penalty_plus_std"),
    )
    print("saved:", aggregate_path)
    print("saved:", output_root / "validation_results.csv")
    print("saved:", output_root / "robust_config.json")


if __name__ == "__main__":
    main()
