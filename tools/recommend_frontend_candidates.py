#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is not installed. Install it first, for example:\n"
        r"  .\.venv\Scripts\python.exe -m pip install scikit-learn"
    ) from exc

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_rows(path: Path) -> List[Dict[str, Any]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def maybe_float(value: Any) -> Any:
    if value is None:
        return value
    text = str(value)
    try:
        if text.strip() == "":
            return value
        return float(text)
    except ValueError:
        return value


def extract_policy_features(optuna_cfg: Dict[str, Any]) -> Dict[str, Any]:
    base_validation = optuna_cfg.get("base_validation_config") or {}
    fallbacks = base_validation.get("candidate_stage_fallbacks") or []
    fallback = fallbacks[0] if fallbacks and isinstance(fallbacks[0], dict) else {}
    override_params = fallback.get("override_params") if isinstance(fallback, dict) else {}
    override_params = override_params if isinstance(override_params, dict) else {}
    enabled = bool(fallback)
    return {
        "policy_rraa_fallback_enabled": int(enabled),
        "policy_rraa_fallback_trigger_stage": fallback.get("trigger_stage") if enabled else "none",
        "policy_rraa_fallback_rerun_from_stage": fallback.get("rerun_from_stage") if enabled else "none",
        "policy_rraa_fallback_deltas_rraa": override_params.get("deltas_rraa") if enabled else "none",
        "policy_effective_deltas_rraa_policy": (
            f"base_then_fallback:{override_params.get('deltas_rraa')}" if enabled else "base_only"
        ),
    }


def build_candidate_rows(optuna_cfg: Dict[str, Any], feature_cols: List[str]) -> List[Dict[str, Any]]:
    search_space = optuna_cfg.get("optuna_search_space") or {}
    param_names = sorted(search_space.keys())
    param_choices = []
    for name in param_names:
        spec = search_space[name]
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Search space for {name} must define non-empty choices")
        param_choices.append(choices)

    policy = extract_policy_features(optuna_cfg)
    rows: List[Dict[str, Any]] = []
    for combo in itertools.product(*param_choices):
        row: Dict[str, Any] = {}
        for idx, name in enumerate(param_names):
            row[f"param_{name}"] = combo[idx]
        row.update(policy)
        for col in feature_cols:
            row.setdefault(col, "")
        rows.append(row)
    return rows


def fit_encoders(train_rows: List[Dict[str, Any]], candidate_rows: List[Dict[str, Any]], feature_cols: List[str]) -> Dict[str, Dict[str, int]]:
    encoders: Dict[str, Dict[str, int]] = {}
    for col in feature_cols:
        values = [row.get(col, "") for row in train_rows] + [row.get(col, "") for row in candidate_rows]
        if all(isinstance(maybe_float(v), float) for v in values):
            continue
        encoders[col] = {value: idx for idx, value in enumerate(sorted(set(values)))}
    return encoders


def encode_rows(rows: List[Dict[str, Any]], feature_cols: List[str], encoders: Dict[str, Dict[str, int]]) -> List[List[float]]:
    matrix: List[List[float]] = []
    for row in rows:
        feats: List[float] = []
        for col in feature_cols:
            value = row.get(col, "")
            parsed = maybe_float(value)
            if isinstance(parsed, float):
                feats.append(parsed)
            else:
                feats.append(float(encoders[col][value]))
        matrix.append(feats)
    return matrix


def main() -> None:
    ap = argparse.ArgumentParser(description="Recommend top-K frontend parameter candidates with RF/XGB.")
    ap.add_argument("--trial_csv", required=True)
    ap.add_argument("--optuna_config", required=True)
    ap.add_argument("--target_col", default="score")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    rows = load_rows(Path(args.trial_csv).resolve())
    if not rows:
        raise SystemExit("Trial CSV is empty.")
    feature_cols = [col for col in rows[0].keys() if col.startswith("param_") or col.startswith("policy_")]
    usable = [row for row in rows if row.get(args.target_col) not in (None, "")]
    y = [float(row[args.target_col]) for row in usable]

    optuna_cfg = load_json(Path(args.optuna_config).resolve())
    candidate_rows = build_candidate_rows(optuna_cfg, feature_cols)
    encoders = fit_encoders(usable, candidate_rows, feature_cols)
    X_train = encode_rows(usable, feature_cols, encoders)
    X_candidates = encode_rows(candidate_rows, feature_cols, encoders)

    rf = RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=1)
    rf.fit(X_train, y)
    rf_pred = list(rf.predict(X_candidates))

    xgb_pred: List[float] | None = None
    if XGBRegressor is not None:
        xgb = XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        xgb.fit(X_train, y)
        xgb_pred = list(xgb.predict(X_candidates))

    recommendations = []
    for idx, row in enumerate(candidate_rows):
        rec = {
            "rank": None,
            "pred_score_rf": float(rf_pred[idx]),
            "pred_score_xgb": float(xgb_pred[idx]) if xgb_pred is not None else None,
        }
        pred_values = [rec["pred_score_rf"]]
        if rec["pred_score_xgb"] is not None:
            pred_values.append(rec["pred_score_xgb"])
        rec["pred_score_mean"] = float(sum(pred_values) / len(pred_values))
        rec.update({col: row.get(col) for col in feature_cols})
        recommendations.append(rec)

    recommendations.sort(key=lambda item: item["pred_score_mean"])
    for idx, rec in enumerate(recommendations, start=1):
        rec["rank"] = idx

    payload = {
        "trial_csv": str(Path(args.trial_csv).resolve()),
        "optuna_config": str(Path(args.optuna_config).resolve()),
        "target_col": args.target_col,
        "feature_cols": feature_cols,
        "candidate_count": len(recommendations),
        "top_k": int(args.top_k),
        "recommendations": recommendations[: max(int(args.top_k), 1)],
    }
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved:", out_json)


if __name__ == "__main__":
    main()
