#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is not installed. Install it first, for example:\n"
        r"  .\.venv\Scripts\python.exe -m pip install scikit-learn pandas"
    ) from exc

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


def load_rows(path: Path) -> List[Dict[str, Any]]:
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


def encode_categorical(rows: List[Dict[str, Any]], feature_cols: List[str]) -> Tuple[List[List[float]], Dict[str, Dict[str, int]]]:
    encoders: Dict[str, Dict[str, int]] = {}
    matrix: List[List[float]] = []
    for col in feature_cols:
        values = [row[col] for row in rows]
        if all(isinstance(maybe_float(v), float) for v in values):
            continue
        encoders[col] = {value: idx for idx, value in enumerate(sorted(set(values)))}

    for row in rows:
        feats: List[float] = []
        for col in feature_cols:
            value = row[col]
            parsed = maybe_float(value)
            if isinstance(parsed, float):
                feats.append(parsed)
            else:
                feats.append(float(encoders[col][value]))
        matrix.append(feats)
    return matrix, encoders


def evaluate(model_name: str, model: Any, X_train: List[List[float]], X_test: List[List[float]], y_train: List[float], y_test: List[float]) -> Dict[str, Any]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "model": model_name,
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(mean_squared_error(y_test, pred) ** 0.5),
        "r2": float(r2_score(y_test, pred)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train RandomForest / XGBoost recommenders on merged Optuna trial tables.")
    ap.add_argument("--trial_csv", required=True)
    ap.add_argument("--target_col", default="score")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    rows = load_rows(Path(args.trial_csv).resolve())
    feature_cols = [
        col
        for col in rows[0].keys()
        if col.startswith("param_") or col.startswith("policy_")
    ]
    usable = [row for row in rows if row.get(args.target_col) not in (None, "")]
    X, encoders = encode_categorical(usable, feature_cols)
    y = [float(row[args.target_col]) for row in usable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    reports: List[Dict[str, Any]] = []
    rf = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=1)
    reports.append(evaluate("RandomForestRegressor", rf, X_train, X_test, y_train, y_test))

    if XGBRegressor is not None:
        xgb = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        reports.append(evaluate("XGBRegressor", xgb, X_train, X_test, y_train, y_test))

    feature_importance = []
    for name, importance in sorted(zip(feature_cols, rf.feature_importances_), key=lambda item: item[1], reverse=True):
        feature_importance.append({"feature": name, "importance": float(importance)})

    payload = {
        "trial_csv": str(Path(args.trial_csv).resolve()),
        "target_col": args.target_col,
        "row_count": len(usable),
        "feature_cols": feature_cols,
        "categorical_encoders": encoders,
        "reports": reports,
        "random_forest_feature_importance": feature_importance,
    }
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved:", out_json)


if __name__ == "__main__":
    main()
