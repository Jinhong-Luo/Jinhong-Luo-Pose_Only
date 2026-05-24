#!/usr/bin/env python3
import json
import os
from typing import Any, Dict

import numpy as np


def safe_ratio(num: float, den: float) -> float:
    den = float(den)
    if abs(den) < 1e-12:
        return 0.0
    return float(num) / den


def summarize_array(arr, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)) -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    out = {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }
    for q in quantiles:
        out[f"q{int(round(100 * q)):02d}"] = float(np.quantile(arr, q))
    return out


def json_ready(obj: Any):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def dump_json(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, ensure_ascii=False)
