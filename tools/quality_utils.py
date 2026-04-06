#!/usr/bin/env python3
import json
import math
import os
from typing import Dict, Optional

import numpy as np


def clip01(x):
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


def safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return v


def logistic_like_ratio(x: float, ref: float) -> float:
    ref = max(float(ref), 1e-12)
    x = max(float(x), 0.0)
    return x / (x + ref)


def normalized_log_count(x: float, ref: float) -> float:
    x = max(float(x), 0.0)
    ref = max(float(ref), 1.0)
    return float(np.clip(np.log1p(x) / np.log1p(ref), 0.0, 1.0))


def gap_decay(delta: float, alpha: float) -> float:
    delta = max(float(delta), 1.0)
    alpha = max(float(alpha), 0.0)
    return 1.0 / (1.0 + alpha * (delta - 1.0))


def method_factor(method_flag: Optional[float], homography_factor: float = 0.7) -> float:
    if method_flag is None:
        return 1.0
    method_flag = int(method_flag)
    if method_flag == 1:
        return float(homography_factor)
    if method_flag < 0:
        return float(homography_factor)
    return 1.0


def weighted_average_score(components: Dict[str, float], weights: Dict[str, float]) -> float:
    denom = 0.0
    numer = 0.0
    for key, value in components.items():
        w = max(float(weights.get(key, 0.0)), 0.0)
        denom += w
        numer += w * float(np.clip(value, 0.0, 1.0))
    if denom <= 0:
        return 1.0
    return float(np.clip(numer / denom, 0.0, 1.0))


def log_additive_score(
    components: Dict[str, float],
    weights: Dict[str, float],
    *,
    eps: float = 1e-6,
) -> float:
    denom = 0.0
    numer = 0.0
    for key, value in components.items():
        w = max(float(weights.get(key, 0.0)), 0.0)
        if w <= 0.0:
            continue
        v = float(np.clip(value, eps, 1.0))
        numer += w * np.log(v)
        denom += w
    if denom <= 0.0:
        return 1.0
    return float(np.clip(np.exp(numer / denom), 0.0, 1.0))


DEFAULT_QPAIR_WEIGHTS = {
    "inlier_ratio": 0.35,
    "ninliers": 0.30,
    "score": 0.20,
    "gap": 0.10,
    "method": 0.05,
}

DEFAULT_QTRACK_WEIGHTS = {
    "length": 0.20,
    "pair_mean": 0.30,
    "pair_min": 0.15,
    "base_u": 0.20,
    "g_stat": 0.15,
}

DEFAULT_QPAIR_CONFIG = {
    "weights": DEFAULT_QPAIR_WEIGHTS,
    "inlier_ratio_ref": 0.6,
    "ninliers_ref": 80.0,
    "gap_alpha": 0.25,
    "homography_factor": 0.7,
    "score_default": 0.5,
}

DEFAULT_QTRACK_CONFIG = {
    "weights": DEFAULT_QTRACK_WEIGHTS,
    "track_len_ref": 6.0,
    "base_u_ref": 0.02,
    "g_ref": 0.02,
}


def deep_update(base: Dict, updates: Optional[Dict]) -> Dict:
    out = dict(base)
    if not updates:
        return out
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_quality_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def make_quality_config_record(
    *,
    section: str,
    raw_payload: Optional[Dict] = None,
    resolved_section: Optional[Dict] = None,
    effective_section: Optional[Dict] = None,
    mode: Optional[str] = None,
    auto_quality_refs: bool = False,
    quality_config_path: Optional[str] = None,
    stage: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    payload = raw_payload or {}
    record = {
        "stage": stage,
        "section": section,
        "quality_config_path": os.path.abspath(quality_config_path) if quality_config_path else None,
        "auto_quality_refs": bool(auto_quality_refs),
        "input_quality_payload": _json_safe(payload),
        "input_section_raw": _json_safe(payload.get(section, {})) if isinstance(payload, dict) else {},
        "resolved_section_before_auto": _json_safe(resolved_section or {}),
        "effective_section": _json_safe(effective_section or {}),
        "mode": mode,
        "source": (
            "auto_inferred_from_scene_stats"
            if auto_quality_refs
            else ("manual_config_or_defaults" if payload else "defaults_only")
        ),
    }
    if extra:
        record.update(_json_safe(extra))
    return record


def save_quality_config_record(path: str, record: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(record), f, indent=2, ensure_ascii=False)


def resolve_quality_section(config_payload: Optional[Dict], section: str, defaults: Dict) -> Dict:
    base = dict(defaults)
    if not config_payload:
        return base
    return deep_update(base, config_payload.get(section, {}))


def _finite_values(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _robust_ref(values, default: float, quantile: float = 0.5, min_value: float = 1e-6) -> float:
    arr = _finite_values(values)
    if arr.size == 0:
        return float(default)
    return float(max(np.quantile(arr, quantile), min_value))


def infer_qpair_config(
    ninliers,
    inlier_ratio,
    score_values=None,
    base_config: Optional[Dict] = None,
) -> Dict:
    cfg = dict(DEFAULT_QPAIR_CONFIG)
    if base_config:
        cfg = deep_update(cfg, base_config)
    score_arr = _finite_values(score_values if score_values is not None else [])
    cfg["ninliers_ref"] = _robust_ref(ninliers, cfg["ninliers_ref"], quantile=0.5, min_value=1.0)
    cfg["inlier_ratio_ref"] = float(np.clip(_robust_ref(inlier_ratio, cfg["inlier_ratio_ref"], quantile=0.5, min_value=1e-3), 1e-3, 1.0))
    if score_arr.size > 0:
        cfg["score_default"] = float(np.clip(np.median(score_arr), 0.0, 1.0))
    return cfg


def infer_qtrack_config(
    track_len,
    base_u,
    g_stat,
    base_config: Optional[Dict] = None,
) -> Dict:
    cfg = dict(DEFAULT_QTRACK_CONFIG)
    if base_config:
        cfg = deep_update(cfg, base_config)
    cfg["track_len_ref"] = _robust_ref(track_len, cfg["track_len_ref"], quantile=0.5, min_value=2.0)
    cfg["base_u_ref"] = _robust_ref(base_u, cfg["base_u_ref"], quantile=0.5, min_value=1e-6)
    cfg["g_ref"] = _robust_ref(g_stat, cfg["g_ref"], quantile=0.5, min_value=1e-6)
    return cfg


def compute_qpair_components(
    ninliers: float,
    nmatches: float,
    inlier_ratio: Optional[float] = None,
    score_mean: Optional[float] = None,
    score_med: Optional[float] = None,
    delta: float = 1.0,
    method: Optional[float] = None,
    config: Optional[Dict] = None,
) -> Dict[str, float]:
    cfg = dict(DEFAULT_QPAIR_CONFIG)
    if config:
        cfg.update(config)

    if inlier_ratio is None:
        inlier_ratio = safe_float(ninliers, 0.0) / max(safe_float(nmatches, 1.0), 1.0)

    score_candidates = [safe_float(score_mean, np.nan), safe_float(score_med, np.nan)]
    score_candidates = [v for v in score_candidates if np.isfinite(v)]
    score_value = float(np.mean(score_candidates)) if score_candidates else float(cfg["score_default"])

    return {
        "inlier_ratio": logistic_like_ratio(inlier_ratio, cfg["inlier_ratio_ref"]),
        "ninliers": normalized_log_count(ninliers, cfg["ninliers_ref"]),
        "score": float(np.clip(score_value, 0.0, 1.0)),
        "gap": gap_decay(delta, cfg["gap_alpha"]),
        "method": method_factor(method, cfg["homography_factor"]),
    }


def compute_qpair(
    ninliers: float,
    nmatches: float,
    inlier_ratio: Optional[float] = None,
    score_mean: Optional[float] = None,
    score_med: Optional[float] = None,
    delta: float = 1.0,
    method: Optional[float] = None,
    mode: str = "weighted_sum",
    config: Optional[Dict] = None,
) -> float:
    cfg = dict(DEFAULT_QPAIR_CONFIG)
    if config:
        cfg.update(config)
    components = compute_qpair_components(
        ninliers=ninliers,
        nmatches=nmatches,
        inlier_ratio=inlier_ratio,
        score_mean=score_mean,
        score_med=score_med,
        delta=delta,
        method=method,
        config=cfg,
    )
    if mode == "product":
        out = 1.0
        for value in components.values():
            out *= float(np.clip(value, 0.0, 1.0))
        return float(np.clip(out, 0.0, 1.0))
    if mode == "log_additive":
        return log_additive_score(components, cfg["weights"])
    if mode != "weighted_sum":
        raise ValueError(f"Unknown q_pair mode: {mode}")
    return weighted_average_score(components, cfg["weights"])


def compute_qtrack_components(
    track_len: float,
    pair_q_mean: float,
    pair_q_min: float,
    base_u: float,
    g_stat: float,
    config: Optional[Dict] = None,
) -> Dict[str, float]:
    cfg = dict(DEFAULT_QTRACK_CONFIG)
    if config:
        cfg.update(config)
    return {
        "length": logistic_like_ratio(track_len, cfg["track_len_ref"]),
        "pair_mean": float(np.clip(pair_q_mean, 0.0, 1.0)),
        "pair_min": float(np.clip(pair_q_min, 0.0, 1.0)),
        "base_u": logistic_like_ratio(base_u, cfg["base_u_ref"]),
        "g_stat": logistic_like_ratio(g_stat, cfg["g_ref"]),
    }


def compute_qtrack(
    track_len: float,
    pair_q_mean: float,
    pair_q_min: float,
    base_u: float,
    g_stat: float,
    mode: str = "weighted_sum",
    config: Optional[Dict] = None,
) -> float:
    cfg = dict(DEFAULT_QTRACK_CONFIG)
    if config:
        cfg.update(config)
    components = compute_qtrack_components(
        track_len=track_len,
        pair_q_mean=pair_q_mean,
        pair_q_min=pair_q_min,
        base_u=base_u,
        g_stat=g_stat,
        config=cfg,
    )
    if mode == "product":
        out = 1.0
        for value in components.values():
            out *= float(np.clip(value, 0.0, 1.0))
        return float(np.clip(out, 0.0, 1.0))
    if mode == "log_additive":
        return log_additive_score(components, cfg["weights"])
    if mode != "weighted_sum":
        raise ValueError(f"Unknown q_track mode: {mode}")
    return weighted_average_score(components, cfg["weights"])
