#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np


def to_numpy_cpu(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return to_numpy_cpu(x[0])
        return np.asarray([to_numpy_cpu(v) for v in x])
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.cpu().numpy()
    try:
        import torch

        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def get_pair_scores(matches_dict: Dict[str, Any], matches: np.ndarray) -> Optional[np.ndarray]:
    if matches is None or len(matches) == 0:
        return None

    if "matching_scores0" in matches_dict:
        s0 = to_numpy_cpu(matches_dict["matching_scores0"]).reshape(-1)
        if s0.shape[0] > int(matches[:, 0].max(initial=-1)):
            return s0[matches[:, 0]]

    for key in ["scores", "matching_scores", "mscores"]:
        if key in matches_dict:
            s = to_numpy_cpu(matches_dict[key]).reshape(-1)
            if s.shape[0] == matches.shape[0]:
                return s
    return None


def extract_pair_matches(matches_dict: Dict[str, Any], mutual: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    matches_t = matches_dict.get("matches", None)
    if matches_t is not None:
        matches = to_numpy_cpu(matches_t).astype(np.int64)
        if matches.ndim == 3:
            matches = matches[0]
        matches = matches.reshape(-1, 2)
        scores = get_pair_scores(matches_dict, matches)
        return matches, scores

    m0 = matches_dict.get("matches0", None)
    if m0 is None:
        return np.zeros((0, 2), dtype=np.int64), None
    m0 = to_numpy_cpu(m0).reshape(-1).astype(np.int64)
    valid = m0 >= 0

    if mutual and ("matches1" in matches_dict):
        m1 = to_numpy_cpu(matches_dict["matches1"]).reshape(-1).astype(np.int64)
        idx0_all = np.arange(m0.shape[0], dtype=np.int64)
        jj = m0.copy()
        ok = valid.copy()
        ok[valid] &= (m1[jj[valid]] == idx0_all[valid])
        valid &= ok

    idx0 = np.where(valid)[0]
    idx1 = m0[idx0]
    matches = np.stack([idx0, idx1], axis=1) if idx0.size > 0 else np.zeros((0, 2), dtype=np.int64)
    scores = get_pair_scores(matches_dict, matches)
    return matches, scores


def pair_match_cache_path(cache_dir: str, i: int, j: int) -> str:
    return os.path.join(cache_dir, f"pair_{int(i):06d}_{int(j):06d}.npz")


def pair_match_manifest_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "cache_manifest.json")


def save_pair_match_cache(
    cache_dir: str,
    i: int,
    j: int,
    matches: np.ndarray,
    pair_scores: Optional[np.ndarray] = None,
    pts0: Optional[np.ndarray] = None,
    pts1: Optional[np.ndarray] = None,
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = pair_match_cache_path(cache_dir, i, j)
    save_kwargs = {
        "matches": np.asarray(matches, dtype=np.int64).reshape(-1, 2),
        "has_scores": np.asarray([1 if pair_scores is not None else 0], dtype=np.uint8),
        "has_points": np.asarray([1 if pts0 is not None and pts1 is not None else 0], dtype=np.uint8),
    }
    if pair_scores is not None:
        save_kwargs["pair_scores"] = np.asarray(pair_scores, dtype=np.float32).reshape(-1)
    if pts0 is not None and pts1 is not None:
        save_kwargs["pts0"] = np.asarray(pts0, dtype=np.float32).reshape(-1, 2)
        save_kwargs["pts1"] = np.asarray(pts1, dtype=np.float32).reshape(-1, 2)
    np.savez_compressed(path, **save_kwargs)
    return path


def load_pair_match_cache(cache_dir: Optional[str], i: int, j: int) -> Optional[Dict[str, Any]]:
    if not cache_dir:
        return None
    path = pair_match_cache_path(cache_dir, i, j)
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as data:
        matches = np.asarray(data["matches"], dtype=np.int64).reshape(-1, 2)
        has_scores = bool(np.asarray(data["has_scores"]).reshape(-1)[0]) if "has_scores" in data else ("pair_scores" in data)
        pair_scores = np.asarray(data["pair_scores"], dtype=np.float32).reshape(-1) if has_scores and "pair_scores" in data else None
        has_points = bool(np.asarray(data["has_points"]).reshape(-1)[0]) if "has_points" in data else ("pts0" in data and "pts1" in data)
        pts0 = np.asarray(data["pts0"], dtype=np.float32).reshape(-1, 2) if has_points and "pts0" in data else None
        pts1 = np.asarray(data["pts1"], dtype=np.float32).reshape(-1, 2) if has_points and "pts1" in data else None
    return {
        "path": path,
        "matches": matches,
        "pair_scores": pair_scores,
        "pts0": pts0,
        "pts1": pts1,
    }


def save_pair_cache_manifest(cache_dir: str, payload: Dict[str, Any]) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = pair_match_manifest_path(cache_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path
