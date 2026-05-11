#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import _bootstrap  # noqa: F401
from frontend_cache import build_lightglue_frontend, load_or_extract_feature, read_gray_u8, resolve_device
from pair_match_cache import extract_pair_matches, load_pair_match_cache, save_pair_cache_manifest, save_pair_match_cache


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_structured_config(path: str) -> Dict[str, Any]:
    path = Path(path).resolve()
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8-sig") as f:
        if suffix in (".yml", ".yaml"):
            import yaml

            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dict: {path}")
    data["_config_path"] = str(path)
    data["_config_dir"] = str(path.parent)
    return data


def resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    norm = path.replace("/", os.sep).replace("\\", os.sep)
    if norm.startswith(f".{os.sep}") or norm.startswith(f"..{os.sep}") or norm in (".", ".."):
        return os.path.abspath(os.path.join(base_dir, path))
    return os.path.abspath(os.path.join(str(REPO_ROOT), path))


def parse_deltas(text: str) -> List[int]:
    vals = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(int(item))
    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        raise ValueError("Empty --deltas")
    return vals


def parse_only_scenes(text: str | None) -> set[str]:
    if not text:
        return set()
    return {item.strip() for item in text.split(",") if item.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch precompute pair raw match cache for scenes listed in a search config.")
    ap.add_argument("--config", required=True, help="Phase-2 Optuna config containing base_validation_config.scenes")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--deltas", default="1,2,3,5,8")
    ap.add_argument("--max_kpts", type=int, default=2048)
    ap.add_argument("--filter_th", type=float, default=0.1)
    ap.add_argument("--mutual", action="store_true")
    ap.add_argument("--feature_cache_dir_name", default="feature_cache_optuna")
    ap.add_argument("--pair_cache_dir_name", default="pair_match_cache_optuna")
    ap.add_argument("--only_scenes", default=None, help="Comma-separated scene names from config, e.g. DTU_scan1,strecha_fountain_P11")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_structured_config(args.config)
    config_dir = cfg["_config_dir"]
    scenes = ((cfg.get("base_validation_config") or {}).get("scenes") or [])
    if not scenes:
        raise ValueError("No scenes found under base_validation_config.scenes")

    only_scenes = parse_only_scenes(args.only_scenes)
    deltas = parse_deltas(args.deltas)
    dev = resolve_device(args.device)
    print("[Device]", dev)
    extractor, matcher = build_lightglue_frontend(max_kpts=args.max_kpts, filter_th=args.filter_th, device=dev)

    import torch

    for scene_cfg in scenes:
        scene_name = str(scene_cfg["name"])
        if only_scenes and scene_name not in only_scenes:
            continue

        image_glob = resolve_path(str(scene_cfg["image_glob"]), config_dir)
        image_paths = sorted(glob.glob(image_glob))
        if not image_paths:
            raise FileNotFoundError(image_glob)

        scene_root = Path(resolve_path(str(scene_cfg["scene_root"]), config_dir))
        feature_cache_dir = scene_root / args.feature_cache_dir_name
        pair_cache_dir = scene_root / args.pair_cache_dir_name
        pair_cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Scene] {scene_name}")
        print(f"  images            : {len(image_paths)}")
        print(f"  feature_cache_dir : {feature_cache_dir}")
        print(f"  pair_cache_dir    : {pair_cache_dir}")

        feats_cache = []
        kpts_cache = []
        for img_path in image_paths:
            feats, kpts = load_or_extract_feature(
                img_path,
                extractor=extractor,
                device=dev,
                max_kpts=args.max_kpts,
                cache_dir=str(feature_cache_dir),
                image_loader=read_gray_u8,
            )
            feats_cache.append(feats)
            kpts_cache.append(np.asarray(kpts, dtype=np.float32))

        N = len(image_paths)
        created = 0
        reused = 0
        for i in range(N):
            feats0 = feats_cache[i]
            for d in deltas:
                j = i + d
                if j >= N:
                    continue
                if (not args.overwrite) and load_pair_match_cache(str(pair_cache_dir), i, j) is not None:
                    reused += 1
                    continue
                feats1 = feats_cache[j]
                with torch.no_grad():
                    out = matcher({"image0": feats0, "image1": feats1})
                matches, pair_scores = extract_pair_matches(out, mutual=args.mutual)
                pts0 = kpts_cache[i][matches[:, 0]] if matches.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
                pts1 = kpts_cache[j][matches[:, 1]] if matches.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
                save_pair_match_cache(str(pair_cache_dir), i, j, matches, pair_scores, pts0, pts1)
                created += 1

        save_pair_cache_manifest(
            str(pair_cache_dir),
            {
                "scene_name": scene_name,
                "image_glob": image_glob,
                "num_images": len(image_paths),
                "deltas": deltas,
                "max_kpts": int(args.max_kpts),
                "filter_th": float(args.filter_th),
                "mutual": bool(args.mutual),
                "feature_cache_dir": str(feature_cache_dir),
                "pair_cache_dir": str(pair_cache_dir),
            },
        )
        print(f"  created pairs     : {created}")
        print(f"  reused pairs      : {reused}")


if __name__ == "__main__":
    main()
