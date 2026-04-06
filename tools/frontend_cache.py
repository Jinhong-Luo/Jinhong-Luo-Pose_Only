#!/usr/bin/env python3
import hashlib
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np


def read_gray_u8(path: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("L")
    return np.array(img)


def to_torch_image_u8(img_u8: np.ndarray, device):
    import torch

    t = torch.from_numpy(np.asarray(img_u8)).to(device=device)
    t = t.float() / 255.0
    return t[None, None]


def resolve_device(device_arg: str, strict_cuda: bool = False):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        if strict_cuda:
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        print("[WARN] --device cuda requested but CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_lightglue_frontend(max_kpts: int, filter_th: float, device):
    from lightglue import LightGlue, SuperPoint

    extractor = SuperPoint(max_num_keypoints=max_kpts).eval().to(device)
    matcher = LightGlue(features="superpoint", filter_threshold=filter_th).eval().to(device)
    return extractor, matcher


def _cpu_feature_dict(feats: Dict) -> Dict:
    out = {}
    for key, value in feats.items():
        if hasattr(value, "detach"):
            out[key] = value.detach().cpu()
        else:
            out[key] = value
    return out


def _move_feature_dict(feats: Dict, device) -> Dict:
    out = {}
    for key, value in feats.items():
        if hasattr(value, "to"):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _feature_cache_path(cache_dir: str, image_path: str, max_kpts: int) -> str:
    abs_path = os.path.abspath(image_path)
    digest = hashlib.sha1(f"{abs_path}|{int(max_kpts)}".encode("utf-8")).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(cache_dir, f"{stem}_{digest}.pt")


def load_or_extract_feature(
    image_path: str,
    extractor,
    device,
    max_kpts: int,
    cache_dir: Optional[str] = None,
    image_loader: Optional[Callable[[str], np.ndarray]] = None,
) -> Tuple[Dict, np.ndarray]:
    import torch

    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = _feature_cache_path(cache_dir, image_path, max_kpts)
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location="cpu")
            feats_cpu = cached["features"]
            feats = _move_feature_dict(feats_cpu, device)
            kpts = feats_cpu["keypoints"][0].detach().cpu().numpy()
            return feats, kpts

    loader = image_loader if image_loader is not None else read_gray_u8
    img_u8 = loader(image_path)
    tensor = to_torch_image_u8(img_u8, device)
    with torch.no_grad():
        feats = extractor.extract(tensor)

    kpts = feats["keypoints"][0].detach().cpu().numpy()
    if cache_path is not None:
        torch.save({"image_path": os.path.abspath(image_path), "features": _cpu_feature_dict(feats)}, cache_path)
    return feats, kpts
