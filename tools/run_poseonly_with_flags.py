#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Small wrapper to forward bool flags to Pose_Only_patched_v3_fixed.py")
    ap.add_argument("--enable_quality_weighting_bool", required=True)
    ap.add_argument("--ligt_use_qtrack_weight_bool", required=True)
    ap.add_argument("--prepared_scene_dir_autok", default=None)
    args, rest = ap.parse_known_args()

    cmd = [sys.executable, str(REPO_ROOT / "tools" / "Pose_Only_patched_v3_fixed.py")] + rest
    if parse_bool(args.enable_quality_weighting_bool):
        cmd.append("--enable_quality_weighting")
    if parse_bool(args.ligt_use_qtrack_weight_bool):
        cmd.append("--ligt_use_qtrack_weight")
    if args.prepared_scene_dir_autok:
        prepared_dir = Path(args.prepared_scene_dir_autok)
        k_npy = prepared_dir / "K.npy"
        ks_npy = prepared_dir / "Ks.npy"
        image_k_idx_npy = prepared_dir / "image_K_idx.npy"
        if k_npy.exists():
            cmd.extend(["--K_npy", str(k_npy)])
        elif ks_npy.exists() and image_k_idx_npy.exists():
            cmd.extend(["--Ks_npy", str(ks_npy), "--image_K_idx_npy", str(image_k_idx_npy)])
        else:
            raise SystemExit(
                f"Could not find intrinsics under prepared scene dir: {prepared_dir}. "
                "Expected either K.npy or Ks.npy + image_K_idx.npy."
            )

    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
