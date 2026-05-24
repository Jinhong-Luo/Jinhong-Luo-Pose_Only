#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Wrapper for generate_rraa_input.py with automatic K/Ks selection.")
    ap.add_argument("--prepared_scene_dir_autok", required=True)
    args, rest = ap.parse_known_args()

    prepared_dir = Path(args.prepared_scene_dir_autok)
    k_npy = prepared_dir / "K.npy"
    ks_npy = prepared_dir / "Ks.npy"
    image_k_idx_npy = prepared_dir / "image_K_idx.npy"

    cmd = [sys.executable, str(REPO_ROOT / "tools" / "generate_rraa_input.py")] + rest
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
