#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from validation_search import load_structured_config, resolve_path


REPO_ROOT = Path(__file__).resolve().parents[1]


def normalize_filters(values: Optional[List[str]]) -> List[str]:
    items: List[str] = []
    for value in values or []:
        for part in str(value).split(","):
            item = part.strip()
            if item and item not in items:
                items.append(item)
    return items


def make_singleton_space(params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        key: {
            "type": "categorical",
            "choices": [value],
        }
        for key, value in params.items()
    }


def build_config(
    template_cfg: Dict[str, Any],
    *,
    study_name: str,
    output_root: str,
    scenes: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = deepcopy(template_cfg)
    cfg["study_name"] = study_name
    cfg["output_root"] = output_root
    cfg["n_trials"] = 1
    cfg["optuna_search_space"] = make_singleton_space(params)
    cfg["base_validation_config"]["scenes"] = scenes
    if params.get("run_pa"):
        configure_pa_stages(cfg, params)
    return cfg


def append_flag(cmd: List[Any], flag: str, value: Any = None) -> None:
    cmd.append(flag)
    if value is not None:
        cmd.append(value)


def configure_pa_stages(cfg: Dict[str, Any], params: Dict[str, Any]) -> None:
    stages = cfg.get("base_validation_config", {}).get("default_candidate_stages") or []
    for stage in stages:
        name = stage.get("name")
        cmd = stage.get("cmd")
        if not isinstance(cmd, list):
            continue

        if name == "pose_only":
            append_flag(cmd, "--run_pa")
            for key, flag in [
                ("pa_iters", "--pa_iters"),
                ("pa_point_rms", "--pa_point_rms"),
                ("pa_max_nfev", "--pa_max_nfev"),
                ("pa_loss", "--pa_loss"),
                ("pa_f_scale", "--pa_f_scale"),
                ("pa_max_step_t", "--pa_max_step_t"),
                ("pa_max_step_r_deg", "--pa_max_step_r_deg"),
                ("pa_min_points", "--pa_min_points"),
                ("pa_max_init_rmse", "--pa_max_init_rmse"),
            ]:
                if key in params:
                    append_flag(cmd, flag, params[key])
            for key, flag in [
                ("pa_no_refine_rot", "--pa_no_refine_rot"),
                ("pa_no_refine_trans", "--pa_no_refine_trans"),
                ("pa_use_qtrack", "--pa_use_qtrack"),
                ("pa_accept_any_update", "--pa_accept_any_update"),
                ("enable_degeneracy_guard", "--enable_degeneracy_guard"),
                ("pa_skip_on_degenerate", "--pa_skip_on_degenerate"),
            ]:
                if params.get(key):
                    append_flag(cmd, flag)
            reuse = stage.setdefault("reuse_outputs", [])
            reuse.extend([
                "{candidate_root}\\pose_only\\poses_c2w.txt",
                "{candidate_root}\\pose_only\\pa_degeneracy_stats.json",
            ])

        elif name == "eval_translation":
            for idx, token in enumerate(cmd):
                if token == "--est_poses" and idx + 1 < len(cmd):
                    cmd[idx + 1] = "{candidate_root}\\pose_only\\poses_c2w.txt"
                    break

        elif name == "summary":
            if "--pa_degeneracy_json" not in cmd:
                cmd.extend([
                    "--pa_degeneracy_json",
                    "{candidate_root}\\pose_only\\pa_degeneracy_stats.json",
                ])


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def should_run(
    run_doc: Dict[str, Any],
    *,
    only_runs: List[str],
    only_tables: List[str],
    include_optional: bool,
) -> bool:
    if run_doc.get("optional") and not include_optional:
        return False
    if only_runs and run_doc["id"] not in only_runs:
        return False
    if only_tables and str(run_doc.get("table")) not in only_tables:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate and run the final paper ablation matrix.")
    ap.add_argument("--config", required=True, help="Matrix config JSON.")
    ap.add_argument("--only-runs", action="append", default=[], help="Comma-separated run ids to execute.")
    ap.add_argument("--only-tables", action="append", default=[], help="Comma-separated table ids to execute (A/F/I).")
    ap.add_argument("--include-optional", action="store_true", help="Run optional groups such as A5/F3/I3.")
    ap.add_argument("--reuse-mode", choices=["reuse", "resume", "full"], default="reuse")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    matrix = load_structured_config(args.config)
    config_dir = Path(matrix["_config_dir"]).resolve()
    template_cfg = load_structured_config(resolve_path(matrix["base_template_config"], str(config_dir)))
    output_root_base = Path(resolve_path(matrix["output_root_base"], str(config_dir))).resolve()
    scenes = resolve_path(matrix["scenes"], str(config_dir))
    only_runs = normalize_filters(args.only_runs)
    only_tables = normalize_filters(args.only_tables)

    defaults = dict(matrix.get("defaults") or {})
    generated_dir = output_root_base / "_generated_configs"
    manifest: Dict[str, Any] = {
        "matrix_name": matrix.get("name"),
        "output_root_base": str(output_root_base),
        "generated_configs": [],
        "logical_tables": matrix.get("logical_tables", {}),
    }

    optuna_driver = REPO_ROOT / "tools" / "optuna_validation_search.py"
    python_exe = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

    for run_doc in matrix.get("runs", []):
        if not should_run(
            run_doc,
            only_runs=only_runs,
            only_tables=only_tables,
            include_optional=args.include_optional,
        ):
            continue

        params = dict(defaults)
        params.update(run_doc.get("params") or {})
        study_name = str(run_doc["id"])
        output_root = str((output_root_base / str(run_doc["output_subdir"])).resolve())
        generated_cfg = build_config(
            template_cfg,
            study_name=study_name,
            output_root=output_root,
            scenes=scenes,
            params=params,
        )
        generated_path = generated_dir / f"{study_name}.json"
        write_json(generated_path, generated_cfg)
        manifest["generated_configs"].append(
            {
                "id": study_name,
                "table": run_doc.get("table"),
                "description": run_doc.get("description"),
                "generated_config": str(generated_path),
                "output_root": output_root,
                "params": params,
                "optional": bool(run_doc.get("optional", False)),
            }
        )

        if args.dry_run:
            print(f"[dry-run] generated {generated_path}")
            continue

        cmd = [
            str(python_exe),
            str(optuna_driver),
            "--config",
            str(generated_path),
            "--reuse_mode",
            args.reuse_mode,
        ]
        print("[ablation-paper-final]", " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    write_json(output_root_base / "run_manifest.json", manifest)
    print("saved:", output_root_base / "run_manifest.json")


if __name__ == "__main__":
    main()
