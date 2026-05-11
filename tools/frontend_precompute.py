#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_structured_config(path: str) -> Dict[str, Any]:
    path = os.path.abspath(path)
    suffix = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8-sig") as f:
        if suffix in (".yml", ".yaml"):
            import yaml

            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dict: {path}")
    data["_config_path"] = path
    data["_config_dir"] = os.path.dirname(path)
    return data


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def resolve_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    norm = path.replace("/", os.sep).replace("\\", os.sep)
    if norm.startswith(f".{os.sep}") or norm.startswith(f"..{os.sep}") or norm in (".", ".."):
        return os.path.abspath(os.path.join(base_dir, path))
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def render_value(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**context)
    if isinstance(value, list):
        return [render_value(v, context) for v in value]
    if isinstance(value, dict):
        return {k: render_value(v, context) for k, v in value.items()}
    return value


def outputs_exist(paths: List[str]) -> bool:
    return bool(paths) and all(os.path.exists(p) for p in paths)


def dump_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_stage(stage: Dict[str, Any], context: Dict[str, Any], *, reuse_mode: str) -> Dict[str, Any]:
    rendered = render_value(stage, context)
    stage_name = rendered.get("name", "unnamed_stage")
    cmd = rendered.get("cmd")
    if not isinstance(cmd, list) or not cmd:
        raise ValueError(f"Stage '{stage_name}' must provide a non-empty cmd list.")

    cwd = resolve_path(rendered.get("cwd"), REPO_ROOT) or REPO_ROOT
    reuse_outputs = [resolve_path(p, REPO_ROOT) for p in ensure_list(rendered.get("reuse_outputs"))]

    if reuse_mode != "full" and outputs_exist(reuse_outputs) and not rendered.get("always_run", False):
        return {
            "name": stage_name,
            "status": "reused",
            "cmd": [stringify(x) for x in cmd],
            "cwd": cwd,
            "reuse_outputs": reuse_outputs,
        }

    print(f"[frontend_precompute] stage={stage_name}")
    result = subprocess.run([stringify(x) for x in cmd], cwd=cwd, check=False)
    return {
        "name": stage_name,
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": int(result.returncode),
        "cmd": [stringify(x) for x in cmd],
        "cwd": cwd,
        "reuse_outputs": reuse_outputs,
    }


def build_context(
    scene_cfg: Dict[str, Any],
    config: Dict[str, Any],
    args: argparse.Namespace,
    output_root: str,
) -> Dict[str, Any]:
    config_dir = config["_config_dir"]
    scene_root = resolve_path(scene_cfg.get("scene_root", "."), config_dir)
    frontend_cache_dir = resolve_path(
        scene_cfg.get("frontend_cache_dir", os.path.join(scene_root or ".", "frontend_cache")),
        config_dir,
    )

    context: Dict[str, Any] = {
        "python": args.python or sys.executable,
        "repo_root": REPO_ROOT,
        "config_dir": config_dir,
        "quality_config": resolve_path(config.get("quality_config"), config_dir) or "",
        "output_root": output_root,
        "scene_root": scene_root or "",
        "frontend_cache_dir": frontend_cache_dir or "",
    }
    context.update({k: v for k, v in scene_cfg.items() if isinstance(v, (str, int, float, bool))})
    return context


def run_frontend(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    output_root = resolve_path(args.output_root or config.get("output_root") or "runs/frontend_precompute", config["_config_dir"])
    assert output_root is not None
    os.makedirs(output_root, exist_ok=True)

    results: Dict[str, Any] = {
        "config_path": config["_config_path"],
        "output_root": output_root,
        "reuse_mode": args.reuse_mode,
        "scenes": [],
    }

    default_stages = ensure_list(config.get("default_frontend_stages"))
    for scene_cfg in ensure_list(config.get("scenes")):
        context = build_context(scene_cfg, config, args, output_root)
        scene_name = scene_cfg["name"]
        scene_record: Dict[str, Any] = {
            "scene_name": scene_name,
            "scene_group": scene_cfg.get("group"),
            "scene_root": context.get("scene_root"),
            "frontend_cache_dir": context.get("frontend_cache_dir"),
            "status": "ok",
            "stages": [],
        }

        stages = ensure_list(scene_cfg.get("frontend_stages")) or default_stages
        if not stages:
            raise ValueError(f"Scene '{scene_name}' has no frontend stages and config has no default_frontend_stages.")

        for stage in stages:
            stage_result = run_stage(stage, context, reuse_mode=args.reuse_mode)
            scene_record["stages"].append(stage_result)
            if stage_result["status"] == "failed":
                scene_record["status"] = "failed"
                break

        results["scenes"].append(scene_record)

    dump_json(os.path.join(output_root, "frontend_precompute_results.json"), results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute reusable frontend outputs (feature cache, tracks, rraa_input) for one or more scenes.")
    parser.add_argument("--config", required=True, help="JSON/YAML config describing scenes and frontend stages.")
    parser.add_argument("--reuse_mode", choices=["reuse", "resume", "full"], default="reuse")
    parser.add_argument("--output_root", default=None, help="Override summary output directory.")
    parser.add_argument("--python", default=None, help="Python executable to use for subprocess stages.")
    args = parser.parse_args()

    config = load_structured_config(args.config)
    results = run_frontend(config, args)
    summary_path = os.path.join(results["output_root"], "frontend_precompute_results.json")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
