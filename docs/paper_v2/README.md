# Paper V2 Layout

This project tree is now organized around the next rerun cycle:

- main comparison rerun:
  - 3 Strecha scenes
  - 5 ETH3D scenes
  - `facade` excluded
- ablation rerun:
  - Strecha 3 scenes only

## Active Structure

- `configs/paper_v2/`
  - `params_best.json`: frozen best parameters
  - `scenes_main8.json`: main rerun scene list
  - `scenes_ablation_strecha3.json`: ablation scene list
  - `experiments_main.json`: four main groups
  - `experiments_ablation.json`: minimal ablation groups
  - `paper_benchmark_manifest_v2.json`: evaluation manifest for the new outputs
- `scripts/paper_v2/`
  - `run_experiment_matrix.ps1`: one batch runner for main and ablation reruns
- `runs/paper_v2/`
  - `main/`: new main comparison outputs
  - `ablation/`: new ablation outputs
- `tools/`
  - split into `data_prep`, `frontend`, `solvers`, `evaluation`, `orchestration`
  - root `tools/*.py` remain as compatibility entry points

## Archive Location

Old configs, docs, scripts, notes, and historical experiment outputs were archived under:

- `archive/2026-04-05_project_cleanup/`

## Frozen Best Parameters

Current active best defaults:

- `qtrack_mode = weighted_sum`
- `u_min = 1e-3`
- `g_min = 1e-3`
- `pa_max_init_rmse = 0.05`
- main RRAA route keeps quality weighting on
- main RRAA route keeps `rraa_use_qpair_weight` on

## Main Comparison Command

```powershell
.\scripts\paper_v2\run_experiment_matrix.ps1 `
  -ScenesConfig configs\paper_v2\scenes_main8.json `
  -ExperimentsConfig configs\paper_v2\experiments_main.json `
  -ParamsConfig configs\paper_v2\params_best.json `
  -SkipIfSummaryExists
```

## Ablation Command

```powershell
.\scripts\paper_v2\run_experiment_matrix.ps1 `
  -ScenesConfig configs\paper_v2\scenes_ablation_strecha3.json `
  -ExperimentsConfig configs\paper_v2\experiments_ablation.json `
  -ParamsConfig configs\paper_v2\params_best.json `
  -SkipIfSummaryExists
```

## Evaluate New Main Outputs

```powershell
.\.venv\Scripts\python.exe tools\eval_paper_benchmark.py `
  --manifest configs\paper_v2\paper_benchmark_manifest_v2.json `
  --experiment_tags main `
  --output_dir runs\paper_v2\eval_main
```

## Evaluate New Ablation Outputs

```powershell
.\.venv\Scripts\python.exe tools\eval_paper_benchmark.py `
  --manifest configs\paper_v2\paper_benchmark_manifest_v2.json `
  --scene_tags ablation `
  --experiment_tags ablation `
  --output_dir runs\paper_v2\eval_ablation
```
