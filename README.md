# Jinhong-Luo-Pose_Only

This repository contains the current local runnable code for the paper-oriented multi-view pose-only pipeline.

## Active Layout

- `tools/`: active scripts and shared utilities
- `configs/paper_v2/`: current main-experiment and ablation configs
- `scripts/paper_v2/`: batch runner scripts
- `docs/paper_v2/`: current protocol notes
- `runs/paper_v2/`: local experiment outputs, ignored by git

Historical materials were moved to local archive storage and are not pushed by default.

## Current Main Commands

Main experiment rerun:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\paper_v2\run_experiment_matrix.ps1 `
  -ScenesConfig .\configs\paper_v2\scenes_main8.json `
  -ExperimentsConfig .\configs\paper_v2\experiments_main.json `
  -ParamsConfig .\configs\paper_v2\params_best.json `
  -SkipIfSummaryExists
```

Main evaluation:

```powershell
.\.venv\Scripts\python.exe .\tools\eval_paper_benchmark.py `
  --manifest .\configs\paper_v2\paper_benchmark_manifest_v2.json `
  --experiment_tags main `
  --output_dir D:\Program\PyCharmMiscProject\Pose_estimation\runs\paper_v2\eval_main
```

Ablation rerun:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\paper_v2\run_experiment_matrix.ps1 `
  -ScenesConfig .\configs\paper_v2\scenes_ablation_strecha3.json `
  -ExperimentsConfig .\configs\paper_v2\experiments_ablation.json `
  -ParamsConfig .\configs\paper_v2\params_best.json `
  -SkipIfSummaryExists
```

Ablation evaluation:

```powershell
.\.venv\Scripts\python.exe .\tools\eval_paper_benchmark.py `
  --manifest .\configs\paper_v2\paper_benchmark_manifest_v2.json `
  --scene_tags ablation `
  --experiment_tags ablation `
  --output_dir D:\Program\PyCharmMiscProject\Pose_estimation\runs\paper_v2\eval_ablation
```
