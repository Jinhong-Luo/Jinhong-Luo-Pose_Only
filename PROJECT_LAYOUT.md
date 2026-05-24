# Project Layout

This repository is organized around the paper-v2 pose-only experiments.

## Versioned in Git

- `README.md`  
  GitHub-facing project overview, method summary, commands, and upload checklist.

- `environment_local.yml`  
  Reference conda environment for the local backend scripts.

- `configs/`  
  Scene lists, experiment matrices, staged-search configs, quality configs, and paper benchmark manifests.

- `scripts/paper_v2/`  
  PowerShell and Python wrappers for batch experiments, plotting, and report generation.

- `tools/`  
  Core implementation, frontend conversion utilities, validation/search drivers, evaluation scripts, table exporters, and plotting scripts.

- `docs/`  
  Notes for paper protocols and experiment organization.

## Local Only

The following folders are intentionally ignored by Git because they contain datasets, caches, bulky outputs, or private writing bundles:

- `.venv/`
- `.cache/`
- `data/raw/`
- `data/prepared/`
- `runs/`
- `Colmap_runs/`
- `paper_ai_bundle/`
- `paper_ai_bundle*.zip`
- `archive/`
- `third_party/`

## Main Paper Paths

- Clean ablation matrix: `configs/paper_v2/experiments_ablation_clean_main.json`
- Full 12-scene list: `configs/paper_v2/scenes_strecha6_dtu6_12scenes.json`
- Dev8 list: `configs/paper_v2/scenes_strecha4_dtu4_dev8_v2.json`
- Holdout4 list: `configs/paper_v2/scenes_strecha2_dtu2_holdout4_v2.json`
- Stage-1 graph search config: `configs/paper_v2/staged_search/stage1_graph_span_dev8_v2.json`
- Stage-2 IRLS search config: `configs/paper_v2/staged_search/stage2_irls_on_A4_uniform_dev8_v2.json`

## Main Runners

- Clean ablation runner: `scripts/paper_v2/run_ablation_clean_main.ps1`
- Ablation matrix driver: `tools/run_ablation_paper_final.py`
- Validation/search driver: `tools/validation_search.py`
- Optuna/TPE wrapper: `tools/optuna_validation_search.py`
- Paper-ready table exporter: `tools/export_ablation_paper_ready_tables.py`
- Split Chinese staged-search plot exporter: `tools/plot_staged_search_split_zh.py`
