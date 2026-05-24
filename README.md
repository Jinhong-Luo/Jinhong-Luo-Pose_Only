# Jinhong-Luo-Pose_Only

This repository contains a paper-oriented multi-view camera pose estimation pipeline. The current main line is a lightweight pose-only backend built around relative rotation averaging and LiGT-style translation estimation, with SuperPoint + LightGlue frontend matches prepared separately.

本仓库用于毕业论文实验中的多视图相机位姿估计流程。核心目标是在已有图像序列和前端匹配结果的基础上，研究轻量级 pose-only 后端的旋转与平移估计稳定性。

## Method Overview

The pipeline is organized into a frontend and a backend.

Frontend:

- SuperPoint is used for local feature extraction.
- LightGlue is used for pairwise feature matching.
- RANSAC-based geometric verification filters outliers.
- Verified matches are converted into multi-view tracks and relative-pose graph inputs.

Backend:

- RRAA performs robust relative rotation averaging.
- LiGT-style pose-only translation estimation solves camera positions from tracks and fixed rotations.
- Equation-level IRLS is used to stabilize LiGT translation estimation.
- Optional translation-only PA can be enabled as a refinement, but it is not the default main method because of its runtime cost.

The default paper method is denoted as **M3**:

| Item | Setting |
|---|---|
| Track-building span | `{1,2,3}` |
| RRAA graph span | `{1,2,3,5,8}` |
| LiGT IRLS | enabled |
| IRLS iterations | `2` |
| Huber coefficient | `1.5` |
| qpair/qtrack quality weighting | disabled |
| fallback | disabled |
| PA refinement | disabled |

The optional **M4** setting is M3 plus translation-only PA.

## Staged Parameter Selection

The final configuration is selected by a staged Optuna/TPE protocol rather than by a single scene or a single metric.

- **Stage 1:** search graph-span protocol, including track-building deltas and RRAA graph deltas.
- **Stage 2:** search LiGT IRLS iterations and Huber coefficient on the selected graph protocol.
- **Stage 3:** check whether auxiliary modules such as quality weighting, fallback, and PA provide stable gains.

The internal selection score is a robustness-oriented score:

```text
score =
    mean_primary_metric
  + 1000 * failure_rate
  + 100  * skip_rate
  + std_primary_metric
  + 0.5  * worst_primary_metric
  + 0.5  * mean_rotation_median_deg
  + 50   * mean_reject_ratio
```

The primary metric is `translation_vs_colmap_ratio`. This score is used only for internal configuration selection. The paper tables still report standard translation error, rotation error, runtime, and memory metrics.

## Dataset Split Used in the Paper

The staged search uses a dev/holdout split.

Development set, `configs/paper_v2/scenes_strecha4_dtu4_dev8_v2.json`:

- Strecha: `fountain-P11`, `entry-P10`, `Herz-Jesus-P8`, `Castle-P19`
- DTU: `scan1`, `scan40`, `scan69`, `scan106`

Holdout set, `configs/paper_v2/scenes_strecha2_dtu2_holdout4_v2.json`:

- Strecha: `Castle-P30-first29`, `Herz-Jesus-P25`
- DTU: `scan97`, `scan114`

Full 12-scene evaluation, `configs/paper_v2/scenes_strecha6_dtu6_12scenes.json`, uses all of the above scenes.

Large image data, prepared scene folders, feature caches, COLMAP outputs, and experiment runs are not included in this repository.

## Repository Layout

Files intended to be uploaded to GitHub:

- `README.md`: project overview and reproduction notes
- `PROJECT_LAYOUT.md`: short local layout note
- `environment_local.yml`: local Python/conda environment reference
- `configs/`: experiment, scene, staged-search, and quality configs
- `scripts/paper_v2/`: PowerShell and Python batch runners used for paper experiments
- `tools/`: active implementation and export scripts
- `docs/`: paper protocol notes and experiment documentation

Files and directories intentionally excluded from GitHub:

- `.venv/`, `venv/`
- `.cache/`
- `data/raw/`
- `data/prepared/`
- `runs/`
- `Colmap_runs/`
- `paper_ai_bundle/`
- `paper_ai_bundle*.zip`
- `archive/`
- `third_party/` weights/checkpoints

These ignored folders contain local datasets, cached features, model checkpoints, intermediate experiment outputs, or private writing bundles.

## Environment

The local backend experiments were run on Windows with Python 3.10. The reference environment is recorded in:

```text
environment_local.yml
```

Create a conda environment:

```powershell
conda env create -f environment_local.yml
conda activate pose_estimation_local
```

Or use the existing local virtual environment if available:

```powershell
.\.venv\Scripts\python.exe --version
```

The frontend feature extraction and matching experiments used a cloud GPU environment with PyTorch 2.1.2, Python 3.10, CUDA 11.8, and RTX 4090 24GB.

## Important Commands

### Clean Main Ablation

Run the full clean main ablation matrix:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\paper_v2\run_ablation_clean_main.ps1 `
  -ReuseMode full `
  -IncludeOptional `
  -PlotSearch
```

Run only the track-span sensitivity table:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\paper_v2\run_ablation_clean_main.ps1 `
  -ReuseMode full `
  -OnlyTables T `
  -IncludeOptional
```

Generate paper-ready ablation tables:

```powershell
.\.venv\Scripts\python.exe tools\generate_ablation_clean_tables.py `
  --config configs\paper_v2\experiments_ablation_clean_main.json

.\.venv\Scripts\python.exe tools\export_ablation_paper_ready_tables.py
```

Main outputs:

```text
runs/paper_v2/ablation_clean_main/tables/
runs/paper_v2/ablation_clean_main/tables/paper_ready/
```

### Staged Search Figures

Generate the original staged-search figures:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\paper_v2\plot_stage1_graph_span_search.ps1
powershell -ExecutionPolicy Bypass -File scripts\paper_v2\plot_stage2_irls_search.ps1
```

Generate split Chinese figures for thesis editing:

```powershell
.\.venv\Scripts\python.exe tools\plot_staged_search_split_zh.py
```

Outputs:

```text
runs/paper_v2/staged_search/stage1_graph_span_dev8_v2/figures/
runs/paper_v2/staged_search/stage2_irls_on_A4_uniform_dev8_v2/figures/
```

### External Comparison Tables

Export M3/M4 scene-level metrics and selected external comparison tables:

```powershell
.\.venv\Scripts\python.exe tools\export_m3_m4_scene_metrics.py
.\.venv\Scripts\python.exe tools\export_colmap_pose_errors.py
.\.venv\Scripts\python.exe tools\export_selected_external_comparison_tables.py
```

Typical outputs are written under:

```text
runs/paper_v2/paper_ablation_assets_2026-05-02/external_compare/
```

## Key Configs

Final clean ablation matrix:

```text
configs/paper_v2/experiments_ablation_clean_main.json
```

Stage-1 graph-span search:

```text
configs/paper_v2/staged_search/stage1_graph_span_dev8_v2.json
```

Stage-2 IRLS search:

```text
configs/paper_v2/staged_search/stage2_irls_on_A4_uniform_dev8_v2.json
```

Full 12-scene list:

```text
configs/paper_v2/scenes_strecha6_dtu6_12scenes.json
```

No-RRAA-weighting quality config:

```text
configs/quality_config.no_rraa_weighting.json
```

## Key Scripts

Core backend:

- `tools/RRAA_fast.py`
- `tools/Pose_Only_patched_v3_fixed.py`
- `tools/run_poseonly_with_flags.py`

Frontend and input generation:

- `tools/run_build_tracks_with_autok.py`
- `tools/run_generate_rraa_input_with_autok.py`
- `tools/build_lightglue_pycolmap_tracks_npz.py`
- `tools/frontend_cache.py`
- `tools/pair_match_cache.py`

Validation and search:

- `tools/validation_search.py`
- `tools/optuna_validation_search.py`
- `tools/run_ablation_paper_final.py`

Evaluation and table export:

- `tools/eval_rraa_rotation.py`
- `tools/eval_poseonly_strecha_mm.py`
- `tools/experiment_summary.py`
- `tools/generate_ablation_clean_tables.py`
- `tools/export_ablation_paper_ready_tables.py`
- `tools/export_m3_m4_scene_metrics.py`
- `tools/export_colmap_pose_errors.py`
- `tools/export_selected_external_comparison_tables.py`

Plotting:

- `tools/plot_stage1_graph_span_search.py`
- `tools/plot_stage2_irls_search.py`
- `tools/plot_staged_search_split_zh.py`

## Upload Workflow

Check what will be uploaded:

```powershell
git status
```

Add the paper-relevant files:

```powershell
git add README.md PROJECT_LAYOUT.md environment_local.yml .gitignore `
  configs docs scripts\paper_v2 tools
```

Commit:

```powershell
git commit -m "Update paper pose-only project documentation and configs"
```

Push:

```powershell
git push origin main
```

Before pushing, make sure `git status` does not show large local folders such as `runs/`, `data/prepared/`, `Colmap_runs/`, `.venv/`, or `paper_ai_bundle/` staged for commit.

## Notes

This repository is intended to preserve the runnable research code, staged-search configs, and paper table/figure generation scripts. It does not include private datasets or bulky experiment outputs. To fully reproduce the reported numbers, the prepared Strecha/DTU scene directories and cached frontend features must be restored locally according to the paths in the scene config files.
