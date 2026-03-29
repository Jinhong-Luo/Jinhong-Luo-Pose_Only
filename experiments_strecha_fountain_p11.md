# Experiments: Strecha `fountain-P11`

## Goal

Record reproducible experiments for:

- LightGlue / SuperPoint matching and track building
- RRAA global rotation
- PoseOnly (LiGT)
- Optional PA refinement

This file is intended to be appended over time for ablation studies and final thesis reporting.

## Dataset Snapshot

- Scene: `data/raw/strecha/fountain-P11`
- Prepared data: `data/prepared/strecha/fountain-P11`
- Frames: `11`
- Intrinsics source: Strecha `.camera`
- GT preparation status:
  - `tools/strecha_prepare.py` fixed to correctly parse the extra `0 0 0` row in `.camera`
  - saved GT files:
    - `R_abs_gt_c2w.npy`
    - `R_abs_gt_w2c.npy`
    - `gt_centers.npy`
    - `gt_poses_c2w.txt`
    - `gt_poses_w2c.txt`

## Canonical Commands

### Prepare

```powershell
.\.venv\Scripts\python.exe tools/strecha_prepare.py `
  --scene_dir data/raw/strecha/fountain-P11 `
  --out_dir data/prepared/strecha/fountain-P11
```

### Tracks

```powershell
.\.venv\Scripts\python.exe tools/build_lightglue_tracks_npz.py `
  --dataset custom `
  --K_npy data/prepared/strecha/fountain-P11/K.npy `
  --image_glob "data/raw/strecha/fountain-P11/images/*.jpg" `
  --out_dir runs/strecha/fountain-P11/tracks_npz `
  --device cpu `
  --deltas 1,2,3,5 `
  --max_kpts 2048 `
  --filter_th 0.1 `
  --mutual `
  --min_score 0.2 `
  --use_ransac `
  --ransac_thresh 1.0 `
  --min_inliers 50
```

### RRAA input

```powershell
.\.venv\Scripts\python.exe tools/generate_rraa_input.py `
  --dataset custom `
  --K_npy data/prepared/strecha/fountain-P11/K.npy `
  --img_dir data/raw/strecha/fountain-P11/images `
  --out_npz runs/strecha/fountain-P11/rraa_input/rraa_input_fountain-P11.npz `
  --device cpu `
  --deltas 1,2,3,5 `
  --max_kpts 2048 `
  --filter_th 0.1 `
  --ransac_px 2.0 `
  --add_reverse `
  --use_h_fallback `
  --save_meta `
  --save_names
```

### RRAA

```powershell
.\.venv\Scripts\python.exe tools/RRAA_fast.py `
  --npz runs/strecha/fountain-P11/rraa_input/rraa_input_fountain-P11.npz `
  --out runs/strecha/fountain-P11/rraa_output/R_abs_fountain-P11.npy `
  --initial_l1_iters 3 `
  --loss l1_2 `
  --a_deg 5 `
  --irls_max_iter 50
```

## Experiment Table

| ID | Tracks | Rotation Input | PA | Output Dir | Rotation Median (deg) | Translation Median (mm) | Notes |
|---|---|---|---|---|---:|---:|---|
| E001 | `deltas=1,2,3,5` | `RRAA` | No | `runs/strecha/fountain-P11/poseonly_rraa` | `0.2019` | `24.6486` | Main LiGT result after GT parsing fix |
| E002 | `deltas=1,2,3,5` | `GT w2c` | No | `runs/strecha/fountain-P11/poseonly_gt` | `GT` | `2.3992` | Upper-bound baseline for translation |
| E003 | `deltas=1,2,3,5` | `RRAA` | Yes | `runs/strecha/fountain-P11/poseonly_rraa_pa` | `0.2019` | `6.2277` | PA gives large improvement over LiGT-only |

## Detailed Records

### E001: RRAA -> PoseOnly (LiGT)

- Tracks:
  - source dir: `runs/strecha/fountain-P11/tracks_npz`
  - pair summary from track build:
    - `pair count = 33`
    - `kept matches median = 488`
    - `merged median = 469`
    - `conflicts total = 251`
- RRAA:
  - output: `runs/strecha/fountain-P11/rraa_output/R_abs_fountain-P11.npy`
  - evaluation:
    - best convention: `est + right gauge`
    - rotation median: `0.201928 deg`
    - rotation mean: `0.194178 deg`
    - rotation p90: `0.263836 deg`
    - rotation max: `0.328314 deg`
- PoseOnly:
  - output: `runs/strecha/fountain-P11/poseonly_rraa`
  - evaluation:
    - translation median: `24.648645 mm`
    - translation mean: `25.892242 mm`
    - translation rmse: `28.639810 mm`
    - translation p90: `41.417611 mm`
    - translation max: `47.063557 mm`
- Interpretation:
  - Rotation is already very accurate.
  - Translation still has noticeable gap before PA.

### E002: GT Rotation -> PoseOnly (LiGT)

- GT rotation input:
  - `data/prepared/strecha/fountain-P11/R_abs_gt_w2c.npy`
- PoseOnly:
  - output: `runs/strecha/fountain-P11/poseonly_gt`
  - evaluation:
    - translation median: `2.399196 mm`
    - translation mean: `2.535740 mm`
    - translation rmse: `2.782129 mm`
    - translation p90: `3.854819 mm`
    - translation max: `5.193680 mm`
- Interpretation:
  - This is the practical upper bound under current tracks and LiGT implementation.
  - Remaining gap from E001 is not due to track/GT parsing alone.

### E003: RRAA -> PoseOnly + PA

- PoseOnly + PA:
  - output: `runs/strecha/fountain-P11/poseonly_rraa_pa`
  - PA settings:
    - `--run_pa`
    - `--pa_iters 5`
    - `--pa_max_nfev 60`
- PA optimization behavior:
  - init residual rmse: `9.588e-04`
  - after 5 accepted PA steps: residual rmse decreased to `8.352e-04`
- evaluation:
  - translation median: `6.227746 mm`
  - translation mean: `5.942468 mm`
  - translation rmse: `6.259186 mm`
  - translation p90: `8.928158 mm`
  - translation max: `9.131765 mm`
- Interpretation:
  - PA substantially improves over E001.
  - Current best practical result under estimated rotations.

## Track Notes

- Current `tracks_npz` was generated with `--deltas 1,2,3,5`.
- This is now the recommended default for Strecha.
- A previous hypothesis that rotation failure was caused by `RRAA` convention mismatch is no longer supported after fixing GT parsing.

## Current Best Summary

- Best estimated-rotation result:
  - `RRAA + PoseOnly + PA`
  - median translation error: `6.2277 mm`
- Best upper-bound baseline:
  - `GT rotation + PoseOnly`
  - median translation error: `2.3992 mm`

## Next Candidate Ablations

- Compare `tracks: deltas=1` vs `deltas=1,2,3,5`
- Run `GT rotation + PA`
- Tune PA:
  - `--pa_iters`
  - `--pa_max_nfev`
  - `--pa_loss`
  - `--pa_f_scale`
- Tune LiGT:
  - `--min_track_len`
  - `--u_min`
  - `--g_min`
  - `--eq_norm`
