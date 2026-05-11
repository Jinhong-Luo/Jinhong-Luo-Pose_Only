# Tools Layout

`tools/` now keeps only one active implementation per script at the root level.

Main active scripts:

- data preparation:
  - `strecha_prepare.py`
  - `eth3d_prepare.py`
  - `generate_rraa_input.py`
  - `prepare_rotation_source.py`
- frontend:
  - `build_lightglue_tracks_npz.py`
  - `build_lightglue_pycolmap_tracks_npz.py`
  - `frontend_cache.py`
  - `frontend_precompute.py`
  - `colmap_points_to_tracks_npz.py`
- solvers:
  - `RRAA_fast.py`
  - `Pose_Only_patched_v3_fixed.py`
- evaluation:
  - `eval_rraa_rotation.py`
  - `eval_poseonly_strecha_mm.py`
  - `experiment_summary.py`
  - `eval_paper_benchmark.py`
  - `plot_ready_export.py`
- orchestration:
  - `validation_search.py`

Shared utility modules:

- `calib_utils.py`
- `degeneracy_utils.py`
- `quality_utils.py`
- `_bootstrap.py`
