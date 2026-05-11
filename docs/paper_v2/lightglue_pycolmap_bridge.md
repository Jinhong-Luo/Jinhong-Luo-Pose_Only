# LightGlue + pycolmap Track Bridge

This bridge replaces the current in-repo multi-view track construction with a
pycolmap-backed pipeline while keeping the existing LightGlue matcher and the
LiGT / PA backend unchanged.

Pipeline:

1. Extract / reuse SuperPoint features with `frontend_cache.py`
2. Match selected image pairs with LightGlue
3. Write cameras, images, keypoints, and raw matches into a pycolmap database
4. Run pycolmap geometric verification and incremental mapping
5. Export the resulting sparse-model observations back to `tracks/*.npz`
6. Reuse the exported tracks with `Pose_Only_patched_v3_fixed.py`

Main tool:

- `tools/build_lightglue_pycolmap_tracks_npz.py`

Batch frontend config example:

- `configs/paper_v2/frontend_eth3d5_pycolmap.json`

Outputs:

- `database.db`
- `pairs.txt`
- `sparse/0/*.bin`
- `tracks/*.npz`
- `tracks/track_quality_summary.npz`
- `tracks/pair_quality_edges.npz`
- `tracks/track_build_quality_stats.json`
- `pycolmap_track_build_stats.json`

Example:

```powershell
.\.venv\Scripts\python.exe .\tools\build_lightglue_pycolmap_tracks_npz.py `
  --image_list .\data\prepared\ETH3D\office\image_list.txt `
  --out_dir .\runs\paper_v2\pycolmap_bridge_full\office `
  --dataset custom `
  --K_npy .\data\prepared\ETH3D\office\K.npy `
  --deltas 1,2,3,5 `
  --device auto `
  --max_kpts 2048 `
  --filter_th 0.1 `
  --mutual `
  --feature_cache_dir .\runs\paper_v2\pycolmap_bridge_full\office\feature_cache `
  --min_track_len 2 `
  --quality_config .\configs\quality_config.template.json `
  --auto_quality_refs
```

Then run Pose-Only with the new tracks:

```powershell
.\.venv\Scripts\python.exe .\tools\Pose_Only_patched_v3_fixed.py `
  --track_npz_dir .\runs\paper_v2\pycolmap_bridge_full\office\tracks `
  --r_abs_npy .\data\prepared\ETH3D\office\R_abs_gt_w2c.npy `
  --dataset custom `
  --K_npy .\data\prepared\ETH3D\office\K.npy `
  --gt_pose_txt .\data\prepared\ETH3D\office\gt_poses_c2w.txt `
  --out_dir .\runs\paper_v2\pycolmap_bridge_full\office\pose_only `
  --min_track_len 5 `
  --max_tracks 20000 `
  --u_min 0.001 `
  --g_min 0.001 `
  --base_pair_candidates 80 `
  --base_pair_full_search_len 50 `
  --irls_iters 2 `
  --qtrack_mode weighted_sum `
  --qtrack_threshold 0.0 `
  --quality_config .\configs\quality_config.template.json `
  --auto_quality_refs `
  --dump_quality_stats `
  --dump_degeneracy_stats `
  --enable_quality_weighting `
  --ligt_use_qtrack_weight
```

Or batch-precompute the ETH3D bridge frontend:

```powershell
.\.venv\Scripts\python.exe .\tools\frontend_precompute.py `
  --config .\configs\paper_v2\frontend_eth3d5_pycolmap.json `
  --reuse_mode reuse
```

Notes:

- The current bridge is fully scriptable and does not require the COLMAP GUI.
- In the current implementation, track construction is obtained from the
  verified-match graph plus `pycolmap.incremental_mapping`; the exported
  multi-view tracks come from the resulting sparse model (`images.bin` and
  `points3D.bin`).
- In this environment, pycolmap database writes may require running outside the
  filesystem sandbox.
- This bridge is intended as a validation path first: it answers whether
  better multi-view tracks materially improve LiGT on ETH3D.
