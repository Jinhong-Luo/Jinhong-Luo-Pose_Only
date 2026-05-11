# Paper Final Ablation Table Templates

This document defines the recommended table headers and the run-to-table mapping for the final paper ablation suite.

## 1. Main Cumulative Ablation

Suggested caption:

`Main cumulative ablation from the paper baseline to the final proposed protocol.`

Header:

| Label | Protocol | Mean rot. median ↓ | Mean trans. ↓ | Worst ↓ | Std ↓ | Failure | RRAA edge count | Largest component ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|

Logical rows:

| Label | Source |
|---|---|
| `M0` | `A3_paper_base` |
| `M1` | `A4_plus_gap8_uniform` |
| `M2` | `F2_gap_adaptive_strict` |
| `M3` | `I1_irls1_huber15` |
| `M4` | reuse `I1_irls1_huber15` metrics, but describe it as the `split_v2` dev/holdout frozen final protocol |

## 2. Graph Protocol Ablation

Suggested caption:

`Ablation of graph-span design from chain graph to middle/long-gap rotation graph.`

Header:

| Label | Tracks deltas | RRAA deltas | Mean rot. median ↓ | Mean trans. ↓ | Worst ↓ | Std ↓ | Failure | RRAA edge count | Largest component ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|

Logical rows:

| Label | Source |
|---|---|
| `A0` | `A0_chain_lower_bound` |
| `A1` | `A1_tracks_multigap_only` |
| `A2` | `A2_short_rotation_graph` |
| `A3` | `A3_paper_base` |
| `A4` | `A4_plus_gap8_uniform` |
| `A5` | `A5_plus_gap13_optional` |

## 3. Frontend Filtering Ablation

Suggested caption:

`Ablation of structured frontend filtering after fixing the long-gap rotation graph.`

Header:

| Label | min_score | min_inliers_map | Mean rot. median ↓ | Mean trans. ↓ | Worst ↓ | Std ↓ | Failure | RRAA edge count | Largest component ratio |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|

Logical rows:

| Label | Source |
|---|---|
| `F0` | `A4_plus_gap8_uniform` |
| `F1` | `F1_score_filter_only` |
| `F2` | `F2_gap_adaptive_strict` |
| `F3` | `F3_gap_adaptive_strict_nearby_optional` |

## 4. IRLS Stabilization Ablation

Suggested caption:

`Ablation of equation-level IRLS stabilization on top of the strongest frontend protocol.`

Main header:

| Label | IRLS iters | Huber k | Mean trans. ↓ | Std ↓ | Worst ↓ | Failure | Equations kept |
|---|---:|---:|---:|---:|---:|---:|---:|

Residual diagnostics header:

| Label | LiGT residual median ↓ | p90 ↓ | p99 ↓ | Mean trans. ↓ |
|---|---:|---:|---:|---:|

Logical rows:

| Label | Source |
|---|---|
| `I0` | reuse `F2_gap_adaptive_strict` metrics |
| `I1` | `I1_irls1_huber15` |
| `I2` | `I2_irls2_huber15` |
| `I3` | `I3_irls1_huber20_optional` |

## 5. Innovation 3 Figures

Recommended figures:

1. `Optimization history on the development set`
   - x: `trial number`
   - y: `score`
   - curves: raw trial scores + best-so-far

2. `Parameter importance`
   - parameters: `min_inliers_map`, `irls_iters`, `deltas_tracks`, `irls_huber_k`, `min_score`, `ransac_px`, `qpair_mode_tracks`

3. `Parallel coordinate plot`
   - axes: `deltas_tracks`, `min_score`, `min_inliers_map`, `irls_iters`, `irls_huber_k`, `score`

4. `Development vs. holdout protocol comparison`
   - protocols: initial manual, mid manual, dev-optuna-best, final frozen

## 6. One-Click Run Commands

Run all non-optional groups:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/paper_v2/run_ablation_paper_final.ps1
```

Run only the main cumulative sources:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/paper_v2/run_ablation_paper_final.ps1 -OnlyRuns A3_paper_base,A4_plus_gap8_uniform,F2_gap_adaptive_strict,I1_irls1_huber15
```

Run only graph protocol ablation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/paper_v2/run_ablation_paper_final.ps1 -OnlyTables A
```

Run optional groups too:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/paper_v2/run_ablation_paper_final.ps1 -IncludeOptional
```
