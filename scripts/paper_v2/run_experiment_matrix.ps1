param(
    [Parameter(Mandatory = $true)]
    [string]$ScenesConfig,

    [Parameter(Mandatory = $true)]
    [string]$ExperimentsConfig,

    [string]$ParamsConfig = "configs/paper_v2/params_best.json",
    [string[]]$OnlyExperiments = @(),
    [string[]]$OnlyScenes = @(),
    [switch]$SkipIfSummaryExists
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath([string]$PathText) {
    if ([System.IO.Path]::IsPathRooted($PathText)) {
        return $PathText
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $PathText))
}

function Resolve-FirstExistingPath([string[]]$Candidates) {
    foreach ($item in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($item)) {
            continue
        }
        $resolved = Resolve-RepoPath $item
        if (Test-Path $resolved) {
            return $resolved
        }
    }
    return $null
}

function Invoke-Step([string]$Label, [string[]]$CommandArgs) {
    Write-Host "[paper-v2][$Label] $($CommandArgs -join ' ')" -ForegroundColor Cyan
    & $PythonExe @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Label"
    }
}

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\\.."))
$PythonExe = Resolve-RepoPath ".venv\\Scripts\\python.exe"

$ScenesDoc = Get-Content (Resolve-RepoPath $ScenesConfig) -Raw | ConvertFrom-Json
$ExperimentsDoc = Get-Content (Resolve-RepoPath $ExperimentsConfig) -Raw | ConvertFrom-Json
$ParamsDoc = Get-Content (Resolve-RepoPath $ParamsConfig) -Raw | ConvertFrom-Json
$OutputRootBase = Resolve-RepoPath $ExperimentsDoc.output_root_base

$Failures = New-Object System.Collections.Generic.List[object]

Push-Location $ProjectRoot
try {
    foreach ($Group in $ExperimentsDoc.groups) {
        if ($OnlyExperiments.Count -gt 0 -and $OnlyExperiments -notcontains $Group.name) {
            continue
        }

        foreach ($Scene in $ScenesDoc.scenes) {
            if ($OnlyScenes.Count -gt 0 -and $OnlyScenes -notcontains $Scene.scene_id -and $OnlyScenes -notcontains $Scene.scene_key) {
                continue
            }

            $RunDir = Join-Path (Join-Path $OutputRootBase $Group.name) $Scene.scene_key
            $RraaOut = Join-Path $RunDir "rraa_output\\R_abs.npy"
            $RraaStats = Join-Path $RunDir "rraa_output\\R_abs_stats.json"
            $PoseDir = Join-Path $RunDir "pose_only"
            $SummaryJson = Join-Path $RunDir "experiment_summary.json"
            $PreparedSceneDir = Resolve-RepoPath $Scene.prepared_scene_dir
            $SceneRoot = Resolve-RepoPath $Scene.scene_root
            $GtRot = Join-Path $PreparedSceneDir "R_abs_gt_w2c.npy"
            $GtCenters = Join-Path $PreparedSceneDir "gt_centers.npy"
            $GtPoses = Join-Path $PreparedSceneDir "gt_poses_c2w.txt"
            $TrackDir = Resolve-FirstExistingPath @(
                (Join-Path $SceneRoot "tracks"),
                ("runs\ETH3D\{0}\tracks" -f $Scene.scene_id),
                ("runs\ETH3D_repaired\{0}\tracks" -f $Scene.scene_id),
                ("runs\strecha\{0}\tracks" -f $Scene.scene_id)
            )
            $TrackStatsJson = if ($TrackDir) { Join-Path $TrackDir "track_build_quality_stats.json" } else { $null }
            $RraaInput = Resolve-FirstExistingPath @(
                (Join-Path $SceneRoot ("rraa_input\\rraa_input_{0}.npz" -f $Scene.scene_id)),
                ("runs\ETH3D\{0}\rraa_input\rraa_input_{0}.npz" -f $Scene.scene_id),
                ("runs\ETH3D_repaired\{0}\rraa_input\rraa_input_{0}.npz" -f $Scene.scene_id),
                ("runs\strecha\{0}\rraa_input\rraa_input_{0}.npz" -f $Scene.scene_id)
            )
            $RotationEvalJson = Join-Path $RunDir "rraa_output\\eval_rotation.json"
            $TranslationEvalJson = Join-Path $PoseDir "eval_translation.json"
            $QualityConfig = Resolve-RepoPath $ParamsDoc.quality_config
            $GtUnitToMm = if ($null -ne $Scene.gt_unit_to_mm) { [string]$Scene.gt_unit_to_mm } else { [string]$ParamsDoc.evaluation.gt_unit_to_mm }
            $KPath = Join-Path $PreparedSceneDir "K.npy"
            $KsPath = Join-Path $PreparedSceneDir "Ks.npy"
            $ImageKIdxPath = Join-Path $PreparedSceneDir "image_K_idx.npy"

            if ($SkipIfSummaryExists -and (Test-Path $SummaryJson)) {
                Write-Host "[paper-v2][skip] $($Group.name) / $($Scene.scene_key)" -ForegroundColor Yellow
                continue
            }

            New-Item -ItemType Directory -Force $RunDir | Out-Null
            New-Item -ItemType Directory -Force (Join-Path $RunDir "rraa_output") | Out-Null
            New-Item -ItemType Directory -Force $PoseDir | Out-Null

            try {
                if ($Group.rotation_source -eq "GT") {
                    Invoke-Step "$($Group.name):prepare_rotation:$($Scene.scene_key)" @(
                        "tools\\prepare_rotation_source.py",
                        "--source_npy", $GtRot,
                        "--out_npy", $RraaOut,
                        "--stats_json", $RraaStats,
                        "--source_label", "gt_rotation"
                    )
                }
                else {
                    $RraaArgs = @(
                        "tools\\RRAA_fast.py",
                        "--npz", $RraaInput,
                        "--out", $RraaOut,
                        "--initial_l1_iters", [string]$ParamsDoc.rraa.initial_l1_iters,
                        "--loss", [string]$ParamsDoc.rraa.loss,
                        "--a_deg", [string]$ParamsDoc.rraa.a_deg,
                        "--irls_max_iter", [string]$ParamsDoc.rraa.irls_max_iter,
                        "--qpair_threshold", [string]$ParamsDoc.qpair_threshold,
                        "--quality_config", $QualityConfig,
                        "--dump_quality_stats"
                    )
                    if ($ParamsDoc.rraa.diagnose_anchor_sensitivity) {
                        $RraaArgs += @("--diagnose_anchor_sensitivity", "--anchor_sensitivity_max_anchors", [string]$ParamsDoc.rraa.anchor_sensitivity_max_anchors)
                    }
                    if ($Group.enable_quality_weighting) {
                        $RraaArgs += "--enable_quality_weighting"
                    }
                    if ($Group.rraa_use_qpair_weight) {
                        $RraaArgs += "--rraa_use_qpair_weight"
                    }
                    Invoke-Step "$($Group.name):rraa:$($Scene.scene_key)" $RraaArgs
                }

                $PoseArgs = @(
                    "tools\\Pose_Only_patched_v3_fixed.py",
                    "--track_npz_dir", $TrackDir,
                    "--r_abs_npy", $RraaOut,
                    "--dataset", [string]$ParamsDoc.poseonly.dataset,
                    "--gt_pose_txt", $GtPoses,
                    "--out_dir", $PoseDir,
                    "--min_track_len", [string]$ParamsDoc.poseonly.min_track_len,
                    "--max_tracks", [string]$ParamsDoc.poseonly.max_tracks,
                    "--u_min", [string]$Group.u_min,
                    "--g_min", [string]$Group.g_min,
                    "--base_pair_candidates", [string]$ParamsDoc.poseonly.base_pair_candidates,
                    "--base_pair_full_search_len", [string]$ParamsDoc.poseonly.base_pair_full_search_len,
                    "--irls_iters", [string]$ParamsDoc.poseonly.irls_iters,
                    "--qtrack_mode", [string]$Group.qtrack_mode,
                    "--qtrack_threshold", [string]$ParamsDoc.qtrack_threshold,
                    "--quality_config", $QualityConfig,
                    "--auto_quality_refs",
                    "--dump_quality_stats",
                    "--dump_degeneracy_stats"
                )
                if (Test-Path $KPath) {
                    $PoseArgs += @("--K_npy", $KPath)
                }
                elseif ((Test-Path $KsPath) -and (Test-Path $ImageKIdxPath)) {
                    $PoseArgs += @("--Ks_npy", $KsPath, "--image_K_idx_npy", $ImageKIdxPath)
                }
                else {
                    throw "Missing intrinsics for $($Scene.scene_key): expected K.npy or Ks.npy + image_K_idx.npy under $PreparedSceneDir"
                }
                if ($Group.enable_quality_weighting) {
                    $PoseArgs += "--enable_quality_weighting"
                }
                if ($Group.ligt_use_qtrack_weight) {
                    $PoseArgs += "--ligt_use_qtrack_weight"
                }
                if ($Group.use_pa) {
                    $PoseArgs += @(
                        "--run_pa",
                        "--pa_loss", [string]$ParamsDoc.pa.loss,
                        "--pa_max_init_rmse", [string]$ParamsDoc.pa.max_init_rmse
                    )
                    if ($ParamsDoc.pa.use_qtrack) {
                        $PoseArgs += "--pa_use_qtrack"
                    }
                }
                Invoke-Step "$($Group.name):pose_only:$($Scene.scene_key)" $PoseArgs

                Invoke-Step "$($Group.name):eval_rotation:$($Scene.scene_key)" @(
                    "tools\\eval_rraa_rotation.py",
                    "--est_npy", $RraaOut,
                    "--gt_npy", $GtRot,
                    "--out_json", $RotationEvalJson
                )

                Invoke-Step "$($Group.name):eval_translation:$($Scene.scene_key)" @(
                    "tools\\eval_poseonly_strecha_mm.py",
                    "--est_poses", (Join-Path $PoseDir "poses_c2w.txt"),
                    "--est_type", "c2w",
                    "--gt_centers_npy", $GtCenters,
                    "--gt_unit_to_mm", $GtUnitToMm,
                    "--out_json", $TranslationEvalJson
                )

                Invoke-Step "$($Group.name):summary:$($Scene.scene_key)" @(
                    "tools\\experiment_summary.py",
                    "--run_dir", $RunDir,
                    "--track_stats_json", $TrackStatsJson,
                    "--rraa_stats_json", $RraaStats,
                    "--ligt_quality_json", (Join-Path $PoseDir "quality_stats.json"),
                    "--ligt_degeneracy_json", (Join-Path $PoseDir "ligt_degeneracy_stats.json"),
                    "--pa_degeneracy_json", (Join-Path $PoseDir "pa_degeneracy_stats.json"),
                    "--qtrack_npz", (Join-Path $PoseDir "track_quality_scores.npz"),
                    "--rraa_eval_json", $RotationEvalJson,
                    "--pose_eval_json", $TranslationEvalJson,
                    "--out_json", $SummaryJson
                )
            }
            catch {
                $Failures.Add([pscustomobject]@{
                    experiment_group = $Group.name
                    scene = $Scene.scene_key
                    message = $_.Exception.Message
                }) | Out-Null
                Write-Warning "[paper-v2][failed] $($Group.name) / $($Scene.scene_key): $($_.Exception.Message)"
            }
        }
    }
}
finally {
    Pop-Location
}

if ($Failures.Count -gt 0) {
    $Failures | Format-Table -AutoSize
    exit 1
}

Write-Host "[paper-v2] all requested runs finished." -ForegroundColor Green
