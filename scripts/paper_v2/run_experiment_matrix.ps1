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

function Normalize-NameFilter([string[]]$Values) {
    $Normalized = New-Object System.Collections.Generic.List[string]
    foreach ($Value in $Values) {
        if ([string]::IsNullOrWhiteSpace($Value)) {
            continue
        }
        foreach ($Part in ($Value -split ",")) {
            $Item = $Part.Trim()
            if ([string]::IsNullOrWhiteSpace($Item)) {
                continue
            }
            if (-not $Normalized.Contains($Item)) {
                $Normalized.Add($Item) | Out-Null
            }
        }
    }
    return @($Normalized)
}

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

function Get-GroupOrDefault($Group, [string]$Key, $DefaultValue) {
    if ($null -ne $Group.PSObject.Properties[$Key]) {
        return $Group.$Key
    }
    return $DefaultValue
}

function Save-StepMetric([string]$RunDir, [string]$StepName, [hashtable]$Metric) {
    if ([string]::IsNullOrWhiteSpace($RunDir) -or [string]::IsNullOrWhiteSpace($StepName)) {
        return
    }
    $MetricsPath = Join-Path $RunDir "step_metrics.json"
    $Existing = @{}
    if (Test-Path $MetricsPath) {
        try {
            $Raw = Get-Content $MetricsPath -Raw
            if (-not [string]::IsNullOrWhiteSpace($Raw)) {
                $Loaded = $Raw | ConvertFrom-Json
                if ($Loaded) {
                    foreach ($Prop in $Loaded.PSObject.Properties) {
                        $Existing[$Prop.Name] = $Prop.Value
                    }
                }
            }
        }
        catch {
        }
    }
    $Existing[$StepName] = $Metric
    $Existing | ConvertTo-Json -Depth 8 | Set-Content -Path $MetricsPath -Encoding UTF8
}

function Quote-CommandArg([string]$Arg) {
    if ($null -eq $Arg) {
        return '""'
    }
    if ($Arg -notmatch '[\s"]') {
        return $Arg
    }
    $Escaped = $Arg -replace '(\\*)"', '$1$1\"'
    $Escaped = $Escaped -replace '(\\+)$', '$1$1'
    return '"' + $Escaped + '"'
}

function Invoke-Step([string]$Label, [string[]]$CommandArgs, [string]$RunDir = $null, [string]$StepName = $null) {
    Write-Host "[paper-v2][$Label] $($CommandArgs -join ' ')" -ForegroundColor Cyan
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $PythonExe
    $psi.WorkingDirectory = $ProjectRoot
    $psi.UseShellExecute = $false
    $psi.Arguments = (($CommandArgs | ForEach-Object { Quote-CommandArg ([string]$_) }) -join " ")
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    $startedAt = [DateTimeOffset]::UtcNow
    [void]$proc.Start()
    $peakWorkingSetBytes = 0
    while (-not $proc.HasExited) {
        try {
            $proc.Refresh()
            if ($proc.PeakWorkingSet64 -gt $peakWorkingSetBytes) {
                $peakWorkingSetBytes = $proc.PeakWorkingSet64
            }
        }
        catch {
        }
        Start-Sleep -Milliseconds 100
    }
    $proc.WaitForExit()
    try {
        $proc.Refresh()
        if ($proc.PeakWorkingSet64 -gt $peakWorkingSetBytes) {
            $peakWorkingSetBytes = $proc.PeakWorkingSet64
        }
    }
    catch {
    }
    $exitCode = $proc.ExitCode
    if (-not [string]::IsNullOrWhiteSpace($RunDir) -and -not [string]::IsNullOrWhiteSpace($StepName)) {
        Save-StepMetric $RunDir $StepName @{
            label = $Label
            started_at_utc = $startedAt.ToString("o")
            finished_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
            elapsed_sec = [Math]::Round(([DateTimeOffset]::UtcNow - $startedAt).TotalSeconds, 6)
            peak_working_set_mb = [Math]::Round(($peakWorkingSetBytes / 1MB), 3)
            exit_code = $exitCode
            command = @($CommandArgs)
        }
    }
    if ($exitCode -ne 0) {
        throw "Step failed: $Label"
    }
}

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\\.."))
$PythonExe = Resolve-RepoPath ".venv\\Scripts\\python.exe"

$ScenesDoc = Get-Content (Resolve-RepoPath $ScenesConfig) -Raw | ConvertFrom-Json
$ExperimentsDoc = Get-Content (Resolve-RepoPath $ExperimentsConfig) -Raw | ConvertFrom-Json
$ParamsDoc = Get-Content (Resolve-RepoPath $ParamsConfig) -Raw | ConvertFrom-Json
$OutputRootBase = Resolve-RepoPath $ExperimentsDoc.output_root_base
$ExperimentsQualityConfigOverride = $null
if ($null -ne $ExperimentsDoc.PSObject.Properties["quality_config_override"]) {
    $ExperimentsQualityConfigOverride = [string]$ExperimentsDoc.quality_config_override
}
$OnlyExperiments = Normalize-NameFilter $OnlyExperiments
$OnlyScenes = Normalize-NameFilter $OnlyScenes

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
            $QualityConfigCandidate = $ParamsDoc.quality_config
            if (-not [string]::IsNullOrWhiteSpace($ExperimentsQualityConfigOverride)) {
                $QualityConfigCandidate = $ExperimentsQualityConfigOverride
            }
            if ($null -ne $Group.PSObject.Properties["quality_config_override"] -and -not [string]::IsNullOrWhiteSpace([string]$Group.quality_config_override)) {
                $QualityConfigCandidate = [string]$Group.quality_config_override
            }
            $QualityConfig = Resolve-RepoPath $QualityConfigCandidate
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
                $RraaEnableQualityWeighting = Get-GroupOrDefault $Group "rraa_enable_quality_weighting" (Get-GroupOrDefault $Group "enable_quality_weighting" $false)
                $PoseEnableQualityWeighting = Get-GroupOrDefault $Group "pose_enable_quality_weighting" (Get-GroupOrDefault $Group "enable_quality_weighting" $false)
                if ($Group.rotation_source -eq "GT") {
                    Invoke-Step "$($Group.name):prepare_rotation:$($Scene.scene_key)" @(
                        "tools\\prepare_rotation_source.py",
                        "--source_npy", $GtRot,
                        "--out_npy", $RraaOut,
                        "--stats_json", $RraaStats,
                        "--source_label", "gt_rotation"
                    ) $RunDir "prepare_rotation"
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
                    if ($RraaEnableQualityWeighting) {
                        $RraaArgs += "--enable_quality_weighting"
                    }
                    if ($Group.rraa_use_qpair_weight) {
                        $RraaArgs += "--rraa_use_qpair_weight"
                    }
                    Invoke-Step "$($Group.name):rraa:$($Scene.scene_key)" $RraaArgs $RunDir "rraa"
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
                    "--base_pair_candidates", [string](Get-GroupOrDefault $Group "base_pair_candidates" $ParamsDoc.poseonly.base_pair_candidates),
                    "--base_pair_full_search_len", [string](Get-GroupOrDefault $Group "base_pair_full_search_len" $ParamsDoc.poseonly.base_pair_full_search_len),
                    "--irls_iters", [string](Get-GroupOrDefault $Group "irls_iters" $ParamsDoc.poseonly.irls_iters),
                    "--qtrack_mode", [string]$Group.qtrack_mode,
                    "--qtrack_threshold", [string](Get-GroupOrDefault $Group "qtrack_threshold" $ParamsDoc.qtrack_threshold),
                    "--qtrack_prefilter_mode", [string](Get-GroupOrDefault $Group "qtrack_prefilter_mode" "none"),
                    "--qtrack_prefilter_value", [string](Get-GroupOrDefault $Group "qtrack_prefilter_value" 0.0),
                    "--qtrack_prefilter_topk", [string](Get-GroupOrDefault $Group "qtrack_prefilter_topk" 0),
                    "--qtrack_prefilter_min_kept_tracks", [string](Get-GroupOrDefault $Group "qtrack_prefilter_min_kept_tracks" 0),
                    "--qtrack_prefilter_rollback_min_selected_ratio", [string](Get-GroupOrDefault $Group "qtrack_prefilter_rollback_min_selected_ratio" 0.0),
                    "--qtrack_prefilter_rollback_min_equation_ratio", [string](Get-GroupOrDefault $Group "qtrack_prefilter_rollback_min_equation_ratio" 0.0),
                    "--quality_config", $QualityConfig,
                    "--auto_quality_refs",
                    "--dump_quality_stats",
                    "--dump_degeneracy_stats"
                )
                $BasePairMinGap = Get-GroupOrDefault $Group "base_pair_min_gap" $ParamsDoc.poseonly.base_pair_min_gap
                $BasePairMaxGap = Get-GroupOrDefault $Group "base_pair_max_gap" $ParamsDoc.poseonly.base_pair_max_gap
                if ($null -ne $BasePairMinGap) {
                    $PoseArgs += @("--base_pair_min_gap", [string]$BasePairMinGap)
                }
                if ($null -ne $BasePairMaxGap) {
                    $PoseArgs += @("--base_pair_max_gap", [string]$BasePairMaxGap)
                }
                if (Get-GroupOrDefault $Group "enable_degeneracy_guard" $ParamsDoc.poseonly.enable_degeneracy_guard) {
                    $PoseArgs += "--enable_degeneracy_guard"
                }
                if (Get-GroupOrDefault $Group "pa_skip_on_degenerate" $ParamsDoc.poseonly.pa_skip_on_degenerate) {
                    $PoseArgs += "--pa_skip_on_degenerate"
                }
                if (Test-Path $KPath) {
                    $PoseArgs += @("--K_npy", $KPath)
                }
                elseif ((Test-Path $KsPath) -and (Test-Path $ImageKIdxPath)) {
                    $PoseArgs += @("--Ks_npy", $KsPath, "--image_K_idx_npy", $ImageKIdxPath)
                }
                else {
                    throw "Missing intrinsics for $($Scene.scene_key): expected K.npy or Ks.npy + image_K_idx.npy under $PreparedSceneDir"
                }
                if ($PoseEnableQualityWeighting) {
                    $PoseArgs += "--enable_quality_weighting"
                }
                if ($Group.ligt_use_qtrack_weight) {
                    $PoseArgs += "--ligt_use_qtrack_weight"
                }
                if ((Get-GroupOrDefault $Group "qtrack_prefilter_only" $false)) {
                    $PoseArgs += "--qtrack_prefilter_only"
                }
                if ($Group.use_pa) {
                    $PaUseQtrack = Get-GroupOrDefault $Group "pa_use_qtrack" $ParamsDoc.pa.use_qtrack
                    $PoseArgs += @(
                        "--run_pa",
                        "--pa_loss", [string](Get-GroupOrDefault $Group "pa_loss" $ParamsDoc.pa.loss),
                        "--pa_max_init_rmse", [string](Get-GroupOrDefault $Group "pa_max_init_rmse" $ParamsDoc.pa.max_init_rmse),
                        "--pa_max_step_t", [string](Get-GroupOrDefault $Group "pa_max_step_t" 0.0),
                        "--pa_max_step_r_deg", [string](Get-GroupOrDefault $Group "pa_max_step_r_deg" 0.0)
                    )
                    if ($PaUseQtrack) {
                        $PoseArgs += "--pa_use_qtrack"
                    }
                    if ((Get-GroupOrDefault $Group "pa_accept_any_update" $false)) {
                        $PoseArgs += "--pa_accept_any_update"
                    }
                }
                Invoke-Step "$($Group.name):pose_only:$($Scene.scene_key)" $PoseArgs $RunDir "pose_only"

                Invoke-Step "$($Group.name):eval_rotation:$($Scene.scene_key)" @(
                    "tools\\eval_rraa_rotation.py",
                    "--est_npy", $RraaOut,
                    "--gt_npy", $GtRot,
                    "--out_json", $RotationEvalJson
                ) $RunDir "eval_rotation"

                Invoke-Step "$($Group.name):eval_translation:$($Scene.scene_key)" @(
                    "tools\\eval_poseonly_strecha_mm.py",
                    "--est_poses", (Join-Path $PoseDir "poses_c2w.txt"),
                    "--est_type", "c2w",
                    "--gt_centers_npy", $GtCenters,
                    "--gt_unit_to_mm", $GtUnitToMm,
                    "--out_json", $TranslationEvalJson
                ) $RunDir "eval_translation"

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
                ) $RunDir "summary"
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
