param(
    [string]$ScenesConfig = "configs/paper_v2/scenes_strecha6_dtu6_12scenes.json",
    [string]$FrontendConfig = "configs/paper_v2/frontend_baseline_phase2_5_12scenes.json",
    [string[]]$OnlyScenes = @(),
    [switch]$Overwrite
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

function Invoke-Step([string]$Label, [string[]]$CommandArgs) {
    Write-Host "[phase2.5-frontend][$Label] $($CommandArgs -join ' ')" -ForegroundColor Cyan
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $PythonExe
    $psi.WorkingDirectory = $ProjectRoot
    $psi.UseShellExecute = $false
    $psi.Arguments = (($CommandArgs | ForEach-Object { Quote-CommandArg ([string]$_) }) -join " ")
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    [void]$proc.Start()
    $proc.WaitForExit()
    if ($proc.ExitCode -ne 0) {
        throw "Step failed: $Label"
    }
}

function Get-ImageGlob($Scene) {
    if ($Scene.dataset -eq "strecha") {
        $preparedLeaf = Split-Path -Leaf ([string]$Scene.prepared_scene_dir)
        return "data/raw/strecha/$preparedLeaf/images/*.jpg"
    }
    if ($Scene.dataset -eq "DTU") {
        return "data/raw/DTU/Rectified/$($Scene.scene_id)/rect_*_3_r5000.png"
    }
    throw "Unsupported dataset for image glob: $($Scene.dataset)"
}

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\\.."))
$PythonExe = Resolve-RepoPath ".venv\\Scripts\\python.exe"
$ScenesDoc = Get-Content (Resolve-RepoPath $ScenesConfig) -Raw | ConvertFrom-Json
$FrontendDoc = Get-Content (Resolve-RepoPath $FrontendConfig) -Raw | ConvertFrom-Json
$OnlyScenes = Normalize-NameFilter $OnlyScenes

Push-Location $ProjectRoot
try {
    foreach ($Scene in $ScenesDoc.scenes) {
        if ($OnlyScenes.Count -gt 0 -and $OnlyScenes -notcontains $Scene.scene_id -and $OnlyScenes -notcontains $Scene.scene_key) {
            continue
        }

        $PreparedSceneDir = Resolve-RepoPath $Scene.prepared_scene_dir
        $SceneRoot = Resolve-RepoPath $Scene.scene_root
        $FeatureCacheDir = Join-Path $SceneRoot $FrontendDoc.frontend.tracks.feature_cache_dirname
        $PairMatchCacheDir = Join-Path $SceneRoot $FrontendDoc.frontend.tracks.pair_match_cache_dirname
        $TrackDir = Join-Path $SceneRoot $FrontendDoc.frontend.tracks.output_dirname
        $RraaInputDir = Join-Path $SceneRoot $FrontendDoc.frontend.rraa_input.output_dirname
        $RraaInputPath = Join-Path $RraaInputDir ("rraa_input_{0}.npz" -f $Scene.scene_id)

        if ((-not $Overwrite) -and (Test-Path (Join-Path $TrackDir "track_build_quality_stats.json")) -and (Test-Path $RraaInputPath)) {
            Write-Host "[phase2.5-frontend][skip] $($Scene.scene_key)" -ForegroundColor Yellow
            continue
        }

        New-Item -ItemType Directory -Force $TrackDir | Out-Null
        New-Item -ItemType Directory -Force $RraaInputDir | Out-Null

        $ImageGlob = Get-ImageGlob $Scene

        Invoke-Step "tracks:$($Scene.scene_key)" @(
            "tools\\run_build_tracks_with_autok.py",
            "--prepared_scene_dir_autok", $PreparedSceneDir,
            "--dataset", "custom",
            "--image_glob", $ImageGlob,
            "--out_dir", $TrackDir,
            "--device", [string]$FrontendDoc.frontend.tracks.device,
            "--max_kpts", [string]$FrontendDoc.frontend.tracks.max_kpts,
            "--filter_th", [string]$FrontendDoc.frontend.tracks.filter_th,
            "--mutual",
            "--min_score", [string]$FrontendDoc.frontend.tracks.min_score,
            "--feature_cache_dir", $FeatureCacheDir,
            "--pair_match_cache_dir", $PairMatchCacheDir,
            "--use_ransac",
            "--ransac_thresh", [string]$FrontendDoc.frontend.tracks.ransac_thresh,
            "--min_inliers", [string]$FrontendDoc.frontend.tracks.min_inliers,
            "--deltas", [string]$FrontendDoc.frontend.tracks.deltas,
            "--qpair_mode", [string]$FrontendDoc.frontend.tracks.qpair_mode,
            "--quality_config", (Resolve-RepoPath "configs/quality_config.template.json"),
            "--auto_quality_refs",
            "--dump_quality_stats"
        )

        Invoke-Step "rraa_input:$($Scene.scene_key)" @(
            "tools\\run_generate_rraa_input_with_autok.py",
            "--prepared_scene_dir_autok", $PreparedSceneDir,
            "--dataset", "custom",
            "--image_list", (Join-Path $PreparedSceneDir "image_list.txt"),
            "--out_npz", $RraaInputPath,
            "--device", [string]$FrontendDoc.frontend.rraa_input.device,
            "--deltas", [string]$FrontendDoc.frontend.rraa_input.deltas,
            "--max_kpts", [string]$FrontendDoc.frontend.rraa_input.max_kpts,
            "--feature_cache_dir", $FeatureCacheDir,
            "--pair_match_cache_dir", $PairMatchCacheDir,
            "--filter_th", [string]$FrontendDoc.frontend.rraa_input.filter_th,
            "--min_match_score", [string]$FrontendDoc.frontend.rraa_input.min_match_score,
            "--ransac_px", [string]$FrontendDoc.frontend.rraa_input.ransac_px,
            "--min_inliers_map", [string]$FrontendDoc.frontend.rraa_input.min_inliers_map,
            "--add_reverse",
            "--use_h_fallback",
            "--save_meta",
            "--save_names",
            "--qpair_mode", [string]$FrontendDoc.frontend.rraa_input.qpair_mode,
            "--quality_config", (Resolve-RepoPath "configs/quality_config.template.json"),
            "--auto_quality_refs",
            "--dump_quality_stats"
        )
    }
}
finally {
    Pop-Location
}

Write-Host "[phase2.5-frontend] all requested scenes finished." -ForegroundColor Green
