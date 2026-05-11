param(
    [string]$ScenesConfig = "configs/paper_v2/scenes_strecha6_dtu6_12scenes.json",
    [string]$ParamsConfig = "configs/paper_v2/params_best.json",
    [string[]]$OnlyScenes = @(),
    [switch]$SkipMaterialize,
    [switch]$SkipRuns,
    [switch]$SkipTable
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath([string]$PathText) {
    if ([System.IO.Path]::IsPathRooted($PathText)) {
        return $PathText
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $PathText))
}

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\\.."))
$PythonExe = Resolve-RepoPath ".venv\\Scripts\\python.exe"

Push-Location $ProjectRoot
try {
    if (-not $SkipMaterialize) {
        $MaterializeArgs = @(
            "-ExecutionPolicy", "Bypass",
            "-File", "scripts/paper_v2/materialize_frontend_phase2_5_baseline.ps1",
            "-ScenesConfig", $ScenesConfig
        )
        if ($OnlyScenes.Count -gt 0) {
            $MaterializeArgs += @("-OnlyScenes", $OnlyScenes)
        }
        & powershell @MaterializeArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during frontend materialization."
        }
    }

    if (-not $SkipRuns) {
        $RunArgs = @(
            "-ExecutionPolicy", "Bypass",
            "-File", "scripts/paper_v2/run_experiment_matrix.ps1",
            "-ScenesConfig", $ScenesConfig,
            "-ExperimentsConfig", "configs/paper_v2/experiments_ablation_v4_qtrack_prefilter_guarded.json",
            "-ParamsConfig", $ParamsConfig
        )
        if ($OnlyScenes.Count -gt 0) {
            $RunArgs += @("-OnlyScenes", $OnlyScenes)
        }
        & powershell @RunArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during guarded qtrack-prefilter ablation runs."
        }
    }

    if (-not $SkipTable) {
        & $PythonExe tools/generate_single_ablation_table.py `
            --run_root runs/paper_v2/ablation_v4/qtrack_prefilter_guarded `
            --out_csv runs/paper_v2/ablation_v4/tables/table_qtrack_prefilter_guarded_ablation.csv `
            --groups V0_no_quality,V1_prefilter_quantile_002_guarded,V2_prefilter_quantile_005_guarded,V3_prefilter_quantile_008_guarded,V4_prefilter_threshold_005_guarded `
            --labels V0,V1,V2,V3,V4
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during guarded qtrack-prefilter ablation table generation."
        }
    }
}
finally {
    Pop-Location
}

Write-Host "[ablation-v4-qtrack-prefilter-guarded] all requested steps finished." -ForegroundColor Green
