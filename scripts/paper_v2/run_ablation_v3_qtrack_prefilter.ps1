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
            "-ExperimentsConfig", "configs/paper_v2/experiments_ablation_v3_qtrack_prefilter.json",
            "-ParamsConfig", $ParamsConfig
        )
        if ($OnlyScenes.Count -gt 0) {
            $RunArgs += @("-OnlyScenes", $OnlyScenes)
        }
        & powershell @RunArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during qtrack-prefilter ablation runs."
        }
    }

    if (-not $SkipTable) {
        & $PythonExe tools/generate_single_ablation_table.py `
            --run_root runs/paper_v2/ablation_v3/qtrack_prefilter `
            --out_csv runs/paper_v2/ablation_v3/tables/table_qtrack_prefilter_ablation.csv `
            --groups P0_no_quality,P1_prefilter_quantile_010,P2_prefilter_quantile_015,P3_prefilter_quantile_020,P4_prefilter_topk,P5_prefilter_threshold `
            --labels P0,P1,P2,P3,P4,P5
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during qtrack-prefilter ablation table generation."
        }
    }
}
finally {
    Pop-Location
}

Write-Host "[ablation-v3-qtrack-prefilter] all requested steps finished." -ForegroundColor Green
