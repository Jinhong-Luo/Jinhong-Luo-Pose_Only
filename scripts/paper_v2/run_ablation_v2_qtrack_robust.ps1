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
            "-ExperimentsConfig", "configs/paper_v2/experiments_ablation_v2_qtrack_robust.json",
            "-ParamsConfig", $ParamsConfig
        )
        if ($OnlyScenes.Count -gt 0) {
            $RunArgs += @("-OnlyScenes", $OnlyScenes)
        }
        & powershell @RunArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during qtrack-focused robust ablation runs."
        }
    }

    if (-not $SkipTable) {
        & $PythonExe tools/generate_single_ablation_table.py `
            --run_root runs/paper_v2/ablation_v2/qtrack_robust `
            --out_csv runs/paper_v2/ablation_v2/tables/table_qtrack_robust_ablation.csv `
            --groups T0_robust_no_quality,T1_threshold_only,T2_weight_only,T3_weighted_sum_full,T4_product_full,T5_log_additive_full `
            --labels T0,T1,T2,T3,T4,T5
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during qtrack-focused robust ablation table generation."
        }
    }
}
finally {
    Pop-Location
}

Write-Host "[ablation-v2-qtrack-robust] all requested steps finished." -ForegroundColor Green
