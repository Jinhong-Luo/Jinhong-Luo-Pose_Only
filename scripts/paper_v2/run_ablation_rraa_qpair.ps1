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
            "-ExperimentsConfig", "configs/paper_v2/experiments_ablation_paper_v2_rraa_qpair.json",
            "-ParamsConfig", $ParamsConfig
        )
        if ($OnlyScenes.Count -gt 0) {
            $RunArgs += @("-OnlyScenes", $OnlyScenes)
        }
        & powershell @RunArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during pure RRAA qpair ablation runs."
        }
    }

    if (-not $SkipTable) {
        & $PythonExe tools/generate_single_ablation_table.py `
            --run_root runs/paper_v2/ablation_paper_v2/rraa_qpair `
            --out_csv runs/paper_v2/ablation_paper_v2/tables/table_QP_rraa_qpair_main.csv `
            --groups QP0_rraa_qpair_off,QP1_rraa_qpair_on `
            --labels QP0,QP1
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during pure RRAA qpair ablation table generation."
        }
    }
}
finally {
    Pop-Location
}

Write-Host "[ablation-rraa-qpair] all requested steps finished." -ForegroundColor Green
