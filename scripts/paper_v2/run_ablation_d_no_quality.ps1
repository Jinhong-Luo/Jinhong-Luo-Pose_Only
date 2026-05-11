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
            "-ExperimentsConfig", "configs/paper_v2/experiments_ablation_v1_degeneracy_no_quality.json",
            "-ParamsConfig", $ParamsConfig
        )
        if ($OnlyScenes.Count -gt 0) {
            $RunArgs += @("-OnlyScenes", $OnlyScenes)
        }
        & powershell @RunArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during no-quality degeneracy ablation runs."
        }
    }

    if (-not $SkipTable) {
        & $PythonExe tools/generate_single_ablation_table.py `
            --run_root runs/paper_v2/ablation_v1/degeneracy_no_quality `
            --out_csv runs/paper_v2/ablation_v1/tables_v2/table_D_degeneracy_ablation_no_quality.csv `
            --groups D0_no_degeneracy,D1_geom_filter_only,D2_basepair_only,D3_robust_only,D4_full_degeneracy `
            --labels D0,D1,D2,D3,D4
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during no-quality degeneracy table generation."
        }
    }
}
finally {
    Pop-Location
}

Write-Host "[ablation-d-no-quality] all requested steps finished." -ForegroundColor Green
