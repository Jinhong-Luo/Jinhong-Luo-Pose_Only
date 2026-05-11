param(
    [string]$ScenesConfig = "configs/paper_v2/scenes_strecha6_dtu6_12scenes.json",
    [string]$ParamsConfig = "configs/paper_v2/params_best.json",
    [string[]]$OnlyScenes = @(),
    [switch]$SkipMaterialize,
    [switch]$SkipDegeneracy,
    [switch]$SkipRefinement,
    [switch]$SkipTables
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

    if (-not $SkipDegeneracy) {
        $DegArgs = @(
            "-ExecutionPolicy", "Bypass",
            "-File", "scripts/paper_v2/run_experiment_matrix.ps1",
            "-ScenesConfig", $ScenesConfig,
            "-ExperimentsConfig", "configs/paper_v2/experiments_ablation_paper_v2_degeneracy.json",
            "-ParamsConfig", $ParamsConfig
        )
        if ($OnlyScenes.Count -gt 0) {
            $DegArgs += @("-OnlyScenes", $OnlyScenes)
        }
        & powershell @DegArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during paper-v2 degeneracy runs."
        }
    }

    if (-not $SkipRefinement) {
        $RefArgs = @(
            "-ExecutionPolicy", "Bypass",
            "-File", "scripts/paper_v2/run_experiment_matrix.ps1",
            "-ScenesConfig", $ScenesConfig,
            "-ExperimentsConfig", "configs/paper_v2/experiments_ablation_paper_v2_refinement.json",
            "-ParamsConfig", $ParamsConfig
        )
        if ($OnlyScenes.Count -gt 0) {
            $RefArgs += @("-OnlyScenes", $OnlyScenes)
        }
        & powershell @RefArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during paper-v2 refinement runs."
        }
    }

    if (-not $SkipTables) {
        & $PythonExe tools/generate_ablation_paper_v2_tables.py
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during paper-v2 ablation table generation."
        }
    }
}
finally {
    Pop-Location
}

Write-Host "[ablation-paper-v2] all requested steps finished." -ForegroundColor Green
