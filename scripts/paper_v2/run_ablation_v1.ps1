param(
    [string]$ScenesConfig = "configs/paper_v2/scenes_strecha6_dtu6_12scenes.json",
    [string]$ParamsConfig = "configs/paper_v2/params_best.json",
    [string[]]$OnlyScenes = @(),
    [switch]$SkipMaterialize,
    [switch]$SkipQuality,
    [switch]$SkipDegeneracy,
    [switch]$SkipStablePA,
    [switch]$SkipTables,
    [string]$ValidationReuseMode = "reuse"
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

    if (-not $SkipQuality) {
        $QualityArgs = @(
            "-ExecutionPolicy", "Bypass",
            "-File", "scripts/paper_v2/run_experiment_matrix.ps1",
            "-ScenesConfig", $ScenesConfig,
            "-ExperimentsConfig", "configs/paper_v2/experiments_ablation_v1_quality.json",
            "-ParamsConfig", $ParamsConfig
        )
        if ($OnlyScenes.Count -gt 0) {
            $QualityArgs += @("-OnlyScenes", $OnlyScenes)
        }
        & powershell @QualityArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during quality ablation runs."
        }
    }

    if (-not $SkipDegeneracy) {
        $DegeneracyArgs = @(
            "-ExecutionPolicy", "Bypass",
            "-File", "scripts/paper_v2/run_experiment_matrix.ps1",
            "-ScenesConfig", $ScenesConfig,
            "-ExperimentsConfig", "configs/paper_v2/experiments_ablation_v1_degeneracy.json",
            "-ParamsConfig", $ParamsConfig
        )
        if ($OnlyScenes.Count -gt 0) {
            $DegeneracyArgs += @("-OnlyScenes", $OnlyScenes)
        }
        & powershell @DegeneracyArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during degeneracy ablation runs."
        }
    }

    if (-not $SkipStablePA) {
        & $PythonExe tools/validation_search.py `
            --config configs/paper_v2/validation_ablation_v1_stable_protocol_with_pa.json `
            --reuse_mode $ValidationReuseMode
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during stable protocol with PA validation."
        }
    }

    if (-not $SkipTables) {
        & $PythonExe tools/generate_ablation_v1_tables.py
        if ($LASTEXITCODE -ne 0) {
            throw "Failed during ablation table generation."
        }
    }
}
finally {
    Pop-Location
}

Write-Host "[ablation-v1] all requested steps finished." -ForegroundColor Green
