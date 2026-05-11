param(
    [switch]$SkipOptunaBest,
    [switch]$SkipDevBest
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
if (-not (Test-Path $python)) {
    throw "Python venv not found: $python"
}

function Invoke-Validation {
    param(
        [Parameter(Mandatory = $true)][string]$ConfigPath
    )

    & $python "tools\\optuna_validation_search.py" "--config" $ConfigPath
    if ($LASTEXITCODE -ne 0) {
        throw "Validation failed: $ConfigPath"
    }
}

if (-not $SkipOptunaBest) {
    Invoke-Validation -ConfigPath "configs/paper_v2/holdout4_v2_compare_optuna_best_nofallback.json"
}

if (-not $SkipDevBest) {
    Invoke-Validation -ConfigPath "configs/paper_v2/holdout4_v2_compare_devbest_nofallback.json"
}

