param(
    [switch]$SkipFallbackOn,
    [switch]$SkipFallbackOff
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
if (-not (Test-Path $python)) {
    throw "Python venv not found: $python"
}

function Invoke-Study {
    param(
        [Parameter(Mandatory = $true)][string]$ConfigPath
    )

    & $python "tools\\optuna_validation_search.py" "--config" $ConfigPath
    if ($LASTEXITCODE -ne 0) {
        throw "Study failed: $ConfigPath"
    }
}

if (-not $SkipFallbackOn) {
    Invoke-Study -ConfigPath "configs/paper_v2/optuna_frontend_policy_dev8_v2_no_quality.json"
}

if (-not $SkipFallbackOff) {
    Invoke-Study -ConfigPath "configs/paper_v2/optuna_frontend_policy_dev8_v2_no_quality_nofallback.json"
}
