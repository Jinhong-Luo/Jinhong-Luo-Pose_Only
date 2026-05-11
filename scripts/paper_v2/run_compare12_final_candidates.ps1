param(
    [switch]$SkipOptunaBestStrict,
    [switch]$SkipRecommenderTopStrict,
    [switch]$SkipOptunaBestStrictNearby
)

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"

function Invoke-Study($configPath) {
    & $python tools\optuna_validation_search.py --config $configPath
    if ($LASTEXITCODE -ne 0) {
        throw "Study failed: $configPath"
    }
}

if (-not $SkipOptunaBestStrict) {
    Invoke-Study "configs/paper_v2/compare12_optuna_best_strict.json"
}

if (-not $SkipRecommenderTopStrict) {
    Invoke-Study "configs/paper_v2/compare12_recommender_top_strict.json"
}

if (-not $SkipOptunaBestStrictNearby) {
    Invoke-Study "configs/paper_v2/compare12_optuna_best_strict_nearby.json"
}

Write-Host "compare12 final candidates finished."
Write-Host "Outputs:"
Write-Host "  runs/paper_v2/compare12_final_candidates/optuna_best_strict"
Write-Host "  runs/paper_v2/compare12_final_candidates/recommender_top_strict"
Write-Host "  runs/paper_v2/compare12_final_candidates/optuna_best_strict_nearby"
