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
    Invoke-Study "configs/paper_v2/holdout4_compare_optuna_best_strict.json"
}

if (-not $SkipRecommenderTopStrict) {
    Invoke-Study "configs/paper_v2/holdout4_compare_recommender_top_strict.json"
}

if (-not $SkipOptunaBestStrictNearby) {
    Invoke-Study "configs/paper_v2/holdout4_compare_optuna_best_strict_nearby.json"
}

Write-Host "holdout4 final compare finished."
Write-Host "Outputs:"
Write-Host "  runs/paper_v2/holdout4_final_compare/optuna_best_strict"
Write-Host "  runs/paper_v2/holdout4_final_compare/recommender_top_strict"
Write-Host "  runs/paper_v2/holdout4_final_compare/optuna_best_strict_nearby"
