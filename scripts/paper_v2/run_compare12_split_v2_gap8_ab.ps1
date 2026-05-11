param(
    [switch]$SkipBase,
    [switch]$SkipPlusGap8
)

$ErrorActionPreference = "Stop"
$python = ".\\.venv\\Scripts\\python.exe"

if (-not $SkipBase) {
    & $python tools\\optuna_validation_search.py --config configs/paper_v2/compare12_split_v2_final_nofallback.json
    if ($LASTEXITCODE -ne 0) {
        throw "Study failed: configs/paper_v2/compare12_split_v2_final_nofallback.json"
    }
}

if (-not $SkipPlusGap8) {
    & $python tools\\optuna_validation_search.py --config configs/paper_v2/compare12_split_v2_final_nofallback_plus_gap8.json
    if ($LASTEXITCODE -ne 0) {
        throw "Study failed: configs/paper_v2/compare12_split_v2_final_nofallback_plus_gap8.json"
    }
}

Write-Host "compare12 split_v2 gap8 A/B finished."
Write-Host "Outputs:"
Write-Host "  runs/paper_v2/compare12_split_v2_final_nofallback"
Write-Host "  runs/paper_v2/compare12_split_v2_final_nofallback_plus_gap8"
