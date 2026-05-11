param(
    [switch]$SkipCompare12
)

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"

if (-not $SkipCompare12) {
    & $python tools\optuna_validation_search.py --config configs/paper_v2/compare12_split_v2_final_nofallback.json
    if ($LASTEXITCODE -ne 0) {
        throw "Study failed: configs/paper_v2/compare12_split_v2_final_nofallback.json"
    }
}

Write-Host "compare12 split_v2 final nofallback finished."
Write-Host "Output:"
Write-Host "  runs/paper_v2/compare12_split_v2_final_nofallback"
