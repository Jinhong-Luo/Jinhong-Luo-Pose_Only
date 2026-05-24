param(
    [int]$Window = 4,
    [string]$StudyDir = "runs/paper_v2/staged_search/stage2_irls_on_A4_uniform_dev8_v2",
    [string]$Title = "Stage 2: LiGT IRLS search on fixed A4 uniform graph"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PlotScript = Join-Path $ProjectRoot "tools\plot_stage2_irls_search.py"
$TrialsCsv = Join-Path $ProjectRoot (Join-Path $StudyDir "trials_summary.csv")
$OutDir = Join-Path $ProjectRoot (Join-Path $StudyDir "figures")

& $PythonExe $PlotScript `
    --trials_csv $TrialsCsv `
    --out_dir $OutDir `
    --title $Title `
    --window $Window

if ($LASTEXITCODE -ne 0) {
    throw "Failed to plot Stage 2 IRLS search."
}
