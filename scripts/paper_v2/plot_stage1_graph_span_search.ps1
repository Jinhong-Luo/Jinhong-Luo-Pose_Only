param(
    [int]$Window = 5,
    [string]$StudyDir = "runs/paper_v2/staged_search/stage1_graph_span_dev8_v2",
    [string]$Title = "Stage 1: graph-span protocol search on dev8_v2"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PlotScript = Join-Path $ProjectRoot "tools\plot_stage1_graph_span_search.py"
$TrialsCsv = Join-Path $ProjectRoot (Join-Path $StudyDir "trials_summary.csv")
$OutDir = Join-Path $ProjectRoot (Join-Path $StudyDir "figures")

& $PythonExe $PlotScript `
    --trials_csv $TrialsCsv `
    --out_dir $OutDir `
    --title $Title `
    --window $Window

if ($LASTEXITCODE -ne 0) {
    throw "Failed to plot Stage 1 graph-span search."
}
