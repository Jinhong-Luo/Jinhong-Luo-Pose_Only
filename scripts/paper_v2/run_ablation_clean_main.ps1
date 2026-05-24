param(
  [string]$Config = "configs\paper_v2\experiments_ablation_clean_main.json",
  [ValidateSet("reuse", "resume", "full")]
  [string]$ReuseMode = "reuse",
  [switch]$IncludeOptional,
  [string]$OnlyRuns = "",
  [string]$OnlyTables = "",
  [switch]$SkipRun,
  [switch]$SkipTables,
  [switch]$PlotSearch,
  [int]$Stage1Window = 6,
  [int]$Stage2Window = 4
)

$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"

if (-not $SkipRun) {
  $cmd = @(
    "tools\run_ablation_paper_final.py",
    "--config", $Config,
    "--reuse-mode", $ReuseMode
  )
  if ($IncludeOptional) {
    $cmd += "--include-optional"
  }
  if ($OnlyRuns -ne "") {
    $cmd += @("--only-runs", $OnlyRuns)
  }
  if ($OnlyTables -ne "") {
    $cmd += @("--only-tables", $OnlyTables)
  }
  & $python @cmd
}

if (-not $SkipTables) {
  & $python tools\generate_ablation_clean_tables.py --config $Config
}

if ($PlotSearch) {
  powershell -ExecutionPolicy Bypass -File scripts\paper_v2\plot_stage1_graph_span_search.ps1 -Window $Stage1Window
  powershell -ExecutionPolicy Bypass -File scripts\paper_v2\plot_stage2_irls_search.ps1 -Window $Stage2Window
}
