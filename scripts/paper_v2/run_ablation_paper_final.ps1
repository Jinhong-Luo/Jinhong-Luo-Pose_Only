param(
    [string]$Config = "configs/paper_v2/experiments_ablation_paper_final.json",
    [string[]]$OnlyRuns = @(),
    [string[]]$OnlyTables = @(),
    [switch]$IncludeOptional,
    [ValidateSet("reuse", "resume", "full")]
    [string]$ReuseMode = "reuse",
    [switch]$DryRun
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
$Driver = Resolve-RepoPath "tools\\run_ablation_paper_final.py"
$ConfigPath = Resolve-RepoPath $Config

Push-Location $ProjectRoot
try {
    $Args = @(
        $Driver,
        "--config", $ConfigPath,
        "--reuse-mode", $ReuseMode
    )
    if ($OnlyRuns.Count -gt 0) {
        foreach ($Item in $OnlyRuns) {
            $Args += @("--only-runs", $Item)
        }
    }
    if ($OnlyTables.Count -gt 0) {
        foreach ($Item in $OnlyTables) {
            $Args += @("--only-tables", $Item)
        }
    }
    if ($IncludeOptional) {
        $Args += "--include-optional"
    }
    if ($DryRun) {
        $Args += "--dry-run"
    }

    & $PythonExe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Failed during final paper ablation execution."
    }
}
finally {
    Pop-Location
}

Write-Host "[ablation-paper-final] all requested steps finished." -ForegroundColor Green
