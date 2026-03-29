param()

$ErrorActionPreference = "Stop"

function Write-Ok($msg) {
    Write-Host "[OK] $msg" -ForegroundColor Green
}

function Write-Fail($msg) {
    Write-Host "[FAIL] $msg" -ForegroundColor Red
}

function Write-Info($msg) {
    Write-Host "[INFO] $msg" -ForegroundColor Cyan
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$failed = $false

try {
    $pyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    Write-Info "Python version: $pyVersion"
    if (-not $pyVersion.StartsWith("3.10.")) {
        Write-Fail "Expected Python 3.10.x, got $pyVersion"
        $failed = $true
    } else {
        Write-Ok "Python 3.10 detected"
    }
} catch {
    Write-Fail "python command is not available"
    exit 1
}

$importChecks = @(
    @{ Name = "numpy"; Cmd = "import numpy" },
    @{ Name = "scipy"; Cmd = "import scipy" },
    @{ Name = "cv2"; Cmd = "import cv2" },
    @{ Name = "torch"; Cmd = "import torch" },
    @{ Name = "kornia"; Cmd = "import kornia" },
    @{ Name = "yaml"; Cmd = "import yaml" },
    @{ Name = "tqdm"; Cmd = "import tqdm" }
)

foreach ($check in $importChecks) {
    try {
        python -c $check.Cmd | Out-Null
        Write-Ok "Import succeeded: $($check.Name)"
    } catch {
        Write-Fail "Import failed: $($check.Name)"
        $failed = $true
    }
}

try {
    python -c "import importlib.util, pathlib, sys; root = pathlib.Path('.').resolve(); sys.path.insert(0, str(root / 'tools')); import _bootstrap; import lightglue; print('ok')" | Out-Null
    Write-Ok "LightGlue import via _bootstrap succeeded"
} catch {
    Write-Fail "LightGlue import via _bootstrap failed"
    $failed = $true
}

$scripts = @(
    "tools/strecha_prepare.py",
    "tools/build_lightglue_tracks_npz.py",
    "tools/generate_rraa_input.py",
    "tools/RRAA_fast.py",
    "tools/Pose_Only_patched_v3_fixed.py",
    "tools/eval_rraa_rotation.py",
    "tools/eval_poseonly_strecha_mm.py"
)

foreach ($script in $scripts) {
    try {
        python $script --help | Out-Null
        Write-Ok "$script --help"
    } catch {
        Write-Fail "$script --help failed"
        $failed = $true
    }
}

if ($failed) {
    Write-Fail "Local setup check finished with failures"
    exit 1
}

Write-Ok "Local setup check passed"
