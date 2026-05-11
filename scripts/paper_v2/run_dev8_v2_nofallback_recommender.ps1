param(
    [switch]$SkipMerge,
    [switch]$SkipTrain,
    [switch]$SkipRecommend,
    [switch]$SkipDevValidate,
    [switch]$RunHoldoutTop1
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
if (-not (Test-Path $python)) {
    throw "Python venv not found: $python"
}

$studyRoot = "runs/paper_v2/optuna_frontend_policy_dev8_v2_no_quality_nofallback"
$trialCsv = "runs/paper_v2/merged_optuna_trials_dev8_v2_no_quality_nofallback.csv"
$sceneCsv = "runs/paper_v2/merged_optuna_scenes_dev8_v2_no_quality_nofallback.csv"
$modelJson = "runs/paper_v2/frontend_param_recommender_score_models_dev8_v2_no_quality_nofallback.json"
$recommendJson = "runs/paper_v2/frontend_param_recommendations_top5_dev8_v2_no_quality_nofallback.json"
$devValidateRoot = "runs/paper_v2/recommended_top3_dev8_v2_no_quality_nofallback"
$holdoutRoot = "runs/paper_v2/final_holdout4_v2_no_quality_nofallback_eval"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string[]]$Args,
        [Parameter(Mandatory = $true)][string]$Label
    )

    Write-Host "==> $Label"
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Label"
    }
}

if (-not $SkipMerge) {
    Invoke-Step -Exe $python -Args @(
        "tools\\merge_optuna_trials_to_training_table.py",
        "--study_root", $studyRoot,
        "--out_trial_csv", $trialCsv,
        "--out_scene_csv", $sceneCsv
    ) -Label "Merge nofallback dev8_v2 trials"
}

if (-not $SkipTrain) {
    Invoke-Step -Exe $python -Args @(
        "tools\\train_frontend_param_recommender.py",
        "--trial_csv", $trialCsv,
        "--out_json", $modelJson
    ) -Label "Train recommender"
}

if (-not $SkipRecommend) {
    Invoke-Step -Exe $python -Args @(
        "tools\\recommend_frontend_candidates.py",
        "--trial_csv", $trialCsv,
        "--optuna_config", "configs/paper_v2/optuna_frontend_policy_dev8_v2_no_quality_nofallback.json",
        "--top_k", "5",
        "--out_json", $recommendJson
    ) -Label "Recommend top-5 candidates"
}

if (-not $SkipDevValidate) {
    Invoke-Step -Exe $python -Args @(
        "tools\\run_recommended_candidates_validation.py",
        "--recommendations_json", $recommendJson,
        "--optuna_config", "configs/paper_v2/optuna_frontend_policy_dev8_v2_no_quality_nofallback.json",
        "--top_k", "3",
        "--output_root", $devValidateRoot
    ) -Label "Validate top-3 on dev8_v2"
}

if ($RunHoldoutTop1) {
    Invoke-Step -Exe $python -Args @(
        "tools\\run_recommended_candidates_validation.py",
        "--recommendations_json", $recommendJson,
        "--optuna_config", "configs/paper_v2/optuna_frontend_policy_holdout4_v2_no_quality_eval.json",
        "--top_k", "1",
        "--output_root", $holdoutRoot
    ) -Label "Final holdout4_v2 top-1 evaluation"
}

