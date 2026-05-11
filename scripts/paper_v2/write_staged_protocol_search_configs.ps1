param(
    [string]$Template = "configs/paper_v2/compare12_split_v2_final_nofallback_plus_gap8.json",
    [string]$OutDir = "configs/paper_v2/staged_search",
    [string]$Scenes = "configs/paper_v2/scenes_strecha4_dtu4_dev8_v2.json"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))

function Resolve-RepoPath([string]$PathText) {
    if ([System.IO.Path]::IsPathRooted($PathText)) {
        return $PathText
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $PathText))
}

function Choice($Values) {
    return [ordered]@{
        type = "categorical"
        choices = @($Values)
    }
}

function Write-Config($Name, $OutputRoot, $Trials, $SearchSpace) {
    $cfg = Get-Content (Resolve-RepoPath $Template) -Raw | ConvertFrom-Json
    $cfg.study_name = $Name
    $cfg.output_root = $OutputRoot
    $cfg.n_trials = $Trials
    $cfg.base_validation_config.scenes = $Scenes
    $cfg.optuna_search_space = $SearchSpace

    $outPath = Resolve-RepoPath (Join-Path $OutDir "$Name.json")
    New-Item -ItemType Directory -Force -Path (Split-Path $outPath) | Out-Null
    $cfg | ConvertTo-Json -Depth 100 | Set-Content -Path $outPath -Encoding UTF8
    Write-Host "wrote $outPath"
}

function Write-ConfigWithDerived($Name, $OutputRoot, $Trials, $SearchSpace, $DerivedParams) {
    $cfg = Get-Content (Resolve-RepoPath $Template) -Raw | ConvertFrom-Json
    $cfg.study_name = $Name
    $cfg.output_root = $OutputRoot
    $cfg.n_trials = $Trials
    $cfg.base_validation_config.scenes = $Scenes
    $cfg.optuna_search_space = $SearchSpace
    $cfg | Add-Member -NotePropertyName derived_params -NotePropertyValue $DerivedParams -Force

    $outPath = Resolve-RepoPath (Join-Path $OutDir "$Name.json")
    New-Item -ItemType Directory -Force -Path (Split-Path $outPath) | Out-Null
    $cfg | ConvertTo-Json -Depth 100 | Set-Content -Path $outPath -Encoding UTF8
    Write-Host "wrote $outPath"
}

$uniform = "1:30,2:30,3:30,4:30,5:30,8:30,13:30,default:30"
$strict = "1:12,2:24,3:36,4:36,5:48,8:72,13:72,default:72"
$strictNearby = "1:12,2:24,3:32,4:32,5:44,8:68,13:68,default:68"
$relaxed = "1:12,2:24,3:30,4:30,5:40,8:60,13:60,default:60"

Write-Config `
    "stage1_graph_span_dev8_v2" `
    "runs/paper_v2/staged_search/stage1_graph_span_dev8_v2" `
    24 `
    ([ordered]@{
        deltas_tracks = Choice @("1", "1,2,3", "1,2,3,5")
        deltas_rraa = Choice @("1,2,3", "1,2,3,5", "1,2,3,5,8", "1,2,3,5,8,13")
        min_score = Choice @(0.0)
        min_inliers_tracks = Choice @(30)
        qpair_mode_tracks = Choice @("weighted_sum")
        ransac_px = Choice @(1.0)
        min_inliers_map = Choice @($uniform)
        irls_iters = Choice @(0)
        irls_huber_k = Choice @(1.5)
    })

Write-Config `
    "stage2_irls_on_A4_uniform_dev8_v2" `
    "runs/paper_v2/staged_search/stage2_irls_on_A4_uniform_dev8_v2" `
    12 `
    ([ordered]@{
        deltas_tracks = Choice @("1,2,3")
        deltas_rraa = Choice @("1,2,3,5,8")
        min_score = Choice @(0.0)
        min_inliers_tracks = Choice @(30)
        qpair_mode_tracks = Choice @("weighted_sum")
        ransac_px = Choice @(1.0)
        min_inliers_map = Choice @($uniform)
        irls_iters = Choice @(0, 1, 2)
        irls_huber_k = Choice @(1.0, 1.5, 2.0)
    })

Write-Config `
    "stage3_filtering_on_A4_irls1_dev8_v2" `
    "runs/paper_v2/staged_search/stage3_filtering_on_A4_irls1_dev8_v2" `
    24 `
    ([ordered]@{
        deltas_tracks = Choice @("1,2,3")
        deltas_rraa = Choice @("1,2,3,5,8")
        min_score = Choice @(0.0, 0.05)
        min_inliers_tracks = Choice @(20, 30)
        qpair_mode_tracks = Choice @("weighted_sum")
        ransac_px = Choice @(1.0)
        min_inliers_map = Choice @($uniform, $relaxed, $strictNearby, $strict)
        irls_iters = Choice @(1)
        irls_huber_k = Choice @(1.5)
    })

Write-ConfigWithDerived `
    "stage3_filtering_powerlaw_on_A4_irls2_dev8_v2" `
    "runs/paper_v2/staged_search/stage3_filtering_powerlaw_on_A4_irls2_dev8_v2" `
    75 `
    ([ordered]@{
        deltas_tracks = Choice @("1,2,3")
        deltas_rraa = Choice @("1,2,3,5,8")
        min_score = Choice @(0.0, 0.03, 0.05, 0.08)
        min_inliers_tracks = Choice @(20, 30)
        qpair_mode_tracks = Choice @("weighted_sum")
        ransac_px = Choice @(1.0)
        map_t0 = Choice @(8, 12, 16, 20, 24)
        map_alpha = Choice @(0.0, 0.25, 0.5, 0.75, 1.0)
        map_tmax = Choice @(48, 72, 96)
        irls_iters = Choice @(2)
        irls_huber_k = Choice @(1.5)
    }) `
    ([ordered]@{
        min_inliers_map = [ordered]@{
            type = "power_law"
            t0_param = "map_t0"
            alpha_param = "map_alpha"
            tmax_param = "map_tmax"
            t_min = 8
            floor = 8
            round_to = 6
            gaps = @(1, 2, 3, 4, 5, 8, 13)
            default_gap = 13
        }
    })
