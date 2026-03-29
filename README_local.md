# 本地迁移说明

## 1. 推荐目录结构

建议把仓库整理成下面的结构，并在仓库根目录执行所有命令：

```text
Pose_estimation/
├── tools/
├── third_party/
│   └── LightGlue/
├── data/
│   ├── raw/
│   │   └── strecha/
│   │       └── fountain-P11/
│   └── prepared/
│       └── strecha/
│           └── fountain-P11/
├── runs/
│   └── strecha/
│       └── fountain-P11/
├── scripts/
├── environment_local.yml
├── migration_audit.md
└── README_local.md
```

## 2. 需要从云端拷贝哪些文件/目录

最小必拷：

- `tools/`
- `third_party/LightGlue/`
- `data/raw/strecha/<scene>/`

建议一并拷贝：

- `data/prepared/strecha/<scene>/`
  - 若云端已准备好 `K.npy`、`R_abs_gt_w2c.npy`、`gt_centers.npy`、`gt_poses_c2w.txt`
- `runs/strecha/<scene>/`
  - 若你希望保留已有 `tracks_npz`、`rraa_input`、`rraa_output`、`poseonly_*` 中间结果

可以不急着拷：

- `.venv/`
- `.idea/`
- `data/raw/RRAA_dataset/` 下未下载完的大文件
- 旧实验无关场景

## 3. 如何创建 conda 环境

在仓库根目录执行：

```bash
conda env create -f environment_local.yml
conda activate pose_estimation_local
```

如果你本机还没有 `conda`，先安装 Miniconda 或 Anaconda，再执行上面两条命令。

## 4. 如何安装 `third_party/LightGlue`

推荐在激活环境后执行：

```bash
pip install -e ./third_party/LightGlue
```

说明：

- `environment_local.yml` 只负责主环境依赖。
- `LightGlue` 更适合单独 editable 安装，这样仓库内源码改动会直接生效。
- 如果你不想 editable 安装，当前仓库也能依靠 `tools/_bootstrap.py` 直接从 `third_party/LightGlue` 导入，但还是推荐执行上面的安装命令，兼容性更稳。

## 5. 如何验证环境是否正常

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_local_setup.ps1
```

Git Bash / Linux / WSL：

```bash
bash ./scripts/check_local_setup.sh
```

检查脚本会做这些事：

- 检查 Python 主版本是否为 3.10
- 检查 `torch / cv2 / kornia / numpy / scipy / yaml / tqdm` 能否导入
- 检查 `lightglue` 是否可导入
- 检查 `tools/_bootstrap.py` 是否可用
- 检查关键脚本的 `--help` 是否能执行

## 6. 如何先跑最小验证

建议不要一上来就跑整条链路，先做最小验证。

### 6.1 检查帮助信息

```bash
python tools/strecha_prepare.py --help
python tools/build_lightglue_tracks_npz.py --help
python tools/generate_rraa_input.py --help
python tools/RRAA_fast.py --help
python tools/Pose_Only_patched_v3_fixed.py --help
python tools/eval_rraa_rotation.py --help
python tools/eval_poseonly_strecha_mm.py --help
```

### 6.2 准备 Strecha 场景

```bash
python tools/strecha_prepare.py ^
  --scene_dir data/raw/strecha/fountain-P11 ^
  --out_dir data/prepared/strecha/fountain-P11
```

如果你不是在 Windows PowerShell / CMD，可把续行符 `^` 改成 `\`。

### 6.3 只跑评估脚本验证读写

如果 `data/prepared/strecha/fountain-P11/` 已经有 GT 文件，可以先试：

```bash
python tools/eval_rraa_rotation.py ^
  --est_npy data/prepared/strecha/fountain-P11/R_abs_gt_w2c.npy ^
  --gt_npy data/prepared/strecha/fountain-P11/R_abs_gt_w2c.npy
```

这一步主要用于验证 `numpy/scipy` 和脚本读写没问题。

## 7. 如何跑 Strecha 的一条完整链路

下面示例以 `fountain-P11` 为例，并按 CPU-only 运行。

### 7.1 预处理数据

```bash
python tools/strecha_prepare.py ^
  --scene_dir data/raw/strecha/fountain-P11 ^
  --out_dir data/prepared/strecha/fountain-P11
```

### 7.2 生成多帧 tracks

```bash
python tools/build_lightglue_tracks_npz.py ^
  --dataset custom ^
  --K_npy data/prepared/strecha/fountain-P11/K.npy ^
  --image_glob "data/raw/strecha/fountain-P11/images/*.jpg" ^
  --out_dir runs/strecha/fountain-P11/tracks_npz ^
  --device cpu ^
  --deltas 1,2,3,5 ^
  --max_kpts 2048 ^
  --filter_th 0.1 ^
  --mutual ^
  --min_score 0.2 ^
  --use_ransac ^
  --ransac_thresh 1.0 ^
  --min_inliers 50
```

说明：

- `--deltas 1,2,3,5` 会在相邻帧之外额外建立跨帧关联，通常能明显增加长轨数量。
- 如果你只想完全保持旧行为，可改回 `--deltas 1`。

### 7.3 生成 RRAA 输入

```bash
python tools/generate_rraa_input.py ^
  --dataset custom ^
  --K_npy data/prepared/strecha/fountain-P11/K.npy ^
  --img_dir data/raw/strecha/fountain-P11/images ^
  --out_npz runs/strecha/fountain-P11/rraa_input/rraa_input_fountain-P11.npz ^
  --device cpu ^
  --deltas 1,2,3,5 ^
  --max_kpts 2048 ^
  --filter_th 0.1 ^
  --ransac_px 2.0 ^
  --add_reverse ^
  --use_h_fallback ^
  --save_meta ^
  --save_names
```

### 7.4 求全局旋转

```bash
python tools/RRAA_fast.py ^
  --npz runs/strecha/fountain-P11/rraa_input/rraa_input_fountain-P11.npz ^
  --out runs/strecha/fountain-P11/rraa_output/R_abs_fountain-P11.npy ^
  --initial_l1_iters 3 ^
  --loss l1_2 ^
  --a_deg 5 ^
  --irls_max_iter 50
```

### 7.5 PoseOnly 求位姿

```bash
python tools/Pose_Only_patched_v3_fixed.py ^
  --track_npz_dir runs/strecha/fountain-P11/tracks_npz ^
  --r_abs_npy runs/strecha/fountain-P11/rraa_output/R_abs_fountain-P11.npy ^
  --dataset custom ^
  --K_npy data/prepared/strecha/fountain-P11/K.npy ^
  --out_dir runs/strecha/fountain-P11/poseonly_rraa ^
  --min_track_len 5 ^
  --max_tracks 20000 ^
  --u_min 1e-3 ^
  --g_min 1e-3 ^
  --base_pair_candidates 80 ^
  --base_pair_full_search_len 50 ^
  --irls_iters 2 ^
  --gt_pose_txt data/prepared/strecha/fountain-P11/gt_poses_c2w.txt
```

### 7.6 旋转评估

```bash
python tools/eval_rraa_rotation.py ^
  --est_npy runs/strecha/fountain-P11/rraa_output/R_abs_fountain-P11.npy ^
  --gt_npy data/prepared/strecha/fountain-P11/R_abs_gt_w2c.npy
```

### 7.7 平移评估

```bash
python tools/eval_poseonly_strecha_mm.py ^
  --est_poses runs/strecha/fountain-P11/poseonly_rraa/poses_c2w.txt ^
  --est_type c2w ^
  --gt_centers_npy data/prepared/strecha/fountain-P11/gt_centers.npy ^
  --gt_unit_to_mm 1000
```

## 8. CPU-only 下哪些命令要改成 `--device cpu`

必须显式改成 `cpu` 的主要是前端 LightGlue 相关脚本：

- `tools/build_lightglue_tracks_npz.py`
- `tools/generate_rraa_input.py`

具体就是：

```bash
--device cpu
```

说明：

- `build_lightglue_tracks_npz.py` 现在默认 `auto`，即使不写也会在无 CUDA 时落到 CPU。
- `generate_rraa_input.py` 默认也是 `auto`。
- 但为了让命令可复现、避免误解，CPU-only 迁移文档里仍建议显式写 `--device cpu`。

`tools/RRAA_fast.py`、`tools/Pose_Only_patched_v3_fixed.py`、评估脚本本身不依赖 CUDA 参数。

## 9. 常见报错和排查办法

### 报错：`No module named lightglue`

处理：

```bash
pip install -e ./third_party/LightGlue
```

如果仍失败，再确认目录存在：

```bash
dir third_party\\LightGlue
```

### 报错：`--device cuda requested but CUDA is not available`

处理：

- 把命令中的 `--device cuda` 改成 `--device cpu`
- 或者直接删掉，让脚本用默认 `auto`

### 报错：`No images found`

处理：

- 检查 `data/raw/strecha/<scene>/images/` 是否存在
- 检查图片后缀是否是 `.jpg/.png/.jpeg/.ppm`
- Windows 下如果用了通配符，确认命令里的引号没有写错

### 报错：`No .camera files found`

处理：

- 检查 `gt_dense_cameras/` 是否完整拷贝
- 确认场景目录结构没变

### 报错：`No valid edges produced. Try lowering thresholds.`

处理：

- 先确认图片顺序和 `K.npy` 是否对应当前场景
- 适当放宽：
  - `--filter_th`
  - `--min_match_score`
  - `--ransac_px`
  - `--min_inliers_map`
- CPU-only 不会直接导致这个错误，通常还是匹配/几何筛选过严

### 报错：首次运行 LightGlue 时下载权重失败

处理：

- 这是本地离线迁移常见问题
- 如果本机无法联网，需要把云端已经下载过的模型缓存一起拷到本地
- 当前仓库里没有直接保存这些权重文件，因此这一步需要你手动确认云端缓存位置

### Python 版本不是 3.10

处理：

- 当前机器默认 `python` 可能不是 3.10
- 请务必先 `conda activate pose_estimation_local`
- 再执行：

```bash
python --version
```

目标结果应为 `Python 3.10.x`
