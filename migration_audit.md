# 本地迁移审计

## 结论

当前仓库已经具备本地迁移的最小主流程骨架，核心算法脚本基本可直接沿用。为了满足“现有流程不变、最小改动、先能跑通”，本次迁移应优先保证以下链路可在 Python 3.10 + CPU-only 环境下跑通：

1. `tools/build_lightglue_tracks_npz.py`
2. `tools/generate_rraa_input.py`
3. `tools/RRAA_fast.py`
4. `tools/Pose_Only_patched_v3_fixed.py`
5. `tools/eval_rraa_rotation.py`
6. `tools/eval_poseonly_strecha_mm.py`

## 必须迁移的代码目录

- `tools/`
  - 主流程全部入口都在这里。
  - `tools/_bootstrap.py` 负责让项目在仓库内找到 `third_party/LightGlue`。
  - `tools/calib_utils.py`、`tools/strecha_prepare.py` 是数据准备和标定读取的基础依赖。
- `main.py`
  - 当前不是主流程核心，但建议一并迁移，避免本地仓库不完整。

## 必须迁移的数据目录

- `data/raw/strecha/`
  - 至少要带需要运行的场景，例如 `fountain-P11`。
  - `images/`、`gt_dense_cameras/`、`K.txt`、`intrinsics.json`、`gt.abc` 都建议保留。
- `data/prepared/`
  - 如果云端已经跑过 `strecha_prepare.py`，则建议把对应场景的 `K.npy`、`R_abs_gt_w2c.npy`、`gt_centers.npy`、`gt_poses_*.txt` 一并拷回本地。
  - 如果本地重新执行 `strecha_prepare.py`，则该目录可以在本地再生成。
- `runs/`
  - 如果希望“现有流程不变、直接续跑/复现实验”，则建议把目标场景已有的 `tracks_npz/`、`rraa_input/`、`rraa_output/`、`poseonly_*` 结果一并迁回。
  - 如果只追求从头本地重跑，则 `runs/` 可为空，由本地重新生成。

## 必须迁移的第三方目录

- `third_party/LightGlue/`
  - 这是当前主流程的直接依赖。
  - 至少需要：
    - `third_party/LightGlue/lightglue/`
    - `third_party/LightGlue/pyproject.toml`
    - `third_party/LightGlue/README.md`
  - 推荐整个目录原样迁移，避免少文件导致 `pip install -e` 失败。

## 可延后迁移项

- `data/raw/RRAA_dataset/`
  - 当前目录里还是未完成下载的 `.qkdownloading` 文件。
  - 对 Strecha 本地最小跑通并非必需。
- `data/raw/Benchmarking_Camera_Calibration_2008.zip`
  - 可作为原始备份保留，但不是主流程运行必需。
- `third_party/LightGlue/demo.ipynb`
  - 体积较大，仅演示用途，可延后。
- `third_party/LightGlue/assets/`
  - 主要用于官方演示和 benchmark，可延后。
- 历史 `runs/` 大结果
  - 若空间紧张，可先只迁目标场景。
- `.idea/`、`.venv/`
  - 不建议从云端同步；本地重新创建环境更稳妥。

## 当前脚本中的硬编码路径与环境依赖

### 已发现的硬编码/旧路径痕迹

以下主要出现在脚本末尾示例命令注释中，不影响算法逻辑，但会误导本地使用：

- `tools/build_lightglue_tracks_npz.py`
  - 旧示例含 `/root/autodl-tmp/lightglue_work/...`
- `tools/generate_rraa_input.py`
  - 旧示例含 `/root/autodl-tmp/lightglue_work/...`
- `tools/strecha_prepare.py`
  - 旧示例含 `~/autodl-tmp/lightglue_work/...`
- `tools/RRAA_fast.py`
  - 旧示例含 `poseonly_work/...`
- `tools/Pose_Only_patched_v3_fixed.py`
  - 旧示例含 `poseonly_work/...`
- `tools/eval_rraa_rotation.py`
  - 旧示例含 `poseonly_work/...`
- `tools/eval_poseonly_strecha_mm.py`
  - 旧示例含 `poseonly_work/...`

### 路径组织相关点

- `tools/_bootstrap.py`
  - 通过 `Path(__file__).resolve().parents[1]` 获取仓库根目录。
  - 通过 `ROOT / "third_party" / "LightGlue"` 注入 LightGlue 路径。
  - 这类写法本身适合本地迁移，已经比绝对路径稳健。
- `tools/generate_rraa_input.py`
  - 对传入路径做了 `expanduser + abspath`，适合本地环境。
- 其余脚本的输入/输出路径大多来自命令行参数，本身没有强绑云端目录。

## 当前脚本中的平台相关点

### CPU / CUDA

- `tools/build_lightglue_tracks_npz.py`
  - 原默认值是 `--device cuda`，不适合本地 AMD CPU-only 目标。
  - 已调整为默认 `auto`，并在无 CUDA 时自动回退 `cpu`。
- `tools/generate_rraa_input.py`
  - 已支持 `--device auto|cuda|cpu`，默认 `auto`，适合本地 CPU-only。
  - 若显式写 `--device cuda` 而本机无 CUDA，会直接报错，这是合理行为。

### Windows / Linux 差异

- 路径拼接基本都走 `os.path` / `pathlib`，跨平台性尚可。
- 当前仓库没有 Linux shell 专属语法写进 Python 主流程。
- 但示例命令原先偏 Linux 风格，已在本次迁移中改成仓库相对路径示例。

## 环境依赖审计

主流程最低必要依赖大致为：

- Python 3.10
- `numpy`
- `scipy`
- `opencv-python` 或 conda `opencv`
- `pillow`
- `pyyaml`
- `tqdm`
- `torch`
- `torchvision`
- `kornia`
- `kornia_rs`

### 不确定项

- LightGlue 第一次运行 SuperPoint / LightGlue 时通常需要模型权重。
  - 如果本地环境首次运行且无法联网，可能需要提前把缓存好的权重拷过来。
  - 这一点属于迁移风险，应在本地首次验证前确认。
- 当前仓库没有锁定一份经过验证的 `requirements.txt`。
  - 因此 `environment_local.yml` 采用了“覆盖主流程的最小依赖”策略，而不是完全复刻旧环境。

## 建议的迁移优先级

1. 先迁 `tools/` + `third_party/LightGlue/` + 目标场景 `data/raw/strecha/<scene>/`
2. 创建 Python 3.10 CPU-only 环境
3. 安装 `third_party/LightGlue`
4. 运行 `scripts/check_local_setup.ps1` 或 `scripts/check_local_setup.sh`
5. 先跑 `--help` 与 `strecha_prepare.py`
6. 再跑一条最小 Strecha 链路
