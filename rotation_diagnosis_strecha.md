# Strecha Rotation Diagnosis

## 结论

本轮排查后，当前 Strecha 场景的核心结论如下：

1. `recoverPose` 输出的相对旋转本身没有明显崩坏。
2. 之前看到的 `~60 deg` 旋转误差，主因不是 `RRAA_fast.py`，而是 Strecha GT 预处理与命名约定存在问题。
3. `tools/strecha_prepare.py` 原先把 `.camera` 文件里的旋转矩阵读错了一行，导致旧版 `R_abs_gt_w2c.npy` 实际不是合法旋转。
4. 修复 `.camera` 解析并重新生成 `data/prepared/strecha/fountain-P11/` 后，当前 `RRAA` 旋转评估已经恢复到正常水平。

## 这次修复了什么

- 修复了 [tools/strecha_prepare.py](/d:/Program/PyCharmMiscProject/Pose_estimation/tools/strecha_prepare.py) 对 Strecha `.camera` 文件的解析。
  - `.camera` 中在 `K` 和 `R` 之间存在一行 `0 0 0` 占位行。
  - 旧代码错误地把这行占位行读进了 `R`。
- 现在会同时保存：
  - `R_abs_gt_c2w.npy`
  - `R_abs_gt_w2c.npy`
  - `gt_poses_c2w.txt`
  - `gt_poses_w2c.txt`

## 关键验证结果

### 1. 修复前 GT 旋转有问题

旧解析下，所谓 `R_abs_gt_w2c.npy` 的矩阵并不是合法旋转：

- `det(R)` 接近 `0`
- `R R^T` 不接近单位阵

这会直接污染：

- `eval_rraa_rotation.py`
- `PoseOnly` 使用 GT rotation 的基线
- 后续任何“和 GT 旋转对齐”的诊断

### 2. 修复后 GT 旋转正常

重新运行：

```powershell
.\.venv\Scripts\python.exe tools/strecha_prepare.py `
  --scene_dir data/raw/strecha/fountain-P11 `
  --out_dir data/prepared/strecha/fountain-P11
```

后，`R_abs_gt_w2c.npy` 满足：

- `det(R)` 约为 `1`
- `R R^T` 误差约为 `1e-6`

### 3. 当前 RRAA 旋转精度其实是正常的

重新评估：

```powershell
.\.venv\Scripts\python.exe tools/eval_rraa_rotation.py `
  --est_npy runs/strecha/fountain-P11/rraa_output/R_abs_fountain-P11.npy `
  --gt_npy data/prepared/strecha/fountain-P11/R_abs_gt_w2c.npy
```

得到：

- 最优约定：`est + right gauge`
- `median ≈ 0.202 deg`

这说明：

- 当前 `RRAA_fast.py` 输出的绝对旋转是正常的
- 之前的 `60 deg` 误差结论已经失效

## 对四个问题的结论

### 1. `recoverPose` 得到的 `R` 与当前 GT 的定义是否来自不同相机坐标系

结论：`recoverPose` 本身没有明显方向错误；真正的问题主要是 Strecha GT 预处理旧版本错了。

在修复后的 GT 下，`generate_rraa_input.py` 产生的 pairwise `Rij` 与 GT 相对旋转

```text
Rij ≈ R_j * R_i^T
```

对齐良好，整体中位误差约：

- `0.41 deg`

因此，当前 `recoverPose -> RRAA` 方向定义是自洽的。

### 2. 是否需要对 Strecha 数据走不同的归一化 / 去畸变 / 相机模型分支

结论：目前不需要额外切到 EuRoC 风格去畸变分支。

原因：

- Strecha 当前使用的是固定内参 `K`
- `intrinsics.json` 中没有显式畸变参数
- `.camera` 文件中给出的 `K` 已经足够用于当前 `custom + K.npy` 分支

因此当前推荐继续使用：

- `--dataset custom`
- `--K_npy data/prepared/strecha/<scene>/K.npy`

而不是：

- `--dataset euroc`
- `--euroc_undistort`

### 3. `findEssentialMat + recoverPose` 在当前场景上是否稳定

结论：稳定。

按修复后的 GT 统计，当前 `rraa_input` 中 pairwise `Rij` 的误差分布如下：

- `delta=1`: median `0.321 deg`
- `delta=2`: median `0.440 deg`
- `delta=3`: median `0.379 deg`
- `delta=5`: median `0.545 deg`

整体都在 `1 deg` 左右以内，说明：

- 这条 `findEssentialMat + recoverPose` 链路在当前 Strecha 场景上是可用的
- 不是当前主瓶颈

### 4. 是否要把 `delta=5` 这类远距离 pair 从旋转图里单独筛掉

结论：从当前统计看，`delta=5` 略差，但没有差到必须删除。

当前 `delta=5`：

- median `0.545 deg`
- max `1.129 deg`

与 `delta=1/2/3` 相比确实略高，但仍处于合理范围。

因此建议：

- 暂时不用因为“旋转不准”而强行删除 `delta=5`
- 如果后续要做保守实验，可以单独跑一版：

```powershell
--deltas 1,2,3
```

然后比较：

- `eval_rraa_rotation.py`
- `PoseOnly` 平移误差

## 对主流程的直接影响

修复 GT 后，当前更可信的结论是：

- `RRAA` 旋转已经很好
- `PoseOnly + GT rotation` 基线也明显改善

例如当前 GT rotation 基线可达到：

- `median_mm ≈ 2.37`

这说明后续若还存在平移误差，重点应更多放在：

- tracks 质量
- LiGT / PA 行为

而不是继续怀疑 `RRAA` 旋转整体方向错了
