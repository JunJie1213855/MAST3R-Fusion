# MASt3R-Fusion 架构设计与实现原理

本文从代码层面梳理 MASt3R-Fusion 的整体架构与核心实现原理，覆盖实时前端、后端因子图、V-I 初始化、回环与全局优化，以及 CUDA 后端。

> 核心思想一句话概括：**用 MASt3R 前馈网络回归的稠密 metric 点图，替代传统 SLAM 的"特征提取 + 匹配 + 三角化"，再把稠密点图的对齐误差压缩成 7×7 的 Sim(3) 相对位姿 Hessian，注入 GTSAM 因子图与 IMU 预积分紧耦合。**

---

## 1. 系统总体架构

系统分为**实时阶段**与**离线阶段**两段式流水线：

```
┌────────────────────────── 实时阶段 (main.py) ──────────────────────────┐
│                                                                        │
│ 相机帧 ──► MASt3R 编码 (feat, pos)                                      │
│   │                                                                    │
│   ├─► 前端跟踪 FrameTracker.track()                                     │
│   │      · 与上一关键帧稠密匹配 (CUDA iter_proj + 描述子细化)             │
│   │      · Sim(3) 高斯牛顿配准 → T_CkCf                                 │
│   │      · 关键帧选择                                                    │
│   │                                                                    │
│   └─► 后端 FactorGraph（滑动窗口）                                       │
│          · retrieval 检索共视帧 → 加视觉因子                             │
│          · AlignCore(CUDA) 算相对 Sim(3) Hessian → GTSAM HessianFactor   │
│          · IMU 预积分因子 + 外参因子 + 先验                               │
│          · LM 优化 + 滑窗边际化                                          │
│                                                                        │
│  输出: result.txt / graph.pkl / data.h5                                 │
└────────────────────────────────────────────────────────────────────────┘
                              │ graph.pkl + data.h5
                              ▼
┌───────────────── 离线阶段 (main_loop.py + main_global_optimization.py) ─┐
│  回环检测: RetrievalDatabase(ASMK/IVF) 检索 → MASt3R 重匹配 → loop.pkl   │
│  全局优化: 视觉因子 + IMU 因子 + 回环因子(Cauchy) + 可选 GNSS → 全局轨迹  │
└────────────────────────────────────────────────────────────────────────┘
```

主循环状态机（`frame.py:10`）：`INIT → TRACKING → RELOC → TERMINATED`。跟踪失败时进入 `RELOC`，用单帧 mono 推理重定位。

**进程模型**：`main.py` 用 `mp.set_start_method("spawn")` 启动，可视化跑在独立进程（`run_visualization`），主进程与可视化进程通过共享内存（`share_memory_()`）+ `multiprocessing.Manager` 的队列/锁通信。

---

## 2. 核心数据表示

### 2.1 Frame（`frame.py:17`）

单帧数据的最小单元：

| 字段 | 含义 |
| --- | --- |
| `feat` / `pos` | MASt3R ViT 编码的 token 与位置编码 |
| `X_canon` / `C` | 该帧稠密点图（相机系 3D 坐标）与置信度 |
| `T_WC` | `lietorch.Sim3` 世界系位姿，**含尺度**（t + R + s，共 7 维） |
| `uimg` / `img` | 原始图（CPU）与归一化图（GPU） |
| `K` | 内参（使用标定时） |
| `T_CkCf` / `ref_kf` | 相对上一关键帧的位姿与参考关键帧 id |

`update_pointmap()`（`frame.py:43`）支持多种点图融合策略，EuRoC 默认 `weighted_pointmap`（置信度加权平均），另有 `best_score` / `indep_conf` / `weighted_spherical` 等。

### 2.2 SharedKeyframes / SharedStates（`frame.py:127/222`）

跨进程共享的关键帧/状态缓冲区：

- 用 `torch.share_memory_()` 实现 GPU/CPU 张量的零拷贝共享；
- 用 `manager.RLock()` / `manager.Value` / `manager.list` 做跨进程同步（mode、paused、全局优化任务队列 `global_optimizer_tasks`、边 `edges_ii/jj` 等）；
- 关键帧缓冲固定大小 `buffer`，滑窗通过 `roll_up()`（`frame.py:334`）把整块缓冲 `torch.roll` 前移，只保留最近 buffer 个关键帧——这是"8GB 显存即可跑任意长序列"的关键。`rollup_sum` 记录已滚出的帧数，用于把窗口内索引映射回全局帧号。

### 2.3 位姿表示 Sim(3)

使用 `lietorch.Sim3`（平移 + 旋转 + 尺度）。MASt3R 输出 metric 尺度，但系统仍显式估计每帧尺度（`S(i)`），以校正长序列的尺度漂移。在进入 GTSAM 时拆解为 `Pose3`（`X(i)`）+ `double`（`S(i)`），即 `Sim3 = (R, t, s)`。

---

## 3. 前端：MASt3R 推理 + 稠密匹配 + 跟踪

### 3.1 MASt3R 前馈点图（`mast3r_utils.py`）

- 加载 `AsymmetricMASt3R` 预训练模型（`mast3r_utils.py:14`）；
- 编码：`_encode_image` → `feat`（token）、`pos`；
- 解码：`decoder`（`mast3r_utils.py:35`）→ `pts3d`（点图 X）、`conf`（C）、`desc`（D）、`desc_conf`（Q）；
- `mast3r_inference_mono`（`mast3r_utils.py:119`）：单帧自解码，用于初始化和重定位；
- `mast3r_match_symmetric/asymmetric`：两帧联合解码得到双向匹配。

### 3.2 稠密匹配（`matching.py`）

不用手工特征描述子，而是：

1. **`iter_proj`（CUDA 核）**：把源帧点图归一化成"射线图像"并配上图像梯度，从初始像素猜测出发做迭代投影匹配（`matching.py:53`）；
2. **`refine_matches`（CUDA 核）**：用 MASt3R 稠密描述子 D 做局部相关，细化到亚像素（`matching.py:90`）；
3. 距离/遮挡校验（`matching.py:78`）。

### 3.3 跟踪 `FrameTracker.track`（`tracker.py:34`）

1. 与上一关键帧 `mast3r_match_asymmetric` 得像素对应 `idx_f2k`；
2. `opt_pose_calib_sim3`（`tracker.py:232`）：**Sim(3) 高斯牛顿**最小化重投影误差，解相对位姿 `T_CkCf`（有标定用 pixel + log-depth 残差，无标定用 ray-dist 残差）；
3. Huber 鲁棒核 + Cholesky 求解（`tracker.py:172`）；
4. 关键帧选择（`tracker.py:125`）：匹配比例、唯一匹配比例、平移量三重阈值。

---

## 4. 后端：把稠密点图"压"进因子图（核心设计）

`global_opt.py` 的 `FactorGraph` 是理解整个系统的关键。

### 4.1 因子图节点（GTSAM symbol）

| 符号 | 含义 |
| --- | --- |
| `X(i)` | 相机位姿 `Pose3`（世界系） |
| `S(i)` | 尺度 `double`（与 `X` 合为 Sim(3)） |
| `Z(i)` | body/IMU 位姿 `Pose3` |
| `C(i)` | 相机-IMU 外参 `Pose3`（`enable_excalib` 时可在线估计） |
| `B(i)` / `V(i)` | IMU bias / 速度 |

### 4.2 视觉因子：稠密点图 → 7×7 Hessian

这是与普通 SLAM 最大的区别。`AlignCoreCalib`（C++/CUDA，`backend/`）+ `Align2GTSAM_factors`（`global_opt.py:33`）：

1. 对每对帧 (i,j)，`AlignCoreCalib` 在 GPU 上对**两帧点图的重投影/射线对齐误差**做高斯牛顿线性化，得到**相对 Sim(3) 的 7×7 Hessian 块 H 和 7 维梯度 v**（`hessian_pieces`）；
2. `Align2GTSAM_factors` 用 Jacobian 链式法则，把 7 维相对位姿的 H 展开成对两个绝对位姿 `[S(i),X(i),S(j),X(j)]` 的 14×14 Hessian，构造成 `HessianFactor`/`LinearContainerFactor`。

**意义**：普通 SLAM 中每个 3D 点/每个重投影残差都是一个因子；这里成千上万个稠密点在一次 CUDA 高斯牛顿中被折叠成每边一个 7×7 矩阵，GTSAM 中几乎没有逐点因子，后端因此极快。

### 4.3 IMU / 外参 / 先验因子

- `CombinedImuFactor`：GTSAM 预积分因子（`global_opt.py:795`）；
- `ExPoseConstraintFactor`：`Z = X · C⁻¹` 联系 body 与相机（`global_opt.py:810`）；
- bias 先验（`PriorFactorConstantBias`）、外参先验（`PriorFactorPose3`）、首帧位姿/尺度先验。

### 4.4 滑动窗口与边际化

`solve_GN_calib`（`global_opt.py:506`）每轮：重新线性化视觉因子 → 组图 → `LevenbergMarquardtOptimizer` → 回写位姿。滑出窗口的老帧用 `gtsam.marginalizeOut` 边际化（`global_opt.py:678`），结果作为 `marg_factor` 先验保留，实现固定窗口、长序列一致性。窗口大小由 `ms_opt.window_num` 控制（EuRoC 为 10）。

---

## 5. V-I 初始化（`vio_utils.py`）

`solve_VI_init`（`global_opt.py:878`）在累计约 7 个关键帧后触发，参照 VINS-Fusion：

1. **陀螺仪 bias**：旋转约束最小二乘（`vio_utils.py:58`）；
2. **线性对齐**：联立求解尺度 s、重力 g、速度 v（`vio_utils.py:104`）；
3. **重力精化**：在重力切平面迭代精化方向（`vio_utils.py:147`）；
4. **g2R**：把重力旋转到世界系 z 轴（`vio_utils.py:215`）。

成功后 `enable_ms=True`，后端进入"metric-scale + IMU 紧耦合"模式（`global_opt.py:952`）。

---

## 6. 回环检测与全局优化

### 6.1 检索式回环（`retrieval_database.py` + `main_loop.py`）

- `RetrievalDatabase` 基于 MASt3R 的 **retrieval 模型（ASMK/IVF 倒排索引）**，把每帧局部描述子量化入库、查询相似帧（`retrieval_database.py:43`）；
- 实时阶段 `run_backend`（`main.py:44`）在窗口内检索共视帧；离线 `main_loop.py` 从 `data.h5` 重载关键帧，对检索到的候选对做 `mast3r_match_symmetric` 精确重匹配，生成回环视觉因子存 `loop.pkl`。

### 6.2 全局优化（`main_global_optimization.py`）

- 加载 `graph.pkl`（实时阶段保存的视觉 Hessian + 位姿/bias 参数）与 `loop.pkl`（回环因子）；
- 重建全图：视觉因子 + IMU 因子 + 外参 + **回环因子**（`ExPoseConstraintFactor`，先宽松后鲁棒 Cauchy 核，多轮迭代剔除 outlier，`main_global_optimization.py:441`）；
- 可选 GNSS 因子（`GPSFactorLever`，`main_global_optimization.py:529`），用 IMU 预积分桥接关键帧与 GNSS 观测；
- 输出全局一致轨迹 `result_global.txt`。

---

## 7. CUDA 后端（`backend/`）

| 文件 | 职责 |
| --- | --- |
| `src/gn_kernels.cu` | `AlignCoreCalib` 的高斯牛顿：对两帧点图对齐误差线性化，输出相对 Sim(3) 的 Hessian/梯度 |
| `src/matching_kernels.cu` | `iter_proj`（迭代投影匹配）与 `refine_matches`（描述子相关亚像素细化） |
| `include/aligncore.h` / `gn.h` | 后端接口声明 |

后端通过 `setup.py` 编译成 `mast3r_fusion_backends` 扩展，同时暴露 CPU（`gauss_newton_calib`）与 CUDA（`gauss_newton_calib_cuda`）两个实现。

---

## 8. 模块与文件索引

| 文件 | 职责 |
| --- | --- |
| `main.py` | 实时 SLAM 主循环、三状态机、后端调度 |
| `tracker.py` | 前端跟踪、Sim(3) 高斯牛顿配准、关键帧选择 |
| `mast3r_utils.py` | MASt3R 模型加载、编码/解码、匹配包装、图像 resize |
| `matching.py` | 稠密匹配（调用 CUDA 核） |
| `global_opt.py` | `FactorGraph`、视觉 Hessian → GTSAM 因子、滑窗边际化、V-I 初始化入口 |
| `vio_utils.py` | V-I 初始化（陀螺仪 bias、尺度/重力/速度对齐） |
| `frame.py` | `Frame`、`SharedKeyframes`/`SharedStates`、点图融合 |
| `dataloader.py` | 数据集加载、`Intrinsics`（去畸变/标定） |
| `retrieval_database.py` | ASMK/IVF 检索式回环数据库 |
| `geometry.py` / `image.py` | 几何工具、图像梯度 |
| `lietorch_utils.py` / `tictoc.py` | Lie 群工具、计时 |
| `geoFunc/` | IMU 数据池（`data_utils.IMUPool`）、坐标/姿态变换（`trans`） |
| `evaluate.py` | 结果输出（轨迹/reconstruction） |
| `visualization.py` | OpenGL 点图/surfel 可视化进程 |
| `main_loop.py` | 离线回环检测 |
| `main_global_optimization.py` | 离线全局优化（回环 + 可选 GNSS） |
| `backend/` | C++/CUDA 高斯牛顿与匹配核 |

---

## 9. 与传统 SLAM 的对比

| 环节 | 传统 SLAM | MASt3R-Fusion |
| --- | --- | --- |
| 前端特征 | 手工特征点（ORB/SIFT）+ 描述子匹配 | MASt3R 回归的**稠密点图 + 射线匹配 + 描述子细化** |
| 深度/尺度 | 三角化或深度图 | 前馈网络直接回归 **metric 尺度**点图 |
| 位姿表示 | SE(3) | **Sim(3)**（显式估计尺度以校正漂移） |
| 视觉因子 | 逐点重投影因子 | **CUDA 高斯牛顿折叠成 7×7 相对 Hessian** |
| 回环 | BoW + 特征重匹配 | **MASt3R retrieval（ASMK/IVF）+ 点图重匹配** |
| 融合 | 视惯紧耦合 | 同构 GTSAM 因子图，可扩展 GNSS |

---

## 10. 关键配置（`config/base_euroc.yaml`）

- `ms_opt.window_num: 10` —— 滑动窗口大小（保留 10 帧 + 边际化）；
- `ms_opt.imu_format: euroc` —— IMU 数据格式；
- `ms_opt.vi_init_repropagate_gyro_bias: True` —— VI 初始化后重传陀螺仪 bias；
- `tracking` / `local_opt` —— 前端与后端各自的残差 sigma、置信度阈值、迭代次数；
- `retrieval.k / min_thresh` —— 检索 top-k 与相似度阈值。
