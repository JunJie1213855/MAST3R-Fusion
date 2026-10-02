# MASt3R-Fusion 快速上手（EuRoC MAV 序列）

本文记录在 WSL2 + NVIDIA GPU 环境下，从安装到跑通 EuRoC 序列（V + I）并出轨迹图 / 3D 重建可视化的完整流程。

## 0. 已验证环境

| 项 | 版本 |
| --- | --- |
| 系统 | WSL2 (Ubuntu, 带 WSLg) |
| Python | 3.10.20（conda 环境 `dlpy310`） |
| PyTorch | 2.5.0+cu124 |
| CUDA | 12.4 |
| OpenCV | 4.8.0 |
| NumPy | 1.26.4 |
| evo | 1.36.5 |
| GTSAM | 修改版（`thirdparty/gtsam`） |

> 后端扩展 `mast3r_fusion_backends.cpython-310-x86_64-linux-gnu.so` 按 Python 3.10 编译。
> 若换 Python / CUDA 版本，需重新编译该扩展。

---

## 1. 安装

### 1.1 创建 conda 环境

```bash
conda create -n dlpy310 python=3.10
conda activate dlpy310
```

### 1.2 安装 PyTorch（cu124）

```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

### 1.3 安装其余依赖

```bash
pip install opencv-python==4.8.0.76 opencv-contrib-python==4.8.0.76
pip install h5py pyparsing evo matplotlib scipy natsort pyrealsense2
```

### 1.4 编译安装修改版 GTSAM

```bash
conda activate dlpy310
git clone https://github.com/yuxuanzhou97/gtsam.git thirdparty/gtsam
cd thirdparty/gtsam
mkdir build && cd build
cmake .. -DGTSAM_BUILD_PYTHON=1 -DGTSAM_PYTHON_VERSION=3.10 \
  -DPYTHON_EXECUTABLE=$(which python)
make python-install -j12
cd ../../..
```

### 1.5 安装项目与后端扩展

```bash
# 若尚未克隆（需带子模块）
git clone https://github.com/GREAT-WHU/MASt3R-Fusion.git --recursive
cd MASt3R-Fusion/

pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .
```

`setup.py` 会编译 CUDA 后端 `mast3r_fusion_backends`。确认生成：

```bash
ls mast3r_fusion_backends*.so
```

### 1.6 下载模型权重

```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

---

## 2. 两个必做的代码修复

以下两处在全新克隆上会导致运行报错，需在 `mast3r_fusion/dataloader.py` 中处理（当前仓库已修复，重新克隆时手动应用即可）。

### 2.1 torchcodec 惰性导入

`torchcodec` 只在 `MP4Dataset`（视频输入）中使用，但原代码在模块顶部导入，导致 EuRoC 图片序列也在启动时就加载 libtorchcodec（依赖 FFmpeg 的 `libavutil.so.59/58/57`），报
`RuntimeError: Could not load libtorchcodec`。

修复：删除顶部 `from torchcodec.decoders import VideoDecoder`，改为在 `MP4Dataset.__init__` 内部按需导入：

```python
# MP4Dataset.__init__ 内
from torchcodec.decoders import VideoDecoder
self.decoder = VideoDecoder(str(self.dataset_path))
```

### 2.2 cv2.imread 传入 str

`EurocDataset.rgb_files` 等存的是 `pathlib.Path`，而本环境 OpenCV 4.8.0 的 `cv2.imread` 不接受 `Path`，报
`TypeError: Can't convert object to 'str' for 'filename'`。

修复：所有 `cv2.imread(...)` 调用改为 `cv2.imread(str(self.rgb_files[idx]), ...)`（基类 `MonocularDataset.read_img` 与 `EurocDataset.read_img` 两处）。

---

## 3. EuRoC 数据准备

下载 [EuRoC MAV](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) 的 ASL 数据格式序列，解压到：

```bash
dataset/euroc/V1_01_easy/mav0/   # 保持原始 mav0 目录结构
```

目录结构应为：

```text
dataset/euroc/V1_01_easy/mav0/
├── cam0/data.csv
├── imu0/data.csv
└── state_groundtruth_estimate0/data.csv   # 评估用真值
```

> ⚠️ 注意路径是**单数 `dataset/`**，而脚本默认值写的是**复数 `datasets/`**。所以运行时必须显式传入第一个参数 `dataset/euroc`，否则报
> `Missing EuRoC sequence or MAV files`。

---

## 4. 运行（实时 VI-SLAM）

```bash
bash batch_euroc_vi.sh dataset/euroc V1_01_easy
```

- 不传序列名则跑全部 11 个序列；换 GPU 用 `GPU_ID=1`。
- 生成结果（在 `results/euroc/` 下）：
  - `result_V1_01_easy.txt` —— 实时位姿估计结果；
  - `graph_V1_01_easy.pkl` —— 因子图，供后续全局优化；
  - `data_V1_01_easy.h5` —— 关键帧信息，供可视化。

---

## 5. 实时可视化（边跑边看）

`batch_euroc_vi.sh` 写死了 `--no-viz`，所以批处理不会弹窗。要看实时可视化（OpenGL 窗口边跑边显示点云 / surfel 地图），去掉 `--no-viz` 直接运行 `main.py`：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset dataset/euroc/V1_01_easy \
    --config config/base_euroc.yaml \
    --calib config/intrinsics_euroc.yaml \
    --imu_path dataset/euroc/V1_01_easy/mav0/imu0/data.csv \
    --imu_dt 0.0 \
    --result_path results/euroc/result_V1_01_easy.txt \
    --save_h5
```

- 需在图形环境下运行（WSL2 需 WSLg，`DISPLAY=:0` 已就绪）。
- `--save_h5` 仍会保存结果，跑完可在窗口关闭后用第 7 节离线回放。
- 若只看实时窗口、不保存结果，可去掉 `--save_h5` 与 `--result_path`。



OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset dataset/difficult/ \
    --config config/base_euroc.yaml \
    --calib config/intrinsics_euroc.yaml \
    --imu_path dataset/difficult/mav0/imu0/data.csv \
    --imu_dt 0.0 \
    --result_path results/euroc/result_V1_03_difficult.txt \
    --save_h5

---

## 6. 评估（轨迹对比真值）

用 evo 计算 APE 并画 3D 轨迹对比图：

```bash
python evaluation/evaluate_euroc.py V1_01_easy \
  --dataset-root dataset/euroc \
  --results-root results/euroc
```

- 输出到 `results/euroc/evaluation/V1_01_easy/`：`ape.png`（轨迹对比图）、`estimate_aligned.tum`、`ape.zip`，终端打印 `APE RMSE=... m`，并生成 `summary.csv`。
- `--dataset-root` 必须写单数 `dataset/euroc`（脚本默认 `datasets/euroc`）。

---

## 7. 可视化（3D 重建回放）

`batch_euroc_vi.sh` 默认 `--no-viz`，但 `--save_h5` 已把每帧重建结果存进 h5，可用 OpenGL 窗口事后回放：

```bash
python evaluation/check_h5.py \
  --h5 results/euroc/data_V1_01_easy.h5 \
  --config config/base_euroc.yaml \
  --calib config/intrinsics_euroc.yaml \
  --frame_id 40
```

- 需指定 `--frame_id`（默认 240，超出 h5 帧数会 `KeyError`；V1_01_easy 约 93 帧，取中间值即可）。
- 弹出 OpenGL 窗口后可鼠标旋转/缩放查看点云。
- 若要看全局优化后的位姿，加 `--pose_file results/euroc/result_global_V1_01_easy.txt`。

---

## 8. （可选）回环 + 全局优化

实时运行结束后，先检测回环，再做全局优化：

```bash
sequence=V1_01_easy
dataset_root=dataset/euroc

# 回环检测
python main_loop.py \
  --config config/base_euroc.yaml \
  --h5_file "results/euroc/data_${sequence}.h5" \
  --loop_output "results/euroc/loop_${sequence}.pkl"

# 全局优化（VIO 因子图 + 回环约束）
python main_global_optimization.py \
  --config config/base_euroc.yaml \
  --graph_path "results/euroc/graph_${sequence}.pkl" \
  --loop_path "results/euroc/loop_${sequence}.pkl" \
  --calib_path config/intrinsics_euroc.yaml \
  --imu_path "${dataset_root}/${sequence}/mav0/imu0/data.csv" \
  --imu_dt 0.0 \
  --result_path "results/euroc/result_global_${sequence}.txt"
```

对全局优化后的轨迹评估：

```bash
python evaluation/evaluate_euroc.py V1_01_easy \
  --dataset-root dataset/euroc \
  --results-root results/euroc \
  --result-pattern 'result_global_{sequence}.txt' \
  --output-dir results/euroc/evaluation_global
```
