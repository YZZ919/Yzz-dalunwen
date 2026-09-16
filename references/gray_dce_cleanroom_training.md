# 训练版灰度 DCE clean-room 记录

这份记录对应 `scripts/train_gray_dce_cleanroom.py` 的两个固定配置。它们实现了论文
对 IR-VIO 增强网络的公开描述：灰度单通道输入、8 张像素级曲线参数图、8 次递归
曲线、曲线估计下采样倍率 16。作者源码和作者 checkpoint 仍未公开，因此结果不能
标作官方 IR-VIO。

## 环境和数据

- Windows 工作区：`D:\大论文实验\IR-VIO的复现`
- WSL：Ubuntu-20.04，CUDA PyTorch `2.4.1+cu124`，CUDA 可见，设备为 RTX 4060 Laptop GPU
- AMP：CUDA float16；梯度范数裁剪 `0.1`
- 数据：可访问的 `okhater/SICE` 镜像，只取每个样本的 `low1.jpg`
- 实际规模：259 train / 47 val；统一转灰度、归一化到 `[0,1]`，训练 crop `512×512`
- 重要边界：镜像派生数据不是论文所述官方 SICE Part1 `2422/600` 拆分；没有使用 EuRoC 训练或选 epoch

数据入口和整理命令见 `references/sice_training_data.md`。官方 SICE 页面和镜像分别为：

- <https://github.com/csjcai/SICE>
- <https://huggingface.co/datasets/okhater/SICE>

## 固定训练配置

| profile | `L_spa` | `L_exp` | `L_tv(A)` | epoch | batch | Adam lr | E |
|---|---:|---:|---:|---:|---:|---:|---:|
| `official` | 1 | 10 | 200 | 100 | 8 | 1e-4 | 0.6 |
| `paper` | 1 | 1 | 20 | 100 | 8 | 1e-4 | 0.6 |

两套结果目录分别是：

- `checkpoints/gray_dce_cleanroom/gray_official_mirror/`
- `checkpoints/gray_dce_cleanroom/gray_paper_mirror/`

每个目录包含 `history.csv`、`training_config.json`、`epoch_005.pth` …
`epoch_100.pth`、`best_val.pth` 和 best 验证样本预览。best 只按 SICE validation loss
选择；两个 profile 的 loss 总尺度不同，不能用 val loss 的绝对值判断哪个更好。

| profile | best epoch | best val loss | 最后一 epoch train loss | 最后一 epoch val loss |
|---|---:|---:|---:|---:|
| `official` | 100 | 0.2315814273 | 0.1664699472 | 0.2315814273 |
| `paper` | 91 | 0.0305458146 | 0.0243373766 | 0.0307077223 |

## 11 条 EuRoC 结果

ATE 为估计轨迹与 `state_groundtruth_estimate0/data.csv` 的线性时间插值关联、SE(3)
刚体对齐后的位移 RMSE。工程成功判据为：至少 100 个匹配位姿、时间覆盖率至少 70%、
ATE 有限。以下为训练版双分支结果；基线/T.W. 取同一序列已有的 ASL 或标准结果文件。

| 序列 | 基线 | T.W. | official | paper | official 覆盖率 | paper 覆盖率 |
|---|---:|---:|---:|---:|---:|---:|
| MH_01_easy | 0.151688 | 0.160691 | 0.195394 | 0.235764 | 99.27% | 99.21% |
| MH_02_easy | 0.181004 | 0.115112 | 0.114428 | 0.146224 | 99.18% | 99.11% |
| MH_03_medium | 0.132526 | 0.142992 | 0.152852 | 0.170471 | 97.67% | 97.67% |
| MH_04_difficult | 0.390537 | 0.358070 | 0.331830 | 0.310186 | 97.64% | 97.74% |
| MH_05_difficult | 0.574752 | 0.495007 | 0.461714 | 0.447005 | 96.30% | 94.89% |
| V1_01_easy | 0.065291 | 0.056241 | 0.057470 | 0.059706 | 96.05% | 95.84% |
| V1_02_medium | 0.196562 | 0.195031 | 0.182419 | 0.158610 | 93.74% | 93.39% |
| V1_03_difficult | 0.250598 | 0.280473 | 0.138109 | 0.203812 | 91.81% | 92.36% |
| V2_01_easy | 0.113514 | 0.119863 | 0.118212 | 0.126720 | 94.95% | 94.87% |
| V2_02_medium | 0.157479 | 0.146688 | 0.135182 | 0.124926 | 96.55% | 95.95% |
| V2_03_difficult | 0.199653 | 0.244217 | 0.199971 | 0.227364 | 94.35% | 94.35% |
| **平均** | **0.219419** | **0.210399** | **0.189780** | **0.200981** | — | — |

两套训练版都为 `11/11 (100%)` 工程流水线成功。相对于 11 序列平均：

- `official`：比基线降低 `13.51%`，比 T.W. 降低 `9.80%`；7/11 条低于基线，8/11 条低于 T.W.
- `paper`：比基线降低 `8.40%`，比 T.W. 降低 `4.48%`；7/11 条低于基线，6/11 条低于 T.W.

这些是固定配置、一次性全序列测试结果，不是把 EuRoC 当调参集。逐序列 JSON/CSV 和
机器可读总表为：

- `results/dual_gray_dce_trained_official_11seq_summary.json` / `.csv`
- `results/dual_gray_dce_trained_paper_11seq_summary.json` / `.csv`
- `results/gray_dce_cleanroom_comparison_11seq.json` / `.csv`

## 与论文 IR-VIO 列的差距

论文 Table I 报告的 IR-VIO 平均 ATE 是四舍五入后的 `0.12 m`。以这个公开数值作
同口径的近似比较：`official` clean-room 为 `0.189780 m`，高 `0.069780 m`
（`+58.15%`，约 `1.58×`）；`paper` clean-room 为 `0.200981 m`，高 `0.080981 m`
（`+67.48%`，约 `1.67×`）。逐序列差值和比例见
`results/gray_dce_vs_paper_irvio_11seq.json` / `.csv`。由于论文表格本身是取整值，
这里应理解为量级对照，而不是重新运行作者实现后的精确误差分解。

## 运行和后续

训练：

```bash
bash /mnt/d/大论文实验/IR-VIO的复现/scripts/run_gray_dce_training.sh
bash /mnt/d/大论文实验/IR-VIO的复现/scripts/run_gray_dce_paper_training.sh
```

导出后使用 `scripts/run_trained_gray_dce_suite.ps1` 可续跑任意序列；已有完整 ATE
默认跳过。当前没有失败序列，主要限制是数据规模和来源与论文原始训练条件不完全一致，
以及作者未公开的增强网络细节仍无法验证。若取得官方 SICE Part1 压缩包或作者权重，
应在新的目录和标签下独立重跑，不能覆盖本记录。
