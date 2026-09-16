# IR-VIO 论文复现协议摘录

来源 PDF（用户提供路径实际文件名）：

`C:\Users\Administrator\Desktop\大论文\很相关论文\IR-VIO_Illumination-Robust_Visual-Inertial_Odometry_Based_on_Adaptive_Weighting_Algorithm_With_Two-Layer_Confidence_Maximization.pdf`

论文为 Song 等人的 IEEE/ASME TMECH 2023 文章，DOI `10.1109/TMECH.2023.3277031`。方法在 VINS-Mono 上增加自适应多传感器/特征加权、图像增强和块噪声去除。

## 公共数据集实验设置

- EuRoC MAV 的 11 条序列：MH_01_easy、MH_02_easy、MH_03_medium、MH_04_difficult、MH_05_difficult、V1_01_easy、V1_02_medium、V1_03_difficult、V2_01_easy、V2_02_medium、V2_03_difficult。
- 数据为双目灰度相机（20 Hz）和 IMU（200 Hz）；实验只使用左相机。
- 关闭回环检测。
- 为公平比较，IR-VIO、VINS-Mono、R-VIO2 的直方图均衡关闭；VINS-Mono 与 IR-VIO 的相机-IMU 在线时间标定关闭。
- 论文表格报告 Absolute Trajectory Error（ATE）的 RMSE，参考 `evo` 工具。

## 论文 Table I（ATE RMSE，m）

| 序列 | ROVIO | OKVIS | R-VIO | VINS-Mono | R-VIO2 | IR-VIO |
|---|---:|---:|---:|---:|---:|---:|
| MH_01_easy | 0.30 | 0.16 | 0.34 | 0.22 | 0.17 | 0.16 |
| MH_02_easy | 0.59 | 0.22 | 0.15 | 0.14 | 0.13 | 0.12 |
| MH_03_medium | 0.39 | 0.24 | 0.29 | 0.18 | 0.20 | 0.11 |
| MH_04_difficult | 0.85 | 0.34 | 0.31 | 0.32 | 0.23 | 0.20 |
| MH_05_difficult | 1.23 | 0.47 | 0.44 | 0.32 | 0.31 | 0.21 |
| V1_01_easy | 0.19 | 0.09 | 0.33 | 0.11 | 0.10 | 0.08 |
| V1_02_medium | 0.20 | 0.20 | 0.10 | 0.09 | 0.12 | 0.08 |
| V1_03_difficult | 0.17 | 0.24 | 0.14 | 0.27 | 0.11 | 0.10 |
| V2_01_easy | 0.40 | 0.13 | 0.12 | 0.09 | 0.09 | 0.06 |
| V2_02_medium | 0.59 | 0.14 | 0.15 | 0.13 | 0.10 | 0.10 |
| V2_03_difficult | 0.22 | 0.29 | 0.32 | 0.27 | 0.14 | 0.17 |
| 平均 | 0.47 | 0.23 | 0.24 | 0.19 | 0.15 | 0.12 |

VINS-Mono 基线和式（2）—（9）的双层权重 clean-room 实现已经在全部 11 条序列完成对照。修正轨迹表头识别后的实测平均 ATE 为基线 `0.21942 m`、双层权重 `0.21040 m`；加权版改善 6/11 条，按均值降低 `4.11%`。式（10）BNR、原图/增强图双分支及分源权重的 clean-room 代理也已在全部 11 条序列完成，平均 ATE `0.19864 m`，相对基线降低 `9.47%`，相对仅权重版降低 `5.59%`，流水线成功率 11/11。双分支低于基线的序列为 7/11，负结果已保留。以上仍低于论文表中的完整 IR-VIO `0.12 m`，也不能声称复现论文报告的 35.6% 平均改善。因作者单通道增强网络的训练与 checkpoint 未公开，这些只能列为 clean-room proxy。若未来获得作者源码，应将其作为独立“官方实现”实验，不与当前结果混淆。

为检查未公开参数的影响，另在 `V1_03_difficult` 上做了 `adaptive_delta_alpha=2/4/8`
的独立敏感性实验（其余配置固定、1× 回放）。ATE RMSE 为 `0.199085/0.197663/0.182272 m`，
三次均满足工程成功判据；该结果用于估计复现不确定性，不改变论文协议下正式采用的 `alpha=4.0`。

在上述固定评估协议下，另完成了两个单通道灰度 DCE clean-room checkpoint 的 11 序列测试：
`official` profile 平均 ATE `0.189780 m`，`paper` profile 平均 ATE `0.200981 m`，
均为 `11/11` 工程流水线成功。训练数据是镜像派生的 SICE `259/47` 子集，作者单通道
checkpoint 仍未公开；完整训练配置和逐序列结果见 `references/gray_dce_cleanroom_training.md`。
