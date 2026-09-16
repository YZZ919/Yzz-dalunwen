# IR-VIO 复现工作区

本目录记录 IR-VIO 论文对应的公开数据集实验准备、VINS-Mono 基线，以及依据论文公式独立实现的“双层置信度自适应权重”。论文原始作者代码和完整参数仍未找到，所以这里明确称为 **clean-room（干净室）复现**，不声称是作者官方 IR-VIO。当前已实现论文式（2）—（10）中的双层权重与块噪声剔除，并依据论文给出的单通道曲线网络描述训练了两个 clean-room checkpoint；它们仍不是作者发布的 IR-VIO 权重。

## 当前状态（2026-09-11）

论文列出的 11 条 EuRoC 序列已经全部完成原图/增强图双分支代理测试。下表同时保留
VINS-Mono 基线、仅双层权重（T.W.）和双分支代理三列，正值表示双分支 ATE 降低：

| 序列 | 基线 ATE | T.W. ATE | 双分支代理 ATE | 双分支 vs 基线 | 双分支 vs T.W. | 覆盖率 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| MH_01_easy | 0.15169 m | 0.16069 m | 0.20710 m | -36.53% | -28.88% | 99.21% | 成功 |
| MH_02_easy | 0.18100 m | 0.11511 m | 0.14106 m | +22.07% | -22.54% | 99.11% | 成功 |
| MH_03_medium | 0.13253 m | 0.14299 m | 0.12259 m | +7.50% | +14.27% | 97.15% | 成功 |
| MH_04_difficult | 0.39054 m | 0.35807 m | 0.39956 m | -2.31% | -11.59% | 97.83% | 成功 |
| MH_05_difficult | 0.57475 m | 0.49501 m | 0.45377 m | +21.05% | +8.33% | 96.30% | 成功 |
| V1_01_easy | 0.06529 m | 0.05624 m | 0.05330 m | +18.37% | +5.23% | 96.19% | 成功 |
| V1_02_medium | 0.19656 m | 0.19503 m | 0.12671 m | +35.54% | +35.03% | 93.62% | 成功 |
| V1_03_difficult | 0.25060 m | 0.28047 m | 0.19766 m | +21.12% | +29.52% | 93.30% | 成功 |
| V2_01_easy | 0.11351 m | 0.11986 m | 0.12525 m | -10.34% | -4.50% | 95.04% | 成功 |
| V2_02_medium | 0.15748 m | 0.14669 m | 0.13333 m | +15.33% | +9.11% | 95.53% | 成功 |
| V2_03_difficult | 0.19965 m | 0.24422 m | 0.22473 m | -12.56% | +7.98% | 93.75% | 成功 |
| **11 条平均** | **0.21942 m** | **0.21040 m** | **0.19864 m** | **+9.47%** | **+5.59%** | — | **11/11** |

按“至少 100 个有效匹配位姿、时间覆盖率至少 70%、ATE 有限”判定，双分支代理
流水线成功率为 **11/11（100%）**。双分支 ATE 低于基线和 T.W. 的序列均为 7/11
（63.6%）；其余序列的负结果原样保留。因此这里的 100% 是工程流水线成功率，
不是完整 IR-VIO 的复现成功率，也不能替代论文报告的作者模型结果。

## 图像增强与块噪声剔除进展

论文式（10）对应的块噪声剔除（BNR）现已同时提供 Python 验证版和可链接到
VINS-Mono 的 C++ 库。实现采用论文给定的 `6 × 8` 分块、噪声比阈值 `10.8`
和最少保留 `8` 个约束，并补上引用噪声估计实现所说明的 `1/6` 归一化因子。
Python 4 项测试和 C++ 2 项测试均通过，VINS-Mono 的 `feature_tracker` 也已重新
编译通过。

为验证端到端数据通路，另运行了一个**不计入 IR-VIO 指标**的代理实验：把
EuRoC MH_04 中抽样得到的最暗帧（灰度均值 `5.57`）复制为 RGB，使用官方
Zero-DCE `Epoch99.pth` 增强，再转回灰度。输出均值为 `17.20`，BNR 判定
`29/48` 个块超过阈值，在 121 个检测特征中剔除 47 个、保留 74 个。可视化和
机器可读结果位于 `results/bnr_component/`。进一步对 11 条序列分别从每 20
帧抽样中选择最暗帧：代理 BNR 在 6/11 个样本上发生剔除，共检测 1504 个
特征、剔除 137 个。完整表格和逐帧可视化位于
`results/bnr_proxy_sample_suite/`。这些结果只证明增强→噪声估计→约束保护链路
能实际运行，不代表 IR-VIO 的未公开单通道增强网络。

另外按论文“单通道输入、逐像素曲线参数图”的描述重建了 1→32→…→8 通道
灰度 DCE 网络，并从官方 RGB checkpoint 做确定性通道折叠初始化。MH_04 最暗
样本上，它与“灰度复制到 RGB 后再取平均”的输出像素平均绝对差只有
`0.0068/255`、99.32% 像素完全相同，BNR 决策也相同。这个结果验证了实现和
权重折叠的一致性，但没有补回作者未公开的训练过程，因此仍标为 clean-room
代理而不是 IR-VIO 模型。

原图/增强图双分支现已实际接入 VINS：两套跟踪器使用独立特征 ID，消息携带
`source=0/1`，BNR 只处理增强分支，估计器按来源分别计算图像层置信度并把两路
残差加入同一个优化问题。第一条完整代理实验 `V1_01_easy` 已通过正式判据：
1401 个估计位姿、1391 个真值匹配、96.19% 时间覆盖、SE(3) ATE `0.05330 m`。
同一序列原 VINS 基线为 `0.06529 m`，仅式（2）—（9）权重为 `0.05624 m`，
因此该次双分支代理分别降低 `18.37%` 和 `5.23%`。全程 1454 个发布帧中有
10 帧触发 BNR，共从增强分支 149169 个特征观测中移除 14 个。该结果只计为
clean-room 双分支代理，不能与作者未公开模型画等号；完整文件见
`results/V1_01_easy/dual_gray_dce_full*` 和 `results/dual_branch_proxy_summary.*`。

保持全部参数不变后，其余 10 条序列也已完成。11 条的双分支平均 ATE 为
`0.19864 m`，相对基线均值 `0.21942 m` 降低 `9.47%`，相对 T.W. 均值
`0.21040 m` 降低 `5.59%`。`MH_01`、`MH_04`、`V2_01`、`V2_03` 的双分支
ATE 比基线高，作为负结果保留；这说明公开正文公式加上代理增强网络可以稳定
运行，但不能保证在每条序列上改善。

为使完整测试可执行，重建的灰度网络已导出固定 `480×752` ONNX。OpenVINO
CPU 后端平均 `0.144 s/帧`，相对原 PyTorch `8.35 s/帧` 约快 58 倍；与原
PyTorch 样本逐像素完全相同，与 OpenVINO 输出相比 99.9989% 像素相同且最大
只差 1 灰度级。V1_01 的 2912 帧预处理耗时约 451 秒，平均亮度
`134.16 → 177.84`。

ATE 使用估计时间戳对应的 ASL 真值位置，以线性时间插值关联，并以刚体 SE(3) 对齐（不校正尺度）后计算位置 RMSE。所有最终表格均使用 `state_groundtruth_estimate0/data.csv` 和相机时间范围；MH_01 早期基于稀疏 `/leica/position` 的 bag 结果仍保留，但汇总已优先采用完整 ASL 真值重跑结果。两种方法对每条序列使用完全相同的真值和评估选项。

评估器会自动识别带表头的 EuRoC 真值 CSV 和无表头的 VINS 轨迹 CSV，保留第一条有效估计位姿；修复前的汇总副本保存在 `logs/metric_header_fix_pre_20260911/`。

固定配置重复回放了 `V1_01_easy`、`V1_03_difficult`、`V2_03_difficult`。三条重复流水线均成功；ATE 相对正式运行的绝对漂移分别为 `0.003%`、`8.35%`、`0.001%`。其中 `V1_03` 在加入订阅等待并改用 1× 真实时间后仍出现约 7.79%—8.35% 漂移，说明其初始化/特征跟踪路径具有运行间敏感性；后续应报告多次重复统计，不应把单次结果误当成严格确定值。

随后在 `V1_03_difficult` 上固定 `delta_beta=4.0`、增强模型和 1× 回放，仅改变图像层裁剪参数
`delta_alpha`，并使用独立配置与结果标签做了敏感性抽查：`alpha=2` 的 ATE 为 `0.19908 m`，
`alpha=4` 为 `0.19766 m`，`alpha=8` 为 `0.18227 m`；三次流水线均成功。较大的裁剪范围确实
改变了轨迹（`alpha=8` 覆盖率为 `92.74%`，其余为 `93.30%`），但论文正文没有给出该参数，
所以这组结果只说明 clean-room 假设敏感，不能据此事后选参。完整表格见
`results/v1_03_alpha_sensitivity_20260911.csv/json`，过程日志见 `logs/dual_proxy_sensitivity_20260911.txt`。

流水线成功判据：至少 100 个有效关联位姿、估计时间跨度覆盖真值跨度的 70% 以上、ATE 为有限值。另记录 `metric_pass_ate_lt_1m` 作为一个宽松质量检查；它不替代论文的数值比较。

## 训练版灰度 DCE checkpoint（2026-09-11）

已把“1 通道输入、8 张曲线参数图、8 次递归、downsampling scale=16”的 clean-room
训练通路落地，并在 CUDA PyTorch `2.4.1+cu124`、RTX 4060 Laptop GPU 上完成 100
个 epoch。训练数据来自可访问的 SICE 镜像，每个样本固定选 `low1.jpg`，实际为
`259` 张训练图和 `47` 张验证图；这不是论文所写的官方 `2422/600` Part1 划分，
因此不能把 checkpoint 称作作者权重。数据来源和准备记录见
`references/sice_training_data.md`。

两个 loss profile 都完整训练并保存了 `epoch_005.pth` … `epoch_100.pth` 和
`best_val.pth`。验证集只用于选 best checkpoint，EuRoC 只作为一次性测试集；随后
两套 best checkpoint 均在论文列出的 11 条 EuRoC 序列上完成了同步双分支回放。两个
profile 的验证损失数值因权重缩放不同，不能直接横向比较：

| profile | loss | 最优 epoch / SICE val loss | VIO 测试序列 | 11 序列平均 ATE | 流水线成功率 |
|---|---|---:|---|---:|---:|
| `official` | `1 L_spa + 10 L_exp + 200 L_tv` | 100 / 0.231581 | 全部 11 条 | **0.189780 m** | 11/11 |
| `paper` | `1 L_spa + 1 L_exp + 20 L_tv` | 91 / 0.030546 | 全部 11 条 | **0.200981 m** | 11/11 |

固定的 11 序列对照（SE(3) ATE RMSE，单位 m）已写入合并 CSV/JSON；其中代表性
序列如下，基线和 T.W. 使用同一真值、同一
时间关联和同一成功判据：

| 序列 | 基线 | T.W. | `official` | `paper` |
|---|---:|---:|---:|---:|
| V1_01_easy | 0.065291 | 0.056241 | **0.057470** | 0.059706 |
| V1_03_difficult | 0.250598 | 0.280473 | **0.138109** | 0.203812 |
| V2_03_difficult | 0.199653 | 0.244217 | 0.199971 | 0.227364 |
| **11 序列平均** | **0.219419** | **0.210399** | **0.189780** | **0.200981** |

相对于 11 序列平均，`official` 比基线降低 `13.51%`、比 T.W. 降低 `9.80%`；`paper`
分别降低 `8.40%` 和 `4.48%`。`official` 在 7/11 条上低于基线、8/11 条上低于
T.W.；`paper` 在 7/11 条上低于基线、6/11 条上低于 T.W.。负结果完整保留，没有用
EuRoC 反选 epoch 或 loss profile；这仍是 clean-room 训练版，不是论文声称的作者
模型结果。

与论文 Table I 中 IR-VIO 的报告平均值 `0.12 m` 对齐时，`official` clean-room 为
`0.189780 m`，高 `0.069780 m`（约 `58.15%`，误差约为论文值的 `1.58×`）；`paper`
为 `0.200981 m`，高 `0.080981 m`（约 `67.48%`，`1.67×`）。按论文表中的四舍五入
逐序列比较，`official` 仅在 `MH_02_easy` 和 `V1_01_easy` 低于报告值，其余序列仍
有差距；这正是当前最主要的复现缺口。

逐序列原始 JSON、CSV 和合并报告位于：

- `checkpoints/gray_dce_cleanroom/gray_official_mirror/`
- `checkpoints/gray_dce_cleanroom/gray_paper_mirror/`
- `results/dual_gray_dce_trained_official_11seq_summary.json` / `.csv`
- `results/dual_gray_dce_trained_paper_11seq_summary.json` / `.csv`
- `results/gray_dce_cleanroom_comparison_11seq.json` / `.csv`
- `results/gray_dce_vs_paper_irvio_11seq.json` / `.csv`

三条代表序列的早期独立汇总仍保留在 `*_3seq_summary.*`，不覆盖 11 序列结果。

复跑训练和导出命令（均在 WSL 中执行）为：

```bash
bash /mnt/d/大论文实验/IR-VIO的复现/scripts/run_gray_dce_training.sh
bash /mnt/d/大论文实验/IR-VIO的复现/scripts/run_gray_dce_paper_training.sh
python3 /mnt/d/大论文实验/IR-VIO的复现/scripts/export_gray_dce_onnx.py \
  /mnt/d/大论文实验/IR-VIO的复现/.deps/gray_dce_cleanroom_scale16_best_480x752.onnx \
  --weights /mnt/d/大论文实验/IR-VIO的复现/checkpoints/gray_dce_cleanroom/gray_official_mirror/best_val.pth \
  --downsample-scale 16
```

已导出的 `official` 和 `paper` ONNX 均用于 OpenVINO 逐帧增强；Windows PowerShell
全序列测试示例为：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_trained_gray_dce_suite.ps1 `
  -Sequences @(
    'MH_01_easy','MH_02_easy','MH_03_medium','MH_04_difficult','MH_05_difficult',
    'V1_01_easy','V1_02_medium','V1_03_difficult','V2_01_easy','V2_02_medium','V2_03_difficult'
  ) `
  -Model (Resolve-Path .deps\gray_dce_cleanroom_scale16_best_480x752.onnx).Path `
  -RunLabel dual_gray_dce_trained_official
```

`paper` 只需把模型换成 `gray_dce_cleanroom_scale16_paper_best_480x752.onnx` 并使用
`dual_gray_dce_trained_paper` 标签。合并两个 profile 的脚本是
`scripts/compare_gray_dce_trained_runs.py`。

## 复现实验命令

先启动 WSL Ubuntu 20.04，并在本目录执行：

```bash
source /opt/ros/noetic/setup.bash
source /mnt/d/大论文实验/IR-VIO的复现/devel/setup.bash
```

ASL 格式序列（图像+IMU CSV）直接回放，不生成 rosbag：

```bash
bash /mnt/d/大论文实验/IR-VIO的复现/scripts/run_sequence_asl.sh \
  V1_01_easy /mnt/d/大论文实验/IR-VIO的复现/datasets/V1_01_easy 1.0 0
```

ROS1 bag：

```bash
bash /mnt/d/大论文实验/IR-VIO的复现/scripts/run_sequence.sh \
  MH_01_easy /mnt/d/大论文实验/IR-VIO的复现/datasets/MH_01_easy.bag
```

双层权重 clean-room 版本在命令末尾指定配置和结果标签：

```bash
bash /mnt/d/大论文实验/IR-VIO的复现/scripts/run_sequence_asl.sh \
  V1_01_easy /mnt/d/大论文实验/IR-VIO的复现/datasets/V1_01_easy 1.0 0 \
  /mnt/d/大论文实验/IR-VIO的复现/config/irvio_weighting_euroc_no_loop.yaml \
  irvio_weighting

bash /mnt/d/大论文实验/IR-VIO的复现/scripts/run_sequence.sh \
  MH_01_easy /mnt/d/大论文实验/IR-VIO的复现/datasets/MH_01_easy.bag \
  /mnt/d/大论文实验/IR-VIO的复现/config/irvio_weighting_euroc_no_loop.yaml \
  irvio_weighting
```

评估示例：

```bash
python3 /mnt/d/大论文实验/IR-VIO的复现/scripts/evaluate_ate.py \
  --estimate /mnt/d/大论文实验/IR-VIO的复现/results/V1_01_easy/vins_mono.csv \
  --groundtruth /mnt/d/大论文实验/IR-VIO的复现/datasets/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv \
  --camera-csv /mnt/d/大论文实验/IR-VIO的复现/datasets/V1_01_easy/mav0/cam0/data.csv \
  --interpolate --output /mnt/d/大论文实验/IR-VIO的复现/results/V1_01_easy/metrics.json
```

Machine Hall 五条序列的可续跑批处理：

```bash
bash /mnt/d/大论文实验/IR-VIO的复现/scripts/run_machine_hall_suite.sh
```

BNR 单帧验证（输入外部增强图）：

```bash
python3 /mnt/d/大论文实验/IR-VIO的复现/scripts/block_noise_removal.py \
  RAW.png ENHANCED.png --json bnr.json --visualization bnr.png
```

官方 RGB Zero-DCE 灰度代理（仅组件验证，不计入 IR-VIO 结果）：

```bash
python3 /mnt/d/大论文实验/IR-VIO的复现/scripts/enhance_zero_dce_proxy.py \
  RAW.png ENHANCED_PROXY.png --report enhancement_proxy.json
```

重建灰度网络导出 ONNX、预处理完整序列并运行双分支代理：

```bash
# 首次使用：WSL 导出依赖与 Windows 快速推理依赖
bash /mnt/d/大论文实验/IR-VIO的复现/scripts/bootstrap_onnx_export.sh
# 在 Windows PowerShell 另执行：
# powershell -ExecutionPolicy Bypass -File scripts\bootstrap_fast_inference.ps1

python3 /mnt/d/大论文实验/IR-VIO的复现/scripts/export_gray_dce_onnx.py

# Windows PowerShell：工作区内隔离的 OpenCV/OpenVINO 依赖
$env:PYTHONPATH=(Resolve-Path .deps\onnx_runner).Path + ';' + \
  (Resolve-Path .deps\openvino_windows).Path
python scripts\enhance_euroc_onnx.py datasets\V1_01_easy \
  results\V1_01_easy\enhanced_openvino_full --backend openvino \
  --report results\V1_01_easy\enhanced_openvino_full.json

# WSL：原图与增强图同步回放
bash /mnt/d/大论文实验/IR-VIO的复现/scripts/run_sequence_dual_asl.sh \
  V1_01_easy /mnt/d/大论文实验/IR-VIO的复现/datasets/V1_01_easy \
  /mnt/d/大论文实验/IR-VIO的复现/results/V1_01_easy/enhanced_openvino_full \
  1.0 0
```

全部序列可在 Windows PowerShell 中续跑；已经存在完整指标的序列会自动跳过：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_dual_proxy_suite.ps1
```

批处理默认按真实时间 `1×` 回放以减小 VINS 队列压力；确认只追求速度时可显式传入
`-ReplaySpeed 3`，但困难序列的初始化结果可能出现更大运行间漂移。

固定配置重复性抽查（不覆盖正式结果）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_dual_proxy_repeatability.ps1
```

图像层裁剪参数敏感性（默认只跑 `V1_03_difficult` 的 `alpha=2/4/8`，不覆盖正式结果）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_dual_proxy_sensitivity.ps1
```

公式到代码的映射、参数假设和权重统计见 `references/clean_room_weighting.md`；汇总对照见 `results/weighting_comparison.csv`。每条序列保留基线/加权轨迹、评估 JSON、运行日志和逐窗口权重 CSV。

## 目录说明

- `references/paper_extracted.txt`：论文文本提取；论文 PDF 本身位于用户给定的 Desktop 路径。
- `references/code_search_20260911.md`：本轮作者源码、DOI 和公开页面检索记录。
- `references/official_current_euroc_config.yaml`：官方 VINS-Mono 当前仓库中的 EuRoC 配置快照。
- `src/VINS-Mono`：可完整获取的历史镜像，提交 `8a7a9622a8c41feb3bc909cf138edbb067f83ca3`；不是 IR-VIO 作者仓库。
- `config/vins_mono_euroc_no_loop.yaml`：左相机、关闭直方图均衡和回环、关闭在线时间标定的基线配置。
- `config/irvio_weighting_euroc_no_loop.yaml`：式（2）—（9）干净室复现配置，包含全部参数假设。
- `references/clean_room_weighting.md`：公式映射、与官方实现的边界和实测权重统计。
- `references/image_enhancement_bnr_gap.md`：图像增强缺失信息、式（10）实现依据与复现边界。
- `references/gray_dce_cleanroom_training.md`：两个灰度 DCE profile 的训练配置、11 序列指标和边界。
- `references/dual_branch_integration_plan.md`：原图/增强图联合优化的接入点与验收条件。
- `references/third_party/`：Zero-DCE 和噪声度量所需的最小官方代码/权重快照。
- `scripts/bootstrap_wsl_noetic.sh`：安装 ROS Noetic/Ceres/OpenCV/Eigen 等依赖。
- `scripts/bootstrap_zero_dce_proxy.sh`：把可选 CPU PyTorch 依赖隔离安装到本目录。
- `scripts/bootstrap_onnx_export.sh` / `bootstrap_fast_inference.ps1`：隔离安装 ONNX 导出与 OpenVINO 推理依赖。
- `scripts/publish_euroc_asl.py`：发布 ASL 图像/IMU；可选同步发布增强图。
- `scripts/run_sequence*.sh`：基线、权重和双分支运行与日志/轨迹保存。
- `scripts/evaluate_ate.py`：时间关联、SE(3)/Sim(3) 对齐及 ATE RMSE。
- `scripts/aggregate_results.py`：从逐序列 JSON 重新生成汇总 CSV/JSON 和成功率。
- `scripts/aggregate_dual_proxy_results.py`：自动发现完整双分支结果并生成独立汇总。
- `scripts/run_dual_proxy_suite.ps1`：可续跑的 11 序列增强、双分支、ATE 与汇总流水线。
- `scripts/run_dual_proxy_repeatability.ps1` / `compare_dual_proxy_repeatability.py`：固定配置重复回放与漂移比较。
- `scripts/run_dual_proxy_sensitivity.ps1`：在独立结果标签下扫描 `adaptive_delta_alpha`，用于参数敏感性记录。
- `scripts/recompute_metrics_header_safe.py`：兼容有/无表头 CSV 的 ATE 重算工具。
- `scripts/run_machine_hall_suite.sh`：Machine Hall 五条 ASL 序列的基线/加权批处理。
- `scripts/block_noise_removal.py`：BNR 独立实现、特征筛选、JSON 与可视化输出。
- `scripts/enhance_zero_dce_proxy.py`：官方 RGB Zero-DCE 的明确标注代理测试。
- `scripts/enhance_gray_dce_cleanroom.py`：按论文文字重建的单通道 DCE 代理。
- `scripts/export_gray_dce_onnx.py`：把重建网络导出为固定 EuRoC 尺寸 ONNX。
- `scripts/train_gray_dce_cleanroom.py`：SICE 灰度数据上的 clean-room 训练、验证、checkpoint 和预览。
- `scripts/prepare_sice_gray.py` / `scripts/download_sice_mirror.py`：镜像数据下载与单通道整理。
- `scripts/compare_gray_dce_trained_runs.py`：合并两个 loss profile 的训练和 EuRoC 对照结果。
- `scripts/validate_gray_dce_checkpoints.py`：在 WSL/CUDA 下加载两个 best checkpoint 并做形状、有限值检查。
- `scripts/enhance_euroc_onnx.py`：用 OpenCV/ONNX Runtime/OpenVINO 批量增强 EuRoC。
- `scripts/aggregate_tracking_stats.py` / `aggregate_tracking_stats_suite.py`：按明确分母汇总 LK、F 矩阵和 BNR 统计，不冒充论文 TSR。
- `scripts/evaluate_rpe.py` / `aggregate_rpe.py`：固定时间间隔的 SE(3) 对齐后平移/旋转 RPE。
- `scripts/evaluate_temporal_brightness.py` / `run_temporal_brightness_suite.sh`：记录原图与增强图相邻帧全局亮度变化。
- `scripts/aggregate_temporal_brightness.py`：汇总各序列亮度连续性诊断，明确不冒充论文 TSR。
- `scripts/build_reference_comparison.py` / `compare_training_reproducibility.py`：生成新旧同协议对照和 deterministic 小样本复现检查。
- `scripts/build_run_manifest.py`：记录环境、数据源、运行标签和停止时的部分完成状态。
- `scripts/run_bnr_proxy_sample_suite.py`：11 条 EuRoC 最暗抽样帧的代理组件批测。
- `src/VINS-Mono/feature_tracker/src/block_noise_removal.*`：可接入双图特征分支的 C++ BNR 库。
- `logs/`：依赖安装、编译、启动和环境记录。
- `results/<sequence>/`：每条序列的回放日志、VINS 日志、轨迹和指标 JSON。
- `results/dual_branch_proxy_repeatability.csv/json`：代表序列重复性比较结果。
- `results/v1_03_alpha_sensitivity_20260911.csv/json`：`V1_03_difficult` 的 `delta_alpha=2/4/8` 敏感性结果。
- `results/irvio_reproduction_summary.xlsx`：基线/双层权重的 11 条序列 Excel 汇总；双分支代理的权威汇总为 `results/dual_branch_proxy_summary.csv/json`。

## 已知限制和下一步

1. 尚未定位到 IR-VIO 作者源码和论文补充参数；现在能严谨声称的是“依据正文公式独立实现了双层权重”，不是复现完整官方 IR-VIO。
2. 块噪声剔除和双图特征消息链已接入并在全部 11 条序列完整跑通；单通道 Zero-DCE 的
   clean-room 训练通路也已完成，但使用的是镜像派生的 `259/47` 图像，不是官方
   `2422/600` SICE Part1，也没有作者 checkpoint。
3. 原有折叠权重双分支代理的 11 序列平均 ATE 为 `0.19864 m`；训练版 `official`
   和 `paper` checkpoint 也已各自完成 11/11 工程流水线，但它们使用镜像派生数据，
   仍不能等同于完整 IR-VIO 成功率或作者论文数值。
4. 已完成代表序列重复性和 `delta_alpha` 敏感性抽查；正式比较仍使用固定 `alpha=4.0`，
   不用单次敏感性结果事后挑参。若获得作者源码或补充参数，应建立独立“官方实现”结果列，
   不覆盖本次 clean-room 记录。

## 2026-09-11 审计与参考实现增补

本轮实现/公式/数据审计见 `references/implementation_audit_20260911.md`。
官方 Zero-DCE 代码快照固定在 commit
`e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd`，本地快照位于
`references/third_party/Zero-DCE/`；审计确认灰度版本删除颜色一致性损失，并把
Spatial/TV 的真实归约尺度分别与 public 实现的 4 倍/8 倍差异单独记录，未把系数
盲目整体放大。

新的参考 `official` checkpoint 使用 `official_resize`、Zero-DCE 卷积初始化、
Adam `weight_decay=1e-4`、显式 seed/shuffle、scale-16 代理和 FP32 loss 归约；
AMP/FP32 1 epoch 对照、checkpoint 重载、ONNX Runtime 与 OpenVINO 一致性均已
通过。100 epoch 参考 `official` 11 序列回放标签为
`dual_gray_dce_reference_official_11seq_tracking`，ATE 平均 `0.195955 m`、
成功率 `11/11`；RPE、跟踪 proxy、BNR 生存率和亮度诊断均保留在 `results/`。

`paper` 参考训练也已完成 100 epoch，最佳验证 loss `0.07609365`，checkpoint
位于 `checkpoints/gray_dce_cleanroom_v2/gray_reference_paper_mirror_resize256/`，
对应 ONNX 与验证诊断均已保存。`V1_03_difficult` 的一次同协议 ATE 为
`0.144772 m`；它不是作者 checkpoint。

为分离“数据量”因素，新增镜像全低曝光组：下载清单
`datasets/sice_mirror_all_low/download_manifest.json`，共 `1021/176` 张图，
场景审计无 train/test 交集，灰度数据在 `datasets/sice_gray_all_low/`。它仍不是
官方 SICE Part1 2422/600；同一参考 `official` 实现的独立长训已在
`checkpoints/gray_dce_cleanroom_v2/gray_reference_official_all_low_resize256/`
启动，历史 259/47 训练目录和所有轨迹均未覆盖。

所有后续 EuRoC ATE 均属于复现开发/回归评估：EuRoC 已用于参数敏感性试验，不能
再称完全盲测；论文 TSR 公式在可得正文中未定义，因此本项目只报告明确命名的
LK/F 通过率 proxy，不把补点后的特征数冒充 TSR。

本轮停止前又完成了严格 FP32 loss 归约审计和 deterministic CUDA 重复性验收：
固定小样本的两次模型参数逐元素一致，报告见
`results/reproducibility_sanity_20260911.json`；非 deterministic CUDA 的普通重复
仍保留约 `2e-6` 级参数差异。用户要求停止后，`paper` 新 11 序列回放停在 5/11，
all-low 数据 `official` 长训停在 epoch 16（最佳验证 loss `0.396857`），亮度诊断
停在 7/11；所有中断结果和日志均保留，未覆盖历史目录。阶段性同协议比较表在
`results/reference_comparison_20260911.{json,csv}`，完整审计在
`references/implementation_audit_20260911.md`。
