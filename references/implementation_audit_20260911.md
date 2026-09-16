# IR-VIO clean-room 实现审计（2026-09-11）

本审计固定了当前可访问的 IR-VIO 主论文 PDF 和 Zero-DCE public repository 的代码引用，并把“已证实、工程修正、清晰的代理假设、尚未确定”分开。IR-VIO 作者源码和训练 checkpoint 仍未找到，因此以下结果不能命名为作者实现。

## 来源与版本

| 来源 | 固定内容 | 状态 |
|---|---|---|
| IR-VIO PDF | 单通道灰度输入、像素级曲线参数图、raw/enhanced 信息融合、BNR 的 M=6、N=8、μ=10.8、μcp=8、EuRoC 11 序列协议 | 本地 PDF；主文未给出作者仓库或 checkpoint |
| Zero-DCE `model.py` | 7 个卷积层、32 个中间特征、24 个 RGB 曲线通道、8 次递归 LE-curve；maxpool/upsample 在发布代码中被注释 | 固定 URL：`https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/model.py`；本地快照 SHA-256 见下 |
| Zero-DCE `Myloss.py` | L_spa 四方向张量相加后由训练脚本 `torch.mean`；L_exp；L_TV 按空间计数归一化但不除曲线通道数；另含 RGB L_color | 固定 URL：`https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/Myloss.py` |
| Zero-DCE `lowlight_train.py` | Conv 权重 N(0,0.02)；Adam、weight_decay=1e-4、lr=1e-4、clip=0.1、batch=8、200 epoch；实际加权 TV=200、exp=10、color=5 | 固定 URL：`https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/lowlight_train.py` |
| Zero-DCE `dataloader.py` | `random.seed(1143)`，文件列表 shuffle，每张图 resize 到 256×256、/255、CHW | 固定 URL：`https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/dataloader.py` |

本地 Zero-DCE 模型快照：

```text
references/third_party/Zero-DCE/model.py
SHA-256 8ADCC8B73B5BFDB1B6E5A16E672C59A7F8F664EE21AAFC3EEA44BB316D4B97F8
```

本次把官方仓库 `master` 固定到 commit
`e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd`，并保存了
`Myloss.py`、`lowlight_train.py`、`dataloader.py` 的本地副本。对应
SHA-256 分别为 `8555A3B0C7F74CC5601E7EDCF648DF5524A604FE2E10357722F1C65EA2E74F80`、
`089D52391755B22265828308059B8BE1148466989CD0A15B6F6CE7C61CA3C9EE`、
`ADECD0309298365DB79A5B8426FA70B156854D64746BF085A13CF6292113D0C4`。
快照目录还包含 `COMMIT.txt`；原始链接示例为
`https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd/Zero-DCE_code/Myloss.py`。

## 对应关系

| 项目 | 论文/官方证据 | 当前 clean-room 实现 | 判定 |
|---|---|---|---|
| 输入 | IR-VIO 主文明确 single-channel grayscale | `GrayEnhanceNet` 1 通道 | 已对齐 |
| 输出 | IR-VIO 主文明确 pixelwise curve parameter map | 8 通道标量曲线图 | 结构代理；通道数/训练细节未由主文完整公开 |
| 递归曲线 | Zero-DCE `model.py` 明确 8 次 | 8 次 `x + r(x²-x)` | 已对齐 Zero-DCE 基线 |
| 主干 | Zero-DCE 7-layer、32 feature maps | 1→32→…→8，跳连拓扑相同 | 已对齐基线；IR-VIO 私有改动未知 |
| scale=16 | 当前主论文 PDF 正文没有检索到明确的 downsample=16 语句或实现 | 低分辨率估计 + bilinear 上采样 | **待验证假设**，不能写成作者已证实参数 |
| 下采样尺寸 | 无作者代码 | `ceil(H/scale), ceil(W/scale)`，`align_corners=False` | 工程代理假设 |
| TV 位置 | 无作者代码 | 可选 full（上采样后）或 low（低分辨率图） | 待验证；当前参考短训使用 full |
| 预处理 | Zero-DCE 发布 dataloader 固定 resize 256 | 历史版随机 crop/resize；参考版 `official_resize` | 参考版对齐 public dataloader |
| 颜色损失 | Zero-DCE RGB 基线有 L_color；IR-VIO 输入为灰度 | 灰度训练不加入 L_color | 已按通道数修正 |
| 初始化 | public `weights_init` 对 Conv 使用 N(0,0.02) | 参考版提供 `zerodce_official`；历史版保留 PyTorch 默认 | 参考版对齐 |
| 优化器 | public Adam `weight_decay=1e-4` | 参考版显式设置；历史版为 0 | 参考版对齐 |
| shuffle | public DataLoader `shuffle=True` | 参考版按 `seed + epoch` 的局部 RNG 打乱 | 工程化可重复对齐 |
| checkpoint | public 只存 state_dict | 参考版保存模型、优化器、AMP、best loss、RNG、配置、数据清单 | 工程修正 |

## 损失尺度审计

运行：

```bash
PYTHONPATH=.deps/torch_cuda python3 scripts/audit_loss_scales.py \
  --output results/loss_scale_audit_20260911.json \
  > logs/loss_scale_audit_20260911.txt
```

固定张量结果：

| 项 | 本地历史实现 | public Myloss 等价形式 | 比值 |
|---|---:|---:|---:|
| L_spa | 0.01889835484 | 0.07559341192 | 4.0000 |
| L_exp | 0.00972139835 | 0.00972139835 | 1.0000 |
| L_TV（8 通道） | 2.56208133698 | 20.49665069580 | 8.0000 |
| L_spa 梯度范数 | 0.00601750 | 0.02407001 | 4.0000 |
| L_TV 梯度范数 | 0.22371649 | 1.78973190 | 8.0000 |

因此 `official` 和 `paper` 不是两个作者发布模型：它们是历史 clean-room 中的 loss 权重名字。新参考训练增加了独立的 `--loss-normalization official`，只修正官方代码的数值归约；`--loss-profile official` 再使用 public training script 的 10/200 暴露权重。灰度情况下 L_color 被删除，而不是置为 RGB 的 5 倍。

RTX 4060 的固定模型 AMP/FP32 对照记录在 `results/loss_scale_audit_20260911.json`：总损失相对差 `1.94e-5`，梯度范数相对差 `1.52e-4`。新参考训练启用 AMP 时将损失计算显式转为 FP32。

## 可重复性实现

新增参数和行为在 `scripts/train_gray_dce_cleanroom.py`：

- `--implementation-profile reference` 自动启用 Python、NumPy、PyTorch CPU/CUDA seed 和逐 epoch shuffle；可叠加 `--deterministic`。
- `--initialization zerodce_official`、`--weight-decay 1e-4`、`--grad-clip-norm 0.1`。
- `--preprocess official_resize --crop-size 256` 复现 public dataloader 的 resize 逻辑。
- `--loss-compute-fp32` 在 AMP 前向后用 FP32 计算三项灰度损失。
- 每个 checkpoint 记录 `best_val_loss`、RNG 状态、完整配置、脚本 SHA-256 和 train/val 图像清单及 SHA-256。
- 非空输出目录默认拒绝新跑；需显式使用新目录或 `--allow-overwrite`。
- resume 从 checkpoint 的全局 best loss 恢复，不会把较差的当前 epoch 当成新的 best。

## 数据审计

SICE 官方仓库 README 写明 Part1 为 360 个图像序列，官方 Zero-DCE/相关实验口径为 3022 张图随机划分 2422/600。当前可访问的 `okhater/SICE` 镜像只列出 259 个 train sample 和 47 个 test sample；当前 train/test 的 sample 目录 ID 不重叠，但尚未证明与官方 Part1 的场景集合、曝光选择和 2422/600 划分一致。数据来源、下载和灰度转换记录在 `references/sice_training_data.md`、`datasets/sice_mirror_low1/download_manifest.json` 与 `datasets/sice_gray/dataset_manifest.json`。因此现有 checkpoint 只能称为 mirror-derived clean-room checkpoint。

## 目前通过的验收

- `sanity_reference_official_1ep_retry`：1 epoch，8 train/4 val；loss、梯度、增强图和曲线图均有限。
- `sanity_reference_official_3ep` 与 `sanity_reference_paper_3ep`：相同 seed、同一 32/8 子集、同一初始化和顺序；两者验证 loss 均下降，差别只来自 loss profile。
- `tmp/gray_dce_reference_sanity_480x752.onnx`：checkpoint 重载后 PyTorch 与 ONNXRuntime 对同一输入最大绝对差 `2.38e-7`，RMSE `2.11e-8`，见 `results/gray_dce_reference_sanity_onnx_consistency_20260911.json`。
- 同一 sanity ONNX 在 Windows ONNXRuntime 与 OpenVINO CPU 之间最大绝对差也是 `2.38e-7`，RMSE `2.01e-8`，见 `results/gray_dce_reference_sanity_openvino_consistency_20260911.json`。

## VIO 公式与实现审计

当前 C++ 分支把论文中的视觉层权重落到残差平方项上，而不是直接把
`omega` 乘到残差向量上：`ProjectionFactor` 对残差和雅可比同时乘
`sqrt_weight * sqrt_info`，调用处传入 `sqrt(hybrid_weight)`。因此若
`hybrid_weight` 表示目标函数中的平方权重，这一处的平方关系是自洽的。

| 公式/机制 | 当前实现 | 判定 |
|---|---|---|
| 图像置信层 | 从未加权的目标帧 6-DoF 雅可比构造 `J^T J`，加正则后用 `det(J^T J)^(-1/12)` 得半径 | 论文结构对应；`Lambda` 未公开，半径参考采用窗口中位数并以 0.05 更新，是待验证代理 |
| Eq.(6) 权重 | `reference_radius / radius`，裁剪到 `[1/4, 4]` | 参数来自配置/论文线索；参考半径不是作者已知值 |
| Eq.(7) 约束区间 | 对 `imu_i+1 ... imu_j` 的 alpha 求均值 | 与文字描述对应；边界和缺失观测需继续用日志核验 |
| Eq.(8) 跟踪层 | `used_num / frame_gap`，裁剪到 4 | 与当前论文解释对应；`used_num` 的统计阶段需要和作者 TSR 定义区分 |
| Eq.(9) 融合 | 返回 `sqrt(alpha * beta)` 作为 `omega_C`；因子内部把它转成 `sqrt(omega_C)` 后缩放残差/雅可比 | 与论文 Eq.(1) 的 `omega_C * rho(||r||²)` 形式一致；这里不是重复平方。作者未知的仍是 `Lambda`、观测筛选和双分支融合细节 |
| 边缘化 | 现有 VINS-Mono 先验因子继续保留；新视觉因子只在当前优化窗口加入自适应权重 | 工程实现；尚未有作者边缘化代码可逐行核对 |
| 双分支 | raw/enhanced 使用独立 source、特征 ID 和权重日志，合并后送同一估计器 | 已实现；融合规则的作者细节仍不公开 |
| BNR | Sobel 平方梯度、10% 均匀块、灰度范围 15–235、6×8、阈值 10.8、最少保留 8；不把补点数当跟踪成功 | 与论文图示/参数一致；`Metric_Our.m` 的实现口径已单独记录 |

因此当前没有在 Eq.(9) 处发现明确的平方关系错误；后续若针对未公开的 `Lambda`、观测筛选或分支融合做敏感性实验，必须使用独立标签，只比较同一 checkpoint、同一回放，不能在 EuRoC 上反复挑优。
代数护栏脚本 `scripts/test_adaptive_weight_formula.py` 的结果在
`results/adaptive_weight_formula_check_20260911.json`，验证了“信息更紧时 alpha 更大”和
“因子残差平方后的系数仍为 Eq.(9) 的 omega_C”。

## Table I 数值口径

主论文 Table I 的 11 个 IR-VIO 行（四舍五入到 0.01 m）为
`0.16, 0.12, 0.11, 0.20, 0.21, 0.08, 0.08, 0.10, 0.06, 0.10, 0.17`，其算术平均为 `0.1263636 m`，表中 Avg 行显示 `0.12 m`。因此当前报告同时保留“表中 Avg=0.12 m”和“逐行四舍五入值平均=0.12636 m”两种口径；在原始未四舍五入数据或作者统计脚本出现前，不把二者之一当成精确真值。

## 尚未解决的差距

1. IR-VIO 作者增强网络的完整拓扑、scale/TV 分辨率、训练图像清单和 checkpoint 未公开，不能仅靠 Zero-DCE 代码推断。
2. 镜像数据规模远小于 2422/600，数据差距可能比损失归约差距更大。
3. EuRoC 已经被用于开发和参数敏感性试验；后续 ATE 只能标为开发/回归评估，不能再称完全盲测。
4. IR-VIO 主文 Table I 报告的平均值约 0.12 m，而抄录的 11 个四舍五入行算术平均约 0.12636 m；两种口径在未核明前同时保留。

## 已补的回放证据

- `scripts/evaluate_rpe.py` 和 `scripts/aggregate_rpe.py` 固定以 `1.0 s` 为相隔时间、`±0.06 s` 为配对容差，先做一次 SE(3) 对齐，再报告平移/旋转 RPE；已有 11 序列 clean-room 对照保存在 `results/rpe_existing_1s_20260911.{json,csv}`。
- `scripts/aggregate_tracking_stats.py` 只把 LK 尝试作为分母，把 LK 成功、RANSAC 后保留和 BNR 后保留分别报告；它明确标记这些是工程 proxy，不是论文 TSR。新配置 `config/irvio_dual_proxy_tracking.yaml` 和编译后的 tracker 已准备好，下一次独立回放会生成逐帧 CSV。

## 2026-09-11 后续受控实验

### 参考 `official` 11 序列回放

使用新参考实现、镜像 low1 数据训练的 100 epoch checkpoint
`checkpoints/gray_dce_cleanroom_v2/gray_reference_official_mirror_resize256/best_val.pth`
导出为 `.deps/gray_dce_reference_official_mirror_resize256_best_480x752.onnx`，在
固定的 `irvio_dual_proxy_tracking.yaml` 下独立回放 11 条 EuRoC。新标签为
`dual_gray_dce_reference_official_11seq_tracking`；旧标签和旧轨迹没有覆盖。

| 项目 | 结果 |
|---|---:|
| 完成/成功 | 11/11 |
| ATE SE(3) RMSE 平均 | 0.195955 m |
| ATE 中位数 | 0.171037 m |
| 1 s 平移 RPE RMSE 平均 | 0.043346 m |
| 1 s 旋转 RPE RMSE 平均 | 0.701012° |
| micro enhanced LK 后 F 通过率（工程 proxy） | 0.910198 |
| micro BNR 生存率（独立统计） | 0.996191 |

逐序列 ATE、RPE、逐帧 tracker CSV 和统计 JSON 分别保存在
`results/<sequence>/dual_gray_dce_reference_official_11seq_tracking_*`；汇总为
`results/dual_gray_dce_reference_official_11seq_tracking_summary.{json,csv}`、
`results/rpe_reference_official_11seq_tracking_1s_20260911.{json,csv}` 和
`results/tracking_reference_official_11seq_20260911.json`。这些 ATE 是开发/回归
评估，不是未触碰的盲测。论文只给出 TSR 改善叙述而未在可得正文中给出公式，故
上述 LK 后 F 比率不能写成论文 TSR。

同一 checkpoint 在 `V1_03_difficult` 的三次相同协议回放为
`0.194892 / 0.182440 / 0.182440 m`，均值 `0.186591 ± 0.005870 m`；这是回放
运行间波动，不是训练随机种子方差。新 `paper` 参考 checkpoint 的一次同协议
回放为 `0.144772 m`，其 1 s RPE 为 `0.047504 m / 0.742409°`。这说明在当前
镜像数据和实现下，loss profile 的排序不能由旧单次结果外推。

### 增补数据 B 组

通过镜像 API 完整取得所有可列出的 `low*.jpg` 后，得到 1021 train、176 test
图像，覆盖 259/47 个 sample；场景交集为空。下载和 SHA-256 清单在
`datasets/sice_mirror_all_low/download_manifest.json`，审计在
`results/sice_all_low_split_audit_20260911.json`，灰度清单在
`datasets/sice_gray_all_low/dataset_manifest.json`。这组仍是镜像派生数据，不是
官方 SICE Part1 2422/600。相同 reference/official 配置的 B 组长训已启动，独立
目录为 `checkpoints/gray_dce_cleanroom_v2/gray_reference_official_all_low_resize256`；
前 3 epoch 验证 loss 从 `0.947297` 降到 `0.479480`，第 2 epoch 有 2 个 AMP
非有限梯度 batch，被显式跳过并计入 history，不隐藏该事实。

### 增强层验收

`paper` checkpoint 的独立验证诊断在
`results/gray_reference_paper_mirror_resize256_val_diagnostics_20260911.json`；
`official` 对应诊断在同名 official 文件。两者输出和曲线均为有限值，未出现全黑
或全白输出。`paper` 的 PyTorch→Windows ONNX Runtime 最大绝对差为
`2.98e-7`，ONNX Runtime↔OpenVINO 最大绝对差为 `2.98e-7`，见对应
`results/gray_dce_reference_paper_mirror_resize256_*consistency_20260911.json`。

全序列亮度诊断由 `scripts/run_temporal_brightness_suite.sh` 生成；例如
`V1_03_difficult` 原图相邻帧均值绝对变化为 `4.13238`，新 `paper` 增强后为
`1.23949`。这是全局亮度连续性证据，不是 TSR。

### 仍未解决

IR-VIO 作者的增强拓扑细节、scale=16 是否确为作者参数、TV 计算分辨率、训练清单、
checkpoint、Lambda/观测筛选/双分支融合规则仍未公开。当前新增结果能定位“数据量、
loss 归约、AMP、回放波动”各自的影响，但不能把 clean-room checkpoint 写成作者
官方模型，也不能把表中 `Avg=0.12 m` 与逐行四舍五入均值 `0.12636 m` 之间的差异
当作已解释的精确结论。

### 停止前补充的可重复性与数值检查

在本轮长任务停止前，将 `--loss-compute-fp32` 的实现改为在损失模块内部显式
关闭 CUDA autocast；仅调用 `.float()` 并不能阻止卷积/池化被 autocast 调度为
FP16。新的固定张量审计保存在 `results/loss_scale_audit_strict_fp32_20260911.json`：
全 AMP 与纯 FP32 总损失相对差 `1.9419e-5`，严格 FP32 loss 归约为 `2.3517e-5`，
对应梯度范数相对差分别为 `1.5223e-4` 与 `1.4759e-4`。这说明该修正提高了
归约语义的确定性，但不能单独解释所有 AMP 非有限梯度。

新实现用 16/8 张图做了独立 1 epoch sanity，损失、梯度、输出和曲线均有限；
随后以相同 seed、数据顺序和 deterministic CUDA 运行两次，模型参数逐元素完全
一致（`max_model_abs_difference=0`），报告在
`results/reproducibility_sanity_20260911.json`。非 deterministic CUDA 的重复运行
仍出现约 `2e-6` 级参数差异，已保留对应两个 sanity 目录，不能把普通 AMP 运行
误写成 bitwise 可复现。

按用户要求已停止正在进行的作业，不再启动新的训练或回放。停止时：

- 新 `paper` 11 序列回放完成 `MH_01_easy`、`MH_02_easy`、`MH_03_medium`、
  `MH_04_difficult`、`MH_05_difficult` 共 5/11；其 ATE 算术均值为
  `0.248984 m`，只是部分序列，不能外推 11 序列平均。`V1_01_easy` 已开始增强
  但被中断，未计入完成数。结果和中断日志仍在各序列目录。
- 增补镜像 all-low 数据的 `reference/official` 长训停在第 16 轮，最佳验证 loss
  `0.396857`；训练目录为
  `checkpoints/gray_dce_cleanroom_v2/gray_reference_official_all_low_resize256/`，
  没有导出 ONNX 或进行 EuRoC 定位，因此不把它写成完整新模型结果。
- 全序列亮度诊断完成 7/11，统计脚本已写好；已完成的 JSON 保留在各序列目录，
  当前汇总标记为 partial。亮度变化仍只是全局帧均值诊断，不是论文 TSR。
