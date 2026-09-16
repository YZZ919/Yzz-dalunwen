# 构建与运行记录

## 2026-09-09：块噪声剔除组件

- 增加独立 Python 实现，以及 `feature_tracker` 下的 C++
  `block_noise_removal` 库。
- `catkin_make --pkg feature_tracker -DCATKIN_ENABLE_TESTING=ON` 编译通过。
- `catkin_make run_tests_feature_tracker` 汇总为 4 tests、0 errors、
  0 failures；其中新增 C++ BNR 测试 2 项，Python BNR 测试 4 项也单独通过。
- 官方 Zero-DCE RGB checkpoint 只作为代理集成依赖。CPU PyTorch 2.4.1
  隔离存放于 `.deps/zero_dce_cpu`；只有运行代理脚本时才需要，不是
  ROS/VINS 基线的运行依赖。
- 11 条 EuRoC 序列的最暗抽样帧代理批测完成：6/11 个样本触发 BNR，
  1504 个特征中剔除 137 个；结果位于 `results/bnr_proxy_sample_suite/`。

## 环境

- WSL2：Ubuntu 20.04.6 LTS，ROS Noetic，20 个逻辑 CPU，约 7.6 GiB 内存、2 GiB swap。
- GPU：NVIDIA GeForce RTX 4060 Laptop 8 GiB；当前 VINS-Mono C++ 基线主要使用 CPU。
- GCC 9.4、CMake 3.16.3、OpenCV 4.2.0、Eigen 3.3.7、Ceres 1.14.0。
- 详细原始输出：`logs/environment.txt`、`logs/package_versions.txt`。

## 源码和兼容性修复

可完整获取的源码是 Gitee 上的历史 VINS-Mono 镜像（提交 `8a7a9622a8c41feb3bc909cf138edbb067f83ca3`），不是 IR-VIO 作者源码。Ubuntu 20.04 的 OpenCV 4 编译时，旧代码中的 `CV_AA`、`CV_GRAY2RGB`、`CV_RGB2GRAY`、`<opencv/cv.h>` 等接口已经移除或改名。只做了对应的 OpenCV 4 API 兼容替换和 C 头文件补充，没有改变估计器、特征、噪声或优化算法。修改文件列表及统计见 `logs/source_patch_stat.txt`。

## 构建结果

```text
[100%] Built target vins_estimator
```

可执行文件：`devel/lib/vins_estimator/vins_estimator`。首次构建失败、逐次重试和最终成功日志保存在 `logs/build_vins_mono.log`、`logs/build_vins_mono_retry1.log`、`logs/build_vins_mono_retry2.log`、`logs/build_vins_mono_retry3.log`。

## 运行结果

- 启动冒烟测试成功加载 `euroc.launch` 和自定义配置，日志见 `logs/smoke_launch.log`；仅因没有输入数据而按预期超时。
- EuRoC 11 条序列的 ASL 回放日志、VINS 日志、轨迹和指标均保存在 `results/<sequence>/`。
- MH_01 早期 rosbag/Leica 结果仍保留；最终汇总优先读取文件名带 `_asl` 的完整 ASL 真值重跑结果。
- 运行脚本会清理工作区根目录的临时 `results/vins_result.csv`，然后把最终轨迹复制到对应序列目录；原始数据不会被删除。
- V2_02 第一次并行启动时出现 ROS master/节点名冲突，该轮结果未进入汇总；随后串行重跑基线和加权版，当前 JSON/CSV 均来自有效重跑。

## 双层权重干净室实现

在基线验证后，已依据论文式（2）—（9）独立实现双层权重，并保留 `adaptive_weighting: 0/1` 开关。新增实现位于 `estimator.cpp`、`projection_factor.*` 和 `parameters.*`，配置为 `config/irvio_weighting_euroc_no_loop.yaml`。该实现已经重新编译并在 EuRoC 全部 11 条序列跑通；编译日志为 `logs/build_irvio_weighting.log`。

它不是官方源码：论文主文未给出的 Lambda、`delta_alpha`、`delta_beta` 和 `sigma_I` 采用显式、可修改的复现假设。BNR 与双图特征链已接入，并用重建灰度 DCE 代理在全部 11 条 EuRoC 序列完整跑通；修正轨迹表头识别后的双分支平均 ATE `0.19864 m`，相对基线均值降低 `9.47%`，流水线成功率 11/11。作者增强权重仍缺失，所有结果继续标为 clean-room proxy。固定配置重复性抽查中，`V1_01`/`V2_03` 漂移小于 0.004%，`V1_03` 漂移 8.35%，已单独记录。细节见 `references/clean_room_weighting.md`、`references/dual_branch_integration_plan.md` 和 `references/image_enhancement_bnr_gap.md`。

同日完成 `V1_03_difficult` 的 `adaptive_delta_alpha` 敏感性抽查（2/4/8，
`delta_beta=4.0`、1×回放、独立结果标签）；三次均通过判据，ATE 分别为
`0.199085/0.197663/0.182272 m`。该实验只用于记录未公开参数的敏感性，不改变正式
汇总，也不构成事后选参依据。原始配置、轨迹、权重日志和指标见
`tmp/dual_proxy_sensitivity/`、`results/V1_03_difficult/` 与
`results/v1_03_alpha_sensitivity_20260911.csv/json`。

2026-09-11 评估器修复：VINS 轨迹 CSV 无表头，旧版加载器无条件跳过第一条有效位姿；现改为逐行尝试解析，兼容带表头真值和无表头轨迹。修复前汇总与逐序列 JSON 保存在 `logs/metric_header_fix_pre_20260911/`，修复后指标由 `scripts/recompute_metrics_header_safe.py` 统一重算。
