# 代码来源核查

- 论文对应工作标题：`IR-VIO: Illumination-Robust Visual-Inertial Odometry Based on Adaptive Weighting Algorithm With Two-Layer Confidence Maximization`。
- 截至本次检索，没有定位到论文作者公开的 IR-VIO 源码仓库或可下载 release；论文正文只给出基于 VINS-Mono 的方法和实验结果。因此当前目录没有把第三方实现冒充为 IR-VIO 官方代码。
- 官方 VINS-Mono 仓库：<https://github.com/HKUST-Aerial-Robotics/VINS-Mono>。
- 当前官方 master（检索快照）提交：`90dabb5ec79946ae42fd2e1e91d4e69aabe1e25d`。
- 可完整获取的构建源码：<https://gitee.com/li_xi_yu/VINS-Mono.git>，提交 `8a7a9622a8c41feb3bc909cf138edbb067f83ca3`（2017-12-06），作为历史 VINS-Mono 基线镜像。

官方仓库 clone 在当前网络环境中反复 TLS 超时；Gitee 镜像完整可用。当前构建只将旧 OpenCV API 迁移到 Ubuntu 20.04/OpenCV 4，不修改 VINS-Mono 估计模型或论文算法。
