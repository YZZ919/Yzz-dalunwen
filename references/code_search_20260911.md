# IR-VIO 公开代码检索记录（2026-09-11）

本轮继续检索论文标题、DOI、作者姓名与 `IR-VIO` 组合关键词，仍未发现作者公开的
IR-VIO 源码仓库或单通道增强 checkpoint。以下页面只能确认论文/作者公开记录：

- ResearchGate 论文记录：<https://www.researchgate.net/publication/371371613_IR-VIO_Illumination-Robust_Visual-Inertial_Odometry_Based_on_Adaptive_Weighting_Algorithm_With_Two-Layer_Confidence_Maximization>
- 南开大学团队录用新闻：<https://rh.nankai.edu.cn/info/1009/1181.htm>
- 作者 ResearchGate 作品页：<https://www.researchgate.net/scientific-contributions/Zhixing-Song-2238741245>

这些页面没有给出可下载的 IR-VIO 源码仓库；南开页面提供的是论文信息和补充视频链接。
因此当前工作区继续把实现标为 `clean-room proxy`，不把 VINS-Mono 历史镜像或官方
Zero-DCE 权重误称为 IR-VIO 作者代码。若后续获得作者邮件回复、私有仓库或补充材料，
应新建独立的 `official_irvio` 结果目录，不覆盖现有结果。
