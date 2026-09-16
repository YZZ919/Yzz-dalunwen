# 灰度 DCE 训练数据说明

## 当前可复现实验源

官方 SICE 仓库给出的 Part1 下载入口是 Google Drive；该入口在当前机器上无法稳定访问。当前工作区先使用公开的 `okhater/SICE` 镜像派生数据做 clean-room 训练通路验证：

官方 SICE README 明确 Part1 是 360 个图像序列；下载入口为 [Google Drive](https://goo.gl/gTGfLk) 或 [BaiduYun](https://pan.baidu.com/s/1kXotehL)。README 还说明完整 SICE 数据集共 589 个序列、4,413 张图像。当前没有从这些入口取得可验证的官方 2,422/600 训练清单，因此不把镜像 ID 映射为官方场景 ID。

- 数据卡：<https://huggingface.co/datasets/okhater/SICE>
- 工作区镜像入口：<https://hf-mirror.com/datasets/okhater/SICE>
- 下载脚本：`scripts/download_sice_mirror.py`
- 灰度整理脚本：`scripts/prepare_sice_gray.py`

镜像 API 当前列出 259 个 `train` sample 和 47 个 `test` sample。下载脚本默认每个 sample 只取 `low1.jpg`，不下载 `label.jpg` 或其余曝光版本；整理脚本再把输入转成单通道 JPEG。因此这不是论文中原始 SICE Part1 的 2422/600 图像拆分，不能把镜像训练得到的 checkpoint 写成“作者权重”。

`scripts/audit_sice_split.py` 会对本地镜像重新统计 sample/image 数量、训练/验证 ID 是否重叠、曝光文件名以及文件 SHA-256；本次审计结果为 259/47 个 sample、每个 sample 仅 1 张 `low1.jpg`、ID 和路径均无交集，报告在 `results/sice_split_audit_20260911.json`。这只能证明本地镜像内部没有明显泄漏，不能证明它等同官方 Part1。

### 镜像全低曝光增补（独立实验源）

为验证“每个场景只取 `low1`”是否造成数据量偏差，下载器新增了
`--selection all_low`。它保留同一个 `sample_*` 目录下的所有 `low*.jpg`，但仍把
train/test 场景完全分开；这不是把 train/test 图像随机重切。镜像 API 当前可列出
1197 张低曝光图（1021 train、176 test，覆盖 259/47 个场景），与默认 low1 的
306 张不同。对应源目录为 `datasets/sice_mirror_all_low/`，灰度整理时使用
`scripts/prepare_sice_gray.py --selection all_low`，输出另存为
`datasets/sice_gray_all_low/`。该增补仍然是镜像派生数据，不能声称等同官方
SICE Part1 的 2422/600；它只用于 B 组“同一实现 + 更多镜像曝光”的受控比较。

下载清单保存在 `datasets/sice_mirror_all_low/download_manifest.json`，场景/图像审计
保存在 `results/sice_all_low_split_audit_20260911.json`，灰度转换清单保存在
`datasets/sice_gray_all_low/dataset_manifest.json`。本组的 train/test sample ID
交集为空；同一 sample 的多张曝光图都留在同一个 split 内。

## 工作区目录

```text
datasets/
├─ sice_mirror_low1/       # 镜像低光输入（可重新下载）
└─ sice_gray/              # 训练实际读取的灰度图
```

小规模 sanity 数据放在 `datasets/sice_mirror_low1_sanity/` 和 `datasets/sice_gray_sanity3/`，仅用于验证脚本和 GPU，不计入正式结果。

## 完整镜像下载

在 WSL 中从工作区执行：

```bash
python3 scripts/download_sice_mirror.py \
  --output-root /mnt/d/大论文实验/IR-VIO的复现/datasets/sice_mirror_low1
python3 scripts/prepare_sice_gray.py \
  --source-root /mnt/d/大论文实验/IR-VIO的复现/datasets/sice_mirror_low1 \
  --output-root /mnt/d/大论文实验/IR-VIO的复现/datasets/sice_gray
```

脚本会写出 `download_manifest.json`、`dataset_manifest.json` 和每张整理后图像的 SHA-256。若后续取得官方 Part1 压缩包，只需把 `--source-root` 指向官方 train/val（或 train/test）目录重新整理，并保留原始来源记录。
