# 实验记录规范

每次正式实验使用一个不可复用的独立目录，推荐命名：

```text
experiments/expNNN_<short-name>_<YYYYMMDD>/
```

从 `_template/` 复制后再填写，至少保留：

- `config.yaml`：机器、代码版本、数据版本和训练参数。
- `command.txt`：实际执行的完整命令；不要只写计划命令。
- `metrics.json`：最终指标、最佳 epoch 和运行状态。
- `README.md`：实验目的、结论、异常和大文件位置。

checkpoint、数据集、逐帧图片、视频和完整日志不进入 Git。它们放在本机对应目录或对象存储中，并在实验 README 的 `Artifact URI` 字段记录可恢复的位置。日志中有助于复核的短摘要可以复制进实验目录。

不要覆盖旧实验目录。需要续训时，在原实验 README 中记录 `resume` checkpoint 和新命令；需要改变参数时，新建实验目录。

建议在开跑前记录三个代码版本：

```bash
git rev-parse HEAD
git -C src/VINS-Mono rev-parse HEAD
git -C src/Noise-AwareCameraExposureControl rev-parse HEAD
```

如果任一仓库有未提交改动，也要把 `git status --short` 写入实验 README。只有 commit ID 而没有 dirty 状态，不能完整标识一次实验。

