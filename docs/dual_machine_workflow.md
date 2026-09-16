# RTX 4060 + RTX 4090 双机工作流

## 分工

- Windows/RTX 4060：代码修改、小样本调试、配置检查和结果整理。
- Ubuntu/RTX 4090：正式训练、长时间批量实验和 VIO 回放。
- Git：只同步源码、配置、文档和实验元数据。
- 对象存储或受控文件传输：同步 datasets、checkpoint、完整日志、视频和逐帧结果。

## 当前仓库边界

本目录现在有三个 Git 边界，不能直接用根目录的 `git add -A` 把它们混成一个不完整提交：

| 路径 | 用途 | 当前处理方式 |
|---|---|---|
| 项目根目录 | 脚本、配置、文档、实验索引 | 新建的主仓库 |
| `src/VINS-Mono` | VIO 源码，含本项目改动 | 保留原独立仓库；根仓库忽略 |
| `src/Noise-AwareCameraExposureControl` | 参考实现 | 保留原独立仓库；根仓库忽略 |

`src/VINS-Mono` 当前基于 Gitee 上游，并有未提交改动。不要把这些改动直接推到上游 `origin`。建议先在该仓库创建项目分支并提交，再把一个有写权限的私人 fork 配成新 remote。根项目也应使用私人 remote。获得这两个 remote 之前，4090 端不能只靠一次根仓库 clone 完整恢复当前源码状态。

## Windows/4060 日常流程

在 PowerShell 中：

```powershell
Set-Location 'D:\大论文实验\IR-VIO的复现'
git status --short --branch
git -C src\VINS-Mono status --short --branch
```

开始实验前，从模板创建唯一目录：

```powershell
$exp = 'exp001_gray-dce_YYYYMMDD'
Copy-Item -LiteralPath experiments\_template -Destination ("experiments\" + $exp) -Recurse
```

填写 `config.yaml` 和 `command.txt` 后再运行。正式结束后填写 `metrics.json` 与实验 README。数据、模型和完整输出仍留在被忽略的本地目录，并在 README 中记录对象存储 URI 或受控路径。

## 首次接入私人远程仓库

用户确认远程地址后，在项目根目录执行：

```bash
git remote add origin <PROJECT_PRIVATE_GIT_URL>
git push -u origin main
```

VINS-Mono 改动应在它自己的仓库中单独提交，并推送到私人 fork，而不是当前上游：

```bash
git -C src/VINS-Mono switch -c irvio-reproduction
git -C src/VINS-Mono add -A
git -C src/VINS-Mono commit -m "Add IR-VIO clean-room integration"
git -C src/VINS-Mono remote add project <VINS_PRIVATE_FORK_URL>
git -C src/VINS-Mono push -u project irvio-reproduction
```

提交前务必先审阅 `git -C src/VINS-Mono diff`；上面的命令是确认后的流程，不是要求盲目执行。

## 4090 首次部署

```bash
git clone <PROJECT_PRIVATE_GIT_URL> ir-vio-reproduction
cd ir-vio-reproduction
git clone --branch irvio-reproduction <VINS_PRIVATE_FORK_URL> src/VINS-Mono
git clone https://github.com/UkcheolShin/Noise-AwareCameraExposureControl.git src/Noise-AwareCameraExposureControl
bash scripts/setup_ubuntu_4090.sh --check-only
bash scripts/setup_ubuntu_4090.sh
```

若工作站是 Ubuntu 20.04 且需要编译/回放 VIO，再显式安装 ROS 依赖：

```bash
bash scripts/setup_ubuntu_4090.sh --install-ros
```

随后从 OSS、局域网存储或移动硬盘恢复 `datasets/`。不要把数据和 checkpoint 提交到 Git。恢复完成后再次检查：

```bash
bash scripts/check_ubuntu_4090.sh
```

## 4090 更新与训练

每次运行前分别更新项目仓库和 VINS-Mono 项目分支：

```bash
git pull --ff-only
git -C src/VINS-Mono pull --ff-only
bash scripts/check_ubuntu_4090.sh
```

现有灰度 DCE 训练入口保持不变：

```bash
bash scripts/run_gray_dce_training.sh
# 或论文系数 profile
bash scripts/run_gray_dce_paper_training.sh
```

这两个脚本的算法参数没有因项目整理而改变。它们把权重写到 `checkpoints/`、日志写到 `logs/`，两者都不进入 Git。正式运行前把实际命令和代码/数据版本写入新的 `experiments/<id>/`，运行后把最佳 epoch、验证指标和 artifact URI 写回该目录。

## 结果回传

小型汇总文件和实验元数据通过 Git：

```bash
git add experiments/<id>
git commit -m "Record <id> results"
git push
```

checkpoint、完整日志和大规模结果通过对象存储。推荐保持以下逻辑前缀：

```text
ir-vio/
  datasets/
  checkpoints/<experiment-id>/
  logs/<experiment-id>/
  results/<experiment-id>/
```

下载后用大小或 SHA-256 校验，并把校验值记录到实验 README 或 `config.yaml`。

