# 数据集来源与校验

## 官方数据集

- EuRoC MAV 官方入口：<https://projects.asl.ethz.ch/datasets/euroc-mav/>
- 新 ETH Research Collection 条目：<https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f>
- 论文协议：11 条序列、左相机、20 Hz 图像、200 Hz IMU、关闭回环；ATE RMSE 用于比较。

## 当前下载镜像

### V1_01_easy（ASL 格式）

- 页面：<https://huggingface.co/datasets/pepijn223/euroc-mirror>
- 工作区下载地址：`https://hf-mirror.com/datasets/pepijn223/euroc-mirror/resolve/main/V1_01_easy.zip?download=true`
- 文件大小：1,149,702,102 bytes
- SHA-256：`A920FE5B5E69A6AD19B32F1CFAF90AC2F59D45DFD1CA18CC9722E07684BA45FA`

### MH_01_easy（ROS1 bag）

- 页面：<https://huggingface.co/datasets/kavehsgh/EuRoC_MAV_Dataset_Machine_Hall_Easy_01>
- 工作区下载地址：`https://hf-mirror.com/datasets/kavehsgh/EuRoC_MAV_Dataset_Machine_Hall_Easy_01/resolve/main/MH_01_easy.bag?download=true`
- 文件大小：2,673,818,914 bytes
- SHA-256：`57F440CCD68EC8DC8F9461269F5909656B86198BAC3ADFD677B1FCC7A1428FA9`
- `rosbag info`：186 s，`/cam0/image_raw` 3682 条，`/imu0` 36820 条，`/leica/position` 3099 条。

分段下载工具为 `scripts/download_hf_range.ps1`，支持 HTTP Range、断点续传和校验后拼接；下载产生的 `.chunks` 缓存可用于中断后继续。

### EuRoC 分组 ASL 镜像

- 页面：<https://huggingface.co/datasets/GlowBond/EuRoC_MAV_Dataset>
- Vicon Room 1：`vicon_room1.zip`，6,042,263,426 bytes，SHA-256 `FE73C27BE6DC8AC00493B78B750D36B144DAF49EEA7FDF3163E934527C1B5297`。
- Vicon Room 2：`vicon_room2.zip`，6,013,384,949 bytes，SHA-256 `6DAF2CBC2DE9A6BC4E02866C99ED01C29A5C7C164756F06C4F72656192977CFC`。
- Machine Hall：`machine_hall.zip`，12,683,729,426 bytes，SHA-256 `5ED7D07903F8D19B6C8808E2AE8A0872B281F6E34EF5497023B8AC58C3DE0F6F`。
- 实际下载使用 `https://hf-mirror.com/datasets/GlowBond/EuRoC_MAV_Dataset/resolve/main/<archive>?download=true`。

三个分组包中只提取各序列的 ASL 内层 ZIP；没有为运行复制外层包中的 rosbag。当前 `datasets/` 下已具备 MH_01—MH_05、V1_01—V1_03、V2_01—V2_03 的 `mav0/cam0`、`mav0/imu0` 和 `mav0/state_groundtruth_estimate0`。
