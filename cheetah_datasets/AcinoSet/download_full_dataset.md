# AcinoSet 完整数据集下载指南

## 数据来源

AcinoSet 提供了猎豹的高质量3D位姿数据，共包含：
- **119,490帧** 的2D关键点估计数据（H5/CSV格式）
- **已处理好的FTE 3D位姿数据**（`fte.pickle`格式）

## 下载链接

### Dropbox 主链接
https://www.dropbox.com/sh/kp5kmatbv5cdjx2/AABfJGb7ktVK_L0lybOLQIbJa?dl=0

### 数据组织结构
数据按 `日期 > 动物ID > 奔跑序号` 组织，例如：
```
2019_03_09/
  lily/
    run/
      fte.pickle          # ← 这是我们需要的3D位姿数据
      n_cam_scene_sba.json  # 相机配置
      CamN_*.h5           # 各相机的2D关键点
```

## 需要下载的文件

只需下载 `fte.pickle` 文件即可获得3D位姿数据：

### FTE文件格式说明
```python
import pickle
with open('fte.pickle', 'rb') as f:
    data = pickle.load(f)

# data 包含:
# - positions: (N, 20, 3) - 20个关键点的3D位置
# - x: (N, 45) - 状态向量 (15个关键点 × 3维)
# - dx: (N, 45) - 速度
# - ddx: (N, 45) - 加速度
```

## 批量下载建议

### 方法1: 使用浏览器
1. 打开Dropbox链接
2. 找到包含 `fte.pickle` 的文件夹
3. 下载整个文件夹

### 方法2: 使用命令行 (需要安装 rclone)
```bash
# 安装 rclone
curl https://rclone.org/install.sh | sudo bash

# 配置 Dropbox
rclone config

# 下载 fte.pickle 文件
rclone copy dropbox:AcinoSet ./data --include "fte.pickle"
```

### 方法3: 使用 wget (如果Dropbox允许直接下载)
对于单个文件，可以添加 `?dl=1` 参数：
```bash
wget "https://www.dropbox.com/path/to/fte.pickle?dl=1" -O fte.pickle
```

## 可用的猎豹运动序列

根据README和数据目录，可能包含以下动物的数据：
- sportCentreNaoya (Ex1, Ex2)
- thursday_kiara
- thursday_elliot  
- sunday_amelia
- lily (2019_03_09)
- 其他日期的数据...

## 将2D转换为3D（如果需要）

如果某些序列没有 `fte.pickle`，可以使用 AcinoSet 的工具进行3D重建：

```bash
# 需要多个相机的同步2D数据
python all_optimizations.py \
    --data_dir 2019_03_09/lily/run \
    --start_frame 70 \
    --end_frame 170 \
    --dlc_thresh 0.5
```

## 注意事项

1. 完整数据集较大（视频 + H5），建议只下载需要的 `fte.pickle` 文件
2. FTE (Full Trajectory Estimation) 是最准确的3D重建方法
3. 数据帧率约为 120 FPS
4. 猎豹体型比 Himmy 大约 1.47 倍（体长），需要进行缩放

## 引用

如果使用此数据集，请引用：
```bibtex
@misc{joska2021acinoset,
      title={AcinoSet: A 3D Pose Estimation Dataset and Baseline Models for Cheetahs in the Wild}, 
      author={Daniel Joska and Liam Clark and Naoya Muramatsu and Ricardo Jericevich and Fred Nicolls and Alexander Mathis and Mackenzie W. Mathis and Amir Patel},
      year={2021},
      eprint={2103.13282},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
