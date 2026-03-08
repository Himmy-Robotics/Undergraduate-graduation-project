#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cheetah Pickle Importer
=======================
将AcinoSet猎豹pickle文件转换为npy格式，用于后续retargetting。

使用方法：
1. 修改 load_config.json 中的 file_name, clip, remarks
2. 运行: python pickle_importer.py

猎豹关键点定义（20个）：
- 0: l_eye, 1: r_eye, 2: nose
- 3: neck_base, 4: spine, 5: tail_base, 6: tail_mid, 7: tail_tip
- 8: l_shoulder, 9: l_front_knee, 10: l_front_ankle
- 11: r_shoulder, 12: r_front_knee, 13: r_front_ankle  
- 14: l_hip, 15: l_back_knee, 16: l_back_ankle
- 17: r_hip, 18: r_back_knee, 19: r_back_ankle
"""

import os
import json
import pickle
import numpy as np

# 猎豹关键点索引
CHEETAH_KEYPOINTS = {
    'l_eye': 0, 'r_eye': 1, 'nose': 2, 'neck_base': 3, 'spine': 4,
    'tail_base': 5, 'tail_mid': 6, 'tail_tip': 7,
    'l_shoulder': 8, 'l_front_knee': 9, 'l_front_ankle': 10,
    'r_shoulder': 11, 'r_front_knee': 12, 'r_front_ankle': 13,
    'l_hip': 14, 'l_back_knee': 15, 'l_back_ankle': 16,
    'r_hip': 17, 'r_back_knee': 18, 'r_back_ankle': 19,
}

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), "load_config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def load_pickle(filepath):
    """加载pickle文件"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    positions = data['positions']
    if isinstance(positions, list):
        positions = np.array(positions)
    
    dx = data.get('dx', None)
    if dx is not None and isinstance(dx, list):
        dx = np.array(dx)
    
    return positions, dx

def normalize_positions(positions):
    """
    归一化位置数据：
    - 将第一帧的spine位置作为原点
    - 调整坐标系使猎豹朝向X正方向
    """
    n_frames = positions.shape[0]
    
    # 使用spine(4)作为root参考点
    spine_idx = CHEETAH_KEYPOINTS['spine']
    root_pos = positions[:, spine_idx, :].copy()
    
    # 减去第一帧的root位置，使从原点开始
    positions = positions - root_pos[0:1, :]
    
    return positions, root_pos

def main():
    print("=" * 60)
    print("Cheetah Pickle Importer")
    print("=" * 60)
    
    # 加载配置
    config = load_config()
    file_name = config["file_name"]
    clip = config["clip"]
    remarks = config["remarks"]
    fps = config.get("fps", 120)
    
    print(f"File: {file_name}")
    print(f"Clip: {clip[0]} - {clip[1]}")
    print(f"FPS: {fps}")
    
    # 输入输出路径
    pickle_dir = "../data/cheetah_data_pickle"
    output_dir = "../data/cheetah_data_npy"
    os.makedirs(output_dir, exist_ok=True)
    
    pickle_path = os.path.join(pickle_dir, f"{file_name}.pickle")
    
    # 加载pickle
    print(f"\nLoading: {pickle_path}")
    positions, dx = load_pickle(pickle_path)
    print(f"  Original shape: {positions.shape}")
    
    # 裁剪
    start, end = clip
    positions = positions[start:end]
    if dx is not None:
        dx = dx[start:end]
    print(f"  Clipped shape: {positions.shape}")
    
    # 归一化
    positions, root_pos = normalize_positions(positions)
    
    # 保存为npy格式
    output_name = f"{file_name}_{clip[0]}_{clip[1]}_{remarks}"
    output_path = os.path.join(output_dir, f"{output_name}.npy")
    
    output_data = {
        'positions': positions,  # (N, 20, 3)
        'root_pos': root_pos,    # (N, 3)
        'dx': dx,                # (N, 25) velocities
        'fps': fps,
        'keypoints': CHEETAH_KEYPOINTS,
    }
    
    np.save(output_path, output_data)
    print(f"\nSaved: {output_path}")
    print(f"  Frames: {positions.shape[0]}")
    print(f"  Keypoints: {positions.shape[1]}")
    
    print("\n转换完成！")

if __name__ == "__main__":
    main()
