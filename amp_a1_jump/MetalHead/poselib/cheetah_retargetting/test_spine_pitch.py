#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脊柱pitch角计算方法对比
对比：1. 旋转矩阵方法  2. 直接几何方法（三点计算）
"""

import os
import sys
import numpy as np
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSELIB_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, POSELIB_DIR)

from cheetah_skeleton import reorder_positions, CHEETAH_KP

# 配置
pickle_dir = os.path.join(SCRIPT_DIR, "..", "data", "cheetah_data_pickle")
pickle_file = "gallop1_0-164.pickle"
pickle_path = os.path.join(pickle_dir, pickle_file)

print("=" * 60)
print("脊柱Pitch角计算方法对比测试")
print("=" * 60)
print(f"数据文件: {pickle_file}")

# 加载数据
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)
positions = np.array(data['positions'])
positions = reorder_positions(positions)
print(f"帧数: {positions.shape[0]}")

# 提取关键点
tail_pos = positions[:, CHEETAH_KP['tail_base'], :]
spine_pos = positions[:, CHEETAH_KP['spine'], :]
neck_pos = positions[:, CHEETAH_KP['neck_base'], :]

seq_len = positions.shape[0]
print(f"处理 {seq_len} 帧数据...")

# ==================== 辅助函数 ====================
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else v

def rotation_matrix_to_euler(R):
    """旋转矩阵转欧拉角 [roll, pitch, yaw]"""
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0
    return np.array([roll, pitch, yaw])

# ==================== 方法1: 旋转矩阵方法 ====================
print("\n方法1: 旋转矩阵方法（当前实现）")

l_hip_pos = positions[:, CHEETAH_KP['l_hip'], :]
r_hip_pos = positions[:, CHEETAH_KP['r_hip'], :]
l_shoulder_pos = positions[:, CHEETAH_KP['l_shoulder'], :]
r_shoulder_pos = positions[:, CHEETAH_KP['r_shoulder'], :]

pitch_method1 = np.zeros(seq_len)

for i in range(seq_len):
    # neck_base旋转矩阵
    f_nb = neck_pos[i] - spine_pos[i]
    x_nb = normalize(f_nb)
    s_nb = l_shoulder_pos[i] - r_shoulder_pos[i]
    z_nb = normalize(np.cross(x_nb, normalize(s_nb)))
    y_nb = np.cross(z_nb, x_nb)
    R_nb = np.column_stack([x_nb, y_nb, z_nb])
    
    # spine旋转矩阵
    f_sp = spine_pos[i] - tail_pos[i]
    x_sp = normalize(f_sp)
    s_sp = l_hip_pos[i] - r_hip_pos[i]
    z_sp = normalize(np.cross(x_sp, normalize(s_sp)))
    y_sp = np.cross(z_sp, x_sp)
    R_sp = np.column_stack([x_sp, y_sp, z_sp])
    
    # 局部旋转
    R_local = R_nb.T @ R_sp
    euler = rotation_matrix_to_euler(R_local)
    pitch_method1[i] = euler[1]

print(f"  Pitch范围: [{np.degrees(pitch_method1.min()):.2f}°, {np.degrees(pitch_method1.max()):.2f}°]")
print(f"  Pitch幅度: {np.degrees(pitch_method1.max() - pitch_method1.min()):.2f}°")

# ==================== 方法2: 直接几何方法 ====================
print("\n方法2: 直接几何方法（三点计算）")

pitch_method2 = np.zeros(seq_len)

for i in range(seq_len):
    # 前半段: neck_base到spine
    vec_front = spine_pos[i] - neck_pos[i]
    pitch_front = np.arctan2(vec_front[2], np.sqrt(vec_front[0]**2 + vec_front[1]**2))
    
    # 后半段: spine到tail_base
    vec_back = tail_pos[i] - spine_pos[i]
    pitch_back = np.arctan2(vec_back[2], np.sqrt(vec_back[0]**2 + vec_back[1]**2))
    
    # 脊柱弯曲角 = 后半段pitch - 前半段pitch
    pitch_method2[i] = pitch_back - pitch_front

print(f"  Pitch范围: [{np.degrees(pitch_method2.min()):.2f}°, {np.degrees(pitch_method2.max()):.2f}°]")
print(f"  Pitch幅度: {np.degrees(pitch_method2.max() - pitch_method2.min()):.2f}°")

# ==================== 对比分析 ====================
print("\n" + "=" * 60)
print("对比分析")
print("=" * 60)

diff = pitch_method1 - pitch_method2
print(f"差异: 均值={np.degrees(np.mean(diff)):.2f}°, 最大={np.degrees(np.max(np.abs(diff))):.2f}°")

print(f"\n方法1幅度/方法2幅度 = {(pitch_method1.max()-pitch_method1.min())/(pitch_method2.max()-pitch_method2.min()):.2%}")

print("\n结论:")
if np.abs(pitch_method2.max()-pitch_method2.min()) > np.abs(pitch_method1.max()-pitch_method1.min()):
    print("  方法2（几何法）产生更大的pitch变化幅度，可能更适合捕捉猎豹脊柱弯曲")
else:
    print("  方法1（旋转矩阵法）产生更大的pitch变化幅度")
