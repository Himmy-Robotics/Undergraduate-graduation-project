#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: cheetah_key_points_retargetting.py
@Auth: Based on Himmy retargeting, modified for Cheetah data
@Date: 2024/12

Cheetah to Himmy Mark2 Motion Retargeting (新方案)
===================================================

该模块实现了将猎豹动捕数据重定向到Himmy Mark2机器人的功能。

新方案核心思路：
--------------
1. 猎豹的neck_base对应Himmy的trunk（base_link）
2. 计算spine相对于neck_base的局部旋转，映射到Himmy脊柱关节
3. trunk位置确定后，前腿hip位置已知，通过前脚IK求前腿关节角
4. 脊柱关节角应用后，通过正向运动学得到后腿hip位置，再IK求后腿关节角

映射链：
- neck_base全局位姿 → trunk位姿 → 前腿IK
- spine相对neck_base的roll/pitch/yaw → yaw_spine/pitch_spine/roll_spine关节角
- 脊柱FK → roll_spine_Link位置 → 后腿hip位置 → 后腿IK

Source骨架结构（AcinoSet猎豹模型，20个关键点）：
- nose, neck_base, spine, tail_base, tail_mid, tail_tip (脊柱)
- l_eye, r_eye (眼睛)
- l_shoulder -> l_front_knee -> l_front_ankle (前左腿)
- r_shoulder -> r_front_knee -> r_front_ankle (前右腿)
- l_hip -> l_back_knee -> l_back_ankle (后左腿)
- r_hip -> r_back_knee -> r_back_ankle (后右腿)

Target骨架结构（Himmy Mark2，20个node）：
- trunk (root)
- FL_hip_Link -> FL_thigh_Link -> FL_calf_Link -> FL_foot (前左腿)
- FR_hip_Link -> FR_thigh_Link -> FR_calf_Link -> FR_foot (前右腿)
- yaw_spine_Link -> pitch_spine_Link -> roll_spine_Link (脊柱)
- RL_hip_Link -> RL_thigh_Link -> RL_calf_Link -> RL_foot (后左腿)
- RR_hip_Link -> RR_thigh_Link -> RR_calf_Link -> RR_foot (后右腿)

关键映射（新方案）：
- cheetah neck_base → himmy trunk (base_link)
- cheetah spine相对neck_base的局部旋转 → himmy spine joints (yaw, pitch, roll)
- cheetah l_front_ankle → himmy FL_foot
- cheetah r_front_ankle → himmy FR_foot
- cheetah l_back_ankle → himmy RL_foot
- cheetah r_back_ankle → himmy RR_foot
"""
import copy
import json
import os
import sys
import pickle
import numpy as np

# 添加poselib路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSELIB_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, POSELIB_DIR)

from cheetah_kinematics import HimmyKinematics
from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# ==================== 配置文件加载 ====================

json_dir = os.path.join(SCRIPT_DIR, "load_config.json")
with open(json_dir, "r") as f:
    motion_json = json.load(f)
    file_name = motion_json["file_name"]
    clip = motion_json["clip"]
    remarks = motion_json["remarks"]

# 源数据目录（cheetah_data_npy）
source_dir = os.path.join(SCRIPT_DIR, "..", "data", "cheetah_data_npy")
# 输出目录（cheetah_data_retarget_npy）
output_dir = os.path.join(SCRIPT_DIR, "..", "data", "cheetah_data_retarget_npy")
os.makedirs(output_dir, exist_ok=True)
output_file = '{}_amp_{}_{}_{}'.format(file_name, clip[0], clip[1], remarks)


# ==================== Himmy Mark2机器人参数（从T-pose提取）====================

# 躯体参数
FRONT_BODY_LENGTH = 0.0432      # trunk到前腿hip的X距离
BODY_WIDE = 0.2                 # 机身宽度（左右hip Y方向间距）

# 脊柱参数
SPINE_OFFSET_X = -0.03855       # trunk到yaw_spine的X偏移
SPINE_YAW_LENGTH = 0.12         # yaw_spine的长度
SPINE_PITCH_LENGTH = 0.10528    # pitch_spine的长度

# 后躯体参数
REAR_BODY_LENGTH = 0.0655       # roll_spine到后腿hip的X距离

# 机体总长度（从T-pose计算：FL_foot.x - RL_foot.x ≈ 0.54m）
# 用于计算trunk偏移量，与A1的处理方式一致
BODY_LENGTH = 0.54              # 前后足之间的总长度

# 腿部参数
HIP_OFFSET_X = 0.083275         # hip到thigh的X偏移
HIP_OFFSET_Y = 0.07955          # hip到thigh的Y偏移
THIGH_LENGTH = 0.23             # 大腿长度
CALF_LENGTH = 0.279978          # 小腿长度

# 初始化运动学求解器
kinematic = HimmyKinematics(
    front_body_length=FRONT_BODY_LENGTH,
    rear_body_length=REAR_BODY_LENGTH,
    spine_length_yaw=SPINE_YAW_LENGTH,
    spine_length_pitch=SPINE_PITCH_LENGTH,
    spine_offset_x=SPINE_OFFSET_X,
    body_wide=BODY_WIDE,
    hip_offset_x=HIP_OFFSET_X,
    hip_offset_y=HIP_OFFSET_Y,
    thigh_length=THIGH_LENGTH,
    calf_length=CALF_LENGTH
)


# ==================== 辅助函数 ====================

def quat_from_euler_xyz_numpy(roll, pitch, yaw):
    """将欧拉角转换为四元数（numpy版本）"""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([x, y, z, w])


def spine_forward_kinematics(spine_angles, base_pos, base_rot):
    """
    计算脊柱弯曲后，roll_spine_Link在世界坐标系下的位姿
    
    脊柱链: trunk -> yaw_spine -> pitch_spine -> roll_spine
    
    Args:
        spine_angles: [yaw, pitch, roll] 三个脊柱关节角度（弧度）
        base_pos: trunk在世界坐标系下的位置 [x, y, z]
        base_rot: trunk在世界坐标系下的欧拉角 [roll, pitch, yaw]
    
    Returns:
        roll_spine_pos_w: roll_spine_Link在世界坐标系下的位置
        roll_spine_rot_w: roll_spine_Link在世界坐标系下的欧拉角
    """
    yaw_ang, pitch_ang, roll_ang = spine_angles
    
    # trunk到世界的变换
    T_w_trunk = kinematic.trans_matrix_ba(base_pos, base_rot)
    
    # trunk到yaw_spine的变换
    yaw_spine_pos = np.array([SPINE_OFFSET_X, 0, 0])
    yaw_spine_rot = [0, 0, yaw_ang]
    T_trunk_yaw = kinematic.trans_matrix_ba(yaw_spine_pos, yaw_spine_rot)
    
    # yaw_spine到pitch_spine的变换
    pitch_spine_pos = np.array([-SPINE_YAW_LENGTH, 0, 0])
    pitch_spine_rot = [0, pitch_ang, 0]
    T_yaw_pitch = kinematic.trans_matrix_ba(pitch_spine_pos, pitch_spine_rot)
    
    # pitch_spine到roll_spine的变换
    roll_spine_pos = np.array([-SPINE_PITCH_LENGTH, 0, 0])
    roll_spine_rot = [roll_ang, 0, 0]
    T_pitch_roll = kinematic.trans_matrix_ba(roll_spine_pos, roll_spine_rot)
    
    # 完整变换链
    T_w_roll = T_w_trunk @ T_trunk_yaw @ T_yaw_pitch @ T_pitch_roll
    
    # 提取位置
    roll_spine_pos_w = T_w_roll[:3, 3]
    
    # 提取旋转
    R = T_w_roll[:3, :3]
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    roll_spine_rot_w = np.array([roll, pitch, yaw])
    
    return roll_spine_pos_w, roll_spine_rot_w




# ==================== 数据加载 ====================

print("=" * 60)
print("Himmy Mark2 Motion Retargeting（脊柱感知版本）")
print("=" * 60)

# ==================== 尺寸缩放计算 ====================

# 加载T-pose文件（使用标准T-pose而不是运动数据的第一帧）
# 猎豹T-pose和Himmy T-pose都从t-pose目录加载
source_tpose = SkeletonState.from_file(os.path.join(SCRIPT_DIR, 't-pose', 'cheetah_tpose_skeleton.npy'))
target_tpose = SkeletonState.from_file(os.path.join(SCRIPT_DIR, 't-pose', 'amp_himmy_mark2_tpose.npy'))

skeleton_s = source_tpose.skeleton_tree
skeleton_t = target_tpose.skeleton_tree

# 计算source尺寸（猎豹数据，单位已经是米）
# 使用猎豹的关节名称
source_length = torch.abs(source_tpose.global_translation[0, skeleton_s.index('l_front_ankle'), 0] -
                          source_tpose.global_translation[0, skeleton_s.index('l_back_ankle'), 0])
source_wide = torch.abs(source_tpose.global_translation[0, skeleton_s.index('l_front_ankle'), 1] -
                        source_tpose.global_translation[0, skeleton_s.index('r_front_ankle'), 1])
source_height = torch.abs(source_tpose.global_translation[0, skeleton_s.index('tail_base'), 2] -
                          source_tpose.global_translation[0, skeleton_s.index('l_back_ankle'), 2])

# 计算target T-pose的关键尺寸
target_length = torch.abs(target_tpose.global_translation[skeleton_t.index('FL_foot'), 0] -
                          target_tpose.global_translation[skeleton_t.index('RL_foot'), 0])
target_wide = torch.abs(target_tpose.global_translation[skeleton_t.index('FL_foot'), 1] -
                        target_tpose.global_translation[skeleton_t.index('FR_foot'), 1])
target_height = torch.abs(target_tpose.global_translation[skeleton_t.index('trunk'), 2] -
                          target_tpose.global_translation[skeleton_t.index('FL_foot'), 2])

print(f'\n=== T-pose 尺寸调试 ===')
print(f'Source T-pose: length={source_length:.4f}, wide={source_wide:.4f}, height={source_height:.4f}')
print(f'Target T-pose: length={target_length:.4f}, wide={target_wide:.4f}, height={target_height:.4f}')
print(f'Target FL_foot Y: {target_tpose.global_translation[skeleton_t.index("FL_foot"), 1]:.4f}')
print(f'Target FR_foot Y: {target_tpose.global_translation[skeleton_t.index("FR_foot"), 1]:.4f}')

# 计算缩放比例
# 使用各向异性缩放，保持原始动作数据相对于T-pose的比例
# 这样可以真实还原猎豹的步态特征
zoom_x = (target_length / source_length) * 1.0
zoom_y = (target_wide / source_wide) * 0.3  # 各向异性缩放，Y方向按宽度比例
zoom_z = target_height / source_height

print(f'\n=== 缩放策略：各向异性（保持原始比例）===')
print(f'  zoom_x (长度): {zoom_x:.4f}')
print(f'  zoom_y (宽度): {zoom_y:.4f}')
print(f'  zoom_z (高度): {zoom_z:.4f}')

# ==================== 计算 T-pose 中的 x 轴偏移 ====================
# 新方案：neck_base 对应 trunk，需要计算前脚位置的对齐偏移
# 目的：当 source neck_base 和 himmy trunk 重合时，
# 修正两者前脚位置的 x 轴差距，确保映射后脚的位置基本相同

# Source 中：neck_base 的位置 和 r_front_ankle 的 x 坐标差（单位已是米）
source_neck_base_x = source_tpose.global_translation[0, skeleton_s.index('neck_base'), 0]
source_front_foot_x = source_tpose.global_translation[0, skeleton_s.index('r_front_ankle'), 0]
source_neck_to_front_foot_x = source_front_foot_x - source_neck_base_x  # Source: neck_base 到前脚的 x 偏移

# Target T-pose 中：trunk 的位置 和 FR_foot 的 x 坐标差
target_trunk_x = target_tpose.global_translation[skeleton_t.index('trunk'), 0]
target_fr_foot_x = target_tpose.global_translation[skeleton_t.index('FR_foot'), 0]
target_trunk_to_front_foot_x = target_fr_foot_x - target_trunk_x  # Target: trunk 到前脚的 x 偏移

# 应用缩放后的 source 偏移（因为 source motion 会被缩放）
source_neck_to_front_foot_x_scaled = source_neck_to_front_foot_x * zoom_x

# 计算需要补偿的 x 轴偏移
# root_x_offset = source_neck_to_front_foot_x_scaled - target_trunk_to_front_foot_x
root_x_offset = target_trunk_to_front_foot_x - source_neck_to_front_foot_x_scaled


print(f'\nT-pose 前脚 x 轴偏移计算（新方案：neck_base -> trunk）:')
print(f'  Source neck_base -> r_front_ankle x: {source_neck_to_front_foot_x:.4f}m')
print(f'  Source neck_base -> r_front_ankle x (scaled): {source_neck_to_front_foot_x_scaled:.4f}m')
print(f'  Target trunk -> FR_foot x: {target_trunk_to_front_foot_x:.4f}m')
print(f'  Root x offset (补偿值): {root_x_offset:.4f}m')

print('zoom_x: {}, zoom_y: {}, zoom_z: {}'.format(zoom_x, zoom_y, zoom_z))


# ==================== 新方案：直接从pickle读取原始positions ====================
# 注意：global_transformation是通过FK计算的，由于local_rotation不准确会导致位置错误
# 因此直接读取pickle中的原始positions

print('\n' + '='*60)
print('直接从pickle读取原始positions')
print('='*60)

from cheetah_skeleton import reorder_positions, CHEETAH_KP

# 直接加载pickle获取原始positions
pickle_dir = os.path.join(SCRIPT_DIR, "..", "data", "cheetah_data_pickle")
pickle_path = os.path.join(pickle_dir, f"{file_name}.pickle")
print(f"Loading original positions from: {pickle_path}")

with open(pickle_path, 'rb') as f:
    pickle_data = pickle.load(f)
original_positions = np.array(pickle_data['positions'])  # (N, 20, 3)

# 猎豹数据的帧率
SOURCE_FPS = 120

# 重排序为骨架拓扑顺序
original_positions = reorder_positions(original_positions)

# 裁剪
original_positions = original_positions[clip[0]:clip[1]]
print(f"  Loaded positions shape: {original_positions.shape}")

# 坐标系说明：
# 猎豹数据坐标系：X向前，Y向左，Z向上（与Himmy一致）
# 验证：脚踝Z≈0.1-0.2m，躯干Z≈0.6m → Z向上正确
# 不需要任何坐标轴翻转！

# 归一化：每帧相对于该帧的root（保留相对运动）
root_positions = original_positions[:, 0:1, :].copy()  # tail_base在索引0
positions_relative = original_positions - root_positions
first_frame_root = root_positions[0, 0, :].copy()
root_global = root_positions[:, 0, :] - first_frame_root
original_positions = positions_relative + root_global[:, np.newaxis, :]

print(f"  Positions normalized (relative to root per frame)")

# # 验证相对位置在变化
# l_front_ankle_idx = CHEETAH_KP['l_front_ankle']  # 获取重排序后的索引
# rel_pos_0 = original_positions[0, l_front_ankle_idx] - original_positions[0, 0]
# rel_pos_99 = original_positions[99, l_front_ankle_idx] - original_positions[99, 0]
# print(f"  Verification - l_front_ankle relative to tail_base:")
# print(f"    Frame 0: {rel_pos_0}")
# print(f"    Frame 100: {rel_pos_99}")

# 提取关键点的全局位置索引（使用CHEETAH_KP）
lh_idx = CHEETAH_KP["l_front_ankle"]  # 左前脚
rh_idx = CHEETAH_KP["r_front_ankle"]  # 右前脚
lf_idx = CHEETAH_KP["l_back_ankle"]   # 左后脚
rf_idx = CHEETAH_KP["r_back_ankle"]   # 右后脚
root_idx = CHEETAH_KP["tail_base"]    # root
spine_idx = CHEETAH_KP["spine"]
l_hip_idx = CHEETAH_KP["l_hip"]
r_hip_idx = CHEETAH_KP["r_hip"]
l_shoulder_idx = CHEETAH_KP["l_shoulder"]
r_shoulder_idx = CHEETAH_KP["r_shoulder"]
neck_idx = CHEETAH_KP["neck_base"]

print(f'\n缩放前（第一帧）：')
print(f'  l_front_ankle:  Y={original_positions[0, lh_idx, 1]:.4f}m')
print(f'  r_front_ankle:  Y={original_positions[0, rh_idx, 1]:.4f}m')
print(f'  前脚Y间距: {abs(original_positions[0, lh_idx, 1] - original_positions[0, rh_idx, 1]):.4f}m')

# 直接缩放positions（不是global_transformation）
original_positions[:, :, 0] *= zoom_x.item()  # X
original_positions[:, :, 1] *= zoom_y.item()  # Y  
original_positions[:, :, 2] *= zoom_z.item()  # Z

print(f'\n缩放后（第一帧）：')
print(f'  l_front_ankle: Y={original_positions[0, lh_idx, 1]:.4f}m')
print(f'  r_front_ankle: Y={original_positions[0, rh_idx, 1]:.4f}m')
print(f'  l_back_ankle:  Y={original_positions[0, lf_idx, 1]:.4f}m')
print(f'  r_back_ankle:  Y={original_positions[0, rf_idx, 1]:.4f}m')
print(f'  前脚Y间距: {abs(original_positions[0, lh_idx, 1] - original_positions[0, rh_idx, 1]):.4f}m')
print(f'  后脚Y间距: {abs(original_positions[0, lf_idx, 1] - original_positions[0, rf_idx, 1]):.4f}m')
print(f'  Y轴缩放因子: zoom_y = {zoom_y:.4f}')
print('='*60)


# ==================== 提取关键点（使用缩放后的原始positions）====================

# 直接从original_positions提取足端位置
# 猎豹坐标系：Y越大越"左"，与Himmy一致（Y>0是左边）
# 不需要交换左右脚，也不需要翻转Y坐标
g_fl_trans = original_positions[:, lh_idx, :].copy()  # l_front_ankle -> FL (前左)
g_fr_trans = original_positions[:, rh_idx, :].copy()  # r_front_ankle -> FR (前右)
g_rl_trans = original_positions[:, lf_idx, :].copy()  # l_back_ankle -> RL (后左)
g_rr_trans = original_positions[:, rf_idx, :].copy()  # r_back_ankle -> RR (后右)

# Root位置 (tail_base)
g_root_trans = original_positions[:, root_idx, :].copy()

# ==================== 从关节位置构建旋转矩阵 ====================
# 按照用户描述的数学方法：
# 1. tail_base的全局旋转：使用 spine-tail_base 作为前向，hip连线作为侧向
# 2. spine的全局旋转：使用 neck_base-spine 作为前向，shoulder连线作为侧向
# 3. spine的局部旋转 = R_tail_base^T @ R_spine

def normalize(v):
    """归一化向量"""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0])
    return v / norm

def rotation_matrix_to_quaternion(R):
    """将3x3旋转矩阵转换为四元数 [x, y, z, w]"""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)

def quaternion_multiply(q1, q2):
    """四元数乘法 q1 * q2，格式 [x, y, z, w]"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quaternion_inverse(q):
    """四元数求逆 [x, y, z, w] -> [-x, -y, -z, w]"""
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quaternion_to_euler(q):
    """四元数转欧拉角 [x, y, z, w] -> [roll, pitch, yaw]"""
    x, y, z, w = q[0], q[1], q[2], q[3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

# 准备关键点位置（猎豹坐标系Y向左，与Himmy一致，无需翻转）
tail_pos = g_root_trans.copy()  # tail_base位置

# 新方案：neck_base 作为 root，需要获取 nose 位置
nose_idx = CHEETAH_KP["nose"]
nose_pos = original_positions[:, nose_idx, :].copy()

spine_pos = original_positions[:, spine_idx, :].copy()
neck_pos = original_positions[:, neck_idx, :].copy()

# 不需要交换左右hip和shoulder，也不需要翻转Y
l_hip_pos = original_positions[:, l_hip_idx, :].copy()
r_hip_pos = original_positions[:, r_hip_idx, :].copy()

l_shoulder_pos = original_positions[:, l_shoulder_idx, :].copy()
r_shoulder_pos = original_positions[:, r_shoulder_idx, :].copy()

seq_len = g_root_trans.shape[0]
print(f'处理帧数: {seq_len}')

# 存储全局旋转矩阵和四元数（新方案）
R_neck_base_all = np.zeros((seq_len, 3, 3))    # neck_base 全局旋转
R_spine_all = np.zeros((seq_len, 3, 3))         # spine 全局旋转
q_neck_base_all = np.zeros((seq_len, 4))        # neck_base 全局四元数
q_spine_all = np.zeros((seq_len, 4))            # spine 全局四元数
q_spine_local_all = np.zeros((seq_len, 4))      # spine相对于neck_base的局部旋转

print('\n使用旋转矩阵方法计算全局旋转（新方案：neck_base作为parent）...')

for i in range(seq_len):
    # ==================== 第一步：构建 Neck_Base (NB) 的全局旋转 ====================
    # 修改：使用躯干方向而不是头部方向
    # 原因：猎豹头部经常低垂，使用 nose 会引入过大的 pitch 角度
    # 
    # 主轴 (Forward): 脊柱指向脖子根的方向（身体前半部分方向）
    #    V_fwd = P_NB - P_SP
    f_nb = neck_pos[i] - spine_pos[i]
    x_nb = normalize(f_nb)
    
    # 辅助轴 (Side): 肩膀连线
    #    V_side = P_LS - P_RS (左肩到右肩)
    s_nb = l_shoulder_pos[i] - r_shoulder_pos[i]
    s_nb_temp = normalize(s_nb)
    
    # 正交化
    #    在X-forward坐标系中，Z是上，Y是左
    z_nb = normalize(np.cross(x_nb, s_nb_temp))  # Z轴（上方向）
    y_nb = np.cross(z_nb, x_nb)                   # Y轴（左方向，已归一化）
    
    # 旋转矩阵 R = [x, y, z] (列向量)
    R_neck_base = np.column_stack([x_nb, y_nb, z_nb])
    R_neck_base_all[i] = R_neck_base
    q_neck_base_all[i] = rotation_matrix_to_quaternion(R_neck_base)
    
    # ==================== 第二步：构建 Spine (SP) 的全局旋转 ====================
    # 主轴 (Forward): 从tail_base指向spine的方向（身体后半部分方向）
    #    V_fwd = P_SP - P_TB (从尾巴根指向脊柱中点)
    f_sp = spine_pos[i] - tail_pos[i]
    x_sp = normalize(f_sp)
    
    # 辅助轴 (Side): 使用髋部连线保持后躯一致性
    #    V_side = P_LH - P_RH (左髋到右髋)
    s_sp = l_hip_pos[i] - r_hip_pos[i]
    s_sp_temp = normalize(s_sp)
    
    # 正交化
    z_sp = normalize(np.cross(x_sp, s_sp_temp))  # Z轴（上方向）
    y_sp = np.cross(z_sp, x_sp)                   # Y轴（左方向）
    
    # 旋转矩阵 R = [x, y, z] (列向量)
    R_spine = np.column_stack([x_sp, y_sp, z_sp])
    R_spine_all[i] = R_spine
    q_spine_all[i] = rotation_matrix_to_quaternion(R_spine)
    
    # ==================== 第三步：计算相对旋转 (Local Rotation) ====================
    # Spine 相对于 Neck_Base 的旋转
    # neck_base 代表身体前半部分，spine 代表身体后半部分
    # 局部旋转反映了脊柱的弯曲程度
    # R_local = R_NB^T @ R_SP
    # 或 Q_local = Q_NB^(-1) ⊗ Q_SP
    q_neck_base_inv = quaternion_inverse(q_neck_base_all[i])
    q_spine_local_all[i] = quaternion_multiply(q_neck_base_inv, q_spine_all[i])

# 新方案：使用 neck_base 的全局旋转作为 Himmy trunk 的旋转
g_root_rot_euler = np.array([quaternion_to_euler(q) for q in q_neck_base_all])

# spine 相对于 neck_base 的局部旋转
spine_local_euler = np.array([quaternion_to_euler(q) for q in q_spine_local_all])

print(f'\nneck_base全局旋转范围（欧拉角，弧度）:')
print(f'  Roll:  [{g_root_rot_euler[:, 0].min():.3f}, {g_root_rot_euler[:, 0].max():.3f}]')
print(f'  Pitch: [{g_root_rot_euler[:, 1].min():.3f}, {g_root_rot_euler[:, 1].max():.3f}]')
print(f'  Yaw:   [{g_root_rot_euler[:, 2].min():.3f}, {g_root_rot_euler[:, 2].max():.3f}]')

print(f'\nspine局部旋转范围（相对于neck_base，欧拉角，弧度）:')
print(f'  Roll:  [{spine_local_euler[:, 0].min():.3f}, {spine_local_euler[:, 0].max():.3f}]')
print(f'  Pitch: [{spine_local_euler[:, 1].min():.3f}, {spine_local_euler[:, 1].max():.3f}]')
print(f'  Yaw:   [{spine_local_euler[:, 2].min():.3f}, {spine_local_euler[:, 2].max():.3f}]')

# DEBUG: 查看缩放后的source脚位置
print(f'\n=== 缩放后的source脚位置（第一帧）===')
print(f'FL (l_front_ankle): {g_fl_trans[0]}')
print(f'FR (r_front_ankle): {g_fr_trans[0]}')
print(f'RL (l_back_ankle):  {g_rl_trans[0]}')
print(f'RR (r_back_ankle):  {g_rr_trans[0]}')
print(f'Y间距分析：')
print(f'  前腿: |{g_fl_trans[0][1]:.4f} - {g_fr_trans[0][1]:.4f}| = {abs(g_fl_trans[0][1] - g_fr_trans[0][1]):.4f}m')
print(f'  后腿: |{g_rl_trans[0][1]:.4f} - {g_rr_trans[0][1]:.4f}| = {abs(g_rl_trans[0][1] - g_rr_trans[0][1]):.4f}m')

# 调试：打印脚的 Y 坐标（第一帧）
print(f'\n=== 脚的 Y 坐标（第一帧，缩放后）===')
print(f'g_fl_trans[0] (应该是左前脚，Y>0): Y = {g_fl_trans[0, 1]:.4f}')
print(f'g_fr_trans[0] (应该是右前脚，Y<0): Y = {g_fr_trans[0, 1]:.4f}')
print(f'g_rl_trans[0] (应该是左后脚，Y>0): Y = {g_rl_trans[0, 1]:.4f}')
print(f'g_rr_trans[0] (应该是右后脚，Y<0): Y = {g_rr_trans[0, 1]:.4f}')
print(f'Himmy 坐标系: Y>0 是左边, Y<0 是右边')


# ==================== 旋转到X轴正方向 ====================

# 使用第一帧的yaw角度将所有数据旋转到X轴正方向
initial_yaw = g_root_rot_euler[0, 2]
rot_matrix = kinematic.rot_matrix_ba([0, 0, -initial_yaw])
g_root_trans = np.array([rot_matrix @ t for t in g_root_trans])
g_root_rot_euler[:, 2] -= initial_yaw  # 更新yaw (neck_base的yaw)
# 新方案：同时旋转neck_pos
neck_pos = np.array([rot_matrix @ t for t in neck_pos])
g_fl_trans = np.array([rot_matrix @ t for t in g_fl_trans])
g_fr_trans = np.array([rot_matrix @ t for t in g_fr_trans])
g_rl_trans = np.array([rot_matrix @ t for t in g_rl_trans])
g_rr_trans = np.array([rot_matrix @ t for t in g_rr_trans])

# 调试：旋转后脚的 Y 坐标
print(f'\n=== 旋转到X轴正方向后，脚的 Y 坐标（第20帧）===')
print(f'g_fl_trans[20] (左前脚，应该Y>0): Y = {g_fl_trans[20, 1]:.4f}')
print(f'g_fr_trans[20] (右前脚，应该Y<0): Y = {g_fr_trans[20, 1]:.4f}')
print(f'g_rl_trans[20] (左后脚，应该Y>0): Y = {g_rl_trans[20, 1]:.4f}')
print(f'g_rr_trans[20] (右后脚，应该Y<0): Y = {g_rr_trans[20, 1]:.4f}')


# ==================== 使用旋转矩阵计算脊柱局部角度 ====================

print('\n使用旋转矩阵计算脊柱局部角度...')

# 直接使用之前通过旋转矩阵计算好的spine局部欧拉角
# spine_local_euler = quaternion_to_euler(q_spine_local_all)
# 其中 q_spine_local = q_tail_base^(-1) ⊗ q_spine
# 这正确反映了spine相对于tail_base的旋转（即脊柱的弯曲程度）

# 注意：spine_local_euler的顺序是 [roll, pitch, yaw]
# 需要映射到Himmy的脊柱关节：yaw_spine, pitch_spine, roll_spine
spine_angles_all = np.zeros((seq_len, 3))

# 旋转矩阵方法计算的局部欧拉角已经考虑了yaw的校正
# 需要先将yaw校正应用到局部旋转
for i in range(seq_len):
    # spine局部旋转的欧拉角 [roll, pitch, yaw]
    local_roll = spine_local_euler[i, 0]
    local_pitch = spine_local_euler[i, 1]
    local_yaw = spine_local_euler[i, 2]
    
    # 映射到Himmy脊柱关节
    # Himmy脊柱: trunk -> yaw_spine -> pitch_spine -> roll_spine
    # spine_angles_all: [yaw, pitch, roll]
    spine_angles_all[i, 0] = local_yaw    # yaw (绕Z轴)
    spine_angles_all[i, 1] = local_pitch  # pitch (绕Y轴) - 这才是真正的脊柱弯曲！
    spine_angles_all[i, 2] = local_roll   # roll (绕X轴)

print(f'脊柱局部角度（从旋转矩阵计算，限位前）:')
print(f'  Yaw:   [{np.degrees(spine_angles_all[:, 0].min()):.2f}°, {np.degrees(spine_angles_all[:, 0].max()):.2f}°]')
print(f'  Pitch: [{np.degrees(spine_angles_all[:, 1].min()):.2f}°, {np.degrees(spine_angles_all[:, 1].max()):.2f}°]')
print(f'  Roll:  [{np.degrees(spine_angles_all[:, 2].min()):.2f}°, {np.degrees(spine_angles_all[:, 2].max()):.2f}°]')

# 脊柱关节限位（弧度）
spine_yaw_limit = [-0.5, 0.5]
spine_pitch_limit = [-0.8, 0.8]
spine_roll_limit = [-0.5, 0.5]

spine_angles_all[:, 0] = np.clip(spine_angles_all[:, 0], spine_yaw_limit[0], spine_yaw_limit[1])
spine_angles_all[:, 1] = np.clip(spine_angles_all[:, 1], spine_pitch_limit[0], spine_pitch_limit[1])
spine_angles_all[:, 2] = np.clip(spine_angles_all[:, 2], spine_roll_limit[0], spine_roll_limit[1])

print(f'\n脊柱局部角度（限位后）:')
print(f'  Yaw:   [{np.degrees(spine_angles_all[:, 0].min()):.2f}°, {np.degrees(spine_angles_all[:, 0].max()):.2f}°]')
print(f'  Pitch: [{np.degrees(spine_angles_all[:, 1].min()):.2f}°, {np.degrees(spine_angles_all[:, 1].max()):.2f}°]')
print(f'  Roll:  [{np.degrees(spine_angles_all[:, 2].min()):.2f}°, {np.degrees(spine_angles_all[:, 2].max()):.2f}°]')


# ==================== 新方案：neck_base -> Himmy trunk ====================

# 核心思想（新方案）：
# 1. Source的neck_base直接对应Himmy的trunk (base_link)
# 2. spine相对于neck_base的局部旋转 → Himmy脊柱关节角度
# 3. 前腿：trunk位置确定 → 前腿hip位置确定 → IK求前腿关节
# 4. 脊柱FK：从trunk通过脊柱正向运动学计算roll_spine位置
# 5. 后腿：roll_spine位置确定 → 后腿hip位置确定 → IK求后腿关节

print('\n使用新方案: neck_base -> Himmy trunk')

# neck_base的位置作为Himmy trunk的位置
g_neck_base_trans = neck_pos.copy()  # 使用之前提取的neck_pos
g_trunk_rot_euler = g_root_rot_euler.copy()  # neck_base的全局旋转

# 应用 x 轴偏移补偿（修正 T-pose 中脚位置的差异）
g_neck_base_trans[:, 0] -= root_x_offset.item()




# ==================== 逆运动学求解 ====================

print('\n进行逆运动学求解（新方案）...')

leg_joint_angles = np.zeros((seq_len, 12))  # FR(3), FL(3), RR(3), RL(3)
g_trunk_trans = np.zeros((seq_len, 3))      # 存储trunk位置
g_roll_spine_trans = np.zeros((seq_len, 3)) # 存储roll_spine位置（通过FK计算）

# hip坐标系旋转（绕Y轴旋转90度）
r_hip = [0, np.pi / 2, 0]

for i in range(seq_len):
    spine_ang = spine_angles_all[i]
    trunk_pos = g_neck_base_trans[i]
    trunk_rot = g_trunk_rot_euler[i]
    
    # 保存trunk位置
    g_trunk_trans[i] = trunk_pos
    
    # ========== Step 1: 前腿IK (基于trunk) ==========
    # trunk位置已知，前腿hip相对于trunk的位置是固定的
    T_w_trunk = kinematic.trans_matrix_ba(trunk_pos, trunk_rot)
    T_trunk_w = np.linalg.inv(T_w_trunk)
    
    # 前腿hip相对于trunk的位置
    fl_hip_offset = np.array([FRONT_BODY_LENGTH, BODY_WIDE / 2, 0])
    fr_hip_offset = np.array([FRONT_BODY_LENGTH, -BODY_WIDE / 2, 0])
    
    # 将足端从世界坐标系转换到hip坐标系
    T_trunk_fl = kinematic.trans_matrix_ab(fl_hip_offset, r_hip)
    T_trunk_fr = kinematic.trans_matrix_ab(fr_hip_offset, r_hip)
    
    fl_foot_hip = T_trunk_fl @ T_trunk_w @ np.append(g_fl_trans[i], 1)
    fl_foot_hip = fl_foot_hip[:3]
    fr_foot_hip = T_trunk_fr @ T_trunk_w @ np.append(g_fr_trans[i], 1)
    fr_foot_hip = fr_foot_hip[:3]
    
    # 求解前腿关节角度
    fl_ang = kinematic.inverse_kinematics_leg(fl_foot_hip, is_left=True)
    fl_ang = np.array([fl_ang[0], -fl_ang[1], -fl_ang[2]])
    fr_ang = kinematic.inverse_kinematics_leg(fr_foot_hip, is_left=False)
    fr_ang = np.array([fr_ang[0], -fr_ang[1], -fr_ang[2]])
    
    # ========== Step 2: 脊柱FK (从trunk计算roll_spine位置) ==========
    # 使用脊柱正向运动学计算roll_spine的世界位置
    roll_spine_pos, roll_spine_rot = spine_forward_kinematics(spine_ang, trunk_pos, trunk_rot)
    g_roll_spine_trans[i] = roll_spine_pos
    
    # ========== Step 3: 后腿IK (基于roll_spine) ==========
    # roll_spine位置通过脊柱FK确定，后腿hip相对于roll_spine的位置是固定的
    T_w_roll = kinematic.trans_matrix_ba(roll_spine_pos, roll_spine_rot)
    T_roll_w = np.linalg.inv(T_w_roll)
    
    # 后腿hip相对于roll_spine的位置
    rl_hip_local = np.array([-REAR_BODY_LENGTH, BODY_WIDE / 2, 0])
    rr_hip_local = np.array([-REAR_BODY_LENGTH, -BODY_WIDE / 2, 0])
    
    # 将足端从世界坐标系转换到hip坐标系
    T_roll_rl = kinematic.trans_matrix_ab(rl_hip_local, r_hip)
    T_roll_rr = kinematic.trans_matrix_ab(rr_hip_local, r_hip)
    
    rl_foot_hip = T_roll_rl @ T_roll_w @ np.append(g_rl_trans[i], 1)
    rl_foot_hip = rl_foot_hip[:3]
    rr_foot_hip = T_roll_rr @ T_roll_w @ np.append(g_rr_trans[i], 1)
    rr_foot_hip = rr_foot_hip[:3]
    
    # 求解后腿关节角度
    rl_ang = kinematic.inverse_kinematics_leg(rl_foot_hip, is_left=True)
    rl_ang = np.array([rl_ang[0], -rl_ang[1], -rl_ang[2]])
    rr_ang = kinematic.inverse_kinematics_leg(rr_foot_hip, is_left=False)
    rr_ang = np.array([rr_ang[0], -rr_ang[1], -rr_ang[2]])
    
    # 调试：打印第一帧信息
    if i == 0:
        print(f'\n=== 第一帧：新方案调试信息 ===')
        print(f'Trunk位置: {trunk_pos}')
        print(f'Trunk旋转: {np.degrees(trunk_rot)}°')
        print(f'Roll_spine位置（FK计算）: {roll_spine_pos}')
        
        print(f'\n脚在hip坐标系中的位置:')
        print(f'FL foot in FL_hip frame: {fl_foot_hip}')
        print(f'FR foot in FR_hip frame: {fr_foot_hip}')
        print(f'RL foot in RL_hip frame: {rl_foot_hip}')
        print(f'RR foot in RR_hip frame: {rr_foot_hip}')
        
        print(f'\n脚在世界坐标系中的位置:')
        print(f'FL foot world: {g_fl_trans[i]}')
        print(f'FR foot world: {g_fr_trans[i]}')
        print(f'RL foot world: {g_rl_trans[i]}')
        print(f'RR foot world: {g_rr_trans[i]}')
        
        print(f'\nHip位置（世界坐标系）:')
        fl_hip_world = (T_w_trunk @ np.append(fl_hip_offset, 1))[:3]
        fr_hip_world = (T_w_trunk @ np.append(fr_hip_offset, 1))[:3]
        rl_hip_world = (T_w_roll @ np.append(rl_hip_local, 1))[:3]
        rr_hip_world = (T_w_roll @ np.append(rr_hip_local, 1))[:3]
        print(f'FL hip world: {fl_hip_world}')
        print(f'FR hip world: {fr_hip_world}')
        print(f'RL hip world: {rl_hip_world}')
        print(f'RR hip world: {rr_hip_world}')
    
    # 存储结果: FR(3), FL(3), RR(3), RL(3)
    leg_joint_angles[i] = np.hstack([fr_ang, fl_ang, rr_ang, rl_ang])

# 检查NaN
if np.isnan(leg_joint_angles).any():
    nan_count = np.sum(np.isnan(leg_joint_angles))
    print(f'警告: {nan_count}个NaN值，用前一帧填充')
    for i in range(1, seq_len):
        for j in range(12):
            if np.isnan(leg_joint_angles[i, j]):
                leg_joint_angles[i, j] = leg_joint_angles[i-1, j]

# 打印 IK 计算出的 hip 角度范围（限位前）
print(f'\n=== IK 计算的 Hip 角度（限位前）===')
print(f'FR hip: [{np.degrees(leg_joint_angles[:, 0].min()):.2f}°, {np.degrees(leg_joint_angles[:, 0].max()):.2f}°]')
print(f'FL hip: [{np.degrees(leg_joint_angles[:, 3].min()):.2f}°, {np.degrees(leg_joint_angles[:, 3].max()):.2f}°]')
print(f'RR hip: [{np.degrees(leg_joint_angles[:, 6].min()):.2f}°, {np.degrees(leg_joint_angles[:, 6].max()):.2f}°]')
print(f'RL hip: [{np.degrees(leg_joint_angles[:, 9].min()):.2f}°, {np.degrees(leg_joint_angles[:, 9].max()):.2f}°]')


# ==================== 关节角度限位 ====================

print('\n应用关节限位...')

# hip_limit = [-0.50, 0.50]
# thigh_limit = [-1.5, 3.5]
# calf_limit = [-2.2, -0.7]

hip_limit = [-1.50, 1.50]
thigh_limit = [-2, 2.2]
calf_limit = [-2.2, -0.7]

for i in range(seq_len):
    for j in range(12):
        if j % 3 == 0:  # Hip
            leg_joint_angles[i, j] = np.clip(leg_joint_angles[i, j], hip_limit[0], hip_limit[1])
        elif j % 3 == 1:  # Thigh
            leg_joint_angles[i, j] = np.clip(leg_joint_angles[i, j], thigh_limit[0], thigh_limit[1])
        else:  # Calf
            leg_joint_angles[i, j] = np.clip(leg_joint_angles[i, j], calf_limit[0], calf_limit[1])


# ==================== 构建target motion ====================

print('\n构建Target Motion...')

# 转换欧拉角到四元数（使用通过脊柱IK计算出的trunk旋转）
g_trunk_rot_euler_tensor = torch.from_numpy(g_trunk_rot_euler)
g_trunk_rot_quat = torch.zeros((seq_len, 4))
for i in range(seq_len):
    # 直接使用估算的欧拉角，不需要额外roll校正
    q = quat_from_euler_xyz_numpy(g_trunk_rot_euler[i, 0], 
                                   g_trunk_rot_euler[i, 1], 
                                   g_trunk_rot_euler[i, 2])
    g_trunk_rot_quat[i] = torch.from_numpy(q)

# Himmy骨架结构（20个节点）
# ['trunk', 'FL_hip_Link', 'FL_thigh_Link', 'FL_calf_Link', 'FL_foot',
#  'FR_hip_Link', 'FR_thigh_Link', 'FR_calf_Link', 'FR_foot',
#  'yaw_spine_Link', 'pitch_spine_Link', 'roll_spine_Link',
#  'RL_hip_Link', 'RL_thigh_Link', 'RL_calf_Link', 'RL_foot',
#  'RR_hip_Link', 'RR_thigh_Link', 'RR_calf_Link', 'RR_foot']

local_rotation = torch.zeros((seq_len, 20, 4))
local_rotation[:, :, -1] = 1  # 初始化为单位四元数

for i in range(seq_len):
    # Root旋转 (trunk)
    local_rotation[i, 0, :] = g_trunk_rot_quat[i]
    
    # 前左腿 (FL) - indices 1, 2, 3
    fl_hip_q = kinematic.rpy2quaternion([leg_joint_angles[i, 3], 0, 0])
    fl_thigh_q = kinematic.rpy2quaternion([0, leg_joint_angles[i, 4], 0])
    fl_calf_q = kinematic.rpy2quaternion([0, leg_joint_angles[i, 5], 0])
    local_rotation[i, 1, :] = torch.tensor(fl_hip_q)
    local_rotation[i, 2, :] = torch.tensor(fl_thigh_q)
    local_rotation[i, 3, :] = torch.tensor(fl_calf_q)
    # FL_foot (index=4) 保持单位四元数
    
    # 前右腿 (FR) - indices 5, 6, 7
    fr_hip_q = kinematic.rpy2quaternion([leg_joint_angles[i, 0], 0, 0])
    fr_thigh_q = kinematic.rpy2quaternion([0, leg_joint_angles[i, 1], 0])
    fr_calf_q = kinematic.rpy2quaternion([0, leg_joint_angles[i, 2], 0])
    local_rotation[i, 5, :] = torch.tensor(fr_hip_q)
    local_rotation[i, 6, :] = torch.tensor(fr_thigh_q)
    local_rotation[i, 7, :] = torch.tensor(fr_calf_q)
    # FR_foot (index=8) 保持单位四元数
    
    # 脊柱关节 - indices 9, 10, 11
    yaw_q = kinematic.rpy2quaternion([0, 0, spine_angles_all[i, 0]])
    pitch_q = kinematic.rpy2quaternion([0, spine_angles_all[i, 1], 0])
    roll_q = kinematic.rpy2quaternion([spine_angles_all[i, 2], 0, 0])
    local_rotation[i, 9, :] = torch.tensor(yaw_q)
    local_rotation[i, 10, :] = torch.tensor(pitch_q)
    local_rotation[i, 11, :] = torch.tensor(roll_q)
    
    # 后左腿 (RL) - indices 12, 13, 14
    rl_hip_q = kinematic.rpy2quaternion([leg_joint_angles[i, 9], 0, 0])
    rl_thigh_q = kinematic.rpy2quaternion([0, leg_joint_angles[i, 10], 0])
    rl_calf_q = kinematic.rpy2quaternion([0, leg_joint_angles[i, 11], 0])
    local_rotation[i, 12, :] = torch.tensor(rl_hip_q)
    local_rotation[i, 13, :] = torch.tensor(rl_thigh_q)
    local_rotation[i, 14, :] = torch.tensor(rl_calf_q)
    # RL_foot (index=15) 保持单位四元数
    
    # 后右腿 (RR) - indices 16, 17, 18
    rr_hip_q = kinematic.rpy2quaternion([leg_joint_angles[i, 6], 0, 0])
    rr_thigh_q = kinematic.rpy2quaternion([0, leg_joint_angles[i, 7], 0])
    rr_calf_q = kinematic.rpy2quaternion([0, leg_joint_angles[i, 8], 0])
    local_rotation[i, 16, :] = torch.tensor(rr_hip_q)
    local_rotation[i, 17, :] = torch.tensor(rr_thigh_q)
    local_rotation[i, 18, :] = torch.tensor(rr_calf_q)
    # RR_foot (index=19) 保持单位四元数


# ==================== 计算root translation ====================

# 将第一帧起点设为原点
root_translation = torch.from_numpy(g_trunk_trans).float()
root_translation[:, 0] = root_translation[:, 0] - root_translation[0, 0]
root_translation[:, 1] = root_translation[:, 1] - root_translation[0, 1]

# 添加高度偏移
root_height_offset = 0.55
root_translation[:, 2] = root_translation[:, 2] + root_height_offset


# ==================== 保存结果 ====================

skeleton_state = SkeletonState.from_rotation_and_root_translation(
    target_tpose.skeleton_tree,
    local_rotation,
    root_translation,
    is_local=True
)

target_motion = SkeletonMotion.from_skeleton_state(skeleton_state, fps=SOURCE_FPS)

# 将高度偏移应用到所有关节的全局位移（将每个节点的 z 分量加上 root_height_offset）
height_offset = float(root_height_offset)
try:
    gt = target_motion.global_transformation
    # 如果是 torch Tensor，转换为 numpy
    if hasattr(gt, 'numpy'):
        gt_np = gt.numpy()
    else:
        gt_np = np.array(gt)
    gt_np[..., -1] = gt_np[..., -1] + height_offset
    # 尝试写回修改后的全局变换数组（尽量兼容不同实现）
    try:
        target_motion.global_transformation = gt_np
    except Exception:
        if hasattr(target_motion, '_global_transformation'):
            target_motion._global_transformation = gt_np
        else:
            # 不能写回时打印警告，但不阻塞保存
            print('Warning: 无法直接写回 target_motion.global_transformation；已修改局部数据备份。')
except Exception as e:
    print('Warning: 应用高度偏移到全局变换失败:', e)

output_path = os.path.join(output_dir, '{}.npy'.format(output_file))
target_motion.to_file(output_path)

print(f'\n保存完成: {output_path}')
print(f'  帧数: {seq_len}')
print(f'  FPS: {SOURCE_FPS}')
print(f'  总时长: {seq_len / SOURCE_FPS:.2f}s')

# 重新加载验证
target_motion = SkeletonMotion.from_file(output_path)
print(f'\n验证: 成功加载 {target_motion.local_rotation.shape[0]} 帧')


# ==================== 分析Himmy retargeting后的数据 ====================

print("\n" + "=" * 60)
print("Himmy Retargeting后的数据分析")
print("=" * 60)

# 获取Himmy骨架
himmy_skeleton = target_tpose.skeleton_tree

# 获取足端点的全局位置
himmy_global_trans = target_motion.global_transformation  # (seq_len, 20, 7)

fl_trans = himmy_global_trans[:, himmy_skeleton.node_names.index('FL_foot'), -3:].numpy()
fr_trans = himmy_global_trans[:, himmy_skeleton.node_names.index('FR_foot'), -3:].numpy()
rl_trans = himmy_global_trans[:, himmy_skeleton.node_names.index('RL_foot'), -3:].numpy()
rr_trans = himmy_global_trans[:, himmy_skeleton.node_names.index('RR_foot'), -3:].numpy()

# 计算脚间距
himmy_front_widths = np.abs(fl_trans[:, 1] - fr_trans[:, 1])
himmy_rear_widths = np.abs(rl_trans[:, 1] - rr_trans[:, 1])

print(f'\nHimmy Motion中的脚位置（前10帧，Y轴坐标）：')
print('Frame | FL_foot_Y | FR_foot_Y | 前脚Y间距 | RL_foot_Y | RR_foot_Y | 后脚Y间距')
print('-' * 85)
for i in range(min(10, fl_trans.shape[0])):
    fl_y = fl_trans[i, 1]
    fr_y = fr_trans[i, 1]
    rl_y = rl_trans[i, 1]
    rr_y = rr_trans[i, 1]
    front_width = abs(fl_y - fr_y)
    rear_width = abs(rl_y - rr_y)
    print(f'{i:5d} | {fl_y:9.4f} | {fr_y:9.4f} | {front_width:9.4f} | {rl_y:9.4f} | {rr_y:9.4f} | {rear_width:9.4f}')

print(f'\nHimmy Motion统计（整个序列）：')
print(f'前脚Y间距: min={himmy_front_widths.min():.4f}, max={himmy_front_widths.max():.4f}, mean={himmy_front_widths.mean():.4f}')
print(f'后脚Y间距: min={himmy_rear_widths.min():.4f}, max={himmy_rear_widths.max():.4f}, mean={himmy_rear_widths.mean():.4f}')

# 对比Himmy T-pose
himmy_fl_y = target_tpose.global_translation[himmy_skeleton.node_names.index('FL_foot'), 1].item()
himmy_fr_y = target_tpose.global_translation[himmy_skeleton.node_names.index('FR_foot'), 1].item()
himmy_rl_y = target_tpose.global_translation[himmy_skeleton.node_names.index('RL_foot'), 1].item()
himmy_rr_y = target_tpose.global_translation[himmy_skeleton.node_names.index('RR_foot'), 1].item()
himmy_tpose_front_width = abs(himmy_fl_y - himmy_fr_y)
himmy_tpose_rear_width = abs(himmy_rl_y - himmy_rr_y)

print(f'\nHimmy T-pose中的Y间距：')
print(f'前脚Y间距: {himmy_tpose_front_width:.4f}m')
print(f'后脚Y间距: {himmy_tpose_rear_width:.4f}m')

print(f'\n对比分析：')
print(f'Motion前脚平均Y间距 / T-pose前脚Y间距 = {himmy_front_widths.mean() / himmy_tpose_front_width:.2%}')
print(f'Motion后脚平均Y间距 / T-pose后脚Y间距 = {himmy_rear_widths.mean() / himmy_tpose_rear_width:.2%}')

if himmy_front_widths.mean() < himmy_tpose_front_width * 0.8 or himmy_rear_widths.mean() < himmy_tpose_rear_width * 0.8:
    print('\n⚠️  Himmy retargeting后的数据存在脚间距缩小的问题！')
    print('    这是由于source motion数据本身就是内八字步态导致的。')
else:
    print('\n✓ Himmy retargeting后的脚间距基本保持了T-pose的比例')


print("\n" + "=" * 60)
print("Motion Retargeting 完成！")
print("=" * 60)

# 可视化（如果有GUI）
# plot_skeleton_motion_interactive(target_motion)
