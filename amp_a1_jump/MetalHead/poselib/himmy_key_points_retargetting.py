#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: himmy_key_points_retargetting.py
@Auth: Based on A1 retargeting by Huiqiao, modified for Himmy Mark2
@Date: 2024/12

Himmy Mark2 Motion Retargeting（脊柱感知版本）
=============================================

该模块实现了将动物动捕数据重定向到Himmy Mark2机器人的功能。

与A1 Retargeting的主要区别：
---------------------------
1. Himmy有3个脊柱自由度（yaw, pitch, roll），需要从source的Spine1映射
2. Himmy的root(trunk)在身体前方，source的root(Hips)在尾部
3. 后腿IK需要考虑脊柱弯曲对hip位置的影响

核心算法：
---------
1. 前腿IK：直接从trunk计算（与A1类似，因为前腿hip相对trunk位置固定）
2. 脊柱处理：从Source的Spine1提取姿态(roll, pitch, yaw)，映射到Himmy的3个脊柱关节
3. 后腿IK：
   a. 通过脊柱正向运动学计算后腿hip的实际世界坐标位置
   b. 用source的后脚位置和计算出的hip位置进行IK

Source骨架结构（AI4Animation狗模型，27个joint）：
- Hips (root) -> Spine -> Spine1 -> Neck -> Head
- LeftShoulder -> LeftArm -> LeftForeArm -> LeftHand -> LeftHand_End (前左腿)
- RightShoulder -> RightArm -> RightForeArm -> RightHand -> RightHand_End (前右腿)
- LeftUpLeg -> LeftLeg -> LeftFoot -> LeftFoot_End (后左腿)
- RightUpLeg -> RightLeg -> RightFoot -> RightFoot_End (后右腿)

Target骨架结构（Himmy Mark2，20个node）：
- trunk (root)
- FL_hip_Link -> FL_thigh_Link -> FL_calf_Link -> FL_foot (前左腿)
- FR_hip_Link -> FR_thigh_Link -> FR_calf_Link -> FR_foot (前右腿)
- yaw_spine_Link -> pitch_spine_Link -> roll_spine_Link (脊柱)
- RL_hip_Link -> RL_thigh_Link -> RL_calf_Link -> RL_foot (后左腿)
- RR_hip_Link -> RR_thigh_Link -> RR_calf_Link -> RR_foot (后右腿)
"""
import copy
import json
import numpy as np
from himmy_kinematics import HimmyKinematics
from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


# ==================== 配置文件加载 ====================

json_dir = "data/load_config.json"
with open(json_dir, "r") as f:
    motion_json = json.load(f)
    file_name = motion_json["file_name"]
    clip = motion_json["clip"]
    remarks = motion_json["remarks"]

# 源数据目录（amp_hardware_a1）
source_dir = "data/amp_hardware_a1/{}/".format(file_name)
# 输出目录（amp_himmy_mark2）
output_dir = "data/amp_himmy_mark2/{}/".format(file_name)
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


def spine_inverse_kinematics(spine_angles, roll_spine_pos_w, roll_spine_rot_w):
    """
    脊柱逆向运动学：已知roll_spine的位姿和脊柱角度，反推trunk的位姿
    
    脊柱链: trunk -> yaw_spine -> pitch_spine -> roll_spine
    逆向: roll_spine -> pitch_spine -> yaw_spine -> trunk
    
    Args:
        spine_angles: [yaw, pitch, roll] 三个脊柱关节角度（弧度）
        roll_spine_pos_w: roll_spine_Link在世界坐标系下的位置 [x, y, z]
        roll_spine_rot_w: roll_spine_Link在世界坐标系下的欧拉角 [roll, pitch, yaw]
    
    Returns:
        trunk_pos_w: trunk在世界坐标系下的位置
        trunk_rot_w: trunk在世界坐标系下的欧拉角
    """
    yaw_ang, pitch_ang, roll_ang = spine_angles
    
    # roll_spine到世界的变换
    T_w_roll = kinematic.trans_matrix_ba(roll_spine_pos_w, roll_spine_rot_w)
    
    # roll_spine到pitch_spine的逆变换
    roll_spine_pos = np.array([-SPINE_PITCH_LENGTH, 0, 0])
    roll_spine_rot = [roll_ang, 0, 0]
    T_pitch_roll = kinematic.trans_matrix_ba(roll_spine_pos, roll_spine_rot)
    T_roll_pitch = np.linalg.inv(T_pitch_roll)
    
    # pitch_spine到yaw_spine的逆变换
    pitch_spine_pos = np.array([-SPINE_YAW_LENGTH, 0, 0])
    pitch_spine_rot = [0, pitch_ang, 0]
    T_yaw_pitch = kinematic.trans_matrix_ba(pitch_spine_pos, pitch_spine_rot)
    T_pitch_yaw = np.linalg.inv(T_yaw_pitch)
    
    # yaw_spine到trunk的逆变换
    yaw_spine_pos = np.array([SPINE_OFFSET_X, 0, 0])
    yaw_spine_rot = [0, 0, yaw_ang]
    T_trunk_yaw = kinematic.trans_matrix_ba(yaw_spine_pos, yaw_spine_rot)
    T_yaw_trunk = np.linalg.inv(T_trunk_yaw)
    
    # 完整逆变换链: T_w_trunk = T_w_roll @ T_roll_pitch @ T_pitch_yaw @ T_yaw_trunk
    T_w_trunk = T_w_roll @ T_roll_pitch @ T_pitch_yaw @ T_yaw_trunk
    
    # 提取位置
    trunk_pos_w = T_w_trunk[:3, 3]
    
    # 提取旋转
    R = T_w_trunk[:3, :3]
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    trunk_rot_w = np.array([roll, pitch, yaw])
    
    return trunk_pos_w, trunk_rot_w


# ==================== 数据加载 ====================

print("=" * 60)
print("Himmy Mark2 Motion Retargeting（脊柱感知版本）")
print("=" * 60)

# 加载source motion（从amp_hardware_a1目录）
source_motion = SkeletonMotion.from_file(source_dir + '{}.npy'.format(file_name))


# ==================== 尺寸缩放计算 ====================

# 加载T-pose
source_tpose = SkeletonState.from_file('data/T_pose/dog_tpose.npy')
target_tpose = SkeletonState.from_file('data/T_pose/amp_himmy_mark2_tpose.npy')

skeleton_s = source_tpose.skeleton_tree
skeleton_t = target_tpose.skeleton_tree

# 计算source T-pose的关键尺寸（注意：source T-pose单位是厘米，需要转换为米）
source_length = torch.abs(source_tpose.global_translation[skeleton_s.index('LeftHand_End'), 0] -
                          source_tpose.global_translation[skeleton_s.index('LeftFoot_End'), 0]) / 100.0
source_wide = torch.abs(source_tpose.global_translation[skeleton_s.index('LeftHand_End'), 1] -
                        source_tpose.global_translation[skeleton_s.index('RightHand_End'), 1]) / 100.0
source_height = torch.abs(source_tpose.global_translation[skeleton_s.index('Hips'), 2] -
                          source_tpose.global_translation[skeleton_s.index('LeftFoot_End'), 2]) / 100.0

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
zoom_x = (target_length / source_length) * 1.0
zoom_y = (target_wide / source_wide) * 1.2
zoom_z = target_height / source_height

# ==================== 计算 T-pose 中右后脚的 x 轴偏移 ====================
# 目的：当 source root (Hips) 和 himmy roll_spine_Link 重合时，
# 修正两者右后脚位置的 x 轴差距，确保映射后脚的位置基本相同

# Source T-pose 中：Hips 的位置 和 RightFoot_End 的 x 坐标差（单位：厘米，需转换为米）
source_hips_x = source_tpose.global_translation[skeleton_s.index('Hips'), 0] / 100.0
source_right_foot_x = source_tpose.global_translation[skeleton_s.index('RightFoot_End'), 0] / 100.0
source_hips_to_rr_foot_x = source_right_foot_x - source_hips_x  # Source: Hips 到右后脚的 x 偏移

# Target T-pose 中：roll_spine_Link 的位置 和 RR_foot 的 x 坐标差
target_roll_spine_x = target_tpose.global_translation[skeleton_t.index('roll_spine_Link'), 0]
target_rr_foot_x = target_tpose.global_translation[skeleton_t.index('RR_foot'), 0]
target_roll_spine_to_rr_foot_x = target_rr_foot_x - target_roll_spine_x  # Target: roll_spine 到右后脚的 x 偏移

# 应用缩放后的 source 偏移（因为 source motion 会被缩放）
source_hips_to_rr_foot_x_scaled = source_hips_to_rr_foot_x * zoom_x

# 计算需要补偿的 x 轴偏移
# 当 source root 映射到 himmy roll_spine 时，两者右后脚的 x 差距
# root_x_offset = target 需要的偏移 - source 缩放后的偏移
root_x_offset =  source_hips_to_rr_foot_x_scaled - target_roll_spine_to_rr_foot_x

print(f'\nT-pose 右后脚 x 轴偏移计算:')
print(f'  Source Hips -> RightFoot_End x: {source_hips_to_rr_foot_x:.4f}m')
print(f'  Source Hips -> RightFoot_End x (scaled): {source_hips_to_rr_foot_x_scaled:.4f}m')
print(f'  Target roll_spine -> RR_foot x: {target_roll_spine_to_rr_foot_x:.4f}m')
print(f'  Root x offset (补偿值): {root_x_offset:.4f}m')




# 获取骨架信息
skeleton = source_motion.skeleton_tree
local_translation = skeleton.local_translation
root_translation = source_motion.root_translation

print('zoom_x: {}, zoom_y: {}, zoom_z: {}'.format(zoom_x, zoom_y, zoom_z))

# 调试：检查转换前的数据
print(f'\n转换前 root_translation[0]: {root_translation[0]}')
print(f'转换前 local_translation[0]: {local_translation[0]}')


# ==================== 单位转换：厘米 -> 米 ====================

print('\n将source motion数据从厘米转换为米...')
# Source motion数据是厘米单位，需要转换为米
# 注意：这里需要原地修改，因为skeleton.local_translation是引用
skeleton.local_translation[:] = skeleton.local_translation / 100.0
source_motion.root_translation[:] = source_motion.root_translation / 100.0

# 调试：检查转换后的数据
print(f'转换后 root_translation[0]: {root_translation[0]}')
print(f'转换后 local_translation[0]: {local_translation[0]}')


# ==================== 新方案：直接缩放全局坐标（避免脊柱旋转累积误差）====================

print('\n' + '='*60)
print('使用新方案：直接缩放全局坐标')
print('='*60)

# 获取全局坐标（未缩放）
global_trans_original = source_motion.global_transformation  # (len, 27, 7)

# 提取四个足端的全局位置
lh_idx = skeleton.node_names.index("LeftHand_End")
rh_idx = skeleton.node_names.index("RightHand_End")
lf_idx = skeleton.node_names.index("LeftFoot_End")
rf_idx = skeleton.node_names.index("RightFoot_End")
hips_idx = skeleton.node_names.index("Hips")

print(f'\n缩放前（第一帧）：')
print(f'  LeftHand_End:  Y={global_trans_original[0, lh_idx, -2].item():.4f}m')
print(f'  RightHand_End: Y={global_trans_original[0, rh_idx, -2].item():.4f}m')
print(f'  前脚Y间距: {abs(global_trans_original[0, lh_idx, -2].item() - global_trans_original[0, rh_idx, -2].item()):.4f}m')

# 直接缩放足端的全局坐标（在世界坐标系中）
# 注意：global_transformation的格式是 [qx, qy, qz, qw, tx, ty, tz]
# 我们只修改translation部分（最后3维）
global_trans_original[:, lh_idx, -3] *= zoom_x  # X
global_trans_original[:, lh_idx, -2] *= zoom_y  # Y
global_trans_original[:, lh_idx, -1] *= zoom_z  # Z

global_trans_original[:, rh_idx, -3] *= zoom_x
global_trans_original[:, rh_idx, -2] *= zoom_y
global_trans_original[:, rh_idx, -1] *= zoom_z

global_trans_original[:, lf_idx, -3] *= zoom_x
global_trans_original[:, lf_idx, -2] *= zoom_y
global_trans_original[:, lf_idx, -1] *= zoom_z

global_trans_original[:, rf_idx, -3] *= zoom_x
global_trans_original[:, rf_idx, -2] *= zoom_y
global_trans_original[:, rf_idx, -1] *= zoom_z

# 同时缩放root的全局位置
global_trans_original[:, hips_idx, -3] *= zoom_x
global_trans_original[:, hips_idx, -2] *= zoom_y
global_trans_original[:, hips_idx, -1] *= zoom_z

print(f'\n缩放后（第一帧）：')
print(f'  LeftHand_End:  Y={global_trans_original[0, lh_idx, -2].item():.4f}m')
print(f'  RightHand_End: Y={global_trans_original[0, rh_idx, -2].item():.4f}m')
print(f'  LeftFoot_End:  Y={global_trans_original[0, lf_idx, -2].item():.4f}m')
print(f'  RightFoot_End: Y={global_trans_original[0, rf_idx, -2].item():.4f}m')
print(f'  前脚Y间距: {abs(global_trans_original[0, lh_idx, -2].item() - global_trans_original[0, rh_idx, -2].item()):.4f}m')
print(f'  后脚Y间距: {abs(global_trans_original[0, lf_idx, -2].item() - global_trans_original[0, rf_idx, -2].item()):.4f}m')
print(f'  预期前脚Y间距: {abs(global_trans_original[0, lh_idx, -2].item() - global_trans_original[0, rh_idx, -2].item()) / zoom_y * zoom_y:.4f}m')
print(f'  Y轴缩放因子: zoom_y = {zoom_y:.4f}')
print('='*60)


# ==================== 提取关键点（使用缩放后的全局坐标）====================

global_trans = global_trans_original  # 使用已经缩放过的全局坐标

# Root (Hips)
global_root_translation = global_trans[:, skeleton.node_names.index("Hips"), -3:]
global_root_rotation_quat = global_trans[:, skeleton.node_names.index("Hips"), :4]

# Spine1（用于脊柱映射）
global_spine1_rotation_quat = global_trans[:, skeleton.node_names.index("Spine1"), :4]

# 四个足端位置
# 与A1保持一致的映射：Source的Left对应Target的Left，Source的Right对应Target的Right
# Source LeftHand -> Target FL (前左)
# Source RightHand -> Target FR (前右)
global_left_hand_end_trans = global_trans[:, skeleton.node_names.index("LeftHand_End"), -3:]   # -> FL
global_right_hand_end_trans = global_trans[:, skeleton.node_names.index("RightHand_End"), -3:] # -> FR
global_left_foot_end_trans = global_trans[:, skeleton.node_names.index("LeftFoot_End"), -3:]   # -> RL
global_right_foot_end_trans = global_trans[:, skeleton.node_names.index("RightFoot_End"), -3:] # -> RR

# 转换root四元数到欧拉角
global_root_rotation_quat = global_root_rotation_quat.numpy()
global_root_rotation_euler = [kinematic.quaternion2rpy(q) for q in global_root_rotation_quat]
global_root_rotation_euler = np.array(global_root_rotation_euler)


# ==================== 裁剪指定区间 ====================

g_root_trans = global_root_translation.numpy()[clip[0]:clip[1], :]
g_root_rot_euler = global_root_rotation_euler[clip[0]:clip[1], :]
g_spine1_quat = global_spine1_rotation_quat.numpy()[clip[0]:clip[1], :]
g_hips_quat = global_root_rotation_quat[clip[0]:clip[1], :]
# 与A1一致：LeftHand->FL, RightHand->FR, LeftFoot->RL, RightFoot->RR
g_fl_trans = global_left_hand_end_trans.numpy()[clip[0]:clip[1], :]    # LeftHand -> FL (前左)
g_fr_trans = global_right_hand_end_trans.numpy()[clip[0]:clip[1], :]   # RightHand -> FR (前右)
g_rl_trans = global_left_foot_end_trans.numpy()[clip[0]:clip[1], :]    # LeftFoot -> RL (后左)    
g_rr_trans = global_right_foot_end_trans.numpy()[clip[0]:clip[1], :]   # RightFoot -> RR (后右)

# DEBUG: 查看缩放后、旋转前的source脚位置
print(f'\n=== 缩放后、旋转前的source脚位置（第一帧）===')
print(f'FL (LeftHand):  {g_fl_trans[0]}')
print(f'FR (RightHand): {g_fr_trans[0]}')
print(f'RL (LeftFoot):  {g_rl_trans[0]}')
print(f'RR (RightFoot): {g_rr_trans[0]}')
print(f'Y间距分析：')
print(f'  前腿: |{g_fl_trans[0][1]:.4f} - {g_fr_trans[0][1]:.4f}| = {abs(g_fl_trans[0][1] - g_fr_trans[0][1]):.4f}m')
print(f'  后腿: |{g_rl_trans[0][1]:.4f} - {g_rr_trans[0][1]:.4f}| = {abs(g_rl_trans[0][1] - g_rr_trans[0][1]):.4f}m')

seq_len = g_root_trans.shape[0]
print(f'处理帧数: {seq_len}')

# 调试：打印脚的 Y 坐标（第一帧）
print(f'\n=== 脚的 Y 坐标（第一帧，缩放后）===')
print(f'g_fl_trans[0] (应该是左前脚，Y>0): Y = {g_fl_trans[0, 1]:.4f}')
print(f'g_fr_trans[0] (应该是右前脚，Y<0): Y = {g_fr_trans[0, 1]:.4f}')
print(f'g_rl_trans[0] (应该是左后脚，Y>0): Y = {g_rl_trans[0, 1]:.4f}')
print(f'g_rr_trans[0] (应该是右后脚，Y<0): Y = {g_rr_trans[0, 1]:.4f}')
print(f'Himmy 坐标系: Y>0 是左边, Y<0 是右边')


# ==================== 处理root旋转（坐标系转换）====================

for i in range(g_root_rot_euler.shape[0]):
    if g_root_rot_euler[i, 2] <= np.pi / 2:
        g_root_rot_euler[i, 2] = 2*np.pi + g_root_rot_euler[i, 2]
g_root_rot_euler[:, 2] = g_root_rot_euler[:, 2] - np.pi

g_root_rot_euler[:, 0] = np.pi / 2 - g_root_rot_euler[:, 0]
for i in range(g_root_rot_euler.shape[0]):
    for j in range(3):
        if g_root_rot_euler[i, j] >= np.pi:
            g_root_rot_euler[i, j] = g_root_rot_euler[i, j] - 2 * np.pi
        elif g_root_rot_euler[i, j] <= -np.pi:
            g_root_rot_euler[i, j] = 2 * np.pi + g_root_rot_euler[i, j]
g_root_rot_euler[:, 0] = -g_root_rot_euler[:, 0]
g_root_rot_euler[:, 1] = -g_root_rot_euler[:, 1]


# ==================== 旋转到X轴正方向 ====================

rot_matrix = kinematic.rot_matrix_ba([0, 0, -g_root_rot_euler[0, -1]])
g_root_trans = np.array([rot_matrix@t for t in g_root_trans])
g_root_rot_euler[:, -1] -= g_root_rot_euler[0, -1]
g_fl_trans = np.array([rot_matrix@t for t in g_fl_trans])
g_fr_trans = np.array([rot_matrix@t for t in g_fr_trans])
g_rl_trans = np.array([rot_matrix@t for t in g_rl_trans])
g_rr_trans = np.array([rot_matrix@t for t in g_rr_trans])

# 调试：旋转后脚的 Y 坐标
print(f'\n=== 旋转到X轴正方向后，脚的 Y 坐标（第20帧）===')
print(f'g_fl_trans[20] (左前脚，应该Y>0): Y = {g_fl_trans[20, 1]:.4f}')
print(f'g_fr_trans[20] (右前脚，应该Y<0): Y = {g_fr_trans[20, 1]:.4f}')
print(f'g_rl_trans[20] (左后脚，应该Y>0): Y = {g_rl_trans[20, 1]:.4f}')
print(f'g_rr_trans[20] (右后脚，应该Y<0): Y = {g_rr_trans[20, 1]:.4f}')







# ==================== 从Spine1提取脊柱角度 ====================

print('\n提取脊柱姿态...')
spine_angles_all = np.zeros((seq_len, 3))

for i in range(seq_len):
    # Spine1相对于Hips的旋转 = Hips^(-1) * Spine1
    hips_q = g_hips_quat[i]
    spine1_q = g_spine1_quat[i]
    
    # 四元数求逆: q^(-1) = [-x, -y, -z, w]
    hips_q_inv = np.array([-hips_q[0], -hips_q[1], -hips_q[2], hips_q[3]])
    
    # 四元数乘法: q1 * q2
    x1, y1, z1, w1 = hips_q_inv
    x2, y2, z2, w2 = spine1_q
    rel_q = np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])
    
    # 转换为欧拉角
    spine1_rel_euler = kinematic.quaternion2rpy(rel_q)
    
    # 映射到Himmy的三个脊柱关节
    # Source Spine1的相对旋转: [roll, pitch, yaw]
    # Himmy脊柱关节顺序: yaw -> pitch -> roll
    
    # 【重要】Source骨架中Spine1相对于Hips有约180°的Yaw偏移
    # 这是骨架本身的朝向问题，不是真正的脊柱扭转
    # 需要移除这个偏移，只保留真正的脊柱运动
    raw_yaw = spine1_rel_euler[2]
    if raw_yaw < 0:
        corrected_yaw = raw_yaw + np.pi
    else:
        corrected_yaw = raw_yaw - np.pi
    
    spine_angles_all[i, 0] = corrected_yaw           # yaw (绕Z轴，已校正180°偏移)
    spine_angles_all[i, 1] = spine1_rel_euler[1]     # pitch (绕Y轴)
    spine_angles_all[i, 2] = spine1_rel_euler[0]     # roll (绕X轴)

# === 脊柱角度不做缩放和平滑，直接使用原始角度 ===

# 脊柱关节限位
spine_yaw_limit = [-0.5, 0.5]
spine_pitch_limit = [-0.8, 0.8]
spine_roll_limit = [-0.5, 0.5]

spine_angles_all[:, 0] = np.clip(spine_angles_all[:, 0], spine_yaw_limit[0], spine_yaw_limit[1])
spine_angles_all[:, 1] = np.clip(spine_angles_all[:, 1], spine_pitch_limit[0], spine_pitch_limit[1])
spine_angles_all[:, 2] = np.clip(spine_angles_all[:, 2], spine_roll_limit[0], spine_roll_limit[1])

print(f'脊柱角度范围（原始角度，仅限位）:')
print(f'  Yaw:   [{spine_angles_all[:, 0].min():.3f}, {spine_angles_all[:, 0].max():.3f}]')
print(f'  Pitch: [{spine_angles_all[:, 1].min():.3f}, {spine_angles_all[:, 1].max():.3f}]')
print(f'  Roll:  [{spine_angles_all[:, 2].min():.3f}, {spine_angles_all[:, 2].max():.3f}]')


# ==================== 新方法：Source Root -> Himmy roll_spine_Link ====================

# 核心思想：
# 1. Source的root(Hips)在动物臀部，直接对应Himmy的roll_spine_Link
# 2. 从roll_spine通过脊柱逆向运动学推算trunk位置
# 3. 后腿：roll_spine位置确定 → 后腿hip位置确定 → IK求后腿关节
# 4. 前腿：trunk位置确定 → 前腿hip位置确定 → IK求前腿关节

print('\n使用新方法: Source Root -> Himmy roll_spine_Link')

# Source的root位置和旋转直接作为Himmy的roll_spine_Link
g_roll_spine_trans = g_root_trans.copy()
g_roll_spine_rot_euler = g_root_rot_euler.copy()

# 应用 x 轴偏移补偿（修正 T-pose 中脚位置的差异）
g_roll_spine_trans[:, 0] += root_x_offset.item()

print(f'Roll_spine位置范围（已应用x偏移）: X=[{g_roll_spine_trans[:, 0].min():.3f}, {g_roll_spine_trans[:, 0].max():.3f}]')


# ==================== 逆运动学求解 ====================

print('\n进行逆运动学求解...')

leg_joint_angles = np.zeros((seq_len, 12))  # FR(3), FL(3), RR(3), RL(3)
g_trunk_trans = np.zeros((seq_len, 3))      # 存储计算出的trunk位置
g_trunk_rot_euler = np.zeros((seq_len, 3))  # 存储计算出的trunk旋转

# hip坐标系旋转（绕Y轴旋转90度）
r_hip = [0, np.pi / 2, 0]

for i in range(seq_len):
    spine_ang = spine_angles_all[i]
    roll_spine_pos = g_roll_spine_trans[i]
    roll_spine_rot = g_roll_spine_rot_euler[i]
    
    # ========== Step 1: 从roll_spine通过脊柱IK计算trunk位置 ==========
    trunk_pos, trunk_rot = spine_inverse_kinematics(spine_ang, roll_spine_pos, roll_spine_rot)
    g_trunk_trans[i] = trunk_pos
    g_trunk_rot_euler[i] = trunk_rot
    
    # ========== Step 2: 后腿IK (基于roll_spine) ==========
    # roll_spine位置已确定，后腿hip相对于roll_spine的位置是固定的
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
    
    # 与A1保持一致：不翻转Y轴
    # IK函数通过 is_left 参数自动处理左右腿的hip偏移符号
    
    # 求解后腿关节角度
    rl_ang = kinematic.inverse_kinematics_leg(rl_foot_hip, is_left=True)
    rl_ang = np.array([rl_ang[0], -rl_ang[1], -rl_ang[2]])
    rr_ang = kinematic.inverse_kinematics_leg(rr_foot_hip, is_left=False)
    rr_ang = np.array([rr_ang[0], -rr_ang[1], -rr_ang[2]])
    
    # ========== Step 3: 前腿IK (基于trunk) ==========
    # trunk位置已通过脊柱IK确定，前腿hip相对于trunk的位置是固定的
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
    
    # 与A1保持一致：不翻转Y轴
    # IK函数通过 is_left 参数自动处理左右腿的hip偏移符号
    
    # 调试：打印第一帧的脚在hip坐标系中的位置（与A1一致，不翻转Y轴）
    if i == 0:
        print(f'\n=== 第一帧：脚在 hip 坐标系中的位置（无Y翻转，与A1一致）===')
        print(f'FL foot in FL_hip frame: {fl_foot_hip}')
        print(f'FR foot in FR_hip frame: {fr_foot_hip}')
        print(f'RL foot in RL_hip frame: {rl_foot_hip}')
        print(f'RR foot in RR_hip frame: {rr_foot_hip}')
        
        # 分析脚间距缩小的原因
        print(f'\n=== 第一帧：脚在世界坐标系中的位置 ===')
        print(f'FL foot world: {g_fl_trans[i]}')
        print(f'FR foot world: {g_fr_trans[i]}')
        print(f'RL foot world: {g_rl_trans[i]}')
        print(f'RR foot world: {g_rr_trans[i]}')
        
        print(f'\n=== 第一帧：Hip位置（世界坐标系）===')
        # 前腿hip位置
        T_w_trunk = kinematic.trans_matrix_ba(trunk_pos, trunk_rot)
        fl_hip_world = (T_w_trunk @ np.append(fl_hip_offset, 1))[:3]
        fr_hip_world = (T_w_trunk @ np.append(fr_hip_offset, 1))[:3]
        print(f'FL hip world: {fl_hip_world}')
        print(f'FR hip world: {fr_hip_world}')
        print(f'前腿hip Y间距: {abs(fl_hip_world[1] - fr_hip_world[1]):.4f}m')
        
        # 后腿hip位置
        T_w_roll = kinematic.trans_matrix_ba(roll_spine_pos, roll_spine_rot)
        rl_hip_world = (T_w_roll @ np.append(rl_hip_local, 1))[:3]
        rr_hip_world = (T_w_roll @ np.append(rr_hip_local, 1))[:3]
        print(f'RL hip world: {rl_hip_world}')
        print(f'RR hip world: {rr_hip_world}')
        print(f'后腿hip Y间距: {abs(rl_hip_world[1] - rr_hip_world[1]):.4f}m')
        
        print(f'\n=== BODY_WIDE参数 ===')
        print(f'BODY_WIDE = {BODY_WIDE}m')
        print(f'理论hip间距应该是: {BODY_WIDE}m')
    
    # 求解前腿关节角度
    fl_ang = kinematic.inverse_kinematics_leg(fl_foot_hip, is_left=True)
    fl_ang = np.array([fl_ang[0], -fl_ang[1], -fl_ang[2]])
    fr_ang = kinematic.inverse_kinematics_leg(fr_foot_hip, is_left=False)
    fr_ang = np.array([fr_ang[0], -fr_ang[1], -fr_ang[2]])
    
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

hip_limit = [-0.50, 0.50]
thigh_limit = [-1.5, 3.5]
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
root_height_offset = 0.2
# 让所有关节的高度都加上偏移
local_translation[:, 2] = local_translation[:, 2] + root_height_offset
root_translation[:, 2] = root_translation[:, 2] + root_height_offset


# ==================== 保存结果 ====================

skeleton_state = SkeletonState.from_rotation_and_root_translation(
    target_tpose.skeleton_tree,
    local_rotation,
    root_translation,
    is_local=True
)

target_motion = SkeletonMotion.from_skeleton_state(skeleton_state, fps=source_motion.fps)

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

output_path = output_dir + '{}.npy'.format(output_file)
target_motion.to_file(output_path)

print(f'\n保存完成: {output_path}')
print(f'  帧数: {seq_len}')
print(f'  FPS: {source_motion.fps}')
print(f'  总时长: {seq_len / source_motion.fps:.2f}s')

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
