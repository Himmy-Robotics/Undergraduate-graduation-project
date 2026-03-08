#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: key_points_retargetting.py
@Auth: Huiqiao
@Date: 2022/7/6

A1 Motion Retargeting（基于关键点的运动重定向）
==============================================

该模块实现了将动物动捕数据重定向到A1机器人的功能。

主要流程:
1. 加载source motion（动物动捕数据）和T-pose（source和target）
2. 根据T-pose计算尺寸缩放比例
3. 对source motion应用缩放
4. 提取关键点（root位置、4个足端位置）
5. 处理root旋转，使机器人朝向X轴正方向
6. 调整root位置到机器人重心
7. 使用逆运动学计算关节角度
8. 应用关节限位
9. 生成target motion并保存

Source骨架结构（AI4Animation狗模型，27个关节）:
- Hips (root) -> Spine -> Spine1 -> Neck -> Head
- LeftShoulder -> LeftArm -> LeftForeArm -> LeftHand -> LeftHand_End (前左腿)
- RightShoulder -> RightArm -> RightForeArm -> RightHand -> RightHand_End (前右腿)
- LeftUpLeg -> LeftLeg -> LeftFoot -> LeftFoot_End (后左腿)
- RightUpLeg -> RightLeg -> RightFoot -> RightFoot_End (后右腿)
- Tail -> Tail1 -> Tail1_End

Target骨架结构（A1机器人，17个节点）:
- trunk (root)
- FR_hip -> FR_thigh -> FR_calf -> FR_foot (前右腿)
- FL_hip -> FL_thigh -> FL_calf -> FL_foot (前左腿)
- RR_hip -> RR_thigh -> RR_calf -> RR_foot (后右腿)
- RL_hip -> RL_thigh -> RL_calf -> RL_foot (后左腿)

关键点映射关系:
- Source Hips -> Target trunk (需要平移变换，因为Hips在尾部，trunk在躯干中心)
- Source RightHand_End -> Target FR_foot (前右)
- Source LeftHand_End -> Target FL_foot (前左)
- Source RightFoot_End -> Target RR_foot (后右)
- Source LeftFoot_End -> Target RL_foot (后左)
"""
import copy
import json
import numpy as np
from isaacgym.torch_utils import *
from kinematics import Kinematics
from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from poselib.skeleton.backend.fbx.fbx_read_wrapper import fbx_to_array
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


# ==================== 配置文件加载 ====================

# 加载配置文件，指定要处理的motion文件和裁剪区间
json_dir = "data/load_config.json"
with open(json_dir, "r") as f:
    motion_json = json.load(f)
    file_name = motion_json["file_name"]      # FBX文件名
    clip = motion_json["clip"]                # 裁剪区间 [start, end]
    remarks = motion_json["remarks"]          # 输出文件备注

root_dir = "data/amp_hardware_a1/{}/".format(file_name)
output_file = '{}_amp_{}_{}_{}'.format(file_name, clip[0], clip[1], remarks)


# ==================== A1机器人参数初始化 ====================

# A1机器人几何参数（单位：米）
body_length = 0.366      # 机体长度（前后髋关节距离）
body_wide = 0.094        # 机体宽度（左右髋关节距离）
hip_length = 0.08505     # 髋关节长度
thigh_length = 0.2       # 大腿长度
calf_length = 0.2        # 小腿长度

# 初始化运动学求解器
kinematic = Kinematics(body_length=0.366, body_wide=0.094, hip_length=0.08505, 
                      thigh_length=0.2, calf_length=0.2)


# ==================== 数据加载 ====================

# 加载source motion（动物动捕数据）
source_motion = SkeletonMotion.from_file(root_dir + '{}.npy'.format(file_name))


# ==================== 尺寸缩放计算 ====================

# 加载T-pose（标准站立姿态）
source_tpose = SkeletonState.from_file('data/dog_tpose.npy')     # 动物T-pose
target_tpose = SkeletonState.from_file('data/amp_a1_tpose.npy')  # A1 T-pose

# 获取骨架树结构
skeleton_s = source_tpose.skeleton_tree
skeleton_t = target_tpose.skeleton_tree

# 计算source T-pose的关键尺寸（使用4个足端点的全局位置）
# 注意：source T-pose单位是厘米，需要转换为米
# source_length: 前后腿之间的距离（X方向）
source_length = torch.abs(source_tpose.global_translation[skeleton_s.index('LeftHand_End'), 0] -
                          source_tpose.global_translation[skeleton_s.index('LeftFoot_End'), 0]) / 100.0
# source_wide: 左右腿之间的距离（Y方向）
source_wide = torch.abs(source_tpose.global_translation[skeleton_s.index('LeftHand_End'), 1] -
                        source_tpose.global_translation[skeleton_s.index('RightHand_End'), 1]) / 100.0
# source_height: Hips到地面的高度（Z方向）
source_height = torch.abs(source_tpose.global_translation[skeleton_s.index('Hips'), 2] -
                          source_tpose.global_translation[skeleton_s.index('LeftFoot_End'), 2]) / 100.0

# 计算target T-pose的关键尺寸（使用4个足端点的全局位置）
# target_length: 前后腿之间的距离（X方向）
target_length = torch.abs(target_tpose.global_translation[skeleton_t.index('FL_foot'), 0] -
                          target_tpose.global_translation[skeleton_t.index('RL_foot'), 0])
# target_wide: 左右腿之间的距离（Y方向）
target_wide = torch.abs(target_tpose.global_translation[skeleton_t.index('FL_foot'), 1] -
                        target_tpose.global_translation[skeleton_t.index('FR_foot'), 1])
# target_height: trunk到地面的高度（Z方向）
target_height = torch.abs(target_tpose.global_translation[skeleton_t.index('trunk'), 2] -
                          target_tpose.global_translation[skeleton_t.index('FL_foot'), 2])

# 计算三个方向的缩放比例
zoom_x = (target_length / source_length) * 1.  # X方向缩放（长度）
zoom_y = target_wide / source_wide              # Y方向缩放（宽度）
zoom_z = target_height / source_height          # Z方向缩放（高度）

# 获取source motion的骨架信息
skeleton = source_motion.skeleton_tree
local_translation = skeleton.local_translation   # 局部位移 (27, 3)
root_translation = source_motion.root_translation  # root全局位移序列 (seq_len, 3)

print('zoom_x: {}, zoom_y: {}, zoom_z: {}'.format(zoom_x, zoom_y, zoom_z))


# ==================== 应用缩放到source motion ====================

# 对每个关节的局部位移应用缩放
# 注意：后腿足端点需要特殊处理，交换X和Y方向的缩放
for i in range(local_translation.shape[0]):
    if i == skeleton.node_names.index('LeftFoot_End') or i == skeleton.node_names.index('RightFoot_End'):
        # 后腿足端点：X方向用Y缩放，Y方向用X缩放
        # 这是因为后腿的局部坐标系与前腿不同
        local_translation[i, 0] *= zoom_y
        local_translation[i, 1] *= zoom_x
        local_translation[i, 2] *= zoom_z
        continue
    # 其他关节：正常缩放
    local_translation[i, 0] *= zoom_x  # X方向（前后）
    local_translation[i, 1] *= zoom_y  # Y方向（左右）
    local_translation[i, 2] *= zoom_z  # Z方向（上下）

# 对root位移应用缩放
root_translation[:, 0] *= zoom_x
root_translation[:, 1] *= zoom_y
root_translation[:, 2] *= zoom_z


# ==================== 提取关键点的全局位置 ====================

# 获取全局变换矩阵（缩放后自动更新）
global_trans = source_motion.global_transformation  # (seq_len, 27, 7)
# 每个关节的全局变换表示为7维: [qx, qy, qz, qw, tx, ty, tz]
# 前4维是四元数，后3维是全局位移

# 提取Hips（root）的全局位置
global_root_translation = global_trans[:, skeleton.node_names.index("Hips"), -3:]  # (seq_len, 3)

# 提取4个足端点的全局位置
global_left_hand_end_trans = global_trans[:, skeleton.node_names.index("LeftHand_End"), -3:]    # 左前足 (seq_len, 3)
global_right_hand_end_trans = global_trans[:, skeleton.node_names.index("RightHand_End"), -3:]  # 右前足 (seq_len, 3)
global_left_foot_end_trans = global_trans[:, skeleton.node_names.index("LeftFoot_End"), -3:]    # 左后足 (seq_len, 3)
global_right_foot_end_trans = global_trans[:, skeleton.node_names.index("RightFoot_End"), -3:]  # 右后足 (seq_len, 3)


# ==================== 提取root旋转 ====================

# 提取root的全局旋转（四元数）
global_root_rotation_quat = global_trans[:, skeleton.node_names.index("Hips"), :4]  # (seq_len, 4)

# 将四元数转换为欧拉角（Roll-Pitch-Yaw）
global_root_rotation_quat = global_root_rotation_quat.numpy()
global_root_rotation_euler = [kinematic.quaternion2rpy(q) for q in global_root_rotation_quat]
global_root_rotation_euler = np.array(global_root_rotation_euler)  # (seq_len, 3)
# 注意：这里使用kinematics的quaternion2rpy方法，而不是get_euler_xyz
# 两者可能有微小差异，但kinematics方法与A1的控制系统一致


# ==================== 裁剪指定区间 ====================

# inverse kinematic
g_root_trans = global_root_translation.numpy()[clip[0]:clip[1], :]  # (clip_len, 3)
g_root_rot_euler = global_root_rotation_euler[clip[0]:clip[1], :]   # (clip_len, 3)
g_lh_trans = global_left_hand_end_trans.numpy()[clip[0]:clip[1], :]  # 左前足 (clip_len, 3)
g_rh_trans = global_right_hand_end_trans.numpy()[clip[0]:clip[1], :]  # 右前足 (clip_len, 3)
g_lf_trans = global_left_foot_end_trans.numpy()[clip[0]:clip[1], :]  # 左后足 (clip_len, 3)
g_rf_trans = global_right_foot_end_trans.numpy()[clip[0]:clip[1], :]  # 右后足 (clip_len, 3)


# ==================== 处理root旋转（转换到A1坐标系）====================

# 处理Yaw角（绕Z轴旋转）到[-π, π]范围
# source数据的Yaw角可能在[0, 2π]范围，需要转换
for i in range(g_root_rot_euler.shape[0]):
    if g_root_rot_euler[i, 2] <= np.pi / 2:
        g_root_rot_euler[i, 2] = 2*np.pi + g_root_rot_euler[i, 2]
g_root_rot_euler[:, 2] = g_root_rot_euler[:, 2] - np.pi  # 将[π/2, 2π]映射到[-π/2, π]

# 处理Roll角（绕X轴旋转）
# source坐标系：动物站立时Roll=0（Z轴向上）
# target坐标系：A1站立时Roll=π/2（Y轴向上）
g_root_rot_euler[:, 0] = np.pi / 2 - g_root_rot_euler[:, 0]

# 将所有角度归一化到[-π, π]范围
for i in range(g_root_rot_euler.shape[0]):
    for j in range(3):
        if g_root_rot_euler[i, j] >= np.pi:
            g_root_rot_euler[i, j] = g_root_rot_euler[i, j] - 2 * np.pi
        elif g_root_rot_euler[i, j] <= -np.pi:
            g_root_rot_euler[i, j] = 2 * np.pi + g_root_rot_euler[i, j]

# 反转Roll和Pitch的符号（坐标系转换）
g_root_rot_euler[:, 0] = -g_root_rot_euler[:, 0]  # Roll
g_root_rot_euler[:, 1] = -g_root_rot_euler[:, 1]  # Pitch


# ==================== 旋转到X轴正方向 ====================

# 将整个motion旋转，使第一帧的Yaw角为0（朝向X轴正方向）
# 这样可以保证motion从X轴正方向开始
rot_matrix = kinematic.rot_matrix_ba([0, 0, -g_root_rot_euler[0, -1]])  # 绕Z轴旋转矩阵

# 旋转所有关键点的全局位置
g_root_trans = np.array([rot_matrix@t for t in g_root_trans])
# 旋转所有关键点的全局位置
g_root_trans = np.array([rot_matrix@t for t in g_root_trans])
g_lh_trans = np.array([rot_matrix@t for t in g_lh_trans])
g_rh_trans = np.array([rot_matrix@t for t in g_rh_trans])
g_lf_trans = np.array([rot_matrix@t for t in g_lf_trans])
g_rf_trans = np.array([rot_matrix@t for t in g_rf_trans])

# 更新Yaw角（减去初始Yaw）
g_root_rot_euler[:, -1] -= g_root_rot_euler[0, -1]

# 合并4个足端点位置为一个数组，顺序：FR, FL, RR, RL
g_foot_trans = np.hstack([g_rh_trans, g_lh_trans, g_rf_trans, g_lf_trans])  # (clip_len, 12)


# ==================== 调整root位置（躯干重心） ====================

# 将root位置从Hips（臀部）调整到A1机器人的躯干中心
# A1的trunk位于机体中心，而动物的Hips在尾部
# 需要向前移动约 body_length/3.5 的距离
root_new_local = np.array([-body_length / 3.5, 0, 0, 1])  # 齐次坐标
g_root_trans = np.array([list((kinematic.trans_matrix_ba(m, t) @ root_new_local)[:3])
                         for m, t in zip(g_root_trans, g_root_rot_euler)])


# ==================== 逆运动学求解 ====================

# 对每一帧使用逆运动学计算12个关节角度
joint_rot_euler = np.zeros((g_root_trans.shape[0], 12))
for i in range(g_root_trans.shape[0]):
    ang = kinematic.inverse_kinematics(g_root_trans[i, :], g_root_rot_euler[i, :], g_foot_trans[i, :])
    joint_rot_euler[i, :] = ang

# 检查是否有NaN值（IK求解失败）
if np.isnan(joint_rot_euler).any():
    raise ValueError('Angle can not be nan!')


# ==================== 关节角度限位 ====================

# 根据A1 URDF定义的关节限位进行裁剪
hip_limit = [-0.30, 0.30]      # 髋关节（外展/内收）限位
thigh_limit = [-1.04, 4.18]    # 大腿关节（前后摆动）限位
calf_limit = [-2.69, -0.91]    # 小腿关节（膝关节弯曲）限位

for i in range(joint_rot_euler.shape[0]):
    for j in range(joint_rot_euler.shape[1]):
        if j==0 or j==3 or j==6 or j==9:  # Hip joints
            joint_rot_euler[i, j] = np.clip(joint_rot_euler[i, j], hip_limit[0], hip_limit[1])
        elif j==1 or j==4 or j==7 or j==10:  # Thigh joints
            joint_rot_euler[i, j] = np.clip(joint_rot_euler[i, j], thigh_limit[0], thigh_limit[1])
        elif j==2 or j==5 or j==8 or j==11:  # Calf joints
            joint_rot_euler[i, j] = np.clip(joint_rot_euler[i, j], calf_limit[0], calf_limit[1])


# ==================== 构建target motion ====================

# 将欧拉角转换为四元数
g_root_rot_euler = torch.from_numpy(g_root_rot_euler)
g_root_rot_quat = quat_from_euler_xyz(g_root_rot_euler[:, 0], g_root_rot_euler[:, 1], g_root_rot_euler[:, 2])

# 将关节角度转换为tensor
joint_rot_euler = torch.from_numpy(joint_rot_euler)  # (clip_len, 12)

# 准备局部旋转数组
skeleton_target = target_tpose.skeleton_tree
local_rotation = torch.zeros((joint_rot_euler.shape[0], 17, 4))  # (clip_len, 17, 4)

# 初始化所有四元数为单位四元数 [x=0, y=0, z=0, w=1]
local_rotation[:, :, -1] = 1

# 填充每一帧的局部旋转
for i in range(joint_rot_euler.shape[0]):
    joint_rot_quat = []
    # 将12个关节的欧拉角转换为四元数
    for j in range(joint_rot_euler.shape[1]):
        if j == 0 or j == 3 or j == 6 or j == 9:  # Hip joints (绕X轴旋转)
            joint_quat = kinematic.rpy2quaternion([joint_rot_euler[i, j], 0, 0])
        else:  # Thigh和Calf joints (绕Y轴旋转)
            joint_quat = kinematic.rpy2quaternion([0, joint_rot_euler[i, j], 0])
        joint_quat = torch.from_numpy(np.array(joint_quat))
        joint_rot_quat.append(joint_quat)
    
    # 填充A1骨架的17个节点的局部旋转
    # A1骨架结构: ['trunk', 'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot', 
    #              'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot', 
    #              'RR_hip', 'RR_thigh', 'RR_calf', 'RR_foot', 
    #              'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot']
    local_rotation[i, 0, :] = g_root_rot_quat[i, :]    # trunk (root)
    local_rotation[i, 1, :] = joint_rot_quat[0]         # FR_hip
    local_rotation[i, 2, :] = joint_rot_quat[1]         # FR_thigh
    local_rotation[i, 3, :] = joint_rot_quat[2]         # FR_calf
    # FR_foot (index=4) 保持单位四元数
    local_rotation[i, 5, :] = joint_rot_quat[3]         # FL_hip
    local_rotation[i, 6, :] = joint_rot_quat[4]         # FL_thigh
    local_rotation[i, 7, :] = joint_rot_quat[5]         # FL_calf
    # FL_foot (index=8) 保持单位四元数
    local_rotation[i, 9, :] = joint_rot_quat[6]         # RR_hip
    local_rotation[i, 10, :] = joint_rot_quat[7]        # RR_thigh
    local_rotation[i, 11, :] = joint_rot_quat[8]        # RR_calf
    # RR_foot (index=12) 保持单位四元数
    local_rotation[i, 13, :] = joint_rot_quat[9]        # RL_hip
    local_rotation[i, 14, :] = joint_rot_quat[10]       # RL_thigh
    local_rotation[i, 15, :] = joint_rot_quat[11]       # RL_calf
    # RL_foot (index=16) 保持单位四元数


# ==================== 计算root translation ====================

# 将root位置从trunk调整回机体中心
# g_root_trans是trunk位置，需要变换到机体后部（机体几何中心）
root_new_local = np.array([body_length / 2, 0, 0, 1])  # 向后移动半个机体长度，齐次坐标
root_trans_w = np.array([list((kinematic.trans_matrix_ba(m, t) @ root_new_local)[:3])
                         for m, t in zip(g_root_trans, g_root_rot_euler)])

# 将motion起点设置在原点（X和Y归零）
root_translation = torch.from_numpy(root_trans_w)
root_translation[:, 0] = root_translation[:, 0] - root_translation[0, 0]  # X归零
root_translation[:, 1] = root_translation[:, 1] - root_translation[0, 1]  # Y归零

# 添加一个小的高度偏移，确保足端不会陷入地面
root_height_offset = 0.02  # 2cm的安全余量
root_translation += root_height_offset


# ==================== 生成并保存target motion ====================

# 从局部旋转和root平移构建SkeletonState
skeleton_state = SkeletonState.from_rotation_and_root_translation(
    target_tpose.skeleton_tree, 
    local_rotation,
    root_translation, 
    is_local=True
)

# 生成SkeletonMotion并保存
target_motion = SkeletonMotion.from_skeleton_state(skeleton_state, fps=source_motion.fps)
target_motion.to_file(root_dir + '{}.npy'.format(output_file))

# 重新加载验证
target_motion = SkeletonMotion.from_file(root_dir + '{}.npy'.format(output_file))

print(f'Motion retargeting完成！')
print(f'输出文件: {root_dir}{output_file}.npy')
print(f'帧数: {target_motion.local_rotation.shape[0]}')
print(f'FPS: {target_motion.fps}')
plot_skeleton_motion_interactive(target_motion)
