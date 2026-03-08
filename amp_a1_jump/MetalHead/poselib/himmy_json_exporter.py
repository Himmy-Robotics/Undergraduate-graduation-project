#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: himmy_json_exporter.py
@Auth: Based on json_exporter.py by Huiqiao, modified for Himmy Mark2
@Date: 2024/12

Himmy Mark2 JSON Exporter
=========================

该模块将retargeting后的npy文件导出为训练所需的json文件。

与A1的json_exporter主要区别：
- Himmy有15个DOF（12腿部 + 3脊柱），A1只有12个DOF
- 数据格式为67维（A1为61维）

Himmy Mark2 数据格式 (67维):
    [0:3]   root_pos (3)        - 根位置 [x, y, z]
    [3:7]   root_rot (4)        - 根旋转四元数 [x, y, z, w]
    [7:19]  joint_pos (12)      - 腿部关节角度 [FL(3), FR(3), RL(3), RR(3)]
    [19:22] spine_pos (3)       - 脊柱关节角度 [yaw, pitch, roll]
    [22:34] foot_pos (12)       - 足端位置 [FL(3), FR(3), RL(3), RR(3)]
    [34:37] lin_vel (3)         - 线速度
    [37:40] ang_vel (3)         - 角速度
    [40:52] joint_vel (12)      - 腿部关节速度
    [52:55] spine_vel (3)       - 脊柱关节速度
    [55:67] foot_vel (12)       - 足端速度

Himmy Mark2 骨架结构 (20个节点):
    0: trunk                    - 根节点
    1: FL_hip_Link              - 前左髋
    2: FL_thigh_Link            - 前左大腿
    3: FL_calf_Link             - 前左小腿
    4: FL_foot                  - 前左脚
    5: FR_hip_Link              - 前右髋
    6: FR_thigh_Link            - 前右大腿
    7: FR_calf_Link             - 前右小腿
    8: FR_foot                  - 前右脚
    9: yaw_spine_Link           - 脊柱yaw
    10: pitch_spine_Link        - 脊柱pitch
    11: roll_spine_Link         - 脊柱roll
    12: RL_hip_Link             - 后左髋
    13: RL_thigh_Link           - 后左大腿
    14: RL_calf_Link            - 后左小腿
    15: RL_foot                 - 后左脚
    16: RR_hip_Link             - 后右髋
    17: RR_thigh_Link           - 后右大腿
    18: RR_calf_Link            - 后右小腿
    19: RR_foot                 - 后右脚
"""

import json
import os
import sys
import numpy as np

# 设置路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'poselib'))

# 导入poselib模块
from poselib.skeleton.skeleton3d import SkeletonMotion

# 导入Himmy运动学模块（用于四元数转换）
from himmy_kinematics import HimmyKinematics


def differential(d, dt=1/60):
    """
    计算差分（速度）
    
    Args:
        d: 数据数组 (num_frames, dim)
        dt: 时间间隔
    
    Returns:
        差分数据列表
    """
    diff_data = []
    diff = np.zeros(d.shape[1]) if len(d.shape) > 1 else 0
    for i in range(1, d.shape[0]):
        diff = (d[i, :] - d[i-1, :]) / dt if len(d.shape) > 1 else (d[i] - d[i-1]) / dt
        diff_data.append(diff)
    diff_data.append(diff)  # 最后一帧使用前一帧的速度
    return diff_data


def quaternion_to_axis_angle(quat):
    """
    四元数转轴角表示
    
    Args:
        quat: 四元数 [x, y, z, w]
    
    Returns:
        轴角 [ax, ay, az, angle]
    """
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    
    # 确保w为正（选择最短路径）
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    
    # 计算角度
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    
    # 计算轴
    sin_half_angle = np.sqrt(1 - w * w)
    if sin_half_angle < 1e-6:
        # 角度接近0，轴可以是任意的
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    ax = x / sin_half_angle
    ay = y / sin_half_angle
    az = z / sin_half_angle
    
    return np.array([ax, ay, az, angle])


def export_himmy_json(input_npy_path, output_json_path, visualize=False):
    """
    将Himmy Mark2的npy文件导出为json格式
    
    Args:
        input_npy_path: 输入npy文件路径
        output_json_path: 输出json文件路径
        visualize: 是否打印详细信息
    """
    print("=" * 60)
    print("Himmy Mark2 JSON Exporter")
    print("=" * 60)
    
    # 1. 加载motion数据
    print(f"\n[Step 1] 加载motion数据: {input_npy_path}")
    target_motion = SkeletonMotion.from_file(input_npy_path)
    skeleton = target_motion.skeleton_tree
    
    dt = 1 / target_motion.fps
    num_frames = len(target_motion)
    
    print(f"  帧数: {num_frames}")
    print(f"  FPS: {target_motion.fps}")
    print(f"  dt: {dt:.6f}s")
    
    # 打印骨架结构
    print(f"\n骨架结构 ({len(skeleton.node_names)} 个节点):")
    for i, name in enumerate(skeleton.node_names):
        print(f"  {i}: {name}")
    
    # 2. 提取全局变换
    print(f"\n[Step 2] 提取全局变换...")
    global_trans = target_motion.global_transformation  # (num_frames, num_nodes, 7)
    # 格式: [qx, qy, qz, qw, tx, ty, tz] 或 [rot(4), trans(3)]
    
    # 3. 提取root位置和旋转
    print(f"\n[Step 3] 提取root位置和旋转...")
    trunk_idx = skeleton.node_names.index('trunk')
    root_pos = global_trans[:, trunk_idx, -3:].numpy()  # (num_frames, 3)
    root_rot = global_trans[:, trunk_idx, :4].numpy()   # (num_frames, 4) [x, y, z, w]
    
    print(f"  root_pos 范围: X=[{root_pos[:, 0].min():.3f}, {root_pos[:, 0].max():.3f}], "
          f"Y=[{root_pos[:, 1].min():.3f}, {root_pos[:, 1].max():.3f}], "
          f"Z=[{root_pos[:, 2].min():.3f}, {root_pos[:, 2].max():.3f}]")
    
    # 4. 提取局部旋转并转换为关节角度
    print(f"\n[Step 4] 提取关节角度...")
    local_rot = target_motion.local_rotation  # (num_frames, num_nodes, 4) [x, y, z, w]
    
    # 转换所有四元数为轴角
    local_rot_flat = local_rot.reshape(-1, 4).numpy()
    joint_pos_euler = []
    for i in range(local_rot_flat.shape[0]):
        axis_ang = quaternion_to_axis_angle(local_rot_flat[i, :])
        joint_ang = axis_ang[:3] * axis_ang[-1]  # 轴 * 角度
        joint_pos_euler.append(joint_ang)
    joint_pos_euler = np.array(joint_pos_euler).reshape((num_frames, -1, 3))
    
    # Himmy Mark2 关节索引映射:
    # 腿部关节: hip绕X轴, thigh绕Y轴, calf绕Y轴
    # 骨架顺序: trunk(0), FL_hip(1), FL_thigh(2), FL_calf(3), FL_foot(4),
    #          FR_hip(5), FR_thigh(6), FR_calf(7), FR_foot(8),
    #          yaw_spine(9), pitch_spine(10), roll_spine(11),
    #          RL_hip(12), RL_thigh(13), RL_calf(14), RL_foot(15),
    #          RR_hip(16), RR_thigh(17), RR_calf(18), RR_foot(19)
    
    # 提取腿部关节角度 (12维)
    # 顺序: [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, 
    #        RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
    joint_pos = np.vstack([
        joint_pos_euler[:, 1, 0],   # FL_hip (X轴)
        joint_pos_euler[:, 2, 1],   # FL_thigh (Y轴)
        joint_pos_euler[:, 3, 1],   # FL_calf (Y轴)
        joint_pos_euler[:, 5, 0],   # FR_hip (X轴)
        joint_pos_euler[:, 6, 1],   # FR_thigh (Y轴)
        joint_pos_euler[:, 7, 1],   # FR_calf (Y轴)
        joint_pos_euler[:, 12, 0],  # RL_hip (X轴)
        joint_pos_euler[:, 13, 1],  # RL_thigh (Y轴)
        joint_pos_euler[:, 14, 1],  # RL_calf (Y轴)
        joint_pos_euler[:, 16, 0],  # RR_hip (X轴)
        joint_pos_euler[:, 17, 1],  # RR_thigh (Y轴)
        joint_pos_euler[:, 18, 1],  # RR_calf (Y轴)
    ]).T  # (num_frames, 12)
    
    # 提取脊柱关节角度 (3维)
    # 顺序: [yaw, pitch, roll]
    spine_pos = np.vstack([
        joint_pos_euler[:, 9, 2],   # yaw_spine (Z轴)
        joint_pos_euler[:, 10, 1],  # pitch_spine (Y轴)
        joint_pos_euler[:, 11, 0],  # roll_spine (X轴)
    ]).T  # (num_frames, 3)
    
    print(f"  腿部关节角度范围 (度):")
    leg_names = ['FL_hip', 'FL_thigh', 'FL_calf', 'FR_hip', 'FR_thigh', 'FR_calf',
                 'RL_hip', 'RL_thigh', 'RL_calf', 'RR_hip', 'RR_thigh', 'RR_calf']
    for i, name in enumerate(leg_names):
        print(f"    {name}: [{np.degrees(joint_pos[:, i].min()):.1f}, {np.degrees(joint_pos[:, i].max()):.1f}]")
    
    print(f"  脊柱关节角度范围 (度):")
    spine_names = ['yaw', 'pitch', 'roll']
    for i, name in enumerate(spine_names):
        print(f"    {name}: [{np.degrees(spine_pos[:, i].min()):.1f}, {np.degrees(spine_pos[:, i].max()):.1f}]")
    
    # 5. 提取足端位置
    print(f"\n[Step 5] 提取足端位置...")
    foot_pos_fl = global_trans[:, skeleton.node_names.index('FL_foot'), -3:].numpy()
    foot_pos_fr = global_trans[:, skeleton.node_names.index('FR_foot'), -3:].numpy()
    foot_pos_rl = global_trans[:, skeleton.node_names.index('RL_foot'), -3:].numpy()
    foot_pos_rr = global_trans[:, skeleton.node_names.index('RR_foot'), -3:].numpy()
    
    # 足端位置: (num_frames, 12) 顺序 [FL(3), FR(3), RL(3), RR(3)]
    foot_pos = np.hstack([foot_pos_fl, foot_pos_fr, foot_pos_rl, foot_pos_rr])
    
    print(f"  足端Z高度范围:")
    print(f"    FL: [{foot_pos_fl[:, 2].min():.3f}, {foot_pos_fl[:, 2].max():.3f}]")
    print(f"    FR: [{foot_pos_fr[:, 2].min():.3f}, {foot_pos_fr[:, 2].max():.3f}]")
    print(f"    RL: [{foot_pos_rl[:, 2].min():.3f}, {foot_pos_rl[:, 2].max():.3f}]")
    print(f"    RR: [{foot_pos_rr[:, 2].min():.3f}, {foot_pos_rr[:, 2].max():.3f}]")
    
    # 6. 计算速度
    print(f"\n[Step 6] 计算速度...")
    lin_vel = target_motion.global_root_velocity.numpy()  # (num_frames, 3)
    ang_vel = target_motion.global_root_angular_velocity.numpy()  # (num_frames, 3)
    joint_vel = np.array(differential(joint_pos, dt))  # (num_frames, 12)
    spine_vel = np.array(differential(spine_pos, dt))  # (num_frames, 3)
    foot_vel = np.array(differential(foot_pos, dt))    # (num_frames, 12)
    
    print(f"  线速度范围: [{np.linalg.norm(lin_vel, axis=1).min():.3f}, {np.linalg.norm(lin_vel, axis=1).max():.3f}] m/s")
    print(f"  角速度范围: [{np.linalg.norm(ang_vel, axis=1).min():.3f}, {np.linalg.norm(ang_vel, axis=1).max():.3f}] rad/s")
    
    # 7. 组装motion数据
    print(f"\n[Step 7] 组装motion数据...")
    # 数据格式 (67维):
    # root_pos(3) + root_rot(4) + joint_pos(12) + spine_pos(3) + foot_pos(12) +
    # lin_vel(3) + ang_vel(3) + joint_vel(12) + spine_vel(3) + foot_vel(12)
    motion_data = np.hstack([
        root_pos,     # [0:3]   3维
        root_rot,     # [3:7]   4维
        joint_pos,    # [7:19]  12维
        spine_pos,    # [19:22] 3维
        foot_pos,     # [22:34] 12维
        lin_vel,      # [34:37] 3维
        ang_vel,      # [37:40] 3维
        joint_vel,    # [40:52] 12维
        spine_vel,    # [52:55] 3维
        foot_vel,     # [55:67] 12维
    ])
    
    print(f"  motion_data shape: {motion_data.shape}")
    assert motion_data.shape[1] == 67, f"Expected 67 dimensions, got {motion_data.shape[1]}"
    
    # 8. 创建JSON结构
    print(f"\n[Step 8] 创建JSON结构...")
    mocap_data = {
        "LoopMode": "Wrap",
        "FrameDuration": dt,
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "MotionWeight": 0.5,
    }
    
    # 转换为列表格式
    motion_data_list = [list(d) for d in motion_data]
    mocap_data["Frames"] = motion_data_list
    
    # 9. 保存JSON文件
    print(f"\n[Step 9] 保存JSON文件: {output_json_path}")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(mocap_data, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print("导出完成!")
    print(f"  输入: {input_npy_path}")
    print(f"  输出: {output_json_path}")
    print(f"  帧数: {num_frames}")
    print(f"  数据维度: {motion_data.shape[1]}")
    print("=" * 60)
    
    return output_json_path


# ==================== 主函数 ====================

if __name__ == "__main__":
    # 配置文件路径
    json_dir = os.path.join(script_dir, "data/load_config.json")
    
    # 读取配置
    with open(json_dir, "r") as f:
        motion_json = json.load(f)
        file_name = motion_json["file_name"]
        clip = motion_json["clip"]
        remarks = motion_json["remarks"]
    
    print(f"配置信息:")
    print(f"  file_name: {file_name}")
    print(f"  clip: {clip}")
    print(f"  remarks: {remarks}")
    
    # 输入输出路径 - Himmy Mark2
    root_dir = os.path.join(script_dir, f"data/amp_himmy_mark2/{file_name}/")
    npy_file_name = f"{file_name}_amp_{clip[0]}_{clip[1]}_{remarks}"
    input_file = os.path.join(root_dir, f"{npy_file_name}.npy")
    output_file = os.path.join(root_dir, f"{npy_file_name}.json")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"\n错误: 输入文件不存在: {input_file}")
        print("请先运行 himmy_key_points_retargetting.py 生成 npy 文件")
        sys.exit(1)
    
    # 导出JSON
    export_himmy_json(input_file, output_file, visualize=True)
