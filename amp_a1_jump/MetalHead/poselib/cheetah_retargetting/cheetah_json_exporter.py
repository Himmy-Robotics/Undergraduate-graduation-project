#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: cheetah_json_exporter.py
@Auth: Based on json_exporter.py, modified for Cheetah retargeted data
@Date: 2024/12

将重定向后的猎豹运动数据（.npy）转换为AMP训练所需的JSON格式。

Himmy Mark2骨架结构（20个节点）：
['trunk', 'FL_hip_Link', 'FL_thigh_Link', 'FL_calf_Link', 'FL_foot',
 'FR_hip_Link', 'FR_thigh_Link', 'FR_calf_Link', 'FR_foot',
 'yaw_spine_Link', 'pitch_spine_Link', 'roll_spine_Link',
 'RL_hip_Link', 'RL_thigh_Link', 'RL_calf_Link', 'RL_foot',
 'RR_hip_Link', 'RR_thigh_Link', 'RR_calf_Link', 'RR_foot']
"""
import os
import sys
import json
import glob
import numpy as np

# 添加路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSELIB_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, POSELIB_DIR)

from poselib.skeleton.skeleton3d import SkeletonMotion
from cheetah_kinematics import HimmyKinematics

# 输入输出目录
INPUT_DIR = os.path.join(POSELIB_DIR, "data", "cheetah_data_retarget_npy")
OUTPUT_DIR = os.path.join(POSELIB_DIR, "data", "cheetah_data_json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Cheetah Retargeted Motion -> JSON Exporter")
print("=" * 60)
print(f"输入目录: {INPUT_DIR}")
print(f"输出目录: {OUTPUT_DIR}")


def differential(d, dt=1/60):
    """计算差分（速度）"""
    diff_data = []
    diff = np.zeros(d.shape[1])
    for i in range(1, d.shape[0]):
        diff = (d[i, :] - d[i-1, :]) / dt
        diff_data.append(diff)
    diff_data.append(diff)  # 最后一帧使用前一帧的速度
    return np.array(diff_data)


def quaternion_to_axis_angle(q):
    """四元数转轴角 [x, y, z, w] -> [ax, ay, az, angle]"""
    x, y, z, w = q
    angle = 2 * np.arccos(np.clip(w, -1, 1))
    s = np.sqrt(1 - w*w)
    if s < 1e-6:
        return np.array([1, 0, 0, 0])
    return np.array([x/s, y/s, z/s, angle])


def export_motion_to_json(npy_path, output_path):
    """将单个npy文件转换为JSON格式"""
    
    # 加载运动数据
    target_motion = SkeletonMotion.from_file(npy_path)
    skeleton = target_motion.skeleton_tree
    
    dt = 1 / target_motion.fps
    n_frames = target_motion.local_rotation.shape[0]
    
    # 获取全局变换
    global_trans = target_motion.global_transformation  # (len, 20, 7)
    
    # Root位置和旋转
    root_pos = global_trans[:, skeleton.node_names.index('trunk'), -3:].numpy()
    root_rot = global_trans[:, skeleton.node_names.index('trunk'), :-3].numpy()
    
    # 局部旋转 (len, 20, 4) - [x, y, z, w]格式
    local_rot = target_motion.local_rotation.numpy()
    
    # 提取关节角度 - Himmy Mark2骨架
    # 顺序: FR(3), FL(3), RR(3), RL(3) + spine(3) = 15个关节
    joint_pos = np.zeros((n_frames, 15))
    
    for i in range(n_frames):
        # FR: indices 5,6,7 - hip(roll), thigh(pitch), calf(pitch)
        fr_hip = quaternion_to_axis_angle(local_rot[i, 5])
        fr_thigh = quaternion_to_axis_angle(local_rot[i, 6])
        fr_calf = quaternion_to_axis_angle(local_rot[i, 7])
        
        # FL: indices 1,2,3
        fl_hip = quaternion_to_axis_angle(local_rot[i, 1])
        fl_thigh = quaternion_to_axis_angle(local_rot[i, 2])
        fl_calf = quaternion_to_axis_angle(local_rot[i, 3])
        
        # RR: indices 16,17,18
        rr_hip = quaternion_to_axis_angle(local_rot[i, 16])
        rr_thigh = quaternion_to_axis_angle(local_rot[i, 17])
        rr_calf = quaternion_to_axis_angle(local_rot[i, 18])
        
        # RL: indices 12,13,14
        rl_hip = quaternion_to_axis_angle(local_rot[i, 12])
        rl_thigh = quaternion_to_axis_angle(local_rot[i, 13])
        rl_calf = quaternion_to_axis_angle(local_rot[i, 14])
        
        # Spine: indices 9,10,11 - yaw(z), pitch(y), roll(x)
        yaw_spine = quaternion_to_axis_angle(local_rot[i, 9])
        pitch_spine = quaternion_to_axis_angle(local_rot[i, 10])
        roll_spine = quaternion_to_axis_angle(local_rot[i, 11])
        
        # 提取单轴角度
        joint_pos[i] = [
            fr_hip[0] * fr_hip[3],      # FR hip (roll around X)
            fr_thigh[1] * fr_thigh[3],  # FR thigh (pitch around Y)
            fr_calf[1] * fr_calf[3],    # FR calf (pitch around Y)
            fl_hip[0] * fl_hip[3],      # FL hip
            fl_thigh[1] * fl_thigh[3],  # FL thigh
            fl_calf[1] * fl_calf[3],    # FL calf
            rr_hip[0] * rr_hip[3],      # RR hip
            rr_thigh[1] * rr_thigh[3],  # RR thigh
            rr_calf[1] * rr_calf[3],    # RR calf
            rl_hip[0] * rl_hip[3],      # RL hip
            rl_thigh[1] * rl_thigh[3],  # RL thigh
            rl_calf[1] * rl_calf[3],    # RL calf
            yaw_spine[2] * yaw_spine[3],    # yaw_spine (around Z)
            pitch_spine[1] * pitch_spine[3], # pitch_spine (around Y)
            roll_spine[0] * roll_spine[3],   # roll_spine (around X)
        ]
    
    # 足端位置
    foot_fr = global_trans[:, skeleton.node_names.index('FR_foot'), -3:].numpy()
    foot_fl = global_trans[:, skeleton.node_names.index('FL_foot'), -3:].numpy()
    foot_rr = global_trans[:, skeleton.node_names.index('RR_foot'), -3:].numpy()
    foot_rl = global_trans[:, skeleton.node_names.index('RL_foot'), -3:].numpy()
    foot_pos = np.hstack([foot_fr, foot_fl, foot_rr, foot_rl])  # (len, 12)
    
    # 速度
    lin_vel = target_motion.global_root_velocity.numpy()
    ang_vel = target_motion.global_root_angular_velocity.numpy()
    joint_vel = differential(joint_pos, dt)
    foot_vel = differential(foot_pos, dt)
    
    # 组合数据 (len, 64) - 比原来多3个脊柱关节
    # root_pos(3) + root_rot(4) + joint_pos(15) + foot_pos(12) + 
    # lin_vel(3) + ang_vel(3) + joint_vel(15) + foot_vel(12) = 67
    motion_data = np.hstack([root_pos, root_rot, joint_pos, foot_pos, 
                             lin_vel, ang_vel, joint_vel, foot_vel])
    
    # 创建JSON结构
    mocap_data = {
        "LoopMode": "Wrap",
        "FrameDuration": dt,
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "MotionWeight": 0.5,
        "Frames": [list(d) for d in motion_data]
    }
    
    # 保存
    with open(output_path, 'w') as f:
        json.dump(mocap_data, f, indent=2)
    
    return n_frames, dt


def main():
    # 查找所有npy文件
    npy_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npy")))
    
    if not npy_files:
        print("未找到npy文件！")
        return
    
    print(f"\n找到 {len(npy_files)} 个文件:")
    
    success_count = 0
    for npy_path in npy_files:
        filename = os.path.basename(npy_path)
        json_name = filename.replace('.npy', '.json')
        output_path = os.path.join(OUTPUT_DIR, json_name)
        
        try:
            n_frames, dt = export_motion_to_json(npy_path, output_path)
            print(f"  ✓ {filename} -> {json_name} ({n_frames}帧, {n_frames*dt:.2f}s)")
            success_count += 1
        except Exception as e:
            print(f"  ✗ {filename}: {str(e)}")
    
    print(f"\n完成! 成功导出 {success_count}/{len(npy_files)} 个文件")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
