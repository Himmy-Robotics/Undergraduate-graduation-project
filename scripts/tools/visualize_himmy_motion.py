#!/usr/bin/env python3
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Himmy Mark2 AMP动作数据可视化脚本

使用正向运动学从关节角度计算各连杆位置，用matplotlib 3D动画可视化。

数据格式 (67维):
- [0:3] root_pos
- [3:7] root_rot (quaternion, [x,y,z,w] PyBullet格式)
- [7:19] joint_pos (12 腿部关节: FL, FR, RL, RR 各3个)
- [19:22] spine_pos (3 脊柱关节: yaw, pitch, roll)
- [22:34] foot_pos (12 = 4腿 × 3坐标)
- [34:37] lin_vel
- [37:40] ang_vel
- [40:52] joint_vel (12 腿部关节)
- [52:55] spine_vel (3 脊柱关节)
- [55:67] foot_vel (12)

Himmy Mark2 骨骼结构:
- trunk (躯干/base_link)
- 前腿: FL_hip -> FL_thigh -> FL_calf -> FL_foot
        FR_hip -> FR_thigh -> FR_calf -> FR_foot
- 脊柱: yaw_spine -> pitch_spine -> roll_spine -> R_body
- 后腿: RL_hip -> RL_thigh -> RL_calf -> RL_foot
        RR_hip -> RR_thigh -> RR_calf -> RR_foot

Usage:
    python scripts/tools/visualize_himmy_motion.py --motion_file trot_forward1.txt
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

# ==============================================================================
# Himmy Mark2 运动数据目录
# ==============================================================================
MOTION_DIR = "/data/zmli/Fast-Quadruped/source/robot_lab/robot_lab/tasks/direct/himmy_amp/motions/datasets/mocap_motions_himmy"

# ==============================================================================
# Himmy Mark2 机器人连杆尺寸 (基于URDF测量，单位：米)
# ==============================================================================

# 躯干尺寸
TRUNK_TO_FRONT_HIP_X = 0.0432      # trunk到前腿hip的X距离
TRUNK_TO_SPINE_X = -0.0386         # trunk到yaw_spine的X偏移

# 脊柱连杆长度
SPINE_YAW_LENGTH = 0.1200          # yaw_spine到pitch_spine
SPINE_PITCH_LENGTH = 0.1053        # pitch_spine到roll_spine  
SPINE_ROLL_LENGTH = 0.0223         # roll_spine到R_body

# 后躯体到后腿hip的距离
RBODY_TO_REAR_HIP_X = 0.0655       # R_body到后腿hip的X距离

# 腿部连杆长度
HIP_OFFSET_Y = 0.1                 # hip左右方向偏移
HIP_TO_THIGH_X = 0.0833            # hip到thigh的X偏移
HIP_TO_THIGH_Y = 0.0795            # hip到thigh的Y偏移
THIGH_LENGTH = 0.23                # 大腿长度
CALF_LENGTH = 0.28                 # 小腿长度

# ==============================================================================
# 数据格式索引
# ==============================================================================
ROOT_POS_START = 0
ROOT_POS_END = 3
ROOT_ROT_START = 3
ROOT_ROT_END = 7
JOINT_POS_START = 7
JOINT_POS_END = 19
SPINE_POS_START = 19
SPINE_POS_END = 22


def load_motion_file(filepath):
    """加载运动文件（JSON格式）"""
    with open(filepath, 'r') as f:
        motion_data = json.load(f)
    
    frames = motion_data.get("Frames", [])
    frame_duration = motion_data.get("FrameDuration", 0.0333)
    return np.array(frames), frame_duration


def rotation_matrix_x(angle):
    """绕X轴旋转矩阵"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_matrix_y(angle):
    """绕Y轴旋转矩阵"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_z(angle):
    """绕Z轴旋转矩阵"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def forward_kinematics_himmy(root_pos, root_quat, joint_angles, spine_angles):
    """
    Himmy Mark2 正向运动学
    
    从关节角度计算各连杆末端位置。
    
    关节角度顺序:
    - joint_angles[0:3]: FL (hip, thigh, calf)
    - joint_angles[3:6]: FR (hip, thigh, calf)
    - joint_angles[6:9]: RL (hip, thigh, calf)
    - joint_angles[9:12]: RR (hip, thigh, calf)
    
    - spine_angles[0]: yaw_spine (绕Z轴)
    - spine_angles[1]: pitch_spine (绕Y轴)
    - spine_angles[2]: roll_spine (绕X轴)
    
    Args:
        root_pos: 根位置 [x, y, z]
        root_quat: 根四元数 [x, y, z, w] (PyBullet格式)
        joint_angles: 12个腿部关节角度
        spine_angles: 3个脊柱关节角度
    
    Returns:
        positions: 字典，包含各关节位置
    """
    # 根旋转矩阵
    root_rot = Rotation.from_quat(root_quat)
    R_root = root_rot.as_matrix()
    
    positions = {"root": root_pos.copy()}
    
    # ======================== 脊柱正向运动学 ========================
    # trunk -> yaw_spine
    yaw_spine_local = np.array([TRUNK_TO_SPINE_X, 0, 0])
    yaw_spine_world = root_pos + R_root @ yaw_spine_local
    positions["yaw_spine"] = yaw_spine_world
    
    # yaw_spine旋转 (绕Z轴)
    R_yaw = R_root @ rotation_matrix_z(spine_angles[0])
    
    # yaw_spine -> pitch_spine
    pitch_spine_local = np.array([-SPINE_YAW_LENGTH, 0, 0])
    pitch_spine_world = yaw_spine_world + R_yaw @ pitch_spine_local
    positions["pitch_spine"] = pitch_spine_world
    
    # pitch_spine旋转 (绕Y轴)
    R_pitch = R_yaw @ rotation_matrix_y(spine_angles[1])
    
    # pitch_spine -> roll_spine
    roll_spine_local = np.array([-SPINE_PITCH_LENGTH, 0, 0])
    roll_spine_world = pitch_spine_world + R_pitch @ roll_spine_local
    positions["roll_spine"] = roll_spine_world
    
    # roll_spine旋转 (绕X轴)
    R_roll = R_pitch @ rotation_matrix_x(spine_angles[2])
    
    # roll_spine -> R_body
    rbody_local = np.array([-SPINE_ROLL_LENGTH, 0, 0])
    rbody_world = roll_spine_world + R_roll @ rbody_local
    positions["R_body"] = rbody_world
    
    # ======================== 腿部正向运动学 ========================
    leg_configs = [
        ("FL", 0, root_pos, R_root, TRUNK_TO_FRONT_HIP_X, HIP_OFFSET_Y, True),
        ("FR", 3, root_pos, R_root, TRUNK_TO_FRONT_HIP_X, -HIP_OFFSET_Y, False),
        ("RL", 6, rbody_world, R_roll, -RBODY_TO_REAR_HIP_X, HIP_OFFSET_Y, True),
        ("RR", 9, rbody_world, R_roll, -RBODY_TO_REAR_HIP_X, -HIP_OFFSET_Y, False),
    ]
    
    for leg_name, idx, base_pos, base_rot, hip_x, hip_y, is_left in leg_configs:
        # 获取关节角度
        hip_angle = joint_angles[idx]       # 髋关节外展/内收 (绕X轴)
        thigh_angle = joint_angles[idx + 1] # 大腿前后摆动 (绕Y轴)
        calf_angle = joint_angles[idx + 2]  # 小腿前后摆动 (绕Y轴)
        
        # Hip位置
        hip_local = np.array([hip_x, hip_y, 0])
        hip_world = base_pos + base_rot @ hip_local
        positions[f"{leg_name}_hip"] = hip_world
        
        # Hip旋转 (绕X轴，外展/内收)
        R_hip = base_rot @ rotation_matrix_x(hip_angle)
        
        # Thigh起点（从hip偏移到thigh关节）
        # 注意：hip到thigh有X和Y方向的偏移
        thigh_offset_y = HIP_TO_THIGH_Y if is_left else -HIP_TO_THIGH_Y
        thigh_start_local = np.array([0, thigh_offset_y, 0])
        thigh_start_world = hip_world + R_hip @ thigh_start_local
        positions[f"{leg_name}_thigh_start"] = thigh_start_world
        
        # Thigh旋转 (绕Y轴，前后摆动)
        R_thigh = R_hip @ rotation_matrix_y(thigh_angle)
        
        # Thigh末端（膝关节）
        thigh_end_local = np.array([0, 0, -THIGH_LENGTH])
        thigh_end_world = thigh_start_world + R_thigh @ thigh_end_local
        positions[f"{leg_name}_knee"] = thigh_end_world
        
        # Calf旋转 (绕Y轴，前后摆动)
        R_calf = R_thigh @ rotation_matrix_y(calf_angle)
        
        # 足端位置
        foot_local = np.array([0, 0, -CALF_LENGTH])
        foot_world = thigh_end_world + R_calf @ foot_local
        positions[f"{leg_name}_foot"] = foot_world
    
    return positions


class HimmyVisualizer:
    """Himmy Mark2 机器人可视化器"""
    
    def __init__(self, motion_file, output_dir="logs/motion_vis"):
        self.motion_file = motion_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        if os.path.isabs(motion_file):
            filepath = motion_file
        else:
            filepath = os.path.join(MOTION_DIR, motion_file)
        
        self.frames, self.frame_duration = load_motion_file(filepath)
        self.motion_name = os.path.splitext(os.path.basename(motion_file))[0]
        
        print(f"加载动作: {self.motion_name}")
        print(f"帧数: {len(self.frames)}, 帧间隔: {self.frame_duration:.4f}s")
        print(f"总时长: {len(self.frames) * self.frame_duration:.2f}s")
        
    def extract_frame_data(self, frame_idx):
        """从帧数据中提取位置和关节角度"""
        frame = self.frames[frame_idx]
        
        root_pos = np.array(frame[ROOT_POS_START:ROOT_POS_END])
        root_quat = np.array(frame[ROOT_ROT_START:ROOT_ROT_END])
        joint_pos = np.array(frame[JOINT_POS_START:JOINT_POS_END])
        spine_pos = np.array(frame[SPINE_POS_START:SPINE_POS_END])
        
        return root_pos, root_quat, joint_pos, spine_pos
    
    def create_animation(self, save_video=True, show=False):
        """创建3D动画"""
        # 降低分辨率以提高渲染速度：1280x720 (720p) 而不是 1600x1200
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # 计算坐标轴范围
        root_positions = np.array([self.frames[i][0:3] for i in range(len(self.frames))])
        x_range = [root_positions[:, 0].min() - 0.3, root_positions[:, 0].max() + 0.3]
        y_range = [root_positions[:, 1].min() - 0.3, root_positions[:, 1].max() + 0.3]
        
        # 计算显示范围
        max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], 0.8)
        display_range = max(max_range, 0.8)
        
        # 颜色配置
        colors = {"FL": "red", "FR": "blue", "RL": "green", "RR": "orange"}
        
        # 初始化线条
        leg_lines = {}
        for leg in ["FL", "FR", "RL", "RR"]:
            leg_lines[leg], = ax.plot([], [], [], 'o-', color=colors[leg], 
                                       linewidth=3, markersize=6, label=leg)
        
        # 脊柱线
        spine_line, = ax.plot([], [], [], 'o-', color='purple', 
                              linewidth=4, markersize=7, label='Spine')
        
        # 躯干连接线
        body_lines = []
        for _ in range(6):
            line, = ax.plot([], [], [], 'k-', linewidth=2.5)
            body_lines.append(line)
        
        # 轨迹线
        trajectory_line, = ax.plot([], [], [], 'b--', alpha=0.4, linewidth=1)
        trajectory_data = []
        
        # 时间文本
        time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=11)
        joint_text = ax.text2D(0.02, 0.02, '', transform=ax.transAxes, fontsize=8, 
                               verticalalignment='bottom', family='monospace')
        
        def init():
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.set_zlabel('Z (m)', fontsize=10)
            ax.set_title(f'Himmy Mark2 AMP Motion: {self.motion_name}', 
                        fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            return list(leg_lines.values()) + [spine_line] + body_lines + [trajectory_line, time_text]
        
        def update(frame_idx):
            root_pos, root_quat, joint_pos, spine_pos = self.extract_frame_data(frame_idx)
            positions = forward_kinematics_himmy(root_pos, root_quat, joint_pos, spine_pos)
            
            # 更新脊柱线
            spine_pts = ["root", "yaw_spine", "pitch_spine", "roll_spine", "R_body"]
            spine_x = [positions[p][0] for p in spine_pts]
            spine_y = [positions[p][1] for p in spine_pts]
            spine_z = [positions[p][2] for p in spine_pts]
            spine_line.set_data(spine_x, spine_y)
            spine_line.set_3d_properties(spine_z)
            
            # 更新腿部线条
            for leg in ["FL", "FR", "RL", "RR"]:
                pts = [f"{leg}_hip", f"{leg}_thigh_start", f"{leg}_knee", f"{leg}_foot"]
                x = [positions[p][0] for p in pts]
                y = [positions[p][1] for p in pts]
                z = [positions[p][2] for p in pts]
                leg_lines[leg].set_data(x, y)
                leg_lines[leg].set_3d_properties(z)
            
            # 更新躯干连接线
            fl_hip = positions["FL_hip"]
            fr_hip = positions["FR_hip"]
            rl_hip = positions["RL_hip"]
            rr_hip = positions["RR_hip"]
            root = positions["root"]
            rbody = positions["R_body"]
            
            # 前横梁
            body_lines[0].set_data([fl_hip[0], fr_hip[0]], [fl_hip[1], fr_hip[1]])
            body_lines[0].set_3d_properties([fl_hip[2], fr_hip[2]])
            # 后横梁
            body_lines[1].set_data([rl_hip[0], rr_hip[0]], [rl_hip[1], rr_hip[1]])
            body_lines[1].set_3d_properties([rl_hip[2], rr_hip[2]])
            # FL到root
            body_lines[2].set_data([fl_hip[0], root[0]], [fl_hip[1], root[1]])
            body_lines[2].set_3d_properties([fl_hip[2], root[2]])
            # FR到root
            body_lines[3].set_data([fr_hip[0], root[0]], [fr_hip[1], root[1]])
            body_lines[3].set_3d_properties([fr_hip[2], root[2]])
            # RL到R_body
            body_lines[4].set_data([rl_hip[0], rbody[0]], [rl_hip[1], rbody[1]])
            body_lines[4].set_3d_properties([rl_hip[2], rbody[2]])
            # RR到R_body
            body_lines[5].set_data([rr_hip[0], rbody[0]], [rr_hip[1], rbody[1]])
            body_lines[5].set_3d_properties([rr_hip[2], rbody[2]])
            
            # 更新轨迹
            trajectory_data.append(root_pos.copy())
            if len(trajectory_data) > 1:
                traj = np.array(trajectory_data)
                trajectory_line.set_data(traj[:, 0], traj[:, 1])
                trajectory_line.set_3d_properties(traj[:, 2])
            
            # 更新时间文本
            current_time = frame_idx * self.frame_duration
            time_text.set_text(f'Time: {current_time:.3f}s  Frame: {frame_idx}/{len(self.frames)-1}')
            
            # 显示关节角度
            joint_str = f"Spine: yaw={np.degrees(spine_pos[0]):.1f}° pitch={np.degrees(spine_pos[1]):.1f}° roll={np.degrees(spine_pos[2]):.1f}°\n"
            joint_str += f"FL: hip={np.degrees(joint_pos[0]):.1f}° thigh={np.degrees(joint_pos[1]):.1f}° calf={np.degrees(joint_pos[2]):.1f}°\n"
            joint_str += f"FR: hip={np.degrees(joint_pos[3]):.1f}° thigh={np.degrees(joint_pos[4]):.1f}° calf={np.degrees(joint_pos[5]):.1f}°"
            joint_text.set_text(joint_str)
            
            # 动态更新坐标轴范围（跟随机器人）
            center_x, center_y = root_pos[0], root_pos[1]
            half_range = display_range * 0.5
            ax.set_xlim([center_x - half_range, center_x + half_range])
            ax.set_ylim([center_y - half_range, center_y + half_range])
            ax.set_zlim([-0.1, display_range - 0.1])
            
            # 设置视角
            ax.view_init(elev=25, azim=-60)
            ax.dist = 8
            
            return list(leg_lines.values()) + [spine_line] + body_lines + [trajectory_line, time_text, joint_text]
        
        # 创建动画
        fps = int(1.0 / self.frame_duration)
        anim = FuncAnimation(fig, update, frames=len(self.frames),
                           init_func=init, blit=False, interval=self.frame_duration * 1000)
        
        if save_video:
            video_path = os.path.join(self.output_dir, f"{self.motion_name}_himmy.mp4")
            print(f"保存视频到: {video_path}")
            
            try:
                # 提高bitrate加速编码：从2400提升到8000 kbps
                writer = FFMpegWriter(fps=fps, metadata=dict(artist='Himmy Visualizer'), 
                                     bitrate=8000, codec='libx264', 
                                     extra_args=['-preset', 'fast'])
                anim.save(video_path, writer=writer)
                print(f"✓ 视频保存成功: {video_path}")
            except Exception as e:
                print(f"FFmpeg保存失败: {e}")
                gif_path = os.path.join(self.output_dir, f"{self.motion_name}_himmy.gif")
                print(f"尝试保存为GIF: {gif_path}")
                try:
                    anim.save(gif_path, writer='pillow', fps=min(fps, 20))
                    print(f"✓ GIF保存成功: {gif_path}")
                except Exception as e2:
                    print(f"GIF保存也失败: {e2}")
        
        if show:
            plt.show()
        
        plt.close()
        return anim


def main():
    parser = argparse.ArgumentParser(description="可视化Himmy Mark2 AMP动作数据")
    parser.add_argument("--motion_file", type=str, default="trot_forward1.txt",
                        help="动作文件名")
    parser.add_argument("--output_dir", type=str, default="/data/zmli/Fast-Quadruped/logs/motion_vis",
                        help="输出目录")
    parser.add_argument("--show", action="store_true", help="显示动画窗口")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Himmy Mark2 AMP Motion Visualizer")
    print("="*60)
    print("使用正向运动学从关节角度计算连杆位置")
    print("="*60)
    
    visualizer = HimmyVisualizer(args.motion_file, args.output_dir)
    visualizer.create_animation(save_video=True, show=args.show)
    
    print("\n" + "="*60)
    print("完成！")
    print(f"输出目录: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
