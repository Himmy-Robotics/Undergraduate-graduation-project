#!/usr/bin/env python3
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
简单的AMP动作数据可视化脚本（不需要Isaac Sim）

使用matplotlib 3D动画来可视化AMP运动捕捉数据，并保存为视频。

重要发现（2024-12-16）：
- AMP数据集使用 PyBullet 格式的四元数: [x, y, z, w] (w在最后)
- scipy.spatial.transform.Rotation 也使用 [x, y, z, w] 格式
- 因此不需要任何四元数格式转换！

Usage:
    python scripts/tools/visualize_amp_motion_simple.py
    python scripts/tools/visualize_amp_motion_simple.py --motion_file pace0.txt
    python scripts/tools/visualize_amp_motion_simple.py --all --output_dir logs/motion_vis
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

# AMP数据目录
MOTION_DIR = "/data/zmli/Fast-Quadruped/source/robot_lab/robot_lab/tasks/direct/himmy_amp/motions/datasets/mocap_motions"

# Himmy Mark2 机器人连杆长度 (米) - 基于URDF测量
THIGH_LENGTH = 0.23  # 大腿长度 (从thigh关节到calf关节)
CALF_LENGTH = 0.23   # 小腿长度 (从calf关节到foot)

# 髋关节偏移 (相对于机体中心)
HIP_OFFSET_Y = 0.1    # 左右方向偏移


def load_motion_file(filepath):
    """加载运动文件（JSON格式）"""
    with open(filepath, 'r') as f:
        motion_data = json.load(f)
    
    frames = motion_data.get("Frames", [])
    frame_duration = motion_data.get("FrameDuration", 0.021)
    return np.array(frames), frame_duration


def forward_kinematics(root_pos, root_quat, joint_angles, spine_angles):
    """
    计算正向运动学，返回各关节和足端位置
    
    Himmy Mark2机器人结构:
    - trunk (躯干) 作为root
    - 前腿 (FL, FR) 直接连接到trunk前部
    - 脊柱 (yaw_spine -> pitch_spine -> roll_spine) 从trunk向后延伸
    - 后腿 (RL, RR) 连接到roll_spine末端
    
    Args:
        root_pos: 根位置 [x, y, z]
        root_quat: 根四元数 [x, y, z, w] (PyBullet/AMP格式，w在最后)
        joint_angles: 12个腿部关节角度 [FL(3), FR(3), RL(3), RR(3)]
        spine_angles: 3个脊柱关节角度 [yaw, pitch, roll]
    
    Returns:
        positions: 字典，包含各关节位置（包括脊柱节点）
    """
    # AMP数据使用 PyBullet 格式 [x, y, z, w]，scipy 也使用 [x, y, z, w]
    root_rot = Rotation.from_quat(root_quat)
    
    positions = {"root": root_pos.copy()}
    
    # === 脊柱计算 ===
    # Mark2的脊柱从trunk向后延伸到后腿连接点
    spine_yaw, spine_pitch, spine_roll = spine_angles
    
    # 脊柱连杆长度 (从URDF)
    YAW_TO_PITCH = 0.12
    PITCH_TO_ROLL = 0.10528
    ROLL_TO_REAR = 0.02232
    
    # yaw_spine: 从trunk向后偏移0.043m，绕Z轴旋转
    yaw_spine_local = np.array([-0.043, 0, 0])
    yaw_spine_world = root_pos + root_rot.apply(yaw_spine_local)
    positions["yaw_spine"] = yaw_spine_world
    
    # pitch_spine: 从yaw_spine继续向后
    yaw_rot = root_rot * Rotation.from_euler('z', spine_yaw)
    pitch_spine_local = np.array([-YAW_TO_PITCH, 0, 0])
    pitch_spine_world = yaw_spine_world + yaw_rot.apply(pitch_spine_local)
    positions["pitch_spine"] = pitch_spine_world
    
    # roll_spine: 从pitch_spine继续向后
    pitch_rot = yaw_rot * Rotation.from_euler('y', spine_pitch)
    roll_spine_local = np.array([-PITCH_TO_ROLL, 0, 0])
    roll_spine_world = pitch_spine_world + pitch_rot.apply(roll_spine_local)
    positions["roll_spine"] = roll_spine_world
    
    # 后躯连接点
    roll_rot = pitch_rot * Rotation.from_euler('x', spine_roll)
    rear_body_local = np.array([-ROLL_TO_REAR, 0, 0])
    rear_body_world = roll_spine_world + roll_rot.apply(rear_body_local)
    positions["rear_body"] = rear_body_world
    
    # === 腿部计算 ===
    # 前腿连接到trunk前部
    FRONT_HIP_X = 0.043  # trunk到前腿髋关节的X偏移
    
    # 后腿连接到rear_body
    leg_configs = {
        "FL": {"hip_offset": np.array([FRONT_HIP_X, HIP_OFFSET_Y, 0]), "base_pos": root_pos, "base_rot": root_rot},
        "FR": {"hip_offset": np.array([FRONT_HIP_X, -HIP_OFFSET_Y, 0]), "base_pos": root_pos, "base_rot": root_rot},
        "RL": {"hip_offset": np.array([0, HIP_OFFSET_Y, 0]), "base_pos": rear_body_world, "base_rot": roll_rot},
        "RR": {"hip_offset": np.array([0, -HIP_OFFSET_Y, 0]), "base_pos": rear_body_world, "base_rot": roll_rot},
    }
    
    leg_names = ["FL", "FR", "RL", "RR"]
    
    for i, leg in enumerate(leg_names):
        config = leg_configs[leg]
        base_pos = config["base_pos"]
        base_rot = config["base_rot"]
        hip_offset = config["hip_offset"]
        
        # 髋关节位置
        hip_world = base_pos + base_rot.apply(hip_offset)
        positions[f"{leg}_hip"] = hip_world
        
        # 获取关节角度
        hip_angle = joint_angles[i * 3]      # 髋关节外展/内收
        thigh_angle = joint_angles[i * 3 + 1]  # 大腿前后摆动
        calf_angle = joint_angles[i * 3 + 2]   # 小腿前后摆动
        
        # 大腿方向和末端位置
        # 髋关节绕X轴旋转（外展/内收），大腿关节绕Y轴旋转（前后摆动）
        leg_rot = base_rot * Rotation.from_euler('xy', [hip_angle, thigh_angle])
        thigh_dir = leg_rot.apply(np.array([0, 0, -THIGH_LENGTH]))
        thigh_end = hip_world + thigh_dir
        positions[f"{leg}_thigh"] = thigh_end
        
        # 小腿方向和足端位置
        # 小腿关节继续绕Y轴旋转
        leg_rot_calf = base_rot * Rotation.from_euler('xy', [hip_angle, thigh_angle + calf_angle])
        calf_dir = leg_rot_calf.apply(np.array([0, 0, -CALF_LENGTH]))
        foot_pos = thigh_end + calf_dir
        positions[f"{leg}_foot"] = foot_pos
    
    return positions


class QuadrupedVisualizer:
    """四足机器人可视化器"""
    
    def __init__(self, motion_file, output_dir="logs/motion_vis"):
        self.motion_file = motion_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        filepath = os.path.join(MOTION_DIR, motion_file) if not os.path.isabs(motion_file) else motion_file
        self.frames, self.frame_duration = load_motion_file(filepath)
        self.motion_name = os.path.splitext(os.path.basename(motion_file))[0]
        
        print(f"加载动作: {self.motion_name}")
        print(f"帧数: {len(self.frames)}, 帧间隔: {self.frame_duration}s")
        
    def extract_frame_data(self, frame_idx):
        """从帧数据中提取位置和关节角度"""
        frame = self.frames[frame_idx]
        
        # 数据格式 (67维):
        # [0:3]   root_pos (3)
        # [3:7]   root_rot quaternion (4) - 格式是 [x, y, z, w]
        # [7:19]  joint_pos (12) - 腿部关节
        # [19:22] spine_pos (3) - 脊柱关节 [yaw, pitch, roll]
        
        root_pos = np.array(frame[0:3])
        root_quat = np.array(frame[3:7])
        joint_pos = np.array(frame[7:19])  # 腿部12个关节
        spine_pos = np.array(frame[19:22])  # 脊柱3个关节
        
        return root_pos, root_quat, joint_pos, spine_pos
    
    def create_animation(self, save_video=True, show=False):
        """创建3D动画"""
        # 增大窗口尺寸以显示更多细节
        fig = plt.figure(figsize=(16, 12), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置坐标轴范围
        root_positions = np.array([self.frames[i][0:3] for i in range(len(self.frames))])
        # 计算机器人运动范围
        x_range = [root_positions[:, 0].min() - 0.1, root_positions[:, 0].max() + 0.1]
        y_range = [root_positions[:, 1].min() - 0.1, root_positions[:, 1].max() + 0.1]
        z_range = [-0.05, 0.6]  # 包括地面
        
        # 确保坐标轴比例一致（1:1:1）
        max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])
        # 使用更小的显示范围以放大机器人（0.6m的近距离视距）
        display_range = max(max_range * 1.5, 0.6)  # 更小的显示范围，近距离视距
        x_mid = (x_range[0] + x_range[1]) * 0.5
        y_mid = (y_range[0] + y_range[1]) * 0.5
        z_mid = 0.3  # 固定Z中点在0.3m（腰部高度）
        
        x_range = [x_mid - display_range * 0.5, x_mid + display_range * 0.5]
        y_range = [y_mid - display_range * 0.5, y_mid + display_range * 0.5]
        z_range = [z_mid - display_range * 0.5, z_mid + display_range * 0.5]
        
        # 颜色配置
        colors = {
            "FL": "red",
            "FR": "blue", 
            "RL": "green",
            "RR": "orange"
        }
        
        # 初始化线条
        lines = {}
        leg_names = ["FL", "FR", "RL", "RR"]
        for leg in leg_names:
            # 增加线条粗细和标记点大小
            lines[leg], = ax.plot([], [], [], 'o-', color=colors[leg], linewidth=3, markersize=7, label=leg)
        
        # 脊柱线（更粗更醒目，紫色）
        spine_line, = ax.plot([], [], [], 'o-', color='purple', linewidth=4, markersize=8, label='Spine')
        
        # 躯干框架线（6条线组成矩形框架+前后连接）
        body_lines = []
        for _ in range(8):  # 增加到8条线
            line, = ax.plot([], [], [], 'k-', linewidth=2.5)
            body_lines.append(line)
        
        # 地面
        ground_x, ground_y = np.meshgrid(np.linspace(x_range[0], x_range[1], 20),
                                          np.linspace(y_range[0], y_range[1], 20))
        ground_z = np.zeros_like(ground_x)
        ax.plot_surface(ground_x, ground_y, ground_z, alpha=0.15, color='gray')
        
        # 轨迹线
        trajectory_line, = ax.plot([], [], [], 'b--', alpha=0.5, linewidth=1)
        trajectory_data = []
        
        # 时间文本
        time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=12)
        
        def init():
            # 初始范围设定
            initial_x_mid = (x_range[0] + x_range[1]) * 0.5
            initial_y_mid = (y_range[0] + y_range[1]) * 0.5
            initial_z_mid = 0.3
            half_range = display_range * 0.5
            
            ax.set_xlim([initial_x_mid - half_range, initial_x_mid + half_range])
            ax.set_ylim([initial_y_mid - half_range, initial_y_mid + half_range])
            ax.set_zlim([initial_z_mid - half_range, initial_z_mid + half_range])
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.set_zlabel('Z (m)', fontsize=10)
            ax.set_title(f'AMP Motion: {self.motion_name}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            return list(lines.values()) + [spine_line] + body_lines + [trajectory_line, time_text]
        
        def update(frame_idx):
            root_pos, root_quat, joint_pos, spine_pos = self.extract_frame_data(frame_idx)
            positions = forward_kinematics(root_pos, root_quat, joint_pos, spine_pos)
            
            # 更新脊柱线条 (从trunk向后: root -> yaw -> pitch -> roll -> rear_body)
            spine_x = [positions["root"][0], positions["yaw_spine"][0], positions["pitch_spine"][0], 
                       positions["roll_spine"][0], positions["rear_body"][0]]
            spine_y = [positions["root"][1], positions["yaw_spine"][1], positions["pitch_spine"][1],
                       positions["roll_spine"][1], positions["rear_body"][1]]
            spine_z = [positions["root"][2], positions["yaw_spine"][2], positions["pitch_spine"][2],
                       positions["roll_spine"][2], positions["rear_body"][2]]
            spine_line.set_data(spine_x, spine_y)
            spine_line.set_3d_properties(spine_z)
            
            # 更新腿部线条
            for leg in leg_names:
                hip = positions[f"{leg}_hip"]
                thigh = positions[f"{leg}_thigh"]
                foot = positions[f"{leg}_foot"]
                
                x = [hip[0], thigh[0], foot[0]]
                y = [hip[1], thigh[1], foot[1]]
                z = [hip[2], thigh[2], foot[2]]
                lines[leg].set_data(x, y)
                lines[leg].set_3d_properties(z)
            
            # 更新躯干框架
            # 前腿髋关节连接到trunk（root位置附近）
            fl_hip = positions["FL_hip"]
            fr_hip = positions["FR_hip"]
            rl_hip = positions["RL_hip"]
            rr_hip = positions["RR_hip"]
            root = positions["root"]
            rear = positions["rear_body"]
            
            # 前部横梁: FL_hip - FR_hip
            body_lines[0].set_data([fl_hip[0], fr_hip[0]], [fl_hip[1], fr_hip[1]])
            body_lines[0].set_3d_properties([fl_hip[2], fr_hip[2]])
            
            # 后部横梁: RL_hip - RR_hip
            body_lines[1].set_data([rl_hip[0], rr_hip[0]], [rl_hip[1], rr_hip[1]])
            body_lines[1].set_3d_properties([rl_hip[2], rr_hip[2]])
            
            # 前左到root: FL_hip - root
            body_lines[2].set_data([fl_hip[0], root[0]], [fl_hip[1], root[1]])
            body_lines[2].set_3d_properties([fl_hip[2], root[2]])
            
            # 前右到root: FR_hip - root
            body_lines[3].set_data([fr_hip[0], root[0]], [fr_hip[1], root[1]])
            body_lines[3].set_3d_properties([fr_hip[2], root[2]])
            
            # 后左到rear: RL_hip - rear_body
            body_lines[4].set_data([rl_hip[0], rear[0]], [rl_hip[1], rear[1]])
            body_lines[4].set_3d_properties([rl_hip[2], rear[2]])
            
            # 后右到rear: RR_hip - rear_body
            body_lines[5].set_data([rr_hip[0], rear[0]], [rr_hip[1], rear[1]])
            body_lines[5].set_3d_properties([rr_hip[2], rear[2]])
            
            # 额外的支撑线（可选）
            body_lines[6].set_data([], [])
            body_lines[6].set_3d_properties([])
            body_lines[7].set_data([], [])
            body_lines[7].set_3d_properties([])
            
            # 更新轨迹
            trajectory_data.append(root_pos.copy())
            if len(trajectory_data) > 1:
                traj = np.array(trajectory_data)
                trajectory_line.set_data(traj[:, 0], traj[:, 1])
                trajectory_line.set_3d_properties(traj[:, 2])
            
            # 更新时间文本
            current_time = frame_idx * self.frame_duration
            time_text.set_text(f'Time: {current_time:.3f}s  Frame: {frame_idx}/{len(self.frames)-1}')
            
            # 动态更新坐标轴范围（摄像机跟随机器人）
            center_x = root_pos[0]
            center_y = root_pos[1]
            center_z = 0.3  # 固定Z中点在腰部高度
            half_range = display_range * 0.5
            
            ax.set_xlim([center_x - half_range, center_x + half_range])
            ax.set_ylim([center_y - half_range, center_y + half_range])
            ax.set_zlim([center_z - half_range, center_z + half_range])
            
            # 改进视角：俯视角加透视，更好地展示脊柱
            # 随时间改变视角，展示不同角度
            frame_progress = frame_idx / len(self.frames)
            azimuth = -90 + 60 * np.sin(frame_progress * 2 * np.pi)  # 左右旋转
            elevation = 25  # 轻微俯视角
            # 使用dist参数缩短摄像机距离，使机器人更大
            ax.view_init(elev=elevation, azim=azimuth)
            # 缩短摄像机距离使机器人显得更大
            ax.dist = 8  # 减小距离使图像放大
            
            return list(lines.values()) + [spine_line] + body_lines + [trajectory_line, time_text]
        
        # 创建动画
        fps = int(1.0 / self.frame_duration)
        anim = FuncAnimation(fig, update, frames=len(self.frames),
                           init_func=init, blit=False, interval=self.frame_duration * 1000)
        
        if save_video:
            video_path = os.path.join(self.output_dir, f"{self.motion_name}_animation.mp4")
            print(f"保存视频到: {video_path}")
            
            try:
                writer = FFMpegWriter(fps=fps, metadata=dict(artist='AMP Visualizer'), bitrate=1800)
                anim.save(video_path, writer=writer)
                print(f"✓ 视频保存成功: {video_path}")
            except Exception as e:
                print(f"FFmpeg保存失败: {e}")
                # 尝试使用pillow保存gif
                gif_path = os.path.join(self.output_dir, f"{self.motion_name}_animation.gif")
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
    parser = argparse.ArgumentParser(description="可视化AMP动作数据（不需要Isaac Sim）")
    parser.add_argument("--motion_file", type=str, default=None, help="指定动作文件名（如 pace0.txt）")
    parser.add_argument("--all", action="store_true", help="可视化所有动作文件")
    parser.add_argument("--output_dir", type=str, default="logs/motion_vis", help="输出目录")
    parser.add_argument("--show", action="store_true", help="显示动画窗口")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("AMP Motion Visualizer (Simple Version)")
    print("="*60)
    print("注意：AMP数据使用PyBullet格式四元数 [x,y,z,w]")
    print("="*60)

    # 获取动作文件列表
    if args.motion_file:
        motion_files = [args.motion_file]
    elif args.all:
        motion_files = [f for f in os.listdir(MOTION_DIR) if f.endswith('.txt')]
    else:
        # 默认只处理第一个文件
        motion_files = [f for f in sorted(os.listdir(MOTION_DIR)) if f.endswith('.txt')][:1]
    
    print(f"\n将处理 {len(motion_files)} 个动作文件:")
    for f in motion_files:
        print(f"  - {f}")
    
    # 处理每个文件
    for motion_file in motion_files:
        print(f"\n{'='*40}")
        print(f"处理: {motion_file}")
        print(f"{'='*40}")
        
        visualizer = QuadrupedVisualizer(motion_file, args.output_dir)
        visualizer.create_animation(save_video=True, show=args.show)
    
    print("\n" + "="*60)
    print("完成！")
    print(f"输出目录: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
