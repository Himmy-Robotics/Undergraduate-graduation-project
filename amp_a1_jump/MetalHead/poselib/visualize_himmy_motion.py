#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: visualize_himmy_motion.py
@Auth: Based on T-pose visualization
@Date: 2024/12

Himmy Mark2 Motion Visualizer
==============================

该脚本用于可视化retargeting后的动作数据，并保存为视频文件。
完全基于T-pose的骨架结构和parent连接关系进行绘制。

使用方法:
    python visualize_himmy_motion.py
    python visualize_himmy_motion.py --npy_file path/to/motion.npy
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端，适合服务器环境
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

# 设置路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'poselib'))

from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState


# 颜色配置（与T-pose 3D可视化一致）
COLORS = {
    'trunk': 'red',
    'FL': 'green',
    'FR': 'blue',
    'RL': 'orange',
    'RR': 'purple',
    'spine': 'cyan',
}


def quat_to_euler_deg(q):
    """
    四元数转欧拉角（度）
    q: [x, y, z, w] 格式的四元数
    返回: [roll, pitch, yaw] 度数
    """
    x, y, z, w = q[0], q[1], q[2], q[3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw]) * 180.0 / np.pi  # 转换为度


def quat_to_axis_angle_deg(q):
    """
    四元数转轴角表示，返回主转动角度（度）
    q: [x, y, z, w] 格式的四元数
    返回: 角度（度）
    """
    x, y, z, w = q[0], q[1], q[2], q[3]
    
    # 归一化四元数
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm > 1e-8:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # 确保w为正（取最短路径）
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    
    # 计算角度
    angle = 2 * np.arccos(np.clip(w, -1, 1))
    
    # 转换为度
    angle_deg = np.degrees(angle)
    
    # 如果角度大于180度，取补角
    if angle_deg > 180:
        angle_deg = angle_deg - 360
    
    # 确定旋转轴的主方向，并根据轴方向确定符号
    sin_half = np.sqrt(x*x + y*y + z*z)
    if sin_half > 1e-8:
        axis = np.array([x, y, z]) / sin_half
        # 找出主轴
        abs_axis = np.abs(axis)
        main_axis_idx = np.argmax(abs_axis)
        # 根据主轴方向确定符号
        if axis[main_axis_idx] < 0:
            angle_deg = -angle_deg
    
    return angle_deg


def get_node_color(name):
    """根据节点名称返回颜色"""
    for key, color in COLORS.items():
        if key in name:
            return color
    return 'gray'


class HimmyVisualizer:
    """Himmy Mark2 动作可视化器（基于T-pose结构）"""
    
    def __init__(self, npy_file, output_dir="data/motion_vis"):
        self.npy_file = npy_file
        self.output_dir = os.path.join(script_dir, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载SkeletonMotion
        self.motion = SkeletonMotion.from_file(npy_file)
        self.skeleton = self.motion.skeleton_tree
        self.num_frames = self.motion.global_translation.shape[0]
        self.fps = self.motion.fps
        self.dt = 1.0 / self.fps
        
        # 获取骨架信息
        self.node_names = self.skeleton.node_names
        self.parent_indices = self.skeleton.parent_indices.numpy()
        
        self.motion_name = os.path.splitext(os.path.basename(npy_file))[0]
        
        print(f"加载动作: {self.motion_name}")
        print(f"帧数: {self.num_frames}")
        print(f"FPS: {self.fps}")
        print(f"节点数: {len(self.node_names)}")
    
    def get_frame_positions(self, frame_idx):
        """获取指定帧的所有节点全局位置"""
        return self.motion.global_translation[frame_idx].numpy()
    
    def create_animation(self, save_video=True, max_frames=None):
        """创建3D动画并保存为视频（坐标轴跟随机器人动态变化，显示关节角度）"""
        print("\n创建动画...")
        
        num_frames = self.num_frames
        if max_frames is not None:
            num_frames = min(num_frames, max_frames)
        
        # 创建图形 - 使用更宽的布局来容纳关节角度信息
        fig = plt.figure(figsize=(22, 12), dpi=100)
        
        # 左侧3D视图 (占60%宽度)
        ax = fig.add_axes([0.02, 0.05, 0.55, 0.90], projection='3d')
        
        # 右侧关节角度文本区域 (占38%宽度)
        ax_text = fig.add_axes([0.60, 0.05, 0.38, 0.90])
        ax_text.axis('off')  # 隐藏坐标轴
        
        # 预计算所有帧的root位置
        root_positions = self.motion.global_translation[:num_frames, 0].numpy()
        
        # 动态坐标轴参数
        # 视野范围大小（机器人周围显示的区域大小）
        VIEW_RANGE = 1.0  # 1.0米，让机器人显示更大
        
        print(f"动态坐标轴模式: 视野范围 {VIEW_RANGE:.2f}m，坐标轴跟随机器人移动")
        
        # 初始化线条存储
        bone_lines = []
        joint_points = None
        
        # 根据parent关系创建骨骼线条
        for i in range(len(self.node_names)):
            parent_idx = self.parent_indices[i]
            if parent_idx >= 0:
                # 根据节点名获取颜色和线宽
                color = get_node_color(self.node_names[i])
                # 脊柱骨骼使用更粗的线条
                if 'spine' in self.node_names[i].lower():
                    linewidth = 5
                else:
                    linewidth = 3
                line, = ax.plot([], [], [], '-', color=color, linewidth=linewidth)
                bone_lines.append((i, parent_idx, line))
        
        # 关节点 - 脊柱关节使用更大的点
        joint_scatter = ax.scatter([], [], [], c='black', s=40)
        
        # 轨迹线（root轨迹）- 只显示最近的轨迹
        trajectory_line, = ax.plot([], [], [], 'r--', alpha=0.5, linewidth=1, label='Trajectory')
        trajectory_data = []
        TRAJECTORY_LENGTH = 60  # 只显示最近60帧的轨迹（约2秒）
        
        # 时间文本
        time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=14, fontweight='bold')
        
        # 关节角度文本对象（在右侧文本区域）- 使用更大字体
        joint_angle_text = ax_text.text(0.02, 0.98, '', transform=ax_text.transAxes, 
                                         fontsize=14, fontfamily='monospace',
                                         verticalalignment='top', horizontalalignment='left')
        
        # 定义关节显示顺序和名称（只显示有意义的关节，跳过foot和trunk）
        joint_display_order = [
            # 脊柱关节
            ('yaw_spine_Link', 'Spine Yaw'),
            ('pitch_spine_Link', 'Spine Pitch'),
            ('roll_spine_Link', 'Spine Roll'),
            # 前左腿
            ('FL_hip_Link', 'FL Hip'),
            ('FL_thigh_Link', 'FL Thigh'),
            ('FL_calf_Link', 'FL Calf'),
            # 前右腿
            ('FR_hip_Link', 'FR Hip'),
            ('FR_thigh_Link', 'FR Thigh'),
            ('FR_calf_Link', 'FR Calf'),
            # 后左腿
            ('RL_hip_Link', 'RL Hip'),
            ('RL_thigh_Link', 'RL Thigh'),
            ('RL_calf_Link', 'RL Calf'),
            # 后右腿
            ('RR_hip_Link', 'RR Hip'),
            ('RR_thigh_Link', 'RR Thigh'),
            ('RR_calf_Link', 'RR Calf'),
        ]
        
        # 获取关节索引
        joint_indices = []
        for node_name, display_name in joint_display_order:
            if node_name in self.node_names:
                joint_indices.append((self.node_names.index(node_name), display_name))
        
        # 图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', linewidth=3, label='FL (Front Left)'),
            Line2D([0], [0], color='blue', linewidth=3, label='FR (Front Right)'),
            Line2D([0], [0], color='orange', linewidth=3, label='RL (Rear Left)'),
            Line2D([0], [0], color='purple', linewidth=3, label='RR (Rear Right)'),
            Line2D([0], [0], color='cyan', linewidth=5, label='Spine'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Trajectory'),
        ]
        
        def init():
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.set_zlabel('Z (m)', fontsize=10)
            ax.set_title(f'Himmy Mark2 Motion: {self.motion_name}', fontsize=14, fontweight='bold')
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
            # 初始化时设置初始坐标轴
            root_pos = root_positions[0]
            ax.set_xlim([root_pos[0] - VIEW_RANGE/2, root_pos[0] + VIEW_RANGE/2])
            ax.set_ylim([root_pos[1] - VIEW_RANGE/2, root_pos[1] + VIEW_RANGE/2])
            ax.set_zlim([-0.1, VIEW_RANGE - 0.1])
            ax.view_init(elev=25, azim=-60)
            return []
        
        def update(frame_idx):
            # 获取当前帧所有节点位置
            positions = self.get_frame_positions(frame_idx)
            
            # 获取当前root位置作为视野中心
            root_pos = positions[0]
            
            # 更新骨骼线条（基于parent连接）
            for i, parent_idx, line in bone_lines:
                x = [positions[parent_idx, 0], positions[i, 0]]
                y = [positions[parent_idx, 1], positions[i, 1]]
                z = [positions[parent_idx, 2], positions[i, 2]]
                line.set_data(x, y)
                line.set_3d_properties(z)
            
            # 更新关节点
            joint_scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
            
            # 更新轨迹（root位置）- 只保留最近的轨迹
            trajectory_data.append(positions[0].copy())
            if len(trajectory_data) > TRAJECTORY_LENGTH:
                trajectory_data.pop(0)
            if len(trajectory_data) > 1:
                traj = np.array(trajectory_data)
                trajectory_line.set_data(traj[:, 0], traj[:, 1])
                trajectory_line.set_3d_properties(traj[:, 2])
            
            # 更新时间文本
            current_time = frame_idx * self.dt
            time_text.set_text(f'Time: {current_time:.2f}s  Frame: {frame_idx}/{num_frames-1}')
            
            # 获取当前帧的局部旋转（四元数）
            local_rotations = self.motion.local_rotation[frame_idx].numpy()
            
            # 构建关节角度显示文本
            angle_lines = []
            angle_lines.append("=" * 35)
            angle_lines.append(f"   JOINT ANGLES (Frame {frame_idx})")
            angle_lines.append("=" * 35)
            angle_lines.append("")
            
            # 分组显示
            current_group = ""
            for idx, display_name in joint_indices:
                # 检测分组变化
                if 'Spine' in display_name and current_group != 'Spine':
                    current_group = 'Spine'
                    angle_lines.append("──── SPINE ────")
                elif 'FL' in display_name and current_group != 'FL':
                    current_group = 'FL'
                    angle_lines.append("")
                    angle_lines.append("──── FRONT LEFT LEG ────")
                elif 'FR' in display_name and current_group != 'FR':
                    current_group = 'FR'
                    angle_lines.append("")
                    angle_lines.append("──── FRONT RIGHT LEG ────")
                elif 'RL' in display_name and current_group != 'RL':
                    current_group = 'RL'
                    angle_lines.append("")
                    angle_lines.append("──── REAR LEFT LEG ────")
                elif 'RR' in display_name and current_group != 'RR':
                    current_group = 'RR'
                    angle_lines.append("")
                    angle_lines.append("──── REAR RIGHT LEG ────")
                
                # 获取四元数并转换为单个转角
                quat = local_rotations[idx]
                joint_angle = quat_to_axis_angle_deg(quat)
                
                # 格式化显示 - 只显示单个转角
                angle_lines.append(f"  {display_name:12s}: {joint_angle:+8.2f}°")
            
            # Root位置信息
            angle_lines.append("")
            angle_lines.append("──── ROOT POSITION ────")
            angle_lines.append(f"  X: {root_pos[0]:+8.4f} m")
            angle_lines.append(f"  Y: {root_pos[1]:+8.4f} m")
            angle_lines.append(f"  Z: {root_pos[2]:+8.4f} m")
            
            # 更新关节角度文本
            joint_angle_text.set_text('\n'.join(angle_lines))
            
            # 动态更新坐标轴范围，跟随机器人移动
            ax.set_xlim([root_pos[0] - VIEW_RANGE/2, root_pos[0] + VIEW_RANGE/2])
            ax.set_ylim([root_pos[1] - VIEW_RANGE/2, root_pos[1] + VIEW_RANGE/2])
            ax.set_zlim([-0.1, VIEW_RANGE - 0.1])  # Z轴固定从地面开始
            
            # 固定视角（不旋转）
            ax.view_init(elev=25, azim=-60)
            ax.dist = 8
            
            return []
        
        # 创建动画
        anim = FuncAnimation(fig, update, frames=num_frames,
                           init_func=init, blit=False, interval=self.dt * 1000)
        
        if save_video:
            video_path = os.path.join(self.output_dir, f"{self.motion_name}_animation.mp4")
            print(f"保存视频到: {video_path}")
            
            try:
                writer = FFMpegWriter(fps=int(self.fps), metadata={'title': self.motion_name}, bitrate=3000)
                anim.save(video_path, writer=writer, dpi=100)
                print(f"视频保存成功: {video_path}")
            except Exception as e:
                print(f"FFMpeg保存失败: {e}")
                try:
                    gif_path = os.path.join(self.output_dir, f"{self.motion_name}_animation.gif")
                    print(f"尝试保存为GIF: {gif_path}")
                    anim.save(gif_path, writer='pillow', fps=min(int(self.fps), 15))
                    print(f"GIF保存成功: {gif_path}")
                except Exception as e2:
                    print(f"GIF保存也失败: {e2}")
        
        plt.close()
        return anim


def main():
    parser = argparse.ArgumentParser(description="Himmy Mark2 动作可视化工具")
    parser.add_argument("--npy_file", type=str, default=None, help="NPY格式动作文件路径")
    parser.add_argument("--output_dir", type=str, default="data/motion_vis", help="输出目录")
    parser.add_argument("--max_frames", type=int, default=None, help="最大帧数限制")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Himmy Mark2 Motion Visualizer (基于T-pose骨架结构)")
    print("=" * 60)
    
    # 如果没有指定文件，使用配置文件
    if args.npy_file is None:
        config_path = os.path.join(script_dir, "data/load_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        file_name = config["file_name"]
        clip = config["clip"]
        remarks = config["remarks"]
        
        npy_file = os.path.join(
            script_dir,
            f"data/amp_himmy_mark2/{file_name}/{file_name}_amp_{clip[0]}_{clip[1]}_{remarks}.npy"
        )
    else:
        npy_file = args.npy_file
    
    print(f"动作文件: {npy_file}")
    
    if not os.path.exists(npy_file):
        print(f"错误: 文件不存在: {npy_file}")
        sys.exit(1)
    
    # 创建可视化器并生成视频
    visualizer = HimmyVisualizer(npy_file, args.output_dir)
    visualizer.create_animation(save_video=True, max_frames=args.max_frames)
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
