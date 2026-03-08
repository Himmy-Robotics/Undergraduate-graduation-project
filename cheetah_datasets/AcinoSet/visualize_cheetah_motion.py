#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
猎豹运动数据可视化脚本
========================
将 AcinoSet 的 3D 轨迹数据可视化并保存为视频。
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import os

# 关键点索引
KEYPOINTS = {
    'l_eye': 0, 'r_eye': 1, 'nose': 2, 'neck_base': 3, 'spine': 4,
    'tail_base': 5, 'tail_mid': 6, 'tail_tip': 7,
    'l_shoulder': 8, 'l_front_knee': 9, 'l_front_ankle': 10,
    'r_shoulder': 11, 'r_front_knee': 12, 'r_front_ankle': 13,
    'l_hip': 14, 'l_back_knee': 15, 'l_back_ankle': 16,
    'r_hip': 17, 'r_back_knee': 18, 'r_back_ankle': 19,
}

# 骨骼连接定义
BONES = [
    # 头部和脊柱
    ('nose', 'neck_base'),
    ('neck_base', 'spine'),
    ('spine', 'tail_base'),
    ('tail_base', 'tail_mid'),
    ('tail_mid', 'tail_tip'),
    # 左前腿
    ('neck_base', 'l_shoulder'),
    ('l_shoulder', 'l_front_knee'),
    ('l_front_knee', 'l_front_ankle'),
    # 右前腿
    ('neck_base', 'r_shoulder'),
    ('r_shoulder', 'r_front_knee'),
    ('r_front_knee', 'r_front_ankle'),
    # 左后腿
    ('tail_base', 'l_hip'),
    ('l_hip', 'l_back_knee'),
    ('l_back_knee', 'l_back_ankle'),
    # 右后腿
    ('tail_base', 'r_hip'),
    ('r_hip', 'r_back_knee'),
    ('r_back_knee', 'r_back_ankle'),
]

# 骨骼颜色
BONE_COLORS = {
    'spine': 'blue',      # 脊柱
    'l_front': 'red',     # 左前腿
    'r_front': 'green',   # 右前腿
    'l_back': 'orange',   # 左后腿
    'r_back': 'purple',   # 右后腿
    'tail': 'gray',       # 尾巴
}

def get_bone_color(bone):
    """根据骨骼类型返回颜色"""
    start, end = bone
    if 'tail' in start or 'tail' in end:
        return BONE_COLORS['tail']
    if 'l_front' in start or 'l_front' in end or (start == 'neck_base' and 'l_shoulder' in end):
        return BONE_COLORS['l_front']
    if 'r_front' in start or 'r_front' in end or (start == 'neck_base' and 'r_shoulder' in end):
        return BONE_COLORS['r_front']
    if 'l_hip' in start or 'l_back' in start or 'l_hip' in end or 'l_back' in end:
        return BONE_COLORS['l_back']
    if 'r_hip' in start or 'r_back' in start or 'r_hip' in end or 'r_back' in end:
        return BONE_COLORS['r_back']
    return BONE_COLORS['spine']


def visualize_motion(filepath, output_video, title="Cheetah Motion"):
    """
    可视化猎豹运动并保存为视频
    
    Args:
        filepath: pickle文件路径
        output_video: 输出视频路径
        title: 视频标题
    """
    # 加载数据
    with open(filepath, 'rb') as f:
        traj = pickle.load(f)
    
    # 处理 positions（可能是 list 或 ndarray）
    positions = traj['positions']
    if isinstance(positions, list):
        positions = np.array(positions)  # (N, 20, 3)
    
    # 处理 dx（可能是 list 或 ndarray，维度可能是25或45）
    dx = traj['dx']
    if isinstance(dx, list):
        dx = np.array(dx)
    
    n_frames = positions.shape[0]
    print(f"加载 {filepath}")
    print(f"  帧数: {n_frames}")
    print(f"  positions shape: {positions.shape}")
    print(f"  dx shape: {dx.shape}")
    
    # 计算速度（取前3维作为根节点速度）
    root_speed = np.linalg.norm(dx[:, :3], axis=1)
    print(f"  平均速度: {root_speed.mean():.2f} m/s")
    print(f"  最大速度: {root_speed.max():.2f} m/s")
    
    # 创建图形 - 四个视图
    fig = plt.figure(figsize=(18, 12))
    
    # 3D视图 - 俯视
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.set_title(f'{title} - 3D View')
    
    # 3D视图 - 侧视
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.set_title(f'{title} - Side View (XOZ)')
    
    # 3D视图 - 正视（从正面看YOZ平面）
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.set_title(f'{title} - Front View (YOZ)')
    
    # 3D视图 - 俯视（从上方看XOY平面）
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.set_title(f'{title} - Top View (XOY)')
    
    # 计算边界（使用整个轨迹的范围）
    all_pos = positions.reshape(-1, 3)
    x_min, x_max = all_pos[:, 0].min() - 0.5, all_pos[:, 0].max() + 0.5
    y_min, y_max = all_pos[:, 1].min() - 0.5, all_pos[:, 1].max() + 0.5
    z_min, z_max = all_pos[:, 2].min() - 0.2, all_pos[:, 2].max() + 0.5
    
    # 初始化线条和点
    lines1 = []
    lines2 = []
    lines3 = []
    lines4 = []
    
    for bone in BONES:
        color = get_bone_color(bone)
        line1, = ax1.plot([], [], [], color=color, linewidth=2)
        line2, = ax2.plot([], [], [], color=color, linewidth=2)
        line3, = ax3.plot([], [], [], color=color, linewidth=2)
        line4, = ax4.plot([], [], [], color=color, linewidth=2)
        lines1.append(line1)
        lines2.append(line2)
        lines3.append(line3)
        lines4.append(line4)
    
    # 关键点散点
    scatter1 = ax1.scatter([], [], [], c='black', s=20)
    scatter2 = ax2.scatter([], [], [], c='black', s=20)
    scatter3 = ax3.scatter([], [], [], c='black', s=20)
    scatter4 = ax4.scatter([], [], [], c='black', s=20)
    
    # 轨迹线
    trajectory_x = []
    trajectory_y = []
    trajectory_z = []
    traj_line1, = ax1.plot([], [], [], 'b--', alpha=0.3, linewidth=1)
    traj_line2, = ax2.plot([], [], [], 'b--', alpha=0.3, linewidth=1)
    traj_line3, = ax3.plot([], [], [], 'b--', alpha=0.3, linewidth=1)
    traj_line4, = ax4.plot([], [], [], 'b--', alpha=0.3, linewidth=1)
    
    # 速度文本
    speed_text = ax1.text2D(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12)
    frame_text = ax1.text2D(0.02, 0.90, '', transform=ax1.transAxes, fontsize=10)
    
    def init():
        for line in lines1 + lines2 + lines3 + lines4:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines1 + lines2 + lines3 + lines4 + [scatter1, scatter2, scatter3, scatter4, traj_line1, traj_line2, traj_line3, traj_line4, speed_text, frame_text]
    
    def update(frame):
        pos = positions[frame]  # (20, 3)
        speed = root_speed[frame]
        
        # 更新骨骼
        for i, bone in enumerate(BONES):
            start_idx = KEYPOINTS[bone[0]]
            end_idx = KEYPOINTS[bone[1]]
            
            xs = [pos[start_idx, 0], pos[end_idx, 0]]
            ys = [pos[start_idx, 1], pos[end_idx, 1]]
            zs = [pos[start_idx, 2], pos[end_idx, 2]]
            
            lines1[i].set_data(xs, ys)
            lines1[i].set_3d_properties(zs)
            lines2[i].set_data(xs, ys)
            lines2[i].set_3d_properties(zs)
            lines3[i].set_data(xs, ys)
            lines3[i].set_3d_properties(zs)
            lines4[i].set_data(xs, ys)
            lines4[i].set_3d_properties(zs)
        
        # 更新关键点
        scatter1._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        scatter2._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        scatter3._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        scatter4._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        
        # 更新轨迹 (使用spine位置)
        spine_pos = pos[KEYPOINTS['spine']]
        trajectory_x.append(spine_pos[0])
        trajectory_y.append(spine_pos[1])
        trajectory_z.append(spine_pos[2])
        
        traj_line1.set_data(trajectory_x, trajectory_y)
        traj_line1.set_3d_properties(trajectory_z)
        traj_line2.set_data(trajectory_x, trajectory_y)
        traj_line2.set_3d_properties(trajectory_z)
        traj_line3.set_data(trajectory_x, trajectory_y)
        traj_line3.set_3d_properties(trajectory_z)
        traj_line4.set_data(trajectory_x, trajectory_y)
        traj_line4.set_3d_properties(trajectory_z)
        
        # 更新视角 - 跟随猎豹
        center_x = spine_pos[0]
        center_y = spine_pos[1]
        view_range = 1.5
        
        # 俯视视角
        ax1.set_xlim(center_x - view_range, center_x + view_range)
        ax1.set_ylim(center_y - view_range, center_y + view_range)
        ax1.set_zlim(z_min, z_max)
        ax1.view_init(elev=30, azim=-60)
        
        # 侧视视角
        ax2.set_xlim(center_x - view_range, center_x + view_range)
        ax2.set_ylim(center_y - view_range, center_y + view_range)
        ax2.set_zlim(z_min, z_max)
        ax2.view_init(elev=0, azim=-90)
        
        # 正视视角（从正面看YOZ平面，即沿X轴负方向看）
        ax3.set_xlim(center_x - view_range, center_x + view_range)
        ax3.set_ylim(center_y - view_range, center_y + view_range)
        ax3.set_zlim(z_min, z_max)
        ax3.view_init(elev=0, azim=0)  # azim=0表示从X正方向看向原点
        
        # 俯视视角（从上方看XOY平面）
        ax4.set_xlim(center_x - view_range, center_x + view_range)
        ax4.set_ylim(center_y - view_range, center_y + view_range)
        ax4.set_zlim(z_min, z_max)
        ax4.view_init(elev=90, azim=-90)  # elev=90从上方看
        
        # 更新文本
        speed_text.set_text(f'Speed: {speed:.2f} m/s')
        frame_text.set_text(f'Frame: {frame}/{n_frames-1}')
        
        return lines1 + lines2 + lines3 + lines4 + [scatter1, scatter2, scatter3, scatter4, traj_line1, traj_line2, traj_line3, traj_line4, speed_text, frame_text]
    
    # 设置坐标轴标签
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=1000/30, blit=False)  # 30 FPS 播放
    
    # 保存视频
    print(f"正在保存视频: {output_video}")
    writer = FFMpegWriter(fps=30, metadata=dict(artist='AcinoSet'), bitrate=5000)
    anim.save(output_video, writer=writer, dpi=100)
    print(f"视频保存完成: {output_video}")
    
    plt.close(fig)


def main():
    # 输出目录
    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # 可视化所有数据集
    datasets = [
        # 原始数据
        ("simulated_data/traj/traj_real.pickle", "cheetah_bound_11ms.mp4", "Cheetah BOUND (~11 m/s)"),
        ("simulated_data/traj/traj_real_2.pickle", "cheetah_trot_9ms.mp4", "Cheetah TROT (~9 m/s)"),
        # 新下载的FTE数据
        ("simulated_data/fte19-3-9-lily-run.pickle", "lily_run_12ms.mp4", "Lily Run (~12 m/s)"),
        ("simulated_data/fte19-3-9-jules-flick1.pickle", "jules_flick1_11ms.mp4", "Jules Flick1 (~11 m/s)"),
        ("simulated_data/fte19-3-9-lily-flick.pickle", "lily_flick_11ms.mp4", "Lily Flick (~11 m/s)"),
        ("simulated_data/fte19-3-9-jules-flick2.pickle", "jules_flick2_10ms.mp4", "Jules Flick2 (~10 m/s)"),
    ]
    
    for filepath, output_name, title in datasets:
        if os.path.exists(filepath):
            output_video = os.path.join(output_dir, output_name)
            visualize_motion(filepath, output_video, title)
        else:
            print(f"跳过不存在的文件: {filepath}")
    
    print("\n所有视频生成完成！")
    print(f"视频保存在: {output_dir}/")


if __name__ == "__main__":
    main()
