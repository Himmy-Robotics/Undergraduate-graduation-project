#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cheetah T-Pose Generator
========================
基于猎豹关键点数据生成T-Pose骨架。

猎豹关键点（20个）:
0:l_eye, 1:r_eye, 2:nose, 3:neck_base, 4:spine, 5:tail_base, 
6:tail_mid, 7:tail_tip, 8:l_shoulder, 9:l_front_knee, 10:l_front_ankle,
11:r_shoulder, 12:r_front_knee, 13:r_front_ankle, 14:l_hip, 
15:l_back_knee, 16:l_back_ankle, 17:r_hip, 18:r_back_knee, 19:r_back_ankle
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 关键点索引
KP = {
    'l_eye': 0, 'r_eye': 1, 'nose': 2, 'neck_base': 3, 'spine': 4,
    'tail_base': 5, 'tail_mid': 6, 'tail_tip': 7,
    'l_shoulder': 8, 'l_front_knee': 9, 'l_front_ankle': 10,
    'r_shoulder': 11, 'r_front_knee': 12, 'r_front_ankle': 13,
    'l_hip': 14, 'l_back_knee': 15, 'l_back_ankle': 16,
    'r_hip': 17, 'r_back_knee': 18, 'r_back_ankle': 19,
}

# 骨骼连接
BONES = [
    ('nose', 'neck_base'), ('neck_base', 'spine'), ('spine', 'tail_base'),
    ('tail_base', 'tail_mid'), ('tail_mid', 'tail_tip'),
    ('neck_base', 'l_shoulder'), ('l_shoulder', 'l_front_knee'), 
    ('l_front_knee', 'l_front_ankle'),
    ('neck_base', 'r_shoulder'), ('r_shoulder', 'r_front_knee'), 
    ('r_front_knee', 'r_front_ankle'),
    ('tail_base', 'l_hip'), ('l_hip', 'l_back_knee'), 
    ('l_back_knee', 'l_back_ankle'),
    ('tail_base', 'r_hip'), ('r_hip', 'r_back_knee'), 
    ('r_back_knee', 'r_back_ankle'),
]

"""
猎豹骨架尺寸参考（来自AcinoSet: all_optimizations.py）：
============================================================
脊柱链：
- head -> neck_base: 0.28m
- neck_base -> spine: 0.37m  
- spine -> tail_base: 0.37m
- tail_base -> tail_mid: 0.28m
- tail_mid -> tail_tip: 0.36m

头部：
- head -> l_eye/r_eye: [0, ±0.03, 0]
- head -> nose: [0.055, 0, -0.055]

前腿（从neck_base）：
- neck_base -> shoulder: [-0.04, ±0.08, -0.10]
- shoulder -> front_knee: [0, 0, -0.24]
- front_knee -> front_ankle: [0, 0, -0.28]

后腿（从tail_base）：
- tail_base -> hip: [0.12, ±0.08, -0.06]
- hip -> back_knee: [0, 0, -0.32]
- back_knee -> back_ankle: [0, 0, -0.25]
"""

def create_cheetah_tpose():
    """基于AcinoSet官方骨架尺寸创建T-Pose（单位：米）"""
    positions = np.zeros((20, 3))
    
    # 定义head位置作为起点（抬高使脚接近地面）
    head_pos = np.array([0.28, 0, 0.68])  # head在原点右侧
    
    # 头部
    positions[KP['l_eye']] = head_pos + np.array([0, 0.03, 0])
    positions[KP['r_eye']] = head_pos + np.array([0, -0.03, 0])
    positions[KP['nose']] = head_pos + np.array([0.055, 0, -0.055])
    
    # 脊柱链 (沿X轴负方向)
    positions[KP['neck_base']] = head_pos + np.array([-0.28, 0, 0])
    positions[KP['spine']] = positions[KP['neck_base']] + np.array([-0.37, 0, 0])
    positions[KP['tail_base']] = positions[KP['spine']] + np.array([-0.37, 0, 0])
    positions[KP['tail_mid']] = positions[KP['tail_base']] + np.array([-0.28, 0, 0])
    positions[KP['tail_tip']] = positions[KP['tail_mid']] + np.array([-0.36, 0, 0])
    
    # 前腿 (从neck_base)
    positions[KP['l_shoulder']] = positions[KP['neck_base']] + np.array([-0.04, 0.08, -0.10])
    positions[KP['l_front_knee']] = positions[KP['l_shoulder']] + np.array([0, 0, -0.24])
    positions[KP['l_front_ankle']] = positions[KP['l_front_knee']] + np.array([0, 0, -0.28])
    
    positions[KP['r_shoulder']] = positions[KP['neck_base']] + np.array([-0.04, -0.08, -0.10])
    positions[KP['r_front_knee']] = positions[KP['r_shoulder']] + np.array([0, 0, -0.24])
    positions[KP['r_front_ankle']] = positions[KP['r_front_knee']] + np.array([0, 0, -0.28])
    
    # 后腿 (从tail_base)
    positions[KP['l_hip']] = positions[KP['tail_base']] + np.array([0.12, 0.08, -0.06])
    positions[KP['l_back_knee']] = positions[KP['l_hip']] + np.array([0, 0, -0.32])
    positions[KP['l_back_ankle']] = positions[KP['l_back_knee']] + np.array([0, 0, -0.25])
    
    positions[KP['r_hip']] = positions[KP['tail_base']] + np.array([0.12, -0.08, -0.06])
    positions[KP['r_back_knee']] = positions[KP['r_hip']] + np.array([0, 0, -0.32])
    positions[KP['r_back_ankle']] = positions[KP['r_back_knee']] + np.array([0, 0, -0.25])
    
    return positions

def plot_tpose(positions, save_path=None):
    """可视化T-Pose并标注关节名称"""
    fig = plt.figure(figsize=(16, 6))
    
    # 3D视图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Cheetah T-Pose (3D View)')
    
    # 侧视图
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Cheetah T-Pose (Side View)')
    
    # 关节名称缩写（用于标注）
    short_names = {
        'l_eye': 'LE', 'r_eye': 'RE', 'nose': 'N', 'neck_base': 'NB', 
        'spine': 'SP', 'tail_base': 'TB', 'tail_mid': 'TM', 'tail_tip': 'TT',
        'l_shoulder': 'LS', 'l_front_knee': 'LFK', 'l_front_ankle': 'LFA',
        'r_shoulder': 'RS', 'r_front_knee': 'RFK', 'r_front_ankle': 'RFA',
        'l_hip': 'LH', 'l_back_knee': 'LBK', 'l_back_ankle': 'LBA',
        'r_hip': 'RH', 'r_back_knee': 'RBK', 'r_back_ankle': 'RBA',
    }
    
    for ax in [ax1, ax2]:
        # 绘制骨骼
        for bone in BONES:
            i1, i2 = KP[bone[0]], KP[bone[1]]
            ax.plot([positions[i1, 0], positions[i2, 0]],
                   [positions[i1, 1], positions[i2, 1]],
                   [positions[i1, 2], positions[i2, 2]], 'b-', lw=2)
        
        # 绘制关键点
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='red', s=50)
        
        # 标注关节名称
        for name, idx in KP.items():
            short = short_names[name]
            ax.text(positions[idx, 0], positions[idx, 1], positions[idx, 2] + 0.02,
                   short, fontsize=8, ha='center')
        
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_box_aspect([1.5, 0.5, 0.8])
    
    ax1.view_init(elev=30, azim=-60)
    ax2.view_init(elev=0, azim=-90)
    
    # 添加图例说明
    legend_text = "Joints: N=nose, NB=neck_base, SP=spine, TB=tail_base, TM=tail_mid, TT=tail_tip\n"
    legend_text += "Front legs: LS/RS=shoulder, LFK/RFK=front_knee, LFA/RFA=front_ankle\n"
    legend_text += "Back legs: LH/RH=hip, LBK/RBK=back_knee, LBA/RBA=back_ankle"
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()

if __name__ == "__main__":
    print("Creating Cheetah T-Pose...")
    tpose = create_cheetah_tpose()
    
    # 保存
    np.save("cheetah_tpose.npy", {'positions': tpose, 'keypoints': KP})
    print("Saved: cheetah_tpose.npy")
    
    # 可视化并保存图片
    plot_tpose(tpose, save_path="cheetah_tpose.png")
