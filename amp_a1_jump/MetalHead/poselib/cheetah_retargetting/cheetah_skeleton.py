#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
猎豹骨架树定义
=============
定义猎豹的SkeletonTree结构，用于转换为poselib格式

注意：poselib要求骨架节点按拓扑顺序排列，即父节点索引必须小于子节点索引
"""
import numpy as np
import torch

# 原始pickle数据中的关键点顺序（用于读取数据）
CHEETAH_PICKLE_ORDER = [
    'l_eye', 'r_eye', 'nose',           # 0-2: 头部
    'neck_base', 'spine', 'tail_base',  # 3-5: 脊柱
    'tail_mid', 'tail_tip',             # 6-7: 尾巴
    'l_shoulder', 'l_front_knee', 'l_front_ankle',  # 8-10: 左前腿
    'r_shoulder', 'r_front_knee', 'r_front_ankle',  # 11-13: 右前腿
    'l_hip', 'l_back_knee', 'l_back_ankle',         # 14-16: 左后腿
    'r_hip', 'r_back_knee', 'r_back_ankle',         # 17-19: 右后腿
]

# 重新排序的骨架节点名称（按拓扑顺序：父节点索引 < 子节点索引）
# root (tail_base) 在索引0
CHEETAH_NODE_NAMES = [
    'tail_base',    # 0: ROOT
    'spine',        # 1: <- tail_base(0)
    'neck_base',    # 2: <- spine(1)
    'l_eye',        # 3: <- neck_base(2)
    'r_eye',        # 4: <- neck_base(2)
    'nose',         # 5: <- neck_base(2)
    'l_shoulder',   # 6: <- neck_base(2)
    'l_front_knee', # 7: <- l_shoulder(6)
    'l_front_ankle',# 8: <- l_front_knee(7)
    'r_shoulder',   # 9: <- neck_base(2)
    'r_front_knee', # 10: <- r_shoulder(9)
    'r_front_ankle',# 11: <- r_front_knee(10)
    'l_hip',        # 12: <- tail_base(0)
    'l_back_knee',  # 13: <- l_hip(12)
    'l_back_ankle', # 14: <- l_back_knee(13)
    'r_hip',        # 15: <- tail_base(0)
    'r_back_knee',  # 16: <- r_hip(15)
    'r_back_ankle', # 17: <- r_back_knee(16)
    'tail_mid',     # 18: <- tail_base(0)
    'tail_tip',     # 19: <- tail_mid(18)
]

# 父节点索引（按新顺序）
CHEETAH_PARENT_INDICES = [
    -1,  # tail_base (ROOT)
    0,   # spine -> tail_base
    1,   # neck_base -> spine
    2,   # l_eye -> neck_base
    2,   # r_eye -> neck_base
    2,   # nose -> neck_base
    2,   # l_shoulder -> neck_base
    6,   # l_front_knee -> l_shoulder
    7,   # l_front_ankle -> l_front_knee
    2,   # r_shoulder -> neck_base
    9,   # r_front_knee -> r_shoulder
    10,  # r_front_ankle -> r_front_knee
    0,   # l_hip -> tail_base
    12,  # l_back_knee -> l_hip
    13,  # l_back_ankle -> l_back_knee
    0,   # r_hip -> tail_base
    15,  # r_back_knee -> r_hip
    16,  # r_back_ankle -> r_back_knee
    0,   # tail_mid -> tail_base
    18,  # tail_tip -> tail_mid
]

# 从pickle顺序到骨架顺序的映射
PICKLE_TO_SKELETON_IDX = {name: i for i, name in enumerate(CHEETAH_NODE_NAMES)}
REORDER_INDICES = [PICKLE_TO_SKELETON_IDX[name] for name in CHEETAH_PICKLE_ORDER]

# 关键点索引映射（新顺序）
CHEETAH_KP = {name: i for i, name in enumerate(CHEETAH_NODE_NAMES)}


def reorder_positions(positions):
    """
    将pickle数据的位置从原始顺序重排为骨架拓扑顺序
    
    Args:
        positions: (N, 20, 3) 按CHEETAH_PICKLE_ORDER排列
    Returns:
        positions: (N, 20, 3) 按CHEETAH_NODE_NAMES排列
    """
    # 创建从pickle顺序到skeleton顺序的映射
    new_positions = np.zeros_like(positions)
    for old_idx, name in enumerate(CHEETAH_PICKLE_ORDER):
        new_idx = PICKLE_TO_SKELETON_IDX[name]
        new_positions[:, new_idx, :] = positions[:, old_idx, :]
    return new_positions


def get_cheetah_skeleton_info():
    """返回猎豹骨架信息"""
    return {
        'node_names': CHEETAH_NODE_NAMES,
        'parent_indices': CHEETAH_PARENT_INDICES,
        'keypoint_map': CHEETAH_KP,
        'root_name': 'tail_base',
        'root_idx': 0,  # root现在在索引0
        'pickle_order': CHEETAH_PICKLE_ORDER,
    }
