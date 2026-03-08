# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
观察函数模块 - 定义强化学习环境的观察量

本模块提供了用于构建机器人状态观察的辅助函数。
观察函数计算并返回环境状态的特定部分，这些状态会作为神经网络的输入。

主要功能：
1. joint_pos_rel_without_wheel - 获取关节位置（排除轮子关节）
2. phase - 生成周期性相位信号（用于步态生成）
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    获取关节相对位置（相对于默认位置），排除轮子关节

    该函数用于混合运动机器人（同时具有腿部和轮子），只返回腿部关节的相对位置，
    将轮子关节的值设为0。这样可以让策略专注于腿部运动的控制。

    Args:
        env: 环境对象
        asset_cfg: 资产配置，指定目标机器人和关节
        wheel_asset_cfg: 轮子关节配置，指定哪些关节是轮子

    Returns:
        关节相对位置张量，形状为 (num_envs, num_joints)
        轮子关节的位置被设为0

    示例：
        # 在配置中使用：
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_wheel")
            }
        )
    """
    # 提取资产对象（启用类型提示）
    asset: Articulation = env.scene[asset_cfg.name]

    # 计算关节相对位置（当前位置 - 默认位置）
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    # 将轮子关节的相对位置设为0（不作为观察量）
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0

    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    """
    生成周期性相位信号（用于步态生成）

    该函数生成一个周期性的二维相位向量 [sin(φ), cos(φ)]，用于帮助策略生成周期性步态。
    相位角 φ = 2π × (当前时间 / 周期时间)

    Args:
        env: 环境对象
        cycle_time: 步态周期时间（秒），如 0.5 表示2Hz的步态频率

    Returns:
        相位张量，形状为 (num_envs, 2)，包含 [sin(φ), cos(φ)]

    工作原理：
        - 通过 sin 和 cos 表示相位，避免了角度的周期性不连续问题
        - 例如：cycle_time=1.0秒，当 t=0.25秒时，φ=π/2，输出 [1, 0]
        - 随着时间推移，相位向量沿单位圆旋转

    应用场景：
        - 四足机器人的对角步态（trot gait）
        - 双足机器人的行走步态
        - 任何需要周期性协调运动的任务

    示例：
        # 在配置中使用：
        phase_obs = ObsTerm(
            func=mdp.phase,
            params={"cycle_time": 0.5}  # 0.5秒一个步态周期（2Hz）
        )
    """
    # 初始化 episode_length_buf（如果不存在）
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    # 计算归一化相位：当前时间 / 周期时间
    # episode_length_buf 记录了当前episode已经过的步数
    # 乘以 step_dt 得到经过的时间（秒）
    # 除以 cycle_time 得到相位（0到1之间循环）
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time

    # 将相位转换为 [sin(2πφ), cos(2πφ)] 形式
    # 这样可以平滑地表示周期性，避免从2π跳回0的不连续性
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)

    return phase_tensor
