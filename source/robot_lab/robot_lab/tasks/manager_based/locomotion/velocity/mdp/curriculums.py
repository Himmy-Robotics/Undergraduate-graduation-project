# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
课程学习函数模块（Curriculum Learning）

本模块提供了用于强化学习环境的课程学习函数。
课程学习是一种训练策略，从简单任务开始，逐步增加任务难度，
可以显著提高学习效率和最终策略的性能。

课程学习的核心思想：
1. 初始阶段：给机器人设置简单的目标（如低速运动）
2. 中间阶段：随着性能提升，逐步增加难度（提高目标速度）
3. 最终阶段：达到完整难度的任务要求

使用方法：
这些函数可以传递给 CurriculumTermCfg 对象，以启用相应的课程学习策略。
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """
    速度命令级别课程学习

    该函数实现了一个自适应课程学习策略，用于逐步增加机器人的速度命令范围。

    工作原理：
    1. 初始阶段：速度命令范围缩小到原始范围的10%（由 range_multiplier[0] 控制）
    2. 评估性能：每个episode结束时，评估速度跟踪奖励的平均值
    3. 自适应提升：如果奖励超过80%的最大值，则扩大速度命令范围
    4. 最终阶段：速度命令范围达到原始范围的100%（由 range_multiplier[1] 控制）

    Args:
        env: 强化学习环境对象
        env_ids: 需要更新课程的环境ID列表
        reward_term_name: 用于评估性能的奖励项名称（通常是速度跟踪奖励）
        range_multiplier: 速度范围乘数 (初始倍数, 最终倍数)
            - 默认 (0.1, 1.0) 表示从10%逐步提升到100%

    Returns:
        当前的最大线速度命令（x方向）

    示例：
        # 在配置中使用：
        curriculum.command_levels = CurrTerm(
            func=mdp.command_levels_vel,
            params={
                "reward_term_name": "track_lin_vel_xy_exp",
                "range_multiplier": (0.2, 1.0)  # 从20%开始，逐步提升到100%
            }
        )
    """
    # 获取基础速度命令的范围配置
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    # 【第一步：初始化】仅在第一次调用时（第0步）保存原始速度范围
    if env.common_step_counter == 0:
        # 保存原始速度范围（用于后续计算）
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)

        # 计算初始速度范围（缩小到原始范围的 range_multiplier[0] 倍）
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # 将命令范围初始化为较小的初始值
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # 【第二步：自适应调整】每个episode结束时检查是否需要增加难度
    # 避免每一步都更新课程，因为最大命令值对所有环境都是共同的
    if env.common_step_counter % env.max_episode_length == 0:
        # 获取该奖励项在当前episode中的累计值
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        # 获取奖励项的配置（包含权重）
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        # 每次增加的速度范围量（[-0.1, +0.1] 表示下界-0.1，上界+0.1）
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # 【第三步：性能评估】如果跟踪奖励超过最大值的80%，则增加命令范围
        # 平均奖励 / 每秒最大奖励 > 0.8 × 权重
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            # 计算新的速度范围（在当前基础上增加 delta_command）
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # 限制在最终范围内（不超过 range_multiplier[1] 倍的原始范围）
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # 更新命令范围
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    # 返回当前的最大线速度命令（x方向的上界）
    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)







def penalty_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_names: list[str],
    initial_weights: list[float],
    final_weights: list[float],
) -> None:
    """
    根据当前速度命令范围调整惩罚项权重
    
    随着课程学习的进行（速度范围增大），逐渐减小惩罚项的权重，
    让机器人在高速运行时能够更加自由地运动。
    
    该函数依赖于 command_levels_vel 已经初始化了 env._initial_vel_x 和 env._final_vel_x。
    如果未找到这些属性，函数将不做任何操作。
    
    Args:
        env: 环境对象
        env_ids: 环境ID列表（本函数修改全局权重，不使用此参数）
        reward_term_names: 需要调整权重的奖励项名称列表
        initial_weights: 初始阶段（低速）的权重列表
        final_weights: 最终阶段（高速）的权重列表
    """
    # 检查是否由 command_levels_vel 初始化了速度范围信息
    if not hasattr(env, "_final_vel_x") or not hasattr(env, "_initial_vel_x"):
        return

    # 获取当前速度范围的上界
    cmd_cfg = env.command_manager.get_term("base_velocity").cfg
    current_max_speed = cmd_cfg.ranges.lin_vel_x[1]
    
    final_max_speed = env._final_vel_x[1].item()
    initial_max_speed = env._initial_vel_x[1].item()

    # 计算进度 (0.0 ~ 1.0)
    if final_max_speed > initial_max_speed:
        progress = (current_max_speed - initial_max_speed) / (final_max_speed - initial_max_speed)
    else:
        progress = 1.0
        
    progress = max(0.0, min(1.0, progress))
    
    # 更新每个奖励项的权重
    for name, init_w, final_w in zip(reward_term_names, initial_weights, final_weights):
        # 获取奖励项配置
        # 注意：get_term_cfg 可能返回 None 如果奖励项不存在
        try:
            term_cfg = env.reward_manager.get_term_cfg(name)
            if term_cfg is not None:
                # 线性插值计算新权重
                new_weight = init_w + (final_w - init_w) * progress
                term_cfg.weight = new_weight
        except Exception:
            pass


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z_exp",
    range_multiplier: tuple[float, float] = (0.1, 1.0),
) -> torch.Tensor:
    """
    角速度命令级别课程学习
    
    该函数实现了一个自适应课程学习策略，用于逐步增加机器人的角速度命令范围。
    
    与线速度课程的区别：
    - 角速度范围是对称的 (-max, +max)
    - 课程学习只改变 max 的绝对值大小
    - 正向和反向角速度同时调整
    
    工作原理：
    1. 初始阶段：角速度命令范围缩小到原始范围的 range_multiplier[0] 倍
    2. 评估性能：每个episode结束时，评估角速度跟踪奖励的平均值
    3. 自适应提升：如果奖励超过80%的最大值，则扩大角速度命令范围
    4. 最终阶段：角速度命令范围达到原始范围的 range_multiplier[1] 倍
    
    Args:
        env: 强化学习环境对象
        env_ids: 需要更新课程的环境ID列表
        reward_term_name: 用于评估性能的奖励项名称（默认是角速度跟踪奖励）
        range_multiplier: 角速度范围乘数 (初始倍数, 最终倍数)
            - 默认 (0.1, 1.0) 表示从10%逐步提升到100%
    
    Returns:
        当前的最大角速度命令绝对值
    
    示例：
        # 在配置中使用：
        curriculum.ang_vel_levels = CurrTerm(
            func=mdp.command_levels_ang_vel,
            params={
                "reward_term_name": "track_ang_vel_z_exp",
                "range_multiplier": (0.2, 1.0)  # 从20%开始，逐步提升到100%
            }
        )
    """
    # 获取基础速度命令的范围配置
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    
    # 【第一步：初始化】仅在第一次调用时（第0步）保存原始角速度范围
    if env.common_step_counter == 0:
        # 保存原始角速度范围的绝对值上界（假设范围是对称的 [-max, +max]）
        original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._original_ang_vel_z_max = torch.abs(original_ang_vel_z[1])  # 取正值上界
        
        # 计算初始和最终的角速度上界
        env._initial_ang_vel_z_max = env._original_ang_vel_z_max * range_multiplier[0]
        env._final_ang_vel_z_max = env._original_ang_vel_z_max * range_multiplier[1]
        
        # 将命令范围初始化为较小的初始值（对称范围）
        initial_max = env._initial_ang_vel_z_max.item()
        base_velocity_ranges.ang_vel_z = (-initial_max, initial_max)
    
    # 【第二步：自适应调整】每个episode结束时检查是否需要增加难度
    if env.common_step_counter % env.max_episode_length == 0:
        # 获取该奖励项在当前episode中的累计值
        episode_sums = env.reward_manager._episode_sums.get(reward_term_name)
        
        # 如果奖励项不存在，直接返回当前值
        if episode_sums is None:
            current_max = base_velocity_ranges.ang_vel_z[1]
            return torch.tensor(current_max, device=env.device)
        
        # 获取奖励项的配置（包含权重）
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        
        # 每次增加的角速度量
        delta_ang_vel = 0.1  # 每次增加 0.1 rad/s
        
        # 【第三步：性能评估】如果跟踪奖励超过最大值的60%，则增加命令范围
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.6 * reward_term_cfg.weight:
            # 获取当前角速度上界
            current_max = base_velocity_ranges.ang_vel_z[1]
            
            # 计算新的角速度上界
            new_max = current_max + delta_ang_vel
            
            # 限制在最终范围内
            new_max = min(new_max, env._final_ang_vel_z_max.item())
            
            # 更新命令范围（对称范围）
            base_velocity_ranges.ang_vel_z = (-new_max, new_max)
    
    # 返回当前的最大角速度命令
    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)
