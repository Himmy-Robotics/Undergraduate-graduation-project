# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
四足机器人速度追踪强化学习任务的奖励函数模块

本模块定义了所有用于训练四足机器人运动控制策略的奖励函数。
这些函数在每个时间步计算单个或多个环境的奖励值，用于引导学习算法学习期望的行为。

主要奖励类别:
1. 速度追踪奖励 (Velocity Tracking)
   - track_lin_vel_xy_exp: 追踪XY平面线速度
   - track_ang_vel_z_exp: 追踪Z轴角速度（偏航）
   
2. 运动质量奖励 (Motion Quality)
   - joint_power: 关节功率消耗（节能）
   - feet_air_time: 步长长度
   - feet_height_body: 抬腿高度
   - feet_distance_xy_exp: 脚步间距
   
3. 稳定性奖励 (Stability)
   - upward: 保持机器人直立
   - flat_orientation_l2: 保持平衡姿态
   - base_height_l2: 维持目标身高
   
4. 惩罚项 (Penalty Terms)
   - joint_pos_limits: 关节位置限制
   - undesired_contacts: 不期望接触
   - feet_stumble: 脚步不稳
   - lin_vel_z_l2: Z方向速度
   - ang_vel_xy_l2: 侧翻倾向

每个函数返回形状为 (num_envs,) 的张量，表示每个环境的单步奖励值。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    【核心奖励】追踪XY平面线速度命令（指数核函数）
    
    这是四足机器人运动控制最重要的奖励函数之一！
    目标: 让机器人的实际速度尽可能接近给定的目标速度
    
    数学公式:
        error = ||v_cmd - v_actual||²  (L2 范数平方)
        reward = exp(-error / std²)
    
    exp函数的特性:
    - error = 0 (完全匹配) → reward = 1.0 (最大奖励)
    - error 越大 → reward 指数衰减趋向0
    - std 控制衰减速度: std越小，对误差越敏感
    
    附加条件: 只有当机器人大致直立时才给奖励（防止躺着也能拿奖励）
    """
    # 获取机器人对象
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 计算速度误差（命令速度 vs 实际速度）
    # [:, :2] 表示只取 XY 方向，忽略 Z 方向
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    
    # 指数奖励: 误差越小，奖励越接近1
    reward = torch.exp(-lin_vel_error / std**2)
    
    # 乘以直立度系数（重力在Z轴负方向的投影）
    # projected_gravity_b[:, 2] ≈ -9.8 (完全直立) → clamp后为0.7 → 系数为1.0
    # projected_gravity_b[:, 2] ≈ 0 (翻倒) → clamp后为0 → 系数为0（不给奖励）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    【核心奖励】追踪Yaw角速度命令（旋转速度）
    
    目标: 让机器人的旋转速度匹配目标角速度
    例如: 命令要求以0.5 rad/s向左转，机器人实际也要以这个速度转
    
    与线速度追踪类似，但只关注Z轴的角速度（Yaw，偏航角）
    Roll和Pitch的旋转不受控制（会通过其他奖励函数惩罚）
    
    数学公式:
        error = (cmd_yaw - actual_yaw)²
        reward = exp(-error / std²)
    
    参数:
        env: 强化学习环境实例
        std: 高斯核的标准差，控制对角速度误差的敏感度
        command_name: 命令管理器中的命令名称（默认"base_velocity"）
        asset_cfg: 机器人资产配置
        
    返回:
        shape为(num_envs,)的张量，每个值在0~1之间
    """
    # 获取机器人对象
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 计算角速度误差（只看Z轴，即Yaw方向）
    # [:, 2] 表示Z轴的角速度
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    
    # 指数奖励: 误差越小，奖励越接近1
    reward = torch.exp(-ang_vel_error / std**2)
    
    # 同样需要机器人保持直立才给奖励（防止倒地的机器人也能拿旋转奖励）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    【速度追踪】重力对齐的机器人坐标系中的线速度（可选替代方案）
    
    与 track_lin_vel_xy_exp 的区别:
    - track_lin_vel_xy_exp: 在身体坐标系（随机器人旋转）中计算
    - track_lin_vel_xy_yaw_frame_exp: 在"偏航框架"中计算
    
    偏航框架意义:
    - 身体坐标系中去除Yaw旋转的影响
    - Roll和Pitch仍然被保留
    - 用于需要考虑身体倾斜但忽略转向的场景
    
    通常不需要用到这个，使用 track_lin_vel_xy_exp 就可以了
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    【角速度追踪】世界坐标系中的Yaw速度（可选替代方案）
    
    与 track_ang_vel_z_exp 的区别:
    - track_ang_vel_z_exp: 在身体坐标系中计算
    - track_ang_vel_z_world_exp: 在世界坐标系中计算
    
    世界坐标系的优势:
    - 与全局参考框架一致
    - 不受机器人旋转状态的影响
    - 但由于我们通常在身体坐标系下工作，这个版本很少使用
    
    推荐: 使用 track_ang_vel_z_exp
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    【节能惩罚】关节功率消耗
    
    目标: 鼓励机器人以节能的方式运动
    
    物理公式: P = τ × ω
    - P: 功率（W）
    - τ: 扭矩（N·m）
    - ω: 角速度（rad/s）
    
    这个惩罚会让策略学会:
    - 减少不必要的快速运动
    - 避免在高速运动时施加大扭矩
    - 使用更平滑、更高效的运动模式
    
    注意: 这是惩罚项（weight为负数），值越大惩罚越重
    """
    # 获取机器人对象
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 计算所有关节的瞬时功率之和
    # |τ × ω| 的绝对值，避免正负抵消
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    height_threshold: float = 0.6,
    velocity_threshold: float = 0.1,  # 新增: 垂直速度阈值
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    【站立惩罚】当命令接近0、高度较低且不再下落时，惩罚偏离默认姿态
    
    目标: 让机器人在着陆静止后保持标准姿态，且不抑制空中动作
    
    逻辑:
    1. 命令为0 (希望停止)
    2. 高度 < height_threshold (接近地面)
    3. 垂直速度 < velocity_threshold (不再下落，表明已着陆)
    满足以上所有条件时，惩罚关节偏离。
    """
    # 计算关节偏离默认位置的L1距离
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    
    # 1. 检查命令是否静止
    is_still_cmd = torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 2. 检查高度是否较低
    is_low_height = asset.data.root_link_pos_w[:, 2] < height_threshold
    
    # 3. 检查垂直速度是否很小 (排除下落过程)
    # 自由落体时速度通常很大 (>2m/s)，只有触底后才会接近0
    is_landed_velocity = torch.abs(asset.data.root_lin_vel_w[:, 2]) < velocity_threshold

    # 只有同时满足三个条件才惩罚
    reward *= (is_still_cmd * is_low_height * is_landed_velocity)
    
    # 保持直立约束
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """
    【惩罚】关节位置偏离
    
    目标: 让机器人关节远离极端位置，防止关节卡死或损伤
    
    逻辑:
    - 如果机器人在站立状态，给予较重的惩罚（stand_still_scale倍数）
    - 如果机器人在运动，给予较轻的惩罚
    - 这样允许运动时的大幅度摆动，但在站立时保持端正姿态
    
    判断站立vs运动的条件:
    - 命令速度 < command_threshold (0.01)
    - 且 实际速度 < velocity_threshold (0.1)
    
    参数:
        stand_still_scale: 站立时的惩罚倍数（通常0.5-2.0）
        velocity_threshold: 判断运动的速度阈值 (m/s)
        command_threshold: 判断停止命令的阈值 (m/s)
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    
    # 计算关节位置偏离L2范数
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    
    # 根据运动状态选择惩罚权重
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,  # 运动时：较小的惩罚
        stand_still_scale * running_reward,  # 站立时：较大的惩罚
    )
    
    # 仍然需要保持直立
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    【惩罚】轮子/脚速度（空中摆动时的速度）
    
    目标: 控制脚在空中摆动时的速度，避免过度的腿部摆动
    
    物理意义:
    - 脚在空中摆动速度过高 → 增加冲击、消耗能量
    - 脚接触地面时摆动速度应该很低 → 保证接触稳定性
    
    逻辑:
    - 如果机器人在运动中：只惩罚空中脚的速度（in_air * joint_vel）
    - 如果机器人站立：惩罚所有脚的速度（joint_vel）
    
    参数:
        sensor_cfg: 接触传感器配置，用于检测脚是否接触地面
        velocity_threshold: 判断运动状态的速度阈值
        command_threshold: 判断停止命令的阈值
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    
    # 获取所有关节的绝对速度
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    
    # 获取接触传感器，判断哪些脚在空中
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]  # shape: (num_envs, num_feet)
    
    # 运动时：只惩罚空中脚（地面接触脚允许高速）
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    
    # 站立时：惩罚所有脚（防止不必要的摆动）
    standing_reward = torch.sum(joint_vel, dim=1)
    
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,  # 运动状态
        standing_reward,  # 站立状态
    )
    return reward


class GaitReward(ManagerTermBase):
    """
    【高级奖励】步态强制（步幅同步）
    
    目标: 强制四足机器人采用特定的步态模式（如小跑）
    
    四足机器人的常见步态:
    - 小跑 (Trotting): 对角线的腿同时接触地面 (FR+HL vs FL+HR)
    - 跳跃 (Bounding): 前两条腿同时接触，后两条腿同时接触
    - 踱步 (Pacing): 同侧的腿同时接触 (FR+FL vs HR+HL)
    
    实现原理:
    - 定义"同步脚对"（如小跑中的两对对角线腿）
    - 计算两条腿的接触时间差（phase difference）
    - 奖励时间差小的配置（鼓励同步接触）
    
    关键参数:
    - std: 时间差敏感度
    - max_err: 最大允许的时间差
    - velocity_threshold: 只在高速运动时应用此奖励
    
    这个奖励很重要，因为自然的步态能提高运动稳定性和能效!
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        
        # 获取同步脚对的名称配置
        # 例如小跑(Trotting): [['FL', 'HR'], ['FR', 'HL']]
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        
        # 检查配置是否合法：必须是两对脚
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
            
        # 获取脚的索引
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """
        计算步态奖励
        
        逻辑:
        这个奖励由6个部分相乘组成:
        1. 同步奖励 (Sync): 
           - Pair 0 内部两脚同步 (sync_reward_0)
           - Pair 1 内部两脚同步 (sync_reward_1)
        2. 异步奖励 (Async):
           - Pair 0 和 Pair 1 之间应该反相 (互斥)
           - Pair 0 的脚接触地面时，Pair 1 的脚应该在空中
           - 包含4个交叉项 (async_reward_0...3)
           
        最终奖励 = Sync * Async
        只有当所有条件都满足时，奖励才高。只要有一个条件不满足（如顺拐），奖励就会急剧下降。
        """
        # 1. 计算组内同步奖励 (Synchronous Reward)
        # 要求同一组内的两只脚状态一致（同时落地，同时抬起）
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        
        # 2. 计算组间异步奖励 (Asynchronous Reward)
        # 要求不同组的脚状态相反（一组落地时，另一组必须在空中）
        # 交叉对比: Pair0_Foot0 vs Pair1_Foot0, Pair0_Foot0 vs Pair1_Foot1, ...
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        
        # 3. 应用条件
        # 只有当有明显的速度命令或实际速度较快时，才强制步态
        # 静止时不需要踏步
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        
        # 保持直立约束
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """
        【辅助】计算两脚同步奖励
        
        目标: 奖励两只脚状态一致
        原理: 
        - 比较两只脚的空中时间 (air_time) 和接触时间 (contact_time)
        - 差值越小，奖励越高
        """
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        
        # 计算时间差的平方，并进行截断（防止误差过大导致梯度爆炸或数值问题）
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        
        # 指数核函数
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """
        【辅助】计算两脚反相奖励
        
        目标: 奖励两只脚状态相反
        原理:
        - 脚0的空中时间 应该接近 脚1的接触时间
        - 脚0的接触时间 应该接近 脚1的空中时间
        """
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        
        # 交叉比较: Air vs Contact
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        
        # 指数核函数
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


class GallopGaitReward(ManagerTermBase):
    """
    【高级奖励】Gallop 步态奖励（非对称奔跑步态）
    
    目标: 引导四足机器人学习 gallop 步态，这是一种高速非对称步态
    
    Gallop 的特征:
    - 前两脚落地有时间差（leading leg 先落地，trailing leg 后落地）
    - 后两脚落地也有时间差
    - 有完全腾空的阶段
    - 与 bound（前脚完全同步、后脚完全同步）不同
    
    设计思路:
    - 不是奖励"时差越小越好"（那会收敛到 bound）
    - 而是奖励"时差接近目标值"（环形高斯分布）
    - 同时不限制前后脚之间的相位关系（允许完全腾空）
    
    关键参数:
    - target_phase_diff: 目标相位差（秒），gallop 典型值为 0.05~0.15
    - std: 对相位差偏离目标的敏感度
    - allow_zero_diff: 是否允许零时差（False 会惩罚 bound）
    
    数学公式:
        reward = exp(-((actual_diff - target_diff)² / std²))
    
    这样设计确保:
    - 时差 = target_diff → 奖励最高
    - 时差 = 0 (bound) → 奖励较低（除非 target_diff=0）
    - 时差过大 → 奖励也降低
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        
        # 前后脚对可以分别设置不同的目标相位差
        # 如果只设置了 target_phase_diff，则前后脚使用相同值
        # 如果设置了 front_target_phase_diff 和 rear_target_phase_diff，则分别使用
        default_diff = cfg.params.get("target_phase_diff", 0.08)
        self.front_target_phase_diff: float = cfg.params.get("front_target_phase_diff", default_diff)
        self.rear_target_phase_diff: float = cfg.params.get("rear_target_phase_diff", default_diff)
        
        self.max_err: float = cfg.params.get("max_err", 0.3)
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        
        # 获取脚的名称配置
        # 对于 gallop: front_feet = ["FL_foot", "FR_foot"], rear_feet = ["HL_foot", "HR_foot"]
        front_feet_names = cfg.params["front_feet_names"]  # 前两只脚
        rear_feet_names = cfg.params["rear_feet_names"]    # 后两只脚
        
        # 获取脚的索引
        # 分别查找每只脚的索引，确保顺序与输入名称列表一致
        self.front_feet_ids = [
            self.contact_sensor.find_bodies([front_feet_names[0]])[0][0],
            self.contact_sensor.find_bodies([front_feet_names[1]])[0][0]
        ]
        self.rear_feet_ids = [
            self.contact_sensor.find_bodies([rear_feet_names[0]])[0][0],
            self.contact_sensor.find_bodies([rear_feet_names[1]])[0][0]
        ]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        velocity_threshold: float,
        command_threshold: float,
        front_feet_names: list[str],
        rear_feet_names: list[str],
        max_err: float,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        target_phase_diff: float = 0.08,
        front_target_phase_diff: float = None,
        rear_target_phase_diff: float = None,
    ) -> torch.Tensor:
        """
        计算 gallop 步态奖励
        
        核心逻辑:
        1. 计算前两脚之间的相位差，奖励接近 front_target_phase_diff 的值
        2. 计算后两脚之间的相位差，奖励接近 rear_target_phase_diff 的值
        3. 不限制前后脚之间的相位关系（允许完全腾空）
        
        返回:
            shape (num_envs,) 的奖励张量
        """
        # 1. 计算前两脚的相位差奖励（使用前脚专属的 target）
        front_reward = self._phase_diff_reward(
            self.front_feet_ids[0], self.front_feet_ids[1], self.front_target_phase_diff
        )
        
        # 2. 计算后两脚的相位差奖励（使用后脚专属的 target）
        rear_reward = self._phase_diff_reward(
            self.rear_feet_ids[0], self.rear_feet_ids[1], self.rear_target_phase_diff
        )
        
        # 组合奖励（两者都要满足）
        gait_reward = front_reward * rear_reward
        
        # 3. 应用条件：只在有速度命令或实际运动时才强制步态
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            gait_reward,
            0.0,
        )
        
        # 保持直立约束
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    def _phase_diff_reward(self, foot_0: int, foot_1: int, target_phase_diff: float) -> torch.Tensor:
        """
        【辅助】计算两脚相位差奖励（连续过渡版本）
        
        目标: 奖励两只脚的相位差接近指定的 target_phase_diff
        
        核心改进:
        - 解决了原版在状态切换时（一脚在空中，一脚在地上）计时器归零导致的跳变问题
        - 利用 last_air_time / last_contact_time 作为预测器，假设当前周期时长与上周期相近
        - 在四种状态下都能正确计算连续的时间差
        
        四种情况:
        - Case 1: L在地上，R在空中 → offset = L_contact - (R_air - R_last_air)
        - Case 2: L和R都在地上 → offset = L_contact - R_contact
        - Case 3: L在空中，R在地上 → offset = L_air - (R_contact - R_last_contact)
        - Case 4: L和R都在空中 → offset = L_air - R_air
        
        参数:
            foot_0, foot_1: 两只脚的索引（foot_0 为领先脚 L，foot_1 为跟随脚 R）
            target_phase_diff: 目标相位差（秒），正值表示 foot_0 领先
        """
        # 获取各种时间数据
        current_air_time = self.contact_sensor.data.current_air_time
        current_contact_time = self.contact_sensor.data.current_contact_time
        last_air_time = self.contact_sensor.data.last_air_time
        last_contact_time = self.contact_sensor.data.last_contact_time
        
        # 提取两只脚的数据 (L = foot_0, R = foot_1)
        L_air = current_air_time[:, foot_0]
        L_contact = current_contact_time[:, foot_0]
        L_last_air = last_air_time[:, foot_0]
        L_last_contact = last_contact_time[:, foot_0]
        
        R_air = current_air_time[:, foot_1]
        R_contact = current_contact_time[:, foot_1]
        R_last_air = last_air_time[:, foot_1]
        R_last_contact = last_contact_time[:, foot_1]
        
        # 判断脚的状态：contact > 0 表示在地上，否则在空中
        L_on_ground = L_contact > 0
        R_on_ground = R_contact > 0
        
        # Case 1: L在地上，R在空中
        # R预计还要在空中的时间 = R_last_air - R_air（假设本周期与上周期相同）
        # offset = L已落地时间 - R还需腾空时间 的负值 = L_contact - (R_air - R_last_air)
        case1_offset = L_contact - (R_air - R_last_air)
        
        # Case 2: L和R都在地上
        # 直接比较两者的落地时间
        case2_offset = L_contact - R_contact
        
        # Case 3: L在空中，R在地上
        # L预计还要在空中的时间 = L_last_air - L_air
        # offset = L_air - (R_contact - R_last_contact)
        case3_offset = L_air - (R_contact - R_last_contact)
        
        # Case 4: L和R都在空中
        # 直接比较两者的腾空时间
        case4_offset = L_air - R_air
        
        # 使用嵌套的 torch.where 选择正确的 offset
        offset = torch.where(
            L_on_ground,
            torch.where(R_on_ground, case2_offset, case1_offset),  # L在地上: R在地上用case2, R在空中用case1
            torch.where(R_on_ground, case3_offset, case4_offset),  # L在空中: R在地上用case3, R在空中用case4
        )
        
        # 保留符号信息，支持指定领先/落后关系
        # target_phase_diff > 0: foot_0 领先于 foot_1
        # target_phase_diff < 0: foot_1 领先于 foot_0
        
        # 截断到最大误差范围（防止初始化或异常情况下的极大值）
        offset = torch.clip(offset, min=-self.max_err, max=self.max_err)
        
        
        # 计算奖励：环形高斯，时差接近 target 时奖励最高
        # reward = exp(-((offset - target)² / std))
        reward = torch.exp(-torch.square(offset - target_phase_diff) / self.std)
        
        return reward


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    """
    【对称性奖励】促进左右关节对称运动
    
    目标: 确保机器人左右腿保持对称的关节位置
    
    物理意义:
    - 四足机器人应该有左右对称的构造
    - 对称的运动更加稳定和高效
    - 不对称的运动会导致身体转向或倾斜
    
    实现方法:
    - 定义镜像关节对（如左前髋关节 vs 右前髋关节）
    - 计算对应关节位置的差异
    - 奖励差异小的配置
    
    参数:
        mirror_joints: 镜像关节对的列表，如 [["FL_hip", "FR_hip"], ["HL_hip", "HR_hip"]]
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    """
    【动作对称奖励】促进左右命令的对称性
    
    目标: 确保策略网络输出的左右腿控制命令保持对称
    
    与 joint_mirror 的区别:
    - joint_mirror: 奖励实际的关节位置对称（物理结果）
    - action_mirror: 奖励输出命令的对称性（控制信号）
    
    两者都很重要:
    - action_mirror 直接约束策略输出
    - 有助于学习更对称的行为策略
    
    参数:
        mirror_joints: 镜像关节对，如 [["FL_hip", "FR_hip"], ...]
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    """
    【动作同步奖励】促进同组关节命令的同步性
    
    目标: 确保同一组内的关节收到相似的控制命令
    
    使用场景:
    - 同一条腿的多个关节应该协调
    - 或者所有腿的特定类型关节应该同步
    
    例如: 如果 joint_groups = [["FL_hip", "FR_hip", "HL_hip", "HR_hip"]]
    就是要求所有的髋关节命令保持一致
    
    参数:
        joint_groups: 关节分组列表，每组内的关节应该同步
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
    【步态奖励】鼓励长的摆动步长
    
    目标: 确保机器人在摆腿时能够充分抬腿，而不是拖沓地走
    
    物理意义:
    - 脚在空中的时间长 → 腿抬得高
    - 腿抬得高 → 更大的步长、更快的速度
    - 但也不能随意浪费时间（通过threshold限制）
    
    计算方法:
    - 当脚第一次接地时，获取它最后的"空中时间"
    - 如果空中时间 > threshold，就获得奖励
    - 奖励值 = (实际时间 - threshold)
    
    参数:
        threshold: 最小摆腿时间（秒），低于此不给奖励
        sensor_cfg: 接触传感器配置
        command_name: 速度命令名称（只在有命令时才奖励）
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward




def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    【双足步态奖励】鼓励单脚站立的时间
    
    目标: 为双足机器人设计，奖励单脚支撑的稳定性
    
    与四足机器人的区别:
    - feet_air_time: 适用四足（鼓励脚快速摆动）
    - feet_air_time_positive_biped: 适用双足（鼓励单脚支撑）
    
    计算方法:
    - 检查在每个时刻有多少条腿接触地面
    - 只有当恰好1条腿接触时（single_stance=True），才给奖励
    - 奖励 = 该腿的接触时间
    - 限制最大奖励（不超过threshold）
    
    使用场景: 双足/类人机器人（本项目未使用）
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    【惩罚】摆腿时间方差 (增强版)
    
    目标: 确保所有腿的摆腿时间保持一致，避免不对称的步态
    
    改进逻辑 (Anti-Survivor Bias):
    - 原始逻辑只使用 last_air_time (落地时更新)。若脚一直不落地，该值不更新，导致"幸存者偏差"。
    - 新逻辑引入 current_air_time。若脚卡在空中，current_air_time 会持续增长。
    - 取 max(last, current)，确保卡住的脚产生巨大的时间值，从而推高方差，触发强力惩罚。
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取历史记录 (只在状态切换时更新)
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    
    # 获取当前实时值 (随时间持续增长)
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    # 核心修正：如果当前处于异常长的状态（比如卡在空中），使用当前时间覆盖历史记录
    # 这迫使"装死"的脚暴露出巨大的时间差异
    effective_air_time = torch.max(last_air_time, current_air_time)
    effective_contact_time = torch.max(last_contact_time, current_contact_time)

    # 计算方差惩罚
    # 将上限从0.5放宽到1.0，让长时间滞空的惩罚更剧烈
    reward = torch.var(torch.clip(effective_air_time, max=1.0), dim=1) + torch.var(
        torch.clip(effective_contact_time, max=1.0), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    【接触奖励】控制接触脚的数量
    
    目标: 在运动时维持指定数量的脚同时接触地面
    
    使用场景:
    - 小跑步态需要2条脚接触（对角线）
    - 跳跃步态可能需要特定数量脚接触
    
    逻辑:
    - expect_contact_num: 期望同时接触的脚数（通常为2）
    - 当实际接触脚数 != 期望值时，不给奖励
    - 这种约束在运动时有效，站立时不约束
    
    参数:
        expect_contact_num: 期望同时接触地面的脚数（4足机器人通常为2）
        sensor_cfg: 接触传感器配置
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    【站立奖励】鼓励站立时脚接触地面
    
    目标: 当没有移动命令时，奖励所有脚都接触地面
    
    逻辑与 feet_contact 相反:
    - feet_contact: 运动时约束接触脚数
    - feet_contact_without_cmd: 站立时奖励脚接触
    
    参数:
        command_name: 速度命令名称（用于判断是否在站立）
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    【摔跤惩罚】检测脚踢到竖直障碍物
    
    目标: 惩罚机器人脚踢到竖直表面（如墙壁），防止不自然的行为
    
    物理原理:
    - 脚正常接触地面：垂直力大，水平力小
    - 脚踢到墙壁：水平力大，垂直力相对小
    
    检测方法:
    - 计算接触力的水平分量 (xy) 和垂直分量 (z)
    - 如果 水平力 > 4 × 垂直力，判定为踢到竖直面
    - 这样的情况不应该出现
    
    阈值4倍是为了允许一定的斜面接触，但严格禁止垂直碰撞
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    【姿态奖励】控制站距（前后脚的侧向距离）
    
    目标: 确保机器人的脚步位置保持在期望的宽度范围内
    
    四足机器人的脚位置应该遵循:
    - 前后脚对的距离（Y方向）应该一致
    - 左右脚的侧向距离应该等于 stance_width/2
    
    例如: 
    - stance_width = 0.2m
    - 左侧脚应该在 +0.1m
    - 右侧脚应该在 -0.1m
    
    计算:
    - 获取所有脚在身体坐标系中的Y坐标
    - 计算与期望值的差异
    - 使用指数奖励
    
    参数:
        stance_width: 左右脚的总距离 (米)
        std: 敏感度参数（越小对误差越敏感）
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    【姿态奖励】同时控制站距和步长
    
    目标: 精确控制机器人脚的2D位置（前后+左右）
    
    这是 feet_distance_y_exp 的扩展版本:
    - Y方向约束: 左右脚距离 = stance_width
    - X方向约束: 前后脚距离 = stance_length
    
    四足机器人的脚位置应该形成矩形:
    - 前左脚 (FL): (+stance_length/2, +stance_width/2)
    - 前右脚 (FR): (+stance_length/2, -stance_width/2)
    - 后左脚 (HL): (-stance_length/2, +stance_width/2)
    - 后右脚 (HR): (-stance_length/2, -stance_width/2)
    
    参数:
        stance_width: 左右脚的距离 (米)
        stance_length: 前后脚的距离 (米)
        std: 敏感度参数
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """
    【高度奖励】鼓励脚抬到目标高度
    
    目标: 确保机器人在摆腿时充分抬起脚离地面
    
    使用场景:
    - 防止脚拖地（self-collision风险）
    - 确保充足的地隙空间，应对不平坦地形
    - 提高步行稳定性
    
    计算方法:
    - target_height: 期望脚的高度（相对全局坐标系）
    - 只在脚在水平面上有速度时才施加奖励（tanh加权）
    - 这样防止站立时脚也被抬高
    
    参数:
        target_height: 目标脚高度 (米)
        tanh_mult: 用于平滑过渡的乘数
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """
    【高度奖励】身体坐标系中的脚高度控制
    
    目标: 与 feet_height 相同，但在身体坐标系中计算
    
    区别:
    - feet_height: 全局坐标系（绝对高度）
    - feet_height_body: 身体坐标系（相对高度）
    
    身体坐标系优势:
    - 当机器人在斜坡上时，相对高度更有意义
    - 避免全局高度变化的影响
    - 更适应非平坦地形
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    【摩擦惩罚】惩罚脚的滑动
    
    目标: 要求脚接触地面时保持静止，避免滑动
    
    物理意义:
    - 脚与地面接触时应该有足够摩擦力
    - 脚在地面滑动表示：
      1. 关节命令不匹配实际接触 → 浪费能量
      2. 可能会丢失抓地力 → 不稳定
    
    计算方法:
    - 获取脚在身体坐标系中的侧向速度（水平xy分量）
    - 只在脚接触地面时（contacts=True）才施加惩罚
    - 惩罚 = 脚速度 × 接触状态
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_rate_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    【动作变化率惩罚】奖励动作的平滑性

    目标: 减小相邻时间步之间动作输出的差异

    参数:
        asset_cfg: 关节配置，用于指定要惩罚哪些关节的动作
    """
    # 获取动作差异
    action_diff = env.action_manager.action - env.action_manager.prev_action

    # 如果 joint_names 是默认的 ".*"，则惩罚所有动作
    if asset_cfg.joint_names == ".*":
        reward = torch.sum(torch.square(action_diff), dim=1)
    else:
        # 否则，查找指定的关节索引
        asset: Articulation = env.scene[asset_cfg.name]
        joint_indices, _ = asset.find_joints(asset_cfg.joint_names)
        # 计算差异并只取指定关节的部分
        reward = torch.sum(torch.square(action_diff[:, joint_indices]), dim=1)

    # 同样乘以直立度系数，保持奖励风格一致
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    【稳定性奖励】奖励机器人保持直立
    
    目标: 鼓励机器人的躯体竖直向上，防止翻滚或倾斜
    
    计算方法:
    - projected_gravity_b[:, 2]: 重力在身体坐标系Z轴上的投影
    - 机器人直立时，这个值接近 -9.8 (完全倒向重力)
    - projected_gravity_b[:, 2] ≈ -1.0 时表示水平
    
    公式: reward = (1 - projected_gravity_b[:, 2])²
    - 直立（-1）→ (1-(-1))² = 4 → 惩罚2
    - 水平（0）→ (1-0)² = 1 → 惩罚1
    - 倒立（1）→ (1-1)² = 0 → 惩罚0
    
    这个奖励在所有其他奖励中都有使用，确保机器人优先保持直立！
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """
    【高度奖励】控制躯体的绝对高度
    
    目标: 让机器人躯体保持在目标高度
    
    用途:
    - 平坦地形: target_height 是恒定值（如0.35m）
    - 不平坦地形: 使用传感器（height_scanner）动态调整目标
    
    传感器调整的意义:
    - 在斜坡上，地面高度变化
    - 使用ray_hits获取周围地面高度
    - 动态调整target_height以适应地形
    
    参数:
        target_height: 躯体目标高度 (米)
        sensor_cfg: 高度扫描仪配置（可选，用于不平坦地形）
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    【惩罚】垂直速度
    
    目标: 防止机器人向上或向下运动（只允许水平运动）
    
    原理:
    - 四足机器人的控制通常只考虑水平运动（XY平面）
    - 垂直运动（Z轴）应该被最小化
    - 垂直跳跃不受欢迎（浪费能量）
    - 下沉到地面也不行（会碰撞）
    
    计算: reward = (v_z)²
    - v_z: 躯体在身体坐标系中的Z轴速度
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    【惩罚】侧翻和俯仰角速度
    
    目标: 防止机器人绕X和Y轴旋转（翻滚和俯仰）
    
    原理:
    - 机器人应该只绕Z轴旋转（偏航，Yaw，转向）
    - X轴旋转（Roll，翻滚）= 机器人要倒下
    - Y轴旋转（Pitch，俯仰）= 机器人要后倾/前倾
    
    这两个旋转都很危险，应该严格禁止！
    
    计算: reward = ω_x² + ω_y²
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    【接触惩罚】检测不期望的接触
    
    目标: 惩罚躯体或其他非脚部位与地面接触
    
    使用场景:
    - 躯体接触地面 → 机器人太低/摔倒 → 惩罚
    - 腿部中点接触地面 → 自碰撞 → 惩罚
    - 只有脚应该接触地面（由feet_contact函数管理）
    
    参数:
        threshold: 接触力的阈值（N）
        sensor_cfg: 接触传感器配置，定义要监测的部位
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    【稳定性奖励】惩罚非水平方向的倾斜
    
    目标: 确保机器人不会侧倾（除了Yaw偏航以外）
    
    与 ang_vel_xy_l2 的区别:
    - ang_vel_xy_l2: 惩罚旋转的快速变化（角速度）
    - flat_orientation_l2: 惩罚倾斜的静态位置（方向）
    
    计算方法:
    - projected_gravity_b[:, :2]: 重力在身体XY方向的投影
    - 机器人直立时，这个值应该接近 [0, 0]
    - 如果有倾斜，这个值就变大
    
    例如:
    - 侧倾30°: projected_gravity_b[:, 0] ≈ -5 → 惩罚25
    - 前倾30°: projected_gravity_b[:, 1] ≈ -5 → 惩罚25
    
    这是保持稳定的关键奖励！
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward





# 只奖励落地时机
# def track_gait_pattern(
#     env: ManagerBasedRLEnv,
#     command_name: str,
#     cycle_time: float,
#     touchdown_phase: list[float],
#     ratios: list[float],
#     sensor_cfg: SceneEntityCfg,
#     velocity_threshold: float = 0.0,
#     min_cycle_time: float = 0.25,
#     cycle_time_slope: float = 0.0,
# ) -> torch.Tensor:
#     """
#     【落地时机奖励】只奖励在正确相位窗口内的落地瞬间
    
#     目标: 引导机器人在正确的时机落地，但不约束支撑时长和抬脚时机
    
#     奖励逻辑:
#     - 当脚在 [touchdown_phase, touchdown_phase + ratio] 相位区间内从腾空变为接触（落地瞬间）时，给予奖励
#     - 其他所有时间（持续接触、持续腾空、离地、错误时机落地）不给奖励
    
#     这是一个稀疏奖励，只在关键时刻（正确的落地瞬间）提供反馈
    
#     参数:
#         cycle_time: 基础步态周期时间（秒），当速度为0时的周期
#         touchdown_phase: 每个脚的开始踏地时刻 (Touchdown Phase) [0, 1)
#                  touchdown_phase直接代表该脚在哪个相位时刻开始允许接触地面
#         ratios: 每个脚的允许落地窗口长度 [0, 1)，
#                 在 [touchdown_phase, touchdown_phase + ratio] 区间内落地可获得奖励
#         sensor_cfg: 接触传感器配置
#         velocity_threshold: 只有当速度命令大于此阈值时才计算奖励
#         min_cycle_time: 最小允许的周期时间（秒），防止周期过短
#         cycle_time_slope: 周期随速度变化的斜率。实际周期 = cycle_time - slope * velocity
#     """
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
#     # 获取当前时间
#     current_time = env.episode_length_buf * env.step_dt
    
#     # 获取速度命令大小
#     cmd_vel = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    
#     # 计算动态周期时间
#     # T = T_base - slope * v
#     target_cycle_time = cycle_time - cycle_time_slope * cmd_vel
#     # 限制最小周期
#     target_cycle_time = torch.clamp(target_cycle_time, min=min_cycle_time)
    
#     # 计算当前周期相位 [0, 1)
#     phase = (current_time.unsqueeze(1) % target_cycle_time.unsqueeze(1)) / target_cycle_time.unsqueeze(1)
#     # phase shape: (num_envs, 1)
    
#     # 转换参数为tensor
#     touchdown_phase_tensor = torch.tensor(touchdown_phase, device=env.device).unsqueeze(0)  # (1, num_feet)
#     ratios_tensor = torch.tensor(ratios, device=env.device).unsqueeze(0)    # (1, num_feet)
    
#     # 计算每个脚的局部相位
#     # 期望: 当 global_phase == touchdown_phase 时，local_phase = 0
#     foot_phases = (phase - touchdown_phase_tensor) % 1.0
    
#     # 判断当前是否在允许落地的相位窗口内 [0, ratio)
#     in_touchdown_window = (foot_phases < ratios_tensor).float()
    
#     # 检测落地瞬间（腾空→接触的那一帧）
#     # compute_first_contact: 只有在脚刚从腾空变为接触的那一帧返回True
#     first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].float()
    
#     # 只有在正确窗口内落地才给奖励
#     # 其他所有情况（持续接触、持续腾空、离地、错误时机落地）奖励为0
#     reward = torch.sum(in_touchdown_window * first_contact, dim=1)
    
#     # 只有在有速度命令时才应用此奖励
#     reward *= (cmd_vel > velocity_threshold).float()
    
#     return reward


# # # 奖励持续的接触状态匹配
# def track_gait_pattern(
#     env: ManagerBasedRLEnv,
#     command_name: str,
#     cycle_time: float,
#     touchdown_phase: list[float],
#     ratios: list[float],
#     sensor_cfg: SceneEntityCfg,
#     velocity_threshold: float = 0.0,
#     min_cycle_time: float = 0.25,
#     cycle_time_slope: float = 0.0,
#     contact_threshold: float = 1.0,
# ) -> torch.Tensor:
#     """
#     【步态模式奖励】基于实际接触状态强制执行特定的步态相位和占空比
    
#     目标: 引导机器人学习特定的步态模式（如Rotary Gallop）
    
#     参数:
#         cycle_time: 基础步态周期时间（秒），当速度为0时的周期
#         touchdown_phase: 每个脚的开始踏地时刻 (Touchdown Phase) [0, 1)。
#                  touchdown_phase直接代表该脚在哪个相位时刻开始接触地面。
#         ratios: 每个脚的占空比（支撑相占比） [0, 1)，长度应等于脚的数量。
#         sensor_cfg: 接触传感器配置
#         velocity_threshold: 只有当速度命令大于此阈值时才计算奖励
#         min_cycle_time: 最小允许的周期时间（秒），防止周期过短
#         cycle_time_slope: 周期随速度变化的斜率。实际周期 = cycle_time - slope * velocity
#         contact_threshold: 判断接触的力阈值 (N)
#     """
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
#     # 获取当前时间
#     current_time = env.episode_length_buf * env.step_dt
    
#     # 获取速度命令大小
#     cmd_vel = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    
#     # 计算动态周期时间
#     # T = T_base - slope * v
#     target_cycle_time = cycle_time - cycle_time_slope * cmd_vel
#     # 限制最小周期
#     target_cycle_time = torch.clamp(target_cycle_time, min=min_cycle_time)
    
#     # 计算当前周期相位 [0, 1)
#     phase = (current_time.unsqueeze(1) % target_cycle_time.unsqueeze(1)) / target_cycle_time.unsqueeze(1)
#     # phase shape: (num_envs, 1)
    
#     # 转换参数为tensor
#     touchdown_phase_tensor = torch.tensor(touchdown_phase, device=env.device).unsqueeze(0)  # (1, num_feet)
#     ratios_tensor = torch.tensor(ratios, device=env.device).unsqueeze(0)    # (1, num_feet)
    
#     # 计算每个脚的局部相位
#     # 期望: 当 global_phase == touchdown_phase 时，local_phase = 0
#     foot_phases = (phase - touchdown_phase_tensor) % 1.0
    
#     # 确定期望的接触状态 (1=接触, 0=腾空)
#     # 如果 foot_phase < ratio，则应该接触
#     desired_contact = (foot_phases < ratios_tensor).float()
    
#     # 获取实际接触状态 - 使用接触力判断
#     # net_forces_w_history: (num_envs, history_len, num_bodies, 3)
#     contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
#     # 取历史中的最大力（防止漏检）
#     contact_force_norm = torch.norm(contact_forces, dim=-1).max(dim=1)[0]  # (num_envs, num_feet)
#     # 判断是否接触
#     actual_contact = (contact_force_norm > contact_threshold).float()
    
#     # 计算匹配度 (相同为1，不同为0)
#     # match = 1 - |desired - actual|
#     match = 1.0 - torch.abs(desired_contact - actual_contact)
    
#     # 对所有脚求和
#     reward = torch.sum(match, dim=1)
    
#     # 只有在有速度命令时才应用此奖励
#     reward *= (cmd_vel > velocity_threshold).float()
    
#     return reward


# 中间态不奖励
def track_gait_pattern(
    env: ManagerBasedRLEnv,
    command_name: str,
    cycle_time: float,
    touchdown_phase: list[float],
    ratios: list[float],
    sensor_cfg: SceneEntityCfg,
    velocity_threshold: float = 0.0,
    min_cycle_time: float = 0.25,
    cycle_time_slope: float = 0.0,
) -> torch.Tensor:
    """
    【步态模式奖励】强制执行特定的步态相位和占空比
    
    目标: 引导机器人学习特定的步态模式（如Rotary Gallop）
    
    参数:
        cycle_time: 基础步态周期时间（秒），当速度为0时的周期
        touchdown_phase: 每个脚的开始踏地时刻 (Touchdown Phase) [0, 1)。
                 注意：这修改了原始逻辑，现在touchdown_phase直接代表该脚在哪个相位时刻开始接触地面。
        ratios: 每个脚的占空比（支撑相占比） [0, 1)，长度应等于脚的数量。
        sensor_cfg: 接触传感器配置
        velocity_threshold: 只有当速度命令大于此阈值时才计算奖励
        min_cycle_time: 最小允许的周期时间（秒），防止周期过短
        cycle_time_slope: 周期随速度变化的斜率。实际周期 = cycle_time - slope * velocity
                          如果为0，则使用固定周期。
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取当前时间
    current_time = env.episode_length_buf * env.step_dt
    
    # 获取速度命令大小
    cmd_vel = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    
    # 计算动态周期时间
    # T = T_base - slope * v
    target_cycle_time = cycle_time - cycle_time_slope * cmd_vel
    # 限制最小周期
    target_cycle_time = torch.clamp(target_cycle_time, min=min_cycle_time)
    
    # 计算当前周期相位 [0, 1)
    # 注意：这种无状态的相位计算在速度变化时会有相位跳变，但在训练中通常可以接受
    phase = (current_time.unsqueeze(1) % target_cycle_time.unsqueeze(1)) / target_cycle_time.unsqueeze(1)
    # phase shape: (num_envs, 1)
    
    # 转换参数为tensor
    touchdown_phase_tensor = torch.tensor(touchdown_phase, device=env.device).unsqueeze(0)  # (1, num_feet)
    ratios_tensor = torch.tensor(ratios, device=env.device).unsqueeze(0)    # (1, num_feet)
    
    # 计算每个脚的局部相位
    # 期望: 当 global_phase == touchdown_phase 时，local_phase = 0
    foot_phases = (phase - touchdown_phase_tensor) % 1.0
    
    # 确定期望的接触状态 (1=接触, 0=腾空)
    # 如果 phase < ratio，则应该接触
    desired_contact = (foot_phases < ratios_tensor).float()
    
    # 获取实际接触状态
    # compute_first_contact 返回的是本步是否发生过接触
    actual_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].float()
    
    # 计算匹配度 (相同为1，不同为0)
    # match = 1 - |desired - actual|
    match = 1.0 - torch.abs(desired_contact - actual_contact)
    
    # 对所有脚求和或求平均
    reward = torch.sum(match, dim=1)
    
    # 只有在有速度命令时才应用此奖励
    reward *= (cmd_vel > velocity_threshold).float()
    
    return reward







# def spine_motivation(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     spine_joint_name: str = "pitch_spine_joint",
#     left_thigh_joint_name: str = "FL_thigh_joint",
#     right_thigh_joint_name: str = "FR_thigh_joint",
#     sigma: float = 100.0,
#     flexion_threshold: float = 0.70,
#     extension_threshold: float = 0.35,
#     boost_weight: float = 1.5
# ) -> torch.Tensor:
#     """
#     【脊柱-前腿相位协同奖励】奖励脊柱 Pitch 关节与前腿大腿关节在相位上的协同运动，并激励更大的运动幅度。
    
#     逻辑：
#     1. 相位协同：通过角速度乘积判断脊柱与前腿的运动方向是否协调
#     2. 状态判断：根据两条前腿的速度方向判断脊柱应该处于什么状态（收缩/伸展/中间态）
#     3. 幅度激励：奖励与预期状态匹配的大幅度运动，超过阈值后封顶
    
#     参数:
#         sigma (float): 缩放系数，用于相位协同评分。默认 100.0。
#         flexion_threshold (float): 收缩状态下的幅度目标阈值（弧度）。默认 0.70 (约40度)。
#         extension_threshold (float): 伸展状态下的幅度目标阈值（弧度）。默认 0.35 (约20度)。
#         boost_weight (float): 幅度激励的权重系数。默认 1.0。
#     """
#     asset: Articulation = env.scene[asset_cfg.name]
    
#     # 获取关节索引
#     spine_idx, _ = asset.find_joints(spine_joint_name)
#     left_thigh_idx, _ = asset.find_joints(left_thigh_joint_name)
#     right_thigh_idx, _ = asset.find_joints(right_thigh_joint_name)
    
#     # 确保找到了关节
#     if len(spine_idx) == 0 or len(left_thigh_idx) == 0 or len(right_thigh_idx) == 0:
#         return torch.zeros(env.num_envs, device=env.device)
        
#     # 获取关节速度和位置
#     spine_vel = asset.data.joint_vel[:, spine_idx[0]]
#     spine_pos = asset.data.joint_pos[:, spine_idx[0]]
#     left_thigh_vel = asset.data.joint_vel[:, left_thigh_idx[0]]
#     right_thigh_vel = asset.data.joint_vel[:, right_thigh_idx[0]]
    
#     # ========== 1. 相位协同评分 ==========
#     product_left = spine_vel * left_thigh_vel
#     product_right = spine_vel * right_thigh_vel
    
#     score_left = -torch.tanh(product_left * sigma)
#     score_right = -torch.tanh(product_right * sigma)
    
#     raw_phase_score = score_left + score_right
    
#     # ========== 2. 判断脊柱应该的状态（向量化） ==========
#     # 两腿都向前 → 脊柱应收缩（spine_state = -1）
#     # 两腿都向后 → 脊柱应伸展（spine_state = +1）
#     # 其他情况 → 中间态（spine_state = 0）
    
#     both_forward = (left_thigh_vel > 0) & (right_thigh_vel > 0)
#     both_backward = (left_thigh_vel < 0) & (right_thigh_vel < 0)
    
#     # 三段式赋值（相当于 if-elif-else）
#     spine_state = torch.zeros(env.num_envs, device=env.device)  # 默认：中间态
#     spine_state[both_forward] = -1.0   # if: 两腿向前 → 应收缩
#     spine_state[both_backward] = 1.0   # elif: 两腿向后 → 应伸展
    
#     # ========== 3. 幅度激励机制 ==========
#     # 计算匹配度：spine_state * spine_pos
#     # 收缩时(state=-1)配合负角度(pos<0) → 正奖励
#     # 伸展时(state=+1)配合正角度(pos>0) → 正奖励
#     # 不匹配或中间态 → 0 或负值
#     amplitude_reward = spine_state * spine_pos
    
#     # 根据 spine_state 选择对应的阈值
#     # 注意：这里用 spine_state 而不是 spine_pos
#     thresholds = torch.where(
#         spine_state < 0,                        # 收缩状态
#         torch.tensor(flexion_threshold, device=env.device), 
#         torch.where(
#             spine_state > 0,                    # 伸展状态
#             torch.tensor(extension_threshold, device=env.device),
#             torch.tensor(1.0, device=env.device)  # 中间态（不会被用到）
#         )
#     )
    
#     # 截断：超过阈值后不再增加
#     clamped_amplitude = torch.clamp(amplitude_reward, min=-thresholds, max=thresholds)
    
    
#     # 中间态时幅度奖励为 0
#     clamped_amplitude = torch.where(
#         spine_state == 0, 
#         torch.zeros_like(clamped_amplitude), 
#         clamped_amplitude
#     )
    

#     # 只使用正向激励（匹配时才 boost）
#     # positive_boost = torch.clamp(clamped_amplitude, min=0.0)
    
#     # ========== 4. 计算最终奖励 ==========
#     final_reward = raw_phase_score * (1.0 + boost_weight * clamped_amplitude)
    
#     return final_reward






def spine_motivation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    spine_joint_name: str = "pitch_spine_joint",
    leg_mode: str = "all",
    sigma: float = 100.0,
    flexion_threshold: float = 0.70,
    extension_threshold: float = 0.2,
    boost_weight: float = 1.5,
    velocity_deadzone: float = 0.0,
    excess_penalty_weight: float = 2.0,
    velocity_bias: float = 0
) -> torch.Tensor:
    """
    【脊柱-腿部相位协同奖励】奖励脊柱 Pitch 关节与腿部大腿关节在相位上的协同运动，并激励更大的运动幅度。
    
    支持三种腿部判断模式：
    - "rear" (后腿模式): 后腿向后蹬(vel>0) → 脊柱伸展; 后腿向前收(vel<0) → 脊柱收缩
    - "front" (前腿模式): 前腿向前伸(vel<0) → 脊柱伸展; 前腿向后收(vel>0) → 脊柱收缩
    - "all" (全腿模式): 同时使用前后腿，将前腿速度取反后与后腿速度平均，平衡前后腿各自造成的偏置
    
    逻辑：
    1. 相位协同：通过角速度乘积判断脊柱与腿部的运动方向是否协调
    2. 状态判断：根据腿部的**平均速度**判断脊柱应该处于什么状态（收缩/伸展/中间态）
       - 使用平均速度而非要求两腿同向，允许腿部有自然的相位差
    3. 幅度激励：奖励与预期状态匹配的大幅度运动，超过阈值后封顶
    4. 超限惩罚：当脊柱角度超过阈值时，施加软惩罚防止过度运动
    
    参数:
        leg_mode (str): 腿部判断模式。"rear"=后腿, "front"=前腿, "all"=全部四条腿。默认 "rear"。
        sigma (float): 缩放系数，用于相位协同评分。默认 100.0。
        flexion_threshold (float): 收缩状态下的幅度目标阈值（弧度）。默认 0.70 (约40度)。
        extension_threshold (float): 伸展状态下的幅度目标阈值（弧度）。默认 0.35 (约20度)。
        boost_weight (float): 幅度激励的权重系数。默认 1.5。
        velocity_deadzone (float): 平均速度的死区阈值（rad/s）。|avg_vel| < deadzone 时为中间态。默认 0.0。
        excess_penalty_weight (float): 超限惩罚权重。默认 2.0。超出阈值的平方乘以此权重作为惩罚。
        velocity_bias (float): 速度偏置值（rad/s）。从effective_vel中减去此值，用于调整收缩/伸展的时间占比。
            正值：增加收缩时间，减少伸展时间。默认 0.0。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取脊柱关节索引
    spine_idx, _ = asset.find_joints(spine_joint_name)
    if len(spine_idx) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    spine_vel = asset.data.joint_vel[:, spine_idx[0]]
    spine_pos = asset.data.joint_pos[:, spine_idx[0]]
    
    # 根据模式获取腿部关节
    if leg_mode == "rear":
        # 后腿模式
        left_idx, _ = asset.find_joints("RL_thigh_joint")
        right_idx, _ = asset.find_joints("RR_thigh_joint")
        if len(left_idx) == 0 or len(right_idx) == 0:
            return torch.zeros(env.num_envs, device=env.device)
        left_vel = asset.data.joint_vel[:, left_idx[0]]
        right_vel = asset.data.joint_vel[:, right_idx[0]]
        # 后腿：直接使用速度
        effective_vel = (left_vel + right_vel) / 2.0
        biased_vel = effective_vel - velocity_bias
        # 相位评分：脊柱与后腿同向为协同
        phase_product = spine_vel * biased_vel
        
    elif leg_mode == "front":
        # 前腿模式
        left_idx, _ = asset.find_joints("FL_thigh_joint")
        right_idx, _ = asset.find_joints("FR_thigh_joint")
        if len(left_idx) == 0 or len(right_idx) == 0:
            return torch.zeros(env.num_envs, device=env.device)
        left_vel = asset.data.joint_vel[:, left_idx[0]]
        right_vel = asset.data.joint_vel[:, right_idx[0]]
        # 前腿：速度取反（前腿vel<0表示向前伸→对应脊柱伸展）
        effective_vel = -(left_vel + right_vel) / 2.0
        biased_vel = effective_vel - velocity_bias
        # 相位评分：脊柱与前腿反向为协同
        phase_product = spine_vel * biased_vel
        
    elif leg_mode == "all":
        # 全腿模式：同时使用前后腿
        rl_idx, _ = asset.find_joints("RL_thigh_joint")
        rr_idx, _ = asset.find_joints("RR_thigh_joint")
        fl_idx, _ = asset.find_joints("FL_thigh_joint")
        fr_idx, _ = asset.find_joints("FR_thigh_joint")
        if len(rl_idx) == 0 or len(rr_idx) == 0 or len(fl_idx) == 0 or len(fr_idx) == 0:
            return torch.zeros(env.num_envs, device=env.device)
        
        rl_vel = asset.data.joint_vel[:, rl_idx[0]]
        rr_vel = asset.data.joint_vel[:, rr_idx[0]]
        fl_vel = asset.data.joint_vel[:, fl_idx[0]]
        fr_vel = asset.data.joint_vel[:, fr_idx[0]]
        
        # 后腿速度直接用，前腿速度取反，然后平均
        # 这样 effective_vel > 0 统一表示脊柱应伸展
        rear_avg = (rl_vel + rr_vel) / 2.0
        front_avg = -(fl_vel + fr_vel) / 2.0  # 前腿取反
        effective_vel = (rear_avg + front_avg) / 2.0
        biased_vel = effective_vel - velocity_bias

        # 相位评分：后腿同向+前腿反向
        phase_product = spine_vel * biased_vel
    else:
        raise ValueError(f"Unknown leg_mode: {leg_mode}. Use 'rear', 'front', or 'all'.")
    
    # ========== 1. 相位协同评分 ==========
    raw_phase_score = torch.tanh(phase_product * sigma) * 2.0  # 乘2保持与原来两腿评分的数量级一致
    
    # ========== 2. 判断脊柱应该的状态（基于有效速度） ==========
    # 应用速度偏置：从effective_vel中减去bias，使得判断阈值整体偏移
    # velocity_bias > 0 时，更容易进入收缩状态（effective_vel更容易 < -deadzone）

    
    # biased_vel > deadzone → 脊柱应伸展 (spine_state = +1)
    # biased_vel < -deadzone → 脊柱应收缩 (spine_state = -1)
    # |biased_vel| < deadzone → 中间态 (spine_state = 0)
    
    spine_state = torch.where(
        biased_vel > velocity_deadzone,
        torch.tensor(1.0, device=env.device),  # 脊柱应伸展
        torch.where(
            biased_vel < -velocity_deadzone,
            torch.tensor(-1.0, device=env.device),  # 脊柱应收缩
            torch.tensor(0.0, device=env.device)   # 中间态
        )
    )
    
    # ========== 3. 幅度激励机制 ==========
    # 计算匹配度：spine_state * spine_pos
    # 收缩时(state=-1)配合负角度(pos<0) → 正奖励
    # 伸展时(state=+1)配合正角度(pos>0) → 正奖励
    # 不匹配或中间态 → 0 或负值
    amplitude_reward = spine_state * spine_pos
    
    # 根据 spine_state 选择对应的阈值
    # 注意：这里用 spine_state 而不是 spine_pos
    thresholds = torch.where(
        spine_state < 0,                        # 收缩状态
        torch.tensor(flexion_threshold, device=env.device), 
        torch.where(
            spine_state > 0,                    # 伸展状态
            torch.tensor(extension_threshold, device=env.device),
            torch.tensor(1.0, device=env.device)  # 中间态（不会被用到）
        )
    )
    
    # 截断：超过阈值后不再增加
    clamped_amplitude = torch.clamp(amplitude_reward, min=-thresholds, max=thresholds)
    
    
    # 中间态时幅度奖励为 0
    clamped_amplitude = torch.where(
        spine_state == 0, 
        torch.zeros_like(clamped_amplitude), 
        clamped_amplitude
    )
    

    # 只使用正向激励（匹配时才 boost）
    # positive_boost = torch.clamp(clamped_amplitude, min=0.0)
    
    # ========== 4. 超限惩罚机制 ==========
    # 当脊柱角度超过阈值时，施加软惩罚防止过度运动
    # excess = max(0, |spine_pos| - threshold) （只有方向正确时才计算）
    # 方向正确时 amplitude_reward > 0，此时计算超限
    excess = torch.clamp(amplitude_reward - thresholds, min=0.0)
    # 中间态时不惩罚
    excess = torch.where(spine_state == 0, torch.zeros_like(excess), excess)
    # 软惩罚：超出部分的平方
    excess_penalty = excess_penalty_weight * torch.square(excess)
    
    # ========== 5. 计算最终奖励 ==========
    final_reward = raw_phase_score * (1 + boost_weight * clamped_amplitude) - excess_penalty
    
    return final_reward


















def feet_height_body_upper_limit(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """
    【高度惩罚】身体坐标系中的足部高度上限限制（单向惩罚）
    
    作用：
    只惩罚足部高度高于 target_height（即过于接近身体或穿模）的情况。
    如果足部低于 target_height（在正常工作空间内），则不惩罚。
    
    参数：
    - target_height: 高度上限（例如 -0.10m）。Z > -0.10 会受到惩罚。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # 1. 计算足部在世界坐标系下的位置和速度（相对于基座）
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    
    # 2. 将相对位置和速度转换到基座（身体）局部坐标系
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    
    # 3. 计算单向误差：只保留 Z > target_height 的部分
    # difference > 0 表示足部太高了（超过了上限）
    difference = footpos_in_body_frame[:, :, 2] - target_height
    difference = torch.clamp(difference, min=0.0)
    
    foot_z_target_error = torch.square(difference).view(env.num_envs, -1)
    
    # 4. 直接惩罚高度误差，不使用速度加权
    # 这是一个安全约束（防止踢到身体），无论速度如何，只要位置越界就应该惩罚
    reward = torch.sum(foot_z_target_error, dim=1)
    
    # 5. 仅当有运动指令时生效 (防止静止状态下的误判，虽然对于上限惩罚通常不需要，但保持一致性)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    
    # 6. 处理重力投影（保持一致性）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    
    return reward


    

def yaw_spine_turn_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    spine_joint_name: str = "yaw_spine_joint",
    scale: float = 3.5,
    cmd_threshold: float = 0.3,
) -> torch.Tensor:
    """
    【Yaw脊柱转向对齐奖励】激励 yaw 脊柱关节朝着与角速度命令相应的方向转动
    
    设计思路：
    1. 转向命令越大，需要的脊柱偏转角度越大
    2. 方向正确时给予正奖励，方向错误时给予负奖励
    3. 使用 tanh 软饱和，防止过度追求极限转角
    
    公式：
        ratio = -spine_pos / cmd_yaw  （方向正确时为正）
        reward = tanh(ratio * scale)
    
    符号约定：
        - cmd_yaw > 0: 逆时针/左转
        - cmd_yaw < 0: 顺时针/右转
        - spine_pos > 0: 头向右偏
        - spine_pos < 0: 头向左偏
        - 命令左转时，脊柱应左偏（cmd>0, pos<0）→ ratio > 0 ✓
        - 命令右转时，脊柱应右偏（cmd<0, pos>0）→ ratio > 0 ✓
    
    参数:
        command_name (str): 速度命令的名称，用于获取 yaw 角速度命令
        spine_joint_name (str): yaw 脊柱关节名称。默认 "yaw_spine_joint"
        scale (float): 饱和速度控制参数。默认 2.0
            - scale 越大，小转角就能饱和
            - scale 越小，需要更大转角才能饱和
        cmd_threshold (float): 角速度命令的死区阈值（rad/s）。默认 0.3
            - |cmd_yaw| < threshold 时，奖励为 0（直行不需要脊柱偏转）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取 yaw 角速度命令（通常是 command 的第 3 个分量，索引为 2）
    cmd_yaw = env.command_manager.get_command(command_name)[:, 2]
    
    # 获取 yaw 脊柱关节位置
    spine_idx, _ = asset.find_joints(spine_joint_name)
    if len(spine_idx) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    spine_pos = asset.data.joint_pos[:, spine_idx[0]]
    


    # 计算命令幅度
    cmd_abs = torch.abs(cmd_yaw)
    
    # 当 |cmd| > threshold 时计算奖励，否则为 0
    # 使用 clamp 避免除零，同时在小命令时平滑过渡
    safe_cmd = torch.clamp(cmd_abs, min=cmd_threshold)
    
    # 计算比例：-spine_pos / cmd_yaw
    # 方向正确时 ratio > 0，方向错误时 ratio < 0
    ratio = -spine_pos / (torch.sign(cmd_yaw) * safe_cmd + 1e-8)
    
    # 软饱和映射
    raw_reward = torch.tanh(ratio * scale)
    
    # 命令小于阈值时，奖励置零（直行状态不激励脊柱偏转）
    active_mask = (cmd_abs > cmd_threshold).float()
    reward = raw_reward * active_mask
    
    return reward
