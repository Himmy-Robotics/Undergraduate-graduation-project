# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
命令生成器模块 - 定义强化学习环境中的命令生成逻辑

本模块提供了两种命令生成器：
1. UniformThresholdVelocityCommand - 带阈值过滤的均匀分布速度命令生成器
2. DiscreteCommandController - 离散命令控制器

命令生成器的作用是为机器人提供目标指令（如期望的运动速度），
强化学习算法需要训练机器人学会跟踪这些命令。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """
    带阈值过滤的均匀分布速度命令生成器

    该类继承自 UniformVelocityCommand，在标准均匀分布速度命令的基础上，
    增加了阈值过滤功能：如果生成的速度命令小于阈值（0.2 m/s），则将其设为零。

    这样可以避免机器人执行过小的速度命令，让训练更加稳定。
    例如：如果生成的命令是 (0.1, 0.15) m/s，L2范数约为0.18 < 0.2，则将其设为 (0, 0)。
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg
    """命令生成器的配置对象"""

    def _resample_command(self, env_ids: Sequence[int]):
        """
        为指定环境重新采样速度命令

        Args:
            env_ids: 需要重新采样命令的环境ID列表

        处理流程：
        1. 调用父类方法生成均匀分布的速度命令
        2. 计算XY平面速度的L2范数（速度大小）
        3. 如果速度大小 < 0.2 m/s，则将该命令置零（避免过小的命令）
        """
        # 调用父类方法，从均匀分布中采样速度命令
        super()._resample_command(env_ids)

        # 对XY平面的速度进行阈值过滤
        # [:, :2] 表示只看x和y方向的速度，忽略z方向的角速度
        # torch.norm计算速度向量的L2范数（速度大小）
        # > 0.2 生成布尔掩码，速度大于0.2的为True，否则为False
        # 乘以这个掩码后，小于0.2的速度被置零
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """带阈值过滤的均匀分布速度命令生成器配置"""

    class_type: type = UniformThresholdVelocityCommand


class DiscreteCommandController(CommandTerm):
    """
    离散命令控制器

    该控制器从预定义的离散命令列表中随机选择命令分配给各个环境。

    使用场景：
    - 当需要机器人学习特定的离散行为时（如：站立、慢走、快跑等）
    - 当命令空间较小且可枚举时

    示例：
        available_commands = [0, 10, 20, 30]  # 分别代表：静止、慢速、中速、快速
        控制器会随机从这4个命令中选择一个分配给每个环境
    """

    cfg: DiscreteCommandControllerCfg
    """命令控制器的配置对象"""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        初始化离散命令控制器

        Args:
            cfg: 命令控制器的配置对象
            env: 强化学习环境对象

        Raises:
            ValueError: 如果 available_commands 为空或包含非整数元素
        """
        # 初始化基类
        super().__init__(cfg, env)

        # 验证 available_commands 非空
        if not self.cfg.available_commands:
            raise ValueError("available_commands 列表不能为空")

        # 确保所有元素都是整数
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("available_commands 中的所有元素必须是整数")

        # 存储可用命令列表
        self.available_commands = self.cfg.available_commands

        # 创建缓冲区来存储命令
        # -- command_buffer: 为每个环境存储当前的离散命令（整数索引）
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: 存储当前命令的快照（整数列表）
        # 默认初始化为第一个可用命令
        self.current_commands = [self.available_commands[0]] * self.num_envs

    def __str__(self) -> str:
        """返回命令控制器的字符串表示（用于调试和日志输出）"""
        return (
            "DiscreteCommandController:\n"
            f"\t环境数量: {self.num_envs}\n"
            f"\t可用命令: {self.available_commands}\n"
        )

    """
    属性（Properties）
    """

    @property
    def command(self) -> torch.Tensor:
        """
        返回当前的命令缓冲区

        Returns:
            torch.Tensor: 命令张量，形状为 (num_envs, 1)，每个元素是整数命令值
        """
        return self.command_buffer

    """
    内部实现函数
    """

    def _update_metrics(self):
        """
        更新命令控制器的指标

        注：当前实现为空，可在子类中重写以添加自定义指标
        """
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """
        为指定环境重新采样命令

        Args:
            env_ids: 需要重新采样命令的环境ID列表

        处理流程：
        1. 随机生成索引（从0到命令数量-1）
        2. 根据索引从 available_commands 中选取命令
        3. 更新对应环境的命令缓冲区
        """
        # 随机生成索引（用于从 available_commands 中选择）
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )

        # 根据索引获取实际的命令值
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices],
            dtype=torch.int32,
            device=self.device
        )

        # 更新指定环境的命令缓冲区
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """
        更新并存储当前命令快照

        将 command_buffer 转换为 Python 列表并存储到 current_commands 中，
        方便后续查询和日志记录。
        """
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """离散命令控制器配置类"""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    可用的离散命令列表，每个元素都是整数

    示例：
        [10, 20, 30, 40, 50]  # 5个离散命令值
        [0, 1, 2]             # 3个离散状态ID

    使用说明：
    - 列表不能为空
    - 所有元素必须是整数
    - 命令会从该列表中随机选择
    """

class UniformAccelerationVelocityCommand(mdp.UniformVelocityCommand):
    """
    带加速度限制的均匀分布速度命令生成器
    
    除了生成目标速度外，还会生成目标加速度。
    命令会从当前值以给定的加速度平滑过渡到目标值。
    """
    cfg: UniformAccelerationVelocityCommandCfg
    
    def __init__(self, cfg: UniformAccelerationVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # 目标速度缓冲区
        self.target_vel_command = torch.zeros_like(self.vel_command_b)
        # 加速度缓冲区
        self.acc_command = torch.zeros_like(self.vel_command_b)
        
    def _resample_command(self, env_ids: Sequence[int]):
        if not self.cfg.use_acceleration:
            # 不使用加速度模式：直接发布目标速度，无需平滑过渡
            super()._resample_command(env_ids)
            self.target_vel_command[env_ids] = self.vel_command_b[env_ids].clone()
            return
        
        # 保存当前命令（作为起始点）
        current_vel = self.vel_command_b[env_ids].clone()
        
        # 调用父类生成新的目标速度（存入 self.vel_command_b）
        super()._resample_command(env_ids)
        
        # 将生成的值移动到 target_vel_command
        self.target_vel_command[env_ids] = self.vel_command_b[env_ids].clone()
        
        # 恢复当前命令（以便平滑过渡）
        self.vel_command_b[env_ids] = current_vel
        
        # 采样加速度
        # X方向
        r = self.cfg.acc_ranges.lin_vel_x
        self.acc_command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(*r)
        # Y方向
        r = self.cfg.acc_ranges.lin_vel_y
        self.acc_command[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(*r)
        # Z方向 (角速度)
        r = self.cfg.acc_ranges.ang_vel_z
        self.acc_command[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(*r)
        
    def compute(self, dt: float):
        # 调用父类处理计时器和重采样
        super().compute(dt)
        
        if not self.cfg.use_acceleration:
            # 不使用加速度模式：速度已在 _resample_command 中直接设置，无需平滑过渡
            return
        
        # 执行平滑过渡 (Ramping)
        # 计算本步最大变化量
        max_change = self.acc_command * dt
        
        # 计算与目标的差值
        diff = self.target_vel_command - self.vel_command_b
        
        # 限制变化量
        change = torch.clamp(diff, min=-max_change, max=max_change)
        
        # 更新当前命令
        self.vel_command_b += change


@configclass
class UniformAccelerationVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """带加速度限制的速度命令配置"""
    class_type: type = UniformAccelerationVelocityCommand
    
    use_acceleration: bool = True
    """是否启用加速度限制。为 True 时速度从当前值按加速度平滑过渡到目标值；
    为 False 时直接发布目标速度，不做平滑过渡。"""
    
    @configclass
    class AccRanges:
        lin_vel_x: tuple[float, float] = (0.0, 0.0)
        lin_vel_y: tuple[float, float] = (0.0, 0.0)
        ang_vel_z: tuple[float, float] = (0.0, 0.0)
        
    acc_ranges: AccRanges = AccRanges()


class PhasedTurnVelocityCommand(mdp.UniformVelocityCommand):
    """
    分阶段转向速度命令生成器
    
    设计思想：
    - 阶段1（加速期）：只有前向速度，角速度为0，让机器人先加速到目标速度
    - 阶段2（转向期）：保持前向速度，施加角速度，测试高速转向能力
    
    这样可以：
    1. 公平评估高速转向能力（机器人已达到稳定速度）
    2. 方便对比有/无脊柱的转向表现
    3. 模拟真实场景中的"先加速后转弯"行为
    """
    cfg: "PhasedTurnVelocityCommandCfg"
    
    def __init__(self, cfg: "PhasedTurnVelocityCommandCfg", env):
        super().__init__(cfg, env)
        # 目标角速度（在阶段2才生效）
        self.target_ang_vel_z = torch.zeros(self.num_envs, device=self.device)
        # 每个环境的阶段计时器
        self.phase_timer = torch.zeros(self.num_envs, device=self.device)
        # 每个环境的转向延迟时间
        self.turn_delay = torch.zeros(self.num_envs, device=self.device)
        # 当前是否在转向阶段
        self.in_turn_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
    def _resample_command(self, env_ids: Sequence[int]):
        """重新采样命令，先只设置线速度，角速度延迟施加"""
        # 调用父类采样（生成 lin_vel_x, lin_vel_y, ang_vel_z）
        super()._resample_command(env_ids)
        
        # 保存目标角速度，但暂时不应用
        self.target_ang_vel_z[env_ids] = self.vel_command_b[env_ids, 2].clone()
        
        # 阶段1：强制角速度为0，让机器人先直线加速
        self.vel_command_b[env_ids, 2] = 0.0
        
        # 采样每个环境的转向延迟时间
        delay_range = self.cfg.turn_delay_range
        self.turn_delay[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            delay_range[0], delay_range[1]
        )
        
        # 重置计时器和阶段标志
        self.phase_timer[env_ids] = 0.0
        self.in_turn_phase[env_ids] = False
        
    def _update_command(self):
        """更新命令，检查是否进入转向阶段"""
        # 更新计时器
        self.phase_timer += self._env.step_dt
        
        # 检查是否有环境进入转向阶段
        enter_turn_phase = (self.phase_timer >= self.turn_delay) & (~self.in_turn_phase)
        
        if enter_turn_phase.any():
            # 进入转向阶段，施加预存的角速度
            self.vel_command_b[enter_turn_phase, 2] = self.target_ang_vel_z[enter_turn_phase]
            self.in_turn_phase[enter_turn_phase] = True


@configclass
class PhasedTurnVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """
    分阶段转向速度命令配置
    
    参数:
        turn_delay_range: 转向延迟时间范围 (min, max)，单位秒
                          在每次重采样时，会从该范围内随机采样一个延迟时间
                          延迟时间过后才会施加角速度命令
    """
    class_type: type = PhasedTurnVelocityCommand
    
    turn_delay_range: tuple[float, float] = (1.0, 3.0)
    """转向延迟时间范围（秒），机器人先直线跑这么久再开始转向"""


class PhasedTurnAccVelocityCommand(UniformAccelerationVelocityCommand):
    """
    分阶段转向 + 加速度限制 的速度命令生成器
    
    结合了两个特性：
    1. 分阶段转向：先直线加速，延迟后再施加角速度
    2. 加速度限制：速度平滑过渡，不会瞬间跳变
    """
    cfg: "PhasedTurnAccVelocityCommandCfg"
    
    def __init__(self, cfg: "PhasedTurnAccVelocityCommandCfg", env):
        super().__init__(cfg, env)
        # 目标角速度（在阶段2才生效）
        self.final_target_ang_vel_z = torch.zeros(self.num_envs, device=self.device)
        # 每个环境的阶段计时器
        self.phase_timer = torch.zeros(self.num_envs, device=self.device)
        # 每个环境的转向延迟时间
        self.turn_delay = torch.zeros(self.num_envs, device=self.device)
        # 当前是否在转向阶段
        self.in_turn_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
    def _resample_command(self, env_ids: Sequence[int]):
        """重新采样命令"""
        # 保存当前命令（作为起始点）
        current_vel = self.vel_command_b[env_ids].clone()
        
        # 调用 UniformVelocityCommand 的采样（跳过 UniformAccelerationVelocityCommand）
        # 这会生成新的 lin_vel_x, lin_vel_y, ang_vel_z
        mdp.UniformVelocityCommand._resample_command(self, env_ids)
        
        # 保存最终目标角速度
        self.final_target_ang_vel_z[env_ids] = self.vel_command_b[env_ids, 2].clone()
        
        # 将线速度目标存入 target_vel_command（用于平滑过渡）
        self.target_vel_command[env_ids, :2] = self.vel_command_b[env_ids, :2].clone()
        # 阶段1：角速度目标为0
        self.target_vel_command[env_ids, 2] = 0.0
        
        # 恢复当前命令（以便平滑过渡）
        self.vel_command_b[env_ids] = current_vel
        
        # 采样加速度
        r = self.cfg.acc_ranges.lin_vel_x
        self.acc_command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(*r)
        r = self.cfg.acc_ranges.lin_vel_y
        self.acc_command[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(*r)
        r = self.cfg.acc_ranges.ang_vel_z
        self.acc_command[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(*r)
        
        # 采样每个环境的转向延迟时间
        delay_range = self.cfg.turn_delay_range
        self.turn_delay[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            delay_range[0], delay_range[1]
        )
        
        # 重置计时器和阶段标志
        self.phase_timer[env_ids] = 0.0
        self.in_turn_phase[env_ids] = False
        
    def compute(self, dt: float):
        """每步计算，处理阶段切换和平滑过渡"""
        # 更新阶段计时器
        self.phase_timer += dt
        
        # 检查是否有环境进入转向阶段
        enter_turn_phase = (self.phase_timer >= self.turn_delay) & (~self.in_turn_phase)
        
        if enter_turn_phase.any():
            # 进入转向阶段，更新目标角速度
            self.target_vel_command[enter_turn_phase, 2] = self.final_target_ang_vel_z[enter_turn_phase]
            self.in_turn_phase[enter_turn_phase] = True
        
        # 调用父类的 compute（处理重采样计时器）
        # 注意：我们需要手动处理平滑过渡，因为父类的 compute 会被覆盖
        # 先调用 CommandTerm 的 compute 处理重采样
        from isaaclab.managers import CommandTerm
        CommandTerm.compute(self, dt)
        
        # 执行平滑过渡 (Ramping)
        max_change = self.acc_command * dt
        diff = self.target_vel_command - self.vel_command_b
        change = torch.clamp(diff, min=-max_change, max=max_change)
        self.vel_command_b += change


@configclass
class PhasedTurnAccVelocityCommandCfg(UniformAccelerationVelocityCommandCfg):
    """
    分阶段转向 + 加速度限制 的速度命令配置
    
    结合了 UniformAccelerationVelocityCommandCfg 和 PhasedTurnVelocityCommandCfg 的特性
    """
    class_type: type = PhasedTurnAccVelocityCommand
    
    turn_delay_range: tuple[float, float] = (1.0, 3.0)
    """转向延迟时间范围（秒），机器人先直线跑这么久再开始转向"""
