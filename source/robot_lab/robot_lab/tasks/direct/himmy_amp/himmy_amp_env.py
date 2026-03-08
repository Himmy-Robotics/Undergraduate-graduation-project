from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane, UsdFileCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_angle_axis
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Define ARROW_CFG locally since it's not available in isaaclab.markers.config
ARROW_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Arrow",
    markers={
        "arrow": UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        ),
    },
)

from .himmy_amp_env_cfg import HimmyAmpEnvCfg



def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower



class HimmyAmpEnv(DirectRLEnv):
    """
    Himmy Mark 2 机器人的 AMP (Adversarial Motion Priors) 环境。
    
    该环境实现了基于 AMP 的模仿学习任务，用于训练四足机器人模仿参考动作。
    它继承自 DirectRLEnv，直接在 GPU 上进行物理模拟和强化学习计算。
    """
    cfg: HimmyAmpEnvCfg

    def __init__(self, cfg: HimmyAmpEnvCfg, render_mode: str | None = None, **kwargs):
        """
        初始化环境。
        
        Args:
            cfg: 环境配置对象。
            render_mode: 渲染模式。
        """
        super().__init__(cfg, render_mode, **kwargs)

        # 动作偏移和缩放
        # 使用默认关节位置作为偏移
        self.action_offset = self.robot.data.default_joint_pos[0].clone()
        
        # 设置动作缩放
        # Hip joints: 0.15, Others: 0.25
        self.action_scale = torch.ones_like(self.action_offset) * 0.25
        
        for joint_name in self.robot.data.joint_names:
            if "hip_joint" in joint_name:
                idx = self.robot.data.joint_names.index(joint_name)
                self.action_scale[idx] = 0.15

        # 定义关键身体部位名称 (用于计算模仿奖励和观测)
        # 对于四足机器人，通常关注四个脚的位置
        # 顺序：FL, FR, RL, RR (与 temp_amp_repo 数据集 reorder_from_pybullet_to_isaac 转换后一致)

        key_body_names = [
            "FL_foot_Link",
            "FR_foot_Link",
            "RL_foot_Link",
            "RR_foot_Link",
        ]
        
        # 定义腿部关节名称 (用于动作映射和观测切片)
        # 重要：顺序必须与 temp_amp_repo 数据集一致
        # 数据集原始顺序是 PyBullet (FR, FL, RR, RL)，经过 reorder_from_pybullet_to_isaac 转换后变为 (FL, FR, RL, RR)
        self.leg_joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        # 获取腿部关节索引
        self.leg_joint_indexes = [self.robot.data.joint_names.index(name) for name in self.leg_joint_names]
        
        # 定义脊柱关节名称 (Himmy Mark2 特有)
        # 顺序: yaw, pitch, roll (与 mocap_motions_himmy 数据格式一致)
        self.spine_joint_names = [
            "yaw_spine_joint",   # 脊柱偏航 (左右转)
            "pitch_spine_joint", # 脊柱俯仰 (前后弯)
            "roll_spine_joint",  # 脊柱横滚 (左右倾)
        ]
        # 获取脊柱关节索引
        self.spine_joint_indexes = [self.robot.data.joint_names.index(name) for name in self.spine_joint_names]
        
        # 合并所有控制关节索引 (腿部 + 脊柱)
        # 动作顺序: 12 个腿部关节 + 3 个脊柱关节
        self.all_joint_indexes = self.leg_joint_indexes + self.spine_joint_indexes

        # 获取身体部位在 Isaac Sim 中的索引
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        
        # 获取非脚部身体索引 (用于 undesired_contacts 惩罚)
        # 这些是我们不希望接触地面的身体部位 (躯干、大腿、小腿)
        self.undesired_contact_body_indexes = []
        for name in self.cfg.undesired_contact_body_names:
            if name in self.robot.data.body_names:
                self.undesired_contact_body_indexes.append(self.robot.data.body_names.index(name))
            else:
                print(f"Warning: Body '{name}' not found in robot body names, skipping.")
        
        # 重新配置 AMP 观测空间
        # AMP 需要历史观测数据，这里根据配置创建缓冲区
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )
        
        # 指令缓冲区 (lin_vel_x, lin_vel_y, ang_vel_yaw)
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_timer = torch.zeros(self.num_envs, device=self.device)

        # Action tracking for action_rate penalty
        self.last_actions = torch.zeros(self.num_envs, cfg.num_actions, device=self.device)
        
        # =============================================================================
        # 课程学习初始化 (Curriculum Learning)
        # =============================================================================
        if self.cfg.curriculum_enabled:
            # 保存原始速度范围 (用于课程学习计算)
            self._original_vel_x = torch.tensor(
                [self.cfg.commands.lin_vel_x_range[0], self.cfg.commands.lin_vel_x_range[1]], 
                device=self.device
            )
            self._original_vel_y = torch.tensor(
                [self.cfg.commands.lin_vel_y_range[0], self.cfg.commands.lin_vel_y_range[1]], 
                device=self.device
            )
            
            # 计算初始和最终速度范围
            self._initial_vel_x = self._original_vel_x * self.cfg.curriculum_initial_multiplier
            self._final_vel_x = self._original_vel_x * self.cfg.curriculum_final_multiplier
            self._initial_vel_y = self._original_vel_y * self.cfg.curriculum_initial_multiplier
            self._final_vel_y = self._original_vel_y * self.cfg.curriculum_final_multiplier
            
            # 当前速度范围 (从初始值开始)
            self._current_vel_x = self._initial_vel_x.clone()
            self._current_vel_y = self._initial_vel_y.clone()
            
            # 课程进度 (0.0 ~ 1.0)
            self._curriculum_progress = torch.tensor(0.0, device=self.device)
            
            # 累计奖励用于课程评估 (所有环境)
            self._curriculum_reward_sum = torch.zeros(self.num_envs, device=self.device)
        
        # 全局步数计数器 (用于固定间隔评估课程学习)
        self.common_step_counter = 0

        # Logging
        self.episode_sums = {
            "rew_lin_vel_xy": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "rew_ang_vel_z": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "rew_action_rate": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "rew_base_height": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "rew_undesired_contacts": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),  # 非脚部接触惩罚
            "rew_termination": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "rew_action_l2": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "rew_joint_pos_limits": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "rew_joint_acc_l2": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "rew_joint_vel_l2": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "rew_spine_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),  # 脊柱惩罚
            "total_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }
        
        # 初始化 extras 字典（基类会在 step 中使用）
        self.extras = {}

    def _setup_scene(self):
        """设置仿真场景，包括机器人、地面和光照。"""
        self.robot = Articulation(self.cfg.robot)
        
        # Debug Visualization
        if self.cfg.debug_vis:
            # Command Arrows (Green)
            cmd_cfg = ARROW_CFG.copy()
            cmd_cfg.prim_path = "/Visuals/CommandArrows"
            cmd_cfg.markers["arrow"].scale = (2.5, 2.5, 2.5)
            cmd_cfg.markers["arrow"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
            self.command_vis = VisualizationMarkers(cmd_cfg)
            
            # Velocity Arrows (Blue)
            vel_cfg = ARROW_CFG.copy()
            vel_cfg.prim_path = "/Visuals/VelocityArrows"
            vel_cfg.markers["arrow"].scale = (2.5, 2.5, 2.5)
            vel_cfg.markers["arrow"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
            self.velocity_vis = VisualizationMarkers(vel_cfg)

        # 添加地面平面
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # 克隆环境 (用于并行训练)
        self.scene.clone_environments(copy_from_source=False)
        # CPU 模拟时需要显式过滤碰撞
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # 将机器人添加到场景管理器
        self.scene.articulations["robot"] = self.robot
        # 添加光照
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, rew, terminated, truncated, extras = super().step(actions)
        
        if self.cfg.debug_vis:
            self._update_debug_vis()
            
        return obs, rew, terminated, truncated, extras

    def _update_debug_vis(self):
        # Base position
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        
        # 1. Command Visualization (Green)
        # Commands are (lin_vel_x, lin_vel_y, ang_vel_yaw) in base frame
        cmd_vel_b = self.commands[:, :2]
        cmd_speed = torch.norm(cmd_vel_b, dim=1)
        
        # Calculate orientation in world frame
        cmd_vel_b_3d = torch.cat([cmd_vel_b, torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        cmd_vel_w = quat_apply(root_quat, cmd_vel_b_3d)
        cmd_yaw_w = torch.atan2(cmd_vel_w[:, 1], cmd_vel_w[:, 0])
        cmd_quat_w = quat_from_angle_axis(cmd_yaw_w, torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1))
        
        # Position: 0.5m above base
        cmd_pos = root_pos.clone()
        cmd_pos[:, 2] += 0.5
        
        # Scale: (speed, 0.1, 0.1)
        cmd_scale = torch.zeros(self.num_envs, 3, device=self.device)
        cmd_scale[:, 0] = cmd_speed * 0.3
        cmd_scale[:, 1] = 0.1
        cmd_scale[:, 2] = 0.1
        
        self.command_vis.visualize(cmd_pos, cmd_quat_w, cmd_scale)
        
        # 2. Velocity Visualization (Blue)
        # Actual velocity in world frame
        lin_vel_w = self.robot.data.root_lin_vel_w
        lin_speed = torch.norm(lin_vel_w[:, :2], dim=1)
        
        vel_yaw_w = torch.atan2(lin_vel_w[:, 1], lin_vel_w[:, 0])
        vel_quat_w = quat_from_angle_axis(vel_yaw_w, torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1))
        
        # Position: 0.5m above base
        vel_pos = root_pos.clone()
        vel_pos[:, 2] += 0.5
        
        vel_scale = torch.zeros(self.num_envs, 3, device=self.device)
        vel_scale[:, 0] = lin_speed * 0.3
        vel_scale[:, 1] = 0.1
        vel_scale[:, 2] = 0.1
        
        self.velocity_vis.visualize(vel_pos, vel_quat_w, vel_scale)

    def _pre_physics_step(self, actions: torch.Tensor):
        """物理步进前的处理，主要是保存动作。"""
        # Save last actions for action_rate penalty
        self.last_actions = self.actions.clone()
        self.actions = actions.clone()
        
        # 更新全局步数计数器
        self.common_step_counter += 1
        
        # 更新指令计时器并重采样指令
        self.command_timer += self.cfg.sim.dt * self.cfg.decimation
        reset_env_ids = (self.command_timer >= self.cfg.commands.resampling_time).nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            self._resample_commands(reset_env_ids)
            self.command_timer[reset_env_ids] = 0.0

    def _resample_commands(self, env_ids: torch.Tensor):
        """
        重采样速度指令。
        
        如果启用了课程学习，使用当前课程的速度范围。
        """
        if self.cfg.curriculum_enabled:
            # 使用课程学习的速度范围
            vel_x_range = self._current_vel_x
            vel_y_range = self._current_vel_y
        else:
            # 使用配置中的默认范围
            vel_x_range = torch.tensor(
                [self.cfg.commands.lin_vel_x_range[0], self.cfg.commands.lin_vel_x_range[1]], 
                device=self.device
            )
            vel_y_range = torch.tensor(
                [self.cfg.commands.lin_vel_y_range[0], self.cfg.commands.lin_vel_y_range[1]], 
                device=self.device
            )
        
        self.commands[env_ids, 0] = torch_rand_float(
            vel_x_range[0].item(), vel_x_range[1].item(), (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(
            vel_y_range[0].item(), vel_y_range[1].item(), (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands[env_ids, 2] = torch_rand_float(
            self.cfg.commands.ang_vel_yaw_range[0], self.cfg.commands.ang_vel_yaw_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()

    def _apply_action(self):
        """
        应用动作到机器人。这里将归一化的动作转换为关节位置目标。
        
        Himmy Mark2 的动作空间:
        - 前 12 维: 腿部关节 (FL/FR/RL/RR × hip/thigh/calf)
        - 后 3 维: 脊柱关节 (yaw, pitch, roll)
        """
        # self.actions 是 (N, 15) 的完整动作
        # 获取默认关节位置
        targets = self.robot.data.default_joint_pos.clone()
        
        # 1. 计算腿部关节的目标位置
        leg_offset = self.action_offset[self.leg_joint_indexes]
        leg_scale = self.action_scale[self.leg_joint_indexes]
        leg_actions = self.actions[:, :12]  # 前 12 维是腿部动作
        leg_targets = leg_offset + leg_scale * leg_actions
        targets[:, self.leg_joint_indexes] = leg_targets
        
        # 2. 计算脊柱关节的目标位置
        # 脊柱使用相同的缩放逻辑，但默认位置为 0
        spine_offset = self.action_offset[self.spine_joint_indexes]
        spine_scale = self.action_scale[self.spine_joint_indexes]
        spine_actions = self.actions[:, 12:15]  # 后 3 维是脊柱动作
        spine_targets = spine_offset + spine_scale * spine_actions
        targets[:, self.spine_joint_indexes] = spine_targets
        
        self.robot.set_joint_position_target(targets)

    def _get_observations(self) -> dict:
        """
        获取环境观测。
        
        Returns:
            dict: 包含策略观测 ("policy") 和 AMP 观测 ("amp_obs") 的字典。
            
        Himmy Mark2 观测空间:
        - Policy Obs (51 dim): gravity(3) + cmd(3) + dof_pos(15) + dof_vel(15) + actions(15)
        - AMP Obs (36 dim): joint_pos(15) + lin_vel(3) + ang_vel(3) + joint_vel(15)
        """
        # 构建任务观测

        # 计算进度: 当前 episode 步数 / 最大步数, shape [num_envs, 1]
        progress = (self.episode_length_buf.squeeze(-1).float() / (self.max_episode_length - 1)).unsqueeze(-1)
        # 转换为相对坐标，保持与参考动作观测一致
        # 根节点相对位置 (相对于环境原点)
        root_pos_relative = self.robot.data.body_pos_w[:, self.ref_body_index] - self.scene.env_origins
        # 关键部位相对位置
        key_body_pos_relative = self.robot.data.body_pos_w[:, self.key_body_indexes] - self.scene.env_origins.unsqueeze(
            1
        )
        
        # 获取所有控制关节的位置和速度 (腿部 + 脊柱)
        all_joint_pos = self.robot.data.joint_pos[:, self.all_joint_indexes]
        all_joint_vel = self.robot.data.joint_vel[:, self.all_joint_indexes]
        all_default_pos = self.robot.data.default_joint_pos[:, self.all_joint_indexes]
        
        # 计算基础观测向量 (包含脊柱)
        obs = compute_obs(
            self.robot.data.projected_gravity_b,
            self.commands,
            all_joint_pos - all_default_pos,  # 相对于默认位置的偏移
            all_joint_vel,
            self.actions,
        )
        
        # 获取 root 状态
        root_pos = self.robot.data.root_pos_w  # [num_envs, 3]
        root_rot = self.robot.data.root_quat_w  # [num_envs, 4] (wxyz)
        # Isaac Lab 使用 wxyz 格式，需要转换为 xyzw 格式
        root_rot_xyzw = torch.cat([root_rot[:, 1:4], root_rot[:, 0:1]], dim=-1)  # [num_envs, 4] (xyzw)
        
        # 获取世界坐标系下的速度（需要从body坐标系转换）
        # root_lin_vel_b 是 body frame，需要转换到 world frame
        root_lin_vel_w = quat_apply(root_rot, self.robot.data.root_lin_vel_b)  # [num_envs, 3]
        root_ang_vel_w = quat_apply(root_rot, self.robot.data.root_ang_vel_b)  # [num_envs, 3]
        
        # 获取 4 个脚的世界位置
        key_body_pos = self.robot.data.body_pos_w[:, self.key_body_indexes]  # [num_envs, 4, 3]
        
        # AMP 观测（与 MetalHead A1 AMP 对齐）
        amp_obs_single = compute_amp_obs(
            all_joint_pos,
            all_joint_vel,
            root_pos,
            root_rot_xyzw,
            root_lin_vel_w,
            root_ang_vel_w,
            key_body_pos,
        )

        # 更新 AMP 观测历史缓冲区
        # 将旧的观测向后移动，腾出第一个位置给最新观测
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # 存入最新观测
        self.amp_observation_buffer[:, 0] = amp_obs_single.clone()
        # 将 AMP 观测展平并存入 extras，供判别器使用
        # 注意：使用 update 而不是覆盖整个字典，以保留 _reset_idx 中设置的 "episode" 日志信息
        self.extras["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)

        return {"policy": obs}

    def get_amp_observations(self):
        return self.amp_observation_buffer.view(-1, self.amp_observation_size)

    def _get_rewards(self) -> torch.Tensor:
        """
        计算奖励。
        
        奖励由两部分组成：
        1. 速度跟踪奖励 (Velocity Tracking Reward): 鼓励机器人跟随指令。
        2. 基础奖励 (Basic Reward): 存活、平滑度等。
        注意：AMP 风格奖励由 PPO 算法中的判别器计算，不在这里显式计算。
        """
        
        # 1. 线速度跟踪奖励 (xy 平面)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.cfg.rew_lin_vel_xy
        
        # 2. 角速度跟踪奖励 (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.cfg.rew_ang_vel_z
        
        # 3. 动作变化率惩罚 (Action Rate)
        rew_action_rate = torch.sum(torch.square(self.actions - self.last_actions), dim=1) * self.cfg.rew_action_rate
        
        # 4. 基座高度奖励 (Base Height)
        # 使用配置中的目标高度，鼓励机器人保持正确的站立高度
        # 这个奖励很重要，因为我们从 AMP 观测中移除了 root height
        # 如果没有这个奖励，机器人可能会以任意高度移动
        base_height_error = torch.square(self.robot.data.root_pos_w[:, 2] - self.cfg.target_base_height)
        rew_base_height = torch.exp(-base_height_error / 0.02) * self.cfg.rew_base_height
        
        # 5. 脊柱运动惩罚 (Spine Penalty) - 已禁用
        # 注释原因: AMP 需要脊柱自由运动来模仿动作捕捉数据
        # spine_vel = self.robot.data.joint_vel[:, self.spine_joint_indexes]
        # rew_spine_penalty = -0.001 * torch.sum(torch.square(spine_vel), dim=1)
        rew_spine_penalty = torch.zeros(self.num_envs, device=self.device)

        # 6. 非脚部接触惩罚 (Undesired Contacts Penalty)
        # 检测躯干、大腿、小腿等非脚部身体部位是否接触地面
        # 通过检查身体部位的高度是否低于阈值来判断接触
        # 这可以防止机器人坐下、跪地或侧躺
        if len(self.undesired_contact_body_indexes) > 0 and self.cfg.rew_undesired_contacts != 0.0:
            # 获取非脚部身体部位的 z 坐标 (高度)
            # body_pos_w shape: (num_envs, num_bodies, 3)
            undesired_body_heights = self.robot.data.body_pos_w[:, self.undesired_contact_body_indexes, 2]
            
            # 检测哪些身体部位低于阈值 (被视为接触地面)
            # is_contact shape: (num_envs, num_undesired_bodies)
            is_contact = undesired_body_heights < self.cfg.undesired_contact_threshold
            
            # 统计每个环境中接触地面的身体部位数量
            # num_contacts shape: (num_envs,)
            num_contacts = torch.sum(is_contact, dim=1).float()
            
            # 计算惩罚 (接触的身体部位越多，惩罚越大)
            # 同时考虑机器人的倾斜程度 (倾斜越大，惩罚越有效)
            # projected_gravity_b[:, 2] < 0 表示机器人正常站立 (z轴向下)
            # 当机器人倒下时，这个值会接近 0
            gravity_factor = torch.clamp(-self.robot.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
            rew_undesired_contacts = num_contacts * gravity_factor * self.cfg.rew_undesired_contacts
        else:
            rew_undesired_contacts = torch.zeros(self.num_envs, device=self.device)

        # ================= basic reward (基础奖励) ==========================
        # 调用 JIT 编译的 compute_rewards 函数计算基础奖励
        # 注意：使用所有控制关节（腿部 + 脊柱）
        basic_reward, basic_reward_log = compute_rewards(
            self.cfg.rew_termination,
            self.cfg.rew_action_l2,
            self.cfg.rew_joint_pos_limits,
            self.cfg.rew_joint_acc_l2,
            self.cfg.rew_joint_vel_l2,
            self.reset_terminated,
            self.actions,
            self.robot.data.joint_pos[:, self.all_joint_indexes],
            self.robot.data.soft_joint_pos_limits[:, self.all_joint_indexes],
            self.robot.data.joint_acc[:, self.all_joint_indexes],
            self.robot.data.joint_vel[:, self.all_joint_indexes],
        )

        # ================= total reward (总奖励) ==========================
        # 注意: rew_spine_penalty 已禁用 (设为0)，保留在公式中以便将来可以重新启用
        total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_action_rate + rew_base_height + rew_undesired_contacts + basic_reward
        
        # ================= 课程学习奖励累计 (Curriculum Learning) ==========================
        if self.cfg.curriculum_enabled:
            # 累计用于课程评估的奖励 (所有环境)
            self._curriculum_reward_sum += rew_lin_vel_xy
            
            # 每 max_episode_length 步评估一次课程学习
            # 避免频繁更新，因为速度命令范围是所有环境共享的
            if self.common_step_counter % self.max_episode_length == 0:
                self._update_curriculum_global()

        # Update episode sums
        self.episode_sums["rew_lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["rew_ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["rew_action_rate"] += rew_action_rate
        self.episode_sums["rew_base_height"] += rew_base_height
        self.episode_sums["rew_undesired_contacts"] += rew_undesired_contacts
        self.episode_sums["rew_termination"] += basic_reward_log["pub_termination"]
        self.episode_sums["rew_action_l2"] += basic_reward_log["pub_action_l2"]
        self.episode_sums["rew_joint_pos_limits"] += basic_reward_log["pub_joint_pos_limits"]
        self.episode_sums["rew_joint_acc_l2"] += basic_reward_log["pub_joint_acc_l2"]
        self.episode_sums["rew_joint_vel_l2"] += basic_reward_log["pub_joint_vel_l2"]
        self.episode_sums["rew_spine_penalty"] += rew_spine_penalty
        self.episode_sums["total_reward"] += total_reward

        # ============== log (日志记录) ================================
        log_dict = {
            "rew_lin_vel_xy": rew_lin_vel_xy.mean(),
            "rew_ang_vel_z": rew_ang_vel_z.mean(),
            "rew_action_rate": rew_action_rate.mean(),
            "rew_base_height": rew_base_height.mean(),
            "rew_undesired_contacts": rew_undesired_contacts.mean(),
            "rew_spine_penalty": rew_spine_penalty.mean(),
            "total_reward": total_reward.mean(),
        }
        
        # 添加基础奖励日志
        for key, value in basic_reward_log.items():
            if isinstance(value, torch.Tensor):
                log_dict[key] = value.mean()
            else:
                log_dict[key] = float(value)

        self.extras["log"] = log_dict

        # 如果存在 skrl agent，直接记录到 TensorBoard
        if hasattr(self, "_skrl_agent") and getattr(self, "_skrl_agent", None) is not None:
            try:
                agent = getattr(self, "_skrl_agent")
                for k, v in log_dict.items():
                    agent.track_data(f"Reward / {k}", v)
            except Exception:
                pass

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        判断 episode 是否结束。
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (died, time_out)
            died: 机器人是否死亡（例如倒地）。
            time_out: 是否达到最大步数。
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            # 如果启用了提前终止，当基座高度低于阈值时判定为死亡
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """
        重置指定环境。
        
        Args:
            env_ids: 需要重置的环境索引。如果为 None，则重置所有环境。
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        
        # 注意：课程学习的累计奖励不在这里重置
        # 而是在全局评估 (_update_curriculum_global) 后统一重置
        # 这样可以确保累计值代表完整的评估周期
        
        # Logging: Fill extras with episode sums for reset environments
        if len(env_ids) > 0:
            episode_log = {}
            for key, value in self.episode_sums.items():
                # Calculate mean for the reset environments
                episode_log[key] = value[env_ids].mean().item()
                # Reset the sums for these environments
                self.episode_sums[key][env_ids] = 0.0
            self.extras["episode"] = episode_log
            
            # 课程学习日志 (与 manager_based 一致，在 reset 时记录)
            # 使用 Curriculum/ 前缀，与 isaaclab 的 CurriculumManager 保持一致
            if self.cfg.curriculum_enabled:
                if "log" not in self.extras:
                    self.extras["log"] = {}
                self.extras["log"]["Curriculum/progress"] = self._curriculum_progress.item()
                self.extras["log"]["Curriculum/current_vel_x_max"] = self._current_vel_x[1].item()
                self.extras["log"]["Curriculum/current_vel_y_max"] = self._current_vel_y[1].item()

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # 重置指令
        self._resample_commands(env_ids)
        self.command_timer[env_ids] = 0.0

        # 根据配置选择重置策略
        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        # 将重置后的状态写入仿真器
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    
    def _update_curriculum_global(self):
        """
        全局课程学习更新（每 max_episode_length 步调用一次）。
        
        采用 MetalHead 风格的课程学习机制：
        1. 固定时间间隔评估（每 max_episode_length 步）
        2. 使用所有环境的平均每步奖励
        3. 判断条件：平均每步奖励 > 0.8 × 权重
        """
        # 计算所有环境的平均累计奖励
        mean_episode_sum = self._curriculum_reward_sum.mean()
        
        # 计算平均每步奖励
        # 注意：这里用的是步数而不是秒数
        avg_reward_per_step = mean_episode_sum / self.max_episode_length
        
        # 获取奖励权重
        reward_weight = self.cfg.rew_lin_vel_xy
        
        # 判断条件：平均每步奖励 > 0.8 × 权重
        # 这与 MetalHead 中的逻辑一致
        threshold = self.cfg.curriculum_reward_threshold * reward_weight
        
        if avg_reward_per_step > threshold:
            # 增加速度范围
            delta = self.cfg.curriculum_delta_command
            
            # 更新 x 方向速度范围（分别处理下界和上界）
            new_vel_x_min = self._current_vel_x[0] - delta
            new_vel_x_max = self._current_vel_x[1] + delta
            # 下界不能低于最终下界，上界不能高于最终上界
            self._current_vel_x[0] = torch.clamp(new_vel_x_min, min=self._final_vel_x[0])
            self._current_vel_x[1] = torch.clamp(new_vel_x_max, max=self._final_vel_x[1])
            
            # 更新 y 方向速度范围（分别处理下界和上界）
            new_vel_y_min = self._current_vel_y[0] - delta
            new_vel_y_max = self._current_vel_y[1] + delta
            self._current_vel_y[0] = torch.clamp(new_vel_y_min, min=self._final_vel_y[0])
            self._current_vel_y[1] = torch.clamp(new_vel_y_max, max=self._final_vel_y[1])
            
            # 打印升级信息（调试用）
            print(f"[Curriculum] 难度提升！avg_reward/step: {avg_reward_per_step:.3f}, threshold: {threshold:.3f}")
            print(f"[Curriculum] 新速度范围 x: [{self._current_vel_x[0]:.2f}, {self._current_vel_x[1]:.2f}], y: [{self._current_vel_y[0]:.2f}, {self._current_vel_y[1]:.2f}]")
        
        # 计算课程进度（基于速度范围的扩展程度）
        # progress = 0 表示初始状态，progress = 1 表示达到最终难度
        initial_range = self._initial_vel_x[1] - self._initial_vel_x[0]
        final_range = self._final_vel_x[1] - self._final_vel_x[0]
        current_range = self._current_vel_x[1] - self._current_vel_x[0]
        
        if final_range > initial_range:
            self._curriculum_progress = (current_range - initial_range) / (final_range - initial_range)
            self._curriculum_progress = torch.clamp(self._curriculum_progress, 0.0, 1.0)
        else:
            self._curriculum_progress = torch.tensor(1.0, device=self.device)
        
        # 重置所有环境的累计奖励（为下一轮评估做准备）
        self._curriculum_reward_sum[:] = 0.0

    # reset strategies (重置策略)

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """默认重置策略：重置到初始姿态。"""
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        随机重置策略：从参考动作中随机采样一个状态作为初始状态。
        
        Args:
            env_ids: 环境索引。
            start: 如果为 True，则从动作的起始时刻开始；否则随机采样时刻。
        """
        # 暂时回退到默认重置策略，因为 MotionLoader 已被移除
        return self._reset_strategy_default(env_ids)

    # env methods

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        """
        收集参考动作数据，用于构建 AMP 观测。
        
        Args:
            num_samples: 采样数量。
            current_times: 当前时间。如果为 None，则随机采样。
            
        Returns:
            torch.Tensor: AMP 观测数据。
        """
        return torch.zeros((num_samples, self.amp_observation_size), device=self.device)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为切线和法线向量表示。
    这通常用于表示根节点的朝向，比直接使用四元数更适合神经网络输入。
    """
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_toe_pos_local_a1(
    joint_pos: torch.Tensor,
) -> torch.Tensor:
    """
    使用 A1 机器人的腿长参数计算脚的局部坐标。
    
    这个函数使用 A1 的运动学参数来计算脚位置，即使是在 Himmy 机器人上，
    这样可以确保 AMP 观测与 A1 数据集兼容。
    
    A1 运动学参数:
    - Hip offset Y: ±0.0838m (FL/RL: +, FR/RR: -)
    - Thigh length: 0.2m
    - Calf length: 0.2m
    
    Args:
        joint_pos: 关节位置 [num_envs, 12]，顺序为 FL(3), FR(3), RL(3), RR(3)
    
    Returns:
        toe_pos_local: 脚的局部坐标 [num_envs, 4, 3]
    """
    num_envs = joint_pos.shape[0]
    device = joint_pos.device
    
    # A1 运动学参数
    HIP_OFFSET_Y = 0.0838  # m
    THIGH_LENGTH = 0.2     # m
    CALF_LENGTH = 0.2      # m
    
    # Hip offsets for each leg (FL, FR, RL, RR)
    # FL: +y, FR: -y, RL: +y, RR: -y
    hip_offset_signs = torch.tensor([1.0, -1.0, 1.0, -1.0], device=device)
    
    # 分离各腿的关节角度
    # joint_pos: [num_envs, 12] -> 4 legs x 3 joints
    joint_pos_reshaped = joint_pos.view(num_envs, 4, 3)  # [num_envs, 4_legs, 3_joints]
    
    hip_angles = joint_pos_reshaped[:, :, 0]    # [num_envs, 4]
    thigh_angles = joint_pos_reshaped[:, :, 1]  # [num_envs, 4]
    calf_angles = joint_pos_reshaped[:, :, 2]   # [num_envs, 4]
    
    # 正向运动学计算
    # Hip 旋转是绕 X 轴，影响 Y 和 Z
    # Thigh 和 Calf 旋转是绕 Y 轴，影响 X 和 Z
    
    # 1. Hip 贡献 (绕 X 轴旋转)
    # Hip offset 在 Y 方向
    hip_y = hip_offset_signs.unsqueeze(0) * HIP_OFFSET_Y * torch.cos(hip_angles)
    hip_z = -hip_offset_signs.unsqueeze(0) * HIP_OFFSET_Y * torch.sin(hip_angles)
    
    # 2. Thigh 贡献 (绕 Y 轴旋转, 长度向下 -Z)
    # 在 hip frame 下计算
    thigh_x = THIGH_LENGTH * torch.sin(thigh_angles)
    thigh_z_local = -THIGH_LENGTH * torch.cos(thigh_angles)
    
    # 考虑 hip 旋转对 thigh 的影响
    thigh_y = thigh_z_local * torch.sin(hip_angles)
    thigh_z = thigh_z_local * torch.cos(hip_angles)
    
    # 3. Calf 贡献 (绕 Y 轴旋转, 相对于 thigh)
    # Calf 角度是相对于 thigh 的
    total_leg_angle = thigh_angles + calf_angles
    calf_x = CALF_LENGTH * torch.sin(total_leg_angle) - THIGH_LENGTH * torch.sin(thigh_angles)
    calf_z_local = -CALF_LENGTH * torch.cos(total_leg_angle) + THIGH_LENGTH * torch.cos(thigh_angles)
    
    # 考虑 hip 旋转对 calf 的影响
    calf_y = calf_z_local * torch.sin(hip_angles)
    calf_z = calf_z_local * torch.cos(hip_angles)
    
    # 总的脚位置 = hip_offset + thigh + calf
    toe_x = thigh_x + calf_x
    toe_y = hip_y + thigh_y + calf_y
    toe_z = hip_z + thigh_z + calf_z
    
    # 组合成 [num_envs, 4, 3]
    toe_pos = torch.stack([toe_x, toe_y, toe_z], dim=-1)
    
    return toe_pos


@torch.jit.script
def my_quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """使用四元数旋转向量 (q: xyzw 格式)"""
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_to_tan_norm(q: torch.Tensor) -> torch.Tensor:
    """将四元数转换为切向量和法向量表示 (6维)"""
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = my_quat_rotate(q, ref_tan)

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = my_quat_rotate(q, ref_norm)

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan


@torch.jit.script
def calc_heading(q: torch.Tensor) -> torch.Tensor:
    """计算四元数的朝向角 (yaw)"""
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = my_quat_rotate(q, ref_dir)
    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading


@torch.jit.script
def calc_heading_quat_inv(q: torch.Tensor) -> torch.Tensor:
    """计算消除朝向后的逆四元数"""
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1
    
    # quat_from_angle_axis
    theta = (-heading).unsqueeze(-1)
    xyz = axis * torch.sin(theta / 2)
    w = torch.cos(theta / 2)
    heading_q = torch.cat([xyz, w], dim=-1)
    return heading_q


@torch.jit.script
def quat_mul_amp(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """四元数乘法 (xyzw 格式)"""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([x, y, z, w], dim=-1)


@torch.jit.script
def quat_rotate_inverse_amp(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """使用四元数的逆旋转向量"""
    q_conj = torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)
    return my_quat_rotate(q_conj, v)


@torch.jit.script
def compute_amp_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    key_body_pos: torch.Tensor,
) -> torch.Tensor:
    """
    计算 AMP 观测向量 (简化版：仅使用根节点旋转和关节位置)
    
    Args:
        dof_positions: [num_envs, 15] - 关节位置 (12 腿部 + 3 脊柱)
        dof_velocities: [num_envs, 15] - 关节速度 (未使用)
        root_pos: [num_envs, 3] - 根节点世界位置 (未使用)
        root_rot: [num_envs, 4] - 根节点四元数 (xyzw)
        root_lin_vel: [num_envs, 3] - 根节点世界坐标系线速度 (未使用)
        root_ang_vel: [num_envs, 3] - 根节点世界坐标系角速度 (未使用)
        key_body_pos: [num_envs, 4, 3] - 4个脚的世界位置 (未使用)
    
    Returns:
        [num_envs, 18] - AMP 观测 (仅根节点旋转和关节位置)
        - root_rot_obs (3): 根节点旋转（tan_norm后3维，表示pitch/roll）
        - dof_pos (15): 关节位置
    """
    # 1. Root Rotation (消除朝向后的旋转)
    heading_rot = calc_heading_quat_inv(root_rot)
    root_rot_obs = quat_mul_amp(heading_rot, root_rot)
    root_rot_obs = quat_to_tan_norm(root_rot_obs)  # [num_envs, 6]
    # 只取后3维 (法向量，表示pitch/roll)
    root_rot_obs = root_rot_obs[:, -3:]  # [num_envs, 3]
    
    # 组合观测 (仅根节点旋转和关节位置):
    # root_rot(3) + dof_pos(15) = 18
    obs = torch.cat([
        root_rot_obs,        # 3  
        dof_positions,       # 15
    ], dim=-1)
    
    return obs


@torch.jit.script
def compute_obs(
    projected_gravity: torch.Tensor,
    commands: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """
    计算策略观测向量 (Policy Observation).
    
    Himmy Mark2 观测包括 (51 dim):
    - Projected Gravity (3)
    - Commands (3)
    - Joint Pos (15): 12 腿部 + 3 脊柱
    - Joint Vel (15): 12 腿部 + 3 脊柱
    - Last Actions (15): 12 腿部 + 3 脊柱
    """
    obs = torch.cat(
        (
            projected_gravity,
            commands,
            joint_pos,
            joint_vel,
            actions,
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_rewards(
    rew_scale_termination: float,
    rew_scale_action_l2: float,
    rew_scale_joint_pos_limits: float,
    rew_scale_joint_acc_l2: float,
    rew_scale_joint_vel_l2: float,
    reset_terminated: torch.Tensor,
    actions: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    joint_acc: torch.Tensor,
    joint_vel: torch.Tensor,
):
    """
    计算基础奖励 (JIT 编译)。
    
    包括：
    - 终止惩罚
    - 动作平滑度惩罚 (L2)
    - 关节限位惩罚
    - 关节加速度惩罚
    - 关节速度惩罚
    """
    rew_termination = rew_scale_termination * reset_terminated.float()
    rew_action_l2 = rew_scale_action_l2 * torch.sum(torch.square(actions), dim=1)

    out_of_limits = -(joint_pos - soft_joint_pos_limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (joint_pos - soft_joint_pos_limits[:, :, 1]).clip(min=0.0)
    rew_joint_pos_limits = rew_scale_joint_pos_limits * torch.sum(out_of_limits, dim=1)

    rew_joint_acc_l2 = rew_scale_joint_acc_l2 * torch.sum(torch.square(joint_acc), dim=1)
    rew_joint_vel_l2 = rew_scale_joint_vel_l2 * torch.sum(torch.square(joint_vel), dim=1)
    total_reward = rew_termination + rew_action_l2 + rew_joint_pos_limits + rew_joint_acc_l2 + rew_joint_vel_l2

    log = {
        "pub_termination": (rew_termination).mean(),
        "pub_action_l2": (rew_action_l2).mean(),
        "pub_joint_pos_limits": (rew_joint_pos_limits).mean(),
        "pub_joint_acc_l2": (rew_joint_acc_l2).mean(),
        "pub_joint_vel_l2": (rew_joint_vel_l2).mean(),
    }
    return total_reward, log


@torch.jit.script
def exp_reward_with_floor(error: torch.Tensor, weight: float, sigma: float, floor: float = 3.0) -> torch.Tensor:
    """
    分段指数奖励函数：大误差区域使用线性惩罚，小误差区域使用指数奖励。
    
    Args:
        error: 误差值 (已平方)。
        weight: 奖励权重。
        sigma: 指数函数的标准差参数。
        floor: 阈值，单位是 sigma² 的倍数。

    Returns:
        分段指数奖励值。
    """
    sigma_sq = sigma * sigma
    threshold = floor * sigma_sq

    # exponential part at threshold and gradient
    exp_val_at_threshold = weight * torch.exp(-floor)
    linear_slope = weight / sigma_sq * torch.exp(-floor)  # ensure first-order continuous

    # large error region: use linear penalty (keep negative slope)
    linear_reward = exp_val_at_threshold - linear_slope * (error - threshold)

    # small error region: use exponential reward
    exp_reward = weight * torch.exp(-error / sigma_sq)

    # choose the corresponding reward function based on the error size
    return torch.where(error > threshold, linear_reward, exp_reward)
