from __future__ import annotations

import os

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import DCMotorCfg
import isaaclab.sim as sim_utils

from robot_lab.assets.unitree import UNITREE_A1_CFG

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")
# 使用 A1 的动作数据集
MOTIONS_A1_DIR = os.path.join(MOTIONS_DIR, "datasets", "A1_dataset")

# =============================================================================
# Robot Configuration Override (A1 for AMP)
# =============================================================================
# 复制原始配置
UNITREE_A1_AMP = UNITREE_A1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

@configclass
class A1AmpEnvCfg(DirectRLEnvCfg):
    """
    A1 AMP 环境配置类。
    
    定义了环境的各种参数，包括奖励权重、观测空间、动作空间、仿真参数等。
    
    A1 特性:
    - 12 DOF: 12 腿部关节
    """

    # basic reward (基础奖励权重 - 速度跟踪任务)
    # 重要：AMP模式下任务奖励会经过 lerp 混合 (lerp=0.3)
    # 最终奖励 = (1-lerp)*AMP奖励 + lerp*任务奖励
    # AMP奖励范围约 [0, 2]，任务奖励需要与之匹配
    # 原始 temp_amp_repo 的 scales 值是 1.5 和 0.5，除以 dt 是为了 IsaacGym 的累积机制
    # 但在 IsaacLab 中，奖励不需要除以 dt，直接使用原始 scales 值
    rew_termination = 0.0
    rew_lin_vel_xy = 1.5    # 线速度跟踪奖励 (原始scales值，不除以dt)
    rew_ang_vel_z = 0.5     # 角速度跟踪奖励 (原始scales值，不除以dt)

    rew_action_rate = -0.00 # 动作变化率惩罚 (平滑, 减少高频抖动)
    rew_base_height = 0.0  # 基座高度奖励 (保持正确站高，防止跪地)
    target_base_height = 0.40  # 目标站高 (Himmy Mark2)
    
    # Undesired contacts penalty (非脚部接触惩罚)
    # 惩罚躯体、大腿、小腿等非脚部身体部位与地面接触
    # 这可以防止机器人坐下或跪地
    rew_undesired_contacts = -0.6  # 负值表示惩罚 (每个接触身体部位的惩罚)
    undesired_contact_threshold = 0.025  # 高度阈值 (m)，低于此高度视为接触地面
    undesired_contact_body_names = [
        # 躯干部位
        "trunk",
        # 大腿部位
        "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
        # 小腿部位 (膝盖)
        "FL_calf", "FR_calf", "RL_calf", "RR_calf",
    ]  # 需要监测的非脚部身体部位名称
    
    # Regularization rewards (正则化惩罚)
    rew_action_l2 = 0.0
    rew_joint_pos_limits = 0.0
    rew_joint_acc_l2 = 0.0
    rew_joint_vel_l2 = 0.0


    # env (环境参数)
    episode_length_s = 20.0
    decimation = 6 # 参考 A1AMPCfg (sim dt=0.005, policy dt=0.03)
    dt = 0.005

    # commands (指令范围) - 与 temp_amp_repo/a1_amp_config.py 对齐
    class CommandsCfg:
        lin_vel_x_range = [0.5, 1.5]  # m/s (参考: [0.0, 5.0])
        lin_vel_y_range = [-0.2, 0.2] # m/s (参考: [-0.3, 0.3])
        ang_vel_yaw_range = [-0.3, 0.30] # rad/s (参考: [-1.57, 1.57])
        heading_command = False # 是否使用朝向指令 (暂不使用)
        resampling_time = 10.0 # 指令重采样时间 (秒)

    commands = CommandsCfg()

    # spaces (空间维度)
    # 观测空间: 
    # project_gravity(3) + commands(3) + dof_pos(12) + dof_vel(12) + actions(12) = 42
    num_observations = 42
    num_actions = 12  # 12 腿部
    observation_space = 42
    action_space = 12
    state_space = 0
    num_amp_observations = 1
    # AMP 观测空间 (与 MetalHead 原始 A1 对齐): 
    # root_h(1) + root_rot(3) + root_vel(3) + root_ang_vel(3) + dof_pos(12) + dof_vel(12) + key_body_pos(12) = 46
    amp_observation_space = 46

    early_termination = True
    termination_height = 0.15

    # motion_file = os.path.join(MOTIONS_DIR, "datasets/himmy_run.npz") # 动作数据文件路径
    motion_file = None
    # 使用 A1 专用的动作数据集
    motions_dir = MOTIONS_A1_DIR
    reference_body = "trunk" # 参考身体部位 (A1 的基座名称)
    reset_strategy = "default" # 重置策略: "default" (初始姿态) 或 "random-start" (随机动作帧)

    # =============================================================================
    # 课程学习配置 (Curriculum Learning)
    # =============================================================================
    # 课程学习通过逐步增加任务难度来提高训练效率和最终策略性能
    # 
    # 工作原理:
    # 1. 初始阶段: 速度命令范围缩小到 curriculum_initial_multiplier 倍
    # 2. 评估性能: 每 max_episode_length 步评估一次速度跟踪奖励
    # 3. 自适应提升: 如果平均每秒奖励超过 0.8 × 权重，则扩大速度命令范围
    # 4. 最终阶段: 速度命令范围达到 curriculum_final_multiplier 倍
    curriculum_enabled = False  # 是否启用课程学习
    curriculum_initial_multiplier = 0.1  # 初始速度范围乘数 (10%)
    curriculum_final_multiplier = 1.0    # 最终速度范围乘数 (100%)
    curriculum_reward_threshold = 0.8    # 奖励阈值 (80% 时升级难度)
    curriculum_delta_command = 0.1       # 每次增加的速度范围量 (m/s)
    curriculum_eval_term = "rew_lin_vel_xy"  # 用于评估的奖励项

    # simulation (仿真参数)
    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**21, # GPU 碰撞对容量
            gpu_total_aggregate_pairs_capacity=2**21,
        ),
    )
    
    # Debug visualization
    debug_vis: bool = False # 是否开启调试可视化 (显示速度箭头等)

    # scene (场景参数)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot (机器人资产配置) - 使用 A1
    robot: ArticulationCfg = UNITREE_A1_AMP
