# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
PPO算法配置文件 - Unitree Go2 机器人

本文件定义了使用RSL-RL库训练Unitree Go2机器人的PPO算法超参数。
包括：
- 训练器配置（迭代次数、保存间隔等）
- 策略网络配置（神经网络结构）
- 算法配置（PPO超参数）

PPO (Proximal Policy Optimization) 是一种稳定高效的策略梯度算法，
广泛应用于机器人运动控制任务。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Unitree Go2 粗糙地形PPO训练配置"""
    
    # 训练器配置
    num_steps_per_env = 24              # 每环境采样步数 (总样本数 = 24 × num_envs)
    max_iterations = 10000              # 最大迭代次数 (总步数 ≈ 10000 × 24 × 4096 ≈ 10亿步)
    save_interval = 100                 # 模型保存间隔
    experiment_name = "unitree_go2_rough"
    
    # 策略网络配置
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                 # 初始探索噪声
        actor_obs_normalization=False,      # Actor输入归一化（已在环境中处理）
        critic_obs_normalization=False,     # Critic输入归一化（已在环境中处理）
        actor_hidden_dims=[512, 256, 128],   # Actor网络: obs → 512 → 256 → 128 → action
        critic_hidden_dims=[512, 256, 128],  # Critic网络: obs → 512 → 256 → 128 → value
        activation="elu",               # 激活函数 (Exponential Linear Unit)
    )
    
    # PPO算法配置
    algorithm = RslRlPpoAlgorithmCfg(
        # 损失函数
        value_loss_coef=1.0,            # 价值损失系数
        use_clipped_value_loss=True,    # 使用裁剪价值损失
        
        # PPO核心参数
        clip_param=0.2,                 # 策略裁剪参数 ε (限制更新幅度)
        entropy_coef=0.01,              # 熵系数 (鼓励探索)
        
        # 训练参数
        num_learning_epochs=5,          # 每次迭代训练轮数
        num_mini_batches=4,             # mini-batch数量
        learning_rate=1.0e-3,           # 学习率 (Adam优化器)
        schedule="adaptive",            # 学习率调度 (根据KL散度自适应)
        
        # 折扣和优势估计
        gamma=0.99,                     # 折扣因子 (长期回报权重)
        lam=0.95,                       # GAE-Lambda (广义优势估计)
        
        # 稳定性参数
        desired_kl=0.01,                # 目标KL散度 (用于自适应学习率)
        max_grad_norm=1.0,              # 梯度裁剪阈值 (防止梯度爆炸)
    )


@configclass
class UnitreeGo2FlatPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    """Unitree Go2 平坦地形PPO训练配置 (继承自粗糙地形，训练更快)"""
    
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 5000          # 减少迭代次数 (平坦地形收敛更快)
        self.experiment_name = "unitree_go2_flat"
