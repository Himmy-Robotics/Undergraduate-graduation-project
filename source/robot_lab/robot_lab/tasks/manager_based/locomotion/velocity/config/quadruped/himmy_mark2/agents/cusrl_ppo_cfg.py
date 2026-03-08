# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""PPO 算法配置文件 - Himmy Mark2 四足机器人 (CusRL 框架)

注意: 脊柱关节已固定，机器人现为标准四足配置 (12个控制关节)
"""

from isaaclab.utils import configclass


@configclass
class HimmyMark2RoughTrainerCfg:
    """
    Himmy Mark2 粗糙地形 CusRL 训练配置.
    
    该配置用于在复杂地形上训练四足机器人的速度控制策略。
    """
    
    # ========== 训练器配置 ==========
    num_steps_per_env = 24  # 每环境采样步数
    max_iterations = 10000  # 最大训练迭代次数
    save_interval = 100  # 模型保存间隔
    experiment_name = "himmy_mark2_rough"
    
    # ========== 策略网络配置 ==========
    # Actor 神经网络结构
    actor_hidden_dims = [512, 256, 128]  # obs → 512 → 256 → 128 → action
    
    # Critic 神经网络结构
    critic_hidden_dims = [512, 256, 128]  # obs → 512 → 256 → 128 → value
    
    # 激活函数
    activation = "elu"
    
    # 初始探索噪声标准差
    init_noise_std = 1.0
    
    # ========== 训练超参数 ==========
    learning_rate = 1.0e-3  # 初始学习率
    
    # PPO 超参数
    clip_param = 0.2  # 策略裁剪参数
    entropy_coef = 0.01  # 熵系数
    value_loss_coef = 1.0  # 价值损失系数
    
    # 训练轮数和批次
    num_learning_epochs = 5  # 每次迭代的训练轮数
    num_mini_batches = 4  # Mini-batch 数量
    
    # 折扣和优势估计
    gamma = 0.99  # 折扣因子
    lam = 0.95  # GAE-Lambda 参数
    
    # 稳定性参数
    max_grad_norm = 1.0  # 梯度裁剪阈值
    desired_kl = 0.01  # 目标 KL 散度


@configclass
class HimmyMark2FlatTrainerCfg(HimmyMark2RoughTrainerCfg):
    """
    Himmy Mark2 平坦地形 CusRL 训练配置.
    
    继承自粗糙地形配置，但减少迭代次数。
    """
    
    # 平坦地形收敛更快，减少迭代次数
    max_iterations = 5000  # 粗糙地形的一半
    
    # 更新实验名称
    experiment_name = "himmy_mark2_flat"
