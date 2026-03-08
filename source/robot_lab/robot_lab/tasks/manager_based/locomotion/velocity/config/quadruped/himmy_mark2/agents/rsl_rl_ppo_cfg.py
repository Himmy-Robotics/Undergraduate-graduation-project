# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
PPO 算法配置文件 - Himmy Mark2 四足机器人 (RSL-RL 框架)

本文件定义了使用 RSL-RL 库训练 Himmy Mark2 机器人的 PPO 算法超参数。
包括：
- 训练器配置（迭代次数、保存间隔等）
- 策略网络配置（神经网络结构）
- 算法配置（PPO 超参数）

PPO (Proximal Policy Optimization) 是一种稳定高效的策略梯度算法，
广泛应用于机器人运动控制任务。

参考:
  - RSL-RL 文档: https://rsl-rl.readthedocs.io/
  - PPO 论文: https://arxiv.org/abs/1707.06347
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class HimmyMark2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    Himmy Mark2 粗糙地形 PPO 训练配置.
    
    该配置用于在复杂地形（起伏、障碍物等）上训练四足机器人的速度控制策略。
    """
    
    # ========== 训练器配置 ==========
    num_steps_per_env = 24  # 每环境采样步数
    # 总样本数 = num_steps_per_env × num_envs
    # 例如: 24 × 4096 = 98,304 样本/迭代
    
    max_iterations = 10000  # 最大训练迭代次数
    # 总训练步数 ≈ max_iterations × num_steps_per_env × num_envs
    # 例如: 10000 × 24 × 4096 ≈ 10 亿步
    
    save_interval = 100  # 每 100 次迭代保存一次模型
    experiment_name = "himmy_mark2_rough"  # 实验文件夹名称
    
    # ========== 策略网络配置 ==========
    policy = RslRlPpoActorCriticCfg(
        # 初始探索噪声标准差(越大探索越充分，但收敛慢；越小收敛快，但可能陷入局部最优)
        init_noise_std=1.0,
        
        # 输入归一化(False = 在环境中已处理，不需重复)
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        
        # Actor 神经网络结构
        # obs(46维) → 512 → 256 → 128 → action(12维)
        # 维度说明:
        #   - obs: 基座速度(3) + 关节位置(12) + 关节速度(12) + 高度扫描(15) = 42维
        #          或不含高度扫描 = 27维，具体取决于配置
        #   - action: 12 个腿部关节
        actor_hidden_dims=[512, 256, 128],
        
        # Critic 神经网络结构(价值函数估计)
        # obs → 512 → 256 → 128 → value scalar
        critic_hidden_dims=[512, 256, 128],
        
        # 激活函数选择
        # - "elu": Exponential Linear Unit，计算快，梯度稳定
        # - "relu": ReLU，简单高效
        # - "silu": Swish，表现力强但计算量大
        activation="elu",
    )
    
    # ========== PPO 算法配置 ==========
    algorithm = RslRlPpoAlgorithmCfg(
        # ---- 损失函数权重 ----
        value_loss_coef=1.0,  # 价值函数损失权重
        # loss_total = actor_loss + value_loss_coef × value_loss + entropy_coef × entropy_loss
        
        use_clipped_value_loss=True,  # 使用裁剪价值损失(防止价值函数抖动)
        
        # ---- PPO 核心参数 ----
        clip_param=0.2,  # 策略裁剪参数 ε
        # 限制新旧策略比值在 [1-ε, 1+ε] = [0.8, 1.2]
        # ε 越小更新越保守，ε 越大更新更激进
        
        entropy_coef=0.01,  # 熵系数(鼓励探索，防止策略过早收敛)
        
        # ---- 训练超参数 ----
        num_learning_epochs=5,  # 每次迭代的训练轮数
        # 每次采集 num_steps_per_env 步数据后，训练 num_learning_epochs 轮
        
        num_mini_batches=4,  # Mini-batch 数量
        # batch_size = total_samples / num_mini_batches
        # 例如: 98,304 / 4 = 24,576
        
        learning_rate=1.0e-3,  # 初始学习率(Adam 优化器)
        schedule="adaptive",  # 学习率调度策略
        # - "fixed": 保持不变
        # - "linear": 线性衰减
        # - "adaptive": 根据 KL 散度自适应调整
        
        # ---- 折扣和优势估计 ----
        gamma=0.99,  # 折扣因子(0~1)
        # 未来奖励的权重: γ=0.99 表示 100 步后的奖励权重为 1/e ≈ 36.8%
        # γ 越接近 1，越重视长期回报；γ 越小，越重视近期回报
        
        lam=0.95,  # GAE-Lambda 参数(广义优势估计)
        # 控制偏差-方差权衡
        # λ=0: 只使用当前值函数估计(低方差，高偏差)
        # λ=1: 使用完整回报(高方差，低偏差)
        # λ=0.95: 平衡点，推荐值
        
        # ---- 稳定性参数 ----
        desired_kl=0.01,  # 目标 KL 散度(用于自适应学习率)
        # 当实际 KL > desired_kl 时，降低学习率；反之增加
        
        max_grad_norm=1.0,  # 梯度裁剪阈值
        # 防止梯度爆炸，限制梯度范数 ≤ max_grad_norm
    )


@configclass
class HimmyMark2FlatPPORunnerCfg(HimmyMark2RoughPPORunnerCfg):
    """
    Himmy Mark2 平坦地形 PPO 训练配置.
    
    继承自粗糙地形配置，但减少迭代次数(平坦地形收敛更快)。
    通常用作预训练，可加速粗糙地形训练的收敛。
    """
    
    def __post_init__(self):
        """初始化后调用，覆盖某些参数."""
        super().__post_init__()
        
        # 平坦地形收敛更快，减少迭代次数
        self.max_iterations = 20000  # 粗糙地形的一半
        
        # 更新实验名称
        self.experiment_name = "himmy_mark2_flat"
