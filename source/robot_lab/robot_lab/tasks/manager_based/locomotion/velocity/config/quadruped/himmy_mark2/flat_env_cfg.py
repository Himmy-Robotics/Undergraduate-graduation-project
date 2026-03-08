# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import HimmyMark2RoughEnvCfg


@configclass
class HimmyMark2FlatEnvCfg(HimmyMark2RoughEnvCfg):
    """
    Himmy Mark2 四足机器人在平坦地形上的强化学习环境配置
    
    该配置类继承自 HimmyMark2RoughEnvCfg，针对平坦地形进行了简化：
    - 移除了地形生成和高度扫描
    - 简化了观察空间
    - 移除了地形课程学习
    
    适用于在简单平面上进行运动控制学习，训练速度更快，适合作为基础训练环境。
    """
    
    def __post_init__(self):
        """初始化平坦地形环境配置（在类实例化后自动调用）"""
        # 调用父类（粗糙地形配置）的初始化方法
        super().__post_init__()

        # 覆盖奖励配置 - 移除传感器配置（平坦地形不需要复杂的高度传感）
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        
        # 将地形改为平面
        self.scene.terrain.terrain_type = "plane"          # 使用简单平面地形
        self.scene.terrain.terrain_generator = None        # 不使用地形生成器
        
        # 移除高度扫描（平坦地形不需要感知地形起伏）
        self.scene.height_scanner = None                   # 禁用场景高度扫描器
        self.observations.policy.height_scan = None        # 禁用策略网络的高度扫描观察
        self.observations.critic.height_scan = None        # 禁用价值网络的高度扫描观察
        
        # 移除地形课程学习（平坦地形无需课程）
        self.curriculum.terrain_levels = None              # 禁用地形难度递增

        # 如果奖励权重为0，则禁用该奖励项
        if self.__class__.__name__ == "HimmyMark2FlatEnvCfg":
            self.disable_zero_weight_rewards()

