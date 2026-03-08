# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
强化学习环境配置基类 - 速度跟踪运动任务

本文件定义了足式机器人速度跟踪任务的基础配置类，包括：
- 场景配置（地形、传感器、光照）
- MDP配置（观察、动作、奖励、终止条件）
- 课程学习配置
- 域随机化事件配置

这是整个项目的配置框架核心，所有具体机器人的配置都从这里继承。
"""

import inspect
import math
import sys
from dataclasses import MISSING

# Isaac Lab 核心模块导入
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# 导入自定义的 MDP（马尔科夫决策过程）函数模块
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

##
# 预定义配置
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# 场景定义
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """
    地形场景配置 - 包含足式机器人、地形、传感器和光照
    
    这个配置类定义了强化学习环境中的物理场景，包括：
    - 地形生成和物理材质
    - 机器人本体（需要在子类中指定）
    - 高度扫描传感器（用于感知地形）
    - 接触力传感器（用于检测足部接触）
    - 环境光照
    """

    # 地形配置
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",                      # 场景中的地形路径
        terrain_type="generator",                       # 使用生成器创建地形
        terrain_generator=ROUGH_TERRAINS_CFG,           # 粗糙地形配置（包含斜坡、阶梯等）
        max_init_terrain_level=5,                       # 最大初始地形难度等级
        collision_group=-1,                             # 碰撞组ID
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",           # 摩擦力组合模式
            restitution_combine_mode="multiply",        # 恢复系数组合模式
            static_friction=1.0,                        # 静摩擦系数
            dynamic_friction=1.0,                       # 动摩擦系数
            restitution=1.0,                            # 恢复系数（弹性）
        ),
        visual_material=sim_utils.MdlFileCfg(
            # 使用大理石纹理材质
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,                           # 启用UV投影
            texture_scale=(0.25, 0.25),                 # 纹理缩放比例
        ),
        debug_vis=False,                                # 调试可视化（默认关闭）
    )
    
    # 机器人配置（需要在子类中具体指定）
    robot: ArticulationCfg = MISSING
    
    # 传感器配置
    # 高度扫描器 - 用于感知机器人前方的地形高度分布
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",          # 附加到机器人基座
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # 从上方20米向下扫描
        ray_alignment="yaw",                            # 射线对齐方式（跟随偏航角）
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 网格模式：分辨率0.1m，大小1.6m×1.0m
        debug_vis=False,                                # 调试可视化
        mesh_prim_paths=["/World/ground"],              # 扫描对象（地面）
    )
    
    # 基座高度扫描器 - 用于精确测量机器人正下方的地面高度
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",          
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),  # 更高分辨率，更小范围
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    
    # 接触力传感器 - 检测机器人各部位与环境的接触
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",           # 监测机器人所有部位
        history_length=3,                               # 保存3个时间步的历史数据
        track_air_time=True                             # 追踪足部腾空时间
    )
    
    # 光照配置
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,                            # 光照强度
            # 使用HDRI环境贴图
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP 配置（马尔科夫决策过程）
# 定义强化学习环境的核心要素：状态、动作、奖励、终止条件
##


@configclass
class CommandsCfg:
    """
    命令配置 - 定义发给机器人的目标速度命令
    
    在速度跟踪任务中，我们会给机器人发送目标速度命令（x、y方向线速度和z轴角速度），
    机器人需要学习如何跟踪这些命令。
    """

    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",                             # 目标资产名称
        resampling_time_range=(10.0, 10.0),            # 命令重采样时间范围（秒）
        rel_standing_envs=0.02,                         # 2%的环境会收到静止命令
        rel_heading_envs=1.0,                           # 100%的环境使用航向命令
        heading_command=True,                           # 启用航向命令
        heading_control_stiffness=0.5,                  # 航向控制刚度
        debug_vis=True,                                 # 调试可视化（显示目标速度箭头）
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),                     # x方向线速度范围（米/秒）
            lin_vel_y=(-1.0, 1.0),                     # y方向线速度范围（米/秒）
            ang_vel_z=(-1.0, 1.0),                     # z轴角速度范围（弧度/秒）
            heading=(-math.pi, math.pi)                # 航向角范围（弧度）
        ),
    )


@configclass
class ActionsCfg:
    """
    动作配置 - 定义强化学习策略输出的动作空间
    
    对于足式机器人，动作通常是关节位置目标值（Position Control）
    或关节力矩目标值（Torque Control）。这里使用位置控制。
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",                             # 目标机器人
        joint_names=[".*"],                             # 控制所有关节（正则表达式匹配）
        scale=0.5,                                      # 动作缩放因子（减小动作幅度）
        use_default_offset=True,                        # 使用默认位置作为偏移（相对控制）
        clip=None,                                      # 动作裁剪范围（None表示不裁剪）
        preserve_order=True                             # 保持关节顺序
    )


@configclass
class ObservationsCfg:
    """
    观察配置 - 定义强化学习中的状态空间
    
    包含两个观察组：
    - PolicyCfg: 策略网络（Actor）的输入观察，会添加噪声以提高鲁棒性
    - CriticCfg: 价值网络（Critic）的输入观察，不添加噪声以准确估计价值
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """
        策略网络观察配置
        
        定义了策略网络的输入特征，包括机器人状态、命令、历史动作等。
        所有观察都会添加噪声以模拟传感器噪声，提高策略的鲁棒性。
        """

        # 观察项定义（顺序会被保留）
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,                      # 基座线速度
            noise=Unoise(n_min=-0.1, n_max=0.1),       # 添加均匀噪声
            clip=(-100.0, 100.0),                       # 数值裁剪范围
            scale=1.0,                                  # 缩放因子
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,                      # 基座角速度
            noise=Unoise(n_min=-0.2, n_max=0.2),       # 角速度噪声较大
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,                 # 投影重力向量（用于感知姿态）
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,                # 当前的目标速度命令
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,                     # 关节位置（相对于默认位置）
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,                     # 关节速度
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),       # 速度噪声较大
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,                       # 上一时刻的动作（历史信息）
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,                       # 地形高度扫描数据
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True               # 启用噪声污染
            self.concatenate_terms = True               # 将所有观察拼接成一个向量

    @configclass
    class CriticCfg(ObsGroup):
        """
        价值网络观察配置
        
        Critic用于估计状态价值，通常需要更准确的状态信息，
        因此不添加噪声。观察项与PolicyCfg相同，但无噪声污染。
        """

        # 观察项定义（顺序会被保留）
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,                      # 基座线速度（无噪声）
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,                      # 基座角速度（无噪声）
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,                 # 投影重力向量（无噪声）
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,                # 目标速度命令
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,                     # 关节位置（无噪声）
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,                     # 关节速度（无噪声）
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,                       # 上一时刻的动作
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,                       # 地形高度扫描（无噪声）
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        # joint_effort = ObsTerm(
        #     func=mdp.joint_effort,                    # 关节力矩（可选）
        #     clip=(-100, 100),
        #     scale=0.01,
        # )

        def __post_init__(self):
            self.enable_corruption = False              # 不启用噪声污染
            self.concatenate_terms = True               # 拼接所有观察

    # 观察组实例化
    policy: PolicyCfg = PolicyCfg()                     # 策略网络使用的观察
    critic: CriticCfg = CriticCfg()                     # 价值网络使用的观察


@configclass
class EventCfg:
    """
    事件配置 - 域随机化（Domain Randomization）
    
    域随机化是提高策略鲁棒性的重要技术，通过在训练时随机改变：
    - 物理参数（质量、摩擦力等）
    - 初始状态（位置、速度等）
    - 外部扰动（推力、力矩等）
    
    使策略能够适应真实世界中的参数不确定性和环境变化。
    
    事件触发模式：
    - startup: 环境启动时触发一次
    - reset: 每次环境重置时触发
    - interval: 按固定时间间隔触发
    """

    # ============ 启动时触发的随机化（startup） ============
    
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,         # 随机化物体材质
        mode="startup",                                 # 启动时触发
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 所有刚体
            "static_friction_range": (0.3, 1.0),        # 静摩擦系数范围
            "dynamic_friction_range": (0.3, 0.8),       # 动摩擦系数范围
            "restitution_range": (0.0, 0.5),            # 恢复系数范围
            "num_buckets": 64,                          # 离散化桶数量
        },
    )

    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,             # 随机化基座质量
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),  # 基座（需在子类中指定）
            "mass_distribution_params": (-1.0, 3.0),    # 质量变化范围（加法）
            "operation": "add",                          # 加法操作
            "recompute_inertia": True,                  # 重新计算惯性
        },
    )

    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,             # 随机化其他部位质量
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 所有刚体
            "mass_distribution_params": (0.7, 1.3),     # 质量缩放范围（乘法）
            "operation": "scale",                        # 缩放操作
            "recompute_inertia": True,                  # 重新计算惯性
        },
    )

    # Skip: 惯性已通过质量随机化自动更新（recompute_inertia=True）
    # randomize_rigid_body_inertia = EventTerm(
    #     func=mdp.randomize_rigid_body_inertia,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "inertia_distribution_params": (0.5, 1.5),
    #         "operation": "scale",
    #     },
    # )

    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,              # 随机化质心位置
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},  # 质心偏移范围
        },
    )

    # ============ 每次重置时触发的随机化（reset） ============
    
    randomize_apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,           # 施加外部力和力矩
        mode="reset",                                   # 重置时触发
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),  # 基座（需在子类中指定）
            "force_range": (-10.0, 10.0),               # 力的范围（牛顿）
            "torque_range": (-10.0, 10.0),              # 力矩范围（牛顿·米）
        },
    )

    randomize_reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,                 # 重置关节状态（缩放方式）
        # func=mdp.reset_joints_by_offset,              # 或使用偏移方式
        mode="reset",
        params={
            "position_range": (1.0, 1.0),               # 位置缩放范围（1.0表示使用默认位置）
            "velocity_range": (0.0, 0.0),               # 速度范围（从零开始）
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,              # 随机化执行器增益
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 2.0),  # 刚度缩放范围
            "damping_distribution_params": (0.5, 2.0),    # 阻尼缩放范围
            "operation": "scale",                         # 缩放操作
            "distribution": "uniform",                    # 均匀分布
        },
    )

    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,              # 随机化初始位姿和速度
        mode="reset",
        params={
            "pose_range": {                              # 位姿随机化范围
                "x": (-0.5, 0.5),                        # x方向位置（米）
                "y": (-0.5, 0.5),                        # y方向位置（米）
                "yaw": (-3.14, 3.14)                     # 偏航角（弧度）
            },
            "velocity_range": {                          # 速度随机化范围
                "x": (-0.5, 0.5),                        # x方向线速度
                "y": (-0.5, 0.5),                        # y方向线速度
                "z": (-0.5, 0.5),                        # z方向线速度
                "roll": (-0.5, 0.5),                     # 滚转角速度
                "pitch": (-0.5, 0.5),                    # 俯仰角速度
                "yaw": (-0.5, 0.5),                      # 偏航角速度
            },
        },
    )

    # ============ 定时间隔触发的随机化（interval） ============
    
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,              # 通过设置速度来"推"机器人
        mode="interval",                                # 间隔触发
        interval_range_s=(10.0, 15.0),                  # 触发间隔（10-15秒）
        params={
            "velocity_range": {                          # 推力速度范围
                "x": (-0.5, 0.5),                        # x方向
                "y": (-0.5, 0.5)                         # y方向
            }
        },
    )


@configclass
class RewardsCfg:
    """
    奖励配置 - 定义强化学习的奖励函数
    
    奖励函数是强化学习中最重要的设计要素，它定义了我们希望机器人学习什么行为。
    好的奖励函数应该：
    1. 明确目标：清楚地指示任务目标（如速度跟踪）
    2. 塑造行为：通过惩罚项引导学习过程（如平滑运动、节能）
    3. 避免不良行为：惩罚危险或不期望的行为（如摔倒、碰撞）
    
    权重为 0.0 表示该奖励项默认禁用，需要在子类中设置具体权重。
    正权重表示奖励，负权重表示惩罚。
    """

    # ============ 通用奖励 ============
    is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)  # 终止惩罚

    # ============ 基座惩罚项（Root penalties）============
    # 这些项用于约束机器人基座的运动，保持稳定姿态
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)    # 惩罚z方向线速度（避免跳跃）
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)  # 惩罚x,y方向角速度（保持水平）
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)  # 惩罚姿态倾斜
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.0,
        },
    )
    body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    )

    # Joint penalties
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )

    def create_joint_deviation_l1_rewterm(self, attr_name, weight, joint_names_pattern):
        rew_term = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=weight,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names_pattern)},
        )
        setattr(self, attr_name, rew_term)

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 1.0},
    )
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    wheel_vel_penalty = RewTerm(
        func=mdp.wheel_vel_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
        },
    )

    action_mirror = RewTerm(
        func=mdp.action_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
        },
    )

    action_sync = RewTerm(
        func=mdp.action_sync,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_groups": [
                ["FR_hip_joint", "FL_hip_joint", "RL_hip_joint", "RR_hip_joint"],
                ["FR_thigh_joint", "FL_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
                ["FR_calf_joint", "FL_calf_joint", "RL_calf_joint", "RR_calf_joint"],
            ],
        },
    )

    # Action penalties
    applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    # smoothness_1 = RewTerm(func=mdp.smoothness_1, weight=0.0)  # Same as action_rate_l2
    # smoothness_2 = RewTerm(func=mdp.smoothness_2, weight=0.0)  # Unvaliable now

    # Contact sensor
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 1.0,
        },
    )
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 100.0},
    )

    # Velocity-tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # Others
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        },
    )

    feet_air_time_variance = RewTerm(
        func=mdp.feet_air_time_variance_penalty,
        weight=0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="")},
    )

    feet_gait = RewTerm(
        func=mdp.GaitReward,
        weight=0.0,
        params={
            "std": math.sqrt(0.5),
            "command_name": "base_velocity",
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("", ""), ("", "")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    feet_contact = RewTerm(
        func=mdp.feet_contact,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
            "expect_contact_num": 2,
        },
    )

    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
        },
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
        },
    )

    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "tanh_mult": 2.0,
            "target_height": 0.05,
            "command_name": "base_velocity",
        },
    )

    feet_height_body = RewTerm(
        func=mdp.feet_height_body,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "tanh_mult": 2.0,
            "target_height": -0.3,
            "command_name": "base_velocity",
        },
    )

    feet_distance_y_exp = RewTerm(
        func=mdp.feet_distance_y_exp,
        weight=0.0,
        params={
            "std": math.sqrt(0.25),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "stance_width": float,
        },
    )

    # feet_distance_xy_exp = RewTerm(
    #     func=mdp.feet_distance_xy_exp,
    #     weight=0.0,
    #     params={
    #         "std": math.sqrt(0.25),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=""),
    #         "stance_length": float,
    #         "stance_width": float,
    #     },
    # )

    upward = RewTerm(func=mdp.upward, weight=0.0)


    


@configclass
class TerminationsCfg:
    """
    终止条件配置 - 定义episode何时结束
    
    终止条件用于判断当前episode是否应该结束。包括：
    - 正常终止：任务完成、时间耗尽
    - 失败终止：机器人摔倒、越界等
    """

    # MDP 终止条件
    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # 时间耗尽（正常终止）
    
    # 越界终止
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,                     # 机器人移动出地形边界
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "distance_buffer": 3.0                           # 边界缓冲区（米）
        },
        time_out=True,                                      # 标记为超时（不是失败）
    )

    # 接触传感器终止条件
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,                           # 非法接触（如身体触地）
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),  # 需在子类中指定
            "threshold": 1.0                                 # 接触力阈值（牛顿）
        },
    )


@configclass
class CurriculumCfg:
    """
    课程学习配置 - 渐进式训练难度
    
    课程学习（Curriculum Learning）是一种训练策略，从简单任务开始，
    逐步增加难度，类似于人类学习过程。这可以：
    1. 加快训练速度
    2. 提高最终性能
    3. 避免训练早期陷入局部最优
    """

    # 地形难度课程
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)  # 根据性能调整地形难度

    # 命令范围课程
    command_levels = CurrTerm(
        func=mdp.command_levels_vel,                        # 根据性能调整命令范围
        params={
            "reward_term_name": "track_lin_vel_xy_exp",    # 用于评估的奖励项
            "range_multiplier": (0.1, 1.0),                 # 命令范围乘数（从10%到100%）
        },
    )


##
# 环境配置主类
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """
    速度跟踪运动环境配置基类（粗糙地形）
    
    这是整个配置系统的核心类，集成了所有MDP要素：
    - scene: 物理场景（机器人、地形、传感器）
    - observations: 状态空间（策略和价值网络的输入）
    - actions: 动作空间（策略网络的输出）
    - commands: 任务命令（目标速度）
    - rewards: 奖励函数（学习目标）
    - terminations: 终止条件（episode结束判断）
    - events: 域随机化（提高鲁棒性）
    - curriculum: 课程学习（渐进式训练）
    
    所有具体机器人的环境配置都应该从这个类继承。
    """

    # ============ 场景配置 ============
    scene: MySceneCfg = MySceneCfg(
        num_envs=4096,                          # 并行环境数量（GPU加速）
        env_spacing=2.5                         # 环境间距（米）
    )
    
    # ============ 基础MDP配置 ============
    observations: ObservationsCfg = ObservationsCfg()    # 观察空间
    actions: ActionsCfg = ActionsCfg()                   # 动作空间
    commands: CommandsCfg = CommandsCfg()                # 命令生成
    
    # ============ 强化学习配置 ============
    rewards: RewardsCfg = RewardsCfg()                   # 奖励函数
    terminations: TerminationsCfg = TerminationsCfg()    # 终止条件
    events: EventCfg = EventCfg()                        # 域随机化事件
    curriculum: CurriculumCfg = CurriculumCfg()          # 课程学习

    def __post_init__(self):
        """
        后初始化方法 - 设置仿真参数和传感器更新频率
        
        在所有配置参数设置完成后自动调用，用于：
        1. 配置仿真时间步长和控制频率
        2. 设置传感器更新周期
        3. 启用/禁用课程学习
        """
        # ============ 通用设置 ============
        self.decimation = 4                     # 控制间隔（每4个物理步执行一次控制）
        self.episode_length_s = 20.0            # episode长度（秒）
        
        # ============ 仿真设置 ============
        self.sim.dt = 0.005                     # 物理仿真时间步长（秒）= 200Hz
        self.sim.render_interval = self.decimation  # 渲染间隔
        self.sim.physics_material = self.scene.terrain.physics_material  # 物理材质
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15  # GPU最大刚体补丁数
        
        # ============ 传感器更新周期配置 ============
        # 所有传感器基于最小更新周期（物理更新周期）进行tick
        if self.scene.height_scanner is not None:
            # 高度扫描器：控制频率更新（每个控制步更新一次）
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            # 接触力传感器：物理频率更新（每个物理步更新一次）
            self.scene.contact_forces.update_period = self.sim.dt

        # ============ 地形课程学习配置 ============
        # 检查是否启用了地形级别课程 - 如果启用，则为地形生成器启用课程
        # 这会生成难度递增的地形，对训练很有用
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    def disable_zero_weight_rewards(self):
        """
        禁用零权重奖励项
        
        遍历所有奖励配置，如果某个奖励的权重为0，则将其设置为None。
        这可以：
        1. 减少计算开销（不计算无用的奖励）
        2. 提高训练效率
        3. 使配置更清晰
        
        通常在具体机器人配置的 __post_init__ 中调用。
        """
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                # 检查是否是奖励项且权重为0
                if not callable(reward_attr) and hasattr(reward_attr, "weight") and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)


def create_obsgroup_class(class_name, terms, enable_corruption=False, concatenate_terms=True):
    """
    动态创建观察组类
    
    这是一个高级功能，允许在运行时动态创建自定义观察组配置类。
    用于需要根据机器人特性动态生成观察配置的场景。
    
    参数说明：
    :param class_name: 配置类名称（字符串）
    :param terms: 配置项字典，键为项名称，值为ObsTerm对象
    :param enable_corruption: 是否启用噪声污染（默认False）
    :param concatenate_terms: 是否将观察项拼接成一个向量（默认True）
    
    :return: 动态创建的ObsGroup类
    
    示例：
    >>> terms = {
    >>>     "base_vel": ObsTerm(func=mdp.base_lin_vel, scale=1.0),
    >>>     "joint_pos": ObsTerm(func=mdp.joint_pos_rel, scale=1.0),
    >>> }
    >>> MyObsClass = create_obsgroup_class("MyCustomObs", terms, enable_corruption=True)
    """
    # Dynamically determine the module name
    module_name = inspect.getmodule(inspect.currentframe()).__name__

    # Define the post-init function
    def post_init_wrapper(self):
        setattr(self, "enable_corruption", enable_corruption)
        setattr(self, "concatenate_terms", concatenate_terms)

    # Dynamically create the class using ObsGroup as the base class
    terms["__post_init__"] = post_init_wrapper
    dynamic_class = configclass(type(class_name, (ObsGroup,), terms))

    # Custom serialization and deserialization
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # Add custom serialization methods to the class
    dynamic_class.__getstate__ = __getstate__
    dynamic_class.__setstate__ = __setstate__

    # Place the class in the global namespace for accessibility
    globals()[class_name] = dynamic_class

    # Register the dynamic class in the module's dictionary
    if module_name in sys.modules:
        sys.modules[module_name].__dict__[class_name] = dynamic_class
    else:
        raise ImportError(f"Module {module_name} not found.")

    # Return the class for external instantiation
    return dynamic_class
