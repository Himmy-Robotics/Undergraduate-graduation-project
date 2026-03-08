"""Configuration for Himmy Mark2 Quadruped Robot (自定义四足机器人配置).

该模块定义了 Himmy Mark2 四足机器人在 IsaacLab 物理仿真环境中的配置参数，
包括机器人模型加载、物理属性、电机参数、初始状态和执行器配置等信息。

Reference: 基于 Solidworks 导出的 URDF 模型
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# 机器人配置常量定义
##

# ==================== Himmy Mark2 机器人配置 ====================
HIMMY_MARK2_CFG = ArticulationCfg(
    # ========== 模型加载和仿真设置 ==========
    spawn=sim_utils.UrdfFileCfg(
        # URDF 文件路径配置
        fix_base=False,  # 是否固定基座(False=机器人可自由运动，True=固定在原地)
        merge_fixed_joints=False,  # 是否合并固定关节(减少计算复杂度)
        replace_cylinders_with_capsules=True,  # 是否将圆柱体替换为胶囊体(提高物理仿真稳定性)
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Himmy/mark2_description/urdf/mark2.urdf",
        # 资源文件路径，指向 Solidworks 导出的 URDF 文件
        
        # 接触传感器配置
        activate_contact_sensors=True,  # 启用接触传感器(用于检测足部与地面接触)
        
        # ========== 刚体物理属性设置 ==========
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # 是否禁用重力(False=启用重力)
            retain_accelerations=False,  # 是否保留加速度(用于连续模拟)
            linear_damping=0.0,  # 线性阻尼系数(空气阻力)
            angular_damping=0.0,  # 角阻尼系数(旋转阻力)
            max_linear_velocity=1000.0,  # 最大线速度限制(m/s)
            max_angular_velocity=1000.0,  # 最大角速度限制(rad/s)
            max_depenetration_velocity=1.0,  # 最大穿透速度(用于碰撞分离)
        ),
        
        # ========== 关节动力学设置 ==========
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # 是否启用自碰撞检测(减少计算开销)
            solver_position_iteration_count=4,  # 位置求解器迭代次数(次数越多精度越高)
            solver_velocity_iteration_count=0,  # 速度求解器迭代次数(0=自动)
        ),
        
        # ========== 关节驱动器基础配置 ==========
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0,  # 初始刚度(由下面的执行器配置覆盖)
                damping=0,  # 初始阻尼(由下面的执行器配置覆盖)
            )
        ),
    ),
    
    # ========== 初始状态配置 ==========
    init_state=ArticulationCfg.InitialStateCfg(
        # 初始位置(x, y, z 坐标，单位: 米)
        pos=(0.0, 0.0, 0.40),  # z = 0.40m (参考 Unitree A1 的 0.38m，略微调高)
        
        # 初始关节位置(以弧度为单位)
        joint_pos={

            ".*_hip_joint": 0.0,    #髋关节

            ".*_thigh_joint": 0.8, # 腿部大腿关节

            ".*_calf_joint": -1.5,  # 小腿关节
            
            # 脊椎关节(机器人躯干): 初始为 0(直立)
            "yaw_spine_joint": 0.0,  # 偏航脊椎关节(绕 Z 轴旋转)
            "pitch_spine_joint": 0.0,  # 俯仰脊椎关节(绕 Y 轴旋转)
            "roll_spine_joint": 0.0,  # 滚转脊椎关节(绕 X 轴旋转)
        },
        
        # 初始关节速度(以 rad/s 为单位)
        joint_vel={".*": 0.0},  # 所有关节初始速度均为 0(静止状态)
    ),
    
    # ========== 关节限制配置 ==========
    soft_joint_pos_limit_factor=0.9,  # 软关节位置限制因子(设置为 90% 的硬限制)
    
    # ========== 执行器配置 ==========
    actuators={
        # 腿部关节执行器配置
        "legs": DCMotorCfg(
            # 适用的关节(使用正则表达式匹配所有腿部关节)
            joint_names_expr=[
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",  # 前左腿
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",  # 前右腿
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",  # 后左腿
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",  # 后右腿
            ],
            
            # 力矩限制(单位: N·m)
            effort_limit=33.5,  # 最大输出力矩(来自 URDF 中的 effort 属性)
            saturation_effort=33.5,  # 饱和力矩(电机最大持续输出)
            
            # 速度限制(单位: rad/s)
            velocity_limit=21,  # 最大角速度(来自 URDF 中的 velocity 属性)
            
            # PD 增益配置(用于关节控制)
            stiffness=80.0,  # 刚度增益(P 增益) - 修正为 50.0 (适配 ~20kg 机身质量)
            damping=2.0,  # 阻尼增益(D 增益)
            friction=0.0,  # 静摩擦系数
        ),
        
        # 脊椎关节执行器配置 (固定脊椎)
         "spine": DCMotorCfg(
            joint_names_expr=[".*_spine_joint"],
            effort_limit=50,
            saturation_effort=50,
            velocity_limit=30,
            stiffness=150.0,  # 保持较高刚度固定脊椎
            damping=3,
            friction=0.0,
        ),



        # # 脊椎关节执行器配置 (固定脊椎)
        #  "spine": DCMotorCfg(
        #     joint_names_expr=[".*_spine_joint"],
        #     effort_limit=33.5,
        #     saturation_effort=33.5,
        #     velocity_limit=21,
        #     stiffness=80.0,  # 保持较高刚度固定脊椎
        #     damping=1,
        #     friction=0.0,
        # ),

    },
)


# ==================== Himmy Mark2 No Spine 机器人配置 ====================
# 与 HIMMY_MARK2_CFG 相同，但脊柱关节为固定刚体，无脊柱执行器
HIMMY_MARK2_NO_SPINE_CFG = ArticulationCfg(
    # ========== 模型加载和仿真设置 ==========
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Himmy/mark2_description/urdf/mark2_no_spine.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0,
                damping=0,
            )
        ),
    ),
    # ========== 初始状态配置 ==========
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # ========== 执行器配置 ==========
    actuators={
        # 腿部关节执行器配置（与 mark2 相同）
        "legs": DCMotorCfg(
            joint_names_expr=[
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            ],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21,
            stiffness=80.0,
            damping=0.0,
            friction=0.0,
        ),
        # 无脊柱执行器 - 脊柱关节已在 URDF 中设为 fixed
    },
)
