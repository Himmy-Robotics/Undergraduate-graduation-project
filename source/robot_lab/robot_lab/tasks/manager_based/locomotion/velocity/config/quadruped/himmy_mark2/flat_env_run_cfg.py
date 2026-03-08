from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg

from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from robot_lab.assets._Himmy_robot import HIMMY_MARK2_CFG  # isort: skip


@configclass
class HimmyMark2FlatRunEnvCfg(LocomotionVelocityRoughEnvCfg):
    """
    Himmy Mark2 四足机器人在平坦地形上的跑步强化学习环境配置
    
    该配置类基于 HimmyMark2RoughEnvCfg 修改，用于定义 Mark2 机器人在
    平坦地形上进行跑步速度控制的运动学习任务。
    """
    # 机器人基座连杆名称
    base_link_name = "base_link"
    # 足部连杆名称（Mark2的URDF中足部连杆为 *_foot_Link）
    foot_link_name = ".*_foot_Link"
    # fmt: off
    # 12个关节名称列表（4条腿 × 3个关节）
    # FR: 前右, FL: 前左, RR: 后右, RL: 后左
    # hip: 髋关节, thigh: 大腿关节, calf: 小腿关节
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",  # 前右腿
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",  # 前左腿
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",  # 后右腿
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",  # 后左腿
        "yaw_spine_joint", "pitch_spine_joint", "roll_spine_joint", # 脊柱关节

    ]
    # fmt: on

    def __post_init__(self):
        """初始化配置参数（在类实例化后自动调用）"""
        # 调用父类的初始化方法
        super().__post_init__()

        # ------------------------------场景配置------------------------------
        # 设置机器人模型配置和在场景中的路径
        self.scene.robot = HIMMY_MARK2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # --- Flat Environment Settings ---
        # 将地形改为平面
        self.scene.terrain.terrain_type = "plane"          # 使用简单平面地形
        self.scene.terrain.terrain_generator = None        # 不使用地形生成器
        
        # 移除高度扫描（平坦地形不需要感知地形起伏）
        self.scene.height_scanner = None                   # 禁用场景高度扫描器
        self.scene.height_scanner_base = None              # 禁用基座高度扫描器
        
        # Disable remote assets to prevent network timeouts
        self.scene.sky_light.spawn.texture_file = None
        self.scene.terrain.visual_material = None

        # ------------------------------观察值配置------------------------------
        # 设置观察值的缩放因子（用于归一化输入到神经网络的数据）
        self.observations.policy.base_lin_vel.scale = 2.0       # 基座线速度缩放
        self.observations.policy.base_ang_vel.scale = 0.25      # 基座角速度缩放
        self.observations.policy.joint_pos.scale = 1.0          # 关节位置缩放
        self.observations.policy.joint_vel.scale = 0.05         # 关节速度缩放
        
        # 禁用某些观察值（设为 None）
        self.observations.policy.base_lin_vel = None            # 不使用基座线速度观察
        self.observations.policy.height_scan = None             # 不使用高度扫描观察
        self.observations.critic.height_scan = None             # 禁用价值网络的高度扫描观察

        # 指定需要观察的关节
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names  
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names  

        # ------------------------------动作配置------------------------------
        # 设置动作缩放（减小动作幅度以提高稳定性）
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.125, 
            ".*_spine_joint": 0.25, 
            "^(?!.*(_hip_joint|_spine_joint)).*": 0.25
        }
        # 设置动作裁剪范围
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        # 指定动作控制的关节
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------事件配置------------------------------
        # 配置环境重置时的随机化参数（用于域随机化，增强策略鲁棒性）
        self.events.randomize_reset_base.params = {
            "pose_range": {  # 位姿随机化范围
                "x": (-0.0, 0.0),          # x方向位置（米）
                "y": (-0.0, 0.0),          # y方向位置（米）
                "z": (0.0, 0.2),           # z方向位置（米）
                "roll": (0, 0),     # 滚转角（弧度）
                "pitch": (0, 0),    # 俯仰角（弧度）
                "yaw": (0, 0),      # 偏航角（弧度）
            },
            "velocity_range": {  # 速度随机化范围
                "x": (-0.0, 0.0),          # x方向线速度（米/秒）
                "y": (-0.0, 0.0),          # y方向线速度（米/秒）
                "z": (-0.0, 0.0),          # z方向线速度（米/秒）
                "roll": (-0.0, 0.0),       # 滚转角速度（弧度/秒）
                "pitch": (-0.0, 0.0),      # 俯仰角速度（弧度/秒）
                "yaw": (-0.0, 0.0),        # 偏航角速度（弧度/秒）
            },
        }
        # 随机化基座质量
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        # 随机化其他刚体质量（除基座外）
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        # 随机化质心位置
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        # 随机施加外力和力矩（模拟外部扰动）
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]


        self.episode_length_s = 40

        # ------------------------------奖励配置------------------------------
        # 通用奖励
        self.rewards.is_terminated.weight = 0  # 终止惩罚权重（已禁用）

        # 基座惩罚项（Root penalties）
        self.rewards.lin_vel_z_l2.weight = -0.0           # 惩罚z方向线速度（避免跳跃）
        self.rewards.ang_vel_xy_l2.weight = -0.0         # 惩罚x,y方向角速度（保持稳定）
        self.rewards.flat_orientation_l2.weight = 0       # 姿态平坦度惩罚（已禁用）


        # 关节惩罚项（Joint penalties）
        self.rewards.joint_torques_l2.weight = -1.0e-9   # 惩罚关节力矩（节能）
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = ["^(?!.*_spine_joint).*"]

        self.rewards.joint_vel_l2.weight = 0         # 惩罚关节速度（已禁用）
        self.rewards.joint_acc_l2.weight = -1.0e-8      # 惩罚关节加速度（平滑运动）
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = ["^(?!pitch_spine_joint).*"]        


        self.rewards.joint_pos_limits.weight = -1.0       # 惩罚接近关节位置极限
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = ["^(?!pitch_spine_joint).*"] 

        self.rewards.joint_power.weight = -1.4e-5         # (修改) 增加功率惩罚，鼓励高效的大步幅步态
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = ["^(?!pitch_spine_joint).*"]

        self.rewards.stand_still.weight = -2.0            # 惩罚静止不动

        self.rewards.joint_pos_penalty.weight = -2.0      # 关节位置惩罚
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = [".*_hip_joint", "yaw_spine_joint", "roll_spine_joint"]




        self.rewards.action_rate_l2.weight = -0.003        # 惩罚动作变化率（平滑控制）
        self.rewards.action_rate_l2.params["asset_cfg"].joint_names = ["^(?!pitch_spine_joint).*"]
    

        # 接触传感器奖励（Contact sensor）
        self.rewards.undesired_contacts.weight = -1.0     # 惩罚非足部接触（如膝盖、身体触地）
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4       # 惩罚过大的接触力
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # 速度跟踪奖励（Velocity-tracking rewards）
        self.rewards.track_lin_vel_xy_exp.weight = 5   # 奖励跟踪目标xy平面线速度
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.8  # 放宽标准，更容易拿高分，加速课程学习
        self.rewards.track_ang_vel_z_exp.weight = 1.5     # 奖励跟踪目标z轴角速度



        
        self.rewards.feet_air_time_variance.weight = -0.8  # 惩罚腾空时间方差（保持规律步态）
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]


        self.rewards.feet_contact.weight = 0              # 足部接触奖励（已禁用）
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1  # 奖励无运动命令时的足部接触（稳定站立）
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0              # 足部绊倒惩罚（已禁用）
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.3            # 惩罚足部滑动（避免打滑）
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]




        # Gallop 步态奖励：奖励前后脚左右两脚有一定的相位差
        self.rewards.gallop_gait = RewTerm(
            func=mdp.GallopGaitReward,
            weight=1.2,
            params={
                "std": 0.01,
                "command_name": "base_velocity",
                "velocity_threshold": 0.5,
                "command_threshold": 0.1,
                "max_err": 0.5,
                "front_feet_names": ["FL_foot_Link", "FR_foot_Link"],
                "rear_feet_names": ["RR_foot_Link", "RL_foot_Link"],
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                # 分别设置前后脚的目标相位差
                "front_target_phase_diff": 0.03,  # 前脚时差
                "rear_target_phase_diff": 0.015,    # 后脚时差
            },
        )

        # 脊柱-前腿相位协同奖励: 鼓励脊柱与前腿大腿反向运动
        self.rewards.spine_motivation = RewTerm(
            func=mdp.spine_motivation,
            weight=0.8,
            params={
                "sigma": 10.0,
                "spine_joint_name": "pitch_spine_joint",
                "leg_mode": "all",  
                "velocity_bias": 0,  
            },
        )




        
         # 3. 配合调整 Air Time 奖励
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time.weight = 1.2
        self.rewards.feet_air_time.params["threshold"] = 0.06



        self.rewards.upward.weight = 2.0                  # 奖励向上的运动方向（保持直立）

        # 如果奖励权重为0，则禁用该奖励项
        if self.__class__.__name__ == "HimmyMark2FlatRunEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------终止条件配置------------------------------
        self.terminations.illegal_contact = None          # 非法接触终止条件（已禁用）

        # ------------------------------课程学习配置------------------------------
        self.curriculum.command_levels.params["range_multiplier"] = (0.1, 1.0)
        self.curriculum.terrain_levels = None             # 禁用地形难度递增
        


        # ------------------------------命令配置------------------------------
        # 使用自定义的带加速度限制的命令生成器
        self.commands.base_velocity = mdp.UniformAccelerationVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            debug_vis=True,
            use_acceleration=False,  # 不使用加速度限制，直接发布目标速度
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(1, 1),   # 目标速度范围
                lin_vel_y=(-0.0, 0.0),
                ang_vel_z=(9.0, 9.0),
            ),
            acc_ranges=mdp.UniformAccelerationVelocityCommandCfg.AccRanges(
                lin_vel_x=(5, 5.0),   # 加速度范围 (0.5 ~ 2.0 m/s^2)
                lin_vel_y=(0.5, 2.0),
                ang_vel_z=(2, 2),
            ),
        )



