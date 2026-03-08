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
class HimmyMark2FlatRunTurnEnvCfg(LocomotionVelocityRoughEnvCfg):
    """
    Himmy Mark2 四足机器人在平坦地形上的高速转向任务环境配置
    
    该配置用于训练机器人：
    1. 先直线加速到目标速度
    2. 延迟一段时间后施加角速度命令
    3. 学习利用脊柱(yaw_spine_joint)辅助高速转向
    
    设计目的：
    - 测试脊柱在高速转向中的作用
    - 方便与无脊柱版本进行对比实验
    """   
    # 机器人基座连杆名称
    base_link_name = "base_link"
    # 足部连杆名称（Mark2的URDF中足部连杆为 *_foot_Link）
    foot_link_name = ".*_foot_Link"
    # fmt: off
    # 15个关节名称列表（4条腿 × 3个关节 + 3个脊柱关节）
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
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        # 移除高度扫描（平坦地形不需要感知地形起伏）
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        
        # Disable remote assets to prevent network timeouts
        self.scene.sky_light.spawn.texture_file = None
        self.scene.terrain.visual_material = None

        # ------------------------------观察值配置------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        
        # 禁用某些观察值
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        
        # 指定需要观察的关节
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        
        # ------------------------------动作配置------------------------------
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.125, 
            ".*_spine_joint": 0.25, 
            "^(?!.*(_hip_joint|_spine_joint)).*": 0.25
        }
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------事件配置------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------奖励配置------------------------------
        # 通用奖励
        self.rewards.is_terminated.weight = 0  # 终止惩罚权重（已禁用）

        # 基座惩罚项（Root penalties）
        self.rewards.lin_vel_z_l2.weight = -0.0           # 惩罚z方向线速度（避免跳跃）
        self.rewards.ang_vel_xy_l2.weight = -0.0         # 惩罚x,y方向角速度（保持稳定）
        self.rewards.flat_orientation_l2.weight = 0       # 姿态平坦度惩罚（已禁用）


        # 关节惩罚项（Joint penalties）
        self.rewards.joint_torques_l2.weight = -1.0e-6   # 惩罚关节力矩（节能）
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = ["^(?!.*_spine_joint).*"]

        self.rewards.joint_vel_l2.weight = 0         # 惩罚关节速度（已禁用）
        self.rewards.joint_acc_l2.weight = -1.2e-8      # 惩罚关节加速度（平滑运动）
        


        self.rewards.joint_pos_limits.weight = -1.0       # 惩罚接近关节位置极限

        self.rewards.joint_power.weight = -1.5e-5         # (修改) 增加功率惩罚，鼓励高效的大步幅步态
        # self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = ["^(?!pitch_spine_joint).*"]

        self.rewards.stand_still.weight = -2.0            # 惩罚静止不动

        self.rewards.joint_pos_penalty.weight = -1.0      # 关节位置惩罚
        # self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = [".*_hip_joint", "yaw_spine_joint", "roll_spine_joint"]



        # 脊柱-前腿相位协同奖励: 鼓励脊柱与前腿大腿反向运动
        self.rewards.spine_motivation = RewTerm(
            func=mdp.spine_motivation,
            weight=0,
            params={
                "sigma": 10.0,
                "spine_joint_name": "pitch_spine_joint",
                "left_thigh_joint_name": "FL_thigh_joint",
                "right_thigh_joint_name": "FR_thigh_joint",
            },
        )



        self.rewards.action_rate_l2.weight = -0.003        # 惩罚动作变化率（平滑控制）

    

        # 接触传感器奖励（Contact sensor）
        self.rewards.undesired_contacts.weight = -1.0     # 惩罚非足部接触（如膝盖、身体触地）
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4       # 惩罚过大的接触力
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # 速度跟踪奖励（Velocity-tracking rewards）
        self.rewards.track_lin_vel_xy_exp.weight = 5   # 奖励跟踪目标xy平面线速度
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.8  # 放宽标准，更容易拿高分，加速课程学习
        self.rewards.track_ang_vel_z_exp.weight = 4     # 奖励跟踪目标z轴角速度
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.7  # 放宽标准，更容易拿高分，加速课程学习


        
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
            weight=0.8,
            params={
                "std": 0.02,
                "command_name": "base_velocity",
                "velocity_threshold": 0.5,
                "command_threshold": 0.1,
                "max_err": 0.4,
                "front_feet_names": ["FL_foot_Link", "FR_foot_Link"],
                "rear_feet_names": ["RR_foot_Link", "RL_foot_Link"],
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                # 分别设置前后脚的目标相位差
                "front_target_phase_diff": 0.04,  # 前脚时差
                "rear_target_phase_diff": 0.02,    # 后脚时差
            },
        )

        
        self.rewards.yaw_spine_alignment = RewTerm(
            func=mdp.yaw_spine_turn_alignment,
            weight=0.8,  # 根据训练效果调整
            params={
                "command_name": "base_velocity",
                "spine_joint_name": "yaw_spine_joint",
                "scale": 3.5,        
                "cmd_threshold": 0.3  # 直行死区
            }
        )



        self.rewards.feet_gait.weight = 0.0               # 奖励gallop步态
        # 修正：参数必须设置在 params 字典中，直接设置属性无效
        self.rewards.feet_gait.params["command_threshold"] = 4.0  # 速度命令阈值（仅在较高速度下奖励步态）
        self.rewards.feet_gait.params["velocity_threshold"] = 3.0  
        # self.rewards.feet_gait.params["std"] = 0.0                     # 步态奖励标准差
        # 定义同步的足部对
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot_Link", "FR_foot_Link"), ("RR_foot_Link", "RL_foot_Link"))
        # self.rewards.feet_gait.params["use_async_reward"] = True

        
         # 3. 配合调整 Air Time 奖励
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time.weight = 1.2
        self.rewards.feet_air_time.params["threshold"] = 0.05



        self.rewards.upward.weight = 2.0                  # 奖励向上的运动方向（保持直立）


        # 如果奖励权重为0，则禁用该奖励项
        if self.__class__.__name__ == "HimmyMark2FlatRunTurnEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------终止条件配置------------------------------
        self.terminations.illegal_contact = None

        # ------------------------------课程学习配置------------------------------
        # 线速度课程学习（从10%到100%）
        self.curriculum.command_levels.params["range_multiplier"] = (0.1, 1.0)
        self.curriculum.terrain_levels = None
        
        # 角速度课程学习（独立于线速度课程）
        # 从20%的角速度范围开始，逐步提升到100%
        # 例如：如果 ang_vel_z = (-2.0, 2.0)，初始范围为 (-0.4, 0.4)
        self.curriculum.ang_vel_levels = CurrTerm(
            func=mdp.command_levels_ang_vel,
            params={
                "reward_term_name": "track_ang_vel_z_exp",  # 根据角速度跟踪奖励升级
                "range_multiplier": (0.1, 1.0),  # 从20%开始，逐步提升到100%
            },
        )

        # ------------------------------命令配置------------------------------
        # 使用分阶段转向命令生成器
        self.commands.base_velocity = mdp.PhasedTurnAccVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 12.0),
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(2, 6.0),
                lin_vel_y=(-0.0, 0.0),
                ang_vel_z=(-1.5, 1.5),
            ),
            acc_ranges=mdp.UniformAccelerationVelocityCommandCfg.AccRanges(
                lin_vel_x=(0.5, 6.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 3.0),
            ),
            turn_delay_range=(5, 10),
        )

    


