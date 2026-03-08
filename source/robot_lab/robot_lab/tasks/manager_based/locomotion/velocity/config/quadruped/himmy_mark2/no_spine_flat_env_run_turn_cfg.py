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
from robot_lab.assets._Himmy_robot import HIMMY_MARK2_NO_SPINE_CFG  # isort: skip


@configclass
class HimmyMark2NoSpineFlatRunTurnEnvCfg(LocomotionVelocityRoughEnvCfg):
    """
    Himmy Mark2 (无脊柱) 四足机器人在平坦地形上的高速转向任务环境配置
    
    该配置用于训练机器人（无脊柱版本）：
    1. 先直线加速到目标速度
    2. 延迟一段时间后施加角速度命令
    3. 仅依靠腿部完成高速转向（用于与有脊柱版本对比）
    
    设计目的：
    - 作为对照组，测试无脊柱时高速转向的表现
    - 与有脊柱版本进行对比实验
    """   
    # 机器人基座连杆名称
    base_link_name = "base_link"
    # 足部连杆名称
    foot_link_name = ".*_foot_Link"
    # fmt: off
    # 12个关节名称列表（4条腿 × 3个关节，无脊柱关节）
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",  # 前右腿
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",  # 前左腿
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",  # 后右腿
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",  # 后左腿
    ]
    # fmt: on

    def __post_init__(self):
        """初始化配置参数（在类实例化后自动调用）"""
        # 调用父类的初始化方法
        super().__post_init__()

        # ------------------------------场景配置------------------------------
        # 使用无脊柱的机器人模型
        self.scene.robot = HIMMY_MARK2_NO_SPINE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # --- Flat Environment Settings ---
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.scene.sky_light.spawn.texture_file = None
        self.scene.terrain.visual_material = None

        # ------------------------------观察值配置------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        
        # 指定需要观察的关节（仅 12 个腿部关节）
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names  
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names   

        # ------------------------------动作配置------------------------------
        # 无脊柱，仅腿部关节
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.125, 
            "^(?!.*_hip_joint).*": 0.25
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
        self.rewards.is_terminated.weight = 0

        # 基座惩罚项
        self.rewards.lin_vel_z_l2.weight = -0.0
        self.rewards.ang_vel_xy_l2.weight = -0.00
        self.rewards.flat_orientation_l2.weight = 0

        # 关节惩罚项（无需排除脊柱关节）
        self.rewards.joint_torques_l2.weight = -1.0e-6
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -1.2e-8
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_power.weight = -1.5e-5
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.003

        # 接触传感器奖励
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # 速度跟踪奖励
        self.rewards.track_lin_vel_xy_exp.weight = 4.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.8
        self.rewards.track_ang_vel_z_exp.weight = 3.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.7

        self.rewards.feet_air_time_variance.weight = -0.8
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.3
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]

        # Gallop 步态奖励
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
                "front_target_phase_diff": 0.06,
                "rear_target_phase_diff": 0.03,
            },
        )



        # Air Time 奖励
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time.weight = 1.2
        self.rewards.feet_air_time.params["threshold"] = 0.05

        self.rewards.upward.weight = 2.0

        # 如果奖励权重为0，则禁用该奖励项
        if self.__class__.__name__ == "HimmyMark2NoSpineFlatRunTurnEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------终止条件配置------------------------------
        self.terminations.illegal_contact = None

        # ------------------------------课程学习配置------------------------------
        self.curriculum.command_levels.params["range_multiplier"] = (0.1, 1.0)
        self.curriculum.terrain_levels = None
        
        # 角速度课程学习
        self.curriculum.ang_vel_levels = CurrTerm(
            func=mdp.command_levels_ang_vel,
            params={
                "reward_term_name": "track_ang_vel_z_exp",
                "range_multiplier": (0.1, 1.0),
            },
        )

        # ------------------------------命令配置------------------------------
        # 使用分阶段转向命令生成器：先直线跑，延迟后再转向
        self.commands.base_velocity = mdp.PhasedTurnAccVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 12.0),  # 较长的采样间隔，让转向持续足够时间
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(1, 6.0),     # 高速前进
                lin_vel_y=(-0.0, 0.0),    # 不允许侧向
                ang_vel_z=(-1.0, 1.0),    # 大角速度范围，测试转向极限（会被课程学习调整）
            ),
            acc_ranges=mdp.UniformAccelerationVelocityCommandCfg.AccRanges(
                lin_vel_x=(0.5, 3.0),     # 线速度加速度
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.5, 2.0),     # 角速度加速度
            ),
            turn_delay_range=(3, 6),  # 延迟3-6秒后开始转向
        )