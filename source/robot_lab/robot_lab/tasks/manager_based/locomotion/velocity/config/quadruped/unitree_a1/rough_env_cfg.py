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
# use cloud assets
# from isaaclab_assets.robots.unitree import UNITREE_A1_CFG  # isort: skip
# use local assets
from robot_lab.assets.unitree import UNITREE_A1_CFG  # isort: skip


@configclass
class UnitreeA1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"
    # fmt: off
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
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
        self.rewards.ang_vel_xy_l2.weight = -0.005         # 惩罚x,y方向角速度（保持稳定）
        self.rewards.flat_orientation_l2.weight = 0       # 姿态平坦度惩罚（已禁用）

        # 关节惩罚项（Joint penalties）— 无需排除脊柱关节
        self.rewards.joint_torques_l2.weight = -2.5e-5   # 惩罚关节力矩（节能）
        self.rewards.joint_vel_l2.weight = 0              # 惩罚关节速度（已禁用）
        self.rewards.joint_acc_l2.weight = -2.5e-7       # 惩罚关节加速度（平滑运动）
        self.rewards.joint_pos_limits.weight = -5.0       # 惩罚接近关节位置极限
        self.rewards.joint_power.weight = -2e-5         # 功率惩罚，鼓励高效步态
        self.rewards.stand_still.weight = -2.0            # 惩罚静止不动
        self.rewards.joint_pos_penalty.weight = -1.0      # 关节位置惩罚
        # 仅惩罚髋关节位置偏差（无脊柱关节）

        self.rewards.action_rate_l2.weight = -0.008       # 惩罚动作变化率（平滑控制）

        # 接触传感器奖励（Contact sensor）
        self.rewards.undesired_contacts.weight = -1.0     # 惩罚非足部接触
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4       # 惩罚过大的接触力
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # 速度跟踪奖励（Velocity-tracking rewards）
        self.rewards.track_lin_vel_xy_exp.weight = 3   # 奖励跟踪目标xy平面线速度
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.8
        self.rewards.track_ang_vel_z_exp.weight = 1.5     # 奖励跟踪目标z轴角速度

        self.rewards.feet_air_time_variance.weight = -0.8  # 惩罚腾空时间方差
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.feet_contact.weight = 0              # 足部接触奖励（已禁用）
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0              # 足部绊倒惩罚（已禁用）
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.3             # 惩罚足部滑动
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
                "front_feet_names": ["FL_foot", "FR_foot"],
                "rear_feet_names": ["RL_foot", "RR_foot"],
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "front_target_phase_diff": 0.03,
                "rear_target_phase_diff": 0.02,
            },
        )

        self.rewards.feet_gait.weight = 0.0
        self.rewards.feet_gait.params["command_threshold"] = 4.0
        self.rewards.feet_gait.params["velocity_threshold"] = 3.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot_Link", "FR_foot_Link"), ("RR_foot_Link", "RL_foot_Link"))

        # Air Time 奖励
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time.weight = 1.2
        self.rewards.feet_air_time.params["threshold"] = 0.05

        self.rewards.upward.weight = 2.0                  # 奖励保持直立

        # 无脊柱：不添加 spine_motivation 奖励

        # 如果奖励权重为0，则禁用该奖励项
        if self.__class__.__name__ == "UnitreeA1RoughEnvCfg":
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
                lin_vel_x=(2, 2),
                lin_vel_y=(-0.0, 0.0),
                ang_vel_z=(0, 0),
            ),
            acc_ranges=mdp.UniformAccelerationVelocityCommandCfg.AccRanges(
                lin_vel_x=(0, 2.0),
                lin_vel_y=(0.5, 2.0),
                ang_vel_z=(0.5, 2.0),
            ),
        )
