# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab.actuators import DCMotorCfg

from .rough_env_cfg import AnymalDRoughEnvCfg


@configclass
class AnymalDFlatEnvCfg(AnymalDRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        # Massively increase height reward to force standing up
        self.rewards.base_height_l2.weight = 10.0
        self.rewards.base_height_l2.params["target_height"] = 0.45
        
        # Reduce penalties that might prevent standing
        self.rewards.stand_still.weight = -0.5  # Reduce from -2.0 to allow learning to stand first
        self.rewards.joint_pos_limits.weight = -1.0  # Reduce from -5.0 during early training
        
        # ------------------------------Physics & Control------------------------------
        # Switch to PD Control (easier to learn standing)
        self.scene.robot.actuators["legs"] = DCMotorCfg(
            joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
            saturation_effort=120.0,
            effort_limit=80.0,
            velocity_limit=7.5,
            stiffness={".*": 40.0},
            damping={".*": 5.0},
        )
        
        # Increase action scale for better control authority
        self.actions.joint_pos.scale = 0.5
        
        # Spawn robot in a standing position (not on the ground)
        self.events.randomize_reset_base.params["pose_range"]["z"] = (0.55, 0.65)

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "AnymalDFlatEnvCfg":
            self.disable_zero_weight_rewards()
