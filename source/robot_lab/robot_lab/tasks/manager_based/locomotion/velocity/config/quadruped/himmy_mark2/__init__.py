# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Himmy Mark2 四足机器人强化学习环境注册模块.

该模块将 Himmy Mark2 机器人的环境配置注册为 Gymnasium 环境，
使其可以与标准的强化学习框架集成。
"""

import gymnasium as gym

from . import agents

##
# 注册 Gymnasium 环境
##

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Himmy-Mark2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:HimmyMark2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HimmyMark2RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:HimmyMark2RoughTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Himmy-Mark2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:HimmyMark2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HimmyMark2FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:HimmyMark2FlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Run-Himmy-Mark2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_run_cfg:HimmyMark2FlatRunEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HimmyMark2FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:HimmyMark2FlatTrainerCfg",
    },
)
gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Run-NoSpine-Himmy-Mark2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.no_spine_flat_env_run_cfg:HimmyMark2NoSpineFlatRunEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HimmyMark2FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:HimmyMark2FlatTrainerCfg",
    },
)


# ==================== 高速转向任务 ====================

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Run-Turn-Himmy-Mark2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_run_turn_cfg:HimmyMark2FlatRunTurnEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HimmyMark2FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:HimmyMark2FlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Run-Turn-NoSpine-Himmy-Mark2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.no_spine_flat_env_run_turn_cfg:HimmyMark2NoSpineFlatRunTurnEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HimmyMark2FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:HimmyMark2FlatTrainerCfg",
    },
)


# ==================== 空中翻转着陆任务 ====================

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Twist-Himmy-Mark2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_twist_cfg:HimmyMark2AirTwistEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HimmyMark2FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:HimmyMark2FlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Twist-NoSpine-Himmy-Mark2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.no_spine_flat_env_twist_cfg:HimmyMark2NoSpineAirTwistEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HimmyMark2FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:HimmyMark2FlatTrainerCfg",
    },
)
