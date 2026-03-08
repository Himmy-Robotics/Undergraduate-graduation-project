# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
MDP（马尔科夫决策过程）函数模块

本模块包含了足式机器人速度跟踪任务的所有MDP相关函数。
这些函数定义了强化学习环境的核心要素，并在配置文件中被引用。

模块结构：
├── commands.py       - 命令生成器（定义机器人的目标行为）
├── observations.py   - 观察函数（定义策略的输入状态）
├── rewards.py        - 奖励函数（定义学习目标和行为塑造）
├── events.py         - 事件函数（定义域随机化策略）
└── curriculums.py    - 课程学习函数（定义训练难度递增策略）

使用方式：
这些函数通过配置文件（如 rough_env_cfg.py）中的 RewardTermCfg、ObsTermCfg 等
配置类被引用，从而构建完整的强化学习环境。

继承关系：
1. 首先导入 isaaclab 和 isaaclab_tasks 的基础MDP函数
2. 然后导入本项目的自定义MDP函数，覆盖或扩展基础功能
"""

# 导入 Isaac Lab 核心MDP函数（提供基础功能）
from isaaclab.envs.mdp import *  # noqa: F401, F403

# 导入 Isaac Lab Tasks 的速度跟踪MDP函数（提供速度任务的标准功能）
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

# 导入本项目的自定义MDP函数（扩展和定制功能）
from .commands import *      # 命令生成器
from .curriculums import *   # 课程学习函数
from .events import *        # 事件和域随机化函数
from .observations import *  # 观察函数
from .rewards import *       # 奖励函数
