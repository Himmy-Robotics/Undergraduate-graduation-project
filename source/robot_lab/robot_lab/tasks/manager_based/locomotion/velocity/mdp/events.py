# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
事件函数模块 - 域随机化（Domain Randomization）

本模块定义了用于强化学习环境的事件函数，主要用于域随机化。
域随机化是提高强化学习策略鲁棒性的重要技术，通过在训练时随机改变物理参数，
使得策略能够适应真实世界中的各种不确定性。

主要功能：
1. randomize_rigid_body_inertia - 随机化刚体惯性张量
2. randomize_com_positions - 随机化质心位置

这些函数通常在以下时机调用：
- startup: 环境启动时调用一次（如材质、质量等不变参数）
- reset: 每次环境重置时调用（如初始位置、速度等）
- interval: 按时间间隔调用（如外部扰动力）
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    随机化刚体的惯性张量（Domain Randomization）

    该函数用于随机化刚体的惯性张量，提高强化学习策略对物体惯性不确定性的鲁棒性。
    只随机化惯性张量的对角分量（Ixx, Iyy, Izz），非对角分量保持为0。

    Args:
        env: 环境对象
        env_ids: 需要随机化的环境ID列表，None表示所有环境
        asset_cfg: 场景实体配置，指定要随机化的资产和刚体
        inertia_distribution_params: 分布参数，如 (min, max) 或 (mean, std)
        operation: 随机化操作类型
            - "add": 加法操作，I_new = I_default + random
            - "scale": 乘法操作，I_new = I_default × random
            - "abs": 绝对值操作，I_new = random
        distribution: 随机分布类型
            - "uniform": 均匀分布
            - "log_uniform": 对数均匀分布
            - "gaussian": 高斯分布

    提示：
        此函数使用CPU张量来分配惯性值，建议仅在环境初始化时使用，
        以避免频繁的GPU-CPU数据传输影响性能。

    示例：
        # 将惯性在默认值基础上缩放0.5到1.5倍
        randomize_rigid_body_inertia(
            env, env_ids, asset_cfg,
            inertia_distribution_params=(0.5, 1.5),
            operation="scale",
            distribution="uniform"
        )
    """
    # 提取资产对象（启用类型提示）
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # 解析环境ID
    if env_ids is None:
        # 如果未指定，则处理所有环境
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        # 将环境ID转移到CPU（PhysX API需要CPU张量）
        env_ids = env_ids.cpu()

    # 解析刚体索引
    if asset_cfg.body_ids == slice(None):
        # 如果未指定，则处理所有刚体
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        # 转换为CPU张量
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # 获取当前的惯性张量（形状: num_assets × num_bodies × 9）
    # 9个元素表示3×3惯性矩阵（按行展开）
    inertias = asset.root_physx_view.get_inertias()

    # 先恢复为默认值，再在默认值基础上进行随机化
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # 随机化对角元素（Ixx, Iyy, Izz -> 索引 0, 4, 8）
    # 惯性矩阵布局：[Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz]
    #               [ 0,   1,   2,   3,   4,   5,   6,   7,   8 ]
    for idx in [0, 4, 8]:
        # 提取并随机化指定的对角元素
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],           # 当前对角元素的值
            inertia_distribution_params,   # 分布参数
            env_ids,                       # 环境ID
            body_ids,                      # 刚体ID
            operation,                     # 操作类型（add/scale/abs）
            distribution,                  # 分布类型（uniform/log_uniform/gaussian）
        )
        # 将随机化后的值赋回惯性张量
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # 将随机化后的惯性张量设置到物理仿真中
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_com_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    随机化刚体的质心（COM）位置（Domain Randomization）

    该函数用于随机化刚体的质心位置，模拟真实世界中质量分布的不确定性。
    质心位置的变化会影响物体的动力学特性，因此这种随机化可以提高策略的鲁棒性。

    Args:
        env: 仿真环境对象
        env_ids: 需要随机化的环境ID列表，None表示所有环境
        asset_cfg: 场景实体配置，指定要随机化的资产和刚体
        com_distribution_params: 分布参数，如 (min, max) 或 (mean, std)
        operation: 随机化操作类型
            - "add": 加法操作，COM_new = COM_default + random
            - "scale": 乘法操作，COM_new = COM_default × random
            - "abs": 绝对值操作，COM_new = random
        distribution: 随机分布类型
            - "uniform": 均匀分布
            - "log_uniform": 对数均匀分布
            - "gaussian": 高斯分布

    提示：
        此函数用于初始化或离线调整，因为它直接修改物理属性。

    示例：
        # 在默认质心位置基础上随机偏移±0.05米
        randomize_com_positions(
            env, env_ids, asset_cfg,
            com_distribution_params=(-0.05, 0.05),
            operation="add",
            distribution="uniform"
        )
    """
    # 提取资产对象（可以是Articulation或RigidObject）
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # 解析环境索引
    if env_ids is None:
        # 如果未指定，则处理所有环境
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        # 将环境ID转移到CPU（PhysX API需要CPU张量）
        env_ids = env_ids.cpu()

    # 解析刚体索引
    if asset_cfg.body_ids == slice(None):
        # 如果未指定，则处理所有刚体
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        # 转换为CPU张量
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # 获取当前的质心偏移量（形状: num_assets × num_bodies × 3）
    # 3个元素表示质心在刚体局部坐标系中的位置 (x, y, z)
    com_offsets = asset.root_physx_view.get_coms()

    # 独立地随机化x、y、z三个维度
    for dim_idx in range(3):
        # 随机化当前维度的质心偏移
        randomized_offset = _randomize_prop_by_op(
            com_offsets[:, :, dim_idx],    # 当前维度的质心偏移值
            com_distribution_params,        # 分布参数
            env_ids,                        # 环境ID
            body_ids,                       # 刚体ID
            operation,                      # 操作类型（add/scale/abs）
            distribution,                   # 分布类型（uniform/log_uniform/gaussian）
        )
        # 将随机化后的值赋回质心偏移张量
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

    # 将随机化后的质心偏移设置到物理仿真中
    asset.root_physx_view.set_coms(com_offsets, env_ids)


"""
内部辅助函数
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """
    根据指定的操作和分布对数据进行随机化（内部辅助函数）

    该函数是所有随机化函数的通用底层实现，支持多种操作和分布类型。

    Args:
        data: 待随机化的数据张量，形状为 (dim_0, dim_1)
        distribution_parameters: 分布参数（如均匀分布的 [min, max] 或高斯分布的 [mean, std]）
        dim_0_ids: 第一维度需要随机化的索引（通常是环境ID）
        dim_1_ids: 第二维度需要随机化的索引（通常是刚体ID或关节ID）
        operation: 随机化操作类型
            - "add": 加法，data_new = data_old + random
            - "scale": 乘法，data_new = data_old × random
            - "abs": 绝对值，data_new = random
        distribution: 随机分布类型
            - "uniform": 均匀分布 U(min, max)
            - "log_uniform": 对数均匀分布 exp(U(log(min), log(max)))
            - "gaussian": 高斯分布 N(mean, std)

    Returns:
        随机化后的数据张量，形状为 (dim_0, dim_1)

    Raises:
        NotImplementedError: 如果操作或分布类型不支持
    """
    # 解析张量形状和索引
    # -- 第一维度（通常是环境维度）
    if dim_0_ids is None:
        # 如果未指定，处理所有第一维度元素
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        # 如果指定了索引，计算需要处理的数量
        n_dim_0 = len(dim_0_ids)
        # 如果第二维度不是切片，需要将第一维度索引扩展为列向量
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]

    # -- 第二维度（通常是刚体/关节维度）
    if isinstance(dim_1_ids, slice):
        # 如果是切片，处理所有第二维度元素
        n_dim_1 = data.shape[1]
    else:
        # 如果是索引列表，计算需要处理的数量
        n_dim_1 = len(dim_1_ids)

    # 解析分布类型并获取对应的采样函数
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform      # 均匀分布采样
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform  # 对数均匀分布采样
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian     # 高斯分布采样
    else:
        raise NotImplementedError(
            f"未知的分布类型: '{distribution}'，用于关节属性随机化。"
            " 请使用 'uniform'、'log_uniform' 或 'gaussian'。"
        )

    # 执行随机化操作
    if operation == "add":
        # 加法操作: data_new = data_old + random_value
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        # 乘法操作: data_new = data_old × random_value
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        # 绝对值操作: data_new = random_value（直接设置为随机值）
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"未知的操作: '{operation}'，用于属性随机化。请使用 'add'、'scale' 或 'abs'。"
        )

    return data
