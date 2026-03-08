# Robot Lab 框架使用指南

## 📚 目录
1. [框架概述](#框架概述)
2. [代码架构](#代码架构)
3. [配置系统详解](#配置系统详解)
4. [调用流程](#调用流程)
5. [开发自己的项目](#开发自己的项目)
6. [最佳实践](#最佳实践)

---

## 框架概述

### 🎯 项目定位
Robot Lab 是一个基于 **Isaac Lab** 的足式机器人强化学习训练框架，专注于速度跟踪运动控制任务。

### 🏗️ 核心设计理念
- **配置驱动**：所有参数通过配置类管理，易于修改和复用
- **层次继承**：基类定义通用功能，子类针对具体机器人定制
- **模块化设计**：场景、观察、动作、奖励等各模块独立可配
- **并行训练**：支持数千个环境并行仿真（GPU加速）

---

## 代码架构

### 📂 目录结构
```
robot_lab/
├── source/robot_lab/robot_lab/
│   ├── tasks/
│   │   └── manager_based/
│   │       └── locomotion/
│   │           └── velocity/
│   │               ├── velocity_env_cfg.py       # 🔴 基础配置类
│   │               ├── mdp/                       # MDP函数实现
│   │               │   ├── rewards.py
│   │               │   ├── observations.py
│   │               │   └── ...
│   │               └── config/
│   │                   └── quadruped/
│   │                       └── unitree_go2/
│   │                           ├── rough_env_cfg.py    # 🟡 机器人配置
│   │                           ├── flat_env_cfg.py     # 🟢 简化配置
│   │                           └── agents/
│   │                               └── rsl_rl_ppo_cfg.py  # 算法配置
│   └── assets/
│       └── unitree.py                             # 机器人资产定义
└── scripts/
    └── reinforcement_learning/
        └── rsl_rl/
            ├── train.py                           # 训练脚本
            └── play.py                            # 推理脚本
```

### 🔗 配置继承关系

```
ManagerBasedRLEnvCfg (Isaac Lab基类)
    ↓
LocomotionVelocityRoughEnvCfg (velocity_env_cfg.py)
    ├─ MySceneCfg              # 场景配置
    ├─ ObservationsCfg         # 观察配置
    ├─ ActionsCfg              # 动作配置
    ├─ CommandsCfg             # 命令配置
    ├─ RewardsCfg              # 奖励配置
    ├─ TerminationsCfg         # 终止条件配置
    ├─ EventCfg                # 事件配置
    └─ CurriculumCfg           # 课程学习配置
    ↓
UnitreeGo2RoughEnvCfg (rough_env_cfg.py)
    ↓
UnitreeGo2FlatEnvCfg (flat_env_cfg.py)
```

---

## 配置系统详解

### 1. 场景配置 (MySceneCfg)

**作用**：定义物理仿真环境

```python
@configclass
class MySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(...)      # 地形
    robot: ArticulationCfg = MISSING       # 机器人（子类指定）
    height_scanner = RayCasterCfg(...)     # 高度扫描传感器
    contact_forces = ContactSensorCfg(...) # 接触力传感器
    sky_light = AssetBaseCfg(...)          # 光照
```

**关键参数**：
- `num_envs`: 并行环境数量（推荐2048-4096）
- `env_spacing`: 环境间距（避免碰撞）
- `terrain_type`: 地形类型（"plane"平面 / "generator"生成器）

### 2. 观察配置 (ObservationsCfg)

**作用**：定义强化学习的状态空间

```python
@configclass
class ObservationsCfg:
    class PolicyCfg(ObsGroup):  # Actor网络输入（带噪声）
        base_lin_vel = ObsTerm(...)
        base_ang_vel = ObsTerm(...)
        projected_gravity = ObsTerm(...)
        velocity_commands = ObsTerm(...)
        joint_pos = ObsTerm(...)
        joint_vel = ObsTerm(...)
        actions = ObsTerm(...)
        height_scan = ObsTerm(...)
    
    class CriticCfg(ObsGroup):  # Critic网络输入（无噪声）
        # 同Policy但无噪声
```

**设计原则**：
- Policy观察加噪声 → 提高鲁棒性
- Critic观察无噪声 → 准确价值估计
- 使用 `scale` 归一化 → 稳定训练

### 3. 动作配置 (ActionsCfg)

**作用**：定义策略网络输出

```python
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],      # 控制所有关节
        scale=0.5,               # 动作幅度缩放
        use_default_offset=True  # 相对于默认姿态
    )
```

**控制模式**：
- `JointPositionActionCfg`: 位置控制（推荐）
- `JointVelocityActionCfg`: 速度控制
- `JointEffortActionCfg`: 力矩控制（高级）

### 4. 奖励配置 (RewardsCfg)

**作用**：定义学习目标（最重要！）

```python
@configclass
class RewardsCfg:
    # 任务目标
    track_lin_vel_xy_exp = RewTerm(weight=3.0)      # 跟踪xy速度
    track_ang_vel_z_exp = RewTerm(weight=1.5)       # 跟踪角速度
    
    # 行为塑造
    lin_vel_z_l2 = RewTerm(weight=-2.0)             # 惩罚跳跃
    ang_vel_xy_l2 = RewTerm(weight=-0.05)           # 保持水平
    joint_torques_l2 = RewTerm(weight=-2.5e-5)      # 节能
    action_rate_l2 = RewTerm(weight=-0.01)          # 平滑控制
    
    # 约束条件
    undesired_contacts = RewTerm(weight=-1.0)       # 避免身体触地
    joint_pos_limits = RewTerm(weight=-5.0)         # 避免关节极限
```

**奖励设计技巧**：
1. **主要目标**权重最大（1.0~5.0）
2. **行为塑造**权重适中（0.01~1.0）
3. **约束惩罚**权重视情况而定（-5.0~-0.01）
4. **权重平衡**：正奖励总和 ≈ 负奖励总和

### 5. 事件配置 (EventCfg)

**作用**：域随机化提高鲁棒性

```python
@configclass
class EventCfg:
    # 启动时触发
    randomize_rigid_body_material = EventTerm(mode="startup", ...)
    randomize_rigid_body_mass = EventTerm(mode="startup", ...)
    
    # 重置时触发
    randomize_reset_base = EventTerm(mode="reset", ...)
    randomize_actuator_gains = EventTerm(mode="reset", ...)
    
    # 间隔触发
    randomize_push_robot = EventTerm(mode="interval", ...)
```

**随机化对象**：
- 物理参数：质量、摩擦力、恢复系数
- 初始状态：位置、姿态、速度
- 执行器：刚度、阻尼
- 外部扰动：推力、力矩

---

## 调用流程

### 🔄 训练流程

```
1. 启动训练
   scripts/reinforcement_learning/rsl_rl/train.py
   ↓
2. 加载环境配置
   from config.quadruped.unitree_go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
   ↓
3. 创建环境
   env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=UnitreeGo2RoughEnvCfg)
   ↓
4. 初始化场景
   - 生成地形
   - 创建机器人
   - 初始化传感器
   ↓
5. 训练循环
   for epoch in range(max_epochs):
       for step in range(steps_per_epoch):
           ├─ 获取观察 obs = env.get_observations()
           ├─ 策略推理 action = policy(obs)
           ├─ 执行动作 obs, reward, done, info = env.step(action)
           ├─ 存储经验 buffer.add(obs, action, reward, done)
           └─ 更新策略 policy.update(buffer)
       ├─ 课程学习 curriculum.update()
       └─ 保存模型 save_checkpoint()
```

### 📊 配置加载顺序

```
1. 基类实例化
   LocomotionVelocityRoughEnvCfg()
   ├─ 设置默认值（所有奖励权重=0）
   ├─ 定义基础场景
   └─ 配置传感器
   ↓
2. 子类覆盖（UnitreeGo2RoughEnvCfg）
   super().__post_init__()
   ├─ 指定机器人模型
   ├─ 设置关节名称
   ├─ 配置奖励权重
   ├─ 调整观察缩放
   └─ 设置动作范围
   ↓
3. 进一步简化（UnitreeGo2FlatEnvCfg）
   super().__post_init__()
   ├─ 改为平坦地形
   ├─ 移除高度扫描
   └─ 禁用地形课程
   ↓
4. 禁用零权重奖励
   self.disable_zero_weight_rewards()
```

### ⚙️ 运行时数据流

```
┌─────────────────┐
│  Environment    │
│   Manager       │
└────────┬────────┘
         │
    ┌────┴────┐
    │  MDP    │
    │ Manager │
    └────┬────┘
         │
    ┌────┼────┐─────────────┐───────────┐
    │    │    │             │           │
┌───▼───┐│┌───▼───┐   ┌────▼────┐  ┌───▼────┐
│Observ │││Action │   │ Reward  │  │Terminal│
│Manager││Manager│   │ Manager │  │Manager │
└───────┘│└───────┘   └─────────┘  └────────┘
         │
    ┌────▼────┐
    │ Event   │
    │ Manager │
    └─────────┘
```

---

## 开发自己的项目

### 🚀 快速开始：添加新机器人

#### 步骤1：定义机器人资产

创建 `source/robot_lab/robot_lab/assets/my_robot.py`：

```python
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

MY_ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/path/to/robot.usd",  # 机器人USD文件路径
        rigid_props=sim_utils.RigidBodyPropertiesCfg(...),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(...),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),  # 初始位置
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.785,
            ".*_calf_joint": -1.57,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=80.0,  # 刚度
            damping=2.0,     # 阻尼
        ),
    },
)
```

#### 步骤2：创建环境配置

创建 `source/robot_lab/robot_lab/tasks/.../my_robot/rough_env_cfg.py`：

```python
from isaaclab.utils import configclass
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg
)
from robot_lab.assets.my_robot import MY_ROBOT_CFG

@configclass
class MyRobotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """我的机器人粗糙地形配置"""
    
    # 定义机器人特定信息
    base_link_name = "base_link"
    foot_link_name = ".*_foot"
    joint_names = [
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
    ]
    
    def __post_init__(self):
        super().__post_init__()
        
        # ===== 场景配置 =====
        self.scene.robot = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/{self.base_link_name}"
        
        # ===== 观察配置 =====
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        
        # ===== 动作配置 =====
        self.actions.joint_pos.joint_names = self.joint_names
        self.actions.joint_pos.scale = 0.3  # 根据机器人调整
        
        # ===== 奖励配置 =====
        # 速度跟踪（主要目标）
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        
        # 基座稳定
        self.rewards.lin_vel_z_l2.weight = -1.5
        self.rewards.ang_vel_xy_l2.weight = -0.05
        
        # 关节约束
        self.rewards.joint_torques_l2.weight = -2e-5
        self.rewards.action_rate_l2.weight = -0.01
        
        # 接触约束
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            f"^(?!.*{self.foot_link_name}).*"
        ]
        
        # 禁用零权重奖励
        if self.__class__.__name__ == "MyRobotRoughEnvCfg":
            self.disable_zero_weight_rewards()
```

#### 步骤3：注册环境

在 `source/robot_lab/robot_lab/tasks/.../velocity/__init__.py` 添加：

```python
import gymnasium as gym
from . import agents
from .my_robot.rough_env_cfg import MyRobotRoughEnvCfg

gym.register(
    id="Isaac-Velocity-Rough-MyRobot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MyRobotRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.MyRobotRoughPPORunnerCfg,
    },
)
```

#### 步骤4：训练

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-MyRobot-v0 \
    --num_envs 4096 \
    --headless
```

---

### 🎨 高级定制

#### 自定义奖励函数

在 `mdp/rewards.py` 添加：

```python
def my_custom_reward(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """自定义奖励：例如惩罚足部相对速度"""
    # 获取足部速度
    foot_vel = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids]
    
    # 计算相对速度
    left_foot = foot_vel[:, 0]
    right_foot = foot_vel[:, 1]
    relative_vel = torch.norm(left_foot - right_foot, dim=-1)
    
    # 返回奖励（惩罚过大的相对速度）
    return -relative_vel
```

在配置中使用：

```python
my_custom = RewTerm(
    func=mdp.my_custom_reward,
    weight=-0.1,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot", "right_foot"])}
)
```

#### 自定义观察

```python
def robot_energy(env) -> torch.Tensor:
    """观察：机器人能量消耗"""
    torques = env.scene["robot"].data.applied_torque
    velocities = env.scene["robot"].data.joint_vel
    power = torch.abs(torques * velocities).sum(dim=-1)
    return power.unsqueeze(-1)
```

添加到观察配置：

```python
robot_energy = ObsTerm(
    func=mdp.robot_energy,
    scale=0.01,
    clip=(0, 100)
)
```

---

## 最佳实践

### ✅ 开发流程建议

```
1. 从简单开始
   ├─ 先在平坦地形训练 (FlatEnvCfg)
   ├─ 使用较少的环境数量 (num_envs=512)
   └─ 启用可视化调试 (headless=False)

2. 逐步增加难度
   ├─ 验证平坦地形性能良好
   ├─ 切换到粗糙地形 (RoughEnvCfg)
   └─ 增加环境数量 (num_envs=4096)

3. 调优奖励权重
   ├─ 记录各奖励项数值
   ├─ 分析哪些奖励不work
   └─ 迭代调整权重

4. 测试鲁棒性
   ├─ 启用域随机化
   ├─ 测试不同地形
   └─ 真机部署前仿真验证
```

### 🎯 参数调优建议

#### 奖励权重调优

```python
# 1. 先设置主要目标（速度跟踪）
track_lin_vel_xy_exp.weight = 1.0  # 基准权重

# 2. 记录训练日志，观察各奖励项数值
# 假设观察到：
# - track_lin_vel_xy_exp 平均值: 0.5
# - lin_vel_z_l2 平均值: 0.1
# - joint_torques_l2 平均值: 0.001

# 3. 调整权重使各项贡献均衡
track_lin_vel_xy_exp.weight = 2.0    # 主要目标加大
lin_vel_z_l2.weight = -2.0           # 使惩罚与主要奖励量级接近
joint_torques_l2.weight = -2e-5      # 小项用小权重微调
```

#### 动作缩放调优

```python
# 太大：机器人动作过激，容易摔倒
self.actions.joint_pos.scale = 1.0  # ❌

# 太小：机器人动作幅度不足，无法完成任务
self.actions.joint_pos.scale = 0.1  # ❌

# 适中：根据机器人关节范围调整
self.actions.joint_pos.scale = 0.3  # ✅
```

#### 并行环境数量

```python
# CPU训练
num_envs = 64    # 少量环境

# GPU训练
num_envs = 2048  # 标准配置
num_envs = 4096  # 高端GPU (>16GB显存)
num_envs = 8192  # 多卡或A100

# 显存不足时减少环境数量或使用更小的机器人模型
```

### 🐛 常见问题解决

#### 1. 机器人一直摔倒

```python
# 检查：
# - 初始高度是否合适？
self.scene.robot.init_state.pos = (0, 0, 0.4)  # 适当高度

# - 是否惩罚了姿态倾斜？
self.rewards.flat_orientation_l2.weight = -1.0

# - 是否惩罚了非法接触？
self.rewards.undesired_contacts.weight = -2.0
```

#### 2. 速度跟踪效果差

```python
# 检查：
# - 速度跟踪奖励权重是否足够大？
self.rewards.track_lin_vel_xy_exp.weight = 3.0  # 增大

# - 动作幅度是否足够？
self.actions.joint_pos.scale = 0.5  # 增大

# - 是否过度惩罚了能量消耗？
self.rewards.joint_torques_l2.weight = -1e-5  # 减小惩罚
```

#### 3. 训练不收敛

```python
# 检查：
# - 奖励权重是否平衡？ （正奖励 ≈ 负奖励）
# - 观察是否正确归一化？
self.observations.policy.joint_vel.scale = 0.05

# - 学习率是否合适？
# 在 agents/rsl_rl_ppo_cfg.py 中调整
learning_rate = 1e-4  # 降低学习率
```

---

## 📚 参考资源

### 代码示例
- **Unitree Go2**: `config/quadruped/unitree_go2/`
- **其他机器人**: `config/quadruped/` 下各子目录

### 文档
- Isaac Lab 官方文档: https://isaac-sim.github.io/IsaacLab/
- RSL-RL 算法库: https://github.com/leggedrobotics/rsl_rl

### 相关论文
- **Learning to Walk in Minutes**: Rudin et al., CoRL 2022
- **Learning Quadrupedal Locomotion over Challenging Terrain**: Lee et al., Science Robotics 2020

---

## 🎓 总结

### 核心要点
1. **配置驱动**：修改配置文件而非代码
2. **继承复用**：从基类继承，只覆盖需要的部分
3. **奖励为王**：花时间设计好的奖励函数
4. **渐进训练**：从简单到复杂，从平坦到粗糙
5. **域随机化**：提高sim-to-real迁移能力

### 开发检查清单
- [ ] 定义机器人资产配置
- [ ] 创建环境配置类
- [ ] 设置奖励权重
- [ ] 配置观察和动作
- [ ] 注册环境
- [ ] 在平坦地形测试
- [ ] 切换到粗糙地形
- [ ] 启用域随机化
- [ ] 调优超参数
- [ ] 真机部署

### 下一步
- 阅读具体机器人的配置示例
- 运行训练脚本熟悉流程
- 尝试修改奖励权重观察效果
- 开发自己的机器人环境

祝您训练顺利！🚀

---

**维护**: Robot Lab Team | **更新**: 2025-10

