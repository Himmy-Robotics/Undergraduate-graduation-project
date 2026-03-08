import os
import glob
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# =============================================================================
# Motion Files Configuration
# =============================================================================
# 使用 Himmy Mark2 专用的动作数据集（包含脊柱运动）
# mocap_motions_himmy: 15个txt文件，67维每帧（包含脊柱关节）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MOTION_DIR_HIMMY = os.path.abspath(os.path.join(CURRENT_DIR, "../motions/datasets/mocap_motions_himmy"))

# 加载所有 Himmy 动作文件
MOTION_FILES_HIMMY = glob.glob(os.path.join(MOTION_DIR_HIMMY, "*.txt"))

# 如果没找到文件，使用空列表（防止启动错误）
if not MOTION_FILES_HIMMY:
    print(f"Warning: No motion files found in {MOTION_DIR_HIMMY}")
    MOTION_FILES_HIMMY = []
else:
    print(f"Found {len(MOTION_FILES_HIMMY)} motion files in {MOTION_DIR_HIMMY}")

@configclass
class HimmyAmpPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 50
    experiment_name = "himmy_amp"
    empirical_normalization = False
    
    # Runner specific
    runner_class_name = "AMPOnPolicyRunner"
    class_name = "AMPOnPolicyRunner"
    algorithm_class_name = "AMPPPO"
    policy_class_name = "ActorCritic"
    
    amp_reward_coef = 2.0
    amp_motion_files = MOTION_FILES_HIMMY  # 使用 Himmy 专用动作数据
    amp_num_preload_transitions = 200000  # Reduced from 2000000 to save GPU memory
    amp_task_reward_lerp = 0.3
    amp_discr_hidden_dims = [1024, 512]
    
    # 最小标准差配置：4 腿 × 3 关节 + 3 脊柱关节 = 15 个动作
    min_normalized_std = [0.05, 0.02, 0.05] * 4 + [0.05, 0.05, 0.05]  # 腿部 + 脊柱

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
