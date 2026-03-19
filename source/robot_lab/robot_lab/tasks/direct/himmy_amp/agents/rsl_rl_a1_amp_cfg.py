import os
import glob
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# =============================================================================
# Motion Files Configuration
# =============================================================================
# A1 target motion files
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MOTION_DIR_A1 = os.path.abspath(os.path.join(CURRENT_DIR, "../motions/datasets/A1_dataset"))

# Load all A1 motion files
MOTION_FILES_A1 = glob.glob(os.path.join(MOTION_DIR_A1, "*.txt")) + glob.glob(os.path.join(MOTION_DIR_A1, "*.json"))

# Use empty list if none found
if not MOTION_FILES_A1:
    print(f"Warning: No motion files found in {MOTION_DIR_A1}")
    MOTION_FILES_A1 = []
else:
    print(f"Found {len(MOTION_FILES_A1)} motion files in {MOTION_DIR_A1}")

@configclass
class A1AmpPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 25000  # 用户提及25000轮可学会
    save_interval = 50
    experiment_name = "a1_amp"
    empirical_normalization = False
    
    # Runner specific
    runner_class_name = "AMPOnPolicyRunner"
    class_name = "AMPOnPolicyRunner"
    algorithm_class_name = "AMPPPO"
    policy_class_name = "ActorCritic"
    
    amp_reward_coef = 2.0
    amp_motion_files = MOTION_FILES_A1
    amp_num_preload_transitions = 2000000
    amp_task_reward_lerp = 0.3  # 与 a1 jump amp 中 amp_task_reward_lerp = 0.3 对应
    amp_discr_hidden_dims = [1024, 512]
    
    # 判别器相关参数
    amp_replay_buffer_size = 1000000
    
    # 最小标准差配置：4 腿 × 3 关节 = 12 个动作
    min_normalized_std = [0.05, 0.02, 0.05] * 4
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],  # 与 a1_amp_jump 保持一致
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=5.0,  # a1_amp_jump 中是 5.
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,     # a1_amp_jump 中是 0.
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,   # a1_amp_jump 中是 5e-4
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # AMP特定参数，rsl_rl或amp_runner可能需要（使用外部或额外传入）
        # bounds_loss_coef = 10.0
        # disc_coef = 5.0
        # disc_logit_reg = 0.05
        # disc_grad_penalty = 0.01
        # disc_weight_decay = 0.0001
    )
