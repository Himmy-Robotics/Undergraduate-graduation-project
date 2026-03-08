# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_name", type=str, default="rl-video", help="Name prefix of the recorded video.")
parser.add_argument("--viewer_world", action="store_true", default=False, help="Use fixed world camera instead of robot-following camera.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument("--debug_vis", action="store_true", default=False, help="Enable debug visualization.")
parser.add_argument("--log", action="store_true", default=False, help="Log spine joints, body attitude and other data to CSV for plotting.")
parser.add_argument("--log_no_spine", action="store_true", default=False, help="Log body attitude and velocity data to CSV (no spine joints, for no-spine robot).")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rl_utils import camera_follow, setup_video_viewport, PlayDataLogger

# --- PATCH START: Fix Warp compatibility for video recording ---
try:
    import omni.replicator.core.scripts.utils.annotator_utils as annotator_utils
    import warp as wp
    import carb
    import numpy as np

    def patched_reshape_output_ptr(
        ptr, height, width, strides, buffer_size, elem_count, source_device, target_device, wp_dtype, np_dtype, **kwargs
    ):
        type_size = wp.types.type_size_in_bytes(wp_dtype) if wp_dtype else np_dtype.itemsize
        if "shape" in kwargs:
            shape = tuple(kwargs.get("shape", 0))
            if len(shape) == 0:
                shape = 0
        elif height and width and elem_count:
            shape = (height, width, elem_count)
        elif buffer_size and width and height and wp_dtype:
            elem_count = buffer_size // height // width // type_size
            shape = (height, width, elem_count)
        else:
            shape = buffer_size // type_size
        carb.profiler.begin(1, "Wrap warp array")
        # PATCH: Removed owner=False to fix Warp compatibility
        data = wp.types.array(
            dtype=wp_dtype, shape=shape, strides=strides, ptr=ptr, device=source_device, requires_grad=False
        )
        # Calculate element count
        if buffer_size and width and height and type_size:
            elem_count = buffer_size // width // height // type_size

        # Data needs to be squeezed after retrieval otherwise strides don't match data
        if height and width and elem_count == 1:
            if not data.is_contiguous:
                data = data.contiguous()
            data = data.reshape((height, width))
        carb.profiler.end(1)

        if not target_device.startswith("cuda"):
            if data.size > 0:  # TEMP fix because warp errors out when having empty array.
                data = data.numpy()

                # Squeeze 3rd dimension if possible
                if len(data.shape) == 3 and data.shape[-1] == 1:
                    data = np.squeeze(data, axis=2)
                if len(data.shape) > 0 and data.shape[0] == 1:
                    data = np.squeeze(data, axis=0)

                data = data.view(np_dtype)
                return data
            else:
                data = np.array([])
        return data

    annotator_utils._reshape_output_ptr = patched_reshape_output_ptr
    print("[INFO] Applied Warp compatibility patch for video recording.")
except Exception as e:
    print(f"[WARNING] Failed to apply Warp patch: {e}")
# --- PATCH END ---

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner
try:
    from rsl_rl.runners import AMPOnPolicyRunner
except ImportError:
    try:
        from amp.amp_on_policy_runner import AMPOnPolicyRunner
    except ImportError:
        AMPOnPolicyRunner = None

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64
    print(f"[INFO] Play Num Envs: {env_cfg.scene.num_envs}")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # Enable debug visualization if requested
    if args_cli.debug_vis and hasattr(env_cfg, "debug_vis"):
        env_cfg.debug_vis = True

    # spawn the robot randomly in the grid (instead of their terrain levels)
    if hasattr(env_cfg.scene, "terrain"):
        env_cfg.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if env_cfg.scene.terrain.terrain_generator is not None:
            env_cfg.scene.terrain.terrain_generator.num_rows = 5
            env_cfg.scene.terrain.terrain_generator.num_cols = 5
            env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    if hasattr(env_cfg, "events") and env_cfg.events is not None:
        env_cfg.events.randomize_apply_external_force_torque = None
        env_cfg.events.push_robot = None
    if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
        env_cfg.curriculum.command_levels = None

    # 禁用速度命令和速度箭头的可视化（用于视频录制）
    if args_cli.video:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
            env_cfg.commands.base_velocity.debug_vis = False

        # === 论文级高质量视频录制设置 ===
        # 1. 提升视频分辨率到 1920x1080 (Full HD)
        env_cfg.viewer.resolution = (1920, 1080)

        # 2. 设置地面颜色为浅灰色（替换默认黑色）
        if hasattr(env_cfg.scene, "terrain"):
            from isaaclab.sim import schemas as sim_schemas
            import isaaclab.sim as sim_utils
            env_cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.5, 0.5),  # 浅灰色地面
            )

        # 3. 开启最高画质渲染模式 (quality preset)
        env_cfg.sim.render.rendering_mode = "quality"
        # 开启所有光影特效
        env_cfg.sim.render.enable_translucency = True      # 半透明效果
        env_cfg.sim.render.enable_reflections = True        # 反射效果
        env_cfg.sim.render.enable_global_illumination = True  # 全局光照
        env_cfg.sim.render.enable_direct_lighting = True    # 直接光照
        env_cfg.sim.render.enable_shadows = True            # 阴影
        env_cfg.sim.render.enable_ambient_occlusion = True  # 环境光遮蔽
        env_cfg.sim.render.enable_dl_denoiser = True        # DL降噪器
        env_cfg.sim.render.samples_per_pixel = 4            # 每像素采样数(更高=更清晰)
        env_cfg.sim.render.antialiasing_mode = "DLAA"       # 最高质量抗锯齿
        env_cfg.sim.render.dlss_mode = 2                    # DLSS Quality模式
        print("[INFO] Video recording: enabled HIGH QUALITY rendering (1920x1080, quality preset, all effects ON)")
    
    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        # resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        # if not resume_path:
        #     print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
        #     return
        raise NotImplementedError("Pre-trained checkpoints are not supported in this environment.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        # 设置viewport配置（light rig等）
        setup_video_viewport(env, world_mode=args_cli.viewer_world)
        
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": args_cli.video_name,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # initialize data logger if --log is enabled
    data_logger = None
    if args_cli.log:
        data_logger = PlayDataLogger(env, log_dir, log_spine=True)
        print(f"[INFO] Data logging enabled (with spine). Will save to: {data_logger.log_path}")
    elif args_cli.log_no_spine:
        data_logger = PlayDataLogger(env, log_dir, log_spine=False)
        print(f"[INFO] Data logging enabled (no spine). Will save to: {data_logger.log_path}")

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- PATCH START: Add missing attributes for AMPOnPolicyRunner ---
    # AMPOnPolicyRunner expects these attributes to be present on the environment
    if not hasattr(env, "num_privileged_obs"):
        env.num_privileged_obs = getattr(env.unwrapped, "num_privileged_obs", None)
    if not hasattr(env, "include_history_steps"):
        env.include_history_steps = getattr(env.unwrapped, "include_history_steps", None)
    if not hasattr(env, "num_obs"):
        if hasattr(env.unwrapped, "num_observations"):
            env.num_obs = env.unwrapped.num_observations
        elif hasattr(env.unwrapped, "observation_space") and env.unwrapped.observation_space is not None:
            # Handle vectorized observation space (num_envs, obs_dim)
            shape = env.unwrapped.observation_space.shape
            if shape and len(shape) == 2:
                env.num_obs = shape[1]
            elif shape:
                env.num_obs = shape[0]
            else:
                 # fallback
                 pass

            
    # FORCE FIX: If the checkpoint expects 42 but env reports 64, and we believe 42 is correct for the task logic,
    # we might need to investigate. But if the env actually produces 64, we can't just change the number.
    # However, if the env produces 42 but reports 64, we MUST fix it.
    
    if not hasattr(env, "num_actions"):
        if hasattr(env.unwrapped, "num_actions"):
            env.num_actions = env.unwrapped.num_actions
        elif hasattr(env.unwrapped, "action_space"):
            env.num_actions = env.unwrapped.action_space.shape[0]
    if not hasattr(env, "dt"):
        env.dt = getattr(env.unwrapped, "step_dt", 0.02)
    if not hasattr(env, "dof_pos_limits"):
        # Try to get dof_pos_limits from the robot data if available
        if hasattr(env.unwrapped, "robot") and hasattr(env.unwrapped.robot, "data"):
             # The shape of soft_joint_pos_limits is (num_envs, num_dof, 2)
             # We need (num_dof, 2)
             env.dof_pos_limits = env.unwrapped.robot.data.soft_joint_pos_limits[0, :, :]
        else:
             # Fallback or dummy if not accessible directly
             # Assuming standard limits or trying to infer
             pass
    
    # --- PATCH START: Fix dof_pos_limits shape mismatch ---
    # The error "The size of tensor a (12) must match the size of tensor b (15) at non-singleton dimension 0"
    # suggests that dof_pos_limits has 15 DOFs but min_normalized_std expects 12 (or vice versa).
    # The robot seems to have 12 DOFs (4 legs * 3 joints).
    # However, the limits might include other joints if not filtered.
    # Let's check if we can filter them based on action space size.
    if hasattr(env, "dof_pos_limits") and hasattr(env, "num_actions"):
        if env.dof_pos_limits.shape[0] != env.num_actions:
            print(f"[WARNING] dof_pos_limits shape {env.dof_pos_limits.shape} does not match num_actions {env.num_actions}. Truncating/Adjusting...")
            # Assuming the first num_actions DOFs are the ones we control
            env.dof_pos_limits = env.dof_pos_limits[:env.num_actions, :]
    # --- PATCH END ---

    # --- PATCH END ---

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "AMPOnPolicyRunner":
        # --- PATCH START: Fix AMPOnPolicyRunner config structure ---
        # AMPOnPolicyRunner expects a specific config structure:
        # { "runner": {...}, "algorithm": {...}, "policy": {...} }
        # agent_cfg.to_dict() might return a nested dict or a flat one depending on the config class.
        # We need to ensure the structure is correct.
        amp_cfg_dict = agent_cfg.to_dict()
        
        # Debug prints
        print(f"[DEBUG] env.num_obs: {getattr(env, 'num_obs', 'Not Set')}")
        print(f"[DEBUG] env.num_privileged_obs: {getattr(env, 'num_privileged_obs', 'Not Set')}")
        print(f"[DEBUG] env.include_history_steps: {getattr(env, 'include_history_steps', 'Not Set')}")
        print(f"[DEBUG] amp_cfg_dict keys: {list(amp_cfg_dict.keys())}")

        if "runner" not in amp_cfg_dict:
            # Construct the expected structure
            root_cfg = {}
            # The runner config usually contains top-level fields
            root_cfg["runner"] = amp_cfg_dict
            
            # Extract policy and algorithm configs if they exist
            if "policy" in amp_cfg_dict:
                root_cfg["policy"] = amp_cfg_dict["policy"]
            else:
                root_cfg["policy"] = amp_cfg_dict # Fallback, though likely wrong if flat
                
            if "algorithm" in amp_cfg_dict:
                root_cfg["algorithm"] = amp_cfg_dict["algorithm"]
            else:
                root_cfg["algorithm"] = amp_cfg_dict # Fallback
                
            amp_cfg = root_cfg
        else:
            amp_cfg = amp_cfg_dict
            
        runner = AMPOnPolicyRunner(env, amp_cfg, log_dir=None, device=agent_cfg.device)
        # --- PATCH END ---
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # Handle dict/TensorDict observations
            # if isinstance(obs, dict):
            #     obs_input = obs["policy"]
            # elif hasattr(obs, "keys") and "policy" in obs.keys():
            #     obs_input = obs["policy"]
            # else:
            #     obs_input = obs
                
            actions = policy(obs)
            # actions = torch.zeros_like(actions)
            # env stepping
            obs, _, _, _ = env.step(actions)

        # Log data if --log is enabled
        if data_logger is not None:
            data_logger.log_step(dt)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        
        # Always follow robot when recording video or using keyboard
        if args_cli.keyboard or args_cli.video:
            if not args_cli.viewer_world:
                camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # save logged data before closing
    if data_logger is not None:
        data_logger.save()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
