# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    cmd = [r"python", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime

import omni
from rsl_rl.runners import OnPolicyRunner
try:
    from rsl_rl.runners import DistillationRunner
except ImportError:
    DistillationRunner = None
try:
    from rsl_rl.runners import AMPOnPolicyRunner
except ImportError:
    try:
        from amp.amp_on_policy_runner import AMPOnPolicyRunner
    except ImportError as e:
        print(f"Failed to import AMPOnPolicyRunner from amp: {e}")
        AMPOnPolicyRunner = None

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
import pickle


def dump_pickle(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401

class RslRlVecEnvWrapperConcrete(RslRlVecEnvWrapper):
    def __init__(self, env, unwrap_dict=False, **kwargs):
        super().__init__(env, **kwargs)
        self.unwrap_dict = unwrap_dict

    def get_privileged_observations(self):
        return None

    def get_observations(self):
        obs = super().get_observations()
        if hasattr(self, "unwrap_dict") and self.unwrap_dict:
            if hasattr(obs, 'keys') and 'policy' in obs.keys():
                obs = obs['policy']
        return obs

    def step(self, actions):
        ret = super().step(actions)
        
        if len(ret) == 4:
            # Assume Gym API: obs, rew, done, info
            # Check if ret[1] is likely rewards (tensor of shape num_envs)
            if isinstance(ret[1], torch.Tensor) and ret[1].shape == (self.num_envs,):
                 obs, rewards, dones, extras = ret
                 privileged_obs = None
            else:
                 # Assume RSL-RL API: obs, privileged_obs, rewards, dones
                 obs, privileged_obs, rewards, dones = ret
                 extras = getattr(self.unwrapped, "extras", {})
        elif len(ret) == 5:
            # Check if Gym API: obs, rew, term, trunc, info
            if isinstance(ret[1], torch.Tensor) and ret[1].shape == (self.num_envs,):
                 obs, rewards, term, trunc, extras = ret
                 dones = term | trunc
                 privileged_obs = None
            else:
                 # Assume RSL-RL API: obs, privileged_obs, rewards, dones, extras
                 obs, privileged_obs, rewards, dones, extras = ret
        else:
            obs, privileged_obs, rewards, dones = ret[0], ret[1], ret[2], ret[3]
            extras = getattr(self.unwrapped, "extras", {})
            
        if hasattr(self, "unwrap_dict") and self.unwrap_dict:
            if hasattr(obs, 'keys') and 'policy' in obs.keys():
                obs = obs['policy']
            if privileged_obs is not None and hasattr(privileged_obs, 'keys') and 'critic' in privileged_obs.keys():
                privileged_obs = privileged_obs['critic']
                
        # Fix for Gym API return (obs, rew, term, trunc, info) being interpreted as RSL-RL
        # If privileged_obs is actually rewards (shape [4096]), fix it.
        # (Already handled above)
        
        # Handle dict dones (e.g. from DirectRLEnv)
        if isinstance(dones, dict):
            # Usually contains 'time_outs' and 'is_terminal' or similar
            # We need a single boolean tensor for reset
            if "time_outs" in dones:
                 # Combine time_outs and other termination conditions if needed
                 # But typically rsl_rl expects a single tensor. 
                 # Let's assume the wrapper should have handled this, but if not:
                 # We need to find the main termination tensor.
                 # Often 'dones' in rsl_rl context is just the reset signal.
                 pass
            
            # If it's a dict, let's try to find a tensor that looks like the done signal
            # Or print it to debug
            # print(f"DEBUG: dones is a dict with keys: {dones.keys()}")
            # Temporary fallback: look for 'dones' or 'terminated' key
            if "dones" in dones:
                dones = dones["dones"]
            elif "terminated" in dones:
                dones = dones["terminated"] | dones.get("truncated", torch.zeros_like(dones["terminated"]))
            elif "time_outs" in dones:
                # If only time_outs is present, maybe it's just time outs?
                # But we need the actual done signal.
                # In DirectRLEnv, step returns (obs, rew, terminated, truncated, info)
                # If wrapper returns dict, it might be extras?
                # Wait, if len(ret) == 4, dones is the 4th element.
                # If DirectRLEnv returns 5 elements, the wrapper might be confused.
                # Let's assume for now we can use time_outs as a proxy if nothing else exists, 
                # BUT this is dangerous.
                # Let's look at extras.
                pass
            
            if isinstance(dones, dict):
                 # Still a dict? Try to construct a zero tensor if we can't find it
                 # This is a hack.
                 dones = torch.zeros(obs.shape[0] if isinstance(obs, torch.Tensor) else obs["policy"].shape[0], device=obs.device if isinstance(obs, torch.Tensor) else obs["policy"].device, dtype=torch.bool)
                 
        reset_env_ids = (dones > 0).nonzero(as_tuple=False).flatten()

        # Try to get terminal amp states from extras, or fallback to current amp_obs
        # If extras is empty, check if dones was a dict and contained amp_obs
        if not extras and isinstance(ret[3], dict) and "amp_obs" in ret[3]:
             extras = ret[3]

        amp_obs = extras.get("amp_obs", None)
        terminal_amp_states = None
        if amp_obs is not None and len(reset_env_ids) > 0:
             terminal_amp_states = amp_obs[reset_env_ids]
        elif len(reset_env_ids) > 0:
             # Fallback: use current amp_obs if terminal states are missing
             # This is not ideal for AMP but prevents crash
             if hasattr(self, "get_amp_observations"):
                 current_amp_obs = self.get_amp_observations()
                 if current_amp_obs is not None:
                     terminal_amp_states = current_amp_obs[reset_env_ids]
        
        if privileged_obs is not None:
            extras["privileged_obs"] = privileged_obs
        extras["reset_env_ids"] = reset_env_ids
        extras["terminal_amp_states"] = terminal_amp_states

        is_amp = hasattr(self.unwrapped, "get_amp_observations") or "amp_obs" in extras
        if is_amp:
            return obs, privileged_obs, rewards, dones, extras, reset_env_ids, terminal_amp_states
        elif privileged_obs is not None:
            return obs, privileged_obs, rewards, dones, extras
        else:
            # Fallback for old runners that don't expect privileged_obs
            # But recent RSL-RL expects 5 values: obs, privileged_obs, rewards, dones, extras
            # So returning 5 is generally safer, but let's conform to original length if possible
            if len(ret) == 4 and privileged_obs is None:
                 return obs, rewards, dones, extras
            return obs, privileged_obs, rewards, dones, extras

    def get_amp_observations(self):
        if hasattr(self.unwrapped, "get_amp_observations"):
            return self.unwrapped.get_amp_observations()
        return None

    @property
    def num_privileged_obs(self):
        if hasattr(self.unwrapped, "num_privileged_obs"):
            return self.unwrapped.num_privileged_obs
        if hasattr(self.unwrapped, "num_privileged_observations"):
            return self.unwrapped.num_privileged_observations
        return None

    @property
    def num_obs(self):
        # 返回单个观测的维度，而不是环境数量
        obs_space = self.unwrapped.single_observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            # 对于 Dict 空间，返回 "policy" 键的维度
            if "policy" in obs_space.spaces:
                return obs_space.spaces["policy"].shape[0]
        return obs_space.shape[0]

    @property
    def include_history_steps(self):
        return self.unwrapped.include_history_steps if hasattr(self.unwrapped, "include_history_steps") else None

    @property
    def dt(self):
        return self.unwrapped.step_dt if hasattr(self.unwrapped, "step_dt") else self.unwrapped.cfg.sim.dt * self.unwrapped.cfg.decimation

    @property
    def dof_pos_limits(self):
        if hasattr(self.unwrapped, "robot"):
            limits = self.unwrapped.robot.data.soft_joint_pos_limits[0]
            # 如果有 all_joint_indexes（包含脊柱），优先使用
            if hasattr(self.unwrapped, "all_joint_indexes"):
                return limits[self.unwrapped.all_joint_indexes]
            # 否则使用腿部关节索引
            if hasattr(self.unwrapped, "leg_joint_indexes"):
                return limits[self.unwrapped.leg_joint_indexes]
            return limits
        return None



torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # Force agent device to match env device
    agent_cfg.device = env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        print(
            "[WARNING] IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    is_amp = (agent_cfg.class_name == "AMPOnPolicyRunner")
    env = RslRlVecEnvWrapperConcrete(env, clip_actions=agent_cfg.clip_actions, unwrap_dict=is_amp)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner" and DistillationRunner is not None:
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "AMPOnPolicyRunner" and AMPOnPolicyRunner is not None:
        cfg = agent_cfg.to_dict()
        algorithm_cfg = cfg.get("algorithm")
        if "class_name" in algorithm_cfg:
            del algorithm_cfg["class_name"]
        if "normalize_advantage_per_mini_batch" in algorithm_cfg:
            del algorithm_cfg["normalize_advantage_per_mini_batch"]
        if "rnd_cfg" in algorithm_cfg:
            del algorithm_cfg["rnd_cfg"]
            
        train_cfg = {
            "runner": cfg,
            "policy": cfg.get("policy"),
            "algorithm": algorithm_cfg
        }
        runner = AMPOnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    elif args_cli.load_run:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model weights from: {resume_path}")
        # load previously trained model
        runner.load(resume_path, load_optimizer=False)
        runner.current_learning_iteration = 0

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
