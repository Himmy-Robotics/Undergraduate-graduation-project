#!/usr/bin/env python3
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Script to visualize AMP motion data on a robot.

This script loads AMP motion capture data and plays it back on the specified robot,
recording the visualization as a video file for remote viewing.

Usage:
    # Play all motions and record video
    python scripts/tools/view_amp_motion.py --task RobotLab-Isaac-Himmy-AMP-Direct-v0 --video --video_length 1000 --headless

    # Play a specific motion file
    python scripts/tools/view_amp_motion.py --task RobotLab-Isaac-Himmy-AMP-Direct-v0 --motion_file path/to/motion.txt --video --headless

    # Interactive viewing (requires display)
    python scripts/tools/view_amp_motion.py --task RobotLab-Isaac-Himmy-AMP-Direct-v0
"""

from __future__ import annotations

import argparse
import os
import sys

# Parse arguments before importing Isaac Sim
parser = argparse.ArgumentParser(description="Visualize AMP motion data on a robot.")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Himmy-AMP-Direct-v0", help="Name of the task/environment.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to specific motion file (optional). If not provided, uses all files from config.")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--video_name", type=str, default="amp_motion_vis", help="Name prefix of the recorded video.")
parser.add_argument("--playback_speed", type=float, default=1.0, help="Playback speed multiplier (1.0 = real-time).")
parser.add_argument("--loop", action="store_true", default=False, help="Loop through all motions continuously.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to visualize.")

# Add Isaac Sim launcher arguments
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras if video recording is requested
if args_cli.video:
    args_cli.enable_cameras = True
    args_cli.offscreen_render = True

# Launch Isaac Sim application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import Isaac Lab and other dependencies
import torch
import numpy as np
import json
import glob

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

# Import the motion loader - use local implementation
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reinforcement_learning", "rsl_rl"))
from amp.motion_loader import AMPLoader

# Import environment config to get motion files path
from robot_lab.tasks.direct.himmy_amp.agents.rsl_rl_amp_cfg import MOTION_FILES, MOTION_DIR


class AMPMotionVisualizer:
    """
    A class to visualize AMP motion data on a robot.
    """

    def __init__(self, args):
        self.args = args
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Setup simulation
        self._setup_simulation()
        
        # Load motion data
        self._load_motion_data()
        
        # Video recording setup
        self.video_writer = None
        self.frame_count = 0
        if args.video:
            self._setup_video_recording()

    def _setup_simulation(self):
        """Setup Isaac Sim simulation and robot."""
        # Create simulation context
        sim_cfg = sim_utils.SimulationCfg(
            dt=0.005,
            render_interval=1,
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view(eye=[2.5, 2.5, 1.5], target=[0.0, 0.0, 0.3])
        
        # Spawn ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # Load robot based on task
        self._load_robot()
        
        # Setup camera for video recording
        if self.args.video:
            self._setup_camera()
        
        # Run a few simulation steps to initialize
        self.sim.reset()
        for _ in range(10):
            self.sim.step()
            if self.args.video and self.camera is not None:
                self.camera.update(self.sim.cfg.dt)

    def _load_robot(self):
        """Load the robot articulation based on task."""
        # Import the appropriate robot config
        if "Himmy" in self.args.task:
            from robot_lab.assets._Himmy_robot import HIMMY_MARK2_CFG
            robot_cfg = HIMMY_MARK2_CFG.replace(prim_path="/World/Robot")
            # Adjust for AMP
            robot_cfg.init_state.pos = (0.0, 0.0, 0.42)
            robot_cfg.init_state.joint_pos = {
                ".*_hip_joint": 0.0,
                ".*_thigh_joint": 0.9,
                ".*_calf_joint": -1.8,
                ".*_spine_joint": 0.0,
            }
            robot_cfg.spawn.merge_fixed_joints = False
        else:
            # Default to A1
            from robot_lab.assets.unitree import UNITREE_A1_CFG
            robot_cfg = UNITREE_A1_CFG.replace(prim_path="/World/Robot")
            robot_cfg.init_state.pos = (0.0, 0.0, 0.42)
            robot_cfg.spawn.merge_fixed_joints = False
        
        self.robot = Articulation(robot_cfg)
        
        # Play the simulation to spawn the robot
        self.sim.reset()
        
        # Get joint info
        print(f"Robot joint names: {self.robot.data.joint_names}")
        print(f"Robot body names: {self.robot.data.body_names}")
        
        # Define leg joint names (FL, FR, RL, RR order to match AMPLoader output)
        self.leg_joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        self.leg_joint_indexes = [self.robot.data.joint_names.index(name) for name in self.leg_joint_names]

    def _setup_camera(self):
        """Setup camera for video recording."""
        try:
            camera_cfg = CameraCfg(
                prim_path="/World/Camera",
                update_period=0.0,  # Update every simulation step
                height=720,
                width=1280,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 100.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(2.5, 2.5, 1.5),
                    rot=(0.83147, 0.02706, 0.10569, 0.54382),  # Looking at origin
                    convention="world",
                ),
            )
            self.camera = Camera(camera_cfg)
            print("Camera setup complete.")
        except Exception as e:
            print(f"Warning: Could not setup camera: {e}")
            self.camera = None

    def _load_motion_data(self):
        """Load AMP motion data."""
        # Determine motion files to use
        if self.args.motion_file:
            motion_files = [self.args.motion_file]
        else:
            motion_files = MOTION_FILES
        
        print(f"Loading motion files from: {MOTION_DIR}")
        print(f"Motion files: {[os.path.basename(f) for f in motion_files]}")
        
        # Time between frames (policy dt)
        time_between_frames = 0.005 * 6  # dt * decimation = 0.03s
        
        self.amp_loader = AMPLoader(
            device=self.device,
            time_between_frames=time_between_frames,
            motion_files=motion_files,
            preload_transitions=False,
        )
        
        print(f"Loaded {len(self.amp_loader.trajectories)} trajectories:")
        for i, name in enumerate(self.amp_loader.trajectory_names):
            duration = self.amp_loader.trajectory_lens[i]
            frames = self.amp_loader.trajectory_num_frames[i]
            print(f"  [{i}] {os.path.basename(name)}: {duration:.2f}s, {int(frames)} frames")

    def _setup_video_recording(self):
        """Setup video recording using OpenCV."""
        try:
            import cv2
            self.cv2 = cv2
            
            video_dir = os.path.join("logs", "motion_vis")
            os.makedirs(video_dir, exist_ok=True)
            
            self.video_path = os.path.join(video_dir, f"{self.args.video_name}.mp4")
            
            # Video settings
            self.fps = 30
            self.video_width = 1280
            self.video_height = 720
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_path, fourcc, self.fps, 
                (self.video_width, self.video_height)
            )
            
            self.frames_buffer = []
            
            print(f"Recording video to: {self.video_path}")
            print(f"Video settings: {self.video_width}x{self.video_height} @ {self.fps}fps")
            
        except ImportError:
            print("Warning: OpenCV not available. Video recording disabled.")
            print("Install with: pip install opencv-python")
            self.video_writer = None

    def _capture_frame(self):
        """Capture a frame for video recording."""
        if self.video_writer is None or self.camera is None:
            return
        
        try:
            # Get RGB data from camera
            rgb_data = self.camera.data.output["rgb"]
            if rgb_data is not None and len(rgb_data) > 0:
                # Convert to numpy and format for OpenCV (BGR)
                frame = rgb_data[0].cpu().numpy()  # [H, W, 4] RGBA
                frame = frame[:, :, :3]  # Remove alpha channel -> RGB
                frame = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR)  # RGB -> BGR
                self.video_writer.write(frame)
                self.frame_count += 1
        except Exception as e:
            if self.frame_count == 0:
                print(f"Warning: Could not capture frame: {e}")

    def run(self):
        """Main visualization loop."""
        traj_idx = 0
        step_count = 0
        
        print("\n" + "="*60)
        print("Starting AMP Motion Visualization")
        print("="*60)
        
        while simulation_app.is_running():
            traj_name = os.path.basename(self.amp_loader.trajectory_names[traj_idx])
            traj_duration = self.amp_loader.trajectory_lens[traj_idx]
            
            print(f"\n▶ Playing: {traj_name} (duration: {traj_duration:.2f}s)")
            
            current_time = 0.0
            playback_dt = 0.005 * 6  # Match policy dt
            
            while current_time < traj_duration:
                # Get motion frame at current time
                frame = self.amp_loader.get_frame_at_time(traj_idx, current_time)
                
                # Extract state from frame
                # Frame format: [joint_pos(12), foot_pos(12), lin_vel(3), ang_vel(3), joint_vel(12)]
                # Note: root_pos and root_rot are in the full frame but get_frame_at_time returns truncated version
                
                # Get full frame for root state
                full_frame = self.amp_loader.get_full_frame_at_time(traj_idx, current_time)
                
                root_pos = AMPLoader.get_root_pos(full_frame)  # [3]
                root_rot = AMPLoader.get_root_rot(full_frame)  # [4] quaternion
                joint_pos = AMPLoader.get_joint_pose(full_frame)  # [12]
                joint_vel = AMPLoader.get_joint_vel(full_frame)  # [12]
                
                # Build full robot state
                full_joint_pos = self.robot.data.default_joint_pos[0].clone()
                full_joint_vel = torch.zeros_like(full_joint_pos)
                
                # Set leg joint positions
                full_joint_pos[self.leg_joint_indexes] = joint_pos
                full_joint_vel[self.leg_joint_indexes] = joint_vel
                
                # Build root state: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
                root_state = torch.zeros(13, device=self.device)
                root_state[:3] = root_pos
                root_state[2] += 0.1  # Small height offset to prevent ground collision
                root_state[3:7] = root_rot
                
                # Write state to simulation
                self.robot.write_root_link_pose_to_sim(root_state[:7].unsqueeze(0))
                self.robot.write_root_com_velocity_to_sim(root_state[7:].unsqueeze(0))
                self.robot.write_joint_state_to_sim(
                    full_joint_pos.unsqueeze(0),
                    full_joint_vel.unsqueeze(0)
                )
                
                # Step simulation for rendering
                self.sim.step()
                self.robot.update(self.sim.cfg.dt)
                
                # Update camera and capture frame
                if self.args.video and self.camera is not None:
                    self.camera.update(self.sim.cfg.dt)
                    self._capture_frame()
                
                # Advance time
                current_time += playback_dt / self.args.playback_speed
                step_count += 1
                
                # Progress indicator
                if step_count % 100 == 0:
                    print(f"  Frame {step_count}/{self.args.video_length if self.args.video else '∞'}")
                
                # Check video length limit
                if self.args.video and step_count >= self.args.video_length:
                    print(f"\n✓ Video recording complete ({self.frame_count} frames captured)")
                    print(f"  Saved to: {self.video_path}")
                    self._cleanup()
                    return
                
                # Check if simulation is still running
                if not simulation_app.is_running():
                    break
            
            # Move to next trajectory
            traj_idx = (traj_idx + 1) % len(self.amp_loader.trajectories)
            
            # If not looping and we've played all trajectories, exit
            if not self.args.loop and traj_idx == 0:
                print("\n✓ All trajectories played")
                break
        
        self._cleanup()

    def _cleanup(self):
        """Cleanup resources."""
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"Video saved to: {self.video_path}")
            print(f"Total frames: {self.frame_count}")


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("AMP Motion Data Visualizer")
    print("="*60)
    
    visualizer = AMPMotionVisualizer(args_cli)
    visualizer.run()
    simulation_app.close()


if __name__ == "__main__":
    main()
