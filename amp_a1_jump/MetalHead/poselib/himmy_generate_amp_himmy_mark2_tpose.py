# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This script imports the Himmy Mark2 MJCF XML file and converts the skeleton into a SkeletonTree format.
It then generates a zero rotation pose, and adjusts the pose into a T-Pose.

Himmy Mark2 has 15 DOF:
- 4 legs x 3 DOF (hip, thigh, calf) = 12 DOF (same as A1)
- 3 spine DOF (yaw, pitch, roll) to mimic animal spine

Joint names in mark2.xml:
['trunk', 'FL_hip_Link', 'FL_thigh_Link', 'FL_calf_Link', 'FL_foot',
 'FR_hip_Link', 'FR_thigh_Link', 'FR_calf_Link', 'FR_foot',
 'yaw_spine_Link', 'pitch_spine_Link', 'roll_spine_Link', 'R_body_Link',
 'RL_hip_Link', 'RL_thigh_Link', 'RL_calf_Link', 'RL_foot',
 'RR_hip_Link', 'RR_thigh_Link', 'RR_calf_Link', 'RR_foot']
"""

import numpy as np
import os
import sys
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving images
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add poselib to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'poselib'))

from poselib.core.rotation3d import quat_mul, quat_from_angle_axis
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
# from poselib.visualization.common import plot_skeleton_state  # Skip interactive plotting


def generate_himmy_mark2_tpose(xml_path: str, output_path: str, visualize: bool = True, save_image: bool = True):
    """
    Generate T-pose for Himmy Mark2 robot.
    
    Args:
        xml_path: Path to the mark2.xml file
        output_path: Path to save the T-pose .npy file
        visualize: Whether to show interactive visualization
        save_image: Whether to save the T-pose image
    """
    
    # Import MJCF file
    print(f"Loading skeleton from: {xml_path}")
    skeleton = SkeletonTree.from_mjcf(xml_path)
    
    # Print skeleton info
    print(f"\nSkeleton node names ({len(skeleton.node_names)} nodes):")
    for i, name in enumerate(skeleton.node_names):
        print(f"  {i}: {name}")
    
    print(f"\nLocal translations shape: {skeleton.local_translation.shape}")
    print(f"Parent indices: {skeleton.parent_indices}")
    
    # Generate zero rotation pose
    zero_pose = SkeletonState.zero_pose(skeleton)
    
    print(f"\nZero pose info:")
    print(f"  Local rotation shape: {zero_pose.local_rotation.shape}")
    print(f"  Root translation: {zero_pose.root_translation}")
    print(f"  Global translation shape: {zero_pose.global_translation.shape}")
    
    # Get local rotation for adjustment
    local_rotation = zero_pose.local_rotation
    
    # Define joint offset for T-pose (similar to A1)
    # The thigh should be slightly forward and calf should be bent back
    joint_offset = 0  # degrees for thigh/calf pitch
    
    # Define hip offset for external rotation (外八角度)
    # This matches the source T-pose where legs are splayed outward
    # Source T-pose: LeftUpLeg has -5.0 deg X rotation, RightUpLeg has +5.0 deg X rotation
    # For Himmy: Left legs need negative X rotation, Right legs need positive X rotation
    hip_external_rotation = -0.0  # degrees - adjustable parameter
    
    # Adjust front legs
    # FL (Front Left) leg - hip external rotation (向外张开)
    if "FL_hip_Link" in skeleton.node_names:
        local_rotation[skeleton.node_names.index("FL_hip_Link")] = quat_mul(
            quat_from_angle_axis(angle=torch.tensor([-hip_external_rotation]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True),
            local_rotation[skeleton.node_names.index("FL_hip_Link")]
        )
    if "FL_thigh_Link" in skeleton.node_names:
        local_rotation[skeleton.node_names.index("FL_thigh_Link")] = quat_mul(
            quat_from_angle_axis(angle=torch.tensor([joint_offset]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
            local_rotation[skeleton.node_names.index("FL_thigh_Link")]
        )
    if "FL_calf_Link" in skeleton.node_names:
        local_rotation[skeleton.node_names.index("FL_calf_Link")] = quat_mul(
            quat_from_angle_axis(angle=torch.tensor([-joint_offset * 2]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
            local_rotation[skeleton.node_names.index("FL_calf_Link")]
        )
    
    # FR (Front Right) leg - hip external rotation (向外张开)
    if "FR_hip_Link" in skeleton.node_names:
        local_rotation[skeleton.node_names.index("FR_hip_Link")] = quat_mul(
            quat_from_angle_axis(angle=torch.tensor([hip_external_rotation]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True),
            local_rotation[skeleton.node_names.index("FR_hip_Link")]
        )
    if "FR_thigh_Link" in skeleton.node_names:
        local_rotation[skeleton.node_names.index("FR_thigh_Link")] = quat_mul(
            quat_from_angle_axis(angle=torch.tensor([joint_offset]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
            local_rotation[skeleton.node_names.index("FR_thigh_Link")]
        )
    if "FR_calf_Link" in skeleton.node_names:
        local_rotation[skeleton.node_names.index("FR_calf_Link")] = quat_mul(
            quat_from_angle_axis(angle=torch.tensor([-joint_offset * 2]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
            local_rotation[skeleton.node_names.index("FR_calf_Link")]
        )
    
    # Adjust rear legs
    # RL (Rear Left) leg - hip external rotation (向外张开)
    if "RL_hip_Link" in skeleton.node_names:
        local_rotation[skeleton.node_names.index("RL_hip_Link")] = quat_mul(
            quat_from_angle_axis(angle=torch.tensor([-hip_external_rotation]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True),
            local_rotation[skeleton.node_names.index("RL_hip_Link")]
        )
    if "RL_thigh_Link" in skeleton.node_names:
        local_rotation[skeleton.node_names.index("RL_thigh_Link")] = quat_mul(
            quat_from_angle_axis(angle=torch.tensor([joint_offset]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
            local_rotation[skeleton.node_names.index("RL_thigh_Link")]
        )
    if "RL_calf_Link" in skeleton.node_names:
        local_rotation[skeleton.node_names.index("RL_calf_Link")] = quat_mul(
            quat_from_angle_axis(angle=torch.tensor([-joint_offset * 2]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
            local_rotation[skeleton.node_names.index("RL_calf_Link")]
        )
    
    # RR (Rear Right) leg - hip external rotation (向外张开)
    if "RR_hip_Link" in skeleton.node_names:
        local_rotation[skeleton.node_names.index("RR_hip_Link")] = quat_mul(
            quat_from_angle_axis(angle=torch.tensor([hip_external_rotation]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True),
            local_rotation[skeleton.node_names.index("RR_hip_Link")]
        )
    if "RR_thigh_Link" in skeleton.node_names:
        local_rotation[skeleton.node_names.index("RR_thigh_Link")] = quat_mul(
            quat_from_angle_axis(angle=torch.tensor([joint_offset]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
            local_rotation[skeleton.node_names.index("RR_thigh_Link")]
        )
    if "RR_calf_Link" in skeleton.node_names:
        local_rotation[skeleton.node_names.index("RR_calf_Link")] = quat_mul(
            quat_from_angle_axis(angle=torch.tensor([-joint_offset * 2]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
            local_rotation[skeleton.node_names.index("RR_calf_Link")]
        )
    
    # Print hip external rotation info
    print(f"\nHip external rotation (外八角度): {hip_external_rotation} degrees")
    print("  Left legs (FL, RL): -{0} deg X rotation".format(hip_external_rotation))
    print("  Right legs (FR, RR): +{0} deg X rotation".format(hip_external_rotation))
    
    # Spine joints - keep at zero rotation for T-pose
    # The spine joints (yaw, pitch, roll) are already at zero rotation
    # which is the neutral standing position
    print("\nSpine joints kept at zero rotation for T-pose:")
    for spine_joint in ["yaw_spine_Link", "pitch_spine_Link", "roll_spine_Link"]:
        if spine_joint in skeleton.node_names:
            print(f"  {spine_joint}: index {skeleton.node_names.index(spine_joint)}")
    
    # Adjust root translation to make feet touch the ground
    translation = zero_pose.root_translation
    global_trans = zero_pose.global_translation
    
    # Find the minimum z-coordinate (lowest point, should be feet)
    min_z = torch.min(global_trans[:, 2])
    translation[2] -= min_z
    
    print(f"\nAdjusted root translation: {translation}")
    print(f"Min Z before adjustment: {min_z}")
    
    # Save T-pose
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    zero_pose.to_file(output_path)
    print(f"\nT-pose saved to: {output_path}")
    
    # Reload and verify
    target_tpose = SkeletonState.from_file(output_path)
    
    # Print foot positions
    print("\nFoot positions in T-pose:")
    for foot_name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]:
        if foot_name in skeleton.node_names:
            idx = skeleton.node_names.index(foot_name)
            pos = target_tpose.global_translation[idx]
            print(f"  {foot_name}: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
    
    # Calculate robot dimensions
    print("\nRobot dimensions from T-pose:")
    fl_foot_idx = skeleton.node_names.index("FL_foot") if "FL_foot" in skeleton.node_names else None
    fr_foot_idx = skeleton.node_names.index("FR_foot") if "FR_foot" in skeleton.node_names else None
    rl_foot_idx = skeleton.node_names.index("RL_foot") if "RL_foot" in skeleton.node_names else None
    trunk_idx = skeleton.node_names.index("trunk") if "trunk" in skeleton.node_names else 0
    
    if fl_foot_idx and rl_foot_idx:
        body_length = torch.abs(target_tpose.global_translation[fl_foot_idx, 0] - 
                               target_tpose.global_translation[rl_foot_idx, 0])
        print(f"  Body length (FL-RL): {body_length:.4f}m")
    
    if fl_foot_idx and fr_foot_idx:
        body_width = torch.abs(target_tpose.global_translation[fl_foot_idx, 1] - 
                              target_tpose.global_translation[fr_foot_idx, 1])
        print(f"  Body width (FL-FR): {body_width:.4f}m")
    
    trunk_height = target_tpose.global_translation[trunk_idx, 2]
    print(f"  Trunk height: {trunk_height:.4f}m")
    
    # Save image
    if save_image:
        save_tpose_image(target_tpose, output_path.replace('.npy', '.png'))
    
    # Show interactive visualization
    if visualize:
        print("\nSkipping interactive visualization (matplotlib Agg backend)")
        # plot_skeleton_state(target_tpose)  # Requires display
    
    return target_tpose


def save_tpose_image(skeleton_state, image_path: str):
    """
    Save T-pose visualization to an image file.
    
    Args:
        skeleton_state: SkeletonState object
        image_path: Path to save the image
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Get global translations
    global_trans = skeleton_state.global_translation.numpy()
    skeleton = skeleton_state.skeleton_tree
    
    # Create three views: front, side, top
    views = [
        (1, 'Front View (Y-Z)', 1, 2, 'Y', 'Z'),
        (2, 'Side View (X-Z)', 0, 2, 'X', 'Z'),
        (3, 'Top View (X-Y)', 0, 1, 'X', 'Y')
    ]
    
    for subplot_idx, title, dim1, dim2, xlabel, ylabel in views:
        ax = fig.add_subplot(1, 3, subplot_idx)
        
        # Plot joints
        ax.scatter(global_trans[:, dim1], global_trans[:, dim2], c='blue', s=50)
        
        # Plot bones (connections between parent-child)
        for i, parent_idx in enumerate(skeleton.parent_indices):
            if parent_idx >= 0:  # Not root
                ax.plot([global_trans[parent_idx, dim1], global_trans[i, dim1]],
                       [global_trans[parent_idx, dim2], global_trans[i, dim2]], 
                       'b-', linewidth=2)
        
        # Annotate joint names
        for i, name in enumerate(skeleton.node_names):
            ax.annotate(name, (global_trans[i, dim1], global_trans[i, dim2]), 
                       fontsize=6, alpha=0.7)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Himmy Mark2 T-Pose', fontsize=14)
    plt.tight_layout()
    plt.savefig(image_path, dpi=150, bbox_inches='tight')
    print(f"T-pose image saved to: {image_path}")
    plt.close()


def save_tpose_3d_image(skeleton_state, image_path: str):
    """
    Save 3D T-pose visualization to an image file.
    
    Args:
        skeleton_state: SkeletonState object
        image_path: Path to save the image
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get global translations
    global_trans = skeleton_state.global_translation.numpy()
    skeleton = skeleton_state.skeleton_tree
    
    # Define colors for different parts
    colors = {
        'trunk': 'red',
        'FL': 'green',
        'FR': 'blue', 
        'RL': 'orange',
        'RR': 'purple',
        'spine': 'cyan',
        'R_body': 'magenta'
    }
    
    # Plot joints with colors
    for i, name in enumerate(skeleton.node_names):
        color = 'gray'
        for key in colors:
            if key in name:
                color = colors[key]
                break
        ax.scatter(global_trans[i, 0], global_trans[i, 1], global_trans[i, 2], 
                  c=color, s=100, label=name if i < 10 else None)
        ax.text(global_trans[i, 0], global_trans[i, 1], global_trans[i, 2], 
               name, fontsize=7)
    
    # Plot bones
    for i, parent_idx in enumerate(skeleton.parent_indices):
        if parent_idx >= 0:
            ax.plot([global_trans[parent_idx, 0], global_trans[i, 0]],
                   [global_trans[parent_idx, 1], global_trans[i, 1]],
                   [global_trans[parent_idx, 2], global_trans[i, 2]], 
                   'k-', linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Himmy Mark2 T-Pose (3D View)')
    
    # Set equal aspect ratio
    max_range = np.max(np.abs(global_trans)) * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range * 2])
    
    plt.tight_layout()
    image_path_3d = image_path.replace('.png', '_3d.png')
    plt.savefig(image_path_3d, dpi=150, bbox_inches='tight')
    print(f"3D T-pose image saved to: {image_path_3d}")
    plt.close()


if __name__ == "__main__":
    # Path configuration
    # XML file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # The xml file is located at:
    # /data/zmli/Fast-Quadruped/source/robot_lab/data/Robots/Himmy/mark2_description/xml/mark2.xml
    # Script is at:
    # /data/zmli/Fast-Quadruped/amp_a1_jump/MetalHead/poselib/generate_amp_himmy_mark2_tpose.py
    # Use absolute path directly
    xml_path = "/data/zmli/Fast-Quadruped/source/robot_lab/data/Robots/Himmy/mark2_description/xml/mark2.xml"
    
    # Output path for T-pose
    output_path = os.path.join(script_dir, "data/amp_himmy_mark2_tpose.npy")
    
    print("=" * 60)
    print("Generating Himmy Mark2 T-Pose")
    print("=" * 60)
    print(f"XML path: {xml_path}")
    print(f"Output path: {output_path}")
    print("=" * 60)
    
    # Generate T-pose
    tpose = generate_himmy_mark2_tpose(
        xml_path=xml_path,
        output_path=output_path,
        visualize=True,  # Set to True to show interactive plot
        save_image=True   # Save image for verification
    )
    
    # Also save 3D view
    save_tpose_3d_image(tpose, output_path.replace('.npy', '.png'))
