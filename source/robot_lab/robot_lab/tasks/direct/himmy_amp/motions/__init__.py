# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
AMP Motion Loader and motion files for Himmy Mark2 robot.
"""

import os

# Get the directory of this file
MOTION_DIR = os.path.dirname(os.path.abspath(__file__))

# Mark2 AMP motion files (retargeted from AI4Animation dog dataset)
MOTION_FILES = [
    # Gallop motions
    os.path.join(MOTION_DIR, "mark2_gallop_stop.npy"),
    os.path.join(MOTION_DIR, "mark2_gallop_fwd0.npy"),
    os.path.join(MOTION_DIR, "mark2_gallop_fwd1.npy"),
    os.path.join(MOTION_DIR, "mark2_gallop_emerg_stop.npy"),
    
    # Trot motions
    os.path.join(MOTION_DIR, "mark2_trot_fwd0.npy"),
    os.path.join(MOTION_DIR, "mark2_trot_fwd1.npy"),
    
    # Turn motions
    os.path.join(MOTION_DIR, "mark2_trot_turn0.npy"),
    os.path.join(MOTION_DIR, "mark2_trot_turn1.npy"),
    os.path.join(MOTION_DIR, "mark2_trot_turn2.npy"),
    os.path.join(MOTION_DIR, "mark2_trot_turn3.npy"),
]

