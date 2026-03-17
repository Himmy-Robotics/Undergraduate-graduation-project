import torch
import math
from isaaclab.utils.math import quat_mul, quat_apply, quat_from_angle_axis
def my_quat_rotate(q, v):
    q_w = q[:, 3]
    q_vec = q[:, 0:3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

def calc_heading_quat_inv(q):
    yaw = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
                     1 - 2 * (q[:, 1]**2 + q[:, 2]**2))
    heading_quat = quat_from_angle_axis(-yaw, torch.tensor([0., 0., 1.]).to(q.device).repeat(q.shape[0], 1))
    return heading_quat

def quat_to_tan_norm(q):
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = my_quat_rotate(q, ref_tan)
    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = my_quat_rotate(q, ref_norm)
    return torch.cat([tan, norm], dim=len(tan.shape) - 1)

def compute_a1_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]  # xyzw format assumed in IsaacLab/Gym
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    # In legged_gym, quat_rotate_inverse expects (w, x, y, z)? Wait!
    # Let's assume my_quat_rotate style since we can mock it here
    from isaaclab.utils.math import quat_apply_inverse
    local_root_vel = quat_apply_inverse(root_rot, root_vel)  # (N, 3)
    local_root_ang_vel = quat_apply_inverse(root_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
                                           local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0],
                                            local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    obs = torch.cat(
        (root_h, root_rot_obs[:, -3:], local_root_vel, local_root_ang_vel, dof_pos, dof_vel, flat_local_key_pos), dim=-1)
    return obs

root_states = torch.zeros((1, 13))
root_states[:, 3:7] = torch.tensor([0., 0., 0., 1.])
dof_pos = torch.zeros((1, 12))
dof_vel = torch.zeros((1, 12))
key_body_pos = torch.zeros((1, 4, 3))
out = compute_a1_observations(root_states, dof_pos, dof_vel, key_body_pos, False)
print('Output shape:', out.shape)
