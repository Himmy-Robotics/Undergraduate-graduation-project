"""
Himmy Mark2 专用的 AMP Motion Loader

支持 67 维数据格式（包含 3 个脊柱关节）：
- [0:3] root_pos
- [3:7] root_rot (quaternion)
- [7:19] joint_pos (12 腿部关节)
- [19:22] spine_pos (3 脊柱关节: yaw, pitch, roll)
- [22:34] foot_pos (12 = 4腿 × 3坐标)
- [34:37] lin_vel
- [37:40] ang_vel
- [40:52] joint_vel (12 腿部关节)
- [52:55] spine_vel (3 脊柱关节)
- [55:67] foot_vel (12)

AMP 观测空间 (18 dim) - 简化版，仅使用根节点旋转和关节位置:
- Root Rot Obs (3): 根节点旋转（tan_norm后3维），约束身体姿态
- Joint Pos (15): 12 腿部 + 3 脊柱
"""

import os
import glob
import json
import logging

import torch
import numpy as np
from pybullet_utils import transformations

from . import utils
from . import pose3d
from . import motion_util


# =============================================================================
# 辅助函数 (与 MetalHead A1 AMP 对齐)
# =============================================================================

def my_quat_rotate(q, v):
    """使用四元数旋转向量 (q: xyzw 格式)"""
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


def quat_to_tan_norm(q):
    """将四元数转换为切向量和法向量表示 (6维)"""
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = my_quat_rotate(q, ref_tan)

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = my_quat_rotate(q, ref_norm)

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan


def calc_heading(q):
    """计算四元数的朝向角 (yaw)"""
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = my_quat_rotate(q, ref_dir)
    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading


def calc_heading_quat_inv(q):
    """计算消除朝向后的逆四元数"""
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1
    
    # quat_from_angle_axis
    theta = (-heading).unsqueeze(-1)
    xyz = axis * torch.sin(theta / 2)
    w = torch.cos(theta / 2)
    heading_q = torch.cat([xyz, w], dim=-1)
    return heading_q


def quat_mul(q1, q2):
    """四元数乘法 (xyzw 格式)"""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([x, y, z, w], dim=-1)


def quat_rotate_inverse(q, v):
    """使用四元数的逆旋转向量"""
    q_conj = torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)
    return my_quat_rotate(q_conj, v)


def build_amp_observations_himmy(root_states, dof_pos, dof_vel, key_body_pos):
    """
    构建 Himmy AMP 观测 (仅使用根节点旋转和关节位置)
    
    Args:
        root_states: [batch, 13] - pos(3) + rot(4) + lin_vel(3) + ang_vel(3)
        dof_pos: [batch, 15] - 12 leg + 3 spine
        dof_vel: [batch, 15] - 12 leg + 3 spine (未使用)
        key_body_pos: [batch, 4, 3] - 4个脚的世界坐标位置 (未使用)
        
    Returns:
        [batch, 18] - AMP 观测 (仅根节点旋转和关节位置)
    """
    root_rot = root_states[:, 3:7]
    
    # 1. Root Rotation (消除朝向后的旋转)
    heading_rot = calc_heading_quat_inv(root_rot)
    root_rot_obs = quat_mul(heading_rot, root_rot)
    root_rot_obs = quat_to_tan_norm(root_rot_obs)  # [batch, 6]
    # 只取后3维 (法向量，表示pitch/roll)
    root_rot_obs = root_rot_obs[:, -3:]  # [batch, 3]
    
    # 组合观测 (仅根节点旋转和关节位置):
    # root_rot(3) + dof_pos(15) = 18
    obs = torch.cat([
        root_rot_obs,        # 3  
        dof_pos,             # 15
    ], dim=-1)
    
    return obs


class HimmyAMPLoader:
    """Himmy Mark2 专用的 AMP 动作数据加载器"""

    # =========================================================================
    # 数据格式常量 (67 维)
    # =========================================================================
    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 12      # 腿部关节
    SPINE_POS_SIZE = 3       # 脊柱关节 (yaw, pitch, roll)
    TAR_TOE_POS_LOCAL_SIZE = 12
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 12      # 腿部关节速度
    SPINE_VEL_SIZE = 3       # 脊柱关节速度
    TAR_TOE_VEL_LOCAL_SIZE = 12

    # 索引定义
    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE  # 3

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX  # 3
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE  # 7

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX  # 7
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE  # 19

    SPINE_POSE_START_IDX = JOINT_POSE_END_IDX  # 19
    SPINE_POSE_END_IDX = SPINE_POSE_START_IDX + SPINE_POS_SIZE  # 22

    TAR_TOE_POS_LOCAL_START_IDX = SPINE_POSE_END_IDX  # 22
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE  # 34

    LINEAR_VEL_START_IDX = TAR_TOE_POS_LOCAL_END_IDX  # 34
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE  # 37

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX  # 37
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE  # 40

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX  # 40
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE  # 52

    SPINE_VEL_START_IDX = JOINT_VEL_END_IDX  # 52
    SPINE_VEL_END_IDX = SPINE_VEL_START_IDX + SPINE_VEL_SIZE  # 55

    TAR_TOE_VEL_LOCAL_START_IDX = SPINE_VEL_END_IDX  # 55
    TAR_TOE_VEL_LOCAL_END_IDX = TAR_TOE_VEL_LOCAL_START_IDX + TAR_TOE_VEL_LOCAL_SIZE  # 67

    # AMP 观测维度 (仅使用根节点旋转和关节位置)
    # root_rot_obs (3) + joint_pos (15) = 18
    AMP_OBS_DIM = 3 + JOINT_POS_SIZE + SPINE_POS_SIZE  # 18

    def __init__(
            self,
            device,
            time_between_frames,
            data_dir='',
            preload_transitions=False,
            num_preload_transitions=1000000,
            motion_files=None,
            ):
        """
        Himmy Mark2 专用的 AMP 动作数据加载器。

        Args:
            device: torch 设备
            time_between_frames: 帧间时间（秒）
            data_dir: 数据目录（已废弃，使用 motion_files）
            preload_transitions: 是否预加载转换
            num_preload_transitions: 预加载的转换数量
            motion_files: 动作文件列表
        """
        self.device = device
        self.time_between_frames = time_between_frames
        
        if motion_files is None:
            motion_files = []
        
        # 存储轨迹数据
        self.trajectories = []           # 原始轨迹（不含 root pos/rot）
        self.trajectories_full = []      # 完整轨迹
        self.trajectories_amp = []       # AMP 观测轨迹（36 dim）
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []        # 轨迹长度（秒）
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        # 加载动作文件
        for i, motion_file in enumerate(motion_files):
            self._load_motion_file(i, motion_file)

        if len(self.trajectory_weights) > 0:
            # 归一化权重
            self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
            self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
            self.trajectory_lens = np.array(self.trajectory_lens)
            self.trajectory_num_frames = np.array(self.trajectory_num_frames)

            # 预加载转换
            self.preload_transitions = preload_transitions
            if self.preload_transitions:
                print(f'Preloading {num_preload_transitions} transitions for Himmy')
                traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
                times = self.traj_time_sample_batch(traj_idxs)
                self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
                self.preloaded_s_next = self.get_full_frame_at_time_batch(
                    traj_idxs, times + self.time_between_frames)
                print(f'Finished preloading')

            self.all_trajectories_full = torch.vstack(self.trajectories_full)
        else:
            print("Warning: No motion files loaded!")
            self.preload_transitions = False
            self.all_trajectories_full = torch.zeros((1, 67), device=device)

    def _load_motion_file(self, idx, motion_file):
        """加载单个动作文件"""
        self.trajectory_names.append(os.path.basename(motion_file).split('.')[0])
        
        with open(motion_file, "r") as f:
            motion_json = json.load(f)
            motion_data = np.array(motion_json["Frames"])
            
            # Himmy 数据已经是正确顺序，不需要重排
            # 但仍需要标准化四元数
            for f_i in range(motion_data.shape[0]):
                root_rot = self.get_root_rot(motion_data[f_i])
                root_rot = pose3d.QuaternionNormalize(root_rot)
                root_rot = motion_util.standardize_quaternion(root_rot)
                motion_data[f_i, self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX] = root_rot
            
            # 存储不含 root pos/rot 的轨迹
            self.trajectories.append(torch.tensor(
                motion_data[:, self.ROOT_ROT_END_IDX:self.SPINE_VEL_END_IDX],
                dtype=torch.float32, device=self.device))
            
            # 存储完整轨迹（包含速度信息和foot_pos，用于构建AMP观测）
            # 需要包含到 TAR_TOE_VEL_LOCAL_END_IDX (67) 来获取所有信息
            # 格式: root_pos(3) + root_rot(4) + joint_pos(12) + spine_pos(3) + foot_pos(12) + 
            #       lin_vel(3) + ang_vel(3) + joint_vel(12) + spine_vel(3) + foot_vel(12) = 67
            self.trajectories_full.append(torch.tensor(
                motion_data,  # 完整67维
                dtype=torch.float32, device=self.device))
            
            self.trajectory_idxs.append(idx)
            self.trajectory_weights.append(float(motion_json.get("MotionWeight", 1.0)))
            frame_duration = float(motion_json["FrameDuration"])
            self.trajectory_frame_durations.append(frame_duration)
            traj_len = (motion_data.shape[0] - 1) * frame_duration
            self.trajectory_lens.append(traj_len)
            self.trajectory_num_frames.append(float(motion_data.shape[0]))

        print(f"Loaded {traj_len:.2f}s motion from {motion_file} (67-dim Himmy format)")

    # =========================================================================
    # 采样方法
    # =========================================================================
    
    def weighted_traj_idx_sample(self):
        """加权采样轨迹索引"""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """批量加权采样轨迹索引"""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """采样轨迹的随机时间点"""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """批量采样轨迹的随机时间点"""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        """线性插值"""
        return (1.0 - blend) * val0 + blend * val1

    # =========================================================================
    # 帧获取方法
    # =========================================================================

    def get_frame_at_time(self, traj_idx, time):
        """
        获取指定时间的 AMP 观测帧 (37 dim)
        
        返回: Root Height (1) + Joint Pos (15) + Lin Vel (3) + Ang Vel (3) + Joint Vel (15)
        """
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_amp[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        idx_high = min(idx_high, n - 1)  # 防止越界
        frame_start = self.trajectories_amp[traj_idx][idx_low]
        frame_end = self.trajectories_amp[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """获取指定时间的完整帧"""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        idx_high = min(idx_high, n - 1)
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        """批量获取指定时间的完整帧 (67维)"""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low = np.floor(p * n).astype(np.int64)
        idx_high = np.minimum(np.ceil(p * n).astype(np.int64), (n - 1).astype(np.int64))
        
        # 预分配张量 (完整67维)
        full_dim = self.TAR_TOE_VEL_LOCAL_END_IDX  # 67
        all_frame_starts = torch.zeros(len(traj_idxs), full_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), full_dim, device=self.device)
        
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        
        # 分别插值位置、四元数和其他数据
        pos_blend = self.slerp(
            all_frame_starts[:, :self.POS_SIZE], 
            all_frame_ends[:, :self.POS_SIZE], blend)
        rot_blend = utils.quaternion_slerp(
            all_frame_starts[:, self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX],
            all_frame_ends[:, self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX], blend)
        rest_blend = self.slerp(
            all_frame_starts[:, self.ROOT_ROT_END_IDX:],
            all_frame_ends[:, self.ROOT_ROT_END_IDX:], blend)
        
        return torch.cat([pos_blend, rot_blend, rest_blend], dim=-1)

    def blend_frame_pose(self, frame0, frame1, blend):
        """混合两帧姿态"""
        root_pos0 = frame0[:self.POS_SIZE]
        root_pos1 = frame1[:self.POS_SIZE]
        root_rot0 = frame0[self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX]
        root_rot1 = frame1[self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX]
        rest0 = frame0[self.ROOT_ROT_END_IDX:]
        rest1 = frame1[self.ROOT_ROT_END_IDX:]

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = transformations.quaternion_slerp(
            root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
        blend_root_rot = torch.tensor(
            motion_util.standardize_quaternion(blend_root_rot),
            dtype=torch.float32, device=self.device)
        blend_rest = self.slerp(rest0, rest1, blend)

        return torch.cat([blend_root_pos, blend_root_rot, blend_rest])

    # =========================================================================
    # 生成器方法
    # =========================================================================

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """
        生成 AMP 训练数据批次
        
        AMP 观测格式 (49 dim) - 移除线速度以实现速度不变性:
        - Root Height (1): 根节点高度
        - Root Rot Obs (3): 根节点旋转（tan_norm后3维）
        - Local Root Ang Vel (3): 本地坐标系角速度
        - Joint Pos (15): 12 腿部 + 3 脊柱
        - Joint Vel (15): 12 腿部 + 3 脊柱
        - Local Key Body Pos (12): 4个脚的局部位置
        """
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                # 从预加载数据中提取 AMP 观测
                s = self._extract_amp_obs(self.preloaded_s[idxs])
                s_next = self._extract_amp_obs(self.preloaded_s_next[idxs])
            else:
                # 非预加载模式：获取完整帧然后提取AMP观测
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                full_s = self.get_full_frame_at_time_batch(traj_idxs, times)
                full_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
                s = self._extract_amp_obs(full_s)
                s_next = self._extract_amp_obs(full_s_next)
            yield s, s_next

    def _extract_amp_obs(self, full_frames):
        """
        从完整帧中提取 AMP 观测 (49 dim) - 无线速度
        
        使用 build_amp_observations_himmy 函数构建观测
        """
        # 提取各部分数据
        # full_frames: [batch, 34] - root_pos(3) + root_rot(4) + joint_pos(12) + spine_pos(3) + foot_pos(12)
        # 但我们需要完整的67维数据来获取速度，所以需要扩展
        
        # 如果full_frames只有34维（到foot_pos），需要从预加载的完整数据中获取速度
        # 这里假设full_frames包含足够的信息
        
        batch_size = full_frames.shape[0]
        
        # 提取 root state (pos + rot + lin_vel + ang_vel)
        root_pos = full_frames[:, self.ROOT_POS_START_IDX:self.ROOT_POS_END_IDX]  # [batch, 3]
        root_rot = full_frames[:, self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX]  # [batch, 4]
        
        # 检查是否有速度信息
        if full_frames.shape[1] > self.TAR_TOE_POS_LOCAL_END_IDX:
            # 有完整的67维数据
            lin_vel = full_frames[:, self.LINEAR_VEL_START_IDX:self.LINEAR_VEL_END_IDX]  # [batch, 3]
            ang_vel = full_frames[:, self.ANGULAR_VEL_START_IDX:self.ANGULAR_VEL_END_IDX]  # [batch, 3]
            joint_vel = full_frames[:, self.JOINT_VEL_START_IDX:self.JOINT_VEL_END_IDX]  # [batch, 12]
            spine_vel = full_frames[:, self.SPINE_VEL_START_IDX:self.SPINE_VEL_END_IDX]  # [batch, 3]
        else:
            # 只有34维数据，速度设为0（这种情况不应该发生）
            lin_vel = torch.zeros(batch_size, 3, device=full_frames.device)
            ang_vel = torch.zeros(batch_size, 3, device=full_frames.device)
            joint_vel = torch.zeros(batch_size, 12, device=full_frames.device)
            spine_vel = torch.zeros(batch_size, 3, device=full_frames.device)
        
        # 构建 root_states [batch, 13]
        root_states = torch.cat([root_pos, root_rot, lin_vel, ang_vel], dim=-1)
        
        # 提取 dof_pos [batch, 15]
        joint_pos = full_frames[:, self.JOINT_POSE_START_IDX:self.JOINT_POSE_END_IDX]  # [batch, 12]
        spine_pos = full_frames[:, self.SPINE_POSE_START_IDX:self.SPINE_POSE_END_IDX]  # [batch, 3]
        dof_pos = torch.cat([joint_pos, spine_pos], dim=-1)
        
        # 提取 dof_vel [batch, 15]
        dof_vel = torch.cat([joint_vel, spine_vel], dim=-1)
        
        # 提取 key_body_pos [batch, 4, 3] - 4个脚的世界坐标位置
        foot_pos_flat = full_frames[:, self.TAR_TOE_POS_LOCAL_START_IDX:self.TAR_TOE_POS_LOCAL_END_IDX]  # [batch, 12]
        key_body_pos = foot_pos_flat.view(batch_size, 4, 3)
        
        # 使用 build_amp_observations_himmy 构建AMP观测
        return build_amp_observations_himmy(root_states, dof_pos, dof_vel, key_body_pos)

    def get_full_frame_batch(self, num_frames):
        """获取随机完整帧批次"""
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    # =========================================================================
    # 属性
    # =========================================================================

    @property
    def observation_dim(self):
        """AMP 观测维度 (36 dim)"""
        return self.AMP_OBS_DIM

    @property
    def num_motions(self):
        """动作文件数量"""
        return len(self.trajectory_names)

    # =========================================================================
    # 静态获取方法
    # =========================================================================

    @staticmethod
    def get_root_pos(pose):
        return pose[HimmyAMPLoader.ROOT_POS_START_IDX:HimmyAMPLoader.ROOT_POS_END_IDX]

    @staticmethod
    def get_root_rot(pose):
        return pose[HimmyAMPLoader.ROOT_ROT_START_IDX:HimmyAMPLoader.ROOT_ROT_END_IDX]

    @staticmethod
    def get_joint_pose(pose):
        return pose[HimmyAMPLoader.JOINT_POSE_START_IDX:HimmyAMPLoader.JOINT_POSE_END_IDX]

    @staticmethod
    def get_spine_pose(pose):
        return pose[HimmyAMPLoader.SPINE_POSE_START_IDX:HimmyAMPLoader.SPINE_POSE_END_IDX]

    @staticmethod
    def get_linear_vel(pose):
        return pose[HimmyAMPLoader.LINEAR_VEL_START_IDX:HimmyAMPLoader.LINEAR_VEL_END_IDX]

    @staticmethod
    def get_angular_vel(pose):
        return pose[HimmyAMPLoader.ANGULAR_VEL_START_IDX:HimmyAMPLoader.ANGULAR_VEL_END_IDX]

    @staticmethod
    def get_joint_vel(pose):
        return pose[HimmyAMPLoader.JOINT_VEL_START_IDX:HimmyAMPLoader.JOINT_VEL_END_IDX]

    @staticmethod
    def get_spine_vel(pose):
        return pose[HimmyAMPLoader.SPINE_VEL_START_IDX:HimmyAMPLoader.SPINE_VEL_END_IDX]
