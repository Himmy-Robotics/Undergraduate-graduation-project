#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: himmy_kinematics.py
@Auth: Based on A1 kinematics by Huiqiao, modified for Himmy Mark2
@Date: 2024/12

Himmy Mark2 运动学模块
===================

该模块为Himmy Mark2四足机器人提供正运动学和逆运动学计算。

Himmy Mark2机器人特点：
- 15个DOF（自由度）：
  - 前腿：FL(3DOF) + FR(3DOF) = 6DOF
  - 脊柱：yaw + pitch + roll = 3DOF（模拟动物脊柱）
  - 后腿：RL(3DOF) + RR(3DOF) = 6DOF

机器人结构示意图（俯视图，X轴正方向为前进方向）:

    FL_foot                              FR_foot
       │                                    │
       └── FL_leg ────────────────────── FR_leg ──┘
                    │ 前躯体(trunk/base_link) │
                    │         │              │
                    │    [SPINE JOINTS]      │
                    │    yaw -> pitch -> roll│
                    │         │              │
                    │ 后躯体(R_body_Link)    │
       ┌── RL_leg ────────────────────── RR_leg ──┐
       │                                    │
    RL_foot                              RR_foot

关键尺寸参数（从URDF提取）：
- 前后躯体总长度（含脊柱）: ~0.5392m
- 体宽: ~0.3551m  
- 躯干高度: ~0.51m
- 大腿长度: 0.23m
- 小腿长度: 0.28m
- hip到thigh的偏移: ~0.0833m (X方向) + ~0.0795m (Y方向)

注意：与A1不同的是，Himmy的后腿通过脊柱关节连接，
因此在计算逆运动学时需要考虑脊柱的转动。
"""

import math
import numpy as np
import torch


class HimmyKinematics:
    """
    Himmy Mark2机器人运动学类
    
    提供正向运动学和逆向运动学的计算方法。
    
    由于Himmy有3个脊柱自由度，逆运动学分为两部分：
    1. 前腿：直接从trunk计算（与A1类似）
    2. 后腿：需要先考虑脊柱的转动，再从roll_spine_Link计算
    
    Attributes:
        front_body_length: 前躯体长度（trunk到脊柱起始）
        rear_body_length: 后躯体长度（脊柱末端到后腿hip）
        spine_length: 脊柱总长度
        body_wide: 机身宽度
        hip_offset_x: hip相对于躯体的X偏移
        hip_offset_y: hip相对于躯体的Y偏移
        thigh_length: 大腿长度
        calf_length: 小腿长度
    """
    
    def __init__(self, 
                 front_body_length=0.0432,      # trunk到FL/FR_hip的X距离
                 rear_body_length=0.0655,       # roll_spine到RL/RR_hip的X距离
                 spine_length_yaw=0.1200,       # yaw_spine长度
                 spine_length_pitch=0.1053,     # pitch_spine长度
                 spine_offset_x=-0.0386,        # trunk到yaw_spine的X偏移
                 body_wide=0.2,                 # 机身宽度（左右hip之间）
                 hip_offset_x=0.0833,           # hip到thigh的X偏移
                 hip_offset_y=0.0795,           # hip到thigh的Y偏移
                 thigh_length=0.23,             # 大腿长度
                 calf_length=0.28):             # 小腿长度
        """
        初始化Himmy运动学参数
        
        Args:
            front_body_length: trunk到前腿hip的X距离
            rear_body_length: roll_spine到后腿hip的X距离
            spine_length_yaw: yaw脊柱关节长度
            spine_length_pitch: pitch脊柱关节长度
            spine_offset_x: trunk到脊柱起点的X偏移
            body_wide: 机身宽度
            hip_offset_x: hip到thigh的X方向偏移
            hip_offset_y: hip到thigh的Y方向偏移
            thigh_length: 大腿长度
            calf_length: 小腿长度
        """
        # 躯体尺寸
        self.front_body_length = front_body_length
        self.rear_body_length = rear_body_length
        self.spine_length_yaw = spine_length_yaw
        self.spine_length_pitch = spine_length_pitch
        self.spine_offset_x = spine_offset_x
        self.body_wide = body_wide
        
        # 腿部尺寸
        self.hip_offset_x = hip_offset_x
        self.hip_offset_y = hip_offset_y
        self.thigh_length = thigh_length
        self.calf_length = calf_length
        
        # 计算总体长度
        self.total_body_length = abs(spine_offset_x) + spine_length_yaw + spine_length_pitch + rear_body_length + front_body_length
        
        # 腿的最大伸展长度（用于逆运动学检查）
        self.max_leg_reach = thigh_length + calf_length
        
    def get_front_leg_positions(self):
        """
        获取前腿hip关节相对于trunk的位置
        
        Returns:
            dict: 包含FL和FR hip位置的字典
        """
        # 从trunk坐标系看，前腿hip的位置
        fl_hip = np.array([self.front_body_length, self.body_wide / 2, 0])
        fr_hip = np.array([self.front_body_length, -self.body_wide / 2, 0])
        return {'FL': fl_hip, 'FR': fr_hip}
    
    def get_rear_leg_positions_from_roll_spine(self):
        """
        获取后腿hip关节相对于roll_spine_Link的位置
        
        Returns:
            dict: 包含RL和RR hip位置的字典
        """
        # 从roll_spine_Link坐标系看，后腿hip的位置
        rl_hip = np.array([-self.rear_body_length, self.body_wide / 2, 0])
        rr_hip = np.array([-self.rear_body_length, -self.body_wide / 2, 0])
        return {'RL': rl_hip, 'RR': rr_hip}
    
    def inverse_kinematics_leg(self, foot_pos_hip, is_left=True):
        """
        计算单条腿的逆运动学（从hip坐标系）
        
        这是基本的3DOF腿部逆运动学求解，适用于所有四条腿。
        
        数学推导：
        假设足端在hip坐标系下的位置为 p = [px, py, pz]
        腿部连杆长度为 l0(hip), l1(thigh), l2(calf)
        
        θ1 (hip角度): 由几何关系求解
        θ2 (thigh角度): 由余弦定理求解
        θ3 (calf角度): 由余弦定理求解
        
        Args:
            foot_pos_hip: 足端在hip坐标系下的位置 [x, y, z]
            is_left: 是否为左腿（左右腿hip长度方向相反）
        
        Returns:
            np.array: [θ_hip, θ_thigh, θ_calf] 三个关节角度（弧度）
        """
        px, py, pz = foot_pos_hip[0], foot_pos_hip[1], foot_pos_hip[2]
        
        # hip长度，左腿为负，右腿为正
        l0 = -self.hip_offset_y if is_left else self.hip_offset_y
        l1 = self.thigh_length
        l2 = self.calf_length
        
        # 计算hip关节角度 θ1
        # 使用几何关系：考虑hip的侧向偏移
        d_xy = np.sqrt(px**2 + py**2)
        if d_xy < 1e-6:
            d_xy = 1e-6
        
        x1 = l0 / d_xy
        x1 = np.clip(x1, -1.0, 1.0)  # 防止arcsin越界
        
        theta1 = np.arctan2(x1, np.sqrt(max(1 - x1**2, 0))) + np.arctan2(py, px)
        
        # 计算从hip到足端的有效距离（去除hip偏移后）
        r = np.sqrt(max(px**2 + py**2 - l0**2, 0))
        
        # 计算thigh关节角度 θ2
        # 使用余弦定理
        leg_length_sq = pz**2 + r**2
        n = (leg_length_sq + l1**2 - l2**2) / (2 * l1)
        
        d_rz = np.sqrt(max(pz**2 + r**2, 1e-6))
        x2 = n / d_rz
        x2 = np.clip(x2, -1.0, 1.0)
        
        theta2 = np.arctan2(x2, np.sqrt(max(1 - x2**2, 0))) - np.arctan2(r, pz)
        
        # 计算calf关节角度 θ3
        # 使用余弦定理
        k = (leg_length_sq - l1**2 - l2**2) / (2 * l1 * l2)
        k = np.clip(k, -1.0, 1.0)
        
        theta3 = np.arctan2(np.sqrt(max(1 - k**2, 0)), k)
        
        return np.array([theta1, theta2, theta3])
    
    def inverse_kinematics(self, base_pose_w, base_rot_w, foot_pos_w, spine_angles=None):
        """
        计算完整的逆运动学（包含脊柱）
        
        这是Himmy的核心逆运动学函数，与A1的主要区别在于：
        1. 前腿：从trunk直接计算（与A1类似）
        2. 后腿：需要通过脊柱变换，从roll_spine_Link计算
        
        坐标系定义：
        - W: 世界坐标系
        - A: trunk（base）坐标系
        - S: roll_spine_Link坐标系（后腿的参考坐标系）
        - B: 各个hip坐标系
        - C: 足端坐标系
        
        Args:
            base_pose_w: trunk在世界坐标系下的位置 [x, y, z]
            base_rot_w: trunk在世界坐标系下的欧拉角 [roll, pitch, yaw]
            foot_pos_w: 四个足端在世界坐标系下的位置，顺序为 [FR, FL, RR, RL]，共12个值
            spine_angles: 脊柱关节角度 [yaw, pitch, roll]，如果为None则设为0
        
        Returns:
            np.array: 12个腿部关节角度 [FR(3), FL(3), RR(3), RL(3)]
            np.array: 3个脊柱关节角度 [yaw, pitch, roll]（如果需要求解的话）
        """
        # 解析足端位置
        foot_fr_w = foot_pos_w[:3]
        foot_fl_w = foot_pos_w[3:6]
        foot_rr_w = foot_pos_w[6:9]
        foot_rl_w = foot_pos_w[9:]
        
        # 脊柱角度，默认为0
        if spine_angles is None:
            spine_angles = np.array([0.0, 0.0, 0.0])
        
        # 计算从世界坐标系到trunk坐标系的变换矩阵
        T_wa = self.trans_matrix_ab(base_pose_w, base_rot_w)
        
        # ========== 前腿逆运动学 ==========
        # 前腿hip相对于trunk的位置
        fl_hip_pos = np.array([self.front_body_length, self.body_wide / 2, 0])
        fr_hip_pos = np.array([self.front_body_length, -self.body_wide / 2, 0])
        
        # hip坐标系旋转（绕Y轴旋转90度）
        r_hip = [0, np.pi / 2, 0]
        
        # 计算从trunk到各hip的变换矩阵
        T_ab_fr = self.trans_matrix_ab(fr_hip_pos, r_hip)
        T_ab_fl = self.trans_matrix_ab(fl_hip_pos, r_hip)
        
        # 将足端位置转换到hip坐标系
        fr_foot_hip = T_ab_fr @ T_wa @ np.append(foot_fr_w, 1)
        fr_foot_hip = fr_foot_hip[:-1]
        fl_foot_hip = T_ab_fl @ T_wa @ np.append(foot_fl_w, 1)
        fl_foot_hip = fl_foot_hip[:-1]
        
        # 计算前腿逆运动学
        fr_ang = self.inverse_kinematics_leg(fr_foot_hip, is_left=False)
        fl_ang = self.inverse_kinematics_leg(fl_foot_hip, is_left=True)
        
        # 调整角度符号（根据关节旋转轴方向）
        fr_ang = np.array([fr_ang[0], -fr_ang[1], -fr_ang[2]])
        fl_ang = np.array([fl_ang[0], -fl_ang[1], -fl_ang[2]])
        
        # ========== 后腿逆运动学（考虑脊柱） ==========
        # 计算从trunk到roll_spine_Link的变换
        # 脊柱链: trunk -> yaw_spine -> pitch_spine -> roll_spine
        
        # yaw_spine相对于trunk的位置和旋转
        yaw_spine_pos = np.array([self.spine_offset_x, 0, 0])
        yaw_spine_rot = [0, 0, spine_angles[0]]  # yaw旋转
        T_trunk_yaw = self.trans_matrix_ba(yaw_spine_pos, yaw_spine_rot)
        
        # pitch_spine相对于yaw_spine的位置和旋转
        pitch_spine_pos = np.array([-self.spine_length_yaw, 0, 0])
        pitch_spine_rot = [0, spine_angles[1], 0]  # pitch旋转
        T_yaw_pitch = self.trans_matrix_ba(pitch_spine_pos, pitch_spine_rot)
        
        # roll_spine相对于pitch_spine的位置和旋转
        roll_spine_pos = np.array([-self.spine_length_pitch, 0, 0])
        roll_spine_rot = [spine_angles[2], 0, 0]  # roll旋转
        T_pitch_roll = self.trans_matrix_ba(roll_spine_pos, roll_spine_rot)
        
        # 从世界坐标系到roll_spine的完整变换
        T_trunk_roll = T_trunk_yaw @ T_yaw_pitch @ T_pitch_roll
        T_w_roll = np.linalg.inv(T_trunk_roll) @ T_wa
        
        # 后腿hip相对于roll_spine_Link的位置
        rl_hip_pos = np.array([-self.rear_body_length, self.body_wide / 2, 0])
        rr_hip_pos = np.array([-self.rear_body_length, -self.body_wide / 2, 0])
        
        # 计算从roll_spine到各hip的变换矩阵
        T_roll_rr = self.trans_matrix_ab(rr_hip_pos, r_hip)
        T_roll_rl = self.trans_matrix_ab(rl_hip_pos, r_hip)
        
        # 将足端位置转换到后腿hip坐标系
        rr_foot_hip = T_roll_rr @ T_w_roll @ np.append(foot_rr_w, 1)
        rr_foot_hip = rr_foot_hip[:-1]
        rl_foot_hip = T_roll_rl @ T_w_roll @ np.append(foot_rl_w, 1)
        rl_foot_hip = rl_foot_hip[:-1]
        
        # 计算后腿逆运动学
        rr_ang = self.inverse_kinematics_leg(rr_foot_hip, is_left=False)
        rl_ang = self.inverse_kinematics_leg(rl_foot_hip, is_left=True)
        
        # 调整角度符号
        rr_ang = np.array([rr_ang[0], -rr_ang[1], -rr_ang[2]])
        rl_ang = np.array([rl_ang[0], -rl_ang[1], -rl_ang[2]])
        
        return np.hstack([fr_ang, fl_ang, rr_ang, rl_ang])
    
    # ==================== 坐标变换辅助函数 ====================
    
    @staticmethod
    def rot_matrix_ba(t):
        """
        计算从A坐标系到B坐标系的旋转矩阵
        
        使用ZYX欧拉角顺序（先绕Z轴旋转yaw，再绕Y轴旋转pitch，最后绕X轴旋转roll）
        
        Args:
            t: 欧拉角 [roll, pitch, yaw]
        
        Returns:
            np.array: 3x3旋转矩阵 R_ba
        """
        roll, pitch, yaw = t[0], t[1], t[2]
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        r = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])
        return r
    
    @staticmethod
    def rot_matrix_ab(t):
        """
        计算从B坐标系到A坐标系的旋转矩阵（即rot_matrix_ba的转置）
        
        Args:
            t: 欧拉角 [roll, pitch, yaw]
        
        Returns:
            np.array: 3x3旋转矩阵 R_ab = R_ba^T
        """
        return HimmyKinematics.rot_matrix_ba(t).T
    
    @staticmethod
    def trans_matrix_ba(m, t):
        """
        计算从A坐标系到B坐标系的齐次变换矩阵
        
        变换矩阵形式：
        T = | R   p |
            | 0   1 |
        
        其中R是旋转矩阵，p是平移向量
        
        Args:
            m: 平移向量 [x, y, z]
            t: 欧拉角 [roll, pitch, yaw]
        
        Returns:
            np.array: 4x4齐次变换矩阵 T_ba
        """
        r = HimmyKinematics.rot_matrix_ba(t)
        trans = np.hstack([r, np.array(m)[:, np.newaxis]])
        trans = np.vstack([trans, np.array([[0, 0, 0, 1]])])
        return trans
    
    @staticmethod
    def trans_matrix_ab(m, t):
        """
        计算从B坐标系到A坐标系的齐次变换矩阵
        
        这是trans_matrix_ba的逆变换：T_ab = T_ba^(-1)
        
        Args:
            m: 平移向量 [x, y, z]
            t: 欧拉角 [roll, pitch, yaw]
        
        Returns:
            np.array: 4x4齐次变换矩阵 T_ab
        """
        r = HimmyKinematics.rot_matrix_ba(t)
        r_t = r.T
        trans = np.hstack([r_t, -np.dot(r_t, np.array(m))[:, np.newaxis]])
        trans = np.vstack([trans, np.array([[0, 0, 0, 1]])])
        return trans
    
    # ==================== 四元数和欧拉角转换函数 ====================
    
    @staticmethod
    def quaternion2rpy(q):
        """
        四元数转欧拉角（roll-pitch-yaw）
        
        四元数格式: [x, y, z, w] (scalar-last)
        
        Args:
            q: 四元数 [x, y, z, w]
        
        Returns:
            list: 欧拉角 [roll, pitch, yaw]
        """
        x, y, z, w = q[0], q[1], q[2], q[3]
        
        # roll (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        
        # pitch (y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = math.asin(t2)
        
        # yaw (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        
        return [roll, pitch, yaw]
    
    @staticmethod
    def rpy2quaternion(r):
        """
        欧拉角转四元数
        
        Args:
            r: 欧拉角 [roll, pitch, yaw]
        
        Returns:
            list: 四元数 [x, y, z, w]
        """
        roll, pitch, yaw = r[0], r[1], r[2]
        
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        w = cr * cp * cy + sr * sp * sy
        
        return [x, y, z, w]


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("Himmy Mark2 运动学测试")
    print("=" * 60)
    
    kinematics = HimmyKinematics()
    
    print("\n机器人参数:")
    print(f"  大腿长度: {kinematics.thigh_length:.4f}m")
    print(f"  小腿长度: {kinematics.calf_length:.4f}m")
    print(f"  腿最大伸展: {kinematics.max_leg_reach:.4f}m")
    
    # 测试四元数转换
    print("\n测试四元数转换:")
    test_rpy = [0.1, 0.2, 0.3]
    q = kinematics.rpy2quaternion(test_rpy)
    rpy_back = kinematics.quaternion2rpy(q)
    print(f"  原始欧拉角: {test_rpy}")
    print(f"  四元数: {q}")
    print(f"  还原欧拉角: {rpy_back}")
    
    print("\n测试完成")
