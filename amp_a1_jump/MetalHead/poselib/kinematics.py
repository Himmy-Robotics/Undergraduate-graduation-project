#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: kinematics.py
@Auth: Huiqiao
@Date: 2022/7/6

A1机器人运动学类
================

该模块实现了A1四足机器人的运动学计算，包括：
1. 逆运动学：从足端位置计算关节角度
2. 坐标变换：world frame <-> body frame <-> hip frame
3. 旋转表示转换：旋转矩阵、欧拉角、四元数、轴角
"""
import math
import numpy as np
from isaacgym.torch_utils import *
import torch


class Kinematics:
    """
    A1机器人运动学求解器
    
    坐标系定义:
    - World frame: 全局坐标系
    - Body frame: 机体坐标系，原点在机体中心，X轴向前，Y轴向左，Z轴向上
    - Hip frame: 髋关节坐标系，原点在髋关节中心
    
    关节定义:
    - Hip joint (髋关节): 绕X轴旋转，控制腿的外展/内收
    - Thigh joint (大腿关节): 绕Y轴旋转，控制大腿前后摆动
    - Calf joint (小腿关节): 绕Y轴旋转，控制小腿前后摆动
    """
    
    def __init__(self, body_length=0.85, body_wide=0.094, hip_length=0.08505, thigh_length=0.2, calf_length=0.2):
        """
        初始化A1机器人几何参数
        
        Args:
            body_length: 机体长度（前后腿髋关节之间的距离），单位：米
            body_wide: 机体宽度（左右腿髋关节之间的距离），单位：米
            hip_length: 髋关节长度（髋关节到大腿关节的距离），单位：米
            thigh_length: 大腿长度（大腿关节到小腿关节的距离），单位：米
            calf_length: 小腿长度（小腿关节到足端的距离），单位：米
        """
        self.body_length = body_length
        self.body_wide = body_wide
        self.hip_length = hip_length
        self.thigh_length = thigh_length
        self.calf_length = calf_length

    def inverse_kinematics(self, base_pose_w, base_rot_w, foot_pos_w):
        """
        A1机器人逆运动学求解
        
        从足端的世界坐标位置计算各腿的关节角度。
        
        求解步骤:
        1. 将足端位置从world frame转换到body frame
        2. 将足端位置从body frame转换到各腿的hip frame
        3. 在hip frame中求解三关节逆运动学
        4. 调整角度符号以匹配URDF定义
        
        Args:
            base_pose_w: 机体中心在世界坐标系中的位置 [x, y, z]
            base_rot_w: 机体在世界坐标系中的姿态（欧拉角RPY） [roll, pitch, yaw]
            foot_pos_w: 四个足端在世界坐标系中的位置 [fr(3), fl(3), rr(3), rl(3)] 共12维
                       顺序: FR(前右), FL(前左), RR(后右), RL(后左)
        
        Returns:
            np.ndarray: 12个关节角度 [fr_hip, fr_thigh, fr_calf, fl_hip, fl_thigh, fl_calf,
                                      rr_hip, rr_thigh, rr_calf, rl_hip, rl_thigh, rl_calf]
        """
        # 提取各腿足端位置
        foot_fr_w = foot_pos_w[:3]
        foot_fl_w = foot_pos_w[3:6]
        foot_rr_w = foot_pos_w[6:9]
        foot_rl_w = foot_pos_w[9:]

        # 计算从world frame到body frame的变换矩阵（齐次变换矩阵）
        t_wa = self.trans_matrix_ab(base_pose_w, base_rot_w)

        # 各腿髋关节在body frame中的位置
        # body_length: 前后髋关节距离, body_wide: 左右髋关节距离
        fr_link_pos = np.array([self.body_length, -self.body_wide / 2, 0])  # 前右
        fl_link_pos = np.array([self.body_length, self.body_wide / 2, 0])   # 前左
        rr_link_pos = np.array([0, -self.body_wide / 2, 0])                 # 后右
        rl_link_pos = np.array([0, self.body_wide / 2, 0])                  # 后左

        # hip frame的姿态：绕Y轴旋转90度，使Z轴指向足端方向
        r_hip = [0, np.pi / 2, 0]
        
        # 计算从body frame到各腿hip frame的变换矩阵
        t_ab_fr = self.trans_matrix_ab(fr_link_pos, r_hip)
        t_ab_fl = self.trans_matrix_ab(fl_link_pos, r_hip)
        t_ab_rr = self.trans_matrix_ab(rr_link_pos, r_hip)
        t_ab_rl = self.trans_matrix_ab(rl_link_pos, r_hip)

        # 将足端位置从world frame转换到各腿的hip frame
        # 步骤: world -> body -> hip
        fr_foot_hip = t_wa @ np.append(foot_fr_w, 1)
        fr_foot_hip = (t_ab_fr @ fr_foot_hip)[:-1]  # 去掉齐次坐标的最后一维
        fl_foot_hip = t_wa @ np.append(foot_fl_w, 1)
        fl_foot_hip = (t_ab_fl @ fl_foot_hip)[:-1]
        rr_foot_hip = t_wa @ np.append(foot_rr_w, 1)
        rr_foot_hip = (t_ab_rr @ rr_foot_hip)[:-1]
        rl_foot_hip = t_wa @ np.append(foot_rl_w, 1)
        rl_foot_hip = (t_ab_rl @ rl_foot_hip)[:-1]

        # 定义连杆长度
        # 右侧腿: hip_length为正（向右）
        # 左侧腿: hip_length为负（向左）
        l_org_r = [self.hip_length, self.thigh_length, self.calf_length]
        l_org_l = [-self.hip_length, self.thigh_length, self.calf_length]

        # 求解各腿的关节角度
        fr_ang = np.array(self.inverse(fr_foot_hip, l_org_r))
        fl_ang = np.array(self.inverse(fl_foot_hip, l_org_l))
        rr_ang = np.array(self.inverse(rr_foot_hip, l_org_r))
        rl_ang = np.array(self.inverse(rl_foot_hip, l_org_l))

        # 调整角度符号以匹配URDF中的关节方向定义
        # thigh和calf关节角度取负值
        fr_ang = np.array([fr_ang[0], -fr_ang[1], -fr_ang[2]])
        fl_ang = np.array([fl_ang[0], -fl_ang[1], -fl_ang[2]])
        rr_ang = np.array([rr_ang[0], -rr_ang[1], -rr_ang[2]])
        rl_ang = np.array([rl_ang[0], -rl_ang[1], -rl_ang[2]])

        return np.hstack([fr_ang, fl_ang, rr_ang, rl_ang])

    def inverse(self, p, l):
        """
        三关节串联机械臂逆运动学求解（在hip frame中）
        
        求解三关节串联结构到达目标位置p所需的关节角度。
        使用几何法求解，基于余弦定理和三角函数。
        
        机械臂结构:
        - 关节1(hip): 起点在原点，绕X轴旋转，连杆长度为l[0]
        - 关节2(thigh): 绕Y轴旋转，连杆长度为l[1]
        - 关节3(calf): 绕Y轴旋转，连杆长度为l[2]
        - 末端(foot): 目标位置为p
        
        Args:
            p: 足端在hip frame中的位置 [x, y, z]
            l: 三段连杆的长度 [hip_length, thigh_length, calf_length]
        
        Returns:
            [t1, t2, t3]: 三个关节角度（弧度）
                t1: hip关节角度（绕X轴，控制腿的外展/内收）
                t2: thigh关节角度（绕Y轴，控制大腿前后摆动）
                t3: calf关节角度（绕Y轴，控制小腿前后摆动）
        """
        # 求解hip关节角度t1
        # 在XY平面投影，利用hip_length和目标点的XY距离
        x1 = l[0] / (p[0] ** 2 + p[1] ** 2) ** 0.5
        t1 = np.arctan2(x1, (max(1 - x1 ** 2, 0.)) ** 0.5) + np.arctan2(p[1], p[0])

        # 计算大腿和小腿在YZ平面的投影长度
        # r是去除hip_length影响后，在YZ平面的投影距离
        r = (max(p[0] ** 2 + p[1] ** 2 - l[0] ** 2, 0.)) ** 0.5
        
        # 求解thigh关节角度t2
        # 利用余弦定理计算大腿和目标点的夹角
        n = (p[2] ** 2 + r ** 2 + l[1] ** 2 - l[2] ** 2) / (2 * l[1])
        x2 = n / (max(p[2] ** 2 + r ** 2, 0.)) ** 0.5
        t2 = np.arctan2(x2, (max(1 - x2 ** 2, 0.)) ** 0.5) - np.arctan2(r, p[2])

        # 求解calf关节角度t3
        # 利用余弦定理计算小腿相对于大腿的夹角
        k = (p[2] ** 2 + r ** 2 - l[1] ** 2 - l[2] ** 2) / (2 * l[1] * l[2])
        t3 = np.arctan2((max(1 - k ** 2, 0.)) ** 0.5, k)
        
        return [t1, t2, t3]

    @staticmethod
    def rot_matrix_ba(t):
        """
        从frame B到frame A的旋转矩阵（RPY欧拉角，固定轴）
        
        使用固定轴（外旋）ZYX顺序的欧拉角构造旋转矩阵。
        旋转顺序: 先绕Z轴旋转yaw, 再绕Y轴旋转pitch, 最后绕X轴旋转roll
        
        Args:
            t: 欧拉角 [roll, pitch, yaw] (弧度)
        
        Returns:
            np.ndarray: 3x3旋转矩阵，将frame B中的向量转换到frame A
        """
        r = np.array([[np.cos(t[2]) * np.cos(t[1]),
                       np.cos(t[2]) * np.sin(t[1]) * np.sin(t[0]) - np.sin(t[2]) * np.cos(t[0]),
                       np.cos(t[2]) * np.sin(t[1]) * np.cos(t[0]) + np.sin(t[2]) * np.sin(t[0])],
                      [np.sin(t[2]) * np.cos(t[1]),
                       np.sin(t[2]) * np.sin(t[1]) * np.sin(t[0]) + np.cos(t[2]) * np.cos(t[0]),
                       np.sin(t[2]) * np.sin(t[1]) * np.cos(t[0]) - np.cos(t[2]) * np.sin(t[0])],
                      [-np.sin(t[1]), np.cos(t[1]) * np.sin(t[0]), np.cos(t[1]) * np.cos(t[0])]])
        return r

    @staticmethod
    def rot_matrix_ab(t):
        """
        从frame A到frame B的旋转矩阵（RPY欧拉角，固定轴）
        
        这是rot_matrix_ba的逆矩阵（转置）。
        
        Args:
            t: 欧拉角 [roll, pitch, yaw] (弧度)
        
        Returns:
            np.ndarray: 3x3旋转矩阵，将frame A中的向量转换到frame B
        """
        r = np.array([[np.cos(t[2]) * np.cos(t[1]),
                       np.cos(t[2]) * np.sin(t[1]) * np.sin(t[0]) - np.sin(t[2]) * np.cos(t[0]),
                       np.cos(t[2]) * np.sin(t[1]) * np.cos(t[0]) + np.sin(t[2]) * np.sin(t[0])],
                      [np.sin(t[2]) * np.cos(t[1]),
                       np.sin(t[2]) * np.sin(t[1]) * np.sin(t[0]) + np.cos(t[2]) * np.cos(t[0]),
                       np.sin(t[2]) * np.sin(t[1]) * np.cos(t[0]) - np.cos(t[2]) * np.sin(t[0])],
                      [-np.sin(t[1]), np.cos(t[1]) * np.sin(t[0]), np.cos(t[1]) * np.cos(t[0])]])
        return r.T

    @staticmethod
    def trans_matrix_ba(m, t):
        """
        从frame B到frame A的齐次变换矩阵（4x4）
        
        构造包含旋转和平移的齐次变换矩阵，将frame B中的点转换到frame A。
        
        齐次变换矩阵格式:
        [R(3x3)  m(3x1)]
        [0(1x3)    1   ]
        
        Args:
            m: frame B原点在frame A中的位置 [x, y, z]
            t: frame B相对于frame A的姿态（欧拉角） [roll, pitch, yaw]
        
        Returns:
            np.ndarray: 4x4齐次变换矩阵
        """
        r = np.array([[np.cos(t[2]) * np.cos(t[1]),
                       np.cos(t[2]) * np.sin(t[1]) * np.sin(t[0]) - np.sin(t[2]) * np.cos(t[0]),
                       np.cos(t[2]) * np.sin(t[1]) * np.cos(t[0]) + np.sin(t[2]) * np.sin(t[0])],
                      [np.sin(t[2]) * np.cos(t[1]),
                       np.sin(t[2]) * np.sin(t[1]) * np.sin(t[0]) + np.cos(t[2]) * np.cos(t[0]),
                       np.sin(t[2]) * np.sin(t[1]) * np.cos(t[0]) - np.cos(t[2]) * np.sin(t[0])],
                      [-np.sin(t[1]), np.cos(t[1]) * np.sin(t[0]), np.cos(t[1]) * np.cos(t[0])]])
        trans = np.hstack([r, np.array(m)[:, np.newaxis]])
        trans = np.vstack([trans, np.array([[0, 0, 0, 1]])])
        return trans

    @staticmethod
    def trans_matrix_ab(m, t):
        """
        从frame A到frame B的齐次变换矩阵（4x4）
        
        构造逆变换矩阵，将frame A中的点转换到frame B。
        
        逆变换计算:
        - 旋转部分: R^T (转置)
        - 平移部分: -R^T * m
        
        Args:
            m: frame B原点在frame A中的位置 [x, y, z]
            t: frame B相对于frame A的姿态（欧拉角） [roll, pitch, yaw]
        
        Returns:
            np.ndarray: 4x4齐次变换矩阵（逆变换）
        """
        r = np.array([[np.cos(t[2]) * np.cos(t[1]),
                       np.cos(t[2]) * np.sin(t[1]) * np.sin(t[0]) - np.sin(t[2]) * np.cos(t[0]),
                       np.cos(t[2]) * np.sin(t[1]) * np.cos(t[0]) + np.sin(t[2]) * np.sin(t[0])],
                      [np.sin(t[2]) * np.cos(t[1]),
                       np.sin(t[2]) * np.sin(t[1]) * np.sin(t[0]) + np.cos(t[2]) * np.cos(t[0]),
                       np.sin(t[2]) * np.sin(t[1]) * np.cos(t[0]) - np.cos(t[2]) * np.sin(t[0])],
                      [-np.sin(t[1]), np.cos(t[1]) * np.sin(t[0]), np.cos(t[1]) * np.cos(t[0])]])
        trans = np.hstack([r.T, -np.dot(r.T, np.array(m))[:, np.newaxis]])
        trans = np.vstack([trans, np.array([[0, 0, 0, 1]])])
        return trans

    @staticmethod
    def isRotationMatrix(R):
        """
        检查矩阵是否为有效的旋转矩阵
        
        有效的旋转矩阵满足: R^T * R = I (单位矩阵)
        
        Args:
            R: 3x3矩阵
        
        Returns:
            bool: 如果是有效的旋转矩阵返回True
        """
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self, R):
        """
        将旋转矩阵转换为欧拉角（ZYX顺序）
        
        注意: 与MATLAB的结果相同，但X和Z的顺序交换了。
        
        Args:
            R: 3x3旋转矩阵
        
        Returns:
            [roll, pitch, yaw]: 欧拉角（弧度）
        
        Raises:
            AssertionError: 如果R不是有效的旋转矩阵
        """
        assert (self.isRotationMatrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return [x, y, z]

    @staticmethod
    def quaternion2rpy(q):
        """
        四元数转欧拉角（RPY，ZYX顺序）
        
        四元数格式: [x, y, z, w]
        
        Args:
            q: 四元数 [x, y, z, w]
        
        Returns:
            [roll, pitch, yaw]: 欧拉角（弧度）
        """
        x, y, z, w = q[0], q[1], q[2], q[3]
        
        # Roll (绕X轴旋转)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        
        # Pitch (绕Y轴旋转)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
        
        # Yaw (绕Z轴旋转)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        
        return [roll, pitch, yaw]

    @staticmethod
    def rpy2quaternion(r):
        """
        欧拉角（RPY）转四元数
        
        Args:
            r: 欧拉角 [roll, pitch, yaw] (弧度)
        
        Returns:
            [x, y, z, w]: 四元数
        """
        roll, pitch, yaw = r[0], r[1], r[2]
        
        # 计算半角
        cr = np.cos(roll/2)
        sr = np.sin(roll/2)
        cp = np.cos(pitch/2)
        sp = np.sin(pitch/2)
        cy = np.cos(yaw/2)
        sy = np.sin(yaw/2)
        
        # 四元数分量
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        w = cr * cp * cy + sr * sp * sy
        
        return [x, y, z, w]

    @staticmethod
    def quaternion2axis_angle(q):
        """
        四元数转轴角表示
        
        轴角表示: 旋转轴向量 * 旋转角度
        
        Args:
            q: 四元数 [x, y, z, w]
        
        Returns:
            [ax, ay, az, angle]: 旋转轴[ax, ay, az]（单位向量）和旋转角度angle（弧度）
        """
        x, y, z, w = q[0], q[1], q[2], q[3]
        
        # 计算旋转角度
        angle = 2 * np.arccos(w)
        
        # 计算旋转轴
        s = np.sqrt(1 - w * w)
        if s < 0.001:  # 角度接近0，旋转轴可以任意
            x = x
            y = y
            z = z
        else:
            x = x / s
            y = y / s
            z = z / s
        
        return [x, y, z, angle]


if __name__ == '__main__':
    kinematic = Kinematics()
    r_org = [0.0, -1.62406, 0.0]
    q_org = [0.0, -0.72568, 0.0, 0.68803]

    q = kinematic.rpy2quaternion(r_org)
    r = kinematic.quaternion2rpy(q)

    r_axis_angle = kinematic.quaternion2axis_angle(q)

    r_org = torch.tensor(r_org)
    q_org = torch.tensor(q_org).reshape(1, -1)
    q2 = quat_from_euler_xyz(r_org[0], r_org[1], r_org[2])
    r2 = torch.vstack(get_euler_xyz(q2.reshape(1, -1))).T
    aa = 1

    # body_length = 0.366
    # body_wide = 0.094
    # hip_length = 0.08505
    # thigh_length = 0.2
    # calf_length = 0.2
    # # body_trans = np.array([0, 0.1, (3**0.5)*(thigh_length + calf_length)/2])
    # body_trans = np.array([0, 0, thigh_length + calf_length - 0.0001])
    # body_rot = np.array([0, 0, 0])
    # fr_x = body_length
    # fr_y = hip_length + body_wide/2
    # fr_z = 0
    # fr = np.array([fr_x, -fr_y, fr_z])
    # fl = np.array([fr_x, fr_y, fr_z])
    # rr = np.array([0, -fr_y, fr_z])
    # rl = np.array([0, fr_y, fr_z])
    # foot_pos = np.hstack([fr, fl, rr, rl])
    # ang = kinematic.inverse_kinematics(body_trans, body_rot, foot_pos)
    # aa = 1
