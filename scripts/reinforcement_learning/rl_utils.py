# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

import isaaclab.utils.math as math_utils


def setup_video_viewport(env, world_mode: bool = False):
    """设置视频录制时的viewport配置：grey studio光照 + world固定视角 + 高画质渲染"""

    # --- 1. 运行时强制开启高画质渲染设置 ---
    try:
        import carb
        settings = carb.settings.get_settings()

        # 强制设置渲染模式为 RaytracedLighting（确保RTX光线追踪生效）
        settings.set_string("/rtx/rendermode", "RaytracedLighting")

        # 开启所有光影特效（覆盖 headless.rendering.kit 的低质量默认值）
        settings.set_bool("/rtx/translucency/enabled", True)
        settings.set_bool("/rtx/reflections/enabled", True)
        settings.set_bool("/rtx/reflections/denoiser/enabled", True)
        settings.set_bool("/rtx/indirectDiffuse/enabled", True)
        settings.set_bool("/rtx/indirectDiffuse/denoiser/enabled", True)
        settings.set_bool("/rtx/directLighting/sampledLighting/enabled", True)
        settings.set_int("/rtx/directLighting/sampledLighting/samplesPerPixel", 4)
        settings.set_float("/rtx/sceneDb/ambientLightIntensity", 1.0)
        settings.set_bool("/rtx/shadows/enabled", True)
        settings.set_bool("/rtx/ambientOcclusion/enabled", True)
        settings.set_int("/rtx/ambientOcclusion/denoiserMode", 0)
        settings.set_bool("/rtx-transient/dldenoiser/enabled", True)
        settings.set_int("/rtx/post/dlss/execMode", 2)  # Quality

        # 高质量子像素和缓存
        settings.set_int("/rtx/raytracing/subpixel/mode", 1)
        settings.set_bool("/rtx/raytracing/cached/enabled", True)

        # replicator / tile 限制
        settings.set_int("/rtx/pathtracing/maxSamplesPerLaunch", 1000000)
        settings.set_int("/rtx/viewTile/limit", 1000000)

        # DLSS-G 关闭（与 tiled camera 不兼容）
        settings.set_bool("/rtx-transient/dlssg/enabled", False)

        print("[INFO] Runtime RTX HIGH-QUALITY render settings applied")
        print(f"  rendermode = {settings.get('/rtx/rendermode')}")
        print(f"  shadows    = {settings.get('/rtx/shadows/enabled')}")
        print(f"  reflections= {settings.get('/rtx/reflections/enabled')}")
        print(f"  GI         = {settings.get('/rtx/indirectDiffuse/enabled')}")
        print(f"  AO         = {settings.get('/rtx/ambientOcclusion/enabled')}")
        print(f"  transluc   = {settings.get('/rtx/translucency/enabled')}")
        print(f"  DLSS mode  = {settings.get('/rtx/post/dlss/execMode')}")
        print(f"  SPP        = {settings.get('/rtx/directLighting/sampledLighting/samplesPerPixel')}")
    except Exception as e:
        print(f"[WARNING] Failed to apply runtime render settings: {e}")
        import traceback
        traceback.print_exc()

    # --- 2. 设置grey studio灯光 + 修改地面颜色 ---
    try:
        import omni.usd
        from pxr import Usd, UsdLux, UsdShade, Sdf, Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()

        # == 2a. 修改 dome light 为灰色（背景色） ==
        sky_light_path = "/World/skyLight"
        sky_light_prim = stage.GetPrimAtPath(sky_light_path)
        if sky_light_prim.IsValid():
            dome_light = UsdLux.DomeLight(sky_light_prim)
            # 清除HDR纹理
            texture_attr = dome_light.GetTextureFileAttr()
            if texture_attr:
                texture_attr.Clear()
            # 设置灰色光照
            dome_light.GetIntensityAttr().Set(2000.0)
            dome_light.GetColorAttr().Set(Gf.Vec3f(0.8, 0.8, 0.8))
            print("[INFO] Modified sky light -> grey studio dome light")
        else:
            # 创建新的 dome light
            dome_light = UsdLux.DomeLight.Define(stage, "/World/GreyStudioLight")
            dome_light.GetIntensityAttr().Set(2000.0)
            dome_light.GetColorAttr().Set(Gf.Vec3f(0.8, 0.8, 0.8))
            print("[INFO] Created grey studio dome light")

        # == 2b. 修改地面颜色为灰色 ==
        # 地面Grid材质路径（Isaac Sim默认grid ground plane的材质）
        grid_shader_path = "/World/ground/terrain/Looks/theGrid/Shader"
        grid_shader_prim = stage.GetPrimAtPath(grid_shader_path)
        if grid_shader_prim.IsValid():
            # 修改 diffuse_tint 颜色
            tint_attr = grid_shader_prim.GetAttribute("inputs:diffuse_tint")
            if tint_attr:
                tint_attr.Set(Gf.Vec3f(0.5, 0.5, 0.5))
                print("[INFO] Set ground plane diffuse_tint to grey (0.5, 0.5, 0.5)")
            else:
                # 尝试创建属性
                tint_attr = grid_shader_prim.CreateAttribute(
                    "inputs:diffuse_tint", Sdf.ValueTypeNames.Color3f
                )
                tint_attr.Set(Gf.Vec3f(0.5, 0.5, 0.5))
                print("[INFO] Created ground plane diffuse_tint attribute (grey)")
        else:
            print(f"[INFO] Grid shader not found at {grid_shader_path}, trying alternative paths...")
            # 尝试搜索地面材质
            ground_prim = stage.GetPrimAtPath("/World/ground")
            if ground_prim.IsValid():
                # 遍历所有子prim查找Shader
                for prim in Usd.PrimRange(ground_prim):
                    if prim.GetTypeName() == "Shader":
                        tint_attr = prim.GetAttribute("inputs:diffuse_tint")
                        if tint_attr:
                            tint_attr.Set(Gf.Vec3f(0.5, 0.5, 0.5))
                            print(f"[INFO] Set ground shader {prim.GetPath()} diffuse_tint to grey")
                        # 也尝试设置 diffuse_color_constant
                        color_attr = prim.GetAttribute("inputs:diffuse_color_constant")
                        if color_attr:
                            color_attr.Set(Gf.Vec3f(0.5, 0.5, 0.5))
                            print(f"[INFO] Set ground shader {prim.GetPath()} diffuse_color_constant to grey")
            else:
                print("[WARNING] Ground plane prim not found")

        # == 2c. 额外：添加一个方向光以增强阴影效果 ==
        dist_light_path = "/World/VideoDirectionalLight"
        if not stage.GetPrimAtPath(dist_light_path).IsValid():
            dist_light = UsdLux.DistantLight.Define(stage, dist_light_path)
            dist_light.GetIntensityAttr().Set(3000.0)
            dist_light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
            dist_light.GetAngleAttr().Set(1.0)
            # 设置光照方向（朝下偏斜）
            xform = UsdGeom.Xformable(dist_light.GetPrim())
            xform.ClearXformOpOrder()
            rot_op = xform.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ)
            rot_op.Set(Gf.Vec3f(-45, 30, 0))
            print("[INFO] Added directional light for enhanced shadows")

        print("[INFO] Grey studio lighting + ground color setup complete")
    except Exception as e:
        print(f"[WARNING] Failed to set grey studio light rig: {e}")
        import traceback
        traceback.print_exc()

    # --- 3. 设置viewer模式 ---
    try:
        controller = env.unwrapped.viewport_camera_controller
        if world_mode:
            # 世界固定视角：相机不跟随机器人
            controller.update_view_to_world()
            controller.update_view_location(
                # eye=np.array([6.1, -3.4, 4.5]),
                # eye=np.array([4.5, 0.3, 2.8]),
                eye=np.array([0, 6, 1.2]),
                lookat=np.array([0, 0.0, 1.5])
            )
            print("[INFO] Set viewer follow mode to 'world' (fixed camera)")
        else:
            # robot跟随模式：相机跟随机器人位置和朝向
            controller.set_view_env_index(env_index=0)
            robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
            controller.update_view_location(
                eye=robot_pos + np.array([5, 5, 6]),
                lookat=robot_pos
            )
            print("[INFO] Set viewer follow mode to 'robot' (camera follows robot)")
    except Exception as e:
        print(f"[WARNING] Failed to set viewer follow mode: {e}")


def camera_follow(env):
    """robot跟随模式 - 相机跟随机器人位置和朝向，始终从侧面拍摄。

    相机偏移量在机器人局部坐标系中定义，随机器人yaw旋转，
    确保无论机器人朝向如何变化，相机始终拍摄相同的相对视角。
    """
    # 局部坐标系偏移: x=前后, y=左右, z=上下（机器人体坐标系）
    local_offset = np.array([6.0, 6.0, 6.0])

    robot = env.unwrapped.scene["robot"]
    robot_pos = robot.data.root_pos_w[0].cpu().numpy()
    robot_quat = robot.data.root_quat_w[0].cpu().numpy()  # (w, x, y, z)

    # 提取 yaw 角（只绕z轴旋转相机，忽略roll/pitch避免相机抖动）
    w, x, y, z = robot_quat
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # 将局部偏移量按yaw旋转到世界坐标系
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    world_offset = np.array([
        cos_yaw * local_offset[0] - sin_yaw * local_offset[1],
        sin_yaw * local_offset[0] + cos_yaw * local_offset[1],
        local_offset[2]
    ])

    env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)
    env.unwrapped.viewport_camera_controller.update_view_location(
        eye=robot_pos + np.array([5, 5, 6]), lookat=robot_pos
        # eye=robot_pos + world_offset, lookat=robot_pos
    )


class PlayDataLogger:
    """记录 play 过程中的机器人状态数据，用于后续绘制曲线。

    时间在机器人 episode 重置时自动归零（基于 episode_length_buf）。

    CSV 列:
        time_s, yaw_spine_pos, pitch_spine_pos, roll_spine_pos,
        body_roll, body_pitch, body_yaw,
        ang_vel_x, ang_vel_y, ang_vel_z,
        lin_vel_x, lin_vel_y, lin_vel_z,
        pos_x, pos_y, pos_z,
        cmd_lin_vel_x, cmd_lin_vel_y, cmd_ang_vel_z
    """

    SPINE_JOINT_NAMES = ["yaw_spine_joint", "pitch_spine_joint", "roll_spine_joint"]

    def __init__(self, env, log_dir: str, log_spine: bool = True):
        import os
        from datetime import datetime

        self._env = env
        self._data: list[dict] = []
        self._log_spine = log_spine

        # 获取机器人 articulation
        robot = self._env.unwrapped.scene["robot"]

        # 查找脊柱关节索引（仅在 log_spine=True 时）
        if self._log_spine:
            self._spine_joint_ids, self._spine_joint_names = robot.find_joints(self.SPINE_JOINT_NAMES)
            print(f"[PlayDataLogger] Found spine joints: {self._spine_joint_names} at indices {self._spine_joint_ids}")
        else:
            self._spine_joint_ids = None
            self._spine_joint_names = []
            print("[PlayDataLogger] No-spine mode: skipping spine joint logging")

        # 保存路径
        tag = "spine" if log_spine else "no_spine"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, "play_data", f"play_log_{tag}_{timestamp}.csv")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_step(self, dt: float):
        """记录当前步的数据。时间由 episode_length_buf 自动管理，重置时归零。

        Args:
            dt: 仿真时间步长 (秒/步)
        """
        from isaaclab.utils.math import euler_xyz_from_quat

        robot = self._env.unwrapped.scene["robot"]

        # 时间：直接用 episode_length_buf * dt，episode 重置时自动归零
        ep_len = self._env.unwrapped.episode_length_buf[0].item()
        time_s = round(ep_len * dt, 6)

        # --- 脊柱关节位置（仅 log_spine 模式） ---
        if self._log_spine:
            joint_pos = robot.data.joint_pos[0]
            spine_pos = joint_pos[self._spine_joint_ids].cpu().numpy()

        # --- 机体姿态（四元数 → 欧拉角） ---
        root_quat = robot.data.root_quat_w[0:1]
        roll, pitch, yaw = euler_xyz_from_quat(root_quat)
        roll, pitch, yaw = roll.item(), pitch.item(), yaw.item()

        # --- 机体角速度（body frame, 同 critic observation） ---
        ang_vel = robot.data.root_ang_vel_b[0].cpu().numpy()

        # --- 机体线速度（body frame, 同 critic observation） ---
        lin_vel = robot.data.root_lin_vel_b[0].cpu().numpy()

        # --- 机体位置 ---
        root_pos = robot.data.root_pos_w[0].cpu().numpy()

        # --- 速度命令 (lin_vel_x, lin_vel_y, ang_vel_z) ---
        vel_cmd = self._env.unwrapped.command_manager.get_command("base_velocity")[0].cpu().numpy()

        row = {
            "time_s": time_s,
        }
        if self._log_spine:
            row["yaw_spine_pos"] = round(float(spine_pos[0]), 6)
            row["pitch_spine_pos"] = round(float(spine_pos[1]), 6)
            row["roll_spine_pos"] = round(float(spine_pos[2]), 6)
        row.update({
            "body_roll": round(roll, 6),
            "body_pitch": round(pitch, 6),
            "body_yaw": round(yaw, 6),
            "ang_vel_x": round(float(ang_vel[0]), 6),
            "ang_vel_y": round(float(ang_vel[1]), 6),
            "ang_vel_z": round(float(ang_vel[2]), 6),
            "lin_vel_x": round(float(lin_vel[0]), 6),
            "lin_vel_y": round(float(lin_vel[1]), 6),
            "lin_vel_z": round(float(lin_vel[2]), 6),
            "pos_x": round(float(root_pos[0]), 6),
            "pos_y": round(float(root_pos[1]), 6),
            "pos_z": round(float(root_pos[2]), 6),
            "cmd_lin_vel_x": round(float(vel_cmd[0]), 6),
            "cmd_lin_vel_y": round(float(vel_cmd[1]), 6),
            "cmd_ang_vel_z": round(float(vel_cmd[2]), 6),
        })
        self._data.append(row)

    def save(self):
        """将记录的数据保存为 CSV 文件。"""
        import csv

        if not self._data:
            print("[PlayDataLogger] No data to save.")
            return

        fieldnames = list(self._data[0].keys())
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._data)

        print(f"[PlayDataLogger] Saved {len(self._data)} steps to: {self.log_path}")
        print(f"  Columns: {', '.join(fieldnames)}")
