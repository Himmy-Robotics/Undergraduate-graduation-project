#!/usr/bin/env python3
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Himmy Mark2 快速启动训练脚本

该脚本简化了训练启动过程，可直接运行开始训练。
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目源路径到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "source"))


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(
        description="Himmy Mark2 快速训练启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 平坦地形训练
  python start_training.py --terrain flat
  
  # 粗糙地形训练
  python start_training.py --terrain rough
  
  # 使用自定义参数
  python start_training.py --terrain flat --num_envs 2048 --headless --video
  
  # 从已有模型继续训练
  python start_training.py --terrain rough --load_run <experiment_name>
        """
    )
    
    parser.add_argument(
        "--terrain",
        type=str,
        choices=["flat", "rough"],
        default="flat",
        help="地形类型 (flat=平坦, rough=粗糙, 默认: flat)"
    )
    
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4096,
        help="并行环境数 (默认: 4096, 根据 GPU 内存调整)"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="无窗口运行模式 (默认启用)"
    )
    
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="录制训练视频 (每 2000 步一次)"
    )
    
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="最大训练迭代次数 (可选覆盖)"
    )
    
    parser.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="从该运行恢复或继续训练"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="加载的检查点迭代号"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子 (可选)"
    )
    
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="运行名称后缀"
    )
    
    args = parser.parse_args()
    
    # 构建训练命令
    task_name = f"RobotLab-Isaac-Velocity-{args.terrain.capitalize()}-Himmy-Mark2-v0"
    
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"),
        f"--task {task_name}",
        f"--num_envs {args.num_envs}",
    ]
    
    if args.headless:
        cmd.append("--headless")
    
    if args.video:
        cmd.append("--video")
    
    if args.max_iterations is not None:
        cmd.append(f"--max_iterations {args.max_iterations}")
    
    if args.load_run is not None:
        cmd.append(f"--load_run {args.load_run}")
    
    if args.checkpoint is not None:
        cmd.append(f"--checkpoint {args.checkpoint}")
    
    if args.seed is not None:
        cmd.append(f"--seed {args.seed}")
    
    if args.run_name is not None:
        cmd.append(f"--run_name {args.run_name}")
    
    # 打印命令信息
    print("=" * 70)
    print("Himmy Mark2 强化学习训练")
    print("=" * 70)
    print(f"\n📋 训练配置:")
    print(f"  地形:          {args.terrain.upper()}")
    print(f"  环境数:        {args.num_envs}")
    print(f"  无窗口模式:    {args.headless}")
    print(f"  录制视频:      {args.video}")
    if args.max_iterations:
        print(f"  迭代次数:      {args.max_iterations}")
    if args.load_run:
        print(f"  加载运行:      {args.load_run}")
    if args.checkpoint:
        print(f"  加载检查点:    {args.checkpoint}")
    print(f"\n🚀 启动命令:")
    print(f"  {' '.join(cmd)}")
    print("\n" + "=" * 70)
    print("训练启动中... 按 Ctrl+C 暂停训练\n")
    
    # 执行训练
    try:
        os.system(" ".join(cmd))
    except KeyboardInterrupt:
        print("\n\n⏸️  训练已暂停")
        print("💾 要恢复训练，运行:")
        print(f"  python start_training.py --terrain {args.terrain} --resume")


if __name__ == "__main__":
    main()
