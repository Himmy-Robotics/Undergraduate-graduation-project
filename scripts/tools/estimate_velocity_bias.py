#!/usr/bin/env python3
"""角速度分布分析脚本 - 估算 velocity_bias 值"""

import numpy as np

def simulate_gallop_velocity(num_steps=2000, cycle_time=0.3, dt=0.02):
    """模拟 gallop 步态中四条腿的角速度"""
    t = np.arange(0, num_steps * dt, dt)
    phase = (t / cycle_time) % 1.0
    
    # 后腿：支撑相占 60%，摆动相占 40%
    duty_rear = 0.6
    rl_vel = np.where(phase < duty_rear, 
                      3.0 * np.sin(np.pi * phase / duty_rear),
                      -2.0 * np.sin(np.pi * (phase - duty_rear) / (1 - duty_rear)))
    rr_vel = rl_vel * 0.95 + np.random.randn(len(t)) * 0.3
    
    # 前腿：相位偏移约 50%
    front_phase = (phase + 0.5) % 1.0
    duty_front = 0.6
    fl_vel = np.where(front_phase < duty_front,
                      2.5 * np.sin(np.pi * front_phase / duty_front),
                      -2.0 * np.sin(np.pi * (front_phase - duty_front) / (1 - duty_front)))
    fr_vel = fl_vel * 0.95 + np.random.randn(len(t)) * 0.3
    
    return rl_vel, rr_vel, fl_vel, fr_vel

def analyze():
    print("="*60)
    print("角速度分布分析 - 估算 velocity_bias")
    print("="*60)
    
    rl_vel, rr_vel, fl_vel, fr_vel = simulate_gallop_velocity()
    
    rear_avg = (rl_vel + rr_vel) / 2.0
    front_avg = -(fl_vel + fr_vel) / 2.0
    effective_vel_all = (rear_avg + front_avg) / 2.0
    
    print(f"\n后腿平均速度:  mean={rear_avg.mean():.3f}")
    print(f"前腿转换后:    mean={front_avg.mean():.3f}")
    print(f"全腿模式:      mean={effective_vel_all.mean():.3f}")
    
    total = len(effective_vel_all)
    
    print(f"\n{'bias':>6} | {'伸展%':>8} | {'收缩%':>8}")
    print("-" * 30)
    
    for bias in np.arange(0.0, 4.0, 0.5):
        biased = effective_vel_all - bias
        ext = np.sum(biased > 0) / total * 100
        flx = np.sum(biased < 0) / total * 100
        print(f"{bias:6.1f} | {ext:7.1f}% | {flx:7.1f}%")
    
    # 找最佳值
    target = 0.70
    best_bias = 0.0
    best_diff = float('inf')
    for bias in np.arange(-2.0, 6.0, 0.05):
        biased = effective_vel_all - bias
        ratio = np.sum(biased < 0) / total
        if abs(ratio - target) < best_diff:
            best_diff = abs(ratio - target)
            best_bias = bias
    
    print(f"\n推荐 velocity_bias = {best_bias:.2f} (达到约70%收缩)")

if __name__ == "__main__":
    analyze()
