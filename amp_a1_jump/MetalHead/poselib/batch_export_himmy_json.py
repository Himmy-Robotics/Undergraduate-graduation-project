#!/usr/bin/env python3
"""批量导出Himmy Mark2 JSON文件
将amp_himmy_mark2目录中的所有npy文件导出为json格式，保存到mocap_motions_himmy目录
"""

import os
import sys
import json
from pathlib import Path
import numpy as np

# 添加当前目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from himmy_json_exporter import export_himmy_json


def find_all_himmy_npy_files(base_dir):
    """查找所有Himmy npy文件
    
    Args:
        base_dir: amp_himmy_mark2基础目录
    
    Returns:
        npy文件路径列表
    """
    npy_files = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"错误: 目录不存在: {base_dir}")
        return npy_files
    
    for npy_file in base_path.rglob("*.npy"):
        npy_files.append(npy_file)
    
    return sorted(npy_files)


def main():
    print("=" * 80)
    print("批量导出Himmy Mark2 JSON文件")
    print("=" * 80)
    
    # 目录配置
    base_dir = Path(script_dir) / "data"
    input_dir = base_dir / "amp_himmy_mark2"
    output_dir = base_dir / "mocap_motions_himmy"
    
    # 指定要处理的文件夹
    folders_to_process = [
        "trot_forward0",
        "trot_forward_left_right_turn1",
        "gallop_forward1",
        "gallop_forward0"
    ]
    
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"处理文件夹: {folders_to_process}")
    print("=" * 80)
    
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 统计
    total_files = 0
    success_files = 0
    failed_files = []
    
    # 处理每个指定的文件夹
    for folder_name in folders_to_process:
        folder_path = input_dir / folder_name
        
        if not folder_path.exists():
            print(f"\n警告: 文件夹不存在: {folder_path}")
            continue
        
        print(f"\n处理文件夹: {folder_name}")
        print("-" * 80)
        
        # 查找该文件夹中的所有.npy文件
        npy_files = list(folder_path.glob("*.npy"))
        
        if not npy_files:
            print(f"  未找到.npy文件")
            continue
        
        print(f"  找到 {len(npy_files)} 个.npy文件")
        
        # 处理每个.npy文件
        for npy_file in sorted(npy_files):
            total_files += 1
            
            # 构造输出路径（保持子文件夹结构）
            relative_path = npy_file.relative_to(input_dir)
            output_path = output_dir / relative_path.parent / (npy_file.stem + ".json")
            
            print(f"\n  [{total_files}] {npy_file.name}")
            print(f"      输入: {npy_file}")
            print(f"      输出: {output_path}")
            
            try:
                # 确保输出目录存在
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 导出JSON
                export_himmy_json(str(npy_file), str(output_path), visualize=False)
                success_files += 1
                print(f"      ✓ 成功")
                
            except Exception as e:
                print(f"      ✗ 失败: {e}")
                failed_files.append((npy_file.name, str(e)))
    
    # 打印总结
    print("\n" + "=" * 80)
    print("批量导出完成!")
    print("=" * 80)
    print(f"总文件数: {total_files}")
    print(f"成功: {success_files}")
    print(f"失败: {len(failed_files)}")
    
    if failed_files:
        print("\n失败的文件:")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    
    print(f"\n输出目录: {output_dir}")
    print("=" * 80)
    print(f"\n输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有npy文件
    print(f"\n正在搜索npy文件...")
    npy_files = find_all_himmy_npy_files(input_dir)
    
    if not npy_files:
        print("错误: 未找到任何npy文件")
        return
    
    print(f"找到 {len(npy_files)} 个npy文件")
    print("-" * 80)
    
    # 统计
    success_count = 0
    failed_count = 0
    failed_files = []
    
    # 批量导出
    for idx, npy_file in enumerate(npy_files, 1):
        rel_path = npy_file.relative_to(input_dir)
        print(f"\n[{idx}/{len(npy_files)}] 处理: {rel_path}")
        
        # 生成输出文件名（使用相同的文件名，但改后缀为.json）
        output_file = output_dir / npy_file.stem
        output_file = output_file.with_suffix(".json")
        
        try:
            # 导出JSON
            export_himmy_json(str(npy_file), str(output_file), visualize=False)
            print(f"  ✅ 成功: {output_file.name}")
            success_count += 1
        except Exception as e:
            print(f"  ❌ 失败: {str(e)[:100]}")
            failed_count += 1
            failed_files.append((str(rel_path), str(e)[:100]))
        
        print("-" * 80)
    
    # 总结
    print("\n" + "=" * 80)
    print("批量导出完成!")
    print(f"成功: {success_count} 个")
    print(f"失败: {failed_count} 个")
    
    if failed_files:
        print("\n失败的文件:")
        for file, reason in failed_files:
            print(f"  - {file}")
            print(f"    原因: {reason}")
    
    print("\n所有JSON文件已保存到:")
    print(f"  {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
