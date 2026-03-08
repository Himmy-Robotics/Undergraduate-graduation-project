#!/usr/bin/env python3
"""将Himmy Mark2 JSON文件转换为txt格式
用于himmy_amp训练（txt文件实际上就是JSON格式，只是扩展名不同）
"""

import os
import shutil
from pathlib import Path


def main():
    print("=" * 80)
    print("Cheetah Data JSON -> TXT 转换")
    print("=" * 80)
    
    # 路径配置
    source_dir = Path("/data/zmli/Fast-Quadruped/amp_a1_jump/MetalHead/poselib/data/cheetah_data_json")
    target_dir = Path("/data/zmli/Fast-Quadruped/source/robot_lab/robot_lab/tasks/direct/himmy_amp/motions/datasets/cheetah_data")
    
    print(f"\n源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    
    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有JSON文件（递归搜索）
    json_files = sorted(source_dir.rglob("*.json"))
    
    if not json_files:
        print("\n错误: 未找到JSON文件")
        return
    
    print(f"\n找到 {len(json_files)} 个JSON文件")
    print("-" * 80)
    
    # 转换统计
    success_count = 0
    failed_count = 0
    
    # 复制并重命名
    for json_file in json_files:
        try:
            # 生成txt文件名（保持相同的文件名，只改扩展名）
            txt_filename = json_file.stem + ".txt"
            txt_file = target_dir / txt_filename
            
            # 复制文件
            shutil.copy2(json_file, txt_file)
            
            print(f"✅ {json_file.name} -> {txt_filename}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ {json_file.name}: {str(e)}")
            failed_count += 1
    
    # 总结
    print("-" * 80)
    print(f"\n转换完成!")
    print(f"成功: {success_count} 个")
    print(f"失败: {failed_count} 个")
    print(f"\n所有txt文件已保存到:")
    print(f"  {target_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
