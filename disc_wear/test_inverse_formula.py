#!/usr/bin/env python3
"""
测试修正后的volume_to_radius_new函数
"""
import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "scripts" / "preprocessing"))

from calculate_wear_per_timestep import radius_to_volume_new, volume_to_radius_new

def test_inverse_formula():
    """测试正向和反向公式的一致性"""
    
    print("=" * 80)
    print("测试修正后的volume_to_radius_new函数")
    print("=" * 80)
    
    # 测试参数
    test_radii = [10.0, 15.0, 20.0, 25.0, 30.0]
    test_cutters = [31, 35, 38, 40, 43]
    
    print("\n测试正向和反向转换的一致性:")
    print("-" * 80)
    print(f"{'刀具':<8} {'原始半径':<12} {'计算体积':<15} {'反算半径':<12} {'误差':<12} {'状态':<8}")
    print("-" * 80)
    
    max_error = 0
    total_tests = 0
    successful_tests = 0
    
    for cutter_id in test_cutters:
        d_deg = (cutter_id - 30) * 70 / 13
        print(f"\n刀具 #{cutter_id} (d={d_deg:.2f}°):")
        
        for radius in test_radii:
            # 正向: 半径 -> 体积
            volume = radius_to_volume_new(radius, cutter_id=cutter_id)
            
            # 反向: 体积 -> 半径
            try:
                radius_recovered = volume_to_radius_new(volume, cutter_id=cutter_id, initial_guess=radius)
                error = abs(radius_recovered - radius)
                max_error = max(max_error, error)
                
                status = "✓" if error < 0.01 else "⚠️" if error < 0.1 else "✗"
                successful_tests += 1 if error < 0.1 else 0
                
                print(f"  {cutter_id:<8} {radius:<12.2f} {volume:<15.2f} {radius_recovered:<12.4f} {error:<12.6f} {status:<8}")
            except Exception as e:
                print(f"  {cutter_id:<8} {radius:<12.2f} {volume:<15.2f} {'失败':<12} {'-':<12} ✗")
                print(f"    错误: {e}")
            
            total_tests += 1
    
    print("\n" + "=" * 80)
    print("测试总结:")
    print(f"  总测试数: {total_tests}")
    print(f"  成功数 (误差<0.1mm): {successful_tests}")
    print(f"  成功率: {successful_tests/total_tests*100:.1f}%")
    print(f"  最大误差: {max_error:.6f} mm")
    print("=" * 80)
    
    # 测试边界情况
    print("\n测试边界情况:")
    print("-" * 80)
    
    # 测试刀具43号（d=70°，临界情况）
    print("\n刀具 #43 (d=70°, 临界情况):")
    for radius in [20.0, 30.0, 40.0]:
        volume = radius_to_volume_new(radius, cutter_id=43)
        try:
            radius_recovered = volume_to_radius_new(volume, cutter_id=43, initial_guess=radius)
            error = abs(radius_recovered - radius)
            print(f"  半径 {radius:.1f}mm -> 体积 {volume:.2f}mm³ -> 半径 {radius_recovered:.4f}mm (误差: {error:.6f}mm)")
        except Exception as e:
            print(f"  半径 {radius:.1f}mm -> 失败: {e}")
    
    # 测试刀具31号（d=5.38°，比值最大）
    print("\n刀具 #31 (d=5.38°, 比值最大):")
    for radius in [20.0, 30.0, 40.0]:
        volume = radius_to_volume_new(radius, cutter_id=31)
        try:
            radius_recovered = volume_to_radius_new(volume, cutter_id=31, initial_guess=radius)
            error = abs(radius_recovered - radius)
            print(f"  半径 {radius:.1f}mm -> 体积 {volume:.2f}mm³ -> 半径 {radius_recovered:.4f}mm (误差: {error:.6f}mm)")
        except Exception as e:
            print(f"  半径 {radius:.1f}mm -> 失败: {e}")

if __name__ == "__main__":
    test_inverse_formula()
