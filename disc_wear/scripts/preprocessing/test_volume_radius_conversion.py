"""
测试体积-半径转换函数的正确性
"""
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "scripts" / "preprocessing"))

from calculate_wear_per_timestep import (
    radius_to_volume, 
    volume_to_radius,
    radius_to_volume_new,
    volume_to_radius_new
)

def test_original_formula():
    """测试原始公式的正向和反向转换"""
    print("=" * 60)
    print("测试原始公式 (1-30号刀具)")
    print("=" * 60)
    
    # 测试一些半径值
    test_radii = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    
    # 使用相同的参数
    T = 25.4
    R = 241.3
    theta_deg = 20
    N = 1
    
    print("\n半径 -> 体积 -> 半径 (往返测试)")
    print(f"{'原始半径 (mm)':<15} {'体积 (mm³)':<20} {'恢复半径 (mm)':<15} {'误差 (mm)':<15}")
    print("-" * 65)
    
    for r in test_radii:
        # 正向：半径 -> 体积
        volume = radius_to_volume(r, T=T, R=R, theta_deg=theta_deg, N=N)
        
        # 反向：体积 -> 半径（使用相同参数）
        recovered_r = volume_to_radius(volume, T=T, R=R, theta_deg=theta_deg, N=N)
        
        # 计算误差
        error = abs(recovered_r - r)
        
        status = "✓" if error < 1e-6 else "✗"
        print(f"{r:<15.4f} {volume:<20.4f} {recovered_r:<15.4f} {error:<15.6e} {status}")
    
    print("\n✓ 原始公式测试完成")


def test_new_formula():
    """测试新公式的正向和反向转换"""
    print("\n" + "=" * 60)
    print("测试新公式 (31-44号刀具)")
    print("=" * 60)
    
    # 测试一些半径值和不同的刀具编号
    test_radii = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    test_cutter_ids = [31, 35, 40, 43]  # 测试几个不同的刀具编号
    
    print("\n半径 -> 体积 -> 半径 (往返测试)")
    print(f"{'刀具ID':<8} {'原始半径 (mm)':<15} {'体积 (mm³)':<20} {'恢复半径 (mm)':<15} {'误差 (mm)':<15}")
    print("-" * 73)
    
    for cutter_id in test_cutter_ids:
        for r in test_radii[:3]:  # 只测试前3个半径值以节省空间
            # 正向：半径 -> 体积
            volume = radius_to_volume_new(r, cutter_id=cutter_id)
            
            # 反向：体积 -> 半径
            recovered_r = volume_to_radius_new(volume, cutter_id=cutter_id)
            
            # 计算误差
            error = abs(recovered_r - r)
            
            status = "✓" if error < 1e-3 else "✗"
            print(f"{cutter_id:<8} {r:<15.4f} {volume:<20.4f} {recovered_r:<15.4f} {error:<15.6e} {status}")
    
    print("\n✓ 新公式测试完成")


def test_array_conversion():
    """测试数组批量转换"""
    print("\n" + "=" * 60)
    print("测试数组批量转换")
    print("=" * 60)
    
    # 创建一组半径值
    radii_array = np.linspace(0.1, 10, 20)
    
    # 参数
    T = 25.4
    R = 241.3
    theta_deg = 20
    N = 1
    
    print("\n原始公式 - 数组转换:")
    volumes = radius_to_volume(radii_array, T=T, R=R, theta_deg=theta_deg, N=N)
    recovered_radii = volume_to_radius(volumes, T=T, R=R, theta_deg=theta_deg, N=N)
    max_error = np.max(np.abs(recovered_radii - radii_array))
    print(f"  测试点数: {len(radii_array)}")
    print(f"  最大误差: {max_error:.6e} mm")
    print(f"  状态: {'✓ 通过' if max_error < 1e-6 else '✗ 失败'}")
    
    print("\n新公式 - 数组转换 (刀具31号):")
    cutter_id = 31
    volumes_new = radius_to_volume_new(radii_array, cutter_id=cutter_id)
    recovered_radii_new = volume_to_radius_new(volumes_new, cutter_id=cutter_id)
    max_error_new = np.max(np.abs(recovered_radii_new - radii_array))
    print(f"  测试点数: {len(radii_array)}")
    print(f"  最大误差: {max_error_new:.6e} mm")
    print(f"  状态: {'✓ 通过' if max_error_new < 1e-3 else '✗ 失败'}")


def compare_formulas():
    """比较两个公式的差异"""
    print("\n" + "=" * 60)
    print("比较原始公式与新公式")
    print("=" * 60)
    
    test_radii = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    cutter_ids = [31, 35, 40, 43]
    
    print(f"\n{'半径 (mm)':<12} {'原始体积 (mm³)':<20} {'新体积 (刀具31) (mm³)':<25} {'差异 (%)':<15}")
    print("-" * 72)
    
    for r in test_radii:
        vol_old = radius_to_volume(r)
        vol_new = radius_to_volume_new(r, cutter_id=31)
        diff_percent = abs(vol_new - vol_old) / vol_old * 100 if vol_old > 0 else 0
        
        print(f"{r:<12.4f} {vol_old:<20.4f} {vol_new:<25.4f} {diff_percent:<15.2f}")
    
    print(f"\n不同刀具编号的体积对比 (半径 = 1.0 mm):")
    print(f"{'刀具ID':<10} {'d (度)':<15} {'体积 (mm³)':<20}")
    print("-" * 45)
    for cid in cutter_ids:
        d_deg = (cid - 30) * 70 / 13
        vol = radius_to_volume_new(1.0, cutter_id=cid)
        print(f"{cid:<10} {d_deg:<15.2f} {vol:<20.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("体积-半径转换函数测试")
    print("=" * 60)
    
    try:
        test_original_formula()
        test_new_formula()
        test_array_conversion()
        compare_formulas()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
