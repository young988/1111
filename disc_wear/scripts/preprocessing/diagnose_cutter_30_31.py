"""
诊断30、31号刀具修正后不过真实点的原因
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # 读取原始磨损数据
    wear_df = pd.read_csv(project_root / "data/processed/基于刀具位置的磨损量.csv", index_col=0)
    wear_df.index = wear_df.index.str.replace('#', '').astype(int)
    wear_df.columns = wear_df.columns.str.replace('R', '').astype(int)
    
    # 读取修正后的数据（如果存在）
    try:
        corrected_df = pd.read_csv(project_root / "data/processed/wear_per_timestep_final.csv")
        has_corrected = True
    except FileNotFoundError:
        print("警告：未找到修正后的数据文件")
        has_corrected = False
    
    measured_rings = wear_df.columns.tolist()
    
    print("=" * 80)
    print("30、31号刀具诊断报告")
    print("=" * 80)
    
    for cutter_id in [30, 31]:
        print(f"\n{'='*80}")
        print(f"刀具 #{cutter_id} 详细分析")
        print(f"{'='*80}\n")
        
        # 1. 真实数据特征
        true_values = wear_df.loc[cutter_id].values
        print("【1. 真实测量数据特征】")
        print(f"环号:   {measured_rings}")
        print(f"磨损值: {true_values.tolist()}")
        print(f"单调性: {'✓ 单调递增' if np.all(np.diff(true_values) >= 0) else '✗ 非单调'}")
        
        if not np.all(np.diff(true_values) >= 0):
            non_mono_idx = np.where(np.diff(true_values) < 0)[0]
            for idx in non_mono_idx:
                print(f"  非单调位置: R{measured_rings[idx]} ({true_values[idx]}) -> R{measured_rings[idx+1]} ({true_values[idx+1]})")
        
        # 2. 单调化后的影响
        print("\n【2. 单调化处理的影响】")
        monotonic_values = np.maximum.accumulate(true_values)
        print(f"单调化后: {monotonic_values.tolist()}")
        
        changed_indices = np.where(true_values != monotonic_values)[0]
        if len(changed_indices) > 0:
            print(f"被修改的点:")
            for idx in changed_indices:
                print(f"  R{measured_rings[idx]}: {true_values[idx]} -> {monotonic_values[idx]} (强制为累积最大值)")
        else:
            print("  无修改（数据本身单调）")
        
        # 3. 修正算法的矛盾
        print("\n【3. 修正算法中的矛盾】")
        if len(changed_indices) > 0:
            print("⚠️  发现矛盾：")
            print("  - 插值计算使用单调化后的值（ensure_monotonic）")
            print("  - 真实点处强制赋值为原始值（true_y_original）")
            print("  - 这导致插值结果与真实点赋值不一致！")
            print("\n  具体影响：")
            for idx in changed_indices:
                ring = measured_rings[idx]
                print(f"    R{ring}处：")
                print(f"      - 插值认为应该是 {monotonic_values[idx]} mm")
                print(f"      - 但被强制赋值为 {true_values[idx]} mm")
                print(f"      - 差异：{abs(monotonic_values[idx] - true_values[idx])} mm")
        else:
            print("✓ 数据单调，无此矛盾")
        
        # 4. 公式转换问题（31-42号刀具）
        if 31 <= cutter_id <= 42:
            print("\n【4. 公式转换的潜在问题】")
            print(f"  刀具 #{cutter_id} 使用 radius_to_volume_new 公式")
            d_deg = (cutter_id - 30) * 70 / 13
            print(f"  位置角度 d = {d_deg:.2f}°")
            print("  问题：")
            print("    1. 体积-半径双向转换涉及复杂的数值求解（fsolve/brentq）")
            print("    2. k值拟合基于体积，但修正时直接使用半径")
            print("    3. 数值误差可能在转换过程中累积")
            print("    4. 修正后的半径可能无法精确还原到真实半径")
        
        # 5. 如果有修正后的数据，检查实际误差
        if has_corrected:
            print("\n【5. 修正后实际误差检查】")
            corrected_col = f'cutter_{cutter_id}_wear_radius_corrected'
            
            if corrected_col in corrected_df.columns:
                # 按环号分组取最大值
                corrected_by_ring = corrected_df.groupby('ring_number')[corrected_col].max()
                
                print(f"{'环号':>6s} | {'真实值':>8s} | {'修正值':>8s} | {'误差':>8s} | {'状态':>6s}")
                print("-" * 50)
                
                max_error = 0
                for ring in measured_rings:
                    if ring in corrected_by_ring.index:
                        true_val = wear_df.loc[cutter_id, ring]
                        corrected_val = corrected_by_ring.loc[ring]
                        error = abs(corrected_val - true_val)
                        max_error = max(max_error, error)
                        
                        status = "✓" if error < 0.01 else "✗"
                        print(f"R{ring:>5d} | {true_val:>8.2f} | {corrected_val:>8.4f} | {error:>8.4f} | {status:>6s}")
                
                print(f"\n最大误差: {max_error:.4f} mm")
                
                if max_error > 0.01:
                    print(f"⚠️  修正后未能精确通过真实点（误差 > 0.01mm）")
            else:
                print(f"  未找到列 {corrected_col}")
    
    # 6. 生成对比图
    print("\n" + "="*80)
    print("生成诊断图表...")
    print("="*80)
    
    if has_corrected:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for idx, cutter_id in enumerate([30, 31]):
            ax1 = axes[idx, 0]
            ax2 = axes[idx, 1]
            
            # 左图：真实值 vs 单调化值
            true_vals = wear_df.loc[cutter_id].values
            mono_vals = np.maximum.accumulate(true_vals)
            
            ax1.plot(measured_rings, true_vals, 'ro-', label='原始真实值', markersize=8, linewidth=2)
            ax1.plot(measured_rings, mono_vals, 'b^--', label='单调化后', markersize=8, linewidth=2, alpha=0.7)
            ax1.set_xlabel('环号', fontsize=12)
            ax1.set_ylabel('磨损半径 (mm)', fontsize=12)
            ax1.set_title(f'刀具 #{cutter_id}: 真实值 vs 单调化值', fontsize=14)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # 标注差异点
            diff_mask = true_vals != mono_vals
            if np.any(diff_mask):
                diff_rings = np.array(measured_rings)[diff_mask]
                diff_true = true_vals[diff_mask]
                diff_mono = mono_vals[diff_mask]
                for r, t, m in zip(diff_rings, diff_true, diff_mono):
                    ax1.annotate(f'Δ={m-t:.1f}', xy=(r, t), xytext=(r, (t+m)/2),
                               fontsize=9, color='red', ha='center',
                               arrowprops=dict(arrowstyle='->', color='red', lw=1))
            
            # 右图：修正值 vs 真实值
            corrected_col = f'cutter_{cutter_id}_wear_radius_corrected'
            if corrected_col in corrected_df.columns:
                corrected_by_ring = corrected_df.groupby('ring_number')[corrected_col].max()
                
                # 获取所有环号的修正值
                all_rings = sorted(corrected_by_ring.index.tolist())
                corrected_all = corrected_by_ring.loc[all_rings].values
                
                ax2.plot(all_rings, corrected_all, 'g-', label='修正曲线', linewidth=1.5, alpha=0.8)
                ax2.plot(measured_rings, true_vals, 'ro', label='真实测量点', markersize=10, zorder=5)
                
                # 标注误差
                for ring in measured_rings:
                    if ring in corrected_by_ring.index:
                        true_val = wear_df.loc[cutter_id, ring]
                        corrected_val = corrected_by_ring.loc[ring]
                        error = abs(corrected_val - true_val)
                        if error > 0.01:
                            ax2.annotate(f'误差={error:.3f}', xy=(ring, true_val),
                                       xytext=(ring, true_val + 2),
                                       fontsize=8, color='red', ha='center',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                
                ax2.set_xlabel('环号', fontsize=12)
                ax2.set_ylabel('磨损半径 (mm)', fontsize=12)
                ax2.set_title(f'刀具 #{cutter_id}: 修正曲线 vs 真实点', fontsize=14)
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = project_root / "results" / "wear_correction_plots" / "diagnosis_cutter_30_31.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"诊断图表已保存: {output_path}")
        plt.close()
    
    print("\n" + "="*80)
    print("诊断完成")
    print("="*80)

if __name__ == "__main__":
    main()
