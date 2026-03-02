"""
诊断：检查friction_df中是否包含所有测量环号
"""
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(".")

# 读取数据
wear_df = pd.read_csv(project_root / "data/processed/基于刀具位置的磨损量.csv", index_col=0)
wear_df.index = wear_df.index.str.replace('#', '').astype(int)
wear_df.columns = wear_df.columns.str.replace('R', '').astype(int)

friction_df = pd.read_csv(project_root / "data/processed/tbm_data_with_friction_energy.csv", low_memory=False)

measured_rings = wear_df.columns.tolist()
actual_rings_in_data = sorted(friction_df['ring_number'].unique())

print("=== 环号覆盖情况诊断 ===\n")

print(f"测量环号（来自磨损数据）: {measured_rings}")
print(f"实际环号（来自friction_df）: 共{len(actual_rings_in_data)}个环号")
print(f"  范围: {min(actual_rings_in_data)} ~ {max(actual_rings_in_data)}")

print("\n【关键检查】测量环号是否都在friction_df中？")
missing_rings = []
for ring in measured_rings:
    if ring not in actual_rings_in_data:
        missing_rings.append(ring)
        print(f"  ✗ R{ring} 不在friction_df中")
    else:
        print(f"  ✓ R{ring} 在friction_df中")

if missing_rings:
    print(f"\n⚠️  发现问题：{len(missing_rings)}个测量环号不在friction_df中")
    print(f"缺失的环号: {missing_rings}")
    print("\n这会导致：")
    print("1. 代码第410行：if ring in interp_wear_per_ring.index")
    print("   对于缺失的环号，这个条件为False")
    print("2. 缺失环号的真实值不会被添加到valid_measured_rings")
    print("3. 修正时无法在这些环号处强制过真实点")
    print("4. 导致修正值 ≠ 真实值")
else:
    print("\n✓ 所有测量环号都在friction_df中")
    print("环号匹配不是问题的原因")

print("\n【进一步检查】31号刀具的情况")
cid = 31
print(f"\n31号刀具的真实测量值:")
for ring in measured_rings:
    true_val = wear_df.loc[cid, ring]
    in_data = "✓" if ring in actual_rings_in_data else "✗"
    print(f"  R{ring}: {true_val} mm  {in_data}")
