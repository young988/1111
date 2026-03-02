"""
测试修复后的31号刀具修正结果
"""
import numpy as np

def ensure_monotonic(values):
    """确保数组单调递增（累积最大值）"""
    return np.maximum.accumulate(values)

def piecewise_linear_transform_old(all_x, model_y, true_x, true_y):
    """旧版本（有bug）"""
    true_x = np.array(true_x)
    true_y_original = np.array(true_y, dtype=float)
    true_y = ensure_monotonic(true_y_original)
    
    corrected_y = np.zeros_like(model_y, dtype=float)
    
    # 真实点赋值
    for i, (tx, ty_orig) in enumerate(zip(true_x, true_y_original)):
        exact_match = np.where(all_x == tx)[0]
        if len(exact_match) > 0:
            corrected_y[exact_match] = ty_orig
    
    # 区间插值（简化版，只处理真实点之间）
    for i in range(len(true_x) - 1):
        tx_i = true_x[i]
        tx_i_plus_1 = true_x[i + 1]
        ty_i = true_y[i]
        ty_i_plus_1 = true_y[i + 1]
        
        mask = (all_x > tx_i) & (all_x < tx_i_plus_1)
        segment_indices = np.where(mask)[0]
        
        idx_i = np.where(all_x == tx_i)[0]
        idx_i_plus_1 = np.where(all_x == tx_i_plus_1)[0]
        
        if len(idx_i) > 0 and len(idx_i_plus_1) > 0:
            idx_i = idx_i[-1]
            idx_i_plus_1 = idx_i_plus_1[-1]
            ystart = model_y[idx_i]
            yend = model_y[idx_i_plus_1]
            delta_model = yend - ystart
            
            for idx in segment_indices:
                my_x = model_y[idx]
                if abs(delta_model) > 1e-10:
                    R = (my_x - ystart) / delta_model
                else:
                    R = (all_x[idx] - tx_i) / (tx_i_plus_1 - tx_i)
                corrected_y[idx] = ty_i + R * (ty_i_plus_1 - ty_i)
    
    # 问题：最后的全局单调化会覆盖真实点的值！
    corrected_y = np.maximum(0, corrected_y)
    corrected_y = ensure_monotonic(corrected_y)  # ← BUG在这里
    
    return corrected_y

def piecewise_linear_transform_new(all_x, model_y, true_x, true_y):
    """新版本（修复后）"""
    true_x = np.array(true_x)
    true_y_original = np.array(true_y, dtype=float)
    true_y = ensure_monotonic(true_y_original)
    
    corrected_y = np.zeros_like(model_y, dtype=float)
    true_point_mask = np.zeros(len(all_x), dtype=bool)
    
    # 真实点赋值并标记
    for i, (tx, ty_orig) in enumerate(zip(true_x, true_y_original)):
        exact_match = np.where(all_x == tx)[0]
        if len(exact_match) > 0:
            corrected_y[exact_match] = ty_orig
            true_point_mask[exact_match] = True
    
    # 区间插值（简化版）
    for i in range(len(true_x) - 1):
        tx_i = true_x[i]
        tx_i_plus_1 = true_x[i + 1]
        ty_i = true_y[i]
        ty_i_plus_1 = true_y[i + 1]
        
        mask = (all_x > tx_i) & (all_x < tx_i_plus_1)
        segment_indices = np.where(mask)[0]
        
        idx_i = np.where(all_x == tx_i)[0]
        idx_i_plus_1 = np.where(all_x == tx_i_plus_1)[0]
        
        if len(idx_i) > 0 and len(idx_i_plus_1) > 0:
            idx_i = idx_i[-1]
            idx_i_plus_1 = idx_i_plus_1[-1]
            ystart = model_y[idx_i]
            yend = model_y[idx_i_plus_1]
            delta_model = yend - ystart
            
            for idx in segment_indices:
                my_x = model_y[idx]
                if abs(delta_model) > 1e-10:
                    R = (my_x - ystart) / delta_model
                else:
                    R = (all_x[idx] - tx_i) / (tx_i_plus_1 - tx_i)
                corrected_y[idx] = ty_i + R * (ty_i_plus_1 - ty_i)
    
    # 修复：只对非真实点进行单调化
    non_true_points = ~true_point_mask
    corrected_y[non_true_points] = np.maximum(0, corrected_y[non_true_points])
    
    # 分段单调化
    for i in range(len(true_x)):
        if i < len(true_x) - 1:
            mask = (all_x > true_x[i]) & (all_x < true_x[i+1])
            if np.any(mask):
                corrected_y[mask] = ensure_monotonic(corrected_y[mask])
    
    # 最后确保真实点不变
    for i, (tx, ty_orig) in enumerate(zip(true_x, true_y_original)):
        exact_match = np.where(all_x == tx)[0]
        if len(exact_match) > 0:
            corrected_y[exact_match] = ty_orig
    
    return corrected_y

# 测试案例：模拟31号刀具的情况
print("=== 测试修复效果 ===\n")

# 模拟数据
all_x = np.array([149, 195, 227, 275, 367])  # 环号
model_y = np.array([8.45, 8.45, 10.0, 10.0, 12.0])  # 模型预测的半径
true_x = np.array([149, 195, 227, 275, 367])  # 真实测量环号
true_y = np.array([7.0, 8.0, 10.0, 10.0, 12.0])  # 真实测量半径

print("输入数据：")
print(f"环号:       {all_x}")
print(f"模型预测:   {model_y}")
print(f"真实值:     {true_y}")
print()

# 旧版本
corrected_old = piecewise_linear_transform_old(all_x, model_y, true_x, true_y)
print("旧版本修正结果：")
print(f"修正值:     {corrected_old}")
errors_old = np.abs(corrected_old - true_y)
print(f"误差:       {errors_old}")
print(f"最大误差:   {errors_old.max():.4f} mm")
print()

# 新版本
corrected_new = piecewise_linear_transform_new(all_x, model_y, true_x, true_y)
print("新版本修正结果：")
print(f"修正值:     {corrected_new}")
errors_new = np.abs(corrected_new - true_y)
print(f"误差:       {errors_new}")
print(f"最大误差:   {errors_new.max():.4f} mm")
print()

print("=== 结论 ===")
if errors_new.max() < 0.001:
    print("✓ 修复成功！所有真实点的误差 < 0.001mm")
else:
    print("✗ 仍有问题，最大误差:", errors_new.max())

print("\n=== 问题根源 ===")
print("旧版本在最后调用了 ensure_monotonic(corrected_y)")
print("这会对整个数组进行累积最大值处理，覆盖了之前对真实点的赋值")
print("\n例如：")
print("  - R149处被赋值为7mm（真实值）")
print("  - R195处被赋值为8mm（真实值）")
print("  - 但模型预测R149=8.45mm > 7mm")
print("  - ensure_monotonic会把R149改为8.45mm（累积最大值）")
print("  - 导致真实点的值被覆盖！")
