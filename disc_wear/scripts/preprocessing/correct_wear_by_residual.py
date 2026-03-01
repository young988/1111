"""
使用分段归一化重映射法修正磨损拟合值（橡皮筋理论）
核心思想：
- 真实值坐标：(Tx, Ty)，10个测量点 + 原点(0,0) = 11个点
- 拟合值坐标：(Mx, My)，稠密序列
- 真实值与拟合值对齐：真实值视为对应环最后一个记录点，与拟合值相应环最后一个点在横轴上对齐
- 在每对相邻真实值点之间对拟合值进行修正，共10个区间

算法：分段归一化重映射
对每个区间 [Tx[i], Tx[i+1]]:
  ystart = My[i] (拟合值在 Tx[i] 处的值)
  yend = My[i+1] (拟合值在 Tx[i+1] 处的值)
  归一化比例：R(x) = [My(x) - ystart] / [yend - ystart]
  修正值：Y = Ty[i] + R(x) * (Ty[i+1] - Ty[i])
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def radius_to_volume(delta_r, T=25.4, R=241.3, theta_deg=20, N=1):
    """将磨损半径转换为磨损体积（原始公式，用于1-30号刀具）"""
    theta_rad = np.radians(theta_deg)
    tan_theta = np.tan(theta_rad)
    
    term1 = T * R
    term2 = -(T / 2) * delta_r
    term3 = R * tan_theta * delta_r
    term4 = -(2 / 3) * tan_theta * (delta_r ** 2)
    
    volume = 2 * np.pi * delta_r * (term1 + term2 + term3 + term4) * N
    return volume


def radius_to_volume_new(delta_r, cutter_id, T=25.4, H=241.3, theta_deg=20):
    """
    Convert wear radius to wear volume (New formula for cutters 31-42).
    
    根据公式文档：
    - 1-30号刀具：使用原始公式
    - 31-42号刀具：使用新公式（本函数）
    - 43号刀具：使用原始公式
    - 44号刀具：磨损量与43号完全一致，使用原始公式
    
    新公式计算步骤：
    1. 计算 d = (cutter_id - 30) × 70/13 (度)
    2. 计算 h₀ = [(T/2 + R/tan(d)) · tan(90°-θ)] / [tan(90°-θ) - tan(d)]
    3. 计算 h₁ = [R/tan(d) - T/2] · [1 - sin(d)·cos(θ)/sin(90°+θ-d)] · tan(d)
    4. 当 h₀ = 2R 时，计算 condition = h₀/tan(d) - h₀/tan(90°-θ)
    5. 判断：
       - 若 condition > T，使用公式1：V = V₀ - V₁
       - 否则，使用公式2：V = π·condition·h₀·[H - h₀/3]
    
    Parameters:
    - delta_r: 磨损半径 R (mm)
    - cutter_id: 刀具编号 (31-42)
    - T: 刀具厚度 (mm), 默认 25.4
    - H: 刀具高度 (mm), 默认 2413
    - theta_deg: 刀具角度 θ (度), 默认 20
    
    Returns:
    - V: 磨损体积 (mm³)
    """
    # 转换角度为弧度
    theta_rad = np.radians(theta_deg)
    tan_theta = np.tan(theta_rad)
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)
    
    # 计算 d = (cutter_id - 30) × 70/13 (度)
    d_deg = (cutter_id - 30) * 70 / 13
    d_rad = np.radians(d_deg)
    tan_d = np.tan(d_rad)
    sin_d = np.sin(d_rad)
    
    # 计算 h₀ = [(T/2 + R/tan(d)) · tan(90°-θ)] / [tan(90°-θ) - tan(d)]
    angle_90_minus_theta = np.radians(90) - theta_rad
    tan_90_minus_theta = np.tan(angle_90_minus_theta)
    
    numerator_h0 = (T / 2 + delta_r / tan_d) * tan_90_minus_theta
    denominator_h0 = tan_90_minus_theta - tan_d
    
    # 避免除零
    epsilon = 1e-6
    if isinstance(denominator_h0, np.ndarray):
        is_singular = np.abs(denominator_h0) < epsilon
        h0 = np.where(is_singular, 2 * delta_r, numerator_h0 / np.where(is_singular, epsilon, denominator_h0))
    else:
        if abs(denominator_h0) < epsilon:
            h0 = 2 * delta_r
        else:
            h0 = numerator_h0 / denominator_h0
    
    # 计算 h₁ = [R/tan(d) - T/2] · [1 - sin(d)·cos(θ)/sin(90°+θ-d)] · tan(d)
    angle_90_plus_theta_minus_d = np.radians(90) + theta_rad - d_rad
    sin_angle_h1 = np.sin(angle_90_plus_theta_minus_d)
    
    # 避免除零
    if isinstance(sin_angle_h1, np.ndarray):
        sin_angle_h1 = np.where(np.abs(sin_angle_h1) < epsilon, epsilon, sin_angle_h1)
    else:
        if abs(sin_angle_h1) < epsilon:
            sin_angle_h1 = epsilon
    
    bracket_h1 = 1 - (sin_d * cos_theta) / sin_angle_h1
    h1 = (delta_r / tan_d - T / 2) * bracket_h1 * tan_d
    
    # 当 h₀ = 2R 时，计算 condition = h₀/tan(d) - h₀/tan(90°-θ)
    condition_value = h0 / tan_d - h0 / tan_90_minus_theta
    
    # 根据条件选择公式：若 condition > T 使用公式1，否则使用公式2
    if isinstance(delta_r, (np.ndarray, pd.Series)) or hasattr(condition_value, '__len__'):
        use_formula1 = condition_value > T
        V = np.zeros_like(delta_r, dtype=float)
        
        # 公式1: V = V₀ - V₁
        if np.any(use_formula1):
            V0 = np.pi * (delta_r / tan_d + T / 2) * h0 * (H - h0 / 3)
            V1 = np.pi * (delta_r / tan_d - T / 2) * h1 * (H - h1 / 3)
            V[use_formula1] = (V0 - V1)[use_formula1]
        
        # 公式2: V = π·condition·h₀·[H - h₀/3]
        if np.any(~use_formula1):
            V[~use_formula1] = (np.pi * condition_value * h0 * (H - h0 / 3))[~use_formula1]
    else:
        # 处理标量情况
        condition_scalar = condition_value.item() if hasattr(condition_value, 'item') else condition_value
        if condition_scalar > T:
            # 公式1: V = V₀ - V₁
            V0 = np.pi * (delta_r / tan_d + T / 2) * h0 * (H - h0 / 3)
            V1 = np.pi * (delta_r / tan_d - T / 2) * h1 * (H - h1 / 3)
            V = V0 - V1
        else:
            # 公式2: V = π·condition·h₀·[H - h₀/3]
            V = np.pi * condition_value * h0 * (H - h0 / 3)
    
    return V


def ensure_monotonic(values):
    """确保数组单调递增（累积最大值）"""
    result = np.copy(values)
    for i in range(1, len(result)):
        if result[i] < result[i-1]:
            result[i] = result[i-1]
    return result


def piecewise_linear_transform(all_x, model_y, true_x, true_y):
    """
    分段归一化重映射（橡皮筋理论）
    
    参数:
        all_x: 所有x坐标（环号），已排序，稠密序列
        model_y: 模型拟合值（My），与all_x对应
        true_x: 真实值的x坐标（Tx），包含原点和测量点，共11个点
        true_y: 真实值的y坐标（Ty），包含原点和测量点，共11个点
    
    返回:
        corrected_y: 修正后的y值
        ratio_values: 每个点的归一化比例 R(x)
    """
    # Step 1: 确保真实值单调递增
    true_y = ensure_monotonic(np.array(true_y))
    true_x = np.array(true_x)
    
    # 初始化修正后的数组和归一化比例数组
    corrected_y = np.zeros_like(model_y, dtype=float)
    ratio_values = np.zeros_like(model_y, dtype=float)
    
    # Step 2: 对每个区间进行处理（共10个区间）
    for i in range(len(true_x) - 1):
        # 区间端点
        tx_i = true_x[i]
        tx_i_plus_1 = true_x[i + 1]
        ty_i = true_y[i]
        ty_i_plus_1 = true_y[i + 1]
        
        # 找到拟合值在真实值点 Tx[i] 和 Tx[i+1] 处的索引
        # 真实值视为对应环最后一个记录点
        idx_i = np.where(all_x == tx_i)[0]
        idx_i_plus_1 = np.where(all_x == tx_i_plus_1)[0]
        
        # 特殊处理：如果 tx_i 不在 all_x 中（比如原点0不在数据中）
        if len(idx_i) == 0:
            # 使用区间内第一个实际存在的点作为起点
            mask_in_range = (all_x >= tx_i) & (all_x <= tx_i_plus_1)
            if not np.any(mask_in_range):
                continue
            idx_i = np.where(mask_in_range)[0][0]  # 区间内第一个点
            ystart = model_y[idx_i]
        else:
            idx_i = idx_i[-1]  # 取该环的最后一个点
            ystart = model_y[idx_i]
        
        if len(idx_i_plus_1) == 0:
            # 如果找不到终点，跳过
            continue
        
        idx_i_plus_1 = idx_i_plus_1[-1]  # 取该环的最后一个点
        yend = model_y[idx_i_plus_1]  # My[i+1]
        
        # 找到当前区间内的所有拟合点（包含端点）
        mask = (all_x >= tx_i) & (all_x <= tx_i_plus_1)
        segment_indices = np.where(mask)[0]
        
        if len(segment_indices) == 0:
            continue
        
        # 对区间内每个拟合点计算归一化比例
        for idx in segment_indices:
            my_x = model_y[idx]  # My(x)
            
            # 计算归一化比例：R(x) = [My(x) - ystart] / [yend - ystart]
            delta_model = yend - ystart
            
            if abs(delta_model) < 1e-10:
                # 如果拟合值在该区间几乎没有变化，使用线性插值
                if tx_i_plus_1 != tx_i:
                    R = (all_x[idx] - tx_i) / (tx_i_plus_1 - tx_i)
                else:
                    R = 0
            else:
                R = (my_x - ystart) / delta_model
            
            # 保存归一化比例
            ratio_values[idx] = R
            
            # 应用归一化趋势到真实值：Y = Ty[i] + R(x) * (Ty[i+1] - Ty[i])
            corrected_y[idx] = ty_i + R * (ty_i_plus_1 - ty_i)
    
    # Step 3: 处理边界外的点
    
    # 第一个真实点之前的点
    first_mask = all_x < true_x[0]
    if np.any(first_mask):
        # 使用第一段的趋势外推
        idx_0 = np.where(all_x == true_x[0])[0]
        idx_1 = np.where(all_x == true_x[1])[0]
        
        if len(idx_0) > 0 and len(idx_1) > 0:
            idx_0 = idx_0[-1]
            idx_1 = idx_1[-1]
            ystart = model_y[idx_0]
            yend = model_y[idx_1]
            delta_model = yend - ystart
            delta_true = true_y[1] - true_y[0]
            
            if abs(delta_model) > 1e-10:
                for idx in np.where(first_mask)[0]:
                    R = (model_y[idx] - ystart) / delta_model
                    ratio_values[idx] = R
                    corrected_y[idx] = true_y[0] + R * delta_true
            else:
                corrected_y[first_mask] = true_y[0]
                ratio_values[first_mask] = 0
        else:
            corrected_y[first_mask] = true_y[0]
            ratio_values[first_mask] = 0
    
    # 最后一个真实点之后的点
    last_mask = all_x > true_x[-1]
    if np.any(last_mask):
        # 使用最后一段的趋势外推
        idx_n_minus_1 = np.where(all_x == true_x[-2])[0]
        idx_n = np.where(all_x == true_x[-1])[0]
        
        if len(idx_n_minus_1) > 0 and len(idx_n) > 0:
            idx_n_minus_1 = idx_n_minus_1[-1]
            idx_n = idx_n[-1]
            ystart = model_y[idx_n_minus_1]
            yend = model_y[idx_n]
            delta_model = yend - ystart
            delta_true = true_y[-1] - true_y[-2]
            
            if abs(delta_model) > 1e-10:
                for idx in np.where(last_mask)[0]:
                    R = (model_y[idx] - yend) / delta_model
                    ratio_values[idx] = R
                    corrected_y[idx] = true_y[-1] + R * delta_true
            else:
                corrected_y[last_mask] = true_y[-1]
                ratio_values[last_mask] = 1.0
        else:
            corrected_y[last_mask] = true_y[-1]
            ratio_values[last_mask] = 1.0
    
    # Step 4: 确保非负
    corrected_y = np.maximum(0, corrected_y)
    
    # Step 5: 最终单调性保证（cumulative max）
    corrected_y = ensure_monotonic(corrected_y)
    
    return corrected_y, ratio_values


def correct_wear_by_piecewise_transform():
    """使用分段归一化重映射法修正磨损拟合值（橡皮筋理论）"""
    
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # 加载数据
    true_wear_path = project_root / "data" / "processed" / "基于刀具位置的磨损量.csv"
    interpolated_path = project_root / "data" / "processed" / "wear_per_timestep.csv"
    
    print("--- Step 1: Loading data ---")
    true_wear_df = pd.read_csv(true_wear_path, index_col=0)
    interp_df = pd.read_csv(interpolated_path, low_memory=False)
    
    # 处理真实数据的索引和列名
    true_wear_df.index = true_wear_df.index.str.replace('#', '').astype(int)
    true_wear_df.columns = true_wear_df.columns.str.replace('R', '').astype(int)
    measured_rings = sorted(true_wear_df.columns.tolist())  # 确保排序
    cutter_ids = true_wear_df.index.tolist()
    
    # 将真实磨损半径转换为体积
    # 1-30号刀具使用原始公式，31-44号刀具使用新公式
    print("Converting wear radius to volume...")
    true_wear_volume_df = pd.DataFrame(index=true_wear_df.index, columns=true_wear_df.columns)
    
    for cid in cutter_ids:
        if cid <= 30:
            # 1-30号刀具使用原始公式
            true_wear_volume_df.loc[cid] = radius_to_volume(true_wear_df.loc[cid])
        elif cid <= 42:
            # 31-42号刀具使用新公式
            true_wear_volume_df.loc[cid] = radius_to_volume_new(true_wear_df.loc[cid], cutter_id=cid)
        elif cid == 43:
            # 43号刀具使用原始公式
            true_wear_volume_df.loc[cid] = radius_to_volume(true_wear_df.loc[cid])
        else:
            # 44号刀具磨损量和43号刀具一致，使用原始公式
            true_wear_volume_df.loc[cid] = radius_to_volume(true_wear_df.loc[43])
    
    print(f"Measured rings: {measured_rings}")
    print(f"Number of cutters: {len(cutter_ids)}")
    print(f"  Cutters 1-30: Using original formula")
    print(f"  Cutters 31-42: Using new formula")
    print(f"  Cutter 43: Using original formula")
    print(f"  Cutter 44: Using cutter 43's wear values (original formula)")
    
    # 获取插值数据中每个环号的最大磨损体积
    wear_cols = [f'cutter_{cid}_wear_volume' for cid in cutter_ids]
    interp_wear_per_ring = interp_df.groupby('ring_number')[wear_cols].max()
    all_rings = np.array(sorted(interp_wear_per_ring.index.tolist()))
    
    print(f"Ring range in interpolated data: {min(all_rings)} - {max(all_rings)}")

    # --- Step 2: 分段归一化重映射修正 ---
    print("\n--- Step 2: Applying piecewise linear transform (rubber band theory) ---")
    
    corrected_wear_per_ring = pd.DataFrame(index=all_rings, columns=cutter_ids)
    ratio_per_ring = pd.DataFrame(index=all_rings, columns=cutter_ids)
    
    for cid in cutter_ids:
        col_name = f'cutter_{cid}_wear_volume'
        
        # 获取该刀具的模型拟合值（蓝色曲线）
        model_values = interp_wear_per_ring.loc[all_rings, col_name].values
        
        # 获取真实值（红色点），包含原点(0,0)
        valid_measured_rings = [0]  # 添加原点
        true_values = [0.0]  # 原点磨损量为0
        
        for ring in measured_rings:
            if ring in interp_wear_per_ring.index:
                valid_measured_rings.append(ring)
                true_values.append(true_wear_volume_df.loc[cid, ring])
        
        # 现在有11个点（原点 + 10个测量点）
        if len(valid_measured_rings) >= 2:
            # 应用分段归一化重映射
            corrected_values, ratio_values = piecewise_linear_transform(
                all_x=all_rings,
                model_y=model_values,
                true_x=np.array(valid_measured_rings),
                true_y=np.array(true_values)
            )
            corrected_wear_per_ring[cid] = corrected_values
            ratio_per_ring[cid] = ratio_values
        else:
            # 如果测量点不足，直接使用原始插值值
            corrected_wear_per_ring[cid] = model_values
            ratio_per_ring[cid] = np.zeros_like(model_values)
    
    print(f"Correction completed for {len(cutter_ids)} cutters")

    # --- Step 3: 将修正后的值写回原始数据 ---
    print("\n--- Step 3: Updating original data with corrected values ---")
    
    # 创建修正后的磨损列
    for cid in cutter_ids:
        corrected_col_name = f'cutter_{cid}_wear_volume_corrected'
        
        # 为每一行数据根据其ring_number查找修正后的值
        def get_corrected_value(ring):
            if ring in corrected_wear_per_ring.index:
                return corrected_wear_per_ring.loc[ring, cid]
            return np.nan
        
        interp_df[corrected_col_name] = interp_df['ring_number'].apply(get_corrected_value)
    
    # 保存修正后的数据
    output_path = project_root / "data" / "processed" / "wear_per_timestep_corrected.csv"
    interp_df.to_csv(output_path, index=False)
    print(f"Saved corrected data to: {output_path}")
    
    # --- Step 4: 绘制对比图 ---
    print("\n--- Step 4: Generating comparison plots ---")
    
    output_dir = project_root / "results" / "wear_correction_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_cutters = len(cutter_ids)
    n_cols = 6
    n_rows = (n_cutters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, cid in enumerate(cutter_ids):
        ax = axes[idx]
        col_name = f'cutter_{cid}_wear_volume'
        
        # 真实值（红色圆点），包含原点
        true_rings_with_origin = [0] + measured_rings
        true_values_with_origin = [0.0] + [true_wear_volume_df.loc[cid, ring] for ring in measured_rings]
        ax.plot(true_rings_with_origin, true_values_with_origin, 'ro', label='True (11 points)', markersize=8)
        
        # 原始插值值（蓝色虚线）
        ax.plot(all_rings, interp_wear_per_ring.loc[all_rings, col_name].values, 
                'b--', label='Original Interp', linewidth=1, alpha=0.5)
        
        # 修正后的值（绿色实线）
        ax.plot(all_rings, corrected_wear_per_ring[cid].values, 
                'g-', label='Corrected', linewidth=1.5)
        
        ax.set_title(f'Cutter #{cid}', fontsize=12)
        ax.set_xlabel('Ring')
        ax.set_ylabel('Wear Volume (mm³)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(n_cutters, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Wear Correction by Piecewise Linear Transform (Rubber Band Theory)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "all_cutters_corrected.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'all_cutters_corrected.png'}")
    
    # --- 绘制缩放因子图（展示橡皮筋拉伸/压缩效果） ---
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(24, 4 * n_rows))
    axes2 = axes2.flatten()
    
    for idx, cid in enumerate(cutter_ids):
        ax = axes2[idx]
        col_name = f'cutter_{cid}_wear_volume'
        
        # 计算每个区间的缩放因子
        model_values = interp_wear_per_ring.loc[all_rings, col_name].values
        
        # 绘制原始模型值和修正后的值的比值
        with np.errstate(divide='ignore', invalid='ignore'):
            scale_factor = corrected_wear_per_ring[cid].values / model_values
            scale_factor = np.where(np.isfinite(scale_factor), scale_factor, 1.0)
        
        ax.plot(all_rings, scale_factor, 'purple', linewidth=1.5, label='Scale Factor')
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # 标记测量点位置
        for ring in measured_rings:
            if ring in all_rings:
                ax.axvline(x=ring, color='red', linestyle=':', linewidth=0.5, alpha=0.5)
        
        ax.set_title(f'Cutter #{cid}', fontsize=12)
        ax.set_xlabel('Ring')
        ax.set_ylabel('Scale Factor (Corrected/Original)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_cutters, len(axes2)):
        axes2[idx].set_visible(False)
    
    plt.suptitle('Piecewise Scale Factors (Rubber Band Stretch/Compress)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "all_cutters_scale_factors.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'all_cutters_scale_factors.png'}")

    # --- 绘制归一化比例图 R(x) ---
    print("\n--- Generating normalization ratio plots ---")
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(24, 4 * n_rows))
    axes3 = axes3.flatten()
    
    for idx, cid in enumerate(cutter_ids):
        ax = axes3[idx]
        
        # 绘制归一化比例 R(x)
        ax.plot(all_rings, ratio_per_ring[cid].values, 'orange', linewidth=1.5, label='R(x)')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='R=0')
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='R=1')
        
        # 标记测量点位置（区间边界）
        true_rings_with_origin = [0] + measured_rings
        for ring in true_rings_with_origin:
            if ring in all_rings:
                ax.axvline(x=ring, color='red', linestyle=':', linewidth=0.8, alpha=0.6)
        
        ax.set_title(f'Cutter #{cid}', fontsize=12)
        ax.set_xlabel('Ring')
        ax.set_ylabel('Normalization Ratio R(x)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
    
    for idx in range(n_cutters, len(axes3)):
        axes3[idx].set_visible(False)
    
    plt.suptitle('Normalization Ratio R(x) = [My(x) - ystart] / [yend - ystart]', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "all_cutters_normalization_ratios.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'all_cutters_normalization_ratios.png'}")

    # --- Step 5: 验证修正效果 ---
    print("\n--- Step 5: Verifying correction results ---")
    
    print("\nVerification at measured points (should be ~0):")
    max_error = 0
    for cid in cutter_ids[:5]:  # 只显示前5个刀具
        errors = []
        # 验证原点
        if 0 in corrected_wear_per_ring.index:
            true_val = 0.0
            corrected_val = corrected_wear_per_ring.loc[0, cid]
            error = abs(corrected_val - true_val)
            errors.append(error)
            max_error = max(max_error, error)
        
        # 验证测量点
        for ring in measured_rings:
            if ring in corrected_wear_per_ring.index:
                true_val = true_wear_volume_df.loc[cid, ring]
                corrected_val = corrected_wear_per_ring.loc[ring, cid]
                error = abs(corrected_val - true_val)
                errors.append(error)
                max_error = max(max_error, error)
        print(f"  Cutter #{cid}: Max error = {max(errors):.4f} mm³ (11 points)")
    
    print(f"\nOverall max error at measured points: {max_error:.4f} mm³")
    print("\nCorrection completed successfully!")

if __name__ == "__main__":
    correct_wear_by_piecewise_transform()
