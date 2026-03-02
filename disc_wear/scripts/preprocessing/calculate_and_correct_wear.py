import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

try:
    from scipy.optimize import fsolve, brentq
    from scipy.interpolate import make_interp_spline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy is not available. Some features will be limited.")

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
    """将磨损半径转换为磨损体积（新公式，用于31-42号刀具）"""
    theta_rad = np.radians(theta_deg)
    tan_theta = np.tan(theta_rad)
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)
    
    d_deg = (cutter_id - 30) * 70 / 13
    d_rad = np.radians(d_deg)
    tan_d = np.tan(d_rad)
    sin_d = np.sin(d_rad)
    
    angle_90_minus_theta = np.radians(90) - theta_rad
    tan_90_minus_theta = np.tan(angle_90_minus_theta)
    
    numerator_h0 = (T / 2 + delta_r / tan_d) * tan_90_minus_theta
    denominator_h0 = tan_90_minus_theta - tan_d
    
    epsilon = 1e-6
    if isinstance(denominator_h0, np.ndarray):
        is_singular = np.abs(denominator_h0) < epsilon
        h0 = np.where(is_singular, 2 * delta_r, numerator_h0 / np.where(is_singular, epsilon, denominator_h0))
    else:
        if abs(denominator_h0) < epsilon:
            h0 = 2 * delta_r
        else:
            h0 = numerator_h0 / denominator_h0
    
    angle_90_plus_theta_minus_d = np.radians(90) + theta_rad - d_rad
    sin_angle_h1 = np.sin(angle_90_plus_theta_minus_d)
    
    if isinstance(sin_angle_h1, np.ndarray):
        sin_angle_h1 = np.where(np.abs(sin_angle_h1) < epsilon, epsilon, sin_angle_h1)
    else:
        if abs(sin_angle_h1) < epsilon:
            sin_angle_h1 = epsilon
    
    bracket_h1 = 1 - (sin_d * cos_theta) / sin_angle_h1
    h1 = (delta_r / tan_d - T / 2) * bracket_h1 * tan_d
    
    condition_value = h0 / tan_d - h0 / tan_90_minus_theta
    
    if isinstance(delta_r, (np.ndarray, pd.Series)) or hasattr(condition_value, '__len__'):
        use_formula1 = condition_value > T
        V = np.zeros_like(delta_r, dtype=float)
        
        if np.any(use_formula1):
            V0 = np.pi * (delta_r / tan_d + T / 2) * h0 * (H - h0 / 3)
            V1 = np.pi * (delta_r / tan_d - T / 2) * h1 * (H - h1 / 3)
            V[use_formula1] = (V0 - V1)[use_formula1]
        
        if np.any(~use_formula1):
            V[~use_formula1] = (np.pi * condition_value * h0 * (H - h0 / 3))[~use_formula1]
    else:
        condition_scalar = condition_value.item() if hasattr(condition_value, 'item') else condition_value
        if condition_scalar > T:
            V0 = np.pi * (delta_r / tan_d + T / 2) * h0 * (H - h0 / 3)
            V1 = np.pi * (delta_r / tan_d - T / 2) * h1 * (H - h1 / 3)
            V = V0 - V1
        else:
            V = np.pi * condition_value * h0 * (H - h0 / 3)
    
    return V

def volume_to_radius(volume, T=25.4, R=241.3, theta_deg=20, N=1):
    """将磨损体积转换回磨损半径（原始公式）"""
    theta_rad = np.radians(theta_deg)
    tan_theta = np.tan(theta_rad)
    
    if isinstance(volume, np.ndarray):
        delta_r = np.zeros_like(volume, dtype=float)
        for i in range(len(volume)):
            v = volume[i]
            a = -(2/3) * tan_theta
            b = R * tan_theta - T/2
            c = T * R
            d = -v / (2 * np.pi * N)
            
            coefficients = [a, b, c, d]
            roots = np.roots(coefficients)
            
            real_roots = roots[np.abs(np.imag(roots)) < 1e-10]
            positive_roots = real_roots[np.real(real_roots) > 0]
            if len(positive_roots) > 0:
                delta_r[i] = np.real(np.min(positive_roots))
            else:
                delta_r[i] = 0
    else:
        a = -(2/3) * tan_theta
        b = R * tan_theta - T/2
        c = T * R
        d = -volume / (2 * np.pi * N)
        
        coefficients = [a, b, c, d]
        roots = np.roots(coefficients)
        
        real_roots = roots[np.abs(np.imag(roots)) < 1e-10]
        positive_roots = real_roots[np.real(real_roots) > 0]
        if len(positive_roots) > 0:
            delta_r = np.real(np.min(positive_roots))
        else:
            delta_r = 0
    
    return np.maximum(0, delta_r)

def volume_to_radius_new(volume, cutter_id, T=25.4, H=241.3, theta_deg=20, initial_guess=None):
    """将磨损体积转换回磨损半径（新公式，用于31-42号刀具）"""
    if not SCIPY_AVAILABLE:
        print("Warning: scipy not available, using approximation")
        return (volume / (2 * np.pi * T * H)) ** 0.5
    
    def equation(delta_r, target_volume):
        if delta_r < 0:
            return 1e10
        calculated_volume = radius_to_volume_new(delta_r, cutter_id=cutter_id, T=T, H=H, theta_deg=theta_deg)
        return calculated_volume - target_volume
    
    if isinstance(volume, np.ndarray):
        delta_r = np.zeros_like(volume, dtype=float)
        for i in range(len(volume)):
            v = volume[i]
            if v <= 0:
                delta_r[i] = 0
                continue
            
            guess = initial_guess if initial_guess is not None else (v / (2 * np.pi * T * H)) ** 0.5
            
            try:
                solution = fsolve(equation, guess, args=(v,), full_output=True)
                if solution[2] == 1:
                    delta_r[i] = max(0, solution[0][0])
                else:
                    try:
                        delta_r[i] = brentq(equation, 0, 100, args=(v,))
                    except:
                        delta_r[i] = 0
            except:
                delta_r[i] = 0
    else:
        if volume <= 0:
            return 0
        
        guess = initial_guess if initial_guess is not None else (volume / (2 * np.pi * T * H)) ** 0.5
        
        try:
            solution = fsolve(equation, guess, args=(volume,), full_output=True)
            if solution[2] == 1:
                delta_r = max(0, solution[0][0])
            else:
                try:
                    delta_r = brentq(equation, 0, 100, args=(volume,))
                except:
                    delta_r = 0
        except:
            delta_r = 0
    
    return np.maximum(0, delta_r)

def ensure_monotonic(values):
    """确保数组单调递增（累积最大值）"""
    result = np.copy(values)
    for i in range(1, len(result)):
        if result[i] < result[i-1]:
            result[i] = result[i-1]
    return result

def piecewise_linear_transform(all_x, model_y, true_x, true_y):
    """分段归一化重映射（橡皮筋理论）"""
    true_y = ensure_monotonic(np.array(true_y))
    true_x = np.array(true_x)
    
    corrected_y = np.zeros_like(model_y, dtype=float)
    ratio_values = np.zeros_like(model_y, dtype=float)
    
    for i in range(len(true_x) - 1):
        tx_i = true_x[i]
        tx_i_plus_1 = true_x[i + 1]
        ty_i = true_y[i]
        ty_i_plus_1 = true_y[i + 1]
        
        idx_i = np.where(all_x == tx_i)[0]
        idx_i_plus_1 = np.where(all_x == tx_i_plus_1)[0]
        
        if len(idx_i) == 0:
            mask_in_range = (all_x >= tx_i) & (all_x <= tx_i_plus_1)
            if not np.any(mask_in_range):
                continue
            idx_i = np.where(mask_in_range)[0][0]
            ystart = model_y[idx_i]
        else:
            idx_i = idx_i[-1]
            ystart = model_y[idx_i]
        
        if len(idx_i_plus_1) == 0:
            continue
        
        idx_i_plus_1 = idx_i_plus_1[-1]
        yend = model_y[idx_i_plus_1]
        
        mask = (all_x >= tx_i) & (all_x <= tx_i_plus_1)
        segment_indices = np.where(mask)[0]
        
        if len(segment_indices) == 0:
            continue
        
        for idx in segment_indices:
            my_x = model_y[idx]
            
            delta_model = yend - ystart
            
            if abs(delta_model) < 1e-10:
                if tx_i_plus_1 != tx_i:
                    R = (all_x[idx] - tx_i) / (tx_i_plus_1 - tx_i)
                else:
                    R = 0
            else:
                R = (my_x - ystart) / delta_model
            
            ratio_values[idx] = R
            corrected_y[idx] = ty_i + R * (ty_i_plus_1 - ty_i)
    
    # 处理边界外的点
    first_mask = all_x < true_x[0]
    if np.any(first_mask):
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
    
    last_mask = all_x > true_x[-1]
    if np.any(last_mask):
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
    
    corrected_y = np.maximum(0, corrected_y)
    corrected_y = ensure_monotonic(corrected_y)
    
    return corrected_y, ratio_values

class Total_wear:
    def __init__(self):
        self.wear = {
            'R149': 168, 'R195': 217, 'R227': 263, 'R275': 305, 'R367': 410,
            'R488': 556, 'R654': 843.5, 'R708': 959, 'R759': 1051, 'R854': 1123
        }
    def get_data_pairs(self):
        rings = [int(k.replace('R', '')) for k in self.wear.keys()]
        wears = list(self.wear.values())
        return rings, wears

def calculate_and_correct_wear():
    """计算磨损并使用分段归一化重映射法修正"""
    
    parser = argparse.ArgumentParser(description="Calculate and correct per-timestep cutter wear.")
    
    project_root = Path(__file__).resolve().parent.parent.parent
    
    parser.add_argument('--wear_data', type=str,
        default=str(project_root / "data" / "processed" / "基于刀具位置的磨损量.csv"))
    parser.add_argument('--friction_data', type=str,
        default=str(project_root / "data" / "processed" / "tbm_data_with_friction_energy.csv"))
    parser.add_argument('--output_csv', type=str,
        default=str(project_root / "data" / "processed" / "wear_per_timestep_final.csv"))
    
    args = parser.parse_args()

    # ===== Step 1: 加载数据 =====
    print("--- Step 1: Loading data ---")
    try:
        wear_df = pd.read_csv(args.wear_data, index_col=0)
        friction_df = pd.read_csv(args.friction_data, low_memory=False)
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # ===== Step 2: 准备数据 =====
    print("--- Step 2: Preparing data ---")
    
    wear_df.index = wear_df.index.str.replace('#', '').astype(int)
    wear_df.columns = wear_df.columns.str.replace('R', '').astype(int)
    measured_rings = wear_df.columns.tolist()
    cutter_ids = wear_df.index.tolist()

    friction_col = 'global_cumulative_friction_energy'
    if friction_col not in friction_df.columns:
        print(f"Error: Friction column '{friction_col}' not found.")
        return

    friction_at_measured_rings = friction_df.groupby('ring_number')[friction_col].max()
    aligned_friction = friction_at_measured_rings.reindex(measured_rings).dropna()
    
    print(f"Aligned friction work for {len(aligned_friction)} measured rings.")
    print(f"Number of cutters: {len(cutter_ids)}")

    # ===== Step 3: 将真实磨损半径转换为体积 =====
    print("--- Step 3: Converting measured wear radius to volume ---")
    
    true_wear_volume_df = pd.DataFrame(index=wear_df.index, columns=wear_df.columns)
    
    for cid in cutter_ids:
        if cid <= 30:
            true_wear_volume_df.loc[cid] = radius_to_volume(wear_df.loc[cid])
        elif cid <= 42:
            true_wear_volume_df.loc[cid] = radius_to_volume_new(wear_df.loc[cid], cutter_id=cid)
        elif cid == 43:
            true_wear_volume_df.loc[cid] = radius_to_volume(wear_df.loc[cid])
        else:
            true_wear_volume_df.loc[cid] = radius_to_volume(wear_df.loc[43])
    
    print("Conversion completed.")

    # ===== Step 4: 拟合k值（基于体积） =====
    print("--- Step 4: Fitting proportionality constant 'k' (volume-based) ---")
    
    cutter_k_values = {}
    
    for cutter_id in wear_df.index:
        wear_radius_values = wear_df.loc[cutter_id, aligned_friction.index]
        friction_values = aligned_friction.values
        
        valid_indices = ~np.isnan(wear_radius_values.values)
        wear_radius_points = wear_radius_values.values[valid_indices]
        friction_points = friction_values[valid_indices]

        if cutter_id <= 30:
            wear_volume_points = radius_to_volume(wear_radius_points)
        elif cutter_id <= 42:
            wear_volume_points = radius_to_volume_new(wear_radius_points, cutter_id=cutter_id)
        elif cutter_id == 43:
            wear_volume_points = radius_to_volume(wear_radius_points)
        else:
            wear_radius_43 = wear_df.loc[43, aligned_friction.index]
            wear_radius_points_43 = wear_radius_43.values[valid_indices]
            wear_volume_points = radius_to_volume(wear_radius_points_43)

        if len(wear_volume_points) > 0 and np.sum(friction_points**2) > 0:
            k = np.sum(friction_points * wear_volume_points) / np.sum(friction_points**2)
            cutter_k_values[cutter_id] = k
        else:
            cutter_k_values[cutter_id] = 0

    print(f"Calculated 'k' for {len(cutter_k_values)} cutters.")

    # ===== Step 5: 生成每个时间步的磨损体积（插值） =====
    print("--- Step 5: Generating wear volume for every timestep ---")
    
    for cutter_id, k in cutter_k_values.items():
        wear_volume_col_name = f'cutter_{cutter_id}_wear_volume'
        friction_df[wear_volume_col_name] = k * friction_df[friction_col]
        friction_df[wear_volume_col_name] = friction_df[wear_volume_col_name].cummax()

    print("Generated per-timestep wear volume columns.")

    # ===== Step 6: 修正磨损体积（分段归一化重映射） =====
    print("--- Step 6: Correcting wear volume using piecewise linear transform ---")
    
    wear_cols_volume = [f'cutter_{cid}_wear_volume' for cid in cutter_ids]
    interp_wear_per_ring = friction_df.groupby('ring_number')[wear_cols_volume].max()
    all_rings = np.array(sorted(interp_wear_per_ring.index.tolist()))
    
    corrected_wear_per_ring = pd.DataFrame(index=all_rings, columns=cutter_ids)
    
    for cid in cutter_ids:
        col_name = f'cutter_{cid}_wear_volume'
        model_values = interp_wear_per_ring.loc[all_rings, col_name].values
        
        valid_measured_rings = [0]
        true_values = [0.0]
        
        for ring in measured_rings:
            if ring in interp_wear_per_ring.index:
                valid_measured_rings.append(ring)
                true_values.append(true_wear_volume_df.loc[cid, ring])
        
        if len(valid_measured_rings) >= 2:
            corrected_values, _ = piecewise_linear_transform(
                all_x=all_rings,
                model_y=model_values,
                true_x=np.array(valid_measured_rings),
                true_y=np.array(true_values)
            )
            corrected_wear_per_ring[cid] = corrected_values
        else:
            corrected_wear_per_ring[cid] = model_values
    
    print(f"Correction completed for {len(cutter_ids)} cutters")

    # ===== Step 7: 将修正后的体积写回原始数据 =====
    print("--- Step 7: Updating data with corrected volume ---")
    
    # 创建环号到修正体积的映射字典，避免DataFrame碎片化
    corrected_volume_columns = {}
    
    for cid in cutter_ids:
        corrected_col_name = f'cutter_{cid}_wear_volume_corrected'
        
        # 使用向量化操作而非apply
        ring_to_value = corrected_wear_per_ring[cid].to_dict()
        corrected_volume_columns[corrected_col_name] = friction_df['ring_number'].map(ring_to_value)
    
    # 一次性添加所有修正后的体积列
    corrected_volume_df = pd.DataFrame(corrected_volume_columns, index=friction_df.index)
    friction_df = pd.concat([friction_df, corrected_volume_df], axis=1)
    
    print("Corrected volume columns added.")

    # ===== Step 8: 将修正后的体积转换为半径 =====
    print("--- Step 8: Converting corrected volume to radius ---")
    print("This may take some time...")
    
    import time
    start_time = time.time()
    
    # 收集所有半径列，避免DataFrame碎片化
    radius_columns = {}
    
    for idx, cid in enumerate(cutter_ids, 1):
        print(f"Processing cutter {cid} ({idx}/{len(cutter_ids)})...", end=' ')
        cutter_start = time.time()
        
        corrected_volume_col = f'cutter_{cid}_wear_volume_corrected'
        corrected_radius_col = f'cutter_{cid}_wear_radius_corrected'
        
        if cid <= 30:
            radius_columns[corrected_radius_col] = volume_to_radius(friction_df[corrected_volume_col].values)
        elif cid <= 42:
            radius_columns[corrected_radius_col] = volume_to_radius_new(
                friction_df[corrected_volume_col].values, 
                cutter_id=cid
            )
        elif cid == 43:
            radius_columns[corrected_radius_col] = volume_to_radius(friction_df[corrected_volume_col].values)
        else:
            radius_columns[corrected_radius_col] = volume_to_radius(friction_df[corrected_volume_col].values)
        
        cutter_elapsed = time.time() - cutter_start
        print(f"Done in {cutter_elapsed:.1f}s")
    
    # 一次性添加所有半径列
    radius_df = pd.DataFrame(radius_columns, index=friction_df.index)
    friction_df = pd.concat([friction_df, radius_df], axis=1)
    
    total_elapsed = time.time() - start_time
    print(f"Converted all volumes to radius in {total_elapsed/60:.1f} minutes.")

    # ===== Step 9: 保存结果 =====
    print("--- Step 9: Saving results ---")
    
    # 保存主表（包含体积和半径）
    friction_df.to_csv(args.output_csv, index=False)
    print(f"Saved main table to: {args.output_csv}")
    
    # 为每把刀单独输出体积和半径表
    output_dir = project_root / "data" / "processed" / "individual_cutters"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for cid in cutter_ids:
        volume_col = f'cutter_{cid}_wear_volume_corrected'
        radius_col = f'cutter_{cid}_wear_radius_corrected'
        
        cutter_df = friction_df[['ring_number', volume_col, radius_col]].copy()
        cutter_df.columns = ['ring_number', 'wear_volume_mm3', 'wear_radius_mm']
        
        output_path = output_dir / f"cutter_{cid:02d}_wear.csv"
        cutter_df.to_csv(output_path, index=False)
    
    print(f"Saved individual cutter tables to: {output_dir}")
    
    # 汇总表：每个环号下44把刀的磨损体积
    volume_summary = friction_df.groupby('ring_number')[[f'cutter_{cid}_wear_volume_corrected' for cid in cutter_ids]].max()
    volume_summary.columns = [f'cutter_{cid}' for cid in cutter_ids]
    volume_summary_path = project_root / "data" / "processed" / "all_cutters_volume_summary.csv"
    volume_summary.to_csv(volume_summary_path)
    print(f"Saved volume summary to: {volume_summary_path}")
    
    # 汇总表：每个环号下44把刀的磨损半径
    radius_summary = friction_df.groupby('ring_number')[[f'cutter_{cid}_wear_radius_corrected' for cid in cutter_ids]].max()
    radius_summary.columns = [f'cutter_{cid}' for cid in cutter_ids]
    radius_summary_path = project_root / "data" / "processed" / "all_cutters_radius_summary.csv"
    radius_summary.to_csv(radius_summary_path)
    print(f"Saved radius summary to: {radius_summary_path}")

    print("\n=== Processing completed successfully! ===")

if __name__ == "__main__":
    calculate_and_correct_wear()
