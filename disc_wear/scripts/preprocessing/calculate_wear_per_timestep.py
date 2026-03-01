import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def radius_to_volume(delta_r, T=25.4, R=241.3, theta_deg=20, N=1):
    """
    Convert wear radius to wear volume (Original formula for cutters 1-30).
    
    Formula: V = 2π * ΔR * [TR - (T/2)ΔR + R*tan(θ)ΔR - (2/3)*tan(θ)ΔR²]
    
    Parameters:
    - delta_r (ΔR): Wear radius change in mm
    - T: Cutter thickness in mm (default: 25.4)
    - R: Cutter position radius in mm (default: 241.3)
    - theta_deg: Cutter angle in degrees (default: 20)
    - N: Number of cutters (default: 1)
    
    Returns:
    - volume: Wear volume in mm³
    """
    theta_rad = np.radians(theta_deg)
    tan_theta = np.tan(theta_rad)
    
    # V = 2π * ΔR * [TR - (T/2)ΔR + R*tan(θ)ΔR - (2/3)*tan(θ)ΔR²]
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

def volume_to_radius(volume, T=25.4, R=241.3, theta_deg=20, N=1):
    """
    Convert wear volume back to wear radius by solving the cubic equation.
    
    Formula: V = 2π * ΔR * [TR - (T/2)ΔR + R*tan(θ)ΔR - (2/3)*tan(θ)ΔR²]
    
    This is a cubic equation in ΔR:
    -(2/3)*tan(θ)*ΔR³ + [R*tan(θ) - T/2]*ΔR² + TR*ΔR - V/(2πN) = 0
    
    We use numpy's polynomial solver for this.
    
    Parameters:
    - volume: Wear volume in mm³
    - T: Cutter thickness in mm (default: 25.4)
    - R: Cutter position radius in mm (default: 241.3)
    - theta_deg: Cutter angle in degrees (default: 20)
    - N: Number of cutters (default: 1)
    
    Returns:
    - delta_r: Wear radius change in mm
    """
    theta_rad = np.radians(theta_deg)
    tan_theta = np.tan(theta_rad)
    
    # Original formula: V = 2π * ΔR * [TR - (T/2)ΔR + R*tan(θ)ΔR - (2/3)*tan(θ)ΔR²] * N
    # Expand: V = 2πN * [TR*ΔR - (T/2)ΔR² + R*tan(θ)ΔR² - (2/3)*tan(θ)ΔR³]
    # Rearrange to: -(2/3)*tan(θ)*ΔR³ + (R*tan(θ) - T/2)*ΔR² + TR*ΔR - V/(2πN) = 0
    
    if isinstance(volume, np.ndarray):
        delta_r = np.zeros_like(volume, dtype=float)
        for i in range(len(volume)):
            v = volume[i]
            a = -(2/3) * tan_theta
            b = R * tan_theta - T/2
            c = T * R
            d = -v / (2 * np.pi * N)
            
            # Solve cubic equation
            coefficients = [a, b, c, d]
            roots = np.roots(coefficients)
            
            # Find the smallest positive real root (physical solution)
            real_roots = roots[np.abs(np.imag(roots)) < 1e-10]
            positive_roots = real_roots[np.real(real_roots) > 0]
            if len(positive_roots) > 0:
                # Choose the smallest positive root
                delta_r[i] = np.real(np.min(positive_roots))
            else:
                delta_r[i] = 0
    else:
        a = -(2/3) * tan_theta
        b = R * tan_theta - T/2
        c = T * R
        d = -volume / (2 * np.pi * N)
        
        # Solve cubic equation
        coefficients = [a, b, c, d]
        roots = np.roots(coefficients)
        
        # Find the smallest positive real root (physical solution)
        real_roots = roots[np.abs(np.imag(roots)) < 1e-10]
        positive_roots = real_roots[np.real(real_roots) > 0]
        if len(positive_roots) > 0:
            # Choose the smallest positive root
            delta_r = np.real(np.min(positive_roots))
        else:
            delta_r = 0
    
    # ===== 旧公式（备用）- 基于二次方程求解 =====
    # 简化版本: V = 2πR(T + ΔR*tan(θ))ΔR*N
    # 对应二次方程: tan(θ)*ΔR² + T*ΔR - V/(2πRN) = 0
    # 
    # a = tan_theta
    # b = T
    # c = -volume / (2 * np.pi * R * N)
    # 
    # discriminant = b**2 - 4*a*c
    # if isinstance(discriminant, np.ndarray):
    #     delta_r = np.zeros_like(discriminant)
    #     positive_mask = discriminant >= 0
    #     delta_r[positive_mask] = (-b + np.sqrt(discriminant[positive_mask])) / (2 * a)
    # else:
    #     if discriminant < 0:
    #         return 0
    #     delta_r = (-b + np.sqrt(discriminant)) / (2 * a)
    # ==========================================
    
    return np.maximum(0, delta_r)


def volume_to_radius_new(volume, cutter_id, T=25.4, H=2413, theta_deg=20, initial_guess=None):
    """
    Convert wear volume back to wear radius for the new formula (cutters 31-42).
    
    使用数值方法求解反向问题（体积→半径）。
    由于新公式包含条件判断和复杂方程，使用 scipy.optimize 进行数值求根。
    
    Parameters:
    - volume: 磨损体积 (mm³)
    - cutter_id: 刀具编号 (31-42)
    - T: 刀具厚度 (mm), 默认 25.4
    - H: 刀具高度 (mm), 默认 2413
    - theta_deg: 角度 θ (度), 默认 20
    - initial_guess: delta_r 的初始猜测值 (可选)
    
    Returns:
    - delta_r: 磨损半径 (mm)
    """
    from scipy.optimize import fsolve, brentq
    
    def equation(delta_r, target_volume):
        """Equation to solve: radius_to_volume_new(delta_r, cutter_id) - target_volume = 0"""
        if delta_r < 0:
            return 1e10  # Penalty for negative radius
        calculated_volume = radius_to_volume_new(delta_r, cutter_id=cutter_id, T=T, H=H, theta_deg=theta_deg)
        return calculated_volume - target_volume
    
    if isinstance(volume, np.ndarray):
        delta_r = np.zeros_like(volume, dtype=float)
        for i in range(len(volume)):
            v = volume[i]
            if v <= 0:
                delta_r[i] = 0
                continue
            
            # Use initial guess or estimate from original formula
            if initial_guess is not None:
                guess = initial_guess if np.isscalar(initial_guess) else initial_guess[i]
            else:
                # Rough estimate: assume linear relationship for initial guess
                guess = (v / (2 * np.pi * T * H)) ** 0.5
            
            try:
                # Try fsolve first
                solution = fsolve(equation, guess, args=(v,), full_output=True)
                if solution[2] == 1:  # Solution found
                    delta_r[i] = max(0, solution[0][0])
                else:
                    # If fsolve fails, try bounded search
                    try:
                        delta_r[i] = brentq(equation, 0, 100, args=(v,))
                    except:
                        delta_r[i] = 0
            except:
                delta_r[i] = 0
    else:
        if volume <= 0:
            return 0
        
        # Use initial guess or estimate
        if initial_guess is not None:
            guess = initial_guess
        else:
            guess = (volume / (2 * np.pi * T * H)) ** 0.5
        
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


import matplotlib.pyplot as plt
try:
    from scipy.optimize import fsolve, brentq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy is not available. volume_to_radius_new will not work properly.")

try:
    from scipy.interpolate import make_interp_spline
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy is not available. The wear trend curve will be linear.")

class Total_wear:
    def __init__(self):
        # Using total wear of all cutters, not individual ones
        self.wear = {
            'R149': 168, 'R195': 217, 'R227': 263, 'R275': 305, 'R367': 410,
            'R488': 556, 'R654': 843.5, 'R708': 959, 'R759': 1051, 'R854': 1123
        }
    def get_data_pairs(self):
        rings = [int(k.replace('R', '')) for k in self.wear.keys()]
        wears = list(self.wear.values())
        return rings, wears

def calculate_wear_per_timestep():
    """
    Calculates wear for each cutter at every timestep by fitting a proportionality
    constant 'k' against global cumulative friction work, using a volume-based approach.
    """
    # --- 1. Define paths and parameters ---
    parser = argparse.ArgumentParser(description="Calculate per-timestep cutter wear based on friction energy.")
    
    project_root = Path(__file__).resolve().parent.parent.parent
    
    parser.add_argument(
        '--wear_data',
        type=str,
        default=str(project_root / "data" / "processed" / "基于刀具位置的磨损量.csv"),
        help="Path to the CSV with measured wear data per cutter at specific rings."
    )
    parser.add_argument(
        '--friction_data',
        type=str,
        default=str(project_root / "data" / "processed" / "tbm_data_with_friction_energy.csv"),
        help="Path to the CSV with friction energy data, must contain 'ring_number' and 'global_cumulative_friction_energy'."
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=str(project_root / "data" / "processed" / "wear_per_timestep.csv"),
        help="Path to save the output file with per-timestep wear data."
    )
    parser.add_argument(
        '--convert_to_radius',
        action='store_true',
        help="Convert wear volume to radius (WARNING: This is very time-consuming for large datasets, may take 8+ hours)."
    )
    args = parser.parse_args()

    # --- 2. Loading data ---
    print("--- Step 1: Loading data ---")
    try:
        wear_df = pd.read_csv(args.wear_data, index_col=0)
        friction_df = pd.read_csv(args.friction_data, low_memory=False)
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure both input files exist.")
        return

    # --- 3. Prepare and align data ---
    print("--- Step 2: Preparing and aligning data ---")
    
    wear_df.index = wear_df.index.str.replace('#', '').astype(int)
    wear_df.columns = wear_df.columns.str.replace('R', '').astype(int)
    measured_rings = wear_df.columns.tolist()

    friction_col = 'global_cumulative_friction_energy'
    if friction_col not in friction_df.columns:
        print(f"Error: Friction column '{friction_col}' not found in the friction data file.")
        return

    friction_at_measured_rings = friction_df.groupby('ring_number')[friction_col].max()
    aligned_friction = friction_at_measured_rings.reindex(measured_rings).dropna()
    
    print(f"Aligned friction work for {len(aligned_friction)} measured rings.")

    # --- 4. Fit proportionality constant 'k' for each cutter (Volume-based) ---
    print("--- Step 3: Fitting proportionality constant 'k' (基于体积) for each cutter ---")
    print("  刀具 1-30号: 使用原始公式")
    print("  刀具 31-42号: 使用新公式（带条件判断）")
    print("  刀具 43号: 使用原始公式")
    print("  刀具 44号: 磨损量与43号完全一致")
    
    cutter_k_values = {}
    
    for cutter_id in wear_df.index:
        wear_radius_values = wear_df.loc[cutter_id, aligned_friction.index]
        friction_values = aligned_friction.values
        
        valid_indices = ~np.isnan(wear_radius_values.values)
        wear_radius_points = wear_radius_values.values[valid_indices]
        friction_points = friction_values[valid_indices]

        # 根据刀具编号选择合适的公式将半径转换为体积
        if cutter_id <= 30:
            # 1-30号刀具: 使用原始公式
            wear_volume_points = radius_to_volume(wear_radius_points)
        elif cutter_id <= 42:
            # 31-42号刀具: 使用新公式
            wear_volume_points = radius_to_volume_new(wear_radius_points, cutter_id=cutter_id)
        elif cutter_id == 43:
            # 43号刀具: 使用原始公式
            wear_volume_points = radius_to_volume(wear_radius_points)
        else:
            # 44号刀具: 磨损量与43号完全一致，使用43号的数据
            wear_radius_43 = wear_df.loc[43, aligned_friction.index]
            wear_radius_points_43 = wear_radius_43.values[valid_indices]
            wear_volume_points = radius_to_volume(wear_radius_points_43)

        if len(wear_volume_points) > 0 and np.sum(friction_points**2) > 0:
            k = np.sum(friction_points * wear_volume_points) / np.sum(friction_points**2)
            cutter_k_values[cutter_id] = k
        else:
            cutter_k_values[cutter_id] = 0

    print(f"Calculated 'k' for {len(cutter_k_values)} cutters.")
    k_values_df = pd.DataFrame(cutter_k_values.items(), columns=['cutter_id', 'k_value_volume'])
    k_values_df = k_values_df.sort_values(by='cutter_id').set_index('cutter_id')
    k_values_output_path = project_root / "results" / "cutter_k_values_volume.csv"
    k_values_df.to_csv(k_values_output_path)
    print(f"Saved volume-based k-values to {k_values_output_path}")

    # --- 5. Generate wear for every timestep ---
    print("--- Step 4: Generating wear for every timestep ---")
    
    for cutter_id, k in cutter_k_values.items():
        wear_volume_col_name = f'cutter_{cutter_id}_wear_volume'
        friction_df[wear_volume_col_name] = k * friction_df[friction_col]
        friction_df[wear_volume_col_name] = friction_df[wear_volume_col_name].cummax()

    print("Generated per-timestep wear columns (in volume mm³) for all cutters.")
    
    # --- 5.5. Convert volume back to radius for each cutter (OPTIONAL) ---
    if args.convert_to_radius:
        print("--- Step 4.5: Converting wear volume back to radius for each cutter ---")
        print("WARNING: This step is very time-consuming and may take 8+ hours for large datasets!")
        print(f"Dataset size: {len(friction_df)} rows")
        
        import time
        start_time = time.time()
        
        for idx, cutter_id in enumerate(wear_df.index, 1):
            print(f"Processing cutter {cutter_id} ({idx}/{len(wear_df.index)})...", end=' ')
            cutter_start = time.time()
            
            wear_volume_col = f'cutter_{cutter_id}_wear_volume'
            wear_radius_col = f'cutter_{cutter_id}_wear_radius'
            
            # 根据刀具编号选择合适的反向转换公式
            if cutter_id <= 30:
                # 1-30号刀具: 使用原始公式的反向转换
                friction_df[wear_radius_col] = volume_to_radius(friction_df[wear_volume_col].values)
            elif cutter_id <= 42:
                # 31-42号刀具: 使用新公式的反向转换
                friction_df[wear_radius_col] = volume_to_radius_new(
                    friction_df[wear_volume_col].values, 
                    cutter_id=cutter_id
                )
            elif cutter_id == 43:
                # 43号刀具: 使用原始公式的反向转换
                friction_df[wear_radius_col] = volume_to_radius(friction_df[wear_volume_col].values)
            else:
                # 44号刀具: 与43号一致
                friction_df[wear_radius_col] = volume_to_radius(friction_df[wear_volume_col].values)
            
            cutter_elapsed = time.time() - cutter_start
            print(f"Done in {cutter_elapsed:.1f}s")
        
        total_elapsed = time.time() - start_time
        print(f"Converted all wear volumes to radius values in {total_elapsed/60:.1f} minutes.")
    else:
        print("--- Step 4.5: Skipping volume-to-radius conversion (use --convert_to_radius to enable) ---")
        print("Note: Radius columns will not be available in the output CSV and radius plots will be skipped.")

    # --- 6. Visualization ---
    print("--- Step 5: Visualizing results ---")
    
    plot_dir = project_root / "results" / "wear_fitting_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每把刀具单独创建子目录
    individual_fit_dir = plot_dir / "individual_k_fitting"
    individual_growth_dir = plot_dir / "individual_wear_growth"
    individual_fit_dir.mkdir(parents=True, exist_ok=True)
    individual_growth_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving individual k-fitting plots to {individual_fit_dir}")
    print(f"Saving individual wear growth plots to {individual_growth_dir}")

    max_friction = aligned_friction.max()
    wear_cols_volume = [f'cutter_{cid}_wear_volume' for cid in wear_df.index]
    wear_growth_volume_df = friction_df.groupby('ring_number')[wear_cols_volume].max()
    
    # 如果启用了半径转换，准备半径数据
    if args.convert_to_radius:
        individual_radius_dir = plot_dir / "individual_radius_growth"
        individual_radius_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving individual radius growth plots to {individual_radius_dir}")
        
        wear_cols_radius = [f'cutter_{cid}_wear_radius' for cid in wear_df.index]
        wear_growth_radius_df = friction_df.groupby('ring_number')[wear_cols_radius].max()
    
    # 为每把刀具单独绘制k值拟合图和磨损增长图
    for cutter_id in wear_df.index:
        # --- 单独的k值拟合图 (摩擦功 vs 体积) ---
        wear_radius_values = wear_df.loc[cutter_id, aligned_friction.index]
        
        # 根据刀具编号选择合适的公式
        if cutter_id <= 30:
            wear_volume_values = radius_to_volume(wear_radius_values)
        elif cutter_id <= 42:
            wear_volume_values = radius_to_volume_new(wear_radius_values, cutter_id=cutter_id)
        elif cutter_id == 43:
            wear_volume_values = radius_to_volume(wear_radius_values)
        else:
            # 44号刀具: 使用43号的数据
            wear_radius_43 = wear_df.loc[43, aligned_friction.index]
            wear_volume_values = radius_to_volume(wear_radius_43)
        
        k = cutter_k_values.get(cutter_id, 0)
        x_fit = np.linspace(0, max_friction, 100)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(aligned_friction.values, wear_volume_values, color='blue', alpha=0.6, s=80, label='Measured Data')
        plt.plot(x_fit, k * x_fit, color='red', linewidth=2, label=f'Fitted Line (k={k:.3e})')
        plt.title(f'Cutter #{cutter_id}: K-value Fitting (Friction vs Volume)', fontsize=14)
        plt.xlabel('Global Cumulative Friction Work', fontsize=12)
        plt.ylabel('Cutter Wear Volume (mm³)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(individual_fit_dir / f"cutter_{cutter_id:02d}_k_fitting.png", dpi=150)
        plt.close()
        
        # --- 单独的磨损增长曲线图 (环号 vs 体积) ---
        plt.figure(figsize=(10, 6))
        plt.plot(wear_growth_volume_df.index.values, wear_growth_volume_df[f'cutter_{cutter_id}_wear_volume'].values, 
                color='green', linewidth=2, label=f'Cutter #{cutter_id}')
        plt.title(f'Cutter #{cutter_id}: Wear Growth Curve (Volume)', fontsize=14)
        plt.xlabel('Ring Number', fontsize=12)
        plt.ylabel('Calculated Cutter Wear Volume (mm³)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(individual_growth_dir / f"cutter_{cutter_id:02d}_wear_growth_volume.png", dpi=150)
        plt.close()
        
        # --- 单独的磨损增长曲线图 (环号 vs 半径) - 仅在启用转换时生成 ---
        if args.convert_to_radius:
            plt.figure(figsize=(10, 6))
            # 绘制插值得到的半径磨损值
            plt.plot(wear_growth_radius_df.index.values, wear_growth_radius_df[f'cutter_{cutter_id}_wear_radius'].values, 
                    color='purple', linewidth=2, label=f'Interpolated (k-based)')
            # 叠加实测数据点
            plt.scatter(aligned_friction.index, wear_radius_values, color='red', s=80, alpha=0.7, 
                       label='Measured Data', zorder=5)
            plt.title(f'Cutter #{cutter_id}: Wear Growth Curve (Radius)', fontsize=14)
            plt.xlabel('Ring Number', fontsize=12)
            plt.ylabel('Cutter Wear Radius (mm)', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(individual_radius_dir / f"cutter_{cutter_id:02d}_wear_growth_radius.png", dpi=150)
            plt.close()
    
    # --- 合并图：所有刀具的半径磨损增长曲线 - 仅在启用转换时生成 ---
    if args.convert_to_radius:
        print("Generating combined radius growth plot for all cutters...")
        plt.figure(figsize=(16, 10))
        colors = plt.cm.jet(np.linspace(0, 1, len(wear_df.index)))
        
        for i, cutter_id in enumerate(wear_df.index):
            plt.plot(wear_growth_radius_df.index.values, 
                    wear_growth_radius_df[f'cutter_{cutter_id}_wear_radius'].values, 
                    color=colors[i], linewidth=1.5, label=f'Cutter {cutter_id}')
        
        plt.title('All Cutters: Wear Growth Curves (Radius)', fontsize=18)
        plt.xlabel('Ring Number', fontsize=14)
        plt.ylabel('Calculated Cutter Wear Radius (mm)', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(plot_dir / "all_cutters_wear_growth_radius.png", dpi=300)
        plt.close()

    # --- Plot 3: Recorded vs Generated Wear Comparison (Volume) ---
    total_wear_instance = Total_wear()
    recorded_rings, recorded_wear_radius_values = total_wear_instance.get_data_pairs()
    
    # Convert recorded radius values to volume
    recorded_wear_volume_values = radius_to_volume(np.array(recorded_wear_radius_values))
    
    wear_per_ring = friction_df.groupby('ring_number')[wear_cols_volume].max()
    wear_per_ring['total_wear_volume'] = wear_per_ring[wear_cols_volume].sum(axis=1)
    
    generated_wear_at_records = wear_per_ring.reindex(recorded_rings)['total_wear_volume'].dropna()
    available_rings = generated_wear_at_records.index.tolist()
    available_recorded_wear = [recorded_wear_volume_values[recorded_rings.index(r)] for r in available_rings]

    if len(available_rings) > 0:
        plt.figure(figsize=(12, 8))
        x_pos = np.arange(len(available_rings))
        width = 0.35
        plt.bar(x_pos - width/2, available_recorded_wear, width, label='Recorded Total Wear (Volume)')
        plt.bar(x_pos + width/2, generated_wear_at_records, width, label='Generated Total Wear (Volume)')
        plt.xlabel('Ring Number')
        plt.ylabel('Total Wear (Volume mm³)')
        plt.title('Recorded vs. Generated Total Wear at Measurement Points')
        plt.xticks(x_pos, available_rings)
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(plot_dir / "wear_comparison_volume.png")
        plt.close()

    print("Visualization complete.")

    # --- 7. Save the result ---
    print(f"--- Step 6: Saving results to {args.output_csv} ---")
    try:
        friction_df.to_csv(args.output_csv, index=False)
        print("Successfully saved the final data with wear columns.")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

if __name__ == "__main__":
    calculate_wear_per_timestep()
