#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from pathlib import Path

class EnergyAnalysis:
    """基于能量分析的TBM刀具磨损预测类"""
    
    def __init__(self):
        "初始化能量分析器"
        print("能量分析模块已初始化。")

    def calculate_energy(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算扭矩功和推力功，包括小段值和累计值。
        使用向量化操作以提高效率。
        
        参数:
        - data: 包含TBM数据的DataFrame，必须包含扭矩、转速、推力、行程等列。
        
        返回:
        - 包含能量计算结果的更新后的DataFrame。
        """
        print("开始计算能量参数...")
        results_df = data.copy()

        # --- 1. 定义常量 ---
        TIME_INTERVAL_S = 5.0
        print(f"使用固定的时间间隔: {TIME_INTERVAL_S} 秒")

        # --- 2. 识别关键列 ---
        torque_col = results_df.filter(regex=r'(cutterhead_torque|刀盘扭矩)').columns[0]
        speed_col = results_df.filter(regex=r'(cutterhead_speed|刀盘转速)').columns[0]
        thrust_col = results_df.filter(regex=r'(propulsion_total_thrust|推进总推力)').columns[0]
        travel_col = results_df.filter(regex=r'(管理行程)').columns[0]
        propulsion_speed_col = results_df.filter(regex=r'(propulsion_average_speed|推进平均速度)').columns[0]

        print("成功识别到以下关键列:")
        print(f"  - 扭矩列: {torque_col}")
        print(f"  - 转速列: {speed_col}")
        print(f"  - 推力列: {thrust_col}")
        print(f"  - 行程列: {travel_col}")
        print(f"  - 掘进速度列: {propulsion_speed_col}")

        # --- 3. 基于行程差计算位移 (方法一) ---
        print("正在计算位移(基于行程差)并处理行程重置...")
        s_from_travel = results_df[travel_col].diff().fillna(0) / 1000  # 转换为米
        
        reset_indices = s_from_travel[s_from_travel < 0].index
        print(f"检测到 {len(reset_indices)} 个行程重置点。")

        for idx in reset_indices:
            if idx > 0:
                prev_idx = idx - 1
                prev_s = s_from_travel.get(prev_idx - 1, 0.0)
                s_from_travel.loc[prev_idx] = prev_s
                s_from_travel.loc[idx] = 0

        # --- 4. 处理掘进速度为0的情况 ---
        print("正在处理掘进速度为0的数据行...")
        imputed_speed_col = 'imputed_propulsion_speed'
        results_df[imputed_speed_col] = results_df[propulsion_speed_col].replace(0, np.nan)
        results_df[imputed_speed_col] = results_df[imputed_speed_col].bfill().ffill()
        results_df[imputed_speed_col] = results_df[imputed_speed_col].fillna(0)
        print("掘进速度为0的数据行已使用最近的有效值进行填充。")

        # --- 5. 基于掘进速度计算位移 (方法二) ---
        print("正在计算位移(基于掘进速度)...")
        propulsion_speed_mps = (results_df[imputed_speed_col] / 1000) / 60 # m/s
        s_from_speed = propulsion_speed_mps * TIME_INTERVAL_S

        # --- 6. 向量化计算瞬时功 ---
        print("正在向量化计算瞬时功...")
        angular_velocity = results_df[speed_col].abs() * 2 * math.pi / 60
        
        # 扭矩功
        torque_work_incremental = results_df[torque_col].abs() * angular_velocity * TIME_INTERVAL_S
        
        # 推力功 (方法一: 基于行程差)
        thrust_work_incremental_travel = results_df[thrust_col] * s_from_travel
        
        # 推力功 (方法二: 基于掘进速度)
        thrust_work_incremental_speed = results_df[thrust_col] * s_from_speed
        
        # --- 7. 将瞬时功添加到DataFrame ---
        results_df['torque_work_incremental'] = torque_work_incremental
        results_df['thrust_work_incremental_travel'] = thrust_work_incremental_travel
        results_df['thrust_work_incremental_speed'] = thrust_work_incremental_speed

        # --- 8. 计算累计功 ---
        print("正在计算累计功 (按环号累计)...")
        torque_work_cumulative_per_ring = results_df.groupby('ring_number')['torque_work_incremental'].cumsum()
        thrust_work_cumulative_travel_per_ring = results_df.groupby('ring_number')['thrust_work_incremental_travel'].cumsum()
        thrust_work_cumulative_speed_per_ring = results_df.groupby('ring_number')['thrust_work_incremental_speed'].cumsum()

        print("正在计算累计功 (从头到尾累计)...")
        torque_work_cumulative_total = torque_work_incremental.cumsum()
        thrust_work_cumulative_travel_total = thrust_work_incremental_travel.cumsum()
        thrust_work_cumulative_speed_total = thrust_work_incremental_speed.cumsum()
        
        # --- 9. 将累计功添加到DataFrame ---
        results_df['torque_work_cumulative_per_ring'] = torque_work_cumulative_per_ring
        results_df['thrust_work_cumulative_travel_per_ring'] = thrust_work_cumulative_travel_per_ring
        results_df['thrust_work_cumulative_speed_per_ring'] = thrust_work_cumulative_speed_per_ring
        
        results_df['torque_work_cumulative_total'] = torque_work_cumulative_total
        results_df['thrust_work_cumulative_travel_total'] = thrust_work_cumulative_travel_total
        results_df['thrust_work_cumulative_speed_total'] = thrust_work_cumulative_speed_total
        
        results_df = results_df.drop(columns=[imputed_speed_col])

        print("能量计算完成！")
        return results_df

def main():
    """
    主函数，执行能量计算流程。
    """
    print("开始执行TBM能量计算脚本...")
    
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
    except NameError:
        project_root = Path(".").resolve()

    input_file = project_root / "data" / "processed" / "extracted_tbm_tunneling_data.csv"
    output_file = project_root / "data" / "processed" / "tbm_data_with_energy.csv"
    
    if not input_file.exists():
        print(f"错误: 输入数据文件不存在: {input_file}")
        print("请先运行 `scripts/preprocessing/extract_data_by_index.py` 脚本生成该文件。")
        return

    print(f"正在从 {input_file} 加载数据...")
    try:
        tbm_data = pd.read_csv(input_file)
    except Exception as e:
        print(f"加载数据失败: {e}")
        return

    analyzer = EnergyAnalysis()
    energy_data = analyzer.calculate_energy(tbm_data)
    
    try:
        energy_data.to_csv(output_file, index=False)
        print(f"\n计算结果已成功保存到: {output_file}")
        print("新增列: ['torque_work_incremental', 'thrust_work_incremental_travel', 'thrust_work_incremental_speed', 'torque_work_cumulative_per_ring', 'thrust_work_cumulative_travel_per_ring', 'thrust_work_cumulative_speed_per_ring', 'torque_work_cumulative_total', 'thrust_work_cumulative_travel_total', 'thrust_work_cumulative_speed_total']")
    except Exception as e:
        print(f"保存结果失败: {e}")

    print("\n脚本执行完毕。")

if __name__ == "__main__":
    main()
