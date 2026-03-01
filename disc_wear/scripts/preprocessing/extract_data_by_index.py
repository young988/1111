#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import glob
from typing import List, Dict, Tuple
from pathlib import Path

class TBMDataExtractor:
    """TBM数据提取器 - 根据列名提取数据"""
    
    def __init__(self, data_folder: str, header_folder: str):
        self.data_folder = data_folder
        self.header_folder = header_folder
        self.headers = self._load_headers()
        
        # 安全地获取掘进状态列名
        tunneling_status_index = 4 # 掘进状态列的0-based索引
        if tunneling_status_index >= len(self.headers):
            raise ValueError(f"错误: 掘进状态列的索引({tunneling_status_index})超出了表头范围(总列数: {len(self.headers)})。")
        self.tunneling_status_column = self.headers[tunneling_status_index]
        
        self.parameter_columns = self._get_parameter_columns()
        self.extracted_data = None

    def _load_headers(self) -> List[str]:
        """从表头文件夹加载列名"""
        header_files = sorted(glob.glob(os.path.join(self.header_folder, "*.Csv")))
        if not header_files:
            raise FileNotFoundError(f"在目录 {self.header_folder} 中未找到任何表头文件 (*.Csv)")
        
        all_headers = []
        for file_path in header_files:
            print(f"正在从 {file_path} 加载表头...")
            try:
                headers = pd.read_csv(file_path, encoding='gbk', nrows=0).columns.tolist()
                all_headers.extend(headers)
            except Exception as e:
                print(f"使用 gbk 解码失败: {e}，尝试使用 utf-8...")
                headers = pd.read_csv(file_path, encoding='utf-8', nrows=0).columns.tolist()
                all_headers.extend(headers)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(all_headers))

    def _get_parameter_columns(self) -> List[str]:
        """根据索引定义获取所有需要提取的列名，并进行安全检查"""
        # 原始的基于0-based索引的参数映射
        parameter_indices_mapping = {
            '行程记录/mm_1': [1],
            '日期_1': [2],
            '时刻_1': [3],
            '掘进状态_1': [4],
            'cutter_motor_current_1_6_1': [9],
            'cutter_motor_current_1_6_2': [12],
            'cutter_motor_current_1_6_3': [15],
            'cutter_motor_current_1_6_4': [18],
            'cutter_motor_current_1_6_5': [21],
            'cutter_motor_current_1_6_6': [24],
            'cutter_motor_frequency_1_6_1': [10],
            'cutter_motor_frequency_1_6_2': [13],
            'cutter_motor_frequency_1_6_3': [16],
            'cutter_motor_frequency_1_6_4': [19],
            'cutter_motor_frequency_1_6_5': [22],
            'cutter_motor_frequency_1_6_6': [25],
            'cutter_motor_torque_1_6_1': [11],
            'cutter_motor_torque_1_6_2': [14],
            'cutter_motor_torque_1_6_3': [17],
            'cutter_motor_torque_1_6_4': [20],
            'cutter_motor_torque_1_6_5': [23],
            'cutter_motor_torque_1_6_6': [26],
            'cutterhead_torque_1': [33],
            'cutterhead_speed_1': [35],
            'soil_pressure_front_1_6_1': [88],
            'soil_pressure_front_1_6_2': [89],
            'soil_pressure_front_1_6_3': [90],
            'soil_pressure_front_1_6_4': [91],
            'soil_pressure_front_1_6_5': [92],
            'soil_pressure_front_1_6_6': [93],
            'cutter_drive_seal_temp_1': [1086],
            'main_bearing_lubrication_1_2_1': [1087],
            'main_bearing_lubrication_1_2_2': [1088],
            'pinion_bearing_lubrication_1': [1089],
            'motor_actual_frequency_1_6_1': [1450],
            'motor_actual_frequency_1_6_2': [1451],
            'motor_actual_frequency_1_6_3': [1452],
            'motor_actual_frequency_1_6_4': [1453],
            'motor_actual_frequency_1_6_5': [1454],
            'motor_actual_frequency_1_6_6': [1455],
            'motor_speed_1_6_1': [1457],
            'motor_speed_1_6_2': [1458],
            'motor_speed_1_6_3': [1459],
            'motor_speed_1_6_4': [1460],
            'motor_speed_1_6_5': [1461],
            'motor_speed_1_6_6': [1462],
            'motor_current_1_6_1': [1464],
            'motor_current_1_6_2': [1465],
            'motor_current_1_6_3': [1466],
            'motor_current_1_6_4': [1467],
            'motor_current_1_6_5': [1468],
            'motor_current_1_6_6': [1469],
            'motor_power_percentage_1_6_1': [1478],
            'motor_power_percentage_1_6_2': [1479],
            'motor_power_percentage_1_6_3': [1480],
            'motor_power_percentage_1_6_4': [1481],
            'motor_power_percentage_1_6_5': [1482],
            'motor_power_percentage_1_6_6': [1483],
            'motor_voltage_1_6_1': [1485],
            'motor_voltage_1_6_2': [1486],
            'motor_voltage_1_6_3': [1487],
            'motor_voltage_1_6_4': [1488],
            'motor_voltage_1_6_5': [1489],
            'motor_voltage_1_6_6': [1490],
            'belt_conveyor_flow_1': [46],
            'screw_gate_travel_1': [98],
            'screw_speed_1': [106],
            'screw_torque_1': [107],
            'spiral_discharge_rate_1': [1430],
            'propulsion_average_speed_1': [52],
            'propulsion_total_oil_pressure_1': [51],
            'propulsion_total_thrust_1': [50],
            'zone_oil_pressure_1_4_1': [56],
            'zone_oil_pressure_1_4_2': [59],
            'zone_oil_pressure_1_4_3': [62],
            'zone_oil_pressure_1_4_4': [65],
            'zone_travel_1_4_1': [57],
            'zone_travel_1_4_2': [60],
            'zone_travel_1_4_3': [63],
            'zone_travel_1_4_4': [66],
            'zone_average_speed_1_4_1': [58],
            'zone_average_speed_1_4_2': [61],
            'zone_average_speed_1_4_3': [64],
            'zone_average_speed_1_4_4': [67],
            'foam_solution_flow_1': [150],
            'foam_air_flow_1': [158],
            'foam_flow_1': [166],
            'foam_solution_pressure_1': [174],
            'foam_air_pressure_1': [182],
            'foam_pressure_1': [190],
            'bentonite_flow_1_4_1': [214],
            'bentonite_flow_1_4_2': [230],
            'water_temperature_1': [1079],
            'oil_temperature_1': [1080],
            'system_voltage_1': [1099],
            'system_current_1': [1100],
            'active_power_1': [1101],
            'reactive_power_1': [1102],
            'frequency_1': [1103],
            'power_factor_1': [1104],
        }
        all_indices = set()
        for indices in parameter_indices_mapping.values():
            all_indices.update(indices)
        
        parameter_columns = []
        max_header_index = len(self.headers) - 1
        for idx in sorted(list(all_indices)):
            if 0 <= idx <= max_header_index:
                parameter_columns.append(self.headers[idx])
            else:
                print(f"警告: 索引 {idx} 超出表头范围 (0-{max_header_index})，将被忽略。")

        print(f"将提取以下 {len(parameter_columns)} 个有效参数列: {parameter_columns}")
        return parameter_columns

    def get_csv_files(self) -> List[str]:
        """获取所有数据CSV文件"""
        csv_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        csv_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        return csv_files
    
    def filter_tunneling_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """根据掘进状态列名过滤数据"""
        if df.empty or self.tunneling_status_column not in df.columns:
            print(f"警告: 掘进状态列 '{self.tunneling_status_column}' 不存在，跳过过滤。")
            return df
        
        original_rows = len(df)
        # Convert column to numeric, coercing errors to NaN. This handles non-numeric values.
        df[self.tunneling_status_column] = pd.to_numeric(df[self.tunneling_status_column], errors='coerce')
        filtered_df = df[df[self.tunneling_status_column] == 2].copy()
        print(f"  原始数据行数: {original_rows}, 掘进状态为2的数据行数: {len(filtered_df)}")
        return filtered_df
    
    def process_single_file(self, file_path: str) -> pd.DataFrame:
        """处理单个CSV文件"""
        print(f"正在处理文件: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path, header=None, names=self.headers, low_memory=False)
            df_filtered = self.filter_tunneling_data(df)
            
            if df_filtered.empty:
                print(f"  该文件没有掘进状态为2的数据，跳过。")
                return pd.DataFrame()
            
            # 检查所需列是否存在
            existing_cols = [col for col in self.parameter_columns if col in df_filtered.columns]
            return df_filtered[existing_cols]
                    
        except Exception as e:
            print(f"  处理文件时出错: {e}")
            return pd.DataFrame()
    
    def process_all_files(self, output_file: str = None) -> pd.DataFrame:
        """处理所有CSV文件"""
        csv_files = self.get_csv_files()
        print(f"在 {self.data_folder} 中找到 {len(csv_files)} 个CSV文件")
        
        all_data = []
        for i, file_path in enumerate(csv_files):
            print(f"处理进度: {i+1}/{len(csv_files)}")
            ring_number = os.path.basename(file_path).split('.')[0]
            extracted_data = self.process_single_file(file_path)
            
            if not extracted_data.empty:
                extracted_data['ring_number'] = ring_number
                all_data.append(extracted_data)
        
        if not all_data:
            print("未能从任何文件中提取有效数据。")
            return pd.DataFrame()

        final_data = pd.concat(all_data, axis=0, ignore_index=True)
        print(f"\n最终合并数据大小: {final_data.shape[0]} 行 x {final_data.shape[1]} 列")
        
        if output_file:
            final_data.to_csv(output_file, index=False)
            print(f"数据已保存到: {output_file}")
        
        self.extracted_data = final_data
        return final_data

def main():
    """
    主函数，执行数据提取流程。
    """
    print("开始执行TBM数据提取脚本...")
    
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
    except NameError:
        project_root = Path(".").resolve()

    data_folder = project_root / "data" / "raw" / "处理后各环数据"
    header_folder = project_root / "data" / "raw" / "原始数据-用于查表头"
    output_file = project_root / "data" / "processed" / "extracted_tbm_tunneling_data.csv"
    
    if not data_folder.exists() or not header_folder.exists():
        print(f"错误: 数据目录 {data_folder} 或表头目录 {header_folder} 不存在。")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        extractor = TBMDataExtractor(str(data_folder), str(header_folder))
        extractor.process_all_files(output_file=str(output_file))
        print("\n数据提取成功！")
    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")

    print("\n脚本执行完毕。")

if __name__ == "__main__":
    main()