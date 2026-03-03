import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class CutterWearDataset(Dataset):
    def __init__(self, csv_path, sequence_length=100, start_ring=None, end_ring=None, 
                 step_size=1, normalize=True):
        """
        刀具磨损预测数据集 - 预测所有刀具的磨损值（使用真实值修正后的数据）
        
        Args:
            csv_path: 数据文件路径 (wear_per_timestep_final.csv)
            sequence_length: 序列长度
            start_ring: 起始环号
            end_ring: 结束环号
            step_size: 滑动窗口步长
            normalize: 是否对特征和标签进行标准化
        """
        self.sequence_length = sequence_length
        self.normalize = normalize
        self.scaler_X = None
        self.scaler_y = None
        
        # Load data
        df = pd.read_csv(csv_path, low_memory=False)

        # Filter by ring number if specified
        if start_ring is not None and end_ring is not None:
            df['ring_number'] = pd.to_numeric(df['ring_number'], errors='coerce')
            df = df.dropna(subset=['ring_number'])
            df['ring_number'] = df['ring_number'].astype(int)
            df = df[(df['ring_number'] >= start_ring) & (df['ring_number'] <= end_ring)]

        # 定义需要排除的列（保留96个参数作为特征）
        features_to_exclude = [
            '[1]管理行程',
            '[2]记录日期', '[3]记录时刻', 'ring_number',
            '[4]系统掘进状态',
            'torque_work_incremental',
            'thrust_work_incremental_travel', 
            'thrust_work_incremental_speed',
            'torque_work_cumulative_per_ring',
            'thrust_work_cumulative_travel_per_ring',
            'thrust_work_cumulative_speed_per_ring',
            'torque_work_cumulative_total',
            'thrust_work_cumulative_travel_total',
            'thrust_work_cumulative_speed_total',
            'global_cumulative_travel',
            'friction_energy_timestep',
            'global_cumulative_friction_energy',
            'friction_energy_timestep_zeroed',
            'global_cumulative_friction_from_zeroed',
        ]
        
        # 添加所有磨损列到排除列表
        wear_columns = [col for col in df.columns if 'cutter_' in col and '_wear' in col]
        features_to_exclude.extend(wear_columns)
        
        # 获取特征列
        feature_columns = [col for col in df.columns if col not in features_to_exclude]

        # 获取所有刀具的修正后磨损标签列（预测所有44把刀）
        corrected_wear_columns = sorted(
            [col for col in df.columns if '_wear_radius_corrected' in col],
            key=lambda x: int(x.split('_')[1])
        )
        
        if len(corrected_wear_columns) == 0:
            raise ValueError("未找到修正后的磨损列 (*_wear_radius_corrected)，请先运行 calculate_and_correct_wear.py")
        
        self.num_cutters = len(corrected_wear_columns)
        self.cutter_ids = [int(col.split('_')[1]) for col in corrected_wear_columns]
        
        print(f"使用的特征列数量: {len(feature_columns)}")
        print(f"预测刀具数量: {self.num_cutters}")
        print(f"刀具编号: {self.cutter_ids[:5]}...{self.cutter_ids[-5:]}")
        
        # 提取特征和标签
        df_features = df[feature_columns].copy()
        df_labels = df[corrected_wear_columns].copy()
        
        # 转换为数值类型并删除NaN行
        df_features = df_features.apply(pd.to_numeric, errors='coerce')
        df_labels = df_labels.apply(pd.to_numeric, errors='coerce')
        
        valid_indices = ~(df_features.isna().any(axis=1) | df_labels.isna().any(axis=1))
        df_features = df_features[valid_indices]
        df_labels = df_labels[valid_indices]

        if df_features.empty or df_labels.empty:
            self.X, self.y = np.array([]), np.array([])
            self.feature_columns = []
            self.label_columns = []
            return
        
        self.feature_columns = feature_columns
        self.label_columns = corrected_wear_columns

        features = df_features.values
        labels = df_labels.values

        # 标准化处理
        if self.normalize:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            features = self.scaler_X.fit_transform(features)
            labels = self.scaler_y.fit_transform(labels)
            print(f"特征和标签标准化完成 (StandardScaler)")

        # Create sequences
        self.X, self.y = [], []

        for i in range(0, len(features) - sequence_length + 1, step_size):
            window_features = features[i : i + sequence_length]
            self.X.append(window_features)
            
            window_labels = labels[i : i + sequence_length]
            if len(window_labels) > 0:
                # 以窗口第一个时间步作为基准，计算相对累计值
                window_labels_cumulative = window_labels - window_labels[0:1]
                self.y.append(window_labels_cumulative)
            else:
                self.y.append(window_labels)

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])
    
    def inverse_transform_y(self, y_normalized):
        """将标准化的标签转换回原始值"""
        if self.scaler_y is None:
            return y_normalized
        
        original_shape = y_normalized.shape
        y_flat = y_normalized.reshape(-1, original_shape[-1])
        y_original = self.scaler_y.inverse_transform(y_flat)
        return y_original.reshape(original_shape)
    
    def inverse_transform_X(self, X_normalized):
        """将归一化的特征转换回原始值"""
        if self.scaler_X is None:
            return X_normalized
        
        original_shape = X_normalized.shape
        X_flat = X_normalized.reshape(-1, original_shape[-1])
        X_original = self.scaler_X.inverse_transform(X_flat)
        return X_original.reshape(original_shape)


if __name__ == '__main__':
    # 使用相对路径
    project_root = Path(__file__).resolve().parent.parent.parent
    CSV_PATH = project_root / "data" / "processed" / "wear_per_timestep_final.csv"
    
    SEQ_LENGTH = 100
    BATCH_SIZE = 64
    START_RING = 1
    END_RING = 400
    STEP_SIZE = 10
    
    try:
        print(f"Loading cutter wear data for rings {START_RING} to {END_RING}...")
        print(f"Using corrected wear values (radius_corrected)")
        print(f"Using sliding window step size: {STEP_SIZE}")
        print(f"Data path: {CSV_PATH}")
            
        dataset = CutterWearDataset(
            csv_path=CSV_PATH, 
            sequence_length=SEQ_LENGTH,
            start_ring=START_RING,
            end_ring=END_RING,
            step_size=STEP_SIZE,
            normalize=True
        )
        
        if len(dataset) == 0:
            print("No data available for the specified parameters.")
        else:
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            features, labels = next(iter(dataloader))

            print(f"\n--- Dataset Verification ---")
            print(f"Total number of samples: {len(dataset)}")
            print(f"Features batch shape: {features.shape}")
            print(f"Labels batch shape (all cutters): {labels.shape}")
            print(f"Number of input features per timestep: {features.shape[2]}")
            print(f"Number of output cutters: {labels.shape[2]}")
            print(f"Sequence length: {features.shape[1]}")
            print(f"Features range: [{features.min():.4f}, {features.max():.4f}]")
            print(f"Labels range: [{labels.min():.4f}, {labels.max():.4f}]")

    except FileNotFoundError:
        print(f"Error: The file was not found at {CSV_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
