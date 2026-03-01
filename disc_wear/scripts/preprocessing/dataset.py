import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path

class TBMDataset(Dataset):
    def __init__(self, csv_path, sequence_length=100, start_ring=None, end_ring=None, thrust_work_method='speed', step_size=1):
        self.sequence_length = sequence_length

        # Load data
        df = pd.read_csv(csv_path)

        # Filter by ring number if specified
        if start_ring is not None and end_ring is not None:
            df['ring_number'] = pd.to_numeric(df['ring_number'], errors='coerce')
            df = df.dropna(subset=['ring_number'])
            df['ring_number'] = df['ring_number'].astype(int)
            df = df[(df['ring_number'] >= start_ring) & (df['ring_number'] <= end_ring)]

        # Define features and label
        features_to_exclude = [
            # Unwanted original features (with column numbers as they appear in CSV)
            '[33]刀盘扭矩', '[1]管理行程', '[2]记录日期', '[3]记录时刻', '[4]系统掘进状态',
            # Identifiers
            'ring_number',
            # Unwanted cumulative/energy columns
            'torque_work_cumulative_per_ring',
            'thrust_work_cumulative_travel_per_ring',
            'thrust_work_cumulative_speed_per_ring',
            'torque_work_cumulative_total',
            'thrust_work_cumulative_travel_total',
            'thrust_work_cumulative_speed_total',
            'global_cumulative_travel'
        ]
        
        thrust_feature_col = ''
        if thrust_work_method == 'speed':
            features_to_exclude.append('thrust_work_incremental_travel')
            thrust_feature_col = 'thrust_work_incremental_speed'
        elif thrust_work_method == 'travel':
            features_to_exclude.append('thrust_work_incremental_speed')
            thrust_feature_col = 'thrust_work_incremental_travel'
        
        df = df.drop(columns=features_to_exclude, errors='ignore')
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        if df.empty:
            self.X, self.y = np.array([]), np.array([])
            return

        self.feature_cols = [col for col in df.columns if col != 'torque_work_incremental']
        self.label_col = 'torque_work_incremental'
        
        features = df[self.feature_cols].values
        labels = df[self.label_col].values.reshape(-1, 1)

        # Create sequences with cumulative values
        self.X, self.y = [], []
        thrust_col_idx = self.feature_cols.index(thrust_feature_col)

        for i in range(0, len(features) - sequence_length + 1, step_size):
            # Input sequence
            window_features = features[i : i + sequence_length].copy()
            
            # Replace thrust work with cumulative sum within the window
            window_features[:, thrust_col_idx] = np.cumsum(window_features[:, thrust_col_idx])
            self.X.append(window_features)
            
            # Label: cumulative torque work sequence over the window
            window_labels = labels[i : i + sequence_length]
            self.y.append(np.cumsum(window_labels, axis=0))

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])

    def save(self, save_path, save_excel=True):
        """
        Save the dataset to a pickle file and optionally to Excel format.
        
        Args:
            save_path (str or Path): Path to save the dataset.
            save_excel (bool): Whether to also save an Excel file for inspection.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data_dict = {
            'X': self.X,
            'y': self.y,
            'feature_cols': self.feature_cols,
            'label_col': self.label_col,
            'sequence_length': self.sequence_length
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"Dataset saved to {save_path}")
        print(f"  - Samples: {len(self.X)}")
        print(f"  - Features: {len(self.feature_cols)}")
        print(f"  - Sequence length: {self.sequence_length}")
        
        # Save Excel file for inspection
        if save_excel and len(self.X) > 0:
            excel_path = save_path.with_suffix('.xlsx')
            print(f"\nSaving Excel file for inspection to {excel_path}...")
            
            # Create a flattened version for Excel viewing
            # We'll save the first few samples with their sequences
            num_samples_to_save = min(100, len(self.X))  # Save first 100 samples
            
            excel_data = []
            for sample_idx in range(num_samples_to_save):
                for seq_idx in range(self.sequence_length):
                    row_data = {
                        'sample_index': sample_idx,
                        'sequence_step': seq_idx,
                        self.label_col: self.y[sample_idx, seq_idx, 0]
                    }
                    # Add all features
                    for feat_idx, feat_name in enumerate(self.feature_cols):
                        row_data[feat_name] = self.X[sample_idx, seq_idx, feat_idx]
                    
                    excel_data.append(row_data)
            
            df_excel = pd.DataFrame(excel_data)
            
            # Use openpyxl engine for better compatibility
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df_excel.to_excel(writer, sheet_name='Dataset_Preview', index=False)
                
                # Add a summary sheet
                summary_data = {
                    'Metric': [
                        'Total Samples',
                        'Sequence Length',
                        'Number of Features',
                        'Samples in Excel (Preview)',
                        'Feature Names'
                    ],
                    'Value': [
                        len(self.X),
                        self.sequence_length,
                        len(self.feature_cols),
                        num_samples_to_save,
                        ', '.join(self.feature_cols)
                    ]
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            print(f"  - Excel file saved with {num_samples_to_save} samples (preview)")
            print(f"  - Contains 2 sheets: 'Dataset_Preview' and 'Summary'")
    
    @classmethod
    def load(cls, load_path):
        """
        Load a dataset from a pickle file.
        
        Args:
            load_path (str or Path): Path to the saved dataset file.
            
        Returns:
            TBMDataset: Loaded dataset instance.
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        # Create an empty dataset instance
        dataset = cls.__new__(cls)
        dataset.X = data_dict['X']
        dataset.y = data_dict['y']
        dataset.feature_cols = data_dict['feature_cols']
        dataset.label_col = data_dict['label_col']
        dataset.sequence_length = data_dict['sequence_length']
        
        print(f"Dataset loaded from {load_path}")
        print(f"  - Samples: {len(dataset.X)}")
        print(f"  - Features: {len(dataset.feature_cols)}")
        print(f"  - Sequence length: {dataset.sequence_length}")
        
        return dataset

if __name__ == '__main__':
    # Parameters
    CSV_PATH = '/media/young/4A8C40028C3FE75B/刀盘数据/disc_wear/data/processed/tbm_data_with_energy.csv'
    SEQ_LENGTH = 100
    BATCH_SIZE = 64
    START_RING = 1
    END_RING = 30
    THRUST_WORK_METHOD = 'speed' # 'speed' or 'travel'
    STEP_SIZE = 1
    
    # Dataset save path
    SAVE_DIR = Path('/media/young/4A8C40028C3FE75B/刀盘数据/disc_wear/data/processed/datasets')
    SAVE_PATH = SAVE_DIR / f'dataset_rings_{START_RING}_{END_RING}_seq{SEQ_LENGTH}_step{STEP_SIZE}_{THRUST_WORK_METHOD}.pkl'
    
    # Create Dataset and DataLoader
    try:
        print(f"Loading data for rings {START_RING} to {END_RING}...")
        print(f"Using '{THRUST_WORK_METHOD}' method for thrust work calculation.")
        print(f"Using sliding window step size: {STEP_SIZE}")
        dataset = TBMDataset(
            csv_path=CSV_PATH, 
            sequence_length=SEQ_LENGTH,
            start_ring=START_RING,
            end_ring=END_RING,
            thrust_work_method=THRUST_WORK_METHOD,
            step_size=STEP_SIZE
        )
        
        if len(dataset) == 0:
            print("No data available for the specified ring range.")
        else:
            # Save the dataset
            dataset.save(SAVE_PATH)
            
            # Test loading
            print("\n--- Testing dataset loading ---")
            loaded_dataset = TBMDataset.load(SAVE_PATH)
            
            # Verify loaded data
            dataloader = DataLoader(loaded_dataset, batch_size=BATCH_SIZE, shuffle=True)

            # Get a batch of data and print shapes
            features, labels = next(iter(dataloader))

            print(f"\n--- Dataset Verification ---")
            print(f"Total number of samples: {len(loaded_dataset)}")
            print(f"Features batch shape: {features.shape}")
            print(f"Labels batch shape: {labels.shape}")
            print(f"Number of features: {features.shape[2]}")
            print(f"Sequence length: {features.shape[1]}")
            print(f"--------------------------")

    except FileNotFoundError:
        print(f"Error: The file was not found at {CSV_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()