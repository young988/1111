import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import pickle
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Add project root to Python path to resolve module not found error
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.preprocessing.dataset import TBMDataset
from scripts.preprocessing.energy_recognition import LSTMModel

def calculate_friction_energy():
    """Load a pre-trained model to infer friction energy and save the results."""
    # --- 1. Parameters ---
    parser = argparse.ArgumentParser(description="Calculate friction energy using a pre-trained model.")
    parser.add_argument('--input_csv', type=str, default=str(project_root / 'data/processed/tbm_data_with_energy.csv'), help='Path to the input CSV file.')
    parser.add_argument('--output_csv', type=str, default=str(project_root / 'data/processed/tbm_data_with_friction_energy.csv'), help='Path to save the output CSV file.')
    parser.add_argument('--model_path', type=str, default=str(project_root / 'checkpoints/kfold_best_energy_model.pth'), help='Path to the pre-trained model.')
    parser.add_argument('--scaler_path', type=str, default=str(project_root / 'checkpoints/best_scalers.pkl'), help='Path to the saved scalers file.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for inference.')
    args = parser.parse_args()

    SEQ_LENGTH = 100 # This must match the model's training sequence length
    STEP_SIZE = 100  # Use non-overlapping windows

    # --- 2. Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 3. Load Original Data and Create Dataset ---
    print(f"Loading data from {args.input_csv}...")
    try:
        original_df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_csv}")
        return

    print(f"Creating dataset with window size {SEQ_LENGTH} and step size {STEP_SIZE}...")
    inference_dataset = TBMDataset(
        csv_path=args.input_csv,
        sequence_length=SEQ_LENGTH,
        step_size=STEP_SIZE
    )

    if len(inference_dataset) == 0:
        print("Error: Dataset is empty. Cannot perform inference.")
        return

    inference_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- 4. Load Pre-trained Model ---
    print(f"Loading model from {args.model_path}...")
    try:
        input_size = inference_dataset.feature_cols.__len__()
        model = LSTMModel(input_size=input_size, hidden_size=128, num_layers=2, output_size=1)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 4.5 Load Scalers ---
    print(f"Loading scalers from {args.scaler_path}...")
    try:
        with open(args.scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        x_scaler = scalers.get('x_scaler')
        y_scaler = scalers.get('y_scaler')
        if x_scaler is None or y_scaler is None:
            raise ValueError("Both 'x_scaler' and 'y_scaler' must be present in the scaler file.")
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {args.scaler_path}")
        return
    except Exception as e:
        print(f"Error loading scalers: {e}")
        return

    # --- 5. Perform Inference ---
    print("Performing inference on the dataset...")
    all_preds_normalized = []
    all_labels_actual = []
    with torch.no_grad():
        for features, labels in inference_loader:
            # Normalize features before sending to model
            batch_size, seq_len, n_features = features.shape
            features_reshaped = features.reshape(-1, n_features).numpy()
            features_scaled_np = x_scaler.transform(features_reshaped)
            features_scaled = torch.from_numpy(features_scaled_np).view(batch_size, seq_len, n_features).float().to(device)
            
            outputs = model(features_scaled)
            all_preds_normalized.append(outputs.cpu())
            all_labels_actual.append(labels.cpu())
    
    preds_tensor = torch.cat(all_preds_normalized)
    labels_tensor = torch.cat(all_labels_actual)

    # --- 6. Denormalize and Calculate Per-Timestep Friction ---
    print("Denormalizing predictions and calculating per-timestep friction...")

    # Reshape for scaler: (num_windows * seq_len, 1)
    preds_reshaped = preds_tensor.reshape(-1, 1)
    # Denormalize the entire sequence of predictions
    preds_denormalized_flat = y_scaler.inverse_transform(preds_reshaped.numpy())
    # Reshape back to (num_windows, seq_len, 1)
    preds_denormalized = preds_denormalized_flat.reshape(preds_tensor.shape)

    # Labels are already in the correct scale
    labels_actual = labels_tensor.numpy()

    # Calculate cumulative friction energy for each window
    cumulative_friction_per_window = labels_actual - preds_denormalized

    # Calculate per-timestep friction energy from the cumulative values
    # We use np.diff to find the increment at each step, prepending 0 for the first step
    timestep_friction_per_window = np.diff(cumulative_friction_per_window, axis=1, prepend=np.zeros((cumulative_friction_per_window.shape[0], 1, 1)))


    # --- 7. Add Friction Columns to DataFrame ---
    print("Adding friction columns to the original dataframe...")

    # Create new columns initialized with NaN or 0
    original_df['friction_energy_timestep'] = 0.0

    # Iterate over each window to insert the 100 timestep values
    num_windows = timestep_friction_per_window.shape[0]
    for i in range(num_windows):
        start_idx = i * STEP_SIZE
        end_idx = start_idx + SEQ_LENGTH
        
        if end_idx > len(original_df):
            print(f"Warning: Window {i} extends beyond dataframe length. Truncating.")
            end_idx = len(original_df)
        
        # Assign the per-timestep values for the window
        timesteps_to_assign = timestep_friction_per_window[i, :end_idx-start_idx].flatten()
        original_df.loc[start_idx:end_idx-1, 'friction_energy_timestep'] = timesteps_to_assign

    # --- 7.5 Calculate Cumulative Friction Columns ---
    print("Calculating global and in-window cumulative friction...")

    # Apply threshold filter: values outside [-2000, 2000] are not counted
    FRICTION_THRESHOLD = 3000
    print(f"Applying threshold filter: friction values outside [-{FRICTION_THRESHOLD}, {FRICTION_THRESHOLD}] will be excluded from cumulative calculations")
    
    # Create a filtered version of friction_energy_timestep for global cumulative calculation
    friction_filtered = original_df['friction_energy_timestep'].copy()
    mask_outside_threshold = (friction_filtered < -FRICTION_THRESHOLD) | (friction_filtered > FRICTION_THRESHOLD)
    num_filtered = mask_outside_threshold.sum()
    print(f"  - Filtering out {num_filtered} timesteps ({num_filtered/len(original_df)*100:.2f}%) outside threshold range")
    
    # Set values outside threshold to 0 for cumulative calculation
    friction_filtered[mask_outside_threshold] = 0

    # Global cumulative sum with threshold filtering
    original_df['global_cumulative_friction_energy'] = friction_filtered.cumsum()

    # --- 7.6 Add Columns with Zeroed Negative Friction ---
    print("Adding columns with negative friction processed as zero...")

    # 1. Create a per-timestep friction column where negative values are treated as zero.
    original_df['friction_energy_timestep_zeroed'] = original_df['friction_energy_timestep'].clip(lower=0)

    # 2. Create global cumulative friction column based on this zeroed per-timestep data.
    original_df['global_cumulative_friction_from_zeroed'] = original_df['friction_energy_timestep_zeroed'].cumsum()


    # --- 8. Save New CSV File ---
    print(f"Saving results to {args.output_csv}...")
    try:
        original_df.to_csv(args.output_csv, index=False)
        print("Successfully saved the new CSV file with friction energy.")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == '__main__':
    calculate_friction_energy()
