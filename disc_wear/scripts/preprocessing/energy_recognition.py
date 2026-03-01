import pickle
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import datetime
import sys
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

# Add project root to Python path to resolve module not found error
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the dataset class from our previous script
from scripts.preprocessing.dataset import TBMDataset

class LSTMModel(nn.Module):
    """LSTM model for sequence-to-sequence prediction."""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bn = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Apply Batch Normalization
        # x shape: (batch, seq_len, input_size)
        # BatchNorm1d expects (batch, input_size, seq_len)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

def train_model(use_saved_dataset=True):
    """Main function to train the LSTM model using K-Fold Cross-Validation."""
    # --- 1. Parameters ---
    CSV_PATH = '/media/young/4A8C40028C3FE75B/刀盘数据/disc_wear/data/processed/tbm_data_with_energy.csv'
    SEQ_LENGTH = 100
    STEP_SIZE = 1
    START_RING = 1
    END_RING = 20
    THRUST_WORK_METHOD = 'speed'
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    K_FOLDS = 5
    RANDOM_SEED = 42

    # --- 2. Setup Project Paths, TensorBoard Writer, and Device ---
    project_root = Path(__file__).resolve().parent.parent.parent
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = project_root / f'runs/energy_recognition_kfold_{timestamp}'
    checkpoint_dir = project_root / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 3. Load or Create Full Dataset ---
    dataset_save_dir = project_root / 'data/processed/datasets'
    dataset_path = dataset_save_dir / f'dataset_rings_{START_RING}_{END_RING}_seq{SEQ_LENGTH}_step{STEP_SIZE}_{THRUST_WORK_METHOD}.pkl'
    
    if use_saved_dataset and dataset_path.exists():
        print(f"Loading saved dataset from {dataset_path}...")
        full_dataset = TBMDataset.load(dataset_path)
    else:
        print(f"Creating new dataset (Rings {START_RING}-{END_RING})...")
        full_dataset = TBMDataset(
            csv_path=CSV_PATH, 
            sequence_length=SEQ_LENGTH,
            start_ring=START_RING,
            end_ring=END_RING,
            thrust_work_method=THRUST_WORK_METHOD,
            step_size=STEP_SIZE
        )
        
        if len(full_dataset) == 0:
            print("Error: Not enough data to create a dataset.")
            return
        
        # Save the dataset for future use
        print(f"Saving dataset to {dataset_path}...")
        full_dataset.save(dataset_path)

    if len(full_dataset) == 0:
        print("Error: Not enough data to create a dataset.")
        return

    # --- 4. K-Fold Cross-Validation Loop ---
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    overall_best_val_loss = float('inf')
    best_x_scaler = None
    best_y_scaler = None

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f"\n{'='*20} FOLD {fold+1}/{K_FOLDS} {'='*20}")

        # --- Data Preparation and Scaling for the current fold ---
        # Extract data for the current fold
        X_train_fold = torch.stack([full_dataset[i][0] for i in train_ids]).float()
        y_train_fold = torch.stack([full_dataset[i][1] for i in train_ids]).float()
        X_val_fold = torch.stack([full_dataset[i][0] for i in val_ids]).float()
        y_val_fold = torch.stack([full_dataset[i][1] for i in val_ids]).float()

        # --- Scale X (features) ---
        # Reshape for scaler: (num_samples * seq_len, num_features)
        n_train_samples, seq_len, n_features = X_train_fold.shape
        n_val_samples = X_val_fold.shape[0]
        X_train_reshaped = X_train_fold.reshape(-1, n_features)
        X_val_reshaped = X_val_fold.reshape(-1, n_features)

        # Apply Min-Max Scaler to features
        x_scaler = MinMaxScaler()
        X_train_scaled_np = x_scaler.fit_transform(X_train_reshaped.numpy())
        X_val_scaled_np = x_scaler.transform(X_val_reshaped.numpy())

        # Reshape back to (num_samples, seq_len, num_features) and convert to tensor
        X_train_scaled = torch.from_numpy(X_train_scaled_np).view(n_train_samples, seq_len, n_features).float()
        X_val_scaled = torch.from_numpy(X_val_scaled_np).view(n_val_samples, seq_len, n_features).float()

        # --- Scale Y (labels) ---
        # Reshape for scaler: (num_samples * seq_len, 1)
        y_train_reshaped = y_train_fold.reshape(-1, 1)
        y_val_reshaped = y_val_fold.reshape(-1, 1)

        # Apply Min-Max Scaler to labels
        y_scaler = MinMaxScaler()
        y_train_scaled_np = y_scaler.fit_transform(y_train_reshaped.numpy())
        y_val_scaled_np = y_scaler.transform(y_val_reshaped.numpy())

        # Reshape back and convert to tensor
        y_train_scaled = torch.from_numpy(y_train_scaled_np).view(n_train_samples, seq_len, 1).float()
        y_val_scaled = torch.from_numpy(y_val_scaled_np).view(n_val_samples, seq_len, 1).float()

        # Create new DataLoaders with scaled data
        train_dataset_fold = TensorDataset(X_train_scaled, y_train_scaled)
        val_dataset_fold = TensorDataset(X_val_scaled, y_val_scaled)

        train_loader = DataLoader(train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset_fold, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        # --- Initialize Model, Loss, and Optimizer for each fold ---
        input_size = full_dataset.feature_cols.__len__()
        hidden_size = 128
        num_layers = 2
        output_size = 1

        model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # --- Inner Training Loop (Epochs) ---
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_train_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
            
            avg_train_loss = running_train_loss / len(train_loader)
            writer.add_scalar(f'Fold_{fold+1}/Loss/Train', avg_train_loss, epoch)

            # --- Validation Phase ---
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()
            
            avg_val_loss = running_val_loss / len(val_loader)
            writer.add_scalar(f'Fold_{fold+1}/Loss/Validation', avg_val_loss, epoch)
            
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

            # --- Save Best Model (Overall) ---
            if avg_val_loss < overall_best_val_loss:
                overall_best_val_loss = avg_val_loss
                model_path = checkpoint_dir / 'kfold_best_energy_model.pth'
                torch.save(model.state_dict(), model_path)
                best_x_scaler = x_scaler
                best_y_scaler = y_scaler
                print(f"  -> New OVERALL best model saved to {model_path} (Val Loss: {avg_val_loss:.6f})")

    # --- Save the best scalers ---
    if best_x_scaler and best_y_scaler:
        scaler_path = checkpoint_dir / 'best_scalers.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump({'x_scaler': best_x_scaler, 'y_scaler': best_y_scaler}, f)
        print(f"Best scalers saved to {scaler_path}")

    writer.close()
    print(f"\n--- K-Fold Training Finished ---")
    print(f"Best validation loss across all folds: {overall_best_val_loss:.6f}")

def inverse_transform_output(scaled_output, scaler_path, scaler_type='y'):
    """
    Loads a saved scaler and applies inverse transformation.

    Args:
        scaled_output (np.ndarray): The scaled data from the model output or features.
        scaler_path (str or Path): Path to the saved .pkl file containing the scalers.
        scaler_type (str): 'x' for feature scaler, 'y' for label scaler. Default is 'y'.

    Returns:
        np.ndarray: The data in its original, un-normalized scale.
    """
    import pickle

    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)

    scaler_name = f'{scaler_type}_scaler'
    scaler = scalers.get(scaler_name)

    if scaler is None:
        raise ValueError(f"Scaler '{scaler_name}' not found in {scaler_path}")

    # The scaler expects a 2D array of shape (n_samples, n_features)
    original_shape = scaled_output.shape
    # Reshape for scaler, assuming last dimension is feature dimension
    reshaped_for_scaling = scaled_output.reshape(-1, original_shape[-1])

    unscaled_output = scaler.inverse_transform(reshaped_for_scaling)

    # Reshape back to the original input shape
    unscaled_output = unscaled_output.reshape(original_shape)

    return unscaled_output


if __name__ == '__main__':
    train_model()