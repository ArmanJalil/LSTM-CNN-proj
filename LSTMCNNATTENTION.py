import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os, random, json, joblib
import re
import matplotlib.dates as mdates
# ============ CONFIG ============
SAVE_DIR = r'D:\testNN'
os.makedirs(SAVE_DIR, exist_ok=True)
INPUT_FILE = r'C:\Users\arman\OneDrive\Desktop\AQIorgonized\gapfiledfinal.csv'

INPUT_WINDOW = 48
HORIZON = 40
TEST_SIZE = 750
VAL_SIZE = 750
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.0005
SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Predicting ONLY the {HORIZON}th hour from now")

# ============ LOAD DATA ============
df = pd.read_csv(INPUT_FILE, parse_dates=['Date'])
target_col = 'Veldan_PM2.5(ug/m3)'
feature_cols = df.columns[[57, 16, 6, 22, 45, 28,67,68,69,70,71,72,73,74,75,76]].tolist()#[c for c in df.columns if c != 'Date']

print(f"Data shape: {df.shape}")
print(f"Target column: {target_col}")
print(f"Number of features: {len(feature_cols)}")

X_raw = df[feature_cols].values.astype(np.float32)
Y_raw = df[[target_col]].values.astype(np.float32)

# Handle missing values
print(f"Missing values in X: {np.isnan(X_raw).sum()}")
print(f"Missing values in Y: {np.isnan(Y_raw).sum()}")

X_raw[np.isnan(X_raw)] = 0.0
Y_raw[np.isnan(Y_raw)] = 0.0

x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X_raw)
y_scaler = StandardScaler()
Y_scaled = y_scaler.fit_transform(Y_raw)

print(f"X scaled range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
print(f"Y scaled range: [{Y_scaled.min():.2f}, {Y_scaled.max():.2f}]")

# ============ CREATE SEQUENCES ============
def create_single_target_sequences(X, Y, input_window, horizon):
    Xs, Ys = [], []
    for i in range(len(X) - input_window - horizon + 1):
        Xs.append(X[i:i+input_window])
        Ys.append(Y[i+input_window+horizon-1, 0])
    return np.array(Xs), np.array(Ys)

X_seq, Y_seq = create_single_target_sequences(X_scaled, Y_scaled, INPUT_WINDOW, HORIZON)
print(f"Total sequences: {len(X_seq)}")
print(f"X_seq shape: {X_seq.shape}, Y_seq shape: {Y_seq.shape}")

# ============ DATA SPLITTING ============
# REMOVED SHUFFLING: Comment out these two lines
# indices = np.random.permutation(len(X_seq))
# X_seq = X_seq[indices]
# Y_seq = Y_seq[indices]

total_sequences = len(X_seq)
train_size = total_sequences - (TEST_SIZE + VAL_SIZE)

if total_sequences < (TEST_SIZE + VAL_SIZE):
    raise ValueError(f"Not enough data. Need {TEST_SIZE + VAL_SIZE} sequences but only have {total_sequences}")

X_train = X_seq[:train_size]
Y_train = Y_seq[:train_size]
X_val = X_seq[train_size:train_size+VAL_SIZE]
Y_val = Y_seq[train_size:train_size+VAL_SIZE]
X_test = X_seq[train_size+VAL_SIZE:train_size+VAL_SIZE+TEST_SIZE]
Y_test = Y_seq[train_size+VAL_SIZE:train_size+VAL_SIZE+TEST_SIZE]

print(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

# ============ SIMPLIFIED MODEL ============
class SimplifiedPredictor(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size=256, dropout=0.5):
        super().__init__()
        
        self.input_projection = nn.Linear(input_size * seq_len, hidden_size)
        
        self.residual_layers = nn.ModuleList()
        for i in range(4):
            self.residual_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, 1)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x_flat = x.reshape(x.size(0), -1)
        features = self.input_projection(x_flat)
        
        for residual_layer in self.residual_layers:
            residual = features
            features = residual_layer(features)
            features = features + residual
            features = self.layer_norm(features)
        
        output = self.output(features)
        return output.squeeze(-1)

# ============ DATASET WITH GPU OPTIMIZATION ============
class SequenceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Use pin_memory for faster GPU transfer
train_loader = DataLoader(SequenceDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(SequenceDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader = DataLoader(SequenceDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

seq_len = X_train.shape[1]
input_size = X_train.shape[2]
model = SimplifiedPredictor(input_size, seq_len).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============ TRAINING WITH PROPER GPU HANDLING ============
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

train_losses, val_losses = [], []
best_val_loss = float('inf')
patience = 15
patience_counter = 0
best_model_state = None

print("\nStarting training with GPU...")
print("Epoch | Train Loss | Val Loss  | Gap    | Status")
print("-" * 50)

for epoch in range(EPOCHS):
    # ============ TRAINING ============
    model.train()
    train_loss = 0.0
    train_samples = 0
    
    for xb, yb in train_loader:
        # Move data to GPU with non_blocking for faster transfer
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * xb.size(0)
        train_samples += xb.size(0)
    
    avg_train_loss = train_loss / train_samples
    train_losses.append(avg_train_loss)

    # ============ VALIDATION ============
    model.eval()
    val_loss = 0.0
    val_samples = 0
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item() * xb.size(0)
            val_samples += xb.size(0)
    
    avg_val_loss = val_loss / val_samples
    val_losses.append(avg_val_loss)
    
    scheduler.step()
    
    loss_gap = avg_val_loss - avg_train_loss
    
    # Early stopping and model saving
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        status = "✓ Best"
        torch.save(best_model_state, os.path.join(SAVE_DIR, 'best_model.pth'))
    else:
        patience_counter += 1
        status = f"Wait {patience_counter}/{patience}"

    if epoch % 5 == 0 or epoch == 0 or patience_counter == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d} | {avg_train_loss:.6f} | {avg_val_loss:.6f} | {loss_gap:+.4f} | {status}")

    if patience_counter >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Loaded best model for evaluation")

# ============ FIXED EVALUATION WITH PROPER GPU->CPU TRANSFER ============
model.eval()
test_preds, test_trues = [], []
test_loss = 0.0
test_samples = 0

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        test_loss += loss.item() * xb.size(0)
        test_samples += xb.size(0)
        
        # FIX: Move tensors to CPU before converting to numpy
        test_preds.append(pred.cpu().numpy())  # ← FIXED HERE
        test_trues.append(yb.cpu().numpy())    # ← FIXED HERE

avg_test_loss = test_loss / test_samples
test_preds = np.concatenate(test_preds, axis=0)
test_trues = np.concatenate(test_trues, axis=0)

# Inverse transform
preds_inv = y_scaler.inverse_transform(test_preds.reshape(-1, 1)).flatten()
trues_inv = y_scaler.inverse_transform(test_trues.reshape(-1, 1)).flatten()

# Calculate metrics
r2 = r2_score(trues_inv, preds_inv)
mse = mean_squared_error(trues_inv, preds_inv)
mae = mean_absolute_error(trues_inv, preds_inv)
rmse = np.sqrt(mse)

print(f"\n{'='*60}")
print(f"FINAL TEST RESULTS")
print(f"{'='*60}")
print(f"Best Validation Loss: {best_val_loss:.6f}")
print(f"Test Loss: {avg_test_loss:.6f}")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Create a unique model name with target and horizon
model_name = "predictor_" + re.sub(r'[\\/*?:"<>|().]', '_', target_col) + f"_horizon_{HORIZON}"

# Save the model
torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{model_name}.pth'))

# Save all scalers and metadata
import joblib
joblib.dump(x_scaler, os.path.join(SAVE_DIR, f'{model_name}_x_scaler.pkl'))
joblib.dump(y_scaler, os.path.join(SAVE_DIR, f'{model_name}_y_scaler.pkl'))

# Save model configuration
model_config = {
    'target_column': target_col,
    'horizon': HORIZON,
    'input_window': INPUT_WINDOW,
    'input_size': X_train.shape[2],
    'feature_columns': feature_cols,
    'model_architecture': 'SimpleLSTM',
    'training_parameters': {
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'epochs': EPOCHS
    },
    'performance_metrics': {
        'r2_score': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'mse': float(mse)
    },
    'data_info': {
        'total_sequences': total_sequences,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test)
    }
}

with open(os.path.join(SAVE_DIR, f'{model_name}_config.json'), 'w') as f:
    json.dump(model_config, f, indent=4)

# Save predictions and loss history
results_df = pd.DataFrame({
    'Actual': trues_inv,
    'Predicted': preds_inv,
    'Error': preds_inv - trues_inv
})
results_df.to_csv(os.path.join(SAVE_DIR, f'{model_name}_predictions.csv'), index=False)

loss_history_df = pd.DataFrame({
    'epoch': range(1, len(train_losses) + 1),
    'train_loss': train_losses,
    'val_loss': val_losses
})
loss_history_df.to_csv(os.path.join(SAVE_DIR, f'{model_name}_loss_history.csv'), index=False)

# ============ PLOTTING ============
clean_col_name = target_col.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')

fig = plt.figure(figsize=(15, 10))

## ============ PLOTTING WITH CORRECT DATES ============

# Calculate the corresponding dates for test predictions
# Since we didn't shuffle, test sequences are at the END of the original data
test_start_idx = train_size + VAL_SIZE  # This is the starting index of test sequences

test_dates = []
for i in range(len(X_test)):
    # Calculate the actual date index in the original dataframe
    # i goes from 0 to TEST_SIZE-1, so we add test_start_idx to get the correct sequence position
    sequence_start_idx = test_start_idx + i
    date_idx = sequence_start_idx + INPUT_WINDOW + HORIZON - 1
    
    if date_idx < len(df):
        test_dates.append(df['Date'].iloc[date_idx])
    else:
        test_dates.append(df['Date'].iloc[-1])

# Convert to numpy array for easier slicing
test_dates = np.array(test_dates)

# Ensure we have the same number of dates as predictions
assert len(test_dates) == len(trues_inv), f"Date length {len(test_dates)} doesn't match predictions {len(trues_inv)}"

clean_col_name = target_col.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')

fig = plt.figure(figsize=(15, 10))

# Plot 1: R² scatter plot (unchanged)
ax1 = plt.subplot(2, 2, 1)
ax1.scatter(trues_inv, preds_inv, s=30, alpha=0.6, color='blue')
mn = min(trues_inv.min(), preds_inv.min())
mx = max(trues_inv.max(), preds_inv.max())
ax1.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect prediction')
ax1.set_xlabel('Actual Values (ppm)')
ax1.set_ylabel('Predicted Values (ppm)')
ax1.set_title(f'{target_col}\nR² = {r2:.4f} ({HORIZON}-hour ahead)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Loss curves (unchanged)
ax2 = plt.subplot(2, 2, 2)
ax2.plot(train_losses, label='Train Loss', linewidth=2, color='blue')
ax2.plot(val_losses, label='Validation Loss', linewidth=2, color='red')
ax2.legend()
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE Loss')
ax2.set_title('Training History (Simplified Model)')
ax2.grid(True, alpha=0.3)

# Plot 3: Full test period with day dates
ax3 = plt.subplot(2, 2, 3)
ax3.plot(test_dates, trues_inv, label='Actual', linewidth=1, alpha=0.8, color='blue')
ax3.plot(test_dates, preds_inv, label='Predicted', linewidth=1, alpha=0.8, color='red')
ax3.set_xlabel('Date')
ax3.set_ylabel('CO Concentration (ppm)')
ax3.set_title(f'Test Period ({len(trues_inv)} samples) - Correct Time Alignment')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Format x-axis to show ALL dates
import matplotlib.dates as mdates

# Calculate the date range
date_range = (test_dates[-1] - test_dates[0]).days + 1

# Set up date formatting to show every date
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Show year-month-day

# Show more dates - adjust interval based on total days
if date_range <= 10:  # If 10 days or less, show every day
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=1))
elif date_range <= 20:  # If 11-20 days, show every 2nd day
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=2))
else:  # If more than 20 days, show every 3rd-5th day
    interval = max(3, date_range // 8)
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=interval))

plt.xticks(rotation=45)
plt.tight_layout()

# Plot 4: Last 100 samples with 24-hour format
ax4 = plt.subplot(2, 2, 4)
last_100 = min(100, len(trues_inv))
last_dates = test_dates[-last_100:]
ax4.plot(last_dates, trues_inv[-last_100:], label='Actual', linewidth=2, color='blue')
ax4.plot(last_dates, preds_inv[-last_100:], label='Predicted', linewidth=2, color='red')
ax4.set_xlabel('Date and Time')
ax4.set_ylabel('CO Concentration (ppm)')
ax4.set_title('Last 100 Samples - Hourly View')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Format x-axis to show both date and time (24-hour format every 12 hours)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))  # Month-day Hour:Minute
ax4.xaxis.set_major_locator(mdates.HourLocator(interval=12))  # Every 12 hours
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, f'{model_name}_results.png'), dpi=150, bbox_inches='tight')
plt.show()

# Additional verification: Print date ranges
print(f"\nDate range for test predictions:")
print(f"Start: {test_dates[0]}")
print(f"End: {test_dates[-1]}")
print(f"Total duration: {(test_dates[-1] - test_dates[0]).days} days")

# Verify time alignment
print(f"\nTime alignment verification:")
print(f"Each prediction is made {HORIZON} hours ahead of the last input value")
print(f"Input window: {INPUT_WINDOW} hours")
print(f"Horizon: {HORIZON} hours")