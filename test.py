import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import random
import json
import re
import matplotlib.dates as mdates
# ============ RESULTS ============
SAVE_DIR = r'D:\testNN'
os.makedirs(SAVE_DIR, exist_ok=True)
# ============ USER CONFIG ============
INPUT_FILE = r'C:\Users\arman\OneDrive\Desktop\AQIorgonized\gapfiledfinal.csv'
INPUT_WINDOW = 48
HORIZON = 32  # Predicting only one step ahead
TEST_SIZE = 750
VAL_SIZE = 750
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001
SEED = 42

# Set device for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if device.type == 'cuda':
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

print('Device:', device)
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name()}')
print(f'Predicting {HORIZON} hour(s) ahead for 25Aban_CO (ppm)')

# ============ LOAD DATA ============
df = pd.read_csv(INPUT_FILE, parse_dates=['Date'])

# SINGLE TARGET
target_col = 'Veldan_PM2.5(ug/m3)'
target_cols = [target_col]

# Include target column in features for auto-correlation
feature_cols = df.columns[[16, 6, 22, 45, 28,67,68,69,70,71,72,73,74,75,76]].tolist()#[c for c in df.columns if c not in ['Date']]
print(f"Target: {target_col}")
print(f"Features: {len(feature_cols)} (including target for auto-correlation)")

# ============ DATA PREPROCESSING ============
X_raw = df[feature_cols].values.astype(np.float32)
Y_raw = df[target_cols].values.astype(np.float32)

X_raw[np.isnan(X_raw)] = 0.0
Y_raw[np.isnan(Y_raw)] = 0.0

x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X_raw)

# ============ SEQUENCE CREATION WITH HORIZON ============
def create_multi_horizon(X, Y, input_window, horizon):
    Xs, Ys = [], []
    T = len(X)
    for i in range(T - input_window - horizon + 1):
        Xs.append(X[i:i+input_window])
        Ys.append(Y[i+input_window+horizon-1])
    return np.array(Xs), np.array(Ys)

X_seq, Y_seq = create_multi_horizon(X_scaled, Y_raw, INPUT_WINDOW, HORIZON)
if X_seq.shape[0] == 0:
    raise ValueError('No sequences created — reduce INPUT_WINDOW or HORIZON or check data length.')

print(f"Total sequences: {X_seq.shape[0]}")

# ============ DATA SPLIT ============
total_sequences = X_seq.shape[0]
required_total = TEST_SIZE + VAL_SIZE

if total_sequences <= required_total:
    raise ValueError(f'Not enough sequences. Need {required_total} but only have {total_sequences}')

X_train = X_seq[:-(TEST_SIZE + VAL_SIZE)]
Y_train = Y_seq[:-(TEST_SIZE + VAL_SIZE)]
X_val = X_seq[-(TEST_SIZE + VAL_SIZE):-TEST_SIZE]
Y_val = Y_seq[-(TEST_SIZE + VAL_SIZE):-TEST_SIZE]
X_test = X_seq[-TEST_SIZE:]
Y_test = Y_seq[-TEST_SIZE:]

print(f"Data split: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test")

# ============ TARGET SCALING ============
y_scaler = StandardScaler()
Y_train_scaled = y_scaler.fit_transform(Y_train.reshape(-1, 1)).flatten()
Y_val_scaled = y_scaler.transform(Y_val.reshape(-1, 1)).flatten()
Y_test_scaled = y_scaler.transform(Y_test.reshape(-1, 1)).flatten()

print(f"Target scaling - Mean: {y_scaler.mean_[0]:.4f}, Scale: {y_scaler.scale_[0]:.4f}")

# ============ DATASETS & DATA LOADERS ============
class SingleDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_loader = DataLoader(SingleDataset(X_train, Y_train_scaled), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(SingleDataset(X_val, Y_val_scaled), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader = DataLoader(SingleDataset(X_test, Y_test_scaled), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ============ SIMPLIFIED MODEL ============
class CNN_LSTM_Sharp(nn.Module):
    def __init__(self, input_size, output_size=1, dropout=0.1):
        super().__init__()
        self.input_size = input_size

        # --- CNN feature extractor (deeper & sharper) ---
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=5, padding="same"),
            nn.LeakyReLU(0.1),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU()
        )

        # --- LSTM for temporal modeling ---
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2,
                            batch_first=True, dropout=dropout)

        # --- Fully connected output ---
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # → (batch, channels=input_size, seq_len)

        # CNN feature extraction
        x = self.conv_block(x)

        # LSTM expects (batch, seq_len, features)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # last time step

        # Output
        out = self.fc(lstm_out)
        return out

model = CNN_LSTM_Sharp(input_size=X_train.shape[2], output_size=1).to(device)


print(f"Model: input={X_train.shape[2]}, output=1")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

# ============ TRAINING ============
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 15
patience_counter = 0

print("Starting training with simplified model...")
print("Epoch | Train Loss | Val Loss  | LR       | Improvement")
print("-" * 60)

for epoch in range(1, EPOCHS+1):
    # Training
    model.train()
    t_loss_total = 0.0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(xb).squeeze()
        
        loss = criterion(pred, yb)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        t_loss_total += loss.item()
    
    train_loss = t_loss_total / len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    v_loss_total = 0.0
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            v_loss_total += loss.item()
    
    val_loss = v_loss_total / len(val_loader)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    improvement = (best_val_loss - val_loss) / best_val_loss * 100 if best_val_loss != float('inf') else 0
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch {epoch:3d}/{EPOCHS} | {train_loss:.6f} | {val_loss:.6f} | {current_lr:.2e} | {improvement:+.2f}%')
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

# ============ FINAL EVALUATION ============
model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device, non_blocking=True)
        out = model(xb).cpu().numpy()
        preds.append(out)
        trues.append(yb.numpy())

preds = np.concatenate(preds, axis=0).flatten()
trues = np.concatenate(trues, axis=0).flatten()

# Inverse transform predictions
preds_inv = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
trues_inv = y_scaler.inverse_transform(trues.reshape(-1, 1)).flatten()



# Calculate metrics
r2 = r2_score(trues_inv, preds_inv)
mse = np.mean((trues_inv - preds_inv) ** 2)
mae = np.mean(np.abs(trues_inv - preds_inv))
rmse = np.sqrt(mse)

print("\n" + "="*60)
print(f"FINAL TEST RESULTS - {target_col}")
print("="*60)
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"Train/Val ratio: {train_losses[-1]:.6f}/{val_losses[-1]:.6f} = {train_losses[-1]/val_losses[-1]:.2f}")

# ============ SAVE EVERYTHING FOR DEPLOYMENT ============
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
# The test sequences start from: total_sequences - TEST_SIZE
test_start_idx = total_sequences - TEST_SIZE

# Each test sequence corresponds to a date at: input_window + horizon - 1 steps from sequence start
test_dates = []
for i in range(test_start_idx, test_start_idx + TEST_SIZE):
    # Calculate the actual date index in the original dataframe
    date_idx = i + INPUT_WINDOW + HORIZON - 1
    if date_idx < len(df):
        test_dates.append(df['Date'].iloc[date_idx])
    else:
        # Handle edge case
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
ax4.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))  # Month-day Hour:Minute
ax4.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=12))  # Every 12 hours
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