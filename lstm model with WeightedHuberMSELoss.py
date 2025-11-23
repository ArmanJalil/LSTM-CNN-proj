# -*- coding: utf-8 -*-
"""
ARN_LSTM_Predictor.py
Usage: run in Spyder / Python environment with GPU and required packages installed.
Saves outputs to D:\testNN
"""

import os
import re
import json
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Settings (from your message)
# -----------------------
INPUT_WINDOW = 24
HORIZON = 12
TEST_SIZE = 240
EPOCHS = 20
SEED = 42
target_col = 'Veldan_PM2.5(ug/m3)'

CSV_PATH = r'C:\Users\arman\OneDrive\Desktop\AQIorgonized\gapfiledfinal.csv'
SAVE_DIR = r'D:\testNN'
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4  # L2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print GPU status so you can confirm
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
print("Using device:", DEVICE)

# -----------------------
# Reproducibility
# -----------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # for deterministic (may slow)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(CSV_PATH, parse_dates=[0], dayfirst=False)  # first column is Date
# Ensure first column named Date
if df.columns[0].lower() not in ['date', 'time', 'datetime']:
    df.rename({df.columns[0]: 'Date'}, axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

# build feature_cols from given indices (user provided)
# feature_cols = df.columns[[16, 6, 22, 45, 28,67,68,69,70,71,72,73,74,75,76]].tolist()
# but index selection must be within bounds; we'll attempt and fail gracefully
try:
    feature_cols = df.columns[[16, 6, 22, 45,57, 28,67,68,69,70,71,72,73,74,75,76]].tolist()
except Exception as e:
    print("Warning: feature index selection failed — check indices. Using a fallback: choose some plausible feature columns.")
    # fallback: choose many numeric columns except Date and target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    feature_cols = numeric_cols[:15]

print("Selected feature columns:", feature_cols)
if target_col not in df.columns:
    raise ValueError(f"target_col '{target_col}' not found in dataframe columns.")

# Keep only rows where target not null
df = df.loc[~df[target_col].isna()].reset_index(drop=True)

# -----------------------
# Prepare sequences (no shuffling)
# -----------------------
values_X = df[feature_cols].values.astype(float)
values_y = df[[target_col]].values.astype(float)  # shape (N,1)

N = len(df)
print("Total rows:", N)

# We'll build sequences: for i in range(0, N - INPUT_WINDOW - HORIZON + 1)
# Each sequence X = values_X[i : i+INPUT_WINDOW], y = values_y[i+INPUT_WINDOW : i+INPUT_WINDOW+HORIZON]
seq_starts = list(range(0, N - INPUT_WINDOW - HORIZON + 1))
total_sequences = len(seq_starts)
print("Total sequences:", total_sequences)

# Reserve last TEST_SIZE sequences as test (time-ordered)
if TEST_SIZE > total_sequences:
    raise ValueError("TEST_SIZE is larger than total available sequences.")
train_val_end = total_sequences - TEST_SIZE
train_val_indices = seq_starts[:train_val_end]
test_indices = seq_starts[train_val_end:]

# further split train into train/val (time ordered, e.g., last 10% as val)
val_frac = 0.1
val_count = max(1, int(len(train_val_indices) * val_frac))
train_indices = train_val_indices[:-val_count]
val_indices = train_val_indices[-val_count:]

print("Train seq count:", len(train_indices), "Val seq count:", len(val_indices), "Test seq count:", len(test_indices))

# Build datasets arrays (we will scale after building sequences to avoid lookahead)
def build_X_y(indices):
    X = []
    Y = []
    dates = []
    for i in indices:
        x = values_X[i : i+INPUT_WINDOW]
        y = values_y[i+INPUT_WINDOW : i+INPUT_WINDOW+HORIZON].reshape(-1)  # shape (HORIZON,)
        X.append(x)
        Y.append(y)
        dates.append(df['Date'].iloc[i+INPUT_WINDOW:i+INPUT_WINDOW+HORIZON].values)  # array of dates for each horizon
    return np.array(X), np.array(Y), np.array(dates)

X_train_raw, y_train_raw, dates_train = build_X_y(train_indices)
X_val_raw, y_val_raw, dates_val = build_X_y(val_indices)
X_test_raw, y_test_raw, dates_test = build_X_y(test_indices)

print("Shapes X_train, y_train:", X_train_raw.shape, y_train_raw.shape)
print("Shapes X_val, y_val:", X_val_raw.shape, y_val_raw.shape)
print("Shapes X_test, y_test:", X_test_raw.shape, y_test_raw.shape)

# -----------------------
# Scaling (fit only on training data)
# -----------------------
x_scaler = StandardScaler()
# fit on flattened train X (samples*time, features)
X_train_flat = X_train_raw.reshape(-1, X_train_raw.shape[2])
x_scaler.fit(X_train_flat)

def scale_X(X_raw):
    s = X_raw.reshape(-1, X_raw.shape[2])
    s = x_scaler.transform(s)
    return s.reshape(X_raw.shape)

X_train = scale_X(X_train_raw)
X_val = scale_X(X_val_raw)
X_test = scale_X(X_test_raw)

y_scaler = StandardScaler()
y_scaler.fit(y_train_raw.reshape(-1,1))
def scale_y(y_raw):
    s = y_raw.reshape(-1,1)
    s = y_scaler.transform(s)
    return s.reshape(y_raw.shape)

y_train = scale_y(y_train_raw)
y_val = scale_y(y_val_raw)
y_test = scale_y(y_test_raw)

# Save scalers now
joblib.dump(x_scaler, os.path.join(SAVE_DIR, 'x_scaler.pkl'))
joblib.dump(y_scaler, os.path.join(SAVE_DIR, 'y_scaler.pkl'))

# -----------------------
# PyTorch Dataset
# -----------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TimeSeriesDataset(X_train, y_train)
val_ds = TimeSeriesDataset(X_val, y_val)
test_ds = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)  # DO NOT shuffle as user asked (time series), but batching ok
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------
# Model (PyTorch) - architecture matching user desire
# We'll implement an encoder LSTM and a simple decoder: use last hidden state -> MLP to predict HORIZON values.
# -----------------------
input_size = X_train.shape[2]
hidden1 = 64
hidden2 = 64
hidden3 = 32
dropout_p = 0.3

class SeqPredictor(nn.Module):
    def __init__(self, input_size, hidden1=hidden1, hidden2=hidden2, hidden3=hidden3, horizon=HORIZON, dropout=dropout_p):
        super().__init__()
        self.encoder1 = nn.LSTM(input_size=input_size, hidden_size=hidden1, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.encoder2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, num_layers=1, batch_first=True)
        self.encoder3 = nn.LSTM(input_size=hidden2, hidden_size=hidden3, num_layers=1, batch_first=True)
        # final linear from hidden3 to horizon values
        self.fc = nn.Linear(hidden3, horizon)
        # We'll apply fc to last time-step hidden state
        # small weight init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, x):
        # x: batch, seq_len, features
        out, _ = self.encoder1(x)
        out = self.dropout1(out)
        out, _ = self.encoder2(out)
        out, (hn, cn) = self.encoder3(out)
        # hn: (num_layers, batch, hidden3) -> take last layer
        last_h = hn[-1]  # shape (batch, hidden3)
        res = self.fc(last_h)  # (batch, horizon)
        return res

model = SeqPredictor(input_size=input_size, hidden1=hidden1, hidden2=hidden2, hidden3=hidden3, horizon=HORIZON, dropout=dropout_p)
model = model.to(DEVICE)

# Loss and optimizer
# -----------------------
# Custom Weighted + Huber Loss
# -----------------------
class WeightedHuberMSELoss(nn.Module):
    def __init__(self, alpha=3.0, delta=1.0):
        """
        alpha: exponent for weighting large errors (e.g., 3.0–4.0 makes model more sensitive to extremes)
        delta: Huber transition point
        """
        super().__init__()
        self.alpha = alpha
        self.delta = delta

    def forward(self, preds, targets):
        diff = preds - targets
        abs_diff = diff.abs()
        # Weighted MSE part
        weighted_mse = (abs_diff ** self.alpha).mean()
        # Huber part
        huber = torch.where(abs_diff < self.delta, 0.5 * diff ** 2, self.delta * (abs_diff - 0.5 * self.delta))
        huber = huber.mean()
        return weighted_mse + 0.5 * huber  # combine both parts
criterion = WeightedHuberMSELoss(alpha=1.0, delta=5.0)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)

# -----------------------
# Training loop with per-horizon loss logging
# -----------------------
train_losses = []
val_losses = []
# we'll also compute per-horizon losses per epoch for plotting (train and val)
train_per_horizon = []  # list of arrays length HORIZON per epoch
val_per_horizon = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_losses = []
    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)  # shape (batch, HORIZON)
        optimizer.zero_grad()
        preds = model(xb)  # (batch, HORIZON)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    train_epoch_loss = np.mean(epoch_losses)
    train_losses.append(train_epoch_loss)

    # compute validation loss and per-horizon errors
    model.eval()
    val_epoch_losses = []
    # accumulators for per-horizon
    per_h_train = np.zeros(HORIZON)
    per_h_val = np.zeros(HORIZON)
    cnt_train = 0
    cnt_val = 0

    # compute per-horizon on entire training set (may be slow)
    with torch.no_grad():
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            preds = model(xb)
            # per-horizon mse
            err = ((preds - yb)**2).mean(dim=0).cpu().numpy()  # shape (HORIZON,)
            per_h_train += err * xb.size(0)
            cnt_train += xb.size(0)
        for xb, yb in val_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_epoch_losses.append(loss.item())
            err = ((preds - yb)**2).mean(dim=0).cpu().numpy()
            per_h_val += err * xb.size(0)
            cnt_val += xb.size(0)

    # finalize per-horizon
    if cnt_train > 0:
        per_h_train = per_h_train / cnt_train
    if cnt_val > 0:
        per_h_val = per_h_val / cnt_val

    train_per_horizon.append(per_h_train)
    val_per_horizon.append(per_h_val)

    val_epoch_loss = np.mean(val_epoch_losses) if val_epoch_losses else np.nan
    val_losses.append(val_epoch_loss)

    scheduler.step(val_epoch_loss if not np.isnan(val_epoch_loss) else train_epoch_loss)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {train_epoch_loss:.6f} - val_loss: {val_epoch_loss:.6f}")

# -----------------------
# Evaluate on train and test (R2 etc.)
# -----------------------
def predict_on_loader(loader):
    model.eval()
    preds_list = []
    trues_list = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            out = model(xb).cpu().numpy()  # (batch, HORIZON)
            preds_list.append(out)
            trues_list.append(yb.numpy())
    preds = np.vstack(preds_list)
    trues = np.vstack(trues_list)
    return preds, trues

# get scaled preds
train_preds_scaled, train_trues_scaled = predict_on_loader(train_loader)
test_preds_scaled, test_trues_scaled = predict_on_loader(test_loader)

# inverse scale
def inv_scale_y(y_scaled):
    s = y_scaled.reshape(-1,1)
    inv = y_scaler.inverse_transform(s)
    return inv.reshape(y_scaled.shape)

train_preds = inv_scale_y(train_preds_scaled)
train_trues = inv_scale_y(train_trues_scaled)
test_preds = inv_scale_y(test_preds_scaled)
test_trues = inv_scale_y(test_trues_scaled)

# compute metrics (we compute overall across all horizon points flattened)
def metrics(trues, preds):
    mse = mean_squared_error(trues.flatten(), preds.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues.flatten(), preds.flatten())
    r2 = r2_score(trues.flatten(), preds.flatten())
    return r2, rmse, mae, mse

r2_train, rmse_train, mae_train, mse_train = metrics(train_trues, train_preds)
r2_test, rmse_test, mae_test, mse_test = metrics(test_trues, test_preds)

print("Train - R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}, MSE: {:.4f}".format(r2_train, rmse_train, mae_train, mse_train))
print("Test  - R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}, MSE: {:.4f}".format(r2_test, rmse_test, mae_test, mse_test))

# also compute per-horizon R2 for test
per_h_r2 = []
for h in range(HORIZON):
    try:
        r = r2_score(test_trues[:, h], test_preds[:, h])
    except:
        r = np.nan
    per_h_r2.append(r)

# -----------------------
# Save model, scalers, config, predictions, loss history
# -----------------------
import json
model_name = "predictor_" + re.sub(r'[\\/*?:\"<>|().]', '_', target_col) + f"_horizon_{HORIZON}"
torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{model_name}.pth'))
joblib.dump(x_scaler, os.path.join(SAVE_DIR, f'{model_name}_x_scaler.pkl'))
joblib.dump(y_scaler, os.path.join(SAVE_DIR, f'{model_name}_y_scaler.pkl'))

model_config = {
    'target_column': target_col,
    'horizon': HORIZON,
    'input_window': INPUT_WINDOW,
    'input_size': input_size,
    'feature_columns': feature_cols,
    'model_architecture': 'SeqPredictor_LSTM_3layer',
    'training_parameters': {
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'epochs': EPOCHS
    },
    'performance_metrics': {
        'r2_train': float(r2_train),
        'rmse_train': float(rmse_train),
        'mae_train': float(mae_train),
        'mse_train': float(mse_train),
        'r2_test': float(r2_test),
        'rmse_test': float(rmse_test),
        'mae_test': float(mae_test),
        'mse_test': float(mse_test),
    },
    'data_info': {
        'total_sequences': total_sequences,
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'test_size': len(test_indices)
    }
}
with open(os.path.join(SAVE_DIR, f'{model_name}_config.json'), 'w', encoding='utf-8') as f:
    json.dump(model_config, f, indent=4, ensure_ascii=False)

# Save predictions (for test -- include dates aligned per-horizon)
# Build long table: for each sample index in test, for each horizon step, store date, actual, pred, horizon
rows = []
for i in range(len(test_preds)):
    for h in range(HORIZON):
        # dates_test has for each test sample an array of HORIZON np.datetime64 values
        date_val = pd.to_datetime(dates_test[i][h])
        rows.append({
            'sample_idx': i,
            'horizon': h+1,
            'date': date_val,
            'actual': float(test_trues[i, h]),
            'predicted': float(test_preds[i, h])
        })
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(SAVE_DIR, f'{model_name}_predictions.csv'), index=False)

loss_history_df = pd.DataFrame({
    'epoch': range(1, len(train_losses) + 1),
    'train_loss': train_losses,
    'val_loss': val_losses
})
loss_history_df.to_csv(os.path.join(SAVE_DIR, f'{model_name}_loss_history.csv'), index=False)

# -----------------------
# Plotting: page with HORIZON rows x 3 columns (total 3*H plots)
# Column1: loss per epoch for each lag (we have per-horizon arrays: train_per_horizon, val_per_horizon)
# Column2: scatter for test actual vs pred for that lag with r2 in top-left
# Column3: full test period with date (only date labels) for that lag: real vs predicted
# -----------------------
import matplotlib.dates as mdates

fig, axes = plt.subplots(HORIZON, 3, figsize=(18, 4*HORIZON))
fig.suptitle(f"{target_col} - Future Prediction (Horizon {HORIZON} hours)", fontsize=18)

# create a results_df as earlier for plotting convenience
results_df['date_dt'] = pd.to_datetime(results_df['date'])

# Prepare test full series per horizon: expand results_df grouped by horizon
for h in range(HORIZON):
    row = h
    # Column 1: loss per epoch for this lag - FIXED EPOCH PLOTTING
    ax1 = axes[row, 0]
    # collect per-epoch train/val loss for horizon h
    train_h_losses = [per_h[h] for per_h in train_per_horizon]
    val_h_losses = [per_h[h] for per_h in val_per_horizon]
    
    # Use actual epoch numbers (1, 2, 3, ... EPOCHS)
    epochs = list(range(1, len(train_h_losses) + 1))
    
    ax1.plot(epochs, train_h_losses, label='train_loss_per_h', marker='o', markersize=2)
    ax1.plot(epochs, val_h_losses, label='val_loss_per_h', marker='s', markersize=2)
    ax1.set_title(f'Prediction +{h+1}h - Loss per epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Set integer x-axis ticks for epochs
    if len(epochs) <= 20:
        ax1.set_xticks(epochs)
    else:
        # Show every 5th epoch if there are many
        ax1.set_xticks(epochs[::5])

    # Column 2: scatter actual vs predicted for test for this horizon
    ax2 = axes[row, 1]
    y_true_h = test_trues[:, h]
    y_pred_h = test_preds[:, h]
    ax2.scatter(y_true_h, y_pred_h, s=8)
    ax2.set_title(f'Prediction +{h+1}h - Test scatter')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    # r2 for this horizon
    try:
        r2_h = r2_score(y_true_h, y_pred_h)
    except:
        r2_h = np.nan
    ax2.text(0.02, 0.95, f'R2={r2_h:.3f}', transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))
    lims = [min(np.nanmin(y_true_h), np.nanmin(y_pred_h)), max(np.nanmax(y_true_h), np.nanmax(y_pred_h))]
    ax2.plot(lims, lims, '--', linewidth=0.8)
    ax2.grid(True)

    # Column 3: full test period with CORRECT FUTURE TIME ALIGNMENT
    ax3 = axes[row, 2]
    dfh = results_df[results_df['horizon'] == (h+1)].copy()
    dfh = dfh.sort_values('date_dt')
    
    # Plot with proper time alignment - these are FUTURE predictions
    ax3.plot(dfh['date_dt'], dfh['actual'], label='Actual', linewidth=1.5)
    ax3.plot(dfh['date_dt'], dfh['predicted'], label='Predicted', linewidth=1.5, alpha=0.8)
    ax3.set_title(f'Prediction +{h+1}h - Future values')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Value')
    ax3.legend()
    
    # Improved date formatting
    unique_dates = dfh['date_dt'].values
    n_dates = len(unique_dates)
    max_ticks = 8
    if n_dates <= max_ticks:
        ticks = unique_dates
    else:
        idxs = np.linspace(0, n_dates-1, max_ticks).astype(int)
        ticks = unique_dates[idxs]
    ax3.set_xticks(ticks)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plot_path = os.path.join(SAVE_DIR, f'{model_name}_summary_plots.png')
plt.savefig(plot_path, bbox_inches='tight', dpi=200)
plt.show(fig)

print("All saved to:", SAVE_DIR)
print("Model name:", model_name)
print(f"Model predicts {HORIZON} hours into the future using past {INPUT_WINDOW} hours of data")