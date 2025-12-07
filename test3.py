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
# Settings
# -----------------------
INPUT_WINDOW = 24
HORIZON = 12
TEST_SIZE = 240
EPOCHS = 200
SEED = 42
target_col = 'Sepahan_Shahr_NO2(ppb)'

CSV_PATH = r'C:\Users\arman\OneDrive\Desktop\AQIorgonized\gapfiledfinal.csv'
SAVE_DIR = r'D:\testNN'
corr_file_path = r"C:\Users\arman\OneDrive\Desktop\AQIorgonized\gappfilledby9\filtered_correlations_less_24_nan.csv"
corr_df = pd.read_csv(corr_file_path)
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# -----------------------
# KGE metric function
# -----------------------
def kling_gupta_efficiency(obs, sim):
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[mask]
    sim = sim[mask]
    
    if len(obs) == 0:
        return np.nan
    
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge

# -----------------------
# Reproducibility
# -----------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# -----------------------
# Load data and apply logarithmic transformation
# -----------------------
df = pd.read_csv(CSV_PATH, parse_dates=[0], dayfirst=False)
if df.columns[0].lower() not in ['date', 'time', 'datetime']:
    df.rename({df.columns[0]: 'Date'}, axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

# Find target column in dataframe
target_col_found = None
for col in df.columns:
    if target_col in col or col in target_col:
        target_col_found = col
        break

if target_col_found is None:
    raise ValueError(f"target_col '{target_col}' not found in dataframe columns.")

target_col = target_col_found
print(f"Using target column: {target_col}")

# Get target column index
target_col_index = list(df.columns).index(target_col)
print(f"Target column index: {target_col_index}")

# Find correlated indices from correlation file
target_col_clean = target_col.replace(' ', '').replace('(', '').replace(')', '').strip()
target_row = None
for idx, row in corr_df.iterrows():
    corr_col_clean = row['Column_Name'].replace(' ', '').replace('(', '').replace(')', '').strip()
    if target_col_clean in corr_col_clean or corr_col_clean in target_col_clean:
        target_row = row
        break

if target_row is not None:
    correlated_indices = eval(target_row['Top_Correlated_Column_Numbers'])
    indices = [target_col_index] + correlated_indices + [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77]
    print(f"Found in correlation file: {target_row['Column_Name']}")
else:
    print("Target column not found in correlation file. Using default indices.")
    
# Remove duplicates while preserving order
seen = set()
indices_unique = []
for idx in indices:
    if idx not in seen:
        seen.add(idx)
        indices_unique.append(idx)
indices = indices_unique

# Filter valid indices
indices = [idx for idx in indices if idx < len(df.columns)]

# Build feature columns
feature_cols = [df.columns[i] for i in indices]
print(f"\nFeature columns ({len(feature_cols)}):")
for i, col in enumerate(feature_cols[:5]):
    print(f"  {i+1}. {col}")
if len(feature_cols) > 5:
    print(f"  ... and {len(feature_cols)-5} more")

# Apply log transformation to all features except temperature and dew point
print("\nApplying logarithmic transformation to features...")
for col in feature_cols:
    if any(keyword in col.lower() for keyword in ['temp', 'temperature', 'dew', 'dewpoint']):
        continue
    
    if col in df.columns:
        min_val = df[col].min()
        if min_val <= 0:
            shift = abs(min_val) + 1
            df[col] = np.log1p(df[col] + shift)
        else:
            df[col] = np.log1p(df[col])

# Apply log transformation to target - STORE SHIFT VALUE
print(f"Applying log transformation to target: {target_col}")
min_target = df[target_col].min()
if min_target <= 0:
    shift_target = abs(min_target) + 1
    df[target_col] = np.log1p(df[target_col] + shift_target)
    print(f"Applied log(1+x+{shift_target}) to target")
else:
    df[target_col] = np.log1p(df[target_col])
    shift_target = 0
    print("Applied log(1+x) to target")

# -----------------------
# Prepare sequences
# -----------------------
values_X = df[feature_cols].values.astype(float)
values_y = df[[target_col]].values.astype(float)

N = len(df)
print("Total rows:", N)

seq_starts = list(range(0, N - INPUT_WINDOW - HORIZON + 1))
total_sequences = len(seq_starts)
print("Total sequences:", total_sequences)

if TEST_SIZE > total_sequences:
    raise ValueError("TEST_SIZE is larger than total available sequences.")
train_val_end = total_sequences - TEST_SIZE
train_val_indices = seq_starts[:train_val_end]
test_indices = seq_starts[train_val_end:]

val_frac = 0.1
val_count = max(1, int(len(train_val_indices) * val_frac))
train_indices = train_val_indices[:-val_count]
val_indices = train_val_indices[-val_count:]

print("Train seq count:", len(train_indices), "Val seq count:", len(val_indices), "Test seq count:", len(test_indices))

def build_X_y(indices):
    X, Y, dates = [], [], []
    for i in indices:
        x = values_X[i : i+INPUT_WINDOW]
        y = values_y[i+INPUT_WINDOW : i+INPUT_WINDOW+HORIZON].reshape(-1)
        X.append(x)
        Y.append(y)
        dates.append(df['Date'].iloc[i+INPUT_WINDOW:i+INPUT_WINDOW+HORIZON].values)
    return np.array(X), np.array(Y), np.array(dates)

X_train_raw, y_train_raw, dates_train = build_X_y(train_indices)
X_val_raw, y_val_raw, dates_val = build_X_y(val_indices)
X_test_raw, y_test_raw, dates_test = build_X_y(test_indices)

print("Shapes X_train, y_train:", X_train_raw.shape, y_train_raw.shape)

# -----------------------
# Scaling
# -----------------------
x_scaler = StandardScaler()
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
BATCH_SIZE = 64

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------
# Model
# -----------------------
input_size = X_train.shape[2]
hidden1, hidden2, hidden3 = 64, 64, 32
dropout_p = 0.4

class SeqPredictor(nn.Module):
    def __init__(self, input_size, hidden1=hidden1, hidden2=hidden2, hidden3=hidden3, horizon=HORIZON, dropout=dropout_p):
        super().__init__()
        self.encoder1 = nn.LSTM(input_size=input_size, hidden_size=hidden1, num_layers=1, batch_first=True)
        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)
        self.encoder2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, num_layers=1, batch_first=True)
        self.relu1 = nn.ReLU()
        self.encoder3 = nn.LSTM(input_size=hidden2, hidden_size=hidden3, num_layers=1, batch_first=True)
        self.tanh2 = nn.Tanh()
        self.fc = nn.Linear(hidden3, horizon)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        out, _ = self.encoder1(x)
        out = self.tanh1(out)
        out = self.dropout1(out)
        out, _ = self.encoder2(out)
        out = self.relu1(out)
        out, (hn, cn) = self.encoder3(out)
        out = self.tanh2(out)
        last_h = hn[-1]
        return self.fc(last_h)

model = SeqPredictor(input_size=input_size)
model = model.to(DEVICE)

LR = 1e-4
WEIGHT_DECAY = 1e-5
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)

# -----------------------
# Training
# -----------------------
train_losses, val_losses = [], []
train_per_horizon, val_per_horizon = [], []
patience, min_val_loss = 30, float('inf')
patience_counter, best_model_state = 0, None

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_losses.append(loss.item())
    
    train_epoch_loss = np.mean(epoch_losses)
    train_losses.append(train_epoch_loss)

    model.eval()
    val_epoch_losses, per_h_train, per_h_val = [], np.zeros(HORIZON), np.zeros(HORIZON)
    cnt_train, cnt_val = 0, 0

    with torch.no_grad():
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            err = ((preds - yb)**2).mean(dim=0).cpu().numpy()
            per_h_train += err * xb.size(0)
            cnt_train += xb.size(0)
        
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_epoch_losses.append(loss.item())
            err = ((preds - yb)**2).mean(dim=0).cpu().numpy()
            per_h_val += err * xb.size(0)
            cnt_val += xb.size(0)

    if cnt_train > 0:
        per_h_train = per_h_train / cnt_train
    if cnt_val > 0:
        per_h_val = per_h_val / cnt_val

    train_per_horizon.append(per_h_train)
    val_per_horizon.append(per_h_val)

    val_epoch_loss = np.mean(val_epoch_losses) if val_epoch_losses else np.nan
    val_losses.append(val_epoch_loss)
    scheduler.step(val_epoch_loss if not np.isnan(val_epoch_loss) else train_epoch_loss)

    if val_epoch_loss < min_val_loss:
        min_val_loss = val_epoch_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {train_epoch_loss:.6f} - val_loss: {val_epoch_loss:.6f} - patience: {patience_counter}/{patience}")

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        model.load_state_dict(best_model_state)
        break

if best_model_state is not None and patience_counter < patience:
    model.load_state_dict(best_model_state)

# -----------------------
# Evaluation
# -----------------------
def predict_on_loader(loader):
    model.eval()
    preds_list, trues_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            out = model(xb).cpu().numpy()
            preds_list.append(out)
            trues_list.append(yb.numpy())
    return np.vstack(preds_list), np.vstack(trues_list)

train_preds_scaled, train_trues_scaled = predict_on_loader(train_loader)
test_preds_scaled, test_trues_scaled = predict_on_loader(test_loader)

def inv_scale_y(y_scaled):
    s = y_scaled.reshape(-1,1)
    inv = y_scaler.inverse_transform(s)
    return inv.reshape(y_scaled.shape)

train_preds_log = inv_scale_y(train_preds_scaled)
train_trues_log = inv_scale_y(train_trues_scaled)
test_preds_log = inv_scale_y(test_preds_scaled)
test_trues_log = inv_scale_y(test_trues_scaled)

def inv_log_transform(log_values, shift_applied):
    """Correct inverse of log(1+x) transformation"""
    exp_values = np.exp(log_values) - 1
    if shift_applied > 0:
        return exp_values - shift_applied
    return exp_values

train_preds_original = inv_log_transform(train_preds_log, shift_target)
train_trues_original = inv_log_transform(train_trues_log, shift_target)
test_preds_original = inv_log_transform(test_preds_log, shift_target)
test_trues_original = inv_log_transform(test_trues_log, shift_target)

print(f"\n=== TRANSFORMATION VERIFICATION ===")
print(f"Shift applied during log transform: {shift_target}")
print(f"Test predictions range: {test_preds_original.min():.2f} to {test_preds_original.max():.2f}")
print(f"Test actual range: {test_trues_original.min():.2f} to {test_trues_original.max():.2f}")

def metrics(trues, preds):
    mse = mean_squared_error(trues.flatten(), preds.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues.flatten(), preds.flatten())
    r2 = r2_score(trues.flatten(), preds.flatten())
    kge = kling_gupta_efficiency(trues.flatten(), preds.flatten())
    return r2, rmse, mae, mse, kge

r2_train, rmse_train, mae_train, mse_train, kge_train = metrics(train_trues_original, train_preds_original)
r2_test, rmse_test, mae_test, mse_test, kge_test = metrics(test_trues_original, test_preds_original)

print("\n=== FINAL METRICS (ORIGINAL SCALE) ===")
print("Train - R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}, MSE: {:.4f}, KGE: {:.4f}".format(r2_train, rmse_train, mae_train, mse_train, kge_train))
print("Test  - R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}, MSE: {:.4f}, KGE: {:.4f}".format(r2_test, rmse_test, mae_test, mse_test, kge_test))

per_h_r2, per_h_kge = [], []
for h in range(HORIZON):
    try:
        r = r2_score(test_trues_original[:, h], test_preds_original[:, h])
        k = kling_gupta_efficiency(test_trues_original[:, h], test_preds_original[:, h])
    except:
        r, k = np.nan, np.nan
    per_h_r2.append(r)
    per_h_kge.append(k)

# -----------------------
# Save results
# -----------------------
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
    'log_transformation_applied': True,
    'shift_applied_during_log_transform': float(shift_target),
    'performance_metrics': {
        'r2_train': float(r2_train), 'rmse_train': float(rmse_train), 'mae_train': float(mae_train),
        'r2_test': float(r2_test), 'rmse_test': float(rmse_test), 'mae_test': float(mae_test),
        'kge_train': float(kge_train), 'kge_test': float(kge_test)
    }
}
with open(os.path.join(SAVE_DIR, f'{model_name}_config.json'), 'w', encoding='utf-8') as f:
    json.dump(model_config, f, indent=4, ensure_ascii=False)

rows = []
for i in range(len(test_preds_original)):
    for h in range(HORIZON):
        date_val = pd.to_datetime(dates_test[i][h])
        rows.append({
            'sample_idx': i,
            'horizon': h+1,
            'date': date_val,
            'actual': float(test_trues_original[i, h]),
            'predicted': float(test_preds_original[i, h])
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
# Plotting
# -----------------------
import matplotlib.dates as mdates

fig, axes = plt.subplots(HORIZON, 3, figsize=(18, 4*HORIZON))
fig.suptitle(f"{target_col} - Future Prediction (Horizon {HORIZON} hours)", fontsize=18)

results_df['date_dt'] = pd.to_datetime(results_df['date'])

for h in range(HORIZON):
    # Column 1: Loss per epoch
    ax1 = axes[h, 0]
    train_h_losses = [per_h[h] for per_h in train_per_horizon]
    val_h_losses = [per_h[h] for per_h in val_per_horizon]
    epochs = list(range(1, len(train_h_losses) + 1))
    
    ax1.plot(epochs, train_h_losses, label='train_loss_per_h', marker='o', markersize=2)
    ax1.plot(epochs, val_h_losses, label='val_loss_per_h', marker='s', markersize=2)
    ax1.set_title(f'Prediction +{h+1}h - Loss per epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    if len(epochs) <= 20:
        ax1.set_xticks(epochs)
    else:
        ax1.set_xticks(epochs[::5])

    # Column 2: Scatter plot
    ax2 = axes[h, 1]
    y_true_h = test_trues_original[:, h]
    y_pred_h = test_preds_original[:, h]
    ax2.scatter(y_true_h, y_pred_h, s=8)
    ax2.set_title(f'Prediction +{h+1}h - Test scatter')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    
    r2_h = per_h_r2[h] if h < len(per_h_r2) else np.nan
    kge_h = per_h_kge[h] if h < len(per_h_kge) else np.nan
    ax2.text(0.02, 0.95, f'R2={r2_h:.3f}\nKGE={kge_h:.3f}', transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))
    
    lims = [min(np.nanmin(y_true_h), np.nanmin(y_pred_h)), max(np.nanmax(y_true_h), np.nanmax(y_pred_h))]
    ax2.plot(lims, lims, '--', linewidth=0.8)
    ax2.grid(True)

    # Column 3: Time series
    ax3 = axes[h, 2]
    dfh = results_df[results_df['horizon'] == (h+1)].copy()
    dfh = dfh.sort_values('date_dt')
    
    ax3.plot(dfh['date_dt'], dfh['actual'], label='Actual', linewidth=1.5)
    ax3.plot(dfh['date_dt'], dfh['predicted'], label='Predicted', linewidth=1.5, alpha=0.8)
    ax3.set_title(f'Prediction +{h+1}h - Future values')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Value')
    ax3.legend()
    
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

print("\n" + "="*50)
print("All saved to:", SAVE_DIR)
print("Model name:", model_name)
print(f"Model predicts {HORIZON} hours into the future using past {INPUT_WINDOW} hours of data")
print("="*50)