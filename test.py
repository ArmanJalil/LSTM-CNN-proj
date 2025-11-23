
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
HORIZON = 3  # Only predict step 3 (directly)
TEST_SIZE = 240
EPOCHS = 100
SEED = 42
target_col = '25Aban_PM2.5(ug/m3)'

CSV_PATH = r'C:\Users\arman\OneDrive\Desktop\AQIorgonized\gapfiledfinal.csv'
SAVE_DIR = r'D:\testNN'
os.makedirs(SAVE_DIR, exist_ok=True)

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
# KGE metric function
# -----------------------
def kling_gupta_efficiency(obs, sim):
    """
    Calculate Kling-Gupta Efficiency (KGE)
    obs: observed values
    sim: simulated/predicted values
    """
    # Remove NaN values
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[mask]
    sim = sim[mask]
    
    if len(obs) == 0:
        return np.nan
    
    # Calculate components
    r = np.corrcoef(obs, sim)[0, 1]  # correlation
    alpha = np.std(sim) / np.std(obs)  # variability ratio
    beta = np.mean(sim) / np.mean(obs)  # bias ratio
    
    # KGE formula
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
    # for deterministic (may slow)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# -----------------------
# Load data and apply logarithmic transformation
# -----------------------
df = pd.read_csv(CSV_PATH, parse_dates=[0], dayfirst=False)  # first column is Date
# Ensure first column named Date
if df.columns[0].lower() not in ['date', 'time', 'datetime']:
    df.rename({df.columns[0]: 'Date'}, axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

# build feature_cols from given indices (user provided)
try:
    feature_cols = df.columns[[16, 6, 22, 45, 28,57,31,66,67,68,69,70,71,72,73]].tolist()
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

# Apply logarithmic transformation to all features except temperature and dew point
print("Applying logarithmic transformation to features...")
for col in feature_cols:
    # Skip temperature and dew point columns (check for common names)
    if any(keyword in col.lower() for keyword in ['temp', 'temperature', 'dew', 'dewpoint']):
        print(f"Skipping log transform for: {col}")
        continue
    
    # Apply log(1 + x) transformation to handle zeros and negative values
    if col in df.columns:
        min_val = df[col].min()
        if min_val <= 0:
            # Shift data to make all values positive before log transform
            shift = abs(min_val) + 1
            df[col] = np.log1p(df[col] + shift)
            print(f"Applied log(1+x+{shift}) to {col} (had negative values)")
        else:
            df[col] = np.log1p(df[col])
            print(f"Applied log(1+x) to {col}")

# Apply log transformation to target variable (PM2.5)
print(f"Applying log transformation to target: {target_col}")
min_target = df[target_col].min()
if min_target <= 0:
    shift = abs(min_target) + 1
    df[target_col] = np.log1p(df[target_col] + shift)
    print(f"Applied log(1+x+{shift}) to target")
else:
    df[target_col] = np.log1p(df[target_col])
    print("Applied log(1+x) to target")

# -----------------------
# Prepare sequences (no shuffling) - MODIFIED FOR SINGLE STEP PREDICTION
# -----------------------
values_X = df[feature_cols].values.astype(float)
values_y = df[[target_col]].values.astype(float)  # shape (N,1)

N = len(df)
print("Total rows:", N)

# We'll build sequences: for i in range(0, N - INPUT_WINDOW - HORIZON + 1)
# Each sequence X = values_X[i : i+INPUT_WINDOW], y = values_y[i+INPUT_WINDOW+HORIZON-1] (single value for step 3)
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

# Build datasets arrays - MODIFIED FOR SINGLE STEP PREDICTION
def build_X_y(indices):
    X = []
    Y = []
    dates = []
    for i in indices:
        x = values_X[i : i+INPUT_WINDOW]
        # Only take the HORIZON step (step 3) - single value
        y = values_y[i+INPUT_WINDOW+HORIZON-1][0]  # Single value for step 3
        X.append(x)
        Y.append(y)
        # Store only the date for step 3 we're predicting
        dates.append(df['Date'].iloc[i+INPUT_WINDOW+HORIZON-1])
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
# PyTorch Dataset - MODIFIED FOR SINGLE OUTPUT
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
# Model (PyTorch) - MODIFIED FOR SINGLE OUTPUT
# -----------------------
input_size = X_train.shape[2]
hidden1 = 64
hidden2 = 64
hidden3 = 32
dropout_p = 0.4

class SeqPredictor(nn.Module):
    def __init__(self, input_size, hidden1=hidden1, hidden2=hidden2, hidden3=hidden3, dropout=dropout_p):
        super().__init__()
        self.encoder1 = nn.LSTM(input_size=input_size, hidden_size=hidden1, num_layers=1, batch_first=True)
        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)
        self.encoder2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, num_layers=1, batch_first=True)
        self.relu1 = nn.ReLU()
        self.encoder3 = nn.LSTM(input_size=hidden2, hidden_size=hidden3, num_layers=1, batch_first=True)
        self.tanh2 = nn.Tanh()
        # final linear from hidden3 to SINGLE output
        self.fc = nn.Linear(hidden3, 1)  # Single output for step 3
        # small weight init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, x):
        # x: batch, seq_len, features
        out, _ = self.encoder1(x)
        out = self.tanh1(out)
        out = self.dropout1(out)
        out, _ = self.encoder2(out)
        out = self.relu1(out)
        out, (hn, cn) = self.encoder3(out)
        out = self.tanh2(out)
        # hn: (num_layers, batch, hidden3) -> take last layer
        last_h = hn[-1]  # shape (batch, hidden3)
        res = self.fc(last_h)  # (batch, 1)
        return res.squeeze(-1)  # (batch,) - single output

model = SeqPredictor(input_size=input_size, hidden1=hidden1, hidden2=hidden2, hidden3=hidden3, dropout=dropout_p)
model = model.to(DEVICE)

# Loss and optimizer
LR = 1e-4
WEIGHT_DECAY = 1e-5
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)

# -----------------------
# Training loop with early stopping - MODIFIED FOR SINGLE OUTPUT
# -----------------------
train_losses = []
val_losses = []

# Early stopping parameters
patience = 10
min_val_loss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_losses = []
    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)  # shape (batch,) - single value
        optimizer.zero_grad()
        preds = model(xb)  # (batch,) - single output
        loss = criterion(preds, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_losses.append(loss.item())
    train_epoch_loss = np.mean(epoch_losses)
    train_losses.append(train_epoch_loss)

    # compute validation loss
    model.eval()
    val_epoch_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_epoch_losses.append(loss.item())

    val_epoch_loss = np.mean(val_epoch_losses) if val_epoch_losses else np.nan
    val_losses.append(val_epoch_loss)

    scheduler.step(val_epoch_loss if not np.isnan(val_epoch_loss) else train_epoch_loss)

    # Early stopping logic
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
# Evaluate on train and test - MODIFIED FOR SINGLE OUTPUT
# -----------------------
def predict_on_loader(loader):
    model.eval()
    preds_list = []
    trues_list = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            out = model(xb).cpu().numpy()  # (batch,) - single output
            preds_list.append(out)
            trues_list.append(yb.numpy())
    preds = np.concatenate(preds_list)
    trues = np.concatenate(trues_list)
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

# Apply inverse log transformation
def inv_log_transform(log_values, original_min=None):
    exp_values = np.expm1(log_values)
    if original_min is not None and original_min <= 0:
        shift = abs(original_min) + 1
        return exp_values - shift
    return exp_values

original_target_min = min_target

# Convert predictions and true values back to original scale
train_preds_original = inv_log_transform(train_preds, original_target_min)
train_trues_original = inv_log_transform(train_trues, original_target_min)
test_preds_original = inv_log_transform(test_preds, original_target_min)
test_trues_original = inv_log_transform(test_trues, original_target_min)

# compute metrics - FIXED FOR 1D ARRAYS
def metrics(trues, preds):
    # Ensure 1D arrays
    trues = trues.flatten() if trues.ndim > 1 else trues
    preds = preds.flatten() if preds.ndim > 1 else preds
    
    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    kge = kling_gupta_efficiency(trues, preds)
    return r2, rmse, mae, mse, kge

r2_train, rmse_train, mae_train, mse_train, kge_train = metrics(train_trues_original, train_preds_original)
r2_test, rmse_test, mae_test, mse_test, kge_test = metrics(test_trues_original, test_preds_original)

print("Train - R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}, MSE: {:.4f}, KGE: {:.4f}".format(r2_train, rmse_train, mae_train, mse_train, kge_train))
print("Test  - R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}, MSE: {:.4f}, KGE: {:.4f}".format(r2_test, rmse_test, mae_test, mse_test, kge_test))

# -----------------------
# Save results - MODIFIED FOR SINGLE HORIZON
# -----------------------
import json
model_name = f"predictor_{re.sub(r'[\\/*?:\"<>|().]', '_', target_col)}_step{HORIZON}_direct"
torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{model_name}.pth'))
joblib.dump(x_scaler, os.path.join(SAVE_DIR, f'{model_name}_x_scaler.pkl'))
joblib.dump(y_scaler, os.path.join(SAVE_DIR, f'{model_name}_y_scaler.pkl'))

model_config = {
    'target_column': target_col,
    'horizon': HORIZON,
    'input_window': INPUT_WINDOW,
    'input_size': input_size,
    'feature_columns': feature_cols,
    'model_architecture': 'SeqPredictor_LSTM_3layer_SingleOutput',
    'training_parameters': {
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'epochs': EPOCHS
    },
    'log_transformation_applied': True,
    'original_target_min': float(original_target_min),
    'performance_metrics': {
        'r2_train': float(r2_train),
        'rmse_train': float(rmse_train),
        'mae_train': float(mae_train),
        'mse_train': float(mse_train),
        'kge_train': float(kge_train),
        'r2_test': float(r2_test),
        'rmse_test': float(rmse_test),
        'mae_test': float(mae_test),
        'mse_test': float(mse_test),
        'kge_test': float(kge_test),
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

# Save predictions
results_df = pd.DataFrame({
    'date': dates_test,
    'actual': test_trues_original,
    'predicted': test_preds_original
})
results_df.to_csv(os.path.join(SAVE_DIR, f'{model_name}_predictions.csv'), index=False)

loss_history_df = pd.DataFrame({
    'epoch': range(1, len(train_losses) + 1),
    'train_loss': train_losses,
    'val_loss': val_losses
})
loss_history_df.to_csv(os.path.join(SAVE_DIR, f'{model_name}_loss_history.csv'), index=False)

# -----------------------
# Simplified Plotting for Single Step
# -----------------------
# -----------------------
# Simplified Plotting for Single Step
# -----------------------
import matplotlib.dates as mdates

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"{target_col} - Direct Prediction (Step {HORIZON}) - Log Transformed", fontsize=16)

# Plot 1: Loss history
axes[0].plot(loss_history_df['epoch'], loss_history_df['train_loss'], label='Train Loss')
axes[0].plot(loss_history_df['epoch'], loss_history_df['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training History')
axes[0].legend()
axes[0].grid(True)

# Plot 2: Scatter plot
axes[1].scatter(test_trues_original, test_preds_original, s=20, alpha=0.6)
axes[1].set_xlabel('Actual')
axes[1].set_ylabel('Predicted')
axes[1].set_title(f'Step {HORIZON} - Test Scatter')
axes[1].text(0.02, 0.95, f'R2={r2_test:.3f}\nKGE={kge_test:.3f}', transform=axes[1].transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))
lims = [min(np.nanmin(test_trues_original), np.nanmin(test_preds_original)), 
        max(np.nanmax(test_trues_original), np.nanmax(test_preds_original))]
axes[1].plot(lims, lims, '--', linewidth=1, color='red')
axes[1].grid(True)

# Plot 3: Time series with proper date formatting
results_df['date'] = pd.to_datetime(results_df['date'])
results_df = results_df.sort_values('date')

# Plot the data
axes[2].plot(results_df['date'], results_df['actual'], label='Actual', linewidth=1.5)
axes[2].plot(results_df['date'], results_df['predicted'], label='Predicted', linewidth=1.5, alpha=0.8)
axes[2].set_xlabel('Date')
axes[2].set_ylabel('PM2.5 (μg/m³)')
axes[2].set_title(f'Step {HORIZON} - Time Series')
axes[2].legend()

# Get the date range and set ticks at beginning of each day
start_date = results_df['date'].min().normalize()  # Start at 00:00
end_date = results_df['date'].max().normalize()    # End at 00:00

# Create daily ticks from start to end date
date_ticks = pd.date_range(start=start_date, end=end_date, freq='D')

# Limit number of ticks to avoid overcrowding
if len(date_ticks) > 15:
    # Show every N days to have reasonable number of ticks
    step = max(1, len(date_ticks) // 10)
    date_ticks = date_ticks[::step]

axes[2].set_xticks(date_ticks)
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')
axes[2].grid(True)

# Add vertical lines at beginning of each day
for tick in date_ticks:
    axes[2].axvline(x=tick, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

# Set x-axis limits to show full range
axes[2].set_xlim([start_date, end_date])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot_path = os.path.join(SAVE_DIR, f'{model_name}_summary_plots.png')
plt.savefig(plot_path, bbox_inches='tight', dpi=200)
plt.show(fig)

print("All saved to:", SAVE_DIR)
print("Model name:", model_name)
print(f"Model directly predicts step {HORIZON} using past {INPUT_WINDOW} hours of data")
print("Note: Data was log-transformed for training and converted back to original scale for evaluation")