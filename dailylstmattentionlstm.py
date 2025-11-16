# -*- coding: utf-8 -*-
"""
ARN_LSTM_Predictor_DAILY_IMPROVED.py
Enhanced LSTM with Attention, Bidirectional, Residuals, and Robust Training
"""
import os
import random
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import skew  # <-- CORRECT skew import

# -----------------------
# Settings
# -----------------------
INPUT_WINDOW = 10
HORIZON = 5
TEST_SIZE = 60
EPOCHS = 100
SEED = 42
target_col = 'Veldan_PM2.5(ug/m3)'
CSV_PATH = r'C:\Users\arman\OneDrive\Desktop\AQIorgonized\gapfiledfinal.csv'
SAVE_DIR = r'D:\testNN'
os.makedirs(SAVE_DIR, exist_ok=True)
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPU Info
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# -----------------------
# Load and Preprocess Data
# -----------------------
df = pd.read_csv(CSV_PATH, parse_dates=[0], dayfirst=False)
if df.columns[0].lower() not in ['date', 'time', 'datetime']:
    df.rename({df.columns[0]: 'Date'}, axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
print(f"Original data shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# Fix duplicate columns
duplicate_cols = df.columns[df.columns.duplicated()].tolist()
if duplicate_cols:
    print(f"Warning: Found duplicate columns: {duplicate_cols}")
    df.columns = [f'{col}_{i}' if col in df.columns[:i] else col for i, col in enumerate(df.columns)]

# Convert to daily (max per day)
df['Date_Only'] = df['Date'].dt.date
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
daily_df = df[numeric_cols + ['Date_Only']].groupby('Date_Only').max().reset_index()
daily_df.rename(columns={'Date_Only': 'Date'}, inplace=True)
daily_df['Date'] = pd.to_datetime(daily_df['Date'])

# Add temporal features
daily_df['day_of_week'] = daily_df['Date'].dt.dayofweek
daily_df['day_of_month'] = daily_df['Date'].dt.day
daily_df['month'] = daily_df['Date'].dt.month
daily_df['is_weekend'] = (daily_df['Date'].dt.dayofweek >= 5).astype(int)
df = daily_df

print(f"Daily data shape: {df.shape}")

# Feature selection
try:
    feature_cols = df.columns[[16, 6, 22, 45, 28, 31, 37, 51, 43, 57, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76]].tolist()
except:
    print("Using fallback feature selection")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    feature_cols = numeric_cols[:20]

feature_cols.extend(['day_of_week', 'month', 'is_weekend'])
print("Selected feature columns:", feature_cols)

# Validate target
if target_col not in df.columns:
    target_candidates = [col for col in df.columns if 'Veldan' in col and 'PM2.5' in col]
    if target_candidates:
        target_col = target_candidates[0]
        print(f"Using alternative target: {target_col}")
    else:
        raise ValueError(f"Target column '{target_col}' not found")

df = df.loc[~df[target_col].isna()].reset_index(drop=True)
print(f"Final dataset shape: {df.shape}")

# -----------------------
# Prepare Sequences
# -----------------------
values_X = df[feature_cols].values.astype(float)
values_y = df[[target_col]].values.astype(float)
N = len(df)
seq_starts = list(range(0, N - INPUT_WINDOW - HORIZON + 1))
total_sequences = len(seq_starts)

if TEST_SIZE > total_sequences:
    TEST_SIZE = total_sequences // 3
    print(f"Adjusted TEST_SIZE to {TEST_SIZE}")

train_val_end = total_sequences - TEST_SIZE
train_val_indices = seq_starts[:train_val_end]
test_indices = seq_starts[train_val_end:]

val_frac = 0.15
val_count = max(1, int(len(train_val_indices) * val_frac))
train_indices = train_val_indices[:-val_count]
val_indices = train_val_indices[-val_count:]

print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

def build_X_y(indices):
    X, Y = [], []
    for i in indices:
        X.append(values_X[i:i+INPUT_WINDOW])
        Y.append(values_y[i+INPUT_WINDOW:i+INPUT_WINDOW+HORIZON].reshape(-1))
    return np.array(X), np.array(Y)

X_train_raw, y_train_raw = build_X_y(train_indices)
X_val_raw, y_val_raw = build_X_y(val_indices)
X_test_raw, y_test_raw = build_X_y(test_indices)
print("X_train shape:", X_train_raw.shape, "y_train shape:", y_train_raw.shape)

# -----------------------
# Scaling
# -----------------------
x_scaler = RobustScaler()
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
y_train_reshaped = y_train_raw.reshape(-1, 1)

# Check skewness
y_skewness = skew(y_train_reshaped.flatten())
if y_skewness > 1.0:
    print(f"Applying log transform (skew: {y_skewness:.2f})")
    y_train_reshaped = np.log1p(y_train_reshaped)

y_scaler.fit(y_train_reshaped)

def scale_y(y_raw):
    s = y_raw.reshape(-1, 1)
    if y_skewness > 1.0:
        s = np.log1p(s)
    s = y_scaler.transform(s)
    return s.reshape(y_raw.shape)

def inv_scale_y(y_scaled):
    s = y_scaled.reshape(-1, 1)
    inv = y_scaler.inverse_transform(s)
    if y_skewness > 1.0:
        inv = np.expm1(inv)
    return inv.reshape(y_scaled.shape)

y_train = scale_y(y_train_raw)
y_val = scale_y(y_val_raw)
y_test = scale_y(y_test_raw)

joblib.dump(x_scaler, os.path.join(SAVE_DIR, 'x_scaler_daily_improved.pkl'))
joblib.dump(y_scaler, os.path.join(SAVE_DIR, 'y_scaler_daily_improved.pkl'))

# -----------------------
# Dataset
# -----------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.augment = augment
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.augment and random.random() > 0.7:
            noise = np.random.normal(0, 0.01, x.shape).astype(np.float32)
            x = x + noise
        return x, y

train_ds = TimeSeriesDataset(X_train, y_train, augment=True)
val_ds = TimeSeriesDataset(X_val, y_val)
test_ds = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------
# Model
# -----------------------
input_size = X_train.shape[2]
hidden1, hidden2, hidden3 = 128, 64, 32
dropout_p = 0.4

class EnhancedSeqPredictor(nn.Module):
    def __init__(self, input_size, hidden1=hidden1, hidden2=hidden2, hidden3=hidden3, horizon=HORIZON, dropout=dropout_p):
        super().__init__()
        self.encoder1 = nn.LSTM(input_size, hidden1, num_layers=2, batch_first=True, dropout=dropout, bidirectional=True)
        self.encoder2 = nn.LSTM(hidden1*2, hidden2, num_layers=1, batch_first=True, dropout=dropout)
        self.encoder3 = nn.LSTM(hidden2, hidden3, num_layers=1, batch_first=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden3, num_heads=4, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden3 * 2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, horizon)
        )
        self.ln1 = nn.LayerNorm(hidden1*2)
        self.ln2 = nn.LayerNorm(hidden2)
        self.ln3 = nn.LayerNorm(hidden3)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:  # <-- FIXED: Only 2D+ for Xavier
                    if 'lstm' in name:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        out1, (hn1, cn1) = self.encoder1(x)
        out1 = self.ln1(out1)
        out2, (hn2, cn2) = self.encoder2(out1)
        out2 = self.ln2(out2 + out1[:, :, :hidden2])
        out3, (hn3, cn3) = self.encoder3(out2)
        out3 = self.ln3(out3 + out2[:, :, :hidden3])
        attn_out, _ = self.attention(out3, out3, out3)
        last_hidden = hn3[-1]
        attention_context = attn_out[:, -1, :]
        combined = torch.cat([last_hidden, attention_context], dim=1)
        return self.fc_layers(combined)

model = EnhancedSeqPredictor(input_size=input_size).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# -----------------------
# Loss & Optimizer
# -----------------------
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    def forward(self, p, t): return self.alpha * self.mse(p, t) + (1 - self.alpha) * self.mae(p, t)

criterion = CombinedLoss(alpha=0.7)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# -----------------------
# Training
# -----------------------
train_losses, val_losses = [], []
best_val_loss = float('inf')
patience, counter = 20, 0
min_delta = 1e-4

print("Starting training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    train_loss = np.mean(losses)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss_epoch = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            val_loss_epoch.append(criterion(preds, yb).item())
    val_loss = np.mean(val_loss_epoch)
    val_losses.append(val_loss)
    scheduler.step()

    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
    else:
        counter += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e} | Patience: {counter}/{patience}")

    if counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth')))
        break

# -----------------------
# Evaluation
# -----------------------
def predict(loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            out = model(xb).cpu().numpy()
            preds.append(out)
            trues.append(yb.numpy())
    return np.vstack(preds), np.vstack(trues)

train_preds_s, train_trues_s = predict(train_loader)
test_preds_s, test_trues_s = predict(test_loader)

train_preds = inv_scale_y(train_preds_s)
train_trues = inv_scale_y(train_trues_s)
test_preds = inv_scale_y(test_preds_s)
test_trues = inv_scale_y(test_trues_s)

def metrics(t, p):
    mse = mean_squared_error(t.flatten(), p.flatten())
    return r2_score(t.flatten(), p.flatten()), np.sqrt(mse), mean_absolute_error(t.flatten(), p.flatten()), mse

r2_train, rmse_train, mae_train, _ = metrics(train_trues, train_preds)
r2_test, rmse_test, mae_test, _ = metrics(test_trues, test_preds)

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Train → R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
print(f"Test  → R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")
print("="*50)

# Save final model
torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'final_model.pth'))
print(f"Model saved to {SAVE_DIR}")