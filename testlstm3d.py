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

# ============ USER CONFIG ============
INPUT_FILE_X = r'C:\Users\arman\OneDrive\Desktop\AQIorgonized\merged_pollution_weather_with_time.csv'
INPUT_FILE_Y = r'C:\Users\arman\OneDrive\Desktop\AQIorgonized\gapfiledfinal.csv'
INPUT_WINDOW = 48
HORIZON = 12
TEST_SIZE = 1000
VAL_SIZE = 1000  # Added validation size
BATCH_SIZE = 64
EPOCHS = 60
LR = 0.001
SEED = 42

# reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')

# ============ LOAD & ALIGN ============
df_X = pd.read_csv(INPUT_FILE_X, parse_dates=['Date'])
df_Y = pd.read_csv(INPUT_FILE_Y, parse_dates=['Date'])
df = pd.merge(df_X, df_Y, on='Date', how='inner', suffixes=('_X',''))

# targets
target_cols = ['25Aban_CO (ppm)','25Aban_NO (ppb)','25Aban_NO2 (ppb)',
               '25Aban_NOx (ppb)','25Aban_O3 (ppb)','25Aban_PM2.5  (ug/m3)',
               '25Aban_SO2 (ppb)']
target_cols = [c for c in target_cols if c in df.columns]
if len(target_cols) < 7:
    pollution_like = [col for col in df.columns if any(x in col for x in ['CO','NO','NO2','NOx','O3','PM2.5','PM10','SO2'])]
    target_cols = pollution_like[:7]
feature_cols = [c for c in df.columns if c not in ['Date'] + target_cols]

print(f"Targets: {target_cols}")
print(f"Features: {len(feature_cols)}")

# X: fill gaps with -99999
X_raw = df[feature_cols].values.astype(np.float32)
X_raw[np.isnan(X_raw)] = -99999.0
X_mask = (X_raw != -99999.0).astype(np.float32)

# FIXED: Properly store X scalers
x_scalers = []
X_scaled = np.zeros_like(X_raw, dtype=np.float32)
for i in range(X_raw.shape[1]):
    col = X_raw[:, i]
    valid = X_mask[:, i].astype(bool)
    s = StandardScaler()
    if valid.sum() > 0:
        s.fit(col[valid].reshape(-1,1))
        X_scaled[valid, i] = s.transform(col[valid].reshape(-1,1)).flatten()
        X_scaled[~valid, i] = 0.0
    else:
        X_scaled[:, i] = 0.0
    x_scalers.append(s)  # CORRECT: Store each scaler

# Y: use raw values
Y_raw = df[target_cols].values.astype(np.float32)
Y_raw[np.isnan(Y_raw)] = 0.0

# augment X with gap-mask features
X_aug = np.concatenate([X_scaled, X_mask], axis=1)

# ============ SEQUENCE CREATION ============
def create_multi_horizon(X, Y, input_window, horizon):
    Xs, Ys = [], []
    T = len(X)
    for i in range(T - input_window - horizon + 1):
        Xs.append(X[i:i+input_window])
        Ys.append(Y[i+input_window:i+input_window+horizon])
    return np.array(Xs), np.array(Ys)

X_seq, Y_seq = create_multi_horizon(X_aug, Y_raw, INPUT_WINDOW, HORIZON)
if X_seq.shape[0] == 0:
    raise ValueError('No sequences created — reduce INPUT_WINDOW/HORIZON or check data length.')

print(f"Total sequences: {X_seq.shape[0]}")

# FIXED: Proper data split with validation
total_sequences = X_seq.shape[0]
required_total = TEST_SIZE + VAL_SIZE

if total_sequences > required_total:
    # Enough data: use requested sizes
    X_train = X_seq[:-(TEST_SIZE + VAL_SIZE)]
    Y_train = Y_seq[:-(TEST_SIZE + VAL_SIZE)]
    X_val = X_seq[-(TEST_SIZE + VAL_SIZE):-TEST_SIZE]
    Y_val = Y_seq[-(TEST_SIZE + VAL_SIZE):-TEST_SIZE]
    X_test = X_seq[-TEST_SIZE:]
    Y_test = Y_seq[-TEST_SIZE:]
    print(f"Data split: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test")
else:
    # Not enough data: adjust sizes proportionally
    test_ratio = TEST_SIZE / required_total
    val_ratio = VAL_SIZE / required_total
    train_ratio = 1.0 - test_ratio - val_ratio
    
    train_size = int(total_sequences * train_ratio)
    val_size = int(total_sequences * val_ratio)
    test_size = total_sequences - train_size - val_size
    
    X_train, X_val, X_test = X_seq[:train_size], X_seq[train_size:train_size+val_size], X_seq[train_size+val_size:]
    Y_train, Y_val, Y_test = Y_seq[:train_size], Y_seq[train_size:train_size+val_size], Y_seq[train_size+val_size:]
    
    print(f'Adjusted split: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test')

# FIXED: Proper Y scaling with stored scalers
y_scalers = []
Y_train_scaled = np.zeros_like(Y_train, dtype=np.float32)
Y_val_scaled = np.zeros_like(Y_val, dtype=np.float32)
Y_test_scaled = np.zeros_like(Y_test, dtype=np.float32)

for j in range(Y_train.shape[2]):
    s = StandardScaler()
    train_data = Y_train[:, :, j].reshape(-1, 1)
    s.fit(train_data)
    
    Y_train_scaled[:, :, j] = s.transform(train_data).reshape(Y_train.shape[0], Y_train.shape[1])
    Y_val_scaled[:, :, j] = s.transform(Y_val[:, :, j].reshape(-1, 1)).reshape(Y_val.shape[0], Y_val.shape[1])
    Y_test_scaled[:, :, j] = s.transform(Y_test[:, :, j].reshape(-1, 1)).reshape(Y_test.shape[0], Y_test.shape[1])
    y_scalers.append(s)  # CORRECT: Store each scaler

# ============ DATASET & MODEL ============
class MultiDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_loader = DataLoader(MultiDataset(X_train, Y_train_scaled), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(MultiDataset(X_val, Y_val_scaled), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(MultiDataset(X_test, Y_test_scaled), batch_size=BATCH_SIZE, shuffle=False)

# NEW: Your requested LSTM architecture in PyTorch
class SequentialLSTM(nn.Module):
    def __init__(self, input_size, output_size, horizon):
        super().__init__()
        self.horizon = horizon
        self.output_size = output_size
        
        # PyTorch LSTM: return_sequences is always True, we control output via forward()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.dense = nn.Linear(32, horizon * output_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM1: return_sequences=True equivalent
        out, (h_n1, c_n1) = self.lstm1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        
        # LSTM2: return_sequences=True equivalent  
        out, (h_n2, c_n2) = self.lstm2(out)
        out = self.relu(out)
        
        # LSTM3: return_sequences=False equivalent (only last output)
        out, (h_n3, c_n3) = self.lstm3(out)
        out = self.relu(out[:, -1, :])  # Take only the last timestep
        
        out = self.dense(out)
        out = out.view(-1, self.horizon, self.output_size)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SequentialLSTM(X_train.shape[2], len(target_cols), HORIZON).to(device)

print(f"Model: input={X_train.shape[2]}, output={len(target_cols)}, horizon={HORIZON}")

# Using MSELoss as equivalent to mean_squared_logarithmic_error for positive values
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)  # L2 regularization

# ============ TRAIN WITH PER-VARIABLE LOSS TRACKING ============
train_losses_per_var = {col: [] for col in target_cols}
val_losses_per_var = {col: [] for col in target_cols}
overall_train_losses = []
overall_val_losses = []

print("Starting training...")

for epoch in range(1, EPOCHS+1):
    # Training
    model.train()
    t_loss_total = 0.0
    t_loss_per_var = {col: 0.0 for col in target_cols}
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        
        # Calculate overall loss
        total_loss = criterion(pred, yb)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss_total += total_loss.item()
        
        # Calculate per-variable losses
        with torch.no_grad():
            for i, col in enumerate(target_cols):
                var_loss = criterion(pred[:, :, i], yb[:, :, i])
                t_loss_per_var[col] += var_loss.item()
    
    # Store losses
    overall_train_loss = t_loss_total / len(train_loader)
    overall_train_losses.append(overall_train_loss)
    
    for col in target_cols:
        avg_loss = t_loss_per_var[col] / len(train_loader)
        train_losses_per_var[col].append(avg_loss)

    # Validation
    model.eval()
    v_loss_total = 0.0
    v_loss_per_var = {col: 0.0 for col in target_cols}
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            
            total_loss = criterion(pred, yb)
            v_loss_total += total_loss.item()
            
            for i, col in enumerate(target_cols):
                var_loss = criterion(pred[:, :, i], yb[:, :, i])
                v_loss_per_var[col] += var_loss.item()
    
    overall_val_loss = v_loss_total / len(val_loader)
    overall_val_losses.append(overall_val_loss)
    
    for col in target_cols:
        avg_loss = v_loss_per_var[col] / len(val_loader)
        val_losses_per_var[col].append(avg_loss)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch:3d}/{EPOCHS} | Train: {overall_train_loss:.6f} | Val: {overall_val_loss:.6f}')

# ============ EVALUATE ============
model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out = model(xb).cpu().numpy()
        preds.append(out)
        trues.append(yb.numpy())

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)

# FIXED: Proper inverse transform using stored scalers
preds_inv = np.zeros_like(preds)
trues_inv = np.zeros_like(trues)
for k in range(len(target_cols)):
    s = y_scalers[k]  # CORRECT: Use the stored scaler for this variable
    preds_inv[:, :, k] = s.inverse_transform(preds[:, :, k].reshape(-1, 1)).reshape(preds.shape[0], preds.shape[1])
    trues_inv[:, :, k] = s.inverse_transform(trues[:, :, k].reshape(-1, 1)).reshape(trues.shape[0], trues.shape[1])

# ============ SAVE RESULTS ============
SAVE_DIR = os.path.dirname(INPUT_FILE_X)

# Save predictions
for k, col in enumerate(target_cols):
    clean_col_name = col.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    cols = [f'{col}_h{h+1}' for h in range(HORIZON)]
    pd.DataFrame(preds_inv[:, :, k], columns=cols).to_csv(os.path.join(SAVE_DIR, f'predictions_{clean_col_name}.csv'), index=False)

# Calculate R²
horizon_r2 = []
for h in range(HORIZON):
    r2s = []
    for k in range(len(target_cols)):
        y_true = trues_inv[:, h, k]
        y_pred = preds_inv[:, h, k]
        r2 = r2_score(y_true, y_pred)
        r2s.append(r2)
    horizon_r2.append(r2s)
    print(f"Horizon {h+1}: {np.mean(r2s):.4f} (avg)")

# ============ PLOTTING WITH PER-VARIABLE LOSSES ============
for k, col in enumerate(target_cols):
    clean_col_name = col.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: R² scatter plot
    ax1 = plt.subplot(2, 2, 1)
    y_true = trues_inv[:, 0, k]
    y_pred = preds_inv[:, 0, k]
    ax1.scatter(y_true, y_pred, s=8, alpha=0.5)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax1.plot([mn, mx], [mn, mx], 'r--')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'{col}  R2(h1)={horizon_r2[0][k]:.3f}')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: PER-VARIABLE Loss curves
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(train_losses_per_var[col], label='Train Loss', linewidth=2)
    ax2.plot(val_losses_per_var[col], label='Validation Loss', linewidth=2)
    ax2.legend()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Training History - {col}')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time series comparison
    ax3 = plt.subplot(2, 1, 2)
    idx = np.arange(len(trues_inv))
    ax3.plot(idx, trues_inv[:, 0, k], label='Actual (h+1)', linewidth=1)
    ax3.plot(idx, preds_inv[:, 0, k], label='Predicted (h+1)', linewidth=1)
    ax3.set_xlabel('Time Sequence')
    ax3.set_ylabel('Value')
    ax3.set_title(f'Last {len(idx)} test sequences (horizon=1)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'{clean_col_name}_summary.png'), dpi=150)
    plt.close()

# Save results
r2_summary = pd.DataFrame({
    'Variable': target_cols,
    **{f'Horizon_{h+1}': [horizon_r2[h][k] for k in range(len(target_cols))] for h in range(HORIZON)}
})
r2_summary.to_csv(os.path.join(SAVE_DIR, 'r2_summary_all_horizons.csv'), index=False)

# Save loss history
loss_history = pd.DataFrame({
    'epoch': range(1, EPOCHS + 1),
    **{f'train_loss_{col}': train_losses_per_var[col] for col in target_cols},
    **{f'val_loss_{col}': val_losses_per_var[col] for col in target_cols}
})
loss_history.to_csv(os.path.join(SAVE_DIR, 'per_variable_loss_history.csv'), index=False)

print(f'\nAll results saved to: {SAVE_DIR}')
print('Done!')