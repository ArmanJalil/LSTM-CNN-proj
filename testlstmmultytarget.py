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
HORIZON = 1
TEST_SIZE = 1000
VAL_SIZE = 1000
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

# FIXED: Proper scaler storage with gap masking
x_scalers = []
X_scaled = np.zeros_like(X_raw, dtype=np.float32)
for i in range(X_raw.shape[1]):
    col = X_raw[:, i]
    valid = X_mask[:, i].astype(bool)
    s = StandardScaler()
    if valid.sum() > 0:
        # FIXED: Only fit on valid data (exclude -99999)
        valid_data = col[valid].reshape(-1, 1)
        s.fit(valid_data)
        X_scaled[valid, i] = s.transform(valid_data).flatten()
        X_scaled[~valid, i] = 0.0  # Use 0 for gaps after scaling
    else:
        X_scaled[:, i] = 0.0
    x_scalers.append(s)

# Y: use raw values
Y_raw = df[target_cols].values.astype(np.float32)
Y_raw[np.isnan(Y_raw)] = 0.0

# augment X with gap-mask features
X_aug = np.concatenate([X_scaled, X_mask], axis=1)

# ============ SEQUENCE CREATION ============
def create_single_horizon(X, Y, input_window):
    Xs, Ys = [], []
    T = len(X)
    for i in range(T - input_window):
        Xs.append(X[i:i+input_window])
        Ys.append(Y[i+input_window])
    return np.array(Xs), np.array(Ys)

X_seq, Y_seq = create_single_horizon(X_aug, Y_raw, INPUT_WINDOW)
if X_seq.shape[0] == 0:
    raise ValueError('No sequences created — reduce INPUT_WINDOW or check data length.')

print(f"Total sequences: {X_seq.shape[0]}")

# FIXED: Robust data split
total_sequences = X_seq.shape[0]
required_total = TEST_SIZE + VAL_SIZE

if total_sequences > required_total:
    X_train = X_seq[:-(TEST_SIZE + VAL_SIZE)]
    Y_train = Y_seq[:-(TEST_SIZE + VAL_SIZE)]
    X_val = X_seq[-(TEST_SIZE + VAL_SIZE):-TEST_SIZE]
    Y_val = Y_seq[-(TEST_SIZE + VAL_SIZE):-TEST_SIZE]
    X_test = X_seq[-TEST_SIZE:]
    Y_test = Y_seq[-TEST_SIZE:]
    print(f"Data split: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test")
else:
    min_train_size = max(100, int(0.5 * total_sequences))
    min_val_size = max(50, int(0.25 * total_sequences))
    test_size = total_sequences - min_train_size - min_val_size
    
    if test_size < 50:
        min_train_size = int(0.6 * total_sequences)
        min_val_size = int(0.2 * total_sequences)
        test_size = total_sequences - min_train_size - min_val_size
    
    X_train, X_val, X_test = X_seq[:min_train_size], X_seq[min_train_size:min_train_size+min_val_size], X_seq[min_train_size+min_val_size:]
    Y_train, Y_val, Y_test = Y_seq[:min_train_size], Y_seq[min_train_size:min_train_size+min_val_size], Y_seq[min_train_size+min_val_size:]
    
    print(f'Adjusted split: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test')

# FIXED: Proper Y scaling with individual scalers for each variable
y_scalers = []  # This will store separate scalers for each target variable
Y_train_scaled = np.zeros_like(Y_train, dtype=np.float32)
Y_val_scaled = np.zeros_like(Y_val, dtype=np.float32)
Y_test_scaled = np.zeros_like(Y_test, dtype=np.float32)

for j in range(Y_train.shape[1]):  # For each target variable
    s = StandardScaler()
    # Fit on training data only for this specific variable
    train_data = Y_train[:, j].reshape(-1, 1)
    s.fit(train_data)
    
    # Transform all splits using this variable's scaler
    Y_train_scaled[:, j] = s.transform(train_data).flatten()
    Y_val_scaled[:, j] = s.transform(Y_val[:, j].reshape(-1, 1)).flatten()
    Y_test_scaled[:, j] = s.transform(Y_test[:, j].reshape(-1, 1)).flatten()
    
    # FIXED: Store each scaler individually
    y_scalers.append(s)

# ============ DATASETS & DATA LOADERS ============
class SingleDataset(Dataset):
    def __init__(self, X, Y):
        # Replace -99999 with 0 for model input (gaps become 0)
        X_processed = X.copy()
        X_processed[X_processed == -99999.0] = 0.0
        
        self.X = torch.tensor(X_processed, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_loader = DataLoader(SingleDataset(X_train, Y_train_scaled), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SingleDataset(X_val, Y_val_scaled), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(SingleDataset(X_test, Y_test_scaled), batch_size=BATCH_SIZE, shuffle=False)

# ============ MODEL ============
class SequentialLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.output_size = output_size
        
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.batchnorm2 = nn.BatchNorm1d(32)
        
        self.lstm4 = nn.LSTM(32, 64, batch_first=True)
        self.batchnorm3 = nn.BatchNorm1d(64)
        
        self.dense = nn.Linear(64, output_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out, _ = self.lstm2(out)
        out = self.relu(out)
        out = out[:, -1, :]
        out = self.batchnorm1(out)
        out = out.unsqueeze(1)
        out = self.dropout2(out)
        
        out, _ = self.lstm3(out)
        out = self.relu(out[:, -1, :])
        out = self.batchnorm2(out)
        
        out = out.unsqueeze(1)
        out, _ = self.lstm4(out)
        out = out[:, -1, :]
        out = self.batchnorm3(out)
        
        out = self.dense(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SequentialLSTM(X_train.shape[2], len(target_cols)).to(device)

print(f"Model: input={X_train.shape[2]}, output={len(target_cols)}")

criterion = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# ============ TRAINING WITH PROPER VALIDATION LOSS ============
train_losses_per_var = {col: [] for col in target_cols}
val_losses_per_var = {col: [] for col in target_cols}
overall_train_losses = []
overall_val_losses = []

print("Starting training...")
print("Epoch | Overall Train | Overall Val | Variable Losses")
print("-" * 60)

for epoch in range(1, EPOCHS+1):
    # Training
    model.train()
    t_loss_total = 0.0
    t_loss_per_var = {col: 0.0 for col in target_cols}
    t_count_per_var = {col: 0 for col in target_cols}
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        
        loss_per_element = criterion(pred, yb)
        loss_per_var = loss_per_element.mean(dim=0)
        total_loss = loss_per_element.mean()
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        t_loss_total += total_loss.item()
        for i, col in enumerate(target_cols):
            t_loss_per_var[col] += loss_per_var[i].item()
            t_count_per_var[col] += 1
    
    overall_train_loss = t_loss_total / len(train_loader)
    overall_train_losses.append(overall_train_loss)
    
    for col in target_cols:
        avg_loss = t_loss_per_var[col] / max(1, t_count_per_var[col])
        train_losses_per_var[col].append(avg_loss)

    # Validation - FIXED: Using proper validation data
    model.eval()
    v_loss_total = 0.0
    v_loss_per_var = {col: 0.0 for col in target_cols}
    v_count_per_var = {col: 0 for col in target_cols}
    
    with torch.no_grad():
        for xb, yb in val_loader:  # Using VALIDATION loader, not test
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            
            loss_per_element = criterion(pred, yb)
            loss_per_var = loss_per_element.mean(dim=0)
            total_loss = loss_per_element.mean()
            
            v_loss_total += total_loss.item()
            for i, col in enumerate(target_cols):
                v_loss_per_var[col] += loss_per_var[i].item()
                v_count_per_var[col] += 1
    
    overall_val_loss = v_loss_total / len(val_loader)
    overall_val_losses.append(overall_val_loss)
    
    for col in target_cols:
        avg_loss = v_loss_per_var[col] / max(1, v_count_per_var[col])
        val_losses_per_var[col].append(avg_loss)
    
    scheduler.step(overall_val_loss)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch:3d}/{EPOCHS} | Train: {overall_train_loss:.6f} | Val: {overall_val_loss:.6f}')
        for col in target_cols:
            train_loss = train_losses_per_var[col][-1]
            val_loss = val_losses_per_var[col][-1]
            print(f'         {col:25} | Train: {train_loss:.6f} | Val: {val_loss:.6f}')

# ============ FINAL EVALUATION ============
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

# FIXED: Proper inverse transform using individual scalers
preds_inv = np.zeros_like(preds)
trues_inv = np.zeros_like(trues)
for k in range(len(target_cols)):
    s = y_scalers[k]  # Use the specific scaler for this variable
    preds_inv[:, k] = s.inverse_transform(preds[:, k].reshape(-1, 1)).flatten()
    trues_inv[:, k] = s.inverse_transform(trues[:, k].reshape(-1, 1)).flatten()

# ============ RESULTS ============
SAVE_DIR = r'C:\Users\arman\OneDrive\Desktop\AQIorgonized\test results'
os.makedirs(SAVE_DIR, exist_ok=True)

# Calculate R² for each variable
r2_scores = []
for k in range(len(target_cols)):
    y_true = trues_inv[:, k]
    y_pred = preds_inv[:, k]
    r2 = r2_score(y_true, y_pred)
    r2_scores.append(r2)

print("\n" + "="*50)
print("FINAL TEST RESULTS (1-HOUR AHEAD PREDICTION)")
print("="*50)
for col, r2 in zip(target_cols, r2_scores):
    print(f"{col:30} R²: {r2:.4f}")

print(f"\nAverage R²: {np.mean(r2_scores):.4f}")

# ============ PLOTTING ============
for k, col in enumerate(target_cols):
    clean_col_name = col.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: R² scatter plot
    ax1 = plt.subplot(2, 2, 1)
    y_true = trues_inv[:, k]
    y_pred = preds_inv[:, k]
    ax1.scatter(y_true, y_pred, s=20, alpha=0.6)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax1.plot([mn, mx], [mn, mx], 'r--', linewidth=2)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'{col}\nR² = {r2_scores[k]:.3f}')
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
    ax3 = plt.subplot(2, 2, 3)
    time_points = range(len(y_true))
    ax3.plot(time_points, y_true, label='Actual', linewidth=1, alpha=0.8)
    ax3.plot(time_points, y_pred, label='Predicted', linewidth=1, alpha=0.8)
    ax3.set_xlabel('Time Sequence')
    ax3.set_ylabel('Value')
    ax3.set_title(f'Test Period ({len(y_true)} samples)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Last 200 samples
    ax4 = plt.subplot(2, 2, 4)
    last_200 = min(200, len(y_true))
    ax4.plot(range(last_200), y_true[-last_200:], label='Actual', linewidth=1.5)
    ax4.plot(range(last_200), y_pred[-last_200:], label='Predicted', linewidth=1.5)
    ax4.set_xlabel('Time Sequence')
    ax4.set_ylabel('Value')
    ax4.set_title(f'Last 200 Samples')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'{clean_col_name}_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot for {col}')

# Save results
loss_history = pd.DataFrame({
    'epoch': range(1, EPOCHS + 1),
    **{f'train_loss_{col}': train_losses_per_var[col] for col in target_cols},
    **{f'val_loss_{col}': val_losses_per_var[col] for col in target_cols}
})
loss_history.to_csv(os.path.join(SAVE_DIR, 'per_variable_loss_history.csv'), index=False)

r2_summary = pd.DataFrame({'Variable': target_cols, 'R2_Score': r2_scores})
r2_summary.to_csv(os.path.join(SAVE_DIR, 'r2_summary_1hour.csv'), index=False)

torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'lstm_1hour_model.pth'))

print(f'\nAll results saved to: {SAVE_DIR}')
print('Training completed successfully!')