import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import random

# ============ USER CONFIG ============
INPUT_FILE = r'C:\Users\arman\OneDrive\Desktop\AQIorgonized\gapfiledfinal.csv'
INPUT_WINDOW = 72
HORIZON = 1  # CHANGE THIS: 1=next hour, 2=2 hours ahead, 3=3 hours ahead, etc.
TEST_SIZE = 1000
VAL_SIZE = 1000
BATCH_SIZE = 64
EPOCHS = 60
LR = 0.001
SEED = 42

# reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print('Device:', 'GPU' if tf.config.list_physical_devices('GPU') else 'CPU')
print(f'Predicting {HORIZON} hour(s) ahead')

# ============ LOAD DATA ============
df = pd.read_csv(INPUT_FILE, parse_dates=['Date'])

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
        Xs.append(X[i:i+input_window])  # Past 48 hours
        Ys.append(Y[i+input_window+horizon-1])  # Target: input_window + horizon hours ahead
    return np.array(Xs), np.array(Ys)

X_seq, Y_seq = create_multi_horizon(X_scaled, Y_raw, INPUT_WINDOW, HORIZON)
if X_seq.shape[0] == 0:
    raise ValueError('No sequences created — reduce INPUT_WINDOW or HORIZON or check data length.')

print(f"Total sequences: {X_seq.shape}")
print(f"Prediction: Using {INPUT_WINDOW} hours to predict {HORIZON} hour(s) ahead")

# ============ DATA SPLIT ============
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

# ============ TARGET SCALING ============
y_scalers = []
Y_train_scaled = np.zeros_like(Y_train, dtype=np.float32)
Y_val_scaled = np.zeros_like(Y_val, dtype=np.float32)
Y_test_scaled = np.zeros_like(Y_test, dtype=np.float32)

for j in range(Y_train.shape[1]):
    s = StandardScaler()
    train_data = Y_train[:, j].reshape(-1, 1)
    s.fit(train_data)
    
    Y_train_scaled[:, j] = s.transform(train_data).flatten()
    Y_val_scaled[:, j] = s.transform(Y_val[:, j].reshape(-1, 1)).flatten()
    Y_test_scaled[:, j] = s.transform(Y_test[:, j].reshape(-1, 1)).flatten()
    
    y_scalers.append(s)

# ============ YOUR EXACT KERAS MODEL ============
# ============ CORRECTED KERAS MODEL ============
model = Sequential()

# LSTM1: 64 units, return_sequences=True, ReLU, L2 regularization
model.add(LSTM(64, activation="relu", return_sequences=True, 
               kernel_regularizer=l2(0.01), input_shape=(INPUT_WINDOW, X_train.shape[2])))
model.add(Dropout(0.3))

# LSTM2: 64 units, return_sequences=True, ReLU
model.add(LSTM(64, activation="relu", return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# LSTM3: 32 units, return_sequences=True, ReLU, L2 regularization
model.add(LSTM(32, activation="relu", return_sequences=True, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())

# LSTM4: 64 units, return_sequences=False, no activation
model.add(LSTM(64, activation="relu", return_sequences=False))  # Final LSTM - no return_sequences
model.add(BatchNormalization())

# Dense output layer
model.add(Dense(len(target_cols)))  # Output for all target variables

# Using mean_squared_logarithmic_error as specified
model.compile(loss='mean_squared_logarithmic_error', optimizer=Adam(learning_rate=LR))

print("Model summary:")

model.summary()

# ============ TRAINING WITH HISTORY TRACKING ============
print("Starting training...")

# Custom callback to track per-variable losses
class PerVariableLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, Y_val, target_cols):
        super().__init__()
        self.X_val = X_val
        self.Y_val = Y_val
        self.target_cols = target_cols
        self.train_losses_per_var = {col: [] for col in target_cols}
        self.val_losses_per_var = {col: [] for col in target_cols}
        self.overall_train_losses = []
        self.overall_val_losses = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Overall losses
        self.overall_train_losses.append(logs['loss'])
        self.overall_val_losses.append(logs['val_loss'])
        
        # Per-variable validation losses
        val_pred = self.model.predict(self.X_val, verbose=0)
        for i, col in enumerate(self.target_cols):
            # Calculate MSE for each variable (since we can't get per-variable MSLE easily)
            var_mse = np.mean((val_pred[:, i] - self.Y_val[:, i]) ** 2)
            self.val_losses_per_var[col].append(var_mse)
            
            # For training, we'll approximate with overall loss (limitation of Keras)
            self.train_losses_per_var[col].append(logs['loss'])
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:3d}/{EPOCHS} | Train: {logs["loss"]:.6f} | Val: {logs["val_loss"]:.6f}')
            for col in self.target_cols:
                train_loss = self.train_losses_per_var[col][-1]
                val_loss = self.val_losses_per_var[col][-1]
                print(f'         {col:25} | Train: {train_loss:.6f} | Val: {val_loss:.6f}')

# Create callback
loss_callback = PerVariableLossCallback(X_val, Y_val_scaled, target_cols)

# Train model
history = model.fit(
    X_train, Y_train_scaled,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, Y_val_scaled),
    callbacks=[loss_callback],
    verbose=0  # We'll print manually in callback
)

# ============ FINAL EVALUATION ============
# Predict on test set
preds_scaled = model.predict(X_test, verbose=0)
preds = preds_scaled.copy()
trues = Y_test_scaled.copy()

# Inverse transform predictions
preds_inv = np.zeros_like(preds)
trues_inv = np.zeros_like(trues)
for k in range(len(target_cols)):
    s = y_scalers[k]
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
print(f"FINAL TEST RESULTS ({HORIZON}-HOUR AHEAD PREDICTION)")
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
    ax1.set_title(f'{col}\nR² = {r2_scores[k]:.3f} ({HORIZON}-hour ahead)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: PER-VARIABLE Loss curves
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(loss_callback.train_losses_per_var[col], label='Train Loss', linewidth=2)
    ax2.plot(loss_callback.val_losses_per_var[col], label='Validation Loss', linewidth=2)
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
    ax3.set_title(f'Test Period ({len(y_true)} samples) - {HORIZON}-hour ahead')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Last 200 samples
    ax4 = plt.subplot(2, 2, 4)
    last_200 = min(200, len(y_true))
    ax4.plot(range(last_200), y_true[-last_200:], label='Actual', linewidth=1.5)
    ax4.plot(range(last_200), y_pred[-last_200:], label='Predicted', linewidth=1.5)
    ax4.set_xlabel('Time Sequence')
    ax4.set_ylabel('Value')
    ax4.set_title(f'Last 200 Samples - {HORIZON}-hour ahead')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'{clean_col_name}_{HORIZON}hour_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot for {col}')

# Save results
loss_history = pd.DataFrame({
    'epoch': range(1, EPOCHS + 1),
    **{f'train_loss_{col}': loss_callback.train_losses_per_var[col] for col in target_cols},
    **{f'val_loss_{col}': loss_callback.val_losses_per_var[col] for col in target_cols}
})
loss_history.to_csv(os.path.join(SAVE_DIR, f'per_variable_loss_history_{HORIZON}hour.csv'), index=False)

r2_summary = pd.DataFrame({'Variable': target_cols, 'R2_Score': r2_scores})
r2_summary.to_csv(os.path.join(SAVE_DIR, f'r2_summary_{HORIZON}hour.csv'), index=False)

# Save model
model.save(os.path.join(SAVE_DIR, f'lstm_{HORIZON}hour_model.h5'))

print(f'\nAll results saved to: {SAVE_DIR}')
print(f'Training completed successfully for {HORIZON}-hour ahead prediction!')