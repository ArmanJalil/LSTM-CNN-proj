import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import random
import os
from datetime import datetime

# Fix all randomness for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# --- Configurable Section ---
TARGET_COL_INDEX = 9
EXCLUDE_INDICES = {5,6,7, 8,9,41,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27, 28,29,30,31,32,33,37,36,39,35,38,33,34,40}
FEATURE_COL_INDICES = list(set(range(52)) - {TARGET_COL_INDEX} - EXCLUDE_INDICES)
file_path = r"C:\\Users\\arman\\OneDrive\\Desktop\\AQI proj\\test.csv"
df = pd.read_csv(file_path)
df = df.apply(pd.to_numeric, errors='coerce')
"""5:25thAban-CO	,6:Enghelab-CO	,7:Farshadi-CO	,8:Feiz-CO	,9:Kave-CO	,10:Kerdabad-CO	,11:MirzaTaher-CO	
,12:Rehnan-CO	,13:Veldan-CO	,14:25thAban-O3	,15:25thAban-NO	,16:Farshadi-NO	,17:Feiz-NO	,18:Kerdabad-NO	,
19:Veldan-NO	,20:25thAban-NO2	,21:Farshadi-NO2	,22:Feiz-NO2	,23:Kerdabad-NO2	,24:Veldan-NO2	,
25:25thAban-SO2	,26:Enghelab-SO2	,27:Farshadi-SO2	,28:Feiz-SO2	,29:Kave-SO2	,30:MirzaTaher-SO2	,
31:Rehnan-SO2	,32:Veldan-SO2	,33:25thAban-PM2.5	,34:Enghelab-PM2.5	,35:Farshadi-PM2.5	,36:Feiz-PM2.5	,
37:Kave-PM2.5	,38:Kerdabad-PM2.5	,39:MirzaTaher-PM2.5	,40:Rehnan-PM2.5	,41:Veldan-PM2.5
"""
# Step 1: Keep all rows but drop NaNs only from feature columns
target_name = df.columns[TARGET_COL_INDEX]
original_target_column = df[target_name].copy()
df_features_only = df.drop(columns=[target_name])
df = df[~df_features_only.isnull().any(axis=1)]

# Step 2: Separate known and missing target values
df_known = df[df[target_name].notnull()]
df_missing = df[df[target_name].isnull()]

X_known = df_known.iloc[:, FEATURE_COL_INDICES]
y_known = df_known.iloc[:, TARGET_COL_INDEX]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, test_size=200, random_state=seed)

# Scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Model
model = Sequential([
    Dense(48, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
    Dropout(0.4),
    Dense(16, activation='relu'),
    Dense(8, activation='tanh'),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse')

# Train
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_test_scaled, y_test_scaled),
    epochs=400,
    verbose=0
)

# Predict known
y_train_pred_scaled = model.predict(X_train_scaled)
y_test_pred_scaled = model.predict(X_test_scaled)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Reload the original data with NaNs included
df_full = pd.read_csv(file_path)
df_full = df_full.apply(pd.to_numeric, errors='coerce')

# Preserve the full target column before dropping NaNs
original_target_column = df_full.iloc[:, TARGET_COL_INDEX].copy()

# Find where the original target had NaNs
missing_mask = original_target_column.isna()
missing_indices_all = original_target_column[missing_mask].index

# Only fill missing values not in the test set
missing_indices = missing_indices_all.difference(y_test.index)

# Prepare X_missing using the full (pre-dropna) data
feature_col_names = df_full.columns[FEATURE_COL_INDICES]  # <-- FIX
X_missing = df_full.loc[missing_indices, feature_col_names]  # <-- FIXED LINE
X_missing_scaled = scaler_X.transform(X_missing)


# Predict missing target values
y_missing_pred_scaled = model.predict(X_missing_scaled).flatten()
y_missing_pred = scaler_y.inverse_transform(y_missing_pred_scaled.reshape(-1, 1)).flatten()

# Create the filled target column
filled_target = original_target_column.copy()

# Fill only truly missing values (NaNs), not including test set real values
for idx, pred in zip(missing_indices, y_missing_pred):
    filled_target.at[idx] = pred

# Save
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
target_name = df.columns[TARGET_COL_INDEX]
filename = f"{target_name}-r2_{r2_test:.3f}-{timestamp}.csv"
filled_target.to_csv(filename, index_label="Index")
print(f"✅ Saved filled column to: {filename}")


# Save model and scalers
model_dir = f"model_{target_name}_r2_{r2_test:.3f}_{timestamp}"
os.makedirs(model_dir, exist_ok=True)
model.save(f"{model_dir}/model.h5")
np.save(f"{model_dir}/scaler_X_mean.npy", scaler_X.mean_)
np.save(f"{model_dir}/scaler_X_scale.npy", scaler_X.scale_)
np.save(f"{model_dir}/scaler_y_mean.npy", scaler_y.mean_)
np.save(f"{model_dir}/scaler_y_scale.npy", scaler_y.scale_)
print(f"Model and scalers saved in: {model_dir}")

# === PLOTS (your original visualization section) ===
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(2, 2)

# Loss plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Test Loss')
ax1.set_title('Loss per Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Scatter plot
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, y_test_pred, alpha=0.6, color='blue')
ax2.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
ax2.set_title(f'Test Predictions\nTrain R²: {r2_train:.3f} | Test R²: {r2_test:.3f}')
ax2.set_xlabel('Actual')
ax2.set_ylabel('Predicted')
ax2.grid(True)

# Line plot: actual vs predicted
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(y_test.values, label='Actual', marker='o')
ax3.plot(y_test_pred, label='Predicted', marker='x')
ax3.set_title(f"Actual vs Predicted - {target_name}\nR² = {r2_test:.2f}")
ax3.set_xlabel("Sample")
ax3.set_ylabel(target_name)
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()

# Print R²
print(f"Train R²: {r2_train:.3f}")
print(f"Test R²: {r2_test:.3f}")
