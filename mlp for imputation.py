import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import random
import os
from datetime import datetime
import re

# Fix all randomness for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# --- Configurable Section ---
TARGET_COL_INDICES = [6,7]
FEATURE_COL_INDICES = list(set(range(4)) | set(range(7, 52)) - set(TARGET_COL_INDICES)) #+ [6, 9, 12, 26, 30, 37, 39, 40]
file_path = r"C:\\Users\\arman\\OneDrive\\Desktop\\AQI proj\\test.csv"
df = pd.read_csv(file_path)
df = df.apply(pd.to_numeric, errors='coerce')

# Step 1: Drop NaNs only from feature columns
feature_names = df.columns[FEATURE_COL_INDICES]
target_names = df.columns[TARGET_COL_INDICES].tolist()
df = df[~df[feature_names].isnull().any(axis=1)]

# Step 2: Separate known and missing targets
df_known = df.dropna(subset=target_names)
df_missing = df[df[target_names].isnull().any(axis=1)]

X_known = df_known[feature_names]
y_known = df_known[target_names]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, test_size=200, random_state=seed)

# Scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Multi-output model
input_layer = Input(shape=(X_train_scaled.shape[1],))
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(input_layer)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='tanh')(x)
x = Dense(8, activation='relu')(x)
outputs = [Dense(1, name=name)(x) for name in target_names]
model = Model(inputs=input_layer, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

# Split y into dict
y_train_dict = {name: y_train_scaled[:, i] for i, name in enumerate(target_names)}
y_test_dict = {name: y_test_scaled[:, i] for i, name in enumerate(target_names)}

# Train
history = model.fit(
    X_train_scaled, y_train_dict,
    validation_data=(X_test_scaled, y_test_dict),
    epochs=50,
    verbose=0
)

# Predict
y_test_pred_scaled = model.predict(X_test_scaled)
y_test_pred = scaler_y.inverse_transform(np.hstack(y_test_pred_scaled))
r2_test = [r2_score(y_test.iloc[:, i], y_test_pred[:, i]) for i in range(len(target_names))]

# Calculate R² for training set
y_train_pred_scaled = model.predict(X_train_scaled)
y_train_pred = scaler_y.inverse_transform(np.hstack(y_train_pred_scaled))
r2_train = [r2_score(y_train.iloc[:, i], y_train_pred[:, i]) for i in range(len(target_names))]

# Print R² scores for both train and test
for name, r2_tr, r2_te in zip(target_names, r2_train, r2_test):
    print(f"{name} - Train R²: {r2_tr:.3f} | Test R²: {r2_te:.3f}")

# === PLOTS ===
num_targets = len(target_names)
fig = plt.figure(figsize=(14, 5 * num_targets))
gs = fig.add_gridspec(num_targets, 2)

for i, col_name in enumerate(target_names):
    # Loss plot
    ax_loss = fig.add_subplot(gs[i, 0])
    ax_loss.plot(history.history[f'{col_name}_loss'], label='Train Loss')
    ax_loss.plot(history.history[f'val_{col_name}_loss'], label='Val Loss')
    ax_loss.set_title(f'Loss per Epoch - {col_name}')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True)

    # Scatter plot
    ax_scatter = fig.add_subplot(gs[i, 1])
    ax_scatter.scatter(y_test.iloc[:, i], y_test_pred[:, i], alpha=0.6, color='blue')
    ax_scatter.plot(
        [min(y_test.iloc[:, i]), max(y_test.iloc[:, i])],
        [min(y_test.iloc[:, i]), max(y_test.iloc[:, i])],
        'r--'
    )
    ax_scatter.set_title(f'{col_name} - R²: {r2_test[i]:.3f}')
    ax_scatter.set_xlabel('Actual')
    ax_scatter.set_ylabel('Predicted')
    ax_scatter.grid(True)

plt.tight_layout()
plt.show()

# Print R² scores
for name, r2 in zip(target_names, r2_test):
    print(f"{name} R²: {r2:.3f}")

# === Predict Missing Values ===
if not df_missing.empty:
    X_missing_scaled = scaler_X.transform(df_missing[feature_names])
    preds_scaled = model.predict(X_missing_scaled)
    preds = scaler_y.inverse_transform(np.hstack(preds_scaled))

    filled_targets = df_missing.copy()
    for i, col in enumerate(target_names):
        filled_targets[col] = preds[:, i]
else:
    filled_targets = pd.DataFrame(columns=target_names)

# === Save Section ===

# Clean column names for file naming
def clean_name(name):
    return re.sub(r'[^A-Za-z0-9]+', '_', name.strip())

# Build filename from R² scores and target names
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename_parts = [f"{target_names[i]}:{r2_test[i]:.3f}" for i in range(len(target_names))]
filename_str = "-".join([f"r2_{clean_name(p)}" for p in filename_parts])
final_filename = f"{filename_str}-{timestamp}.csv"

# Save filled target DataFrame
filled_target_df = pd.DataFrame(filled_targets)
filled_target_df.to_csv(final_filename, index_label="Index")
print(f"✅ Saved filled targets to: {final_filename}")

# Save model and scalers
model_dir = f"model_{filename_str}_{timestamp}"
os.makedirs(model_dir, exist_ok=True)
model.save(f"{model_dir}/model.h5")
np.save(f"{model_dir}/scaler_X_mean.npy", scaler_X.mean_)
np.save(f"{model_dir}/scaler_X_scale.npy", scaler_X.scale_)

# Save scaler_y values for all targets
np.save(f"{model_dir}/scaler_y_mean.npy", scaler_y.mean_)
np.save(f"{model_dir}/scaler_y_scale.npy", scaler_y.scale_)

print(f"✅ Model and scalers saved to: {model_dir}")
