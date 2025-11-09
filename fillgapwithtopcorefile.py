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

# --- Load Filtered Correlations ---
filtered_corr_path = r"C:\Users\arman\OneDrive\Desktop\AQIorgonized\gappfiledby4\filtered_correlations_less_24_nan.csv"
filtered_corr_df = pd.read_csv(filtered_corr_path)

# Extract target columns (keys) and their correlated columns
target_correlations = {}
for _, row in filtered_corr_df.iterrows():
    target_col = int(row['Column_Number'])  # Ensure it's integer
    # Parse the correlated column numbers from string format
    corr_columns_str = row['Top_Correlated_Column_Numbers']
    # Convert string like "[8, 33, 5, 12, 9]" to list of integers
    corr_columns = [int(x) for x in eval(corr_columns_str)] if pd.notna(corr_columns_str) else []
    target_correlations[target_col] = corr_columns

print("Target columns and their correlated columns:")
for target, corr_cols in target_correlations.items():
    print(f"Target {target}: Correlated columns {corr_cols}")

file_path = r"C:\Users\arman\OneDrive\Desktop\AQIorgonized\merged_pollution_weather_with_time.csv"
df = pd.read_csv(file_path)
df = df.apply(pd.to_numeric, errors='coerce')

print(f"Dataset shape: {df.shape}")
print(f"Dataset columns: {list(df.columns)}")

# Store results for all targets
all_results = {}

# Process each target column separately
for target_col, feature_cols in target_correlations.items():
    print(f"\n{'='*60}")
    print(f"PROCESSING TARGET COLUMN {target_col}")
    print(f"{'='*60}")
    
    # Skip if no correlated features available
    if not feature_cols:
        print(f"⚠️  No correlated features for target {target_col}, skipping...")
        continue
    
    # Convert column indices to actual column positions (1-indexed to 0-indexed)
    target_index = target_col   # Convert target to 0-indexed
    
    # Check if target column exists in dataframe
    if target_index >= len(df.columns):
        print(f"⚠️  Target column {target_col} not found in dataframe, skipping...")
        continue
    
    target_name = df.columns[target_index]
    print(f"Target: {target_name} (Column {target_col}, Index {target_index})")
    
    # Convert feature columns from 1-indexed to 0-indexed and filter valid ones
    valid_feature_indices = []
    valid_feature_names = []
    
    for feature_col in feature_cols:
        feature_index = feature_col   # Convert to 0-indexed
        if feature_index < len(df.columns) and feature_index != target_index:  # Don't include target as feature
            valid_feature_indices.append(feature_index)
            valid_feature_names.append(df.columns[feature_index])
    
    # ADD COLUMNS 68-77 AS ADDITIONAL FEATURES (convert from 1-indexed to 0-indexed)
    additional_features = list(range(67, 78))  # Columns 68-77 in 0-indexed
    additional_feature_names = [df.columns[i] for i in additional_features if i < len(df.columns)]
    
    # Combine correlated features with additional features, remove duplicates and target
    all_feature_indices = list(set(valid_feature_indices + additional_features))
    all_feature_indices = [idx for idx in all_feature_indices if idx != target_index]  # Remove target from features
    
    all_feature_names = [df.columns[i] for i in all_feature_indices]
    
    print(f"Correlated features ({len(valid_feature_names)}): {valid_feature_names}")
    print(f"Additional features (68-77): {additional_feature_names}")
    print(f"Total features ({len(all_feature_names)}): {all_feature_names}")
    
    if not all_feature_names:
        print(f"⚠️  No valid features for target {target_col}, skipping...")
        continue
    
    # Create a clean dataset for this target
    selected_columns = all_feature_indices + [target_index]
    try:
        df_clean = df.iloc[:, selected_columns].copy()
    except Exception as e:
        print(f"❌ Error selecting columns: {e}")
        continue
    
    # Set proper column names
    df_clean.columns = all_feature_names + [target_name]
    
    # Drop rows where features are NaN
    df_clean = df_clean.dropna(subset=all_feature_names)
    
    # Separate known and missing targets
    df_known = df_clean.dropna(subset=[target_name])
    df_missing = df_clean[df_clean[target_name].isnull()]
    
    print(f"Known samples: {len(df_known)}")
    print(f"Missing samples to fill: {len(df_missing)}")
    
    if len(df_known) < 10:  # Minimum samples needed
        print(f"⚠️  Not enough known samples ({len(df_known)}) for target {target_col}, skipping...")
        continue
    
    # Prepare features and target
    X_known = df_known[all_feature_names]
    y_known = df_known[[target_name]]
    
    # Train-test split
    test_size = min(200, len(X_known) // 5)
    if len(X_known) <= test_size:
        print(f"⚠️  Not enough samples for train-test split, skipping...")
        continue
        
    X_train, X_test, y_train, y_test = train_test_split(
        X_known, y_known, test_size=test_size, random_state=seed
    )
    
    # Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Single-output model for this target
    input_layer = Input(shape=(X_train_scaled.shape[1],))
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(input_layer)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='tanh')(x)
    x = Dense(8, activation='relu')(x)
    output = Dense(1, name='output')(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_test_scaled, y_test_scaled),
        epochs=50,
        verbose=0,
        batch_size=32
    )
    
    # Predict on test set
    y_test_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
    r2_test = r2_score(y_test.iloc[:, 0], y_test_pred[:, 0])
    
    # Predict on training set
    y_train_pred_scaled = model.predict(X_train_scaled, verbose=0)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
    r2_train = r2_score(y_train.iloc[:, 0], y_train_pred[:, 0])
    
    print(f"✅ {target_name} - Train R²: {r2_train:.3f} | Test R²: {r2_test:.3f}")
    
    # === Predict Missing Values ===
    if not df_missing.empty:
        X_missing = df_missing[all_feature_names]
        X_missing_scaled = scaler_X.transform(X_missing)
        preds_scaled = model.predict(X_missing_scaled, verbose=0)
        preds = scaler_y.inverse_transform(preds_scaled)
        
        # Store predictions
        filled_values = preds[:, 0]
        missing_indices = df_missing.index.tolist()
        print(f"✅ Filled {len(filled_values)} missing values for {target_name}")
    else:
        filled_values = np.array([])
        missing_indices = []
        print(f"ℹ️  No missing values to fill for {target_name}")
    
    # Store results
    all_results[target_col] = {
        'target_name': target_name,
        'target_index': target_index,
        'feature_names': all_feature_names,
        'feature_indices': all_feature_indices,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'filled_values': filled_values,
        'missing_indices': missing_indices,
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    
    # === Plot for this target ===
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Loss - {target_name}\nTest R²: {r2_test:.3f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test.iloc[:, 0], y_test_pred[:, 0], alpha=0.6, color='blue')
    plt.plot([y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()],
             [y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()], 'r--')
    plt.title(f'Actual vs Predicted - {target_name}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# === Combine all filled values into final dataset ===
print(f"\n{'='*60}")
print("COMBINING ALL FILLED VALUES")
print(f"{'='*60}")

# Create a copy of original dataframe for filling
df_filled = df.copy()

filled_count = 0
for target_col, result in all_results.items():
    if len(result['filled_values']) > 0:
        target_name = result['target_name']
        missing_indices = result['missing_indices']
        filled_values = result['filled_values']
        
        # Fill the missing values in the main dataframe
        df_filled.loc[missing_indices, target_name] = filled_values
        filled_count += len(filled_values)
        
        print(f"✅ Filled {len(filled_values)} values in {target_name}")

print(f"\n🎯 TOTAL: Filled {filled_count} missing values across {len(all_results)} targets")

# === Save the complete filled dataset ===
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save the filled dataset
filled_filename = f"filled_dataset_complete_{timestamp}.csv"
df_filled.to_csv(filled_filename, index=False)
print(f"✅ Saved complete filled dataset to: {filled_filename}")

# Save results summary
summary_data = []
for target_col, result in all_results.items():
    summary_data.append({
        'Target_Column_Number': target_col,
        'Target_Column_Name': result['target_name'],
        'Features_Used': str(result['feature_names']),
        'Train_R2': result['r2_train'],
        'Test_R2': result['r2_test'],
        'Missing_Values_Filled': len(result['filled_values'])
    })

summary_df = pd.DataFrame(summary_data)
summary_filename = f"gapfilling_results_summary_{timestamp}.csv"
summary_df.to_csv(summary_filename, index=False)
print(f"✅ Saved results summary to: {summary_filename}")

# Save models and scalers
"""models_dir = f"gapfill_models_{timestamp}"
os.makedirs(models_dir, exist_ok=True)

for target_col, result in all_results.items():
    target_name_clean = re.sub(r'[^A-Za-z0-9]+', '_', result['target_name'])
    model_dir = f"{models_dir}/model_{target_col}_{target_name_clean}"
    os.makedirs(model_dir, exist_ok=True)
    
    result['model'].save(f"{model_dir}/model.h5")
    np.save(f"{model_dir}/scaler_X_mean.npy", result['scaler_X'].mean_)
    np.save(f"{model_dir}/scaler_X_scale.npy", result['scaler_X'].scale_)
    np.save(f"{model_dir}/scaler_y_mean.npy", result['scaler_y'].mean_)
    np.save(f"{model_dir}/scaler_y_scale.npy", result['scaler_y'].scale_)

print(f"✅ Saved all models and scalers to: {models_dir}")"""

# Print final summary
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(f"Targets processed: {len(all_results)}")
print(f"Total missing values filled: {filled_count}")
print(f"Final dataset shape: {df_filled.shape}")
print(f"Original dataset shape: {df.shape}")

for target_col, result in all_results.items():
    status = "✅ SUCCESS" if result['r2_test'] > 0.5 else "⚠️ POOR" if result['r2_test'] > 0 else "❌ FAILED"
    print(f"{status} - {result['target_name']}: Test R² = {result['r2_test']:.3f}, Filled {len(result['filled_values'])} values")