# -*- coding: utf-8 -*-
"""
3D CNN + LSTM + Tabular AQI Prediction - ENHANCED VERSION
"""

import os, glob, re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
import psutil
import GPUtil
from threading import Thread
import queue

# ---------------- SETTINGS ----------------
IMG_DIR = r"D:\images for1 houre"
CSV_PATH = r"C:\Users\arman\OneDrive\Desktop\AQIorgonized\gapfiledfinal.csv"
SAVE_DIR = r"D:\testNN_meteosat"
os.makedirs(SAVE_DIR, exist_ok=True)

target_col = 'Veldan_PM2.5(ug/m3)'
INPUT_WINDOW = 8
HORIZON = 12
EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- PROGRESS MONITORING ----------------
class ProgressMonitor:
    def __init__(self):
        self.epoch_queue = queue.Queue()
        self.batch_queue = queue.Queue()
        self.metrics_queue = queue.Queue()
        self.running = True
        
    def start_monitoring(self):
        self.monitor_thread = Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        
    def _monitor_resources(self):
        while self.running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_gb = memory.used / (1024**3)
                
                # GPU usage
                gpu_info = ""
                if torch.cuda.is_available():
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        gpu_info += f"GPU{i}: {gpu.load*100:.1f}% | {gpu.memoryUsed:.1f}/{gpu.memoryTotal:.1f}MB | "
                
                metrics = {
                    'cpu': cpu_percent,
                    'memory': memory_percent,
                    'memory_gb': memory_used_gb,
                    'gpu_info': gpu_info,
                    'timestamp': time.time()
                }
                self.metrics_queue.put(metrics)
                
                time.sleep(2)  # Update every 2 seconds
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(5)
                
    def stop_monitoring(self):
        self.running = False

progress_monitor = ProgressMonitor()

# ---------------- REPRODUCIBILITY ----------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_everything(SEED)

# ---------------- LOAD CSV WITH PROGRESS ----------------
print("📊 Loading CSV data...")
df = pd.read_csv(CSV_PATH, parse_dates=[0])
df.rename({df.columns[0]:'Date'}, axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=[target_col]).reset_index(drop=True)
df['UTC_Time'] = df['Date'] - timedelta(hours=3, minutes=30)

print(f"✅ Loaded {len(df)} rows from CSV")

# ---------------- IMAGE FILES WITH PROGRESS ----------------
print("🖼️ Scanning image files...")
img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.tif")))

def parse_img_time(fname):
    base = os.path.basename(fname)
    tstr = re.search(r"(\d{8}T\d{6})", base)
    if tstr:
        return datetime.strptime(tstr.group(1), "%Y%m%dT%H%M%S")
    return None

img_info = [(parse_img_time(f), f) for f in tqdm(img_files, desc="Parsing image timestamps")]
img_info = sorted([x for x in img_info if x[0] is not None], key=lambda x: x[0])

if not img_info:
    raise ValueError("❌ No valid image files found!")

start_t, end_t = img_info[0][0], img_info[-1][0]
df = df[(df['UTC_Time'] >= start_t) & (df['UTC_Time'] <= end_t)].reset_index(drop=True)

if len(df) == 0:
    raise ValueError("❌ No data points within image time range!")

print(f"📅 Data range: {start_t} to {end_t}")
print(f"📈 Remaining data points: {len(df)}")

def nearest_img_time(t):
    diffs = [(abs((t - ti).total_seconds()), fi) for ti, fi in img_info]
    return min(diffs, key=lambda x: x[0])[1]

print("🔗 Matching images to timestamps...")
df['img_path'] = [nearest_img_time(t) for t in tqdm(df['UTC_Time'], desc="Matching images")]

# ---------------- SCALERS ----------------
feature_cols = [c for c in df.columns if c not in ['Date','UTC_Time','img_path',target_col] and df[c].dtype in [np.int64, np.float64]]
print(f"🔧 Using {len(feature_cols)} feature columns")

x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_feat = x_scaler.fit_transform(df[feature_cols])
y_target = y_scaler.fit_transform(df[[target_col]])

# ---------------- ENHANCED DATASET WITH PROGRESS ----------------
class MeteosatDataset(Dataset):
    def __init__(self, df, x_feat, y, input_window, horizon):
        self.df = df.reset_index(drop=True)
        self.x_feat = x_feat
        self.y = y
        self.input_window = input_window
        self.horizon = horizon
        self.seq_starts = [i for i in range(len(df) - input_window - horizon + 1)]
        
    def __len__(self):
        return len(self.seq_starts)
    
    def __getitem__(self, idx):
        i = self.seq_starts[idx]
        imgs = []
        
        # Load input window images with progress tracking
        for j in range(i, i + self.input_window):
            img_path = self.df['img_path'].iloc[j]
            try:
                with rasterio.open(img_path) as src:
                    arr = src.read(out_dtype="float32")
                    # Normalize image data
                    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
                    imgs.append(arr)
            except Exception as e:
                # Create dummy data if image loading fails
                dummy_img = np.random.randn(12, 256, 256).astype(np.float32)
                imgs.append(dummy_img)
        
        imgs = np.stack(imgs, axis=1)  # (bands, time, H, W)
        
        # Use features from the last time step of input window
        feats = self.x_feat[i + self.input_window - 1]
        
        # Target: next horizon steps
        y = self.y[i + self.input_window : i + self.input_window + self.horizon].reshape(-1)
        
        return torch.tensor(imgs), torch.tensor(feats, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

print("📦 Creating dataset...")
dataset = MeteosatDataset(df, X_feat, y_target, INPUT_WINDOW, HORIZON)
n_total = len(dataset)
print(f"✅ Total sequences: {n_total}")

if n_total == 0:
    raise ValueError("❌ No sequences available!")

n_test = max(1, int(0.2 * n_total))
n_val = max(1, int(0.1 * n_total))
n_train = n_total - n_test - n_val

print(f"🎯 Train/Val/Test split: {n_train}/{n_val}/{n_test}")

train_ds, val_ds, test_ds = torch.utils.data.random_split(
    dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------- HYBRID 3D CNN + LSTM MODEL ----------------
try:
    with rasterio.open(df['img_path'].iloc[0]) as src:
        NUM_BANDS = src.count
        IMG_HEIGHT = src.height
        IMG_WIDTH = src.width
    print(f"🖼️ Image dimensions: {NUM_BANDS} bands, {IMG_HEIGHT}x{IMG_WIDTH}")
except Exception as e:
    print(f"⚠️ Error checking image dimensions: {e}")
    NUM_BANDS = 12
    IMG_HEIGHT = 256
    IMG_WIDTH = 256

class HybridCNN3D_LSTM(nn.Module):
    def __init__(self, num_bands, feat_dim, horizon, input_window, img_height=256, img_width=256,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        
        self.input_window = input_window
        
        # 3D CNN for spatiotemporal feature extraction
        self.cnn3d = nn.Sequential(
            # Input: (batch, bands, time, height, width)
            nn.Conv3d(num_bands, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),  # (batch, 32, time, H/2, W/2)
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),  # (batch, 64, time, H/4, W/4)
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1))  # (batch, 128, time, 1, 1)
        )
        
        # LSTM for temporal sequence modeling
        self.lstm = nn.LSTM(
            input_size=128 + feat_dim,  # CNN features + tabular features
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,  # bidirectional
            num_heads=8,
            batch_first=True
        )
        
        # Feature fusion and prediction
        self.fc_fusion = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),  # After attention
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, horizon)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(lstm_hidden * 2)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
                        
    def forward(self, imgs, feats):
        batch_size = imgs.size(0)
        
        # 3D CNN processing
        # imgs shape: (batch, bands, time, H, W)
        cnn_features = self.cnn3d(imgs)  # (batch, 128, time, 1, 1)
        cnn_features = cnn_features.view(batch_size, 128, self.input_window, -1).squeeze(-1)  # (batch, 128, time)
        cnn_features = cnn_features.transpose(1, 2)  # (batch, time, 128)
        
        # Prepare tabular features for each time step
        feats_expanded = feats.unsqueeze(1).expand(-1, self.input_window, -1)  # (batch, time, feat_dim)
        
        # Concatenate CNN features with tabular features
        combined_features = torch.cat([cnn_features, feats_expanded], dim=2)  # (batch, time, 128 + feat_dim)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(combined_features)  # (batch, time, lstm_hidden * 2)
        lstm_out = self.ln1(lstm_out)
        
        # Attention mechanism
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)  # (batch, time, lstm_hidden * 2)
        
        # Use the last time step's attended output
        context_vector = attn_out[:, -1, :]  # (batch, lstm_hidden * 2)
        
        # Final prediction
        output = self.fc_fusion(context_vector)  # (batch, horizon)
        
        return output

model = HybridCNN3D_LSTM(
    num_bands=NUM_BANDS, 
    feat_dim=len(feature_cols), 
    horizon=HORIZON,
    input_window=INPUT_WINDOW,
    img_height=IMG_HEIGHT, 
    img_width=IMG_WIDTH
).to(DEVICE)

print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"🏗️ Model architecture: 3D CNN + LSTM + Attention")

# ---------------- TRAINING WITH COMPREHENSIVE PROGRESS ----------------
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

train_losses, val_losses = [], []
best_val_loss = float('inf')

# Start progress monitoring
print("📊 Starting resource monitoring...")
progress_monitor.start_monitoring()

print("🚀 Starting training...")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    
    # Training phase
    model.train()
    tr_loss = 0
    batch_times = []
    
    train_pbar = tqdm(train_loader, desc=f'🏋️ Epoch {epoch}/{EPOCHS} - Training')
    for batch_idx, (imgs, feats, y) in enumerate(train_pbar):
        batch_start = time.time()
        
        # Data loading to GPU
        imgs, feats, y = imgs.to(DEVICE), feats.to(DEVICE), y.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        preds = model(imgs, feats)
        loss = criterion(preds, y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        tr_loss += loss.item()
        
        # Batch timing
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Update progress bar
        avg_batch_time = np.mean(batch_times[-10:]) if batch_times else 0
        eta = avg_batch_time * (len(train_loader) - batch_idx)
        
        # Get resource metrics
        try:
            metrics = progress_monitor.metrics_queue.get_nowait()
            cpu_usage = metrics['cpu']
            memory_usage = metrics['memory_gb']
            gpu_info = metrics['gpu_info']
        except queue.Empty:
            cpu_usage = 0
            memory_usage = 0
            gpu_info = ""
            
        train_pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'CPU': f'{cpu_usage:.1f}%',
            'Mem': f'{memory_usage:.1f}GB',
            'BatchTime': f'{batch_time:.2f}s'
        })
    
    tr_loss /= len(train_loader)
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_pbar = tqdm(val_loader, desc=f'🧪 Epoch {epoch}/{EPOCHS} - Validation')
    
    with torch.no_grad():
        for imgs, feats, y in val_pbar:
            imgs, feats, y = imgs.to(DEVICE), feats.to(DEVICE), y.to(DEVICE)
            preds = model(imgs, feats)
            val_loss += criterion(preds, y).item()
            
            val_pbar.set_postfix({
                'ValLoss': f'{criterion(preds, y).item():.4f}'
            })
    
    val_loss /= len(val_loader)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': tr_loss
        }, os.path.join(SAVE_DIR, "best_hybrid_model.pth"))
    
    train_losses.append(tr_loss)
    val_losses.append(val_loss)
    
    epoch_time = time.time() - epoch_start
    
    print(f"✅ Epoch {epoch}/{EPOCHS} completed in {epoch_time:.1f}s")
    print(f"   Train Loss: {tr_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {current_lr:.2e}")
    print(f"   Avg Batch Time: {avg_batch_time:.2f}s | Best Val: {best_val_loss:.5f}")
    print("-" * 60)

total_training_time = time.time() - start_time
print(f"🎉 Training completed in {total_training_time:.1f} seconds")

# Stop monitoring
progress_monitor.stop_monitoring()

# ---------------- TESTING WITH PROGRESS ----------------
print("🧪 Starting testing...")
model.eval()
preds_all, trues_all = [], []

# Load best model for testing
checkpoint = torch.load(os.path.join(SAVE_DIR, "best_hybrid_model.pth"))
model.load_state_dict(checkpoint['model_state_dict'])
print(f"📁 Loaded best model from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.5f}")

with torch.no_grad():
    for imgs, feats, y in tqdm(test_loader, desc="🧪 Testing"):
        imgs, feats = imgs.to(DEVICE), feats.to(DEVICE)
        p = model(imgs, feats).cpu().numpy()
        preds_all.append(p)
        trues_all.append(y.numpy())

test_preds = np.vstack(preds_all)
test_trues = np.vstack(trues_all)

# Inverse transform predictions
test_preds_inv = y_scaler.inverse_transform(test_preds.reshape(-1, 1)).reshape(test_preds.shape)
test_trues_inv = y_scaler.inverse_transform(test_trues.reshape(-1, 1)).reshape(test_trues.shape)

# ---------------- COMPREHENSIVE EVALUATION ----------------
print("\n" + "="*70)
print("📊 COMPREHENSIVE TEST RESULTS")
print("="*70)

# Calculate metrics for each horizon
horizon_metrics = []
for h in range(HORIZON):
    r2_h = r2_score(test_trues_inv[:, h], test_preds_inv[:, h])
    rmse_h = mean_squared_error(test_trues_inv[:, h], test_preds_inv[:, h], squared=False)
    mae_h = mean_absolute_error(test_trues_inv[:, h], test_preds_inv[:, h])
    horizon_metrics.append((r2_h, rmse_h, mae_h))
    print(f"Horizon {h+1:2d}: R²={r2_h:.3f} | RMSE={rmse_h:.3f} | MAE={mae_h:.3f}")

# Overall metrics
r2 = r2_score(test_trues_inv.flatten(), test_preds_inv.flatten())
rmse = mean_squared_error(test_trues_inv.flatten(), test_preds_inv.flatten(), squared=False)
mae = mean_absolute_error(test_trues_inv.flatten(), test_preds_inv.flatten())

print("-" * 70)
print(f"📈 OVERALL: R²={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")
print("="*70)

# ---------------- SAVE COMPLETE RESULTS ----------------
torch.save({
    'model_state_dict': model.state_dict(),
    'x_scaler': x_scaler,
    'y_scaler': y_scaler,
    'feature_cols': feature_cols,
    'config': {
        'input_window': INPUT_WINDOW,
        'horizon': HORIZON,
        'num_bands': NUM_BANDS,
        'model_type': 'HybridCNN3D_LSTM'
    },
    'training_info': {
        'total_time': total_training_time,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None
    }
}, os.path.join(SAVE_DIR, "complete_hybrid_model.pth"))

# Save detailed predictions and metrics
results_data = []
for i in range(len(test_preds)):
    for h in range(HORIZON):
        results_data.append({
            'sample_idx': i,
            'horizon': h + 1,
            'actual': test_trues_inv[i, h],
            'predicted': test_preds_inv[i, h],
            'error': test_trues_inv[i, h] - test_preds_inv[i, h],
            'abs_error': abs(test_trues_inv[i, h] - test_preds_inv[i, h])
        })

results_df = pd.DataFrame(results_data)
results_df.to_csv(os.path.join(SAVE_DIR, "hybrid_predictions.csv"), index=False)

# Save training history
history_df = pd.DataFrame({
    'epoch': range(1, len(train_losses) + 1),
    'train_loss': train_losses,
    'val_loss': val_losses
})
history_df.to_csv(os.path.join(SAVE_DIR, "training_history.csv"), index=False)

# ---------------- ENHANCED PLOTTING ----------------
plt.style.use('default')
fig = plt.figure(figsize=(20, 12))

# Plot 1: Training history
plt.subplot(2, 3, 1)
plt.plot(train_losses, 'b-', label="Train Loss", linewidth=2, alpha=0.8)
plt.plot(val_losses, 'r-', label="Val Loss", linewidth=2, alpha=0.8)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Progress\n(Hybrid 3D CNN + LSTM)")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: All predictions vs actual
plt.subplot(2, 3, 2)
plt.scatter(test_trues_inv.flatten(), test_preds_inv.flatten(), alpha=0.6, s=20, 
           c=test_trues_inv.flatten(), cmap='viridis')
min_val = min(test_trues_inv.min(), test_preds_inv.min())
max_val = max(test_trues_inv.max(), test_preds_inv.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"All Predictions\nR² = {r2:.3f}")
plt.colorbar(label='Actual Value')
plt.grid(True, alpha=0.3)

# Plot 3: Horizon-wise R²
plt.subplot(2, 3, 3)
horizon_r2 = [m[0] for m in horizon_metrics]
plt.bar(range(1, HORIZON + 1), horizon_r2, alpha=0.7, color='skyblue')
plt.xlabel("Prediction Horizon")
plt.ylabel("R² Score")
plt.title("R² by Prediction Horizon")
plt.grid(True, alpha=0.3)

# Add value labels
for i, v in enumerate(horizon_r2):
    plt.text(i + 1, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

# Plot 4: Horizon-wise RMSE
plt.subplot(2, 3, 4)
horizon_rmse = [m[1] for m in horizon_metrics]
plt.bar(range(1, HORIZON + 1), horizon_rmse, alpha=0.7, color='lightcoral')
plt.xlabel("Prediction Horizon")
plt.ylabel("RMSE")
plt.title("RMSE by Prediction Horizon")
plt.grid(True, alpha=0.3)

# Add value labels
for i, v in enumerate(horizon_rmse):
    plt.text(i + 1, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

# Plot 5: Time series comparison
plt.subplot(2, 3, 5)
sample_range = min(100, len(test_trues_inv))
plt.plot(test_trues_inv[:sample_range, 0], 'b-', label='Actual', alpha=0.8, linewidth=1)
plt.plot(test_preds_inv[:sample_range, 0], 'r-', label='Predicted', alpha=0.8, linewidth=1)
plt.xlabel("Sample Index")
plt.ylabel(target_col)
plt.title("Time Series - Horizon 1")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Residual analysis
plt.subplot(2, 3, 6)
residuals = test_trues_inv.flatten() - test_preds_inv.flatten()
plt.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "hybrid_comprehensive_results.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"\n💾 All results saved to: {SAVE_DIR}")
print("📁 Files created:")
print("   - complete_hybrid_model.pth (Model + config)")
print("   - best_hybrid_model.pth (Best validation model)")
print("   - hybrid_predictions.csv (Detailed predictions)")
print("   - training_history.csv (Loss history)")
print("   - hybrid_comprehensive_results.png (Results visualization)")
print(f"\n⏱️ Total execution time: {time.time() - start_time:.1f} seconds")