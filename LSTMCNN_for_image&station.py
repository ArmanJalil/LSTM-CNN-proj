# فایل: cnn_lstm_attention_multioutput_torch.py
# اجرا در Spyder یا هر IDE دیگری
# برای GPU مطمئن شو که PyTorch نسخه CUDA نصب شده (print(torch.cuda.is_available()) باید True بده)

import os, glob, re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import rasterio
from rasterio.enums import Resampling
import jdatetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------
# تنظیمات
# ---------------------------
tif_dir = r"D:\cropped"
csv_path = r"C:\Users\arman\OneDrive\Desktop\AQI proj\fianalgappfiled.csv"
target_cols = ["25thAban-CO","25thAban-NO2","25thAban-NO","25thAban-O3","25thAban-PM2.5","25thAban-SO2"]

T_in = 12
T_out = 72
IMG_SIZE = (700, 440)
bands = 12
BATCH_SIZE = 1
EPOCHS = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🟢 Device:", device)

# ---------------------------
# توابع کمکی
# ---------------------------
def jalali_to_gregorian(jalali_str):
    j = jdatetime.datetime.strptime(jalali_str, "%Y/%m/%d %H:%M:%S")
    return j.togregorian()

def parse_tif_timestamp(path):
    base = os.path.basename(path)
    m = re.search(r'(\d{8}T\d{6})', base)
    if not m:
        return None
    s = m.group(1)
    return datetime.strptime(s, "%Y%m%dT%H%M%S")

def ceil_to_next_hour(dt):
    if dt.minute == 0 and dt.second == 0:
        return dt
    return (dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))

def read_tif_as_array(path, target_size=(700,440), bands=bands):
    W,H = target_size
    with rasterio.open(path) as src:
        out = src.read(out_shape=(src.count, H, W), resampling=Resampling.bilinear)
        arr = np.transpose(out, (1,2,0)).astype(np.float32)
        c = arr.shape[2]
        if c != bands:
            new = np.zeros((arr.shape[0], arr.shape[1], bands), dtype=np.float32)
            new[:,:,:min(c,bands)] = arr[:,:,:min(c,bands)]
            arr = new
        m = np.max(np.abs(arr)) + 1e-6
        arr = arr / m
        return arr

# ---------------------------
# خواندن CSV
# ---------------------------
df = pd.read_csv(csv_path, sep=",", encoding="utf-8")
date_col = df.columns[0]
df['datetime'] = df[date_col].astype(str).apply(jalali_to_gregorian)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

# ---------------------------
# نگاشت زمان‌ها به فایل‌های tif
# ---------------------------
tif_paths = glob.glob(os.path.join(tif_dir, "*.tif"))
tif_map = {}
for p in tif_paths:
    ts = parse_tif_timestamp(p)
    if ts is not None:
        tif_map[ts] = p
tif_times = sorted(tif_map.keys())
tifs_by_station_time = defaultdict(list)
for t in tif_times:
    st = ceil_to_next_hour(t)
    tifs_by_station_time[st].append(t)

def choose_tif_for_station_time(station_dt, threshold_minutes=60):
    candidates = tifs_by_station_time.get(station_dt, [])
    if not candidates:
        return None
    best = min(candidates, key=lambda x: abs(x - station_dt))
    if abs(best - station_dt) <= timedelta(minutes=threshold_minutes):
        return tif_map[best]
    return None

# ---------------------------
# ساخت دیتاست PyTorch
# ---------------------------
class AirQualityDataset(Dataset):
    def __init__(self, df, target_cols, tif_dir):
        self.df = df
        self.times = sorted(df.index)
        self.target_cols = target_cols
        self.samples = []
        n_times = len(self.times)
        for idx in range(n_times):
            t0 = self.times[idx]
            start_in = t0 - timedelta(hours=T_in-1)
            end_out = t0 + timedelta(hours=T_out-1)
            if start_in < self.times[0] or end_out > self.times[-1]:
                continue
            ok = True
            in_tifs, in_vals, out_vals = [], [], []
            for k in range(T_in):
                t_step = start_in + timedelta(hours=k)
                try:
                    vals = df.loc[t_step, target_cols].astype(float).values
                except KeyError:
                    ok = False; break
                tif = choose_tif_for_station_time(t_step)
                if tif is None: ok = False; break
                in_tifs.append(tif)
                in_vals.append(vals)
            if not ok: continue
            for k in range(T_out):
                t_step = t0 + timedelta(hours=k)
                try:
                    tv = df.loc[t_step, target_cols].astype(float).values
                except KeyError:
                    ok = False; break
                out_vals.append(tv)
            if not ok: continue
            self.samples.append((in_tifs, np.array(in_vals), np.array(out_vals)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        in_tifs, in_vals, out_vals = self.samples[idx]
        imgs = np.stack([read_tif_as_array(p, IMG_SIZE, bands) for p in in_tifs], axis=0)
        imgs = torch.tensor(imgs).permute(0,3,1,2)  # (T_in, C, H, W)
        station = torch.tensor(in_vals, dtype=torch.float32)
        target = torch.tensor(out_vals, dtype=torch.float32)
        return (imgs, station), target

# ---------------------------
# مدل CNN + LSTM + Attention
# ---------------------------
class CNNEncoder(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(bands, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
    def forward(self, x):
        # x: (B, T, C, H, W)
        B,T,C,H,W = x.shape
        x = x.reshape(B*T, C, H, W)
        feats = self.conv(x).view(B, T, -1)
        return feats

class AttentionLSTMModel(nn.Module):
    def __init__(self, bands, n_targets, feat_dim=256):
        super().__init__()
        self.cnn = CNNEncoder(bands)
        self.lstm_enc = nn.LSTM(128 + n_targets, feat_dim, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(2*feat_dim, 1)
        self.lstm_dec = nn.LSTM(2*feat_dim, 256, batch_first=True)
        self.out = nn.Linear(256, n_targets)

    def forward(self, imgs, stations):
        feats = self.cnn(imgs)  # (B,T,128)
        x = torch.cat([feats, stations], dim=-1)
        enc_out, _ = self.lstm_enc(x)
        attn_weights = F.softmax(self.attn_fc(enc_out), dim=1)
        context = torch.sum(attn_weights * enc_out, dim=1, keepdim=True)
        dec_in = context.repeat(1, T_out, 1)
        dec_out, _ = self.lstm_dec(dec_in)
        out = self.out(dec_out)
        return out

# ---------------------------
# آماده‌سازی داده و آموزش
# ---------------------------
dataset = AirQualityDataset(df, target_cols, tif_dir)
n_val = 500
train_set = torch.utils.data.Subset(dataset, list(range(len(dataset)-n_val)))
val_set = torch.utils.data.Subset(dataset, list(range(len(dataset)-n_val, len(dataset))))
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

model = AttentionLSTMModel(bands, len(target_cols)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

train_loss, val_loss = [], []
for epoch in range(EPOCHS):
    model.train()
    total = 0
    for (imgs, st), y in train_loader:
        imgs, st, y = imgs.to(device), st.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(imgs, st)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    tr_loss = total/len(train_loader)
    train_loss.append(tr_loss)

    model.eval()
    with torch.no_grad():
        total = 0
        for (imgs, st), y in val_loader:
            imgs, st, y = imgs.to(device), st.to(device), y.to(device)
            pred = model(imgs, st)
            loss = criterion(pred, y)
            total += loss.item()
        v_loss = total/len(val_loader)
    val_loss.append(v_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train: {tr_loss:.4f} - Val: {v_loss:.4f}")

torch.save(model.state_dict(), "final_multi_model_torch.pth")
np.savetxt("train_val_loss.csv", np.vstack([train_loss,val_loss]).T, delimiter=",", header="train,val")

plt.plot(train_loss, label="train")
plt.plot(val_loss, label="val")
plt.legend(); plt.title("Loss over epochs")
plt.savefig("training_plots_torch.png")
plt.show()
