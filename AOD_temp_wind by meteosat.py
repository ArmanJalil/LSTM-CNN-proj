# Full Python Code: Extract Raster AOD, Temperature, Wind Speed from SEVIRI TIFFs
# Saves to D:\output.npz (NPZ format - optimal for NN input with spatial data)
# Data shape: (n_times, 3, 20, 20) [AOD, temperature_K, wind_speed_mps]

import glob
import os
import rasterio
import numpy as np
from datetime import datetime
import cv2  # Requires: pip install opencv-python
import pandas as pd  # For time handling, optional

# ===================== CONFIGURATION =====================
folder_path = r'D:\images for1 houre'      # Input folder
output_npz_path = r'D:\output.npz'          # Output on D drive
pixel_resolution_m = 5000                   # Approx. 0.05° resolution in meters
min_time_gap_sec = 300                      # Min valid gap (5 min) for wind

# ===================== DATA COLLECTION =====================
print("Scanning TIFF files...")
files = glob.glob(os.path.join(folder_path, '*_multi.tif'))

file_times = []
for f in files:
    name = os.path.basename(f)[:-10]  # Remove '_multi.tif'
    try:
        dt = datetime.strptime(name, '%Y%m%dT%H%M%S')
        file_times.append((f, dt))
    except ValueError:
        print(f"Warning: Skipping invalid filename: {f}")
        continue

if not file_times:
    raise ValueError("No valid TIFF files found in the folder.")

# Sort by time
file_times.sort(key=lambda x: x[1])
files, times = zip(*file_times)

# ===================== PROCESSING LOOP =====================
data_list = []  # List of (3, 20, 20) arrays
time_list = []  # List of datetimes
prev_img = None
prev_time = None

print(f"Processing {len(files)} files...")

for i, (f, t) in enumerate(zip(files, times)):
    try:
        with rasterio.open(f) as src:
            if src.count < 12:
                print(f"Warning: {f} has only {src.count} bands. Skipping.")
                prev_img = None
                continue

            bands = src.read()  # Shape: (12, 20, 20)

            # === TEMPERATURE (LST): Simplified Split-Window ===
            T108 = bands[4, :, :]  # IR_108 (0-indexed band 5)
            T120 = bands[2, :, :]  # IR_120 (band 3)
            lst = T108 + 3.0 * (T108 - T120)  # (20,20) array in Kelvin

            # === AOD: Empirical from VIS006 (Band 7) ===
            rho06 = bands[6, :, :] / 100.0  # VIS006, assume % to fraction
            aod = np.maximum(0.0, (rho06 - 0.05) / 0.15)  # (20,20)

            # === WIND SPEED: Optical Flow on IR10.8 (Band 5) ===
            wind_speed = np.full((20, 20), np.nan)  # Default NaN
            if prev_img is not None and prev_time is not None:
                dt_sec = (t - prev_time).total_seconds()
                if dt_sec >= min_time_gap_sec:
                    curr_img = bands[4, :, :].astype(np.float32)  # IR10.8
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_img, curr_img, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                    )  # (20,20,2)
                    dx = flow[..., 0]
                    dy = flow[..., 1]
                    u = dx * pixel_resolution_m / dt_sec
                    v = dy * pixel_resolution_m / dt_sec
                    wind_speed = np.sqrt(u**2 + v**2)  # (20,20) in m/s

            # Stack rasters: (3, 20, 20)
            raster_stack = np.stack([aod, lst, wind_speed], axis=0)
            data_list.append(raster_stack)
            time_list.append(t)

            # Update previous
            prev_img = bands[4, :, :].astype(np.float32)
            prev_time = t

    except Exception as e:
        print(f"Error processing {f}: {e}")
        prev_img = None
        continue

# ===================== SAVE TO NPZ ON D:\ =====================
if data_list:
    data_array = np.array(data_list)  # (n_times, 3, 20, 20)
    time_array = np.array([t.strftime('%Y-%m-%d %H:%M:%S') for t in time_list])
    np.savez(output_npz_path, data=data_array, times=time_array)
    print(f"\nSuccess: Data saved to {output_npz_path}")
    print(f"   Shape: {data_array.shape} (times, channels, height, width)")
    print("   Channels: 0=AOD, 1=temperature_K, 2=wind_speed_mps")
else:
    print("No valid data processed.")
loaded = np.load(r'D:\output.npz')
data = loaded['data']  # (n_times, 3, 20, 20)
times = loaded['times']
