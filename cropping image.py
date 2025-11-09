import os
import zipfile
import shutil
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from satpy import Scene
from pyresample import create_area_def
import time
import traceback

# ---------- مسیرها ----------
input_dir = "e:/meteodata"
output_dir = "D:/cropped"
temp_dir = "temp_extracted"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# ---------- محدوده جغرافیایی ----------
lon_min, lon_max = 30.0, 65.0   # درجه شرقی
lat_min, lat_max = 23.0, 45.0   # درجه شمالی

geo_area = create_area_def(
    area_id="isfahan_geo_crop",
    projection={'proj': 'latlong', 'datum': 'WGS84'},
    area_extent=(lon_min, lat_min, lon_max, lat_max),
    resolution=(0.05, 0.05),
    units='degrees'
)

print(f"Target region: Lon [{lon_min}, {lon_max}]  Lat [{lat_min}, {lat_max}]")

# ---------- توابع کمکی ----------
def process_zip(zip_path, output_dir, temp_dir, geo_area, lon_min, lat_min, lon_max, lat_max):
    """بازکردن، بازنمونه‌برداری و ذخیره به صورت GeoTIFF چندباندی"""
    zip_file = os.path.basename(zip_path)
    base_name = os.path.splitext(zip_file)[0]
    out_path = os.path.join(output_dir, f"{base_name}_multi.tif")

    # اگر فایل خروجی از قبل موجوده، ردش کن
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"⏩ Already cropped: {out_path}")
        return True

    # تلاش تا ۳ بار
    for attempt in range(1, 4):
        try:
            print(f"\nProcessing: {zip_file} (try {attempt}/3)")

            # پاکسازی temp
            shutil.rmtree(temp_dir, ignore_errors=True)
            os.makedirs(temp_dir, exist_ok=True)

            # اکسترکت
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(temp_dir)

            # پیدا کردن فایل‌های HRIT
            hrit_files = []
            for root, _, files in os.walk(temp_dir):
                for f in files:
                    if f.lower().endswith((".nat", ".hdr", ".img")):
                        hrit_files.append(os.path.join(root, f))

            if not hrit_files:
                print(f"  ⚠️ No HRIT/native files found in {zip_file}")
                return False

            # بارگذاری صحنه
            scn = Scene(filenames=hrit_files, reader='seviri_l1b_native')
            available = [b for b in scn.available_dataset_names() if b not in ('latitude', 'longitude')]
            print(f"  Found {len(available)} channels: {available}")

            scn.load(available)

            # بازنمونه‌برداری
            print("  Resampling to EPSG:4326 ...")
            scn_geo = scn.resample(geo_area, resampler='nearest')

            ds = scn_geo.to_xarray_dataset()
            band_names = list(ds.data_vars.keys())
            band_arrays = [ds[name].values for name in band_names]

            height, width = band_arrays[0].shape
            count = len(band_arrays)
            data_stack = np.stack(band_arrays, axis=0)

            transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

            print(f"  Saving GeoTIFF → {out_path}")
            with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=count,
                dtype=data_stack.dtype,
                crs='EPSG:4326',
                transform=transform,
                compress='LZW',
                tiled=True
            ) as dst:
                for i, band_name in enumerate(band_names, start=1):
                    dst.write(data_stack[i - 1], i)
                    dst.set_band_description(i, band_name)

            print(f"  ✅ Saved {count} bands: {out_path}")
            return True

        except Exception as e:
            print(f"  ⚠️ Error in {zip_file} (attempt {attempt}): {e}")
            traceback.print_exc()
            if attempt < 3:
                print("  🔁 Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"❌ Skipping {zip_file} after 3 failed attempts.")
                return False

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ---------- اجرای حلقه اصلی ----------
for zip_file in os.listdir(input_dir):
    if not zip_file.lower().endswith(".zip"):
        continue

    zip_path = os.path.join(input_dir, zip_file)
    process_zip(zip_path, output_dir, temp_dir, geo_area, lon_min, lat_min, lon_max, lat_max)

print("\n✅ All done. Multi-band GeoTIFFs are ready for QGIS.")
