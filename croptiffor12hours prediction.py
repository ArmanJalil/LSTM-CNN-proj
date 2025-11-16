import os
import rasterio
from rasterio.windows import from_bounds
import traceback

# ----------------------------------------------------------------------
# -------------------------- CONFIGURATION -----------------------------
# ----------------------------------------------------------------------
input_dir  = r"D:/cropped meteo for 2 days"   # folder with original *_multi.tif
output_dir = r"D:/images for1 houre"             # folder to save cropped images
os.makedirs(output_dir, exist_ok=True)

# ----- Crop box (Isfahan region) -----
lon_min, lon_max = 51.0, 52.0   # °E
lat_min, lat_max = 32.0, 33.0   # °N


# ----------------------------------------------------------------------
# --------------------------- MAIN PROCESS -----------------------------
# ----------------------------------------------------------------------
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(".tif"):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    try:
        with rasterio.open(input_path) as src:
            # Ensure the CRS is geographic (EPSG:4326)
            if src.crs.to_string() != "EPSG:4326":
                print(f"⚠️ Skipping {filename} — CRS is not EPSG:4326.")
                continue

            # Create a rasterio window for the bounding box
            window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)

            # Read the data inside the window
            data = src.read(window=window)

            # Update metadata
            transform = src.window_transform(window)
            meta = src.meta.copy()
            meta.update({
                "height": data.shape[1],
                "width": data.shape[2],
                "transform": transform
            })

            # Write cropped file
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(data)

            print(f"✅ Cropped: {filename}")

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        traceback.print_exc()

print("\n🎯 All done! Cropped files saved to:", output_dir)
