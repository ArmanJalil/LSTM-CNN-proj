import eumdac
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# احراز هویت
credentials = ('JWrNshABjnfPzGnEYCt6mwGixcQa', 'FaOHSqLieQNbWl9v8o9atH3jRz8a')
token = eumdac.AccessToken(credentials)
datastore = eumdac.DataStore(token)
collection = datastore.get_collection('EO:EUM:DAT:MSG:HRSEVIRI-IODC')

# مسیر ذخیره فایل‌ها
save_dir = "E:/IODC_Images_Jan2023"
os.makedirs(save_dir, exist_ok=True)

# بازه زمانی
start_date = datetime(2024, 3, 1)
end_date = datetime(2024, 4, 1)

# تولید بازه‌های ساعتی
hour_ranges = [(start_date + timedelta(hours=i),
                start_date + timedelta(hours=i + 1))
               for i in range(int((end_date - start_date).total_seconds() // 3600))]

def download_product(dtstart, dtend, position=0):
    products = collection.search(dtstart=dtstart, dtend=dtend)
    products_list = list(products)

    if not products_list:
        return f"⚠️ هیچ محصولی برای {dtstart} پیدا نشد."

    product = products_list[0]
    filename = os.path.join(save_dir, f"{product.sensing_start.strftime('%Y%m%dT%H%M%S')}.zip")

    if os.path.exists(filename):
        return f"⚠️ فایل قبلاً وجود دارد: {filename}"

    try:
        with product.open() as fsrc:
            total_size = int(fsrc.headers.get("Content-Length", 0))
            with open(filename, 'wb') as fdst, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=os.path.basename(filename),
                ascii=True,
                position=position,
                leave=True
            ) as pbar:
                for chunk in iter(lambda: fsrc.read(1024 * 1024), b""):  # 1MB chunks
                    fdst.write(chunk)
                    pbar.update(len(chunk))
        return f"✅ ذخیره شد: {filename}"
    except Exception as e:
        return f"❌ خطا در ذخیره فایل برای {dtstart}: {e}"

# دانلود موازی با حداکثر 3 اتصال (پیشنهاد: 2 یا 3 برای تعادل سرعت)
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {}
    for idx, (dtstart, dtend) in enumerate(hour_ranges):
        futures[executor.submit(download_product, dtstart, dtend, idx % 3)] = dtstart

    for future in as_completed(futures):
        print(future.result())

