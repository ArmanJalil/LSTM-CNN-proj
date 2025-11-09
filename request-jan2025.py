import eumdac
import shutil
import os
import time
from datetime import datetime, timedelta

# احراز هویت
credentials = ('JWrNshABjnfPzGnEYCt6mwGixcQa', 'FaOHSqLieQNbWl9v8o9atH3jRz8a')
token = eumdac.AccessToken(credentials)
datastore = eumdac.DataStore(token)
collection = datastore.get_collection('EO:EUM:DAT:MSG:HRSEVIRI-IODC')

# مسیر ذخیره فایل‌ها
save_dir = "e:/data"
os.makedirs(save_dir, exist_ok=True)

# بازه زمانی: از 1 ژانویه تا 1 فوریه 2024
start_date = datetime(2024, 4, 22)
end_date = datetime(2024, 4, 23)

# تولید بازه‌های ساعتی
hour_ranges = [(start_date + timedelta(hours=i),
                start_date + timedelta(hours=i + 1))
               for i in range(int((end_date - start_date).total_seconds() // 3600))]

# دانلود با ری‌ترای
def download_product(product, filename, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            with product.open() as fsrc, open(filename, 'wb') as fdst:
                shutil.copyfileobj(fsrc, fdst)                                                                                                                                                                                                                                                                                          
            print(f"✅ ذخیره شد: {filename}")
            return True
        except Exception as e:
            print(f"⚠️ خطا در دانلود (تلاش {attempt}/{max_retries}): {e}")
            if os.path.exists(filename):
                os.remove(filename)  # فایل ناقص حذف میشه
            time.sleep(5)  # کمی صبر قبل از تلاش دوباره
    print(f"❌ شکست در دانلود بعد از {max_retries} بار: {filename}")
    return False

# دانلود اولین تصویر از هر ساعت
for dtstart, dtend in hour_ranges:
    products = collection.search(dtstart=dtstart, dtend=dtend)
    products_list = list(products)

    if products_list:
        product = products_list[0]
        filename = os.path.join(save_dir, f"{product.sensing_start.strftime('%Y%m%dT%H%M%S')}.zip")

        # اگر قبلا فایل درست دانلود شده باشه، رد میشه
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            print(f"⏩ قبلا دانلود شده: {filename}")
            continue

        download_product(product, filename)

    else:
        print(f"⚠️ هیچ محصولی برای {dtstart} پیدا نشد.")

