import eumdac
import shutil
import os
from datetime import datetime, timedelta

# احراز هویت
credentials = ('JWrNshABjnfPzGnEYCt6mwGixcQa', 'FaOHSqLieQNbWl9v8o9atH3jRz8a')
token = eumdac.AccessToken(credentials)
datastore = eumdac.DataStore(token)
collection = datastore.get_collection('EO:EUM:DAT:MSG:HRSEVIRI-IODC')

# مسیر ذخیره فایل‌ها
save_dir = "D:/IODC_Images_Jan2023"
os.makedirs(save_dir, exist_ok=True)

# بازه زمانی: از 1 ژانویه تا 1 فوریه 2023
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 2, 1)

# تولید بازه‌های ساعتی
hour_ranges = [(start_date + timedelta(hours=i),
                start_date + timedelta(hours=i + 1))
               for i in range(int((end_date - start_date).total_seconds() // 3600))]

# دانلود اولین تصویر از هر ساعت
for dtstart, dtend in hour_ranges:
    products = collection.search(dtstart=dtstart, dtend=dtend)
    products_list = list(products)

    if products_list:
        product = products_list[0]
        filename = os.path.join(save_dir, f"{product.sensing_start.strftime('%Y%m%dT%H%M%S')}.zip")
        try:
            with product.open() as fsrc, open(filename, 'wb') as fdst:
                shutil.copyfileobj(fsrc, fdst)
            print(f"✅ ذخیره شد: {filename}")
        except Exception as e:
            print(f"❌ خطا در ذخیره فایل برای {dtstart}: {e}")
    else:
        print(f"⚠️ هیچ محصولی برای {dtstart} پیدا نشد.")

