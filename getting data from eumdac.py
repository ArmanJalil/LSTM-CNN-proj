import eumdac
from datetime import datetime
import shutil

# احراز هویت
credentials = ('JWrNshABjnfPzGnEYCt6mwGixcQa', 'FaOHSqLieQNbWl9v8o9atH3jRz8a')
token = eumdac.AccessToken(credentials)
datastore = eumdac.DataStore(token)

# گرفتن کالکشن IODC
collection = datastore.get_collection('EO:EUM:DAT:MSG:HRSEVIRI-IODC')

# تعریف بازه زمانی جستجو
dtstart = datetime(2025, 9, 9,11,30)
dtend   = datetime(2025, 9, 9,12,00)

# جستجو برای پیدا کردن محصولات
products = collection.search(dtstart=dtstart, dtend=dtend)

products_list = list(products)  # تبدیل به لیست

print(products_list)
    
if products_list:
    product = products_list[0]
    print(f"✅ URL محصول: {product.url}")
    print(f"📅 زمان سنجش: {product.sensing_start} تا {product.sensing_end}")

    filename = f"D:/{product.sensing_start.strftime('%Y%m%dT%H%M%S')}.zip"

    with product.open() as fsrc, open(filename, 'wb') as fdst:
        shutil.copyfileobj(fsrc, fdst)

    print(f"📥 فایل با موفقیت ذخیره شد: {filename}")
else:
    print("❌ هیچ محصولی پیدا نشد.")