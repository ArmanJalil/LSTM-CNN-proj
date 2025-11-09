from satpy import Scene
from glob import glob

# مسیر فایل HRIT
file_path = r"C:\Users\arman\Downloads\Compressed\202405201600\*"

# پیدا کردن تمام فایل‌های مرتبط با این زمان
filenames = glob(file_path)

# ساخت صحنه (Scene) از فایل‌های HRIT
scn = Scene(reader='seviri_l1b_hrit', filenames=filenames)

# بارگذاری کانال بخار آب 7.3 میکرون
scn.load(['WV_073'])

# نمایش تصویر با matplotlib
scn.show('WV_073')


print(scn['WV_073'].attrs)
