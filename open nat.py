from satpy import Scene
import matplotlib.pyplot as plt

# مسیر فایل
filename = "D:\IODC_Images_Jan2023\MSG2-SEVI-MSG15-0100-NA-20240101005740.457000000Z-NA.nat"

# ساخت Scene با reader مخصوص HRIT SEVIRI
scn = Scene(filenames=[filename], reader='seviri_l1b_native')

# نمایش کانال‌هایی که قابل بارگذاری هستن
print("کانال‌های موجود:", scn.available_dataset_names())

# بارگذاری یکی از کانال‌ها (مثلاً VIS006 یا IR_108 یا WV_062)
scn.load(['VIS006'])

# تصویرسازی
img = scn['VIS006'].values

plt.imshow(img, cmap='gray')
plt.title('SEVIRI VIS006 Channel')
plt.colorbar()
plt.show()

