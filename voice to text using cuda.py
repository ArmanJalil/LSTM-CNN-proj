import torch
import whisper
import os

print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 در حال استفاده از دستگاه: {device}")  

# فقط مسیر فایل صوتی را اینجا قرار بده
audio_file_path = r"D:\mes_calasses\hydrogeology\Geomorphology.m4a"

# ساخت مسیر خروجی‌ها به صورت اتوماتیک
base_name = os.path.splitext(audio_file_path)[0]
output_french_text = base_name + "_french.txt"
output_french_srt = base_name + "_french.srt"
output_english_text = base_name + "_translated.txt"
output_english_srt = base_name + "_translated.srt"

# بارگذاری مدل
model = whisper.load_model("medium").to(device)

# --- مرحله ۱: متن و SRT فرانسوی ---
print("🎙️ در حال تبدیل صدا به متن فرانسوی ...")
result_fr = model.transcribe(audio_file_path, task="transcribe", language="fr")

with open(output_french_text, "w", encoding="utf-8") as f:
    f.write(result_fr["text"])
print("✅ متن فرانسوی ذخیره شد:", output_french_text)

def format_timestamp(seconds: float) -> str:
    millis = int((seconds - int(seconds)) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

with open(output_french_srt, "w", encoding="utf-8") as srt_file:
    for i, seg in enumerate(result_fr["segments"], start=1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        srt_file.write(f"{i}\n{start} --> {end}\n{text}\n\n")
print("✅ فایل SRT فرانسوی ذخیره شد:", output_french_srt)

# --- مرحله ۲: متن و SRT انگلیسی ---
print("🌍 در حال ترجمه صدا به انگلیسی ...")
result_en = model.transcribe(audio_file_path, task="translate")

with open(output_english_text, "w", encoding="utf-8") as f:
    f.write(result_en["text"])
print("✅ متن انگلیسی ذخیره شد:", output_english_text)

with open(output_english_srt, "w", encoding="utf-8") as srt_file:
    for i, seg in enumerate(result_en["segments"], start=1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        srt_file.write(f"{i}\n{start} --> {end}\n{text}\n\n")
print("✅ فایل SRT انگلیسی ذخیره شد:", output_english_srt)
