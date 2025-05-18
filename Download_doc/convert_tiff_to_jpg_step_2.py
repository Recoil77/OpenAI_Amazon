#!/usr/bin/env python3
import os
from pathlib import Path
from PIL import Image

# === Настройки ===
INPUT_DIR = Path("Download_doc/working/tiff")      # Замените на нужную директорию
OUTPUT_DIR = Path("Download_doc/working/jpeg") 
OUTPUT_DIR.mkdir(exist_ok=True)

# === Проход по всем .tiff ===
for tiff_path in sorted(INPUT_DIR.glob("*.tif*")):
    img = Image.open(tiff_path)

    # Преобразование в RGB (если изображение в другом режиме)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Сохраняем в .jpg
    jpg_path = OUTPUT_DIR / (tiff_path.stem + ".jpg")
    img.save(jpg_path, "JPEG", quality=95)

    print(f"✅ {tiff_path.name} → {jpg_path.name}")
