import os
import subprocess
from pathlib import Path

# === НАСТРОЙКИ ===
number = "37"                  # ← номер документа
first_page = 7                 # ← начальная страница (включительно)
last_page = 436                # ← последняя страница (включительно), None = до конца
use_pdftoppm = True            # ← True = использовать pdftoppm (иначе pdfimages)
dpi = 400                      # ← разрешение
output_format = "jpeg"         # ← 'tiff' или 'jpeg'
jpeg_quality = "95"            # ← применимо только если output_format = 'jpeg'

BASE = Path(f"docs/{number}")
PDF = BASE / f"{number}.pdf"
OUT = BASE / output_format

# === ПРОВЕРКИ ===
if not PDF.exists():
    print(f"❌ PDF не найден: {PDF}")
    exit(1)

OUT.mkdir(parents=True, exist_ok=True)

# === СБОР КОМАНДЫ ===
if use_pdftoppm:
    out_prefix = OUT / number
    cmd = ["pdftoppm", f"-{output_format}", "-r", str(dpi)]
    if output_format == "jpeg":
        cmd += ["-jpegopt", f"quality={jpeg_quality}"]
    if first_page:
        cmd += ["-f", str(first_page)]
    if last_page:
        cmd += ["-l", str(last_page)]
    cmd += [str(PDF), str(out_prefix)]
else:
    # pdfimages всегда tiff, независимо от output_format
    cmd = ["pdfimages", "-tiff", "-p"]
    if first_page:
        cmd += ["-f", str(first_page)]
    if last_page:
        cmd += ["-l", str(last_page)]
    cmd += [str(PDF), str(OUT / number)]

# === ЗАПУСК ===
print(f"📥 Извлекаем страницы {first_page}–{last_page or 'EOF'} из {PDF.name}")
print("Команда:", " ".join(str(c) for c in cmd))
subprocess.run(cmd, check=True)
print(f"✔️ Изображения сохранены в: {OUT.resolve()}")

# === УДАЛЕНИЕ ЛИШНИХ TIFF-файлов (pdfimages: только *-000.tif) ===
if not use_pdftoppm and output_format == "tiff":
    for file in OUT.glob(f"{number}-*.tif"):
        if not file.stem.endswith("-000"):
            print(f"🗑 Удаляем неосновной TIFF: {file.name}")
            file.unlink()

print("🎯 Готово!")