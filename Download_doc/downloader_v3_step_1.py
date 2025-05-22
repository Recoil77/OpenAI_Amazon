#!/usr/bin/env python3

import json, pathlib, sys, requests, tqdm
from urllib.parse import urlparse
number = "16"
# === НАСТРОЙКИ ===
MANIFEST = pathlib.Path(f"done/{number}/{number}.json")
OUT = pathlib.Path(f"done/{number}/tiff")
USE_JPEG_FALLBACK = True
UA = {"User-Agent": "Mozilla/5.0 (script)"}


# === ЗАГРУЗКА МАНИФЕСТА ===
if not MANIFEST.exists():
    sys.exit(f"❌ Файл не найден: {MANIFEST}")
data = json.loads(MANIFEST.read_text("utf-8"))

canvases = data["sequences"][0]["canvases"]
print("Страниц:", len(canvases))

# === ОПРЕДЕЛЕНИЕ master-префикса ===
sample = canvases[0]["@id"]
parsed = urlparse(sample)

try:
    after_iiif = parsed.path.split("/iiif/", 1)[1]
except IndexError:
    sys.exit("❌ canvas['@id'] не содержит '/iiif/'")

segments = after_iiif.split(":")
if len(segments) < 2:
    sys.exit("❌ Не удалось выделить путь к TIFF")

# Убираем 'service' если есть
if segments[0] == "service":
    segments = segments[1:]

storage_path = "/".join(segments[:-1])
MASTER = f"https://tile.loc.gov/storage-services/master/{storage_path}"
print("Master-prefix:", MASTER)

# === СОЗДАНИЕ ПАПКИ ===
OUT.mkdir(exist_ok=True)

# === СКАЧИВАНИЕ ===
for cv in tqdm.tqdm(canvases, desc="downloading"):
    try:
        canvas_id = cv["@id"]
        file_id = canvas_id.split(":")[-1]
        tif_path = OUT / f"{file_id}.tif"
        jpg_path = tif_path.with_suffix(".jpg")

        # ⛔ Пропустить, если уже скачан любой формат
        if tif_path.exists() or jpg_path.exists():
            continue

        # 🔽 Сначала пробуем TIFF
        url = f"{MASTER}/{file_id}.tif"
        r = requests.get(url, headers=UA, timeout=60)

        # 🔁 Если TIFF 404 — fallback на JPEG
        if r.status_code == 404 and USE_JPEG_FALLBACK:
            jpeg_url = f"{canvas_id}/full/pct:100/0/default.jpg"
            r = requests.get(jpeg_url, headers=UA, timeout=60)
            r.raise_for_status()
            jpg_path.write_bytes(r.content)
        else:
            r.raise_for_status()
            tif_path.write_bytes(r.content)

    except Exception as e:
        print(f"! Ошибка для {cv.get('label', '?')} → {e}")

print("✔ Готово! Файлы сохранены в", OUT.resolve())
