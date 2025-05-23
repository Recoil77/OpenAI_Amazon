#!/usr/bin/env python3

import json, pathlib, sys, requests, tqdm
from urllib.parse import urlparse
number = "1"
# === НАСТРОЙКИ ===
MANIFEST = pathlib.Path(f"docs/{number}/{number}.json")
OUT = pathlib.Path(f"docs/{number}/jpeg")
USE_JPEG_FALLBACK = True
UA = {"User-Agent": "Mozilla/5.0 (script)"}


# === ЗАГРУЗКА МАНИФЕСТА ===
if not MANIFEST.exists():
    sys.exit(f"❌ Файл не найден: {MANIFEST}")
data = json.loads(MANIFEST.read_text("utf-8"))

# --- определяем список canvases ---
canvases = []
if "sequences" in data:                          # v2
    canvases = data["sequences"][0]["canvases"]
elif "items" in data:                            # v3
    canvases = data["items"]
else:
    sys.exit("❌ Не нашли canvases в манифесте")

# === СОЗДАНИЕ ПАПКИ ===
OUT.mkdir(exist_ok=True)

# --- функция, которая достаёт прямую ссылку на изображение ---
def extract_img_url(cv):
    # 1) LOC  (v2) --------------------------------------------
    try:
        return cv["images"][0]["resource"]["@id"]
    except (KeyError, IndexError):
        pass
    # 2) v3  (Harvard / IA) -----------------------------------
    try:
        return cv["items"][0]["items"][0]["body"]["id"]
    except (KeyError, IndexError):
        pass
    # 3) fallback --------------------------------------------
    raise ValueError("нет image-URL в canvas")

# ...

for idx, cv in enumerate(tqdm.tqdm(canvases, desc="downloading"), start=1):
    try:
        img_url = extract_img_url(cv)
        ext     = img_url.rsplit(".", 1)[-1]          # tif / jpg …

        # ► уникальное имя: порядковый номер + расширение
        save_as = OUT / f"{idx:03d}.{ext}"

        if save_as.exists():
            continue

        r = requests.get(img_url, headers=UA, timeout=60)
        r.raise_for_status()
        save_as.write_bytes(r.content)

    except Exception as e:
        print(f"! Ошибка для canvas #{idx} → {e}")
