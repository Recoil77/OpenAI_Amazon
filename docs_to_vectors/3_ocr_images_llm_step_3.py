#!/usr/bin/env python3
import asyncio
import aiohttp
import json
from pathlib import Path
from aiofiles import open as aio_open
from aiofiles.os import path as aio_path
from tqdm import tqdm

number = "54"

# === НАСТРОЙКИ ===
INPUT_DIR = Path(f"docs/{number}/jpeg")
TMP_DIR = Path(f"docs/{number}/tmp")
FINAL_OUTPUT = Path(f"docs/{number}/{number}.txt")
ENDPOINT = "http://192.168.168.10:8000/ocr_main_text"
CONCURRENCY = 8  # Количество потоков

# === Создание tmp-папки ===
TMP_DIR.mkdir(exist_ok=True)

async def process_batch(name, session, batch):
    bar = tqdm(batch, desc=f"Поток {name}", position=name, leave=False)
    for img_path in bar:
        tmp_path = TMP_DIR / (img_path.stem + ".json")
        if await aio_path.exists(tmp_path):
            bar.set_postfix_str(f"✅ Пропущено: {img_path.name}")
            continue

        try:
            with img_path.open("rb") as f:
                img_data = f.read()

            data = aiohttp.FormData()
            data.add_field("file", img_data, filename=img_path.name, content_type="image/jpeg")

            async with session.post(ENDPOINT, data=data, timeout=aiohttp.ClientTimeout(total=900)) as resp:
                resp.raise_for_status()
                result = await resp.json()
                text = result.get("text", "").strip()

                async with aio_open(tmp_path, "w", encoding="utf-8") as f:
                    await f.write(json.dumps({
                        "filename": img_path.name,
                        "text": text
                    }, ensure_ascii=False, indent=2))

                bar.set_postfix_str(f"✅ Готово: {img_path.name}")
        except Exception as e:
            bar.set_postfix_str(f"❌ Ошибка: {img_path.name} → {e}")

async def main():
    images = sorted(INPUT_DIR.glob("*.jpg"))
    total = len(images)
    print(f"🔍 Найдено изображений: {total}")

    # Разбиваем на N частей
    chunks = [images[i::CONCURRENCY] for i in range(CONCURRENCY)]

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_batch(i, session, chunk)
            for i, chunk in enumerate(chunks)
        ]
        await asyncio.gather(*tasks)

    # Проверяем, что обработано
    tmp_files = sorted(TMP_DIR.glob("*.json"))
    processed = len(tmp_files)

    if processed < total:
        print(f"⏸ Всего обработано: {processed} из {total}.")
        missing = [img.name for img in images if not (TMP_DIR / (img.stem + ".json")).exists()]
        if missing:
            print("❗ Не обработаны следующие изображения:")
            for name in missing:
                print(f" - {name}")
        return

    print(f"✅ Все изображения обработаны: {processed} из {total}.")
    print("📦 Собираем финальный файл…")

    async with aio_open(FINAL_OUTPUT, "w", encoding="utf-8") as out:
        for tmp in tmp_files:
            async with aio_open(tmp, "r", encoding="utf-8") as f:
                data = json.loads(await f.read())
                text = data.get("text", "").strip()

                # 🔍 Очищаем от вставок, markdown и префиксов
                lines = text.splitlines()
                junk_prefixes = [
                    "here is", "respuesta:", "transcripción", "```", "respuesta extraída",
                    "extracted main body text", "respuesta ocr", "json", "output:"
                ]
                while lines and any(p in lines[0].lower() for p in junk_prefixes):
                    lines.pop(0)

                clean_text = "\n".join(lines).strip()
                await out.write(clean_text + "\n\n")

    print(f"✅ Финальный файл сохранён в {FINAL_OUTPUT}")

if __name__ == "__main__":
    asyncio.run(main())
