#!/usr/bin/env python3
import asyncio
import aiohttp
import json
from pathlib import Path
from aiofiles import open as aio_open
from aiofiles.os import path as aio_path
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

# === НАСТРОЙКИ ===
INPUT_DIR = Path("Download_doc/working/jpeg")
TMP_DIR = Path("Download_doc/working/tmp")
# INPUT_DIR = Path("done/3/jpeg")
# TMP_DIR = Path("done/3/tmp")
FINAL_OUTPUT = Path("Download_doc/working/final_output.txt")
ENDPOINT = "http://127.0.0.1:8000/ocr_main_text_o4mini"
CONCURRENCY = 8  # Количество потоков

# === Создание tmp-папки ===
TMP_DIR.mkdir(exist_ok=True)

async def process_batch(name, session, batch, total):
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

            async with session.post(ENDPOINT, data=data, timeout=aiohttp.ClientTimeout(total=300)) as resp:
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
            process_batch(i, session, chunk, total)
            for i, chunk in enumerate(chunks)
        ]
        await asyncio.gather(*tasks)

    # Проверяем, всё ли обработано
    tmp_files = sorted(TMP_DIR.glob("*.json"))
    if len(tmp_files) < total:
        print(f"⏸ Всего обработано: {len(tmp_files)} из {total}. Итог не собирается.")
        return



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
                    lines = lines[1:]

                clean_text = "\n".join(lines).strip()

                # 💾 Записываем без метки filename, только текст
                await out.write(clean_text + "\n\n")

    print(f"✅ Финальный файл сохранён в {FINAL_OUTPUT}")

if __name__ == "__main__":
    asyncio.run(main())
