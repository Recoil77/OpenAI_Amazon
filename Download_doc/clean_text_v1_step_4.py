import os
import json
import asyncio
import httpx
from pathlib import Path

# === НАСТРОЙКИ ===
number = "22"
INPUT_FILE = Path(f"done/{number}/{number}.txt")
OUTPUT_DIR = Path(f"done/{number}/chunks_json")
FINAL_OUTPUT = Path(f"done/{number}/{number}_final.txt")
ENDPOINT = "http://localhost:8000/clean_ocr_extended"
MAX_CHARS = 3000

OUTPUT_DIR.mkdir(exist_ok=True)

# === Новый простой чанкер по пробелам ===
def chunk_text_by_chars(text: str, max_chars: int) -> list[str]:
    words = text.split()
    chunks = []
    current = ""

    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current += (" " if current else "") + word
        else:
            chunks.append(current.strip())
            current = word
    if current:
        chunks.append(current.strip())
    return chunks

def load_existing_chunk(index: int) -> dict | None:
    path = OUTPUT_DIR / f"chunk_{index:04d}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def save_chunk(index: int, data: dict):
    with open(OUTPUT_DIR / f"chunk_{index:04d}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

async def call_llm(text: str) -> dict:
    payload = {"text": text}
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(ENDPOINT, json=payload)
        r.raise_for_status()
        return r.json()

async def main():
    raw_text = INPUT_FILE.read_text(encoding="utf-8")
    chunks = chunk_text_by_chars(raw_text, MAX_CHARS)
    total = len(chunks)
    print(f"🔍 Чанков найдено: {total}")

    for i, chunk in enumerate(chunks):
        if load_existing_chunk(i):
            print(f"✅ chunk_{i:04d} уже обработан.")
            continue

        print(f"⏳ Обработка chunk_{i:04d}…")
        try:
            result = await call_llm(chunk)
            save_chunk(i, result)
        except Exception as e:
            print(f"❌ Ошибка при обработке chunk_{i:04d}: {e}")

    # Сборка финального текста
    print("🧩 Сборка полного документа…")
    with open(FINAL_OUTPUT, "w", encoding="utf-8") as out:
        for i in range(total):
            data = load_existing_chunk(i)
            if data and "cleaned_text" in data:
                out.write(data["cleaned_text"].strip() + "\n\n")

    print(f"📄 Готово: {FINAL_OUTPUT.resolve()}")

# Запуск
if __name__ == "__main__":
    asyncio.run(main())
