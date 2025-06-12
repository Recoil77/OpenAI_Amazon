import os
import json
import asyncio
import httpx
from pathlib import Path
from tqdm import tqdm

# === SETTINGS ===
WORKERS = 16
number = "73"
INPUT_FILE = Path(f"docs/{number}/{number}.txt")
OUTPUT_DIR = Path(f"docs/{number}/chunks_json")
ENDPOINT = "http://192.168.168.10:8000/clean_ocr_extended"
MAX_CHARS = 3000
LOW_SCORE_THRESHOLD = 0.8  # Порог для слабых переводов

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

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

def load_existing_chunk(index: int) -> bool:
    path = OUTPUT_DIR / f"chunk_{index:04d}.json"
    return path.exists()

def save_chunk(index: int, original_text: str, cleaned_text: str, quality_score: float):
    data = {
        "original_text": original_text,
        "cleaned_text": cleaned_text,
        "quality_score": quality_score
    }
    with open(OUTPUT_DIR / f"chunk_{index:04d}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

async def call_llm(text: str) -> dict:
    payload = {"text": text}
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(ENDPOINT, json=payload)
        r.raise_for_status()
        return r.json()

async def process_chunk(index: int, text: str, sem: asyncio.Semaphore, pbar: tqdm,
                        errors: list, low_scores: list):
    async with sem:
        if load_existing_chunk(index):
            pbar.update()
            return
        try:
            #print(text)
            result = await call_llm(text)
            cleaned = result.get("cleaned_text", "").strip()
            score = result.get("quality_score", None)
            if not cleaned or score is None:
                errors.append((index, "empty_or_missing_fields"))
                print(f"❌ chunk_{index:04d}: пустой или некорректный ответ от LLM")
                return
            save_chunk(index, text, cleaned, score)
            if score < LOW_SCORE_THRESHOLD:
                low_scores.append((index, score))
        except Exception as e:
            errors.append((index, str(e)))
            print(f"❌ chunk_{index:04d}: ошибка — {e}")
        finally:
            pbar.update()

async def main():
    raw_text = INPUT_FILE.read_text(encoding="utf-8")
    chunks = chunk_text_by_chars(raw_text, MAX_CHARS)
    total = len(chunks)
    print(f"🔍 Found {total} chunks.")

    sem = asyncio.Semaphore(WORKERS)
    pbar = tqdm(total=total, desc="Processing chunks")
    errors = []
    low_scores = []

    tasks = [
        process_chunk(i, chunk, sem, pbar, errors, low_scores)
        for i, chunk in enumerate(chunks)
    ]
    await asyncio.gather(*tasks)
    pbar.close()

    print("\n📋 Работа завершена.")

    if errors:
        print("❌ Ошибки при обработке следующих чанков:")
        for idx, err in errors:
            print(f"   chunk_{idx:04d} — {err}")
    else:
        print("✅ Все чанки успешно обработаны.")

    if low_scores:
        print("\n⚠️  Низкий quality_score (<{:.2f}) у следующих чанков:".format(LOW_SCORE_THRESHOLD))
        for idx, score in low_scores:
            print(f"   chunk_{idx:04d}: score={score:.2f}")
    else:
        print("👍 Нет чанков с низким quality_score.")

if __name__ == "__main__":
    asyncio.run(main())
