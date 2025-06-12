import os
import json
import asyncio
import httpx
from pathlib import Path
from tqdm import tqdm

# === НАСТРОЙКИ ===
WORKERS = 16
number = "73"
CHUNKS_DIR = Path(f"docs/{number}/chunks_json")
OUTPUT_DIR = Path(f"docs/{number}/chunks_with_metadata")
META_PATH = Path(f"docs/{number}/meta.json")
METADATA_URL = "http://192.168.168.10:8000/generate_metadata"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def load_meta():
    if not META_PATH.exists():
        raise RuntimeError(f"❌ meta.json не найден: {META_PATH.resolve()}")
    return json.loads(META_PATH.read_text("utf-8"))

def load_chunk(index: int) -> dict | None:
    path = CHUNKS_DIR / f"chunk_{index:04d}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8"))

def load_existing_metadata_chunk(index: int) -> bool:
    path = OUTPUT_DIR / f"chunk_{index:04d}.json"
    return path.exists()

def save_metadata_chunk(index: int, data: dict):
    with open(OUTPUT_DIR / f"chunk_{index:04d}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

async def call_metadata(text: str) -> dict | None:
    payload = {
        "document_id": "unknown",  # значение не используется, но поле нужно
        "text": text
    }
    async with httpx.AsyncClient(timeout=900) as client:
        r = await client.post(METADATA_URL, json=payload)
        r.raise_for_status()
        return r.json()

async def process_chunk(index: int, sem: asyncio.Semaphore, pbar: tqdm, errors: list):
    async with sem:
        # Пропуск, если файл с метаданными уже есть
        if load_existing_metadata_chunk(index):
            pbar.update()
            return
        chunk = load_chunk(index)
        if not chunk:
            errors.append((index, "chunk_not_found"))
            print(f"❌ chunk_{index:04d}: файл не найден")
            pbar.update()
            return

        orig = chunk.get("original_text", "")
        trans = chunk.get("cleaned_text", "")
        score = chunk.get("quality_score", None)

        if not orig or not trans or score is None:
            errors.append((index, "missing_fields"))
            print(f"❌ chunk_{index:04d}: отсутствует текст или score")
            pbar.update()
            return

        try:
            meta_orig = await call_metadata(orig)
            entities_orig = meta_orig.get("entities", None)
            if not entities_orig:
                errors.append((index, "meta_original_empty"))
                print(f"❌ chunk_{index:04d}: пустая или невалидная metadata_original")
                pbar.update()
                return

            meta_trans = await call_metadata(trans)
            entities_trans = meta_trans.get("entities", None)
            if not entities_trans:
                errors.append((index, "meta_translated_empty"))
                print(f"❌ chunk_{index:04d}: пустая или невалидная metadata_translated")
                pbar.update()
                return

            meta = load_meta()
            out = {
                "document_id": meta["document_id"],
                "doc_name": meta["doc_name"],
                "year": meta["year"],
                "doc_type": meta["doc_type"],

                "original_text": orig,
                "cleaned_text": trans,
                "quality_score": score,

                "metadata_original": {"entities": entities_orig},
                "metadata_translated": {"entities": entities_trans}
            }
            save_metadata_chunk(index, out)
        except Exception as e:
            errors.append((index, str(e)))
            print(f"❌ chunk_{index:04d}: ошибка — {e}")
        finally:
            pbar.update()

async def main():
    chunk_files = sorted(CHUNKS_DIR.glob("chunk_*.json"))
    total = len(chunk_files)
    print(f"🔍 Найдено чанков: {total}")

    sem = asyncio.Semaphore(WORKERS)
    pbar = tqdm(total=total, desc="Metadata gen")
    errors = []

    tasks = [
        process_chunk(i, sem, pbar, errors)
        for i in range(total)
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

if __name__ == "__main__":
    asyncio.run(main())
