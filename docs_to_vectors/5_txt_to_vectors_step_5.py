import asyncio
import json
import uuid
from pathlib import Path
import asyncpg
import httpx
from tqdm import tqdm
from gateway.gateway_client import embedding
import os
from dotenv import load_dotenv
load_dotenv("/opt2/.env")

number = "1"

# 1) Конфигурация
CHUNKS_DIR = Path(f"docs/{number}/chunks_json")
METADATA_URL = "http://192.168.168.5:8000/generate_metadata"
DATABASE_URL = env = os.getenv("DATABASE_URL")

# 2) Параметры документа (должны быть заданы вручную)
DOCUMENT_UUID = uuid.uuid4() # uuid.UUID("123e4567-e89b-12d3-a456-426614174000")
#DOCUMENT_UUID = uuid.UUID("b7a22453-f532-47ea-94f4-f41660d6d188")

DOCUMENT_NAME = "Dritte Buch Americae : darinn Brasilia durch Johann Staden auss eigener Erfahrung in teutsch beschrieben"
YEAR = 1610
DOC_TYPE = "book" # , , "map" "book"  "diary"   

# 3) Параллелизм: число одновременных задач
MAX_CONCURRENT = 12

def parse_chunk_index(path: Path) -> int:
    # ожидаем файлы вида chunk_0010.json
    return int(path.stem.split("_")[1])

async def process_chunk(conn: asyncpg.Connection, client: httpx.AsyncClient, fp: Path):
    chunk_index = parse_chunk_index(fp)
    raw = json.loads(fp.read_text(encoding="utf-8"))
    text = raw.get("cleaned_text", "").strip()
    if not text:
        raise ValueError(f"chunk {chunk_index} skipped: empty text")

    # 1) Запрос метаданных
    resp = await client.post(
        METADATA_URL,
        json={
            "document_id": DOCUMENT_NAME,
            "year": YEAR,
            "doc_type": DOC_TYPE,
            "text": text
        },
        timeout=900.0
    )
    resp.raise_for_status()
    meta = resp.json()

    # 2) Получение embedding
    emb_resp = await embedding.create(input=meta["text"], model="text-embedding-3-small")
    raw_vector = emb_resp["data"][0]["embedding"]
    # Преобразование в строку для pgvector
    vector_str = "[" + ",".join(str(x) for x in raw_vector) + "]"

    # 3) Формируем JSONB metadata
    metadata_json = {
        "doc_name": DOCUMENT_NAME,
        "year": YEAR,
        "doc_type": DOC_TYPE,
        "entities": meta.get("entities", [])
    }
    metadata_str = json.dumps(metadata_json, ensure_ascii=False)

    # 4) Вставка в базу
    await conn.execute(
        """
        INSERT INTO chunks_metadata 
          (document_id, chunk_index, metadata, text, embedding)
        VALUES ($1, $2, $3::jsonb, $4, $5)
        """,
        DOCUMENT_UUID,
        chunk_index,
        metadata_str,
        meta.get("text", ""),
        vector_str
    )
    return chunk_index

async def main():
    # Подключаемся к базе
    conn = await asyncpg.connect(DATABASE_URL)

    # Узнаём уже обработанные индексы
    rows = await conn.fetch(
        "SELECT chunk_index FROM chunks_metadata WHERE document_id = $1",
        DOCUMENT_UUID
    )
    done_indexes = {row["chunk_index"] for row in rows}
    print(f"Already processed chunk indexes: {sorted(done_indexes)}")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    failed = []

    async with httpx.AsyncClient() as client:
        files = sorted(CHUNKS_DIR.glob("chunk_*.json"))
        print(f"Found {len(files)} chunk files.")

        async def sem_task(fp: Path):
            async with sem:
                idx = parse_chunk_index(fp)
                if idx in done_indexes:
                    print(f"⏭ chunk {idx} skipped (already in DB)")
                    return
                try:
                    chunk_idx = await process_chunk(conn, client, fp)
                    #print(f"✔ chunk {chunk_idx} inserted")
                except Exception as e:
                    print(f"❌ Error on chunk {idx}: {e}")
                    failed.append(idx)

        tasks = [asyncio.create_task(sem_task(fp)) for fp in files]
        for _ in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing chunks"):
            await _

    await conn.close()

    if failed:
        print(f"\n⚠️ Failed to process chunks: {sorted(failed)}")
    else:
        print("\n✅ All chunks processed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
