import asyncio
import json
from pathlib import Path
import uuid

import asyncpg
import httpx
from tqdm import tqdm

from gateway.gateway_client import embedding

# 1) Конфигурация
CHUNKS_DIR = Path("done/1/chunks_json")  # папка с chunk_0001.json и т.д.
METADATA_URL = "http://localhost:8000/generate_metadata"
DATABASE_URL = "postgresql://postgres:Recoil_post_2002%23@db-dev.fullnode.pro/amazon"

# 2) Параметры документа
DOCUMENT_UUID = uuid.uuid4()  # заменить на реальный UUID
DOCUMENT_NAME = "Hans Peter Kraus Collection: Of the Marañon River and its discovery"
YEAR = 1580
DOC_TYPE = "diary"

# 3) Параллелизм: количество одновременных задач (по умолчанию 8)
MAX_CONCURRENT = 1

def parse_chunk_id(path: Path) -> int:
    return int(path.stem.split("_")[1])

async def process_chunk(conn: asyncpg.Connection, client: httpx.AsyncClient, fp: Path):
    chunk_index = parse_chunk_id(fp)
    raw = json.loads(fp.read_text(encoding="utf-8"))
    text = raw.get("cleaned_text", "").strip()
    if not text:
        raise ValueError(f"chunk {chunk_index} skipped: empty text")

    # 1) Генерация метадаты
    resp = await client.post(
        METADATA_URL,
        json={
            "document_id": DOCUMENT_NAME,
            "year": YEAR,
            "doc_type": DOC_TYPE,
            "text": text
        },
        timeout=600.0
    )
    resp.raise_for_status()
    meta = resp.json()

    # 2) Получение embedding
    emb_resp = await embedding.create(input=meta["text"], model="text-embedding-ada-002")
    raw_vector = emb_resp["data"][0]["embedding"]
    vector_str = "[" + ",".join(str(x) for x in raw_vector) + "]"

    # 3) Формируем JSONB metadata
    metadata_json = {
        "doc_name": DOCUMENT_NAME,
        "year": YEAR,
        "doc_type": DOC_TYPE,
        "entities": meta.get("entities", [])
    }
    metadata_str = json.dumps(metadata_json, ensure_ascii=False)

    # 4) Вставка в БД
    await conn.execute(
        """
        INSERT INTO chunks_metadata
          (document_id, chunk_index, metadata, text, embedding)
        VALUES ($1, $2, $3::jsonb, $4, $5)
        ON CONFLICT DO NOTHING
        """,
        DOCUMENT_UUID,
        chunk_index,
        metadata_str,
        meta.get("text", ""),
        vector_str
    )
    return chunk_index

async def main():
    conn = await asyncpg.connect(DATABASE_URL)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    failed = []

    async with httpx.AsyncClient() as client:
        files = sorted(CHUNKS_DIR.glob("chunk_*.json"))
        print(f"Found {len(files)} chunks.")

        async def sem_task(fp):
            async with sem:
                try:
                    idx = await process_chunk(conn, client, fp)
                    print(f"✔ chunk {idx} inserted")
                except Exception as e:
                    ci = parse_chunk_id(fp)
                    print(f"❌ Error on chunk {ci}: {e}")
                    failed.append(ci)

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

