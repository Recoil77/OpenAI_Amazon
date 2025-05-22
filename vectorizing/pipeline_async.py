import asyncio
import json
from pathlib import Path

import asyncpg
import httpx
from tqdm import tqdm

from gateway.gateway_client import embedding

# 1) Конфигурация
CHUNKS_DIR = Path("done/1/chunks_json")  # папка с chunk_0001.json и т.д.
METADATA_URL = "http://localhost:8000/generate_metadata"
DATABASE_URL = "postgresql://postgres:Recoil_post_2002%23@db-dev.fullnode.pro/amazon"

# 2) Параметры документа
DOCUMENT_ID = "Hans Peter Kraus Collection: Of the Marañon River and its discovery"
YEAR = 1580
DOC_TYPE = "diary"

# 3) Параллелизм: количество одновременных задач (по умолчанию 8)
MAX_CONCURRENT = 4

def parse_chunk_id(path: Path) -> int:
    return int(path.stem.split("_")[1])

async def process_chunk(conn: asyncpg.Connection, client: httpx.AsyncClient, fp: Path):
    chunk_id = parse_chunk_id(fp)
    raw = json.loads(fp.read_text(encoding="utf-8"))
    text = raw.get("cleaned_text", "").strip()
    if not text:
        raise ValueError(f"chunk {chunk_id} skipped: empty text")

    # 1) Генерация метадаты
    resp = await client.post(
        METADATA_URL,
        json={
            "document_id": DOCUMENT_ID,
            "year": YEAR,
            "doc_type": DOC_TYPE,
            "text": text
        },
        timeout=600.0
    )
    resp.raise_for_status()
    meta = resp.json()
    print(meta['entities'])
    # 2) Получение embedding
    emb_resp = await embedding.create(input=meta["text"], model="text-embedding-ada-002")
    raw_vector = emb_resp["data"][0]["embedding"]
    vector = "[" + ",".join(str(x) for x in raw_vector) + "]"

    # 3) Вставка в БД
    await conn.execute(
        """
        INSERT INTO chunks_metadata
          (chunk_id, document_id, year, doc_type, entities, text, embedding)
        VALUES ($1,$2,$3,$4,$5,$6,$7)
        ON CONFLICT (chunk_id) DO NOTHING
        """,
        chunk_id,
        DOCUMENT_ID,
        YEAR,
        DOC_TYPE,
        meta.get("entities", []),
        meta.get("text", ""),
        vector
    )
    return chunk_id

async def main():
    conn = await asyncpg.connect(DATABASE_URL)
    failed = []
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async with httpx.AsyncClient() as client:
        files = sorted(CHUNKS_DIR.glob("chunk_*.json"))
        print(f"Found {len(files)} chunks.")

        async def sem_task(fp):
            async with sem:
                try:
                    cid = await process_chunk(conn, client, fp)
                    return cid, None
                except Exception as e:
                    return None, (parse_chunk_id(fp), str(e))

        tasks = [asyncio.create_task(sem_task(fp)) for fp in files]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing chunks"):
            cid, error = await coro
            if error:
                failed.append(error[0])
                print(f"❌ Error on chunk {error[0]}: {error[1]}")
            else:
                print(f"✔ chunk {cid} inserted")

    await conn.close()

    if failed:
        print(f"\n⚠️ Failed to process chunks: {sorted(failed)}")
    else:
        print("\n✅ All chunks processed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
