import os
import json
import asyncio
from pathlib import Path
from tqdm import tqdm
import asyncpg
from gateway.gateway_client import embedding

DATABASE_URL = os.getenv("DATABASE_URL") 
number = "40"
CHUNKS_DIR = Path(f"docs/{number}/chunks_with_metadata")

async def chunk_exists(conn, document_id, chunk_index):
    row = await conn.fetchrow(
        "SELECT 1 FROM chunks_metadata_v2 WHERE document_id = $1 AND chunk_index = $2",
        document_id, chunk_index
    )
    return row is not None

def embedding_to_pgvector(vec):
    return "[" + ",".join(str(x) for x in vec) + "]"

async def main():
    chunk_files = sorted(CHUNKS_DIR.glob("chunk_*.json"))
    total = len(chunk_files)
    print(f"🔍 Найдено чанков: {total}")

    conn = await asyncpg.connect(DATABASE_URL)
    errors = []
    pbar = tqdm(total=total, desc="DB insert")

    for f in chunk_files:
        try:
            chunk = json.loads(f.read_text(encoding="utf-8"))
            document_id = chunk["document_id"]
            doc_name = chunk["doc_name"]
            year = int(chunk["year"]) if chunk["year"] is not None else None
            doc_type = chunk["doc_type"]
            chunk_index = int(f.stem.split("_")[1])
            original_text = chunk["original_text"]
            cleaned_text = chunk["cleaned_text"]
            quality_score = float(chunk["quality_score"])
            metadata_original = json.dumps(chunk["metadata_original"], ensure_ascii=False)
            metadata_translated = json.dumps(chunk["metadata_translated"], ensure_ascii=False)

            # Пропуск если уже есть
            if await chunk_exists(conn, document_id, chunk_index):
                pbar.update()
                continue

            # Получаем embedding
            emb_resp = await embedding.create(input=cleaned_text, model="text-embedding-3-small")
            raw_vector = emb_resp["data"][0]["embedding"]
            vector_str = embedding_to_pgvector(raw_vector)

            await conn.execute(
                """
                INSERT INTO chunks_metadata_v2 (
                    document_id, doc_name, year, doc_type,
                    chunk_index, original_text, cleaned_text, quality_score,
                    metadata_original, metadata_translated,
                    embedding
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11)
                """,
                document_id, doc_name, year, doc_type,
                chunk_index, original_text, cleaned_text, quality_score,
                metadata_original, metadata_translated,
                vector_str
            )
        except Exception as e:
            errors.append((f.name, str(e)))
        finally:
            pbar.update()

    pbar.close()
    await conn.close()

    print("\n📋 Загрузка завершена.")

    if errors:
        print("❌ Ошибки при обработке файлов:")
        for fname, err in errors:
            print(f"   {fname} — {err}")
    else:
        print("✅ Все чанки успешно загружены.")

if __name__ == "__main__":
    asyncio.run(main())
