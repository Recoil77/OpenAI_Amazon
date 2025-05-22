import asyncio
import json
import asyncpg
from gateway.gateway_client import embedding

# Текст для поиска похожих чанков
TEXT_TO_SEARCH = """
During that month, we had covered only about 110 kilometers, and had descended nearly 150 meters—the figures are approximate but fairly accurate
"""

# Конфигурация подключения к PostgreSQL (pgvector)
DATABASE_URL = "postgresql://postgres:Recoil_post_2002%23@db-dev.fullnode.pro/amazon"

async def main():
    # 1) Получаем embedding для запроса
    emb_resp = await embedding.create(input=TEXT_TO_SEARCH, model="text-embedding-3-small")
    vector = emb_resp["data"][0]["embedding"]
    # Преобразуем список float в строку для параметра pgvector
    vector_str = "[" + ",".join(str(x) for x in vector) + "]"

    # 2) Подключаемся к базе
    conn = await asyncpg.connect(DATABASE_URL)

    # 3) Выполняем запрос на ближайший вектор и выводим нужные поля
    row = await conn.fetchrow(
        """
        SELECT
          metadata->>'year'    AS year,
          metadata->>'doc_name' AS doc_name,
          metadata->>'doc_type' AS doc_type,
          chunk_index,
          text
        FROM chunks_metadata
        ORDER BY embedding <=> $1
        LIMIT 1
        """,
        vector_str
    )

    # 4) Выводим результат
    if row:
        print("Most similar chunk:")
        print(f"Year: {row['year']}")
        print(f"Document Name: {row['doc_name']}")
        print(f"Document Type: {row['doc_type']}")
        print(f"Chunk Index: {row['chunk_index']}")
        print("Text:")
        print(row['text'])
    else:
        print("No chunks found in database.")

    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
