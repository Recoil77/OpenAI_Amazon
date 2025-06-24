import os
import random
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text
from dotenv import load_dotenv
import httpx

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL_ASYNC")
assert DATABASE_URL, "DATABASE_URL_ASYNC not set in .env!"

SCORE_MIN = 0.0
SCORE_MAX = 0.1
YEAR_START = 1500
YEAR_END = 1599
ENDPOINT_URL = "http://192.168.168.10:8000/generate_hypothesis"

engine = create_async_engine(DATABASE_URL, future=True)

async def main():
    async with AsyncSession(engine) as session:
        # 1. Найти все id чанков с нужным скором и годом
        select_ids_sql = text("""
            SELECT id, document_id, chunk_index FROM public.chunks_metadata_v2
            WHERE object_score >= :score_min AND object_score <= :score_max
              AND year >= :year_start AND year <= :year_end
        """)
        res = await session.execute(select_ids_sql, {
            "score_min": SCORE_MIN,
            "score_max": SCORE_MAX,
            "year_start": YEAR_START,
            "year_end": YEAR_END
        })
        rows = res.fetchall()
        if not rows:
            print("Нет подходящих чанков.")
            return

        # 2. Рандомный чанк
        ch = random.choice(rows)
        ch_id, doc_id, ch_index = ch
        print(f"Выбран чанк: id={ch_id}, doc_id={doc_id}, chunk_index={ch_index}")

        # 3. Найти максимальный chunk_index для этого документа
        max_idx_sql = text("""
            SELECT MAX(chunk_index) FROM public.chunks_metadata_v2
            WHERE document_id = :doc_id
        """)
        max_idx = (await session.execute(max_idx_sql, {"doc_id": doc_id})).scalar_one()
        
        # 4. Получить три чанка: верх, основной, низ
        get_chunk = lambda idx: None if idx < 0 or idx > max_idx else idx

        above_idx = get_chunk(ch_index - 1)
        below_idx = get_chunk(ch_index + 1)

        chunk_text_sql = text("""
            SELECT cleaned_text FROM public.chunks_metadata_v2
            WHERE document_id = :doc_id AND chunk_index = :idx
        """)

        # Верхний
        context_above = None
        if above_idx is not None:
            res = await session.execute(chunk_text_sql, {"doc_id": doc_id, "idx": above_idx})
            context_above = res.scalar_one_or_none()
        # Основной
        res = await session.execute(chunk_text_sql, {"doc_id": doc_id, "idx": ch_index})
        main_chunk = res.scalar_one_or_none()
        # Нижний
        context_below = None
        if below_idx is not None:
            res = await session.execute(chunk_text_sql, {"doc_id": doc_id, "idx": below_idx})
            context_below = res.scalar_one_or_none()

        # Печатаем payload
        payload = {
            "context_above": context_above,
            "main_chunk": main_chunk,
            "context_below": context_below
        }
        print("\nPayload для endpoint:")
        for k, v in payload.items():
            print(f"{k}:\n{v}\n{'-'*20}")

        # 5. Отправить на endpoint (если нужно)
        async with httpx.AsyncClient(timeout=160) as client:
            resp = await client.post(ENDPOINT_URL, json=payload)
            print("\nОтвет endpoint:")
            print(resp.text)

if __name__ == "__main__":
    asyncio.run(main())
