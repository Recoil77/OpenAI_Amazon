import os
from dotenv import load_dotenv
import asyncio
import httpx
from tqdm.asyncio import tqdm
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

# === Настройки ===
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL_ASYNC")
assert DATABASE_URL, "DATABASE_URL_ASYNC not set in .env!"

SCORE_MIN = float(os.getenv("SCORE_MIN", 0.9))
SCORE_MAX = float(os.getenv("SCORE_MAX", 1.0))
YEAR_START = int(os.getenv("YEAR_START", 1500))
YEAR_END = int(os.getenv("YEAR_END", 1599))
ENDPOINT_URL = os.getenv("ENDPOINT_URL", "http://192.168.168.10:8000/generate_hypothesis")

# === SQLAlchemy Setup ===
engine = create_async_engine(DATABASE_URL, future=True)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def get_context(session, document_id, chunk_index):
    # Получить верхний чанк (если есть)
    sql_above = text("""
        SELECT cleaned_text FROM public.chunks_metadata_v2
        WHERE document_id = :doc_id AND chunk_index = :idx
        LIMIT 1
    """)
    result_above = await session.execute(sql_above, {"doc_id": document_id, "idx": chunk_index - 1})
    context_above = result_above.scalar()

    # Получить нижний чанк (если есть)
    sql_below = text("""
        SELECT cleaned_text FROM public.chunks_metadata_v2
        WHERE document_id = :doc_id AND chunk_index = :idx
        LIMIT 1
    """)
    result_below = await session.execute(sql_below, {"doc_id": document_id, "idx": chunk_index + 1})
    context_below = result_below.scalar()

    return context_above, context_below

async def process_chunk(session, client, row, context_above, context_below):
    # Не трогаем если гипотеза уже есть
    if row["hypothesis"]:
        return

    # Формируем payload для запроса
    payload = {
        "context_above": context_above,
        "main_chunk": row["cleaned_text"],
        "context_below": context_below
    }

    try:
        response = await client.post(ENDPOINT_URL, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        hypothesis = data.get("hypothesis", "").strip()
    except Exception as exc:
        print(f"Ошибка на id={row['id']}: {exc}")
        return

    # Если гипотеза не получена — пропускаем
    if not hypothesis:
        return

    # Сохраняем в базу
    update_sql = text("""
        UPDATE public.chunks_metadata_v2
        SET hypothesis = :hypothesis
        WHERE id = :id
    """)
    await session.execute(update_sql, {"hypothesis": hypothesis, "id": row["id"]})
    await session.commit()

async def main():
    async with async_session() as session, httpx.AsyncClient() as client:
        # Считаем сколько всего чанков
        total_sql = text("""
            SELECT COUNT(*) FROM public.chunks_metadata_v2
            WHERE object_score BETWEEN :score_min AND :score_max
              AND year >= :year_start AND year <= :year_end
              AND (hypothesis IS NULL OR hypothesis = '')
        """)
        result = await session.execute(total_sql, {
            "score_min": SCORE_MIN, "score_max": SCORE_MAX,
            "year_start": YEAR_START, "year_end": YEAR_END
        })
        total = result.scalar_one()
        print(f"Всего чанков к обработке: {total}")

        # Загружаем все чанки для обработки
        select_sql = text("""
            SELECT id, document_id, chunk_index, cleaned_text, hypothesis
            FROM public.chunks_metadata_v2
            WHERE object_score BETWEEN :score_min AND :score_max
              AND year >= :year_start AND year <= :year_end
              AND (hypothesis IS NULL OR hypothesis = '')
            ORDER BY id
        """)
        rows = (await session.execute(select_sql, {
            "score_min": SCORE_MIN, "score_max": SCORE_MAX,
            "year_start": YEAR_START, "year_end": YEAR_END
        })).mappings().all()

        for row in tqdm(rows, total=total, desc="Generating hypotheses"):
            context_above, context_below = await get_context(session, row["document_id"], row["chunk_index"])
            await process_chunk(session, client, row, context_above, context_below)

    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
