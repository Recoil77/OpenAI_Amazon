import os
import asyncio
import httpx
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text, select, update
from sqlalchemy.orm import sessionmaker
from tqdm.asyncio import tqdm

# --- Подключение к базе и эндпоинт ---
load_dotenv()
DATABASE_URL_ASYNC = os.getenv("DATABASE_URL_ASYNC")
CHECK_OBJECT_URL = "http://192.168.168.10:8000/check_object"

YEAR_START = 1500
YEAR_END = 1599

# --- Подключаем движок и сессию ---
engine = create_async_engine(DATABASE_URL_ASYNC, future=True)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# --- Основная обработка одного чанка ---
async def process_chunk(session, client, row):
    cleaned_text = row["cleaned_text"]
    # Оборачиваем в тройные кавычки на всякий случай
    prompt = f'''"""{cleaned_text}"""'''
    try:
        resp = await client.post(CHECK_OBJECT_URL, json={"text": prompt}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        score = float(data["score"])
    except Exception as e:
        # Ошибки пропускаем, NULL останется, идём дальше
        return False

    # Записываем score
    await session.execute(
        update_chunk_score_sql,
        {"score": score, "id": row["id"]}
    )
    await session.commit()
    return True

# --- SQL запрос для обновления score по id ---
update_chunk_score_sql = text(
    "UPDATE public.chunks_metadata_v2 SET object_score = :score WHERE id = :id"
)

# --- Главный асинхронный цикл ---
async def main():
    async with async_session() as session, httpx.AsyncClient() as client:
        # Считаем сколько всего строк надо обработать (для tqdm)
        total_sql = text(
            "SELECT COUNT(*) FROM public.chunks_metadata_v2 "
            "WHERE object_score IS NULL AND year >= :start AND year <= :end"
        )
        result = await session.execute(total_sql, {"start": YEAR_START, "end": YEAR_END})
        total = result.scalar_one()

        # Получаем result, а затем вызываем .mappings().all()
        select_sql = text(
            "SELECT id, cleaned_text FROM public.chunks_metadata_v2 "
            "WHERE object_score IS NULL AND year >= :start AND year <= :end"
        )
        result = await session.execute(select_sql, {"start": YEAR_START, "end": YEAR_END})
        rows = result.mappings().all()

        # tqdm-асинхронный цикл
        for row in tqdm(rows, total=total, desc="Processing chunks"):
            await process_chunk(session, client, row)

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
