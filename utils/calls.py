import httpx
long_timeout = httpx.Timeout(
    connect=60.0,   # 30 секунд на установку соединения
    read=300.0,     # 120 секунд на чтение ответа
    write=300.0,    # 120 секунд на отправку данных
    pool=60.0       # 30 секунд ожидания свободного подключения
)

async def call_fastapi_async(url: str, payload: dict = None, files: dict = None) -> dict:
    async with httpx.AsyncClient(timeout=long_timeout) as client:
        try:
            if files is not None:
                response = await client.post(url, files=files, timeout=60)
            else:
                response = await client.post(url, json=payload, timeout=60)
            response.raise_for_status()
            if response.status_code == 204 or not response.content:
                return {}
            return response.json()
        except Exception as e:
            print("Error calling FastAPI endpoint:", e)
            return {"error": str(e)}