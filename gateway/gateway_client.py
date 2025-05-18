import httpx
from fastapi import UploadFile
import os
from dotenv import load_dotenv
load_dotenv(".env")

CHAT_URL =              "http://192.168.168.2:8000/v1/chat/completions"
EMBEDDING_ENDPOINT_G =  "http://192.168.168.2:8000/embeddings"
AUDIO_ENDPOINT =        "http://192.168.168.2:8000/audio/transcriptions"
REASONING_ENDPOINT =    "http://192.168.168.2:8000/v1/responses"



class OpenAIChatMessage:
    def __init__(self, data):
        self.content = data["message"]["content"]

class OpenAIChatChoice:
    def __init__(self, data):
        self.message = OpenAIChatMessage(data)

class OpenAIChatResponseProxy:
    def __init__(self, raw: dict):
        self.choices = [OpenAIChatChoice(choice) for choice in raw["choices"]]


class GatewayChatCompletion:
    @staticmethod
    async def create(**kwargs):
        payload = {"chat_completion": kwargs}
        async with httpx.AsyncClient(timeout=900) as client:
            response = await client.post(CHAT_URL, json=payload)
            response.raise_for_status()
            raw = response.json()
            return OpenAIChatResponseProxy(raw)

chat_completion = GatewayChatCompletion()

class GatewayEmbedding:
    @staticmethod
    async def create(**kwargs):
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(EMBEDDING_ENDPOINT_G, json=kwargs)
            response.raise_for_status()
            return response.json()

embedding = GatewayEmbedding()

# gateway_client.py

class GatewayTranscription:
    @staticmethod
    async def create(file: UploadFile, model: str = "whisper-1", **kwargs):
        files = {
            "file": (file.filename, await file.read(), file.content_type),
            "model": (None, model)
        }
        for k, v in kwargs.items():
            files[k] = (None, v)

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(AUDIO_ENDPOINT, files=files)
            response.raise_for_status()
            return response.json()

transcribe = GatewayTranscription()




class GatewayResponse:
    @staticmethod
    async def create(**kwargs):
        async with httpx.AsyncClient(timeout=900) as client:
            response = await client.post(REASONING_ENDPOINT, json=kwargs)
            response.raise_for_status()
            return response.json()

response_completion = GatewayResponse()


