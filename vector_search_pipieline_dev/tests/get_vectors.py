import requests

# URL твоего FastAPI сервера
API_URL = "http://localhost:8000/vector_search"  # измени порт, если другой

# Тестовый запрос
payload = {
    "query": "During that month, we had covered only about 110 kilometers, and had descended nearly 150 meters—the figures are approximate but fairly accurate",
    "k": 3
}

response = requests.post(API_URL, json=payload)
response.raise_for_status()

data = response.json()
for i, chunk in enumerate(data.get("results", []), 1):
    print(f"\n=== RESULT #{i} ===")
    print(f"Year: {chunk['year']}")
    print(f"Document: {chunk['doc_name']}")
    print(f"Type: {chunk['doc_type']}")
    print(f"Chunk index: {chunk['chunk_index']}")
    print("Text:")
    print(chunk['text'])

