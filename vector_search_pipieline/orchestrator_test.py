import requests

BASE = "http://localhost:8000"

# 1) Настройки для пайплайна
payload = {
    "question": "Tribes and Animals of the Prairies and Rocky Mountains",
    "k": 32,
    "bge_threshold": 0.25,
    "semantic_threshold": 0.25
}

# 2) Вызов search_pipeline
resp = requests.post(f"{BASE}/search_pipeline", json=payload)
resp.raise_for_status()
chunks = resp.json()

# 3) Вывод результатов
print(f"Found {len(chunks)} relevant chunks:\n")
for i, c in enumerate(chunks, start=1):
    print(f"--- Chunk #{i} ---")
    print(f"Year:           {c['year']}")
    print(f"Document:       {c['doc_name']} ({c['doc_type']})")
    print(f"Chunk index:    {c['chunk_index']}")
    print(f"BGE score:      {c['bge_score']:.3f}")
    print(f"Semantic score: {c['semantic_score']:.3f}")
    print("Text snippet:")
    print(c['text'])
    print("------------------------------\n")
