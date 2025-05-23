import requests

BASE = "http://192.168.168.5:8000"

# 1) Настройки для пайплайна
payload = {
    "question": "description and location of the river Pogubu", # "we had to cross the river Pogubu"
    "k": 64,
    "bge_threshold": 0.1,
    "semantic_threshold": 0.25
}

# 2) Вызов search_pipeline
resp = requests.post(f"{BASE}/search_pipeline", json=payload)
resp.raise_for_status()
chunks = resp.json()

# 3) Вывод результатов с фактами
print(f"Found {len(chunks)} relevant chunks:\n")
for i, c in enumerate(chunks, start=1):
    print(f"--- Chunk #{i} ---")
    print(f"Year:            {c['year']}")
    print(f"Document:        {c['doc_name']} ({c['doc_type']})")
    print(f"Chunk index:     {c['chunk_index']}")
    print(f"BGE score:       {c['bge_score']:.3f}")
    print(f"Semantic score:  {c['semantic_score']:.3f}")
    print("Text snippet:")
    print(c['text'][:100], "...")
    print("\nExtracted facts:")
    if c.get('facts'):
        for fact in c['facts']:
            print(f"  - {fact}")
    else:
        print("  (no facts extracted)")
    print("------------------------------\n")
