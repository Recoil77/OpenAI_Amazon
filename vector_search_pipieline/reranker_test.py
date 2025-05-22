import requests

BASE = "http://localhost:8000"

# 1) Исходный вопрос
original = "Manaus"

# 2) Refine
r1 = requests.post(f"{BASE}/refine_query", json={"query": original})
r1.raise_for_status()
refined = r1.json()["refined_query"]
print("Refined query:", refined)

# 3) Vector search
K = 5
r2 = requests.post(f"{BASE}/vector_search", json={"query": refined, "k": K})
r2.raise_for_status()
cands = r2.json()["results"]
print(f"\nTop {K} candidates from vector_search:")
for i, c in enumerate(cands, 1):
    print(f"{i}. [{c['year']}, {c['doc_name']}] {c['text'][:100]}...")

# 4) Rerank
docs = [c["text"] for c in cands]
r3 = requests.post(f"{BASE}/rerank_bge", json={"question": refined, "answers": docs, "threshold": 0.25})
r3.raise_for_status()
ranked = r3.json()["results"]

print("\nReranked results:")
for r in ranked:
    print(f"{r['index']} → score={r['score']:.4f} | {r['text'][:100]}...")

