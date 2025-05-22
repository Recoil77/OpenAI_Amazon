import requests

BASE = "http://localhost:8000"

# 1) Исходный вопрос
original = "entrance to the Paraná-mirím do Mucambo"

# 2) Refine
r1 = requests.post(f"{BASE}/refine_query", json={"query": original})
r1.raise_for_status()
refined = r1.json()["refined_query"]
print("Refined query:", refined)

# 3) Vector search
K = 32
r2 = requests.post(f"{BASE}/vector_search", json={"query": refined, "k": K})
r2.raise_for_status()
cands = r2.json()["results"]
docs = [c["text"] for c in cands]

# 4) BGE rerank (we don’t print these now, just get the ordering)
r3 = requests.post(f"{BASE}/rerank_bge", json={"question": refined, "answers": docs, "threshold": 0.25})
r3.raise_for_status()
bge_ranked = r3.json().get("results", [])

# 5) Prepare payload for LLM semantic reranker
llm_candidates = [
    {"block_id": item["index"], "text": docs[item["index"]]}
    for item in bge_ranked
]
llm_payload = {
    "question": refined,
    "candidates": llm_candidates,
    "threshold": 0.25
}

# 6) LLM semantic rerank
r4 = requests.post(f"{BASE}/rerank_semantic_v5", json=llm_payload)
r4.raise_for_status()
llm_ranked = r4.json()

# 7) Print full text of each chunk passed LLM rerank
print("\nChunks after LLM semantic rerank:")
for item in llm_ranked:
    bid = item["block_id"]
    score = item["score"]
    text = docs[bid]
    print(f"\n--- Block {bid} (score={score:.3f}) ---")
    print(text)
    print("----------------------------")
