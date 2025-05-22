import requests
import json

BASE_URL = "http://localhost:8000"

# 1) Define the question
question = "What factors led to the decline of Manaus during the seventeenth century?"

# 2) Prepare some candidate text fragments
candidates = [
    {"block_id": 1, "text": "In the early 1600s, Manaus suffered from economic downturn as rubber prices collapsed."},
    {"block_id": 2, "text": "The Amazon rainforest is known for its biodiversity and indigenous cultures."},
    {"block_id": 3, "text": "Historical records mention conflict, disease outbreaks, and trade disruptions around Manaus in the 17th century."},
    {"block_id": 4, "text": "Rubber boom in the 19th century revitalized Manaus with new infrastructure."}
]

# 3) Build the request payload
payload = {
    "question": question,
    "candidates": candidates,
    "threshold": 0.25
}

# 4) Call the rerank_semantic_v5 endpoint
response = requests.post(f"{BASE_URL}/rerank_semantic_v5", json=payload)
response.raise_for_status()

# 5) Parse and display the reranked results
results = response.json()
print("Reranked candidates (block_id → score):\n")
for item in results:
    print(f"Block {item['block_id']}  →  score = {item['score']:.3f}")
    # Optionally print the text:
    # matching_text = next(c['text'] for c in candidates if c['block_id'] == item['block_id'])
    # print(f"   Text: {matching_text}\n")
