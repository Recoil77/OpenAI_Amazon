import requests
import json

BASE = "http://localhost:8000"

# 1) Define your question
question = "What historical and cultural landmarks in Manaus are described in these text fragments?"

# 2) Prepare some example chunks (use the actual texts from your pipeline)
chunks = [
    {
        "chunk_id": 1,
        "text": (
            "Floods of the Amazon extend for a considerable distance. Though the water is crystal clear, "
            "it appears quite dark brown in large volumes; this color, common to many rivers in these regions, "
            "is caused by decomposed plants... Soon, the first houses of Manáos came into view... "
            "The ruins of the small Portuguese fort São José da Barra do Rio Negro are seen on the left. "
            "But they attract much less interest than an old Indian cemetery... Hundreds of large red clay urns..."
        )
    },
    {
        "chunk_id": 2,
        "text": (
            "The position of the city, however, at the junction of the Rio Negro, the Amazon, and the Solimões, "
            "is commanding; and, insignificant as it looks at present, Manaos will no doubt be a great center "
            "of commerce and navigation at some future time... Directly on the riverbank, and overlooking the waters, "
            "once stood a fort known as San Jose..."
        )
    }
]

# 3) Build the request payload
payload = {
    "question": question,
    "chunks": chunks
}

# 4) Call the extract_facts endpoint
resp = requests.post(f"{BASE}/extract_facts", json=payload)
resp.raise_for_status()

# 5) Parse and display the extracted facts
extracted = resp.json()
print("Extracted facts:\n")
for entry in extracted:
    print(f"Chunk {entry['chunk_id']}:")
    for fact in entry["facts"]:
        print(f"  - {fact}")
    print()
