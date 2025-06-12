import uuid
import json
from pathlib import Path

number = "73"
DOC_PATH = Path(f"docs/{number}/meta.json")

meta = {
    "document_id": str(uuid.uuid4()),
    "doc_name": "Voyages dans l'Amérique du Sud",
    "year": 1883,
    "doc_type": "book"  # or "manuscript"
}

DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(DOC_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"✅ meta.json created: {DOC_PATH.resolve()}")
