import uuid
import json
from pathlib import Path

number = "91"
DOC_PATH = Path(f"docs/{number}/meta.json")

meta = {
    "document_id": str(uuid.uuid4()),
    "doc_name": "Relación del nuevo descubrimiento del famoso Río Grande que descubrió Francisco de Orellana",
    "year": 1544,
    "doc_type": "book"  # or "manuscript"
}

DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(DOC_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"✅ meta.json created: {DOC_PATH.resolve()}")
