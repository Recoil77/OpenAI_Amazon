import uuid
import json
from pathlib import Path

number = "76"
DOC_PATH = Path(f"docs/{number}/meta.json")

meta = {
    "document_id": str(uuid.uuid4()),
    "doc_name": "Relación del descubrimiento del río de las Amazonas, hoy San Francisco de Quito",
    "year": 1693,
    "doc_type": "manuscript"  # or"book" 
}

DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(DOC_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"✅ meta.json created: {DOC_PATH.resolve()}")
