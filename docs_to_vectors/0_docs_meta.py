import uuid
import json
from pathlib import Path

# === НАСТРОЙКИ ДЛЯ РУЧНОЙ РЕДАКТУРЫ ===
number = "57"
DOC_PATH = Path(f"docs/{number}/meta.json")

meta = {
    "document_id": str(uuid.uuid4()),
    "doc_name": "Das kaiserreich Brasilien auf der Wiener weltausstellung von 1873",
    "year": 1873,
    "doc_type": "book" # "manuscript"
    # Добавь сюда ещё поля, если нужно
}

DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(DOC_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"✅ meta.json создан: {DOC_PATH.resolve()}")
