import uuid
import json
from pathlib import Path

# === НАСТРОЙКИ ДЛЯ РУЧНОЙ РЕДАКТУРЫ ===
number = "73"
DOC_PATH = Path(f"docs/{number}/meta.json")

meta = {
    "document_id": str(uuid.uuid4()),
    "doc_name": "Voyages dans l'Amérique du Sud",
    "year": 1883,
    "doc_type": "book" # "manuscript"
    # Добавь сюда ещё поля, если нужно
}

DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(DOC_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"✅ meta.json создан: {DOC_PATH.resolve()}")
