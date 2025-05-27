import uuid
import json
from pathlib import Path

# === НАСТРОЙКИ ДЛЯ РУЧНОЙ РЕДАКТУРЫ ===
number = "20"
DOC_PATH = Path(f"docs/{number}/meta.json")

meta = {
    "document_id": str(uuid.uuid4()),
    "doc_name": "Travels on the Amazon, Wallace, Alfred Russel",
    "year": 1858,
    "doc_type": "book"
    # Добавь сюда ещё поля, если нужно
}

DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(DOC_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"✅ meta.json создан: {DOC_PATH.resolve()}")
