from __future__ import annotations
"""Quick test script: send one image to /page_assess and show raw + parsed outputs.

Usage (no CLI flags – edit constants below and run):
    python test_single_page_assess.py

Requires `requests` (pip install requests).
"""

import json
from pathlib import Path
import sys

import requests

# ───────────────── CONFIG ─────────────────
ENDPOINT = "http://192.168.168.10:8000/page_assess"  # FastAPI endpoint
IMAGE_PATH = Path("docs/86/jpeg/71-802.jpg")          # image to test
PAGE_ID = IMAGE_PATH.stem                              # "69-025"


# ───────────────── MAIN ─────────────────

def main() -> None:
    if not IMAGE_PATH.exists():
        sys.exit(f"Image not found: {IMAGE_PATH}")

    files = {
        "file": (IMAGE_PATH.name, IMAGE_PATH.read_bytes(), "image/jpeg"),
    }
    data = {
        "page_id": PAGE_ID,
    }

    print(f"POST {ENDPOINT}  ←  {IMAGE_PATH}")
    try:
        resp = requests.post(ENDPOINT, files=files, data=data, timeout=180)
    except requests.RequestException as exc:
        sys.exit(f"[HTTP error] {exc}")

    print(f"Status code: {resp.status_code}\n")

    # Raw text first (may help with malformed JSON)
    print("Raw response:\n" + "-" * 40)
    print(resp.text)
    print("-" * 40)

    # Try to parse JSON
    try:
        parsed = resp.json()
        print("Parsed JSON (pretty):")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except ValueError:
        print("⚠️  Response is not valid JSON – see raw output above.")


if __name__ == "__main__":
    main()
