#!/usr/bin/env python3
"""
Ска-чать все master-TIFF страницы рукописи mss31013-14100
из локального IIIF-manifest.json.
"""

import json, re, pathlib, sys, requests, tqdm

MANIFEST = pathlib.Path("Download_doc/working/1.json")
if not MANIFEST.exists():
    sys.exit(f"Файл не найден: {MANIFEST}")

data     = json.loads(MANIFEST.read_text("utf-8"))
canvases = data["sequences"][0]["canvases"]
print("Страниц:", len(canvases))

# ── 1. item_id и master-prefix ──────────────────────────────────────────────
sample   = canvases[0]["@id"]                    # быстрее, чем лезть в resource
item_id  = re.search(r"mss\d{5}-\d{5}", sample).group(0)   # mss31013-14100
coll     = item_id.split("-")[0]                             # mss31013
MASTER   = (f"https://tile.loc.gov/storage-services/master/"
            f"mss/{coll}/{item_id}")
print("Master-prefix:", MASTER)

# ── 2. куда сохранять ───────────────────────────────────────────────────────
OUT = pathlib.Path("Download_doc/working/tiff"); OUT.mkdir(exist_ok=True)
UA  = {"User-Agent": "Mozilla/5.0 (script)"}

# ── 3. перебор страниц ──────────────────────────────────────────────────────
for cv in tqdm.tqdm(canvases, desc="downloading"):
    # id canvas или resource — без разницы, у обоих один хвост
    url  = cv["@id"]
    m    = re.search(r":(\d{4})0000\b", url)
    if not m:
        print("! пропуск (не нашёл номер):", url); continue
    page = m.group(1)              # '0005'
    name = f"{page}0000.tif"
    tif  = f"{MASTER}/{name}"

    try:
        r = requests.get(tif, headers=UA, timeout=60)
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"! {name}: {e.response.status_code}"); continue

    (OUT / name).write_bytes(r.content)

print("✔ Все TIFF сохранены →", OUT.resolve())
