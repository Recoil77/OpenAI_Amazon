from __future__ import annotations
"""Batch runner for `/page_assess` endpoint – verbose.

* Targeted run: specify DOC_IDS; None → all document folders.
* Resizes each image so the shortest side ≤ RESIZE_MAX (keeps quality token‑cheap).
* Saves JSON only when the response status is "ok"; pages with errors are retried
  on the next run because their JSON file is absent.
* Shows progress bar and per‑page log lines; prints Rich summary table at the end.

Dependencies: `pip install aiohttp aiofiles pillow tqdm rich`
"""

import asyncio
import io
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import aiofiles
import aiohttp
from PIL import Image
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# ──────────────────────────── CONFIG ────────────────────────────
ROOT_DIR = Path("docs").resolve()                 # root with <doc_id>/jpeg/
DOC_IDS: Optional[List[str]] = ["75"]              # list of doc IDs to process; None → all
ENDPOINT = "http://192.168.168.10:8000/page_assess"  # FastAPI endpoint URL
CONCURRENCY = 8                                     # parallel workers
RESIZE_MAX = 768                                    # px (short side after resize)
JPEG_QUALITY = 85                                   # Pillow JPEG quality
TIMEOUT_SEC = 130                                   # HTTP timeout per request (sec)
VERBOSE = True                                      # True → log each page
# ─────────────────────────────────────────────────────────────────

console = Console()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ──────────────────────────── HELPERS ────────────────────────────

def iter_images(root: Path, allowed_ids: Optional[List[str]]) -> Iterable[tuple[str, Path]]:
    """Yield (page_id, image_path) for every file under docs/<id>/jpeg/*."""
    for doc_dir in root.iterdir():
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        if allowed_ids is not None and doc_id not in allowed_ids:
            continue
        jpeg_dir = doc_dir / "jpeg"
        if not jpeg_dir.is_dir():
            continue
        for img_path in jpeg_dir.iterdir():
            if img_path.is_file():
                yield img_path.stem, img_path


def prepare_image_bytes(img_path: Path) -> bytes:
    """Resize image down to RESIZE_MAX on shorter side, return JPEG bytes."""
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        short = min(w, h)
        if short > RESIZE_MAX:
            scale = RESIZE_MAX / short
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()


async def post_image(session: aiohttp.ClientSession, page_id: str, img_path: Path) -> dict:
    """Send one page to /page_assess; always return JSON dict."""
    try:
        img_bytes = await asyncio.to_thread(prepare_image_bytes, img_path)
        form = aiohttp.FormData()
        form.add_field("file", img_bytes, filename=img_path.name, content_type="image/jpeg")
        form.add_field("page_id", page_id)
        async with session.post(
            ENDPOINT, data=form, timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC)
        ) as resp:
            resp.raise_for_status()
            return await resp.json()
    except Exception as exc:  # network / timeout etc.
        return {"page_id": page_id, "status": "error", "message": str(exc)}


# ──────────────────────────── WORKER ────────────────────────────
async def worker(queue: asyncio.Queue, pbar: tqdm, stats: dict):
    async with aiohttp.ClientSession() as session:
        while True:
            item = await queue.get()
            if item is None:  # poison pill
                break
            page_id, img_path = item
            doc_dir = img_path.parent.parent  # docs/<doc_id>
            json_dir = doc_dir / "json"
            json_dir.mkdir(exist_ok=True)
            json_path = json_dir / f"{img_path.stem}.json"

            # Skip if JSON already exists (i.e. page successfully processed earlier)
            if json_path.exists():
                stats["skipped"] += 1
                if VERBOSE:
                    logging.info("[SKIP] %s (json exists)", img_path)
                pbar.update(1)
                queue.task_done()
                continue

            result = await post_image(session, page_id, img_path)
            status = result.get("status", "ok")

            if status == "ok":
                # Write JSON only on success – ensures retries for failed pages.
                async with aiofiles.open(json_path, "w", encoding="utf-8") as f_out:
                    await f_out.write(json.dumps(result, ensure_ascii=False, indent=2))
                stats["success"] += 1
                if VERBOSE:
                    logging.info("[OK]   %s", img_path)
            else:
                stats["errors"] += 1
                if VERBOSE:
                    logging.error("[ERR]  %s → %s", img_path, result.get("message"))

            pbar.update(1)
            queue.task_done()


# ──────────────────────────── MAIN ────────────────────────────
async def main():
    images = list(iter_images(ROOT_DIR, DOC_IDS))
    total = len(images)
    if total == 0:
        console.print("[bold red]No images found for given DOC_IDS")
        return

    queue: asyncio.Queue = asyncio.Queue()
    for item in images:
        queue.put_nowait(item)
    for _ in range(CONCURRENCY):
        queue.put_nowait(None)  # poison

    stats = {"success": 0, "skipped": 0, "errors": 0}
    pbar = tqdm(total=total, desc="Pages", unit="page")

    workers = [asyncio.create_task(worker(queue, pbar, stats)) for _ in range(CONCURRENCY)]
    await asyncio.gather(*workers)
    pbar.close()

    table = Table(title="Batch summary")
    table.add_column("Total", justify="right")
    table.add_column("Success", justify="right", style="green")
    table.add_column("Skipped", justify="right", style="yellow")
    table.add_column("Errors", justify="right", style="red")
    table.add_row(str(total), str(stats["success"]), str(stats["skipped"]), str(stats["errors"]))
    console.print(table)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("[bold]Interrupted by user")
