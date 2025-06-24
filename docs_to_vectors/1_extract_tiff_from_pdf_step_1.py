import os
import subprocess
from pathlib import Path

# === SETTINGS ===
number = "91"                  # ← document number
first_page = 9               # ← start page (inclusive)
last_page = 39              # ← end page (inclusive), None = until the end
use_pdftoppm = True            # ← True = use pdftoppm (otherwise pdfimages)
dpi = 400                      # ← resolution
output_format = "jpeg"         # ← 'tiff' or 'jpeg'
jpeg_quality = "95"            # ← applies only if output_format = 'jpeg'

BASE = Path(f"docs/{number}")
PDF = BASE / f"{number}.pdf"
OUT = BASE / "jpeg_"

# === CHECKS ===
if not PDF.exists():
    print(f"❌ PDF not found: {PDF}")
    exit(1)

OUT.mkdir(parents=True, exist_ok=True)

# === COMMAND ASSEMBLY ===
if use_pdftoppm:
    out_prefix = OUT / number
    cmd = ["pdftoppm", f"-{output_format}", "-r", str(dpi)]
    if output_format == "jpeg":
        cmd += ["-jpegopt", f"quality={jpeg_quality}"]
    if first_page:
        cmd += ["-f", str(first_page)]
    if last_page:
        cmd += ["-l", str(last_page)]
    cmd += [str(PDF), str(out_prefix)]
else:
    # pdfimages always outputs tiff, regardless of output_format
    cmd = ["pdfimages", "-tiff", "-p"]
    if first_page:
        cmd += ["-f", str(first_page)]
    if last_page:
        cmd += ["-l", str(last_page)]
    cmd += [str(PDF), str(OUT / number)]

# === RUN ===
print(f"📥 Extracting pages {first_page}–{last_page or 'EOF'} from {PDF.name}")
print("Command:", " ".join(str(c) for c in cmd))
subprocess.run(cmd, check=True)
print(f"✔️ Images saved to: {OUT.resolve()}")

# === REMOVE EXTRA TIFF FILES (pdfimages: only *-000.tif are needed) ===
if not use_pdftoppm and output_format == "tiff":
    for file in OUT.glob(f"{number}-*.tif"):
        if not file.stem.endswith("-000"):
            print(f"🗑 Deleting non-primary TIFF: {file.name}")
            file.unlink()

print("🎯 Done!")
