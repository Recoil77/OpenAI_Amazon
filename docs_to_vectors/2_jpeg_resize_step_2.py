from pathlib import Path
from PIL import Image

number = "91"

INPUT_DIR = Path(f"docs/{number}/jpeg_")
OUTPUT_DIR = Path(f"docs/{number}/jpeg")
OUTPUT_DIR.mkdir(exist_ok=True)

QUALITY = 95
MAX_SIDE = 2048
MIN_SHORT = 768

def resize_image(im):
    w, h = im.size
    scale = min(MAX_SIDE / max(w, h), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    im = im.resize((new_w, new_h), Image.LANCZOS)
    short = min(im.size)
    if short < MIN_SHORT:
        scale = MIN_SHORT / short
        new_w, new_h = int(im.size[0] * scale), int(im.size[1] * scale)
        im = im.resize((new_w, new_h), Image.LANCZOS)
    return im

for path in INPUT_DIR.iterdir():
    if path.suffix.lower() in [".jpg", ".jpeg"]:
        out_path = OUTPUT_DIR / path.name
        try:
            im = Image.open(path)
            im = im.convert("RGB")
            im = resize_image(im)
            im.save(out_path, "JPEG", quality=QUALITY, optimize=True)
            print(f"✅ {path.name} -> {out_path.name}  {im.size}")
        except Exception as e:
            print(f"❌ {path.name}: {e}")