from fastapi import FastAPI, UploadFile, File
from gateway.gateway_client import chat_completion
import pytesseract
from PIL import Image
import io
from dotenv import load_dotenv
load_dotenv("/opt2/.env")

app = FastAPI()


@app.post("/extract_text_from_image")
async def extract_text_from_image(file: UploadFile = File(...)):
    """
    Распознаёт текст с изображения через Tesseract OCR
    """
    image = Image.open(io.BytesIO(await file.read()))
    raw_text = pytesseract.image_to_string(image, lang="eng")
    return {"raw_text": raw_text}


@app.post("/llm_clean_ocr")
async def llm_clean_ocr(data: dict):
    """
    Отправляет результат OCR в LLM для очистки и нормализации
    """
    raw_text = data["raw_text"]
    response = await chat_completion.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in early printed books. "
                    "You receive OCR'd English text from 17th–18th century sources. "
                    "Your task is to clean up spelling, long s (ſ), and any layout issues, "
                    "but keep the sentence structure and tone of the original. "
                    "Correct only OCR mistakes, don't modernize the language."
                )
            },
            {"role": "user", "content": raw_text}
        ]
    )
    return {"cleaned_text": response.choices[0].message.content}

from fastapi import UploadFile, File
from pdf2image import convert_from_bytes
import os

@app.post("/ocr_pdf_to_text")
async def ocr_pdf_to_text(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    pages = convert_from_bytes(pdf_bytes, dpi=300)

    all_text = []
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page, lang="eng")
        all_text.append(f"[PAGE:{i+1}]\n{text.strip()}\n")

    full_text = "\n".join(all_text)

    # Сохраняем результат в файл
    os.makedirs("processed/ocr_output", exist_ok=True)
    filename = os.path.splitext(file.filename)[0] + ".txt"
    output_path = os.path.join("processed/ocr_output", filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    return {"text": full_text, "saved_to": output_path}


from pydantic import BaseModel
from typing import Optional
import json 

class CleanOCRRequest(BaseModel):
    text: str
    next_preview: Optional[str] = ""

@app.post("/clean_ocr")
async def clean_ocr(request: CleanOCRRequest):
    prompt = (
        "You are an expert in cleaning OCR-scanned texts from historical English books.\n"
        "You receive the raw OCR output of one page.\n"
        "Clean it up, remove page numbers, chapter headers, hyphenation, garbage lines, and normalize spacing.\n"
        "Preserve early modern English spelling and grammar.\n"
        "Use the short preview of the next page to improve ending clarity.\n"
        "If the beginning of the page seems to continue a sentence from the previous one, preserve and connect it properly.\n"
        "Return your response as a JSON object with a single field: 'cleaned_text'."
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Page OCR:\n{request.text}\n\nNext page starts with:\n{request.next_preview}"}
    ]

    response = await chat_completion.create(
        model="gpt-4.1-2025-04-14", # "o4-mini-2025-04-16" 
        #service_tier="flex",
        messages=messages,
        temperature=0,
        #max_tokens=2048
    )

    cleaned_text = response.choices[0].message.content.strip()

    # добавь:
    if cleaned_text.startswith("{"):
        try:
            return json.loads(cleaned_text)
        except:
            pass

    return {"cleaned_text": cleaned_text}



import base64
import json
import textwrap
import mimetypes

from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from gateway.gateway_client import chat_completion


def _file_to_data_uri(upload: UploadFile) -> str:
    data = upload.file.read()
    size_kb = len(data) / 1024
    print(f"[DEBUG] file {upload.filename} → {size_kb:.1f} KB")
    mime = mimetypes.guess_type(upload.filename)[0] or "image/jpeg"
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


class OCRResponse(BaseModel):
    lines: List[Dict[str, str]]
    error: str | None = None


@app.post("/ocr_translate_manuscript", response_model=OCRResponse)
async def ocr_translate_manuscript(
    file: UploadFile = File(...),
    detail: str     = "high",
    model: str      = "gpt-4o",
) -> Any:
    """
    Принимает JPEG/PNG/WebP ≤2048px,
    вкладывает его в data URI, шлёт в GPT-4o-Vision через chat_completion
    и возвращает JSON:
      { lines:[{"span":…,"eng":…},…], error:null }
    Если модель отказывается или вернёт не-JSON, в error будет текст.
    """

    # 1️⃣ Кодируем файл в data URI
    img_uri = _file_to_data_uri(file)

    # 2️⃣ System + prompt
    system_msg = (
        "You are a world-class paleographer and expert in Early Modern Spanish. "
        "Transcribe every line exactly and then translate it into fluent modern English. "
        "Mark illegible parts as [?]. Do NOT refuse."
    )
    user_txt = textwrap.dedent(f"""\
        detail:{detail}

        TASK:
        1) Below is an image of a 16–17th-century Spanish manuscript page.
        2) Transcribe it line-by-line with original spelling.
        3) Under each line, provide a modern English translation.
        4) Use [?] for any illegible fragments.

        RETURN STRICT JSON: {{
          "lines":[{{"span":"<original>","eng":"<translation>"}},…]
        }}
    """).strip()

    # 3️⃣ Собираем messages: текст + image_url
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role":    "user",
            "content": [
                {"type": "text",      "text": user_txt},
                {"type": "image_url", "image_url": {"url": img_uri}},
            ],
        },
    ]

    print("[DEBUG] prompt first line →", user_txt.splitlines()[0])
    print("[DEBUG] sending to chat_completion…")

    # 4️⃣ Вызываем chat_completion
    try:
        resp = await chat_completion.create(
            model=model,
            temperature=0,
            messages=messages,
        )
    except Exception as exc:
        print("[ERROR] Gateway call failed →", exc)
        raise HTTPException(status_code=502, detail=str(exc))

    # 5️⃣ Получаем ответ модели
    raw = resp.choices[0].message.content.strip()
    print("[DEBUG] raw head →", raw[:120].replace("\n", " "), "…")

    # 6️⃣ Пытаемся распарсить JSON
    try:
        parsed = json.loads(raw)
        return {"lines": parsed.get("lines", []), "error": None}
    except json.JSONDecodeError:
        return {"lines": [], "error": raw}







from fastapi import UploadFile, File
from pydantic import BaseModel
from typing import List
import base64, mimetypes, json
import textwrap

from gateway.gateway_client import chat_completion


class OCRReadResponse(BaseModel):
    lines: List[str]
    lang: str


def _file_to_data_uri(upload: UploadFile) -> str:
    data = upload.file.read()
    mime = mimetypes.guess_type(upload.filename)[0] or "image/jpeg"
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


@app.post("/ocr_read_lines", response_model=OCRReadResponse)
async def ocr_read_lines(
    file: UploadFile = File(...),
    model: str = "gpt-4o"
):
    img_uri = _file_to_data_uri(file)

    system_msg = (
        "You are a world-class OCR system. "
        "You receive historical scanned manuscripts in various languages. "
        "Transcribe **only the visible lines of text**, one per line. "
        "Do not interpret or translate, just transcribe what you see. "
        "At the end, guess the primary language using ISO 639-1 codes like 'en', 'es', 'la', etc."
        "Respond in strict JSON format: {\"lines\": [...], \"lang\": \"xx\"}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe all visible lines from the image."},
                {"type": "image_url", "image_url": {"url": img_uri}},
            ],
        },
    ]

    try:
        resp = await chat_completion.create(
            model=model,
            temperature=0,
            messages=messages,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gateway error: {e}")

    raw = resp.choices[0].message.content.strip()
    print("[RAW OUTPUT]", raw)

    try:
        return json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON returned:\n{raw[:500]}\n\nError: {e}")


@app.post("/ocr_blocks_structured_raw")
async def ocr_blocks_structured_raw(
    file: UploadFile = File(...),
    model: str = "gpt-4o"
):
    img_uri = _file_to_data_uri(file)

    system_msg = (
        "You are a world-class OCR expert working with historical manuscripts. "
        "Your task is to transcribe the page image and organize the content into structured blocks. "
        "Return JSON with:\n"
        "- 'main_blocks': list of full-width body text blocks (with their bounding boxes)\n"
        "- 'marginalia': list of side notes or small insertions (with box and position: left, top, bottom, inline, unclear)\n"
        "- 'lang': guessed ISO 639-1 language code like 'es', 'en', 'la'\n"
        "If there are no marginalia, return an empty list.\n"
        "Example:\n"
        "{\n"
        "  \"main_blocks\": [{\"text\": \"...\", \"box\": [x,y,w,h]}],\n"
        "  \"marginalia\": [{\"text\": \"...\", \"box\": [x,y,w,h], \"position\": \"left\"}],\n"
        "  \"lang\": \"es\"\n"
        "}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze the image below and return structured OCR output as JSON."},
                {"type": "image_url", "image_url": {"url": img_uri}},
            ],
        },
    ]

    try:
        resp = await chat_completion.create(
            model=model,
            temperature=0,
            messages=messages,
        )
    except Exception as e:
        return {"error": f"[Gateway error] {str(e)}"}

    return {"raw": resp.choices[0].message.content.strip()}


@app.post("/ocr_main_text")
#@app.post("/ocr_main_text_strict_json")
async def ocr_main_text_strict_json(
    file: UploadFile = File(...),
    model: str = "gpt-4.1-2025-04-14"
):
    img_uri = _file_to_data_uri(file)

    system_msg = (
        "You are a strict OCR engine for historical manuscripts. "
        "Extract the main body text from the image. "
        "Respond ONLY in strict JSON format: {\"text\": \"...\"}. "
        "Do NOT add explanations, introductions, markdown, or formatting. "
        "Return only the transcription inside the 'text' field. No code blocks, no commentary."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the main body text. Respond only as JSON: {\"text\": \"...\"}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_uri,
                        #"detail": "high"
                    }
                },
            ],
        },
    ]

    try:
        resp = await chat_completion.create(
            model=model,
            temperature=0,
            messages=messages,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"[Gateway error] {str(e)}")

    raw = resp.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        if "text" not in parsed:
            raise ValueError("Missing 'text' field in JSON")
        return parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON returned:\n\n{raw[:500]}\n\nError: {e}")


@app.post("/ocr_main_text_o4mini")
async def ocr_main_text_o4mini(
    file: UploadFile = File(...),
    model: str = "o4-mini-2025-04-16"
):
    img_uri = _file_to_data_uri(file)

    system_msg = (
        "You are a strict OCR engine for historical manuscripts. "
        "Extract only the main body text from the image, line by line. "
        "Some historical pages may have bleed-through (text from the reverse side). "
        "If that occurs, ignore mirrored or background text and focus on the front-facing text only. "
        "Even if the page appears empty, degraded, or unclear, always return your best guess."
        "Respond ONLY in strict JSON format: {\"text\": \"...\"}. "
        "Do not add explanations, introductions, summaries, markdown, or code blocks. "
        "Return only the text field with the raw transcription."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the main body text. Respond only as JSON: {\"text\": \"...\"}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_uri,
                        "effort": "high"
                    }
                },
            ],
        },
    ]

    try:
        resp = await chat_completion.create(
            model=model,
            service_tier="flex",
            messages=messages,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"[Gateway error] {str(e)}")

    raw = resp.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        if "text" not in parsed:
            raise ValueError("Missing 'text' field in JSON")
        return parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON returned:\n\n{raw[:500]}\n\nError: {e}")