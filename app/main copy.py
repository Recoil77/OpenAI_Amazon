from fastapi import FastAPI, UploadFile, File
from gateway.gateway_client import chat_completion
from gateway.gateway_client import response_completion
import pytesseract
from PIL import Image
import io
from fastapi import FastAPI, Query
from fastapi import FastAPI, Body
import asyncio
import asyncpg
from gateway.gateway_client import embedding
import tiktoken
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv("/opt2/.env")


DATABASE_URL = "postgresql://postgres:Recoil_post_2002%23@db-dev.fullnode.pro/amazon"
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

    system_msg_g = (
        "You are a strict OCR engine for historical manuscripts. "
        "Extract the main body text from the image. "
        "Respond ONLY in strict JSON format: {\"text\": \"...\"}. "
        "Do NOT add explanations, introductions, markdown, or formatting. "
        "Return only the transcription inside the 'text' field. No code blocks, no commentary."
    )

    messages = [
        {"role": "system", "content": system_msg_g},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the main body text. Respond only as JSON: {\"text\": \"...\"}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_uri,
                        "detail": "high"
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
    

class ExtendedOCRRequest(BaseModel):
    prev: str = ""
    text: str
    next: str = ""



@app.post("/clean_ocr_extended")
async def clean_ocr_extended(request: ExtendedOCRRequest):
    """
    Очистка и восстановление исторического OCR с контекстом до и после.
    Возвращает JSON с полями: cleaned_text, quality_score, note.
    """
    # system_msg = (
    #     "You are a professional historian and language expert.\n\n"
    #     "You are processing damaged OCR fragments from early modern printed or handwritten sources (1600–1700s). "
    #     "Your job is to restore and translate the main body text of a given fragment.\n\n"
    #     "You are provided:\n"
    #     "- `prev`: the fragment before,\n"
    #     "- `text`: the current fragment to be cleaned and translated,\n"
    #     "- `next`: the fragment after.\n\n"
    #     "Your task is:\n"
    #     "1. Clean the `text` from OCR noise: remove page numbers, headers, broken hyphenation, illegible lines, layout issues.\n"
    #     "2. Then translate the cleaned text into fluent modern English, preserving historical meaning, terminology, and tone.\n"
    #     "3. Use `prev` and `next` to ensure sentence continuity. If `prev` is empty, assume it's the start of the document. If `next` is empty, assume it's the end.\n\n"
    #     "Return your response as strict JSON in the following structure:\n"
    #     "{\n"
    #     "  \"cleaned_text\": \"<cleaned and translated text in English>\",\n"
    #     "  \"quality_score\": <float between 0.0 and 1.0, indicating your confidence in the result>,\n"
    #     "  \"note\": \"<short comment if something was illegible or uncertain, otherwise leave empty>\"\n"
    #     "}\n\n"
    #     "❗️Do NOT include any commentary, explanation, or markdown outside of this JSON."
    # )
    # system_msg = (
    #     "You are a professional historian and language expert.\n\n"
    #     "You are processing damaged OCR fragments from early modern printed or handwritten sources (1600–1700s). "
    #     "Your job is to restore and translate the main body text of a given fragment.\n\n"
    #     "You are provided three fields:\n"
    #     "- `prev`: the fragment that comes immediately *before* `text`, in the original language (not translated).\n"
    #     "- `text`: the current fragment to be cleaned and translated (your main focus).\n"
    #     "- `next`: the fragment that comes immediately *after* `text`, also in the original language.\n\n"
    #     "Your task is:\n"
    #     "1. Clean the `text` from OCR noise: remove page numbers, headers, broken hyphenation, illegible lines, layout issues.\n"
    #     "2. Translate the cleaned `text` into fluent modern English, preserving historical meaning, terminology, and tone.\n"
    #     "3. Use `prev` and `next` ONLY for understanding — to help resolve ambiguous phrases or broken sentences in `text`.\n"
    #     "4. ❗️Do NOT include any translated content from `prev` or `next` in your output.\n"
    #     "5. Your output must represent only the content of `text`.\n"
    #     "6. Important: Your translation must begin exactly at the start of `text`, even if it starts mid-sentence. "
    #     "Do NOT attempt to reconstruct the beginning from `prev`.\n"
    #     "7. Likewise, your translation must end exactly at the last words of `text`, even if the sentence appears incomplete. "
    #     "Do NOT add continuation from `next`.\n\n"
    #     "Return your response as strict JSON in the following structure:\n"
    #     "{\n"
    #     "  \"cleaned_text\": \"<cleaned and translated text in English>\",\n"
    #     "  \"quality_score\": <float between 0.0 and 1.0, indicating your confidence in the result>,\n"
    #     "  \"note\": \"<short comment if something was illegible or uncertain, otherwise leave empty>\"\n"
    #     "}\n\n"
    #     "❗️Do NOT include any commentary, explanation, or markdown outside of this JSON."
    # )

    system_msg_x = (
        "You are a professional historian and language expert.\n\n"
        "You are processing OCR fragments from early modern printed or handwritten sources (1600–1700s). "
        "Your task is to clean and translate the content of a single fragment.\n\n"
        "The input fragment may begin or end mid-sentence. Do NOT attempt to reconstruct missing parts. "
        "Only work with the text exactly as provided.\n\n"
        "Instructions:\n"
        "1. Clean the OCR text: fix broken hyphenation, remove page numbers, headers, illegible lines, and layout issues.\n"
        "2. Translate the cleaned text into fluent modern English, preserving historical meaning and tone.\n"
        "3. Do not invent or assume missing content. Do not continue or complete any incomplete sentences.\n\n"
        "Return your response as strict JSON in the following format:\n"
        "{\n"
        "  \"cleaned_text\": \"<cleaned and translated text in English>\",\n"
        "  \"quality_score\": <float between 0.0 and 1.0>,\n"
        "}\n\n"
        "❗️Do NOT include any commentary, explanation, or markdown outside of this JSON."
    )
    # user_content = (
    #     f"prev:\n{request.prev.strip()}\n\n"
    #     f"text:\n{request.text.strip()}\n\n"
    #     f"next:\n{request.next.strip()}"
    # )

    # messages = [
    #     {"role": "system", "content": system_msg},
    #     {"role": "user", "content": user_content}
    # ]
    user_content = request.text.strip()

    messages = [
        {"role": "system", "content": system_msg_x},
        {"role": "user", "content": user_content}
    ]
    try:
        resp = await chat_completion.create(
            model="gpt-4.1-2025-04-14", #  "o4-mini-2025-04-16"
            #service_tier="flex",
            #reasoning={"effort": "high"},
            temperature=0.2,
            messages=messages,
        )
        raw_output = resp.choices[0].message.content.strip()

        # Попробуем распарсить как JSON
        parsed = json.loads(raw_output)
        return parsed

    except json.JSONDecodeError:
        return {"error": "Invalid JSON returned by model", "raw_output": raw_output}

    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    
class ReasonedOCRRequest(BaseModel):
    prev: str = ""
    text: str
    next: str = ""


system_msg_e = (
    "You are a professional historian and language expert.\n\n"
    "You are processing damaged OCR fragments from early modern printed or handwritten sources (1600–1700s). "
    "Your job is to restore and translate the main body text of a given fragment.\n\n"
    "You are provided:\n"
    "- `prev`: the fragment before,\n"
    "- `text`: the current fragment to be cleaned and translated,\n"
    "- `next`: the fragment after.\n\n"
    "Your task is:\n"
    "1. Clean the `text` from OCR noise: remove page numbers, headers, broken hyphenation, illegible lines, layout issues.\n"
    "2. Then translate the cleaned text into fluent modern English, preserving historical meaning, terminology, and tone.\n"
    "3. Use `prev` and `next` to ensure sentence continuity. If `prev` is empty, assume it's the start of the document. If `next` is empty, assume it's the end.\n\n"
    "Return your response as strict JSON in the following structure:\n"
    "{\n"
    "  \"cleaned_text\": \"<cleaned and translated text in English>\",\n"
    "  \"quality_score\": <float between 0.0 and 1.0, indicating your confidence in the result>,\n"
    "  \"note\": \"<short comment if something was illegible or uncertain, otherwise leave empty>\"\n"
    "}\n\n"
    "❗️Do NOT include any commentary, explanation, or markdown outside of this JSON."
)




@app.post("/clean_ocr_extended_reasoned")
async def clean_ocr_extended_reasoned(request: ReasonedOCRRequest):
    """
    Очистка и перевод OCR через o4-mini-2025-04-16 с reasoning (effort=high).
    """
    full_prompt = (
        system_msg_e + "\n\n"
        f"prev:\n{request.prev.strip()}\n\n"
        f"text:\n{request.text.strip()}\n\n"
        f"next:\n{request.next.strip()}"
    )

    try:
        response = await response_completion.create(
            model="o4-mini-2025-04-16",
            input=full_prompt,
            service_tier="flex",
            reasoning={"effort": "high"}
        )
        print(response)
        content = response["output"][1]["content"][0]["text"]
        parsed = json.loads(content)
        return parsed

    except json.JSONDecodeError:
        return {"error": "Invalid JSON returned by model", "raw_output": content}

    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


class CatalogChunkRequest(BaseModel):
    prev: str = ""
    text: str
    next: str = ""

system_msg_catalog = (
    "You are an expert metadata extractor and translator for early modern library catalogues.\n\n"
    "You receive three fields:\n"
    "- `prev`: the text before this fragment (for context, may be empty)\n"
    "- `text`: the main catalog fragment to process (may include several entries)\n"
    "- `next`: the following fragment (for context, may be empty)\n\n"
    "The catalog is OCR'd and may be noisy. Each main entry:\n"
    "- Begins with a unique number (like `10407.`, `10408.`, or `*10410.—`), which strictly increases by one (can be split across lines by OCR).\n"
    "- The entry data is ALL text between the start of this number and the next (the next number with +1).\n"
    "- Ignore all 'ff. N', 'fl. N', 'v.', etc. as part of the entry; do not split on these.\n"
    "- If the number or structure is ambiguous, include as much context as possible and set `parse_error` to true.\n"
    "- NEVER merge two main entries with different numbers.\n\n"
    "Your task:\n"
    "1. For EACH entry found, extract the following fields:\n"
    "   - `entry_no` (integer)\n"
    "   - `title` (string, English translation)\n"
    "   - `author` (string, English translation)\n"
    "   - `year` (string, English translation, or null)\n"
    "   - `type` (string, English, e.g., 'map', 'book', etc.)\n"
    "   - `city` (string, English)\n"
    "   - `printer` (string, English, or null)\n"
    "   - `notes` (string, English, all other relevant data)\n"
    "   - `raw_text` (the full raw text for this entry, as a backup)\n"
    "   - `parse_error` (boolean, true if the entry was ambiguous or error-prone, else false)\n"
    "2. If there are multiple entries, output a JSON array. If only one, a single JSON object is fine.\n"
    "3. All text fields should be translated to fluent, modern English.\n"
    "4. STRICTLY output ONLY JSON and nothing else. DO NOT use markdown or commentary."
)


@app.post("/extract_catalog_entries")
async def extract_catalog_entries(request: CatalogChunkRequest):
    """
    Catalog entry extraction and translation via GPT-4.1-mini with high effort (reasoning).
    """
    full_prompt = (
        system_msg_catalog + "\n\n"
        f"prev:\n{request.prev.strip()}\n\n"
        f"text:\n{request.text.strip()}\n\n"
        f"next:\n{request.next.strip()}"
    )

    try:
        response = await response_completion.create(
            model="o4-mini-2025-04-16",  # или твой mini endpoint
            input=full_prompt,
            service_tier="flex",
            reasoning={"effort": "medium"}
        )
        #print(response)
        # Если модель возвращает [ { ... }, { ... } ], это JSON array — норм
        content = response["output"][1]["content"][0]["text"]
        parsed = json.loads(content)
        return parsed

    except json.JSONDecodeError:
        return {"error": "Invalid JSON returned by model", "raw_output": content}

    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    





class MetadataRequest(BaseModel):
    document_id: str
    year: Optional[int] = None
    doc_type: Optional[str] = None
    text: str

class MetadataResponse(BaseModel):
    document_id: str
    year: Optional[int] = None
    doc_type: Optional[str] = None
    entities: List[str]
    text: str

PROMPT_TEMPLATE = (
    "Analyze the provided text and identify a comprehensive list of unique, meaningful "
    "references useful for semantic search. Include proper names, locations, specialized terms, "
    "and distinctive contextual markers. Avoid generic or redundant words.\n\n"
    "Respond only with valid JSON:\n"
    "{{\n  \"entities\": [\"...\", \"...\", \"...\"]\n}}\n"
    "Text:\n---\n{TEXT}\n---"
)

@app.post(
    "/generate_metadata",
    response_model=MetadataResponse,
    summary="Generate semantic entities for a text chunk",
    description="Extracts unique, meaningful references from the input text to drive downstream embedding and search."
)
async def generate_metadata(
    req: MetadataRequest,
    effort: str = Query("medium", description="Reasoning effort: low | medium | high"),
    service_tier: str = Query("flex", description="Service tier for the model call")
):
    # build prompt
    prompt = PROMPT_TEMPLATE.format(TEXT=req.text)

    # call the mini‐model via /v1/responses
    response = await response_completion.create(
        model="o4-mini-2025-04-16",
        input=prompt,
        service_tier=service_tier,
        reasoning={"effort": effort}
    )

    # extract the JSON blob from the model's markdown response
    raw = ""
    try:
        raw = response["output"][1]["content"][0]["text"]
    except Exception:
        pass

    entities: List[str] = []
    if raw:
        # strip markdown fences if present
        cleaned = raw.strip().lstrip("```json").rstrip("```").strip()
        try:
            data = json.loads(cleaned)
            entities = data.get("entities", [])
        except json.JSONDecodeError:
            print(f"[generate_metadata] JSON parse error for document {req.document_id}:\n{raw}")

    return MetadataResponse(
        document_id=req.document_id,
        year=req.year,
        doc_type=req.doc_type,
        entities=entities,
        text=req.text
    )



from pydantic import BaseModel

class VectorSearchRequest(BaseModel):
    query: str
    k: int = 5

@app.post("/vector_search")
async def vector_search(req: VectorSearchRequest):
    # 1. Получаем embedding через OpenAI Gateway
    emb_resp = await embedding.create(input=req.query, model="text-embedding-3-small")
    vector = emb_resp["data"][0]["embedding"]
    vector_str = "[" + ",".join(str(x) for x in vector) + "]"

    # 2. Запрашиваем top-K похожих чанков
    conn = await asyncpg.connect(DATABASE_URL)
    rows = await conn.fetch(
        """
        SELECT
          metadata->>'year'    AS year,
          metadata->>'doc_name' AS doc_name,
          metadata->>'doc_type' AS doc_type,
          chunk_index,
          text
        FROM chunks_metadata
        ORDER BY embedding <=> $1
        LIMIT $2
        """,
        vector_str, req.k
    )
    await conn.close()

    # 3. Собираем ответ
    results = [
        {
            "year": row["year"],
            "doc_name": row["doc_name"],
            "doc_type": row["doc_type"],
            "chunk_index": row["chunk_index"],
            "text": row["text"]
        }
        for row in rows
    ]
    return {"results": results}





class RefineQueryRequest(BaseModel):
    query: str

@app.post("/refine_query")
async def refine_query(req: RefineQueryRequest):
    # Tokenize the user's query
    encoding = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoding.encode(req.query))

    # Choose system prompt based on token count
    if token_count < 12:
        system_prompt = (
            "You are an LLM prompt optimizer for embedding-based search over a multimodal archive. "
            "Generate a single, coherent noun phrase of approximately 15 tokens that preserves key entities and the user's intent. "
            "Avoid lists, bullet points, or comma-separated items—use a concise search phrase."
        )
    elif token_count <= 30:
        system_prompt = (
            "You are an LLM prompt optimizer for embedding-based search over a multimodal archive. "
            "Rephrase the query into one coherent sentence of 12–30 tokens optimized for embeddings. "
            "Maintain natural language flow and preserve meaning without using lists or comma-separated items."
        )
    else:
        system_prompt = (
            "You are an LLM prompt optimizer for embedding-based search over a multimodal archive. "
            "Condense the query into one concise sentence of about 20 tokens optimized for embeddings. "
            "Ensure it reads as natural language and retains all key entities, without list formatting."
        )

    # Prepare messages for LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.query}
    ]

    # Call GPT-4.1 to refine the query
    resp = await chat_completion.create(
        model="gpt-4.1-2025-04-14",
        messages=messages,
        temperature=0
    )

    # Extract and return the refined query
    refined_query = resp.choices[0].message.content.strip()
    return {"refined_query": refined_query}


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

class BGERerankFunction:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        # Dynamic device selection: GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        # Thread pool for parallel batch processing
        self.executor = ThreadPoolExecutor()

    def _score_batch(self, query: str, docs: list[str]) -> list[float]:
        # Tokenize paired inputs and run inference with normalized scores
        with torch.inference_mode():
            inputs = self.tokenizer(
                [query] * len(docs),
                docs,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            logits = self.model(**inputs).logits.squeeze(-1)
            # Normalize to [0,1]
            probs = sigmoid(logits)
            return probs.cpu().tolist()

    async def __call__(self, query: str, docs: list[str], batch_size: int = 8) -> list[float]:
        # Split docs into batches and score in parallel
        tasks = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            tasks.append(asyncio.get_running_loop().run_in_executor(
                self.executor,
                self._score_batch,
                query,
                batch
            ))
        # Gather all batch scores
        results = await asyncio.gather(*tasks)
        # Flatten
        scores = [score for batch_scores in results for score in batch_scores]
        return scores

# Initialize reranker
bge_reranker = BGERerankFunction(model_name="BAAI/bge-reranker-base")

class RerankRequest(BaseModel):
    question: str
    answers: list[str]
    threshold: float = 0.25

class RerankResult(BaseModel):
    index: int
    score: float
    text: str

@app.post("/rerank_bge")
async def rerank_bge_endpoint(req: RerankRequest):
    try:
        # Run reranker to get normalized scores
        scores = await bge_reranker(req.question, req.answers)
        # Filter by threshold and pair results
        filtered = [
            RerankResult(index=i, score=float(score), text=req.answers[i])
            for i, score in enumerate(scores)
            if score >= req.threshold
        ]
        # Sort descending by score
        sorted_results = sorted(filtered, key=lambda r: r.score, reverse=True)
        return {"results": [r.dict() for r in sorted_results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Configuration
MAX_CONCURRENT_RERANK = 8
semaphore = asyncio.Semaphore(MAX_CONCURRENT_RERANK)

def make_system_prompt(threshold: float) -> str:
    return (
        "You are a semantic relevance assistant.\n"
        "Your task is to evaluate how well a candidate text fragment answers or supports the given user question.\n"
        "Return a JSON object with a single field 'score' between 0.0 and 1.0.\n"
        "If the score is below the threshold (" + str(threshold) + "), return exactly {\"score\": 0.0}.\n"
        "Do not include explanations or extra fields."
    )

RERANKER_USER_PROMPT = (
    "Question:\n{question}\n\n"
    "Candidate Text:\n{candidate_text}"
)

# Request/Response models
typing_list = list
class RerankBlockCandidate(BaseModel):
    block_id: int
    text: str

class RerankSemanticV5Request(BaseModel):
    question: str
    candidates: typing_list[RerankBlockCandidate]
    threshold: float = 0.25

class RerankMinimalResult(BaseModel):
    block_id: int
    score: float



@app.post("/rerank_semantic_v5", response_model=typing_list[RerankMinimalResult])
async def rerank_semantic_v5(request: RerankSemanticV5Request):
    system_prompt = make_system_prompt(request.threshold)

    async def score_candidate(candidate: RerankBlockCandidate) -> dict:
        user_prompt = RERANKER_USER_PROMPT.format(
            question=request.question,
            candidate_text=candidate.text
        )
        try:
            async with semaphore:
                response = await chat_completion.create(
                    model="gpt-4.1-2025-04-14",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0
                )
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)
            score = float(parsed.get("score", 0.0))
            if not (0.0 <= score <= 1.0):
                score = 0.0
        except Exception as e:
            print(f"❌ Error scoring block {candidate.block_id}: {e}")
            score = 0.0
        return {"block_id": candidate.block_id, "score": score}

    # Score all candidates with concurrency control
    results = await asyncio.gather(*(score_candidate(c) for c in request.candidates))
    # Filter and sort
    filtered = [r for r in results if r["score"] >= request.threshold]
    sorted_results = sorted(filtered, key=lambda r: r["score"], reverse=True)
    return sorted_results



class PipelineRequest(BaseModel):
    question: str
    k: int = 10
    bge_threshold: float = 0.25
    semantic_threshold: float = 0.25

class ChunkPipelineResult(BaseModel):
    year: str
    doc_name: str
    doc_type: str
    chunk_index: int
    text: str
    bge_score: float
    semantic_score: float

@app.post("/search_pipeline", response_model=List[ChunkPipelineResult])
async def search_pipeline(req: PipelineRequest):
    try:
        # 1) Refine the query
        refine_resp = await refine_query(RefineQueryRequest(query=req.question))
        refined = refine_resp["refined_query"]

        # 2) Vector search
        vec_resp = await vector_search(VectorSearchRequest(query=refined, k=req.k))
        vec_results = vec_resp["results"]

        # 3) BGE rerank (pre-filter)
        bge_req = RerankRequest(question=refined, answers=[d["text"] for d in vec_results], threshold=req.bge_threshold)
        bge_resp = await rerank_bge_endpoint(bge_req)
        bge_results = bge_resp["results"]

        # 4) LLM semantic rerank
        semantic_cands = [{"block_id": r["index"], "text": r["text"]} for r in bge_results]
        sem_req = RerankSemanticV5Request(
            question=refined,
            candidates=semantic_cands,
            threshold=req.semantic_threshold
        )
        sem_resp = await rerank_semantic_v5(sem_req)

        # 5) Assemble final chunks
        final = []
        for item in sem_resp:
            bid = item["block_id"]
            sem_score = item["score"]
            # find the corresponding BGE entry
            bge_entry = next(r for r in bge_results if r["index"] == bid)
            meta = vec_results[bid]
            final.append(ChunkPipelineResult(
                year=meta["year"],
                doc_name=meta["doc_name"],
                doc_type=meta["doc_type"],
                chunk_index=meta["chunk_index"],
                text=meta["text"],
                bge_score=bge_entry["score"],
                semantic_score=sem_score
            ))
        return final

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))