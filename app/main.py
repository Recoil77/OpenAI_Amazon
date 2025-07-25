# ====== Standard library imports ======
import os
import io
import json
import asyncio
import base64
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from fastapi.concurrency import run_in_threadpool
from openai import OpenAI
from uuid import UUID
# ====== Third-party imports ======
import asyncpg
import torch
import tiktoken
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, List, Dict
from typing import Literal, List, Optional, Dict

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field, ValidationError

# ====== Local application imports ======
from gateway.gateway_client import chat_completion, response_completion, embedding
from dotenv import load_dotenv
load_dotenv()

from app.classes import MetadataRequest, MetadataResponse, VectorSearchV2Request, VectorSearchV2Result, RerankRequest, RerankResult, RerankSemanticV5Request, RerankBlockCandidate, RerankMinimalResult

DATABASE_URL = env = os.getenv("DATABASE_URL")

app = FastAPI()

def _file_to_data_uri(upload: UploadFile) -> str:
    data = upload.file.read()
    size_kb = len(data) / 1024
    print(f"[DEBUG] file {upload.filename} → {size_kb:.1f} KB")
    mime = mimetypes.guess_type(upload.filename)[0] or "image/jpeg"
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


@app.post("/ocr_main_text")
async def ocr_main_text_strict_json(
    file: UploadFile = File(...),
    model: str = "gpt-4.1-2025-04-14" #,    "gpt-4o-2024-11-20"
):
    img_uri = _file_to_data_uri(file)

    system_msg_g = (
        "You are a strict OCR engine for early-20 th-century printed sources. "
        "Return ONLY JSON: {\"text\": \"...\"}. "
        "Rules:\n"
        "1. Transcribe exactly the text that is printed; never invent or repeat lines.\n"
        "2. If a column is padded with dots or dashes (e.g. '. . . . .'), **collapse the run "
        "into a single space** or remove it entirely.\n"
        "3. Never output more than one consecutive dot.\n"
        "4. Stop when you reach the bottom margin of the page.\n"
        "5. Total output must stay under 3 000 characters.\n"
    )

    messages = [
        {"role": "system", "content": system_msg_g},
        {
            "role": "user",
            "content": [
                {"type": "text",
                "text": "Extract the main body text. Respond only as JSON: {\"text\": \"...\"}"},
                {"type": "image_url",
                "image_url": {"url": img_uri, "detail": "high" }} #  "low" 
            ],
        },
    ]


    try:
        resp = await chat_completion.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=1536,
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
    text: str  # prev/next убраны как неиспользуемые

class ExtendedOCRResponse(BaseModel):
    cleaned_text: str
    quality_score: float

SYSTEM_PROMPT = (
    "You are a professional historian and language expert.\n\n"
    "You process OCR fragments from early‑modern printed or handwritten sources (1600–1700s).\n"
    "Your task is to CLEAN the fragment and, only if it is **not already English**, translate it into fluent modern English while preserving historical meaning and tone.\n\n"
    "The fragment may start or end mid‑sentence. Do **NOT** invent missing parts or continue incomplete sentences.\n\n"
    "Steps:\n"
    "1. Fix OCR artefacts: broken hyphenation, page numbers, headers, layout noise.\n"
    "2. Detect source language. If it is English, keep it; otherwise translate to English.\n"
    "3. Output strict JSON **only** in this form:\n"
    "{\n  \"cleaned_text\": \"<cleaned (and if needed translated) text>\",\n  \"quality_score\": <float 0.0–1.0>\n}\n\n"
    "quality_score scale examples: 1.0 (crystal‑clear), 0.8 (minor noise), 0.5 (readable with effort), 0.2 (hard to read), 0.0 (illegible).\n\n"
    "❗️Return ONLY the JSON. No markdown, no commentary."
)

@app.post("/clean_ocr_extended", response_model=ExtendedOCRResponse)
async def clean_ocr_extended(req: ExtendedOCRRequest):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": req.text.strip()},
    ]

    try:
        resp = await chat_completion.create(
            model="gpt-4.1-2025-04-14",
            temperature=0.2,
            messages=messages,
            max_tokens=2048,
            response_format={"type": "json_object"},  # гарантированный JSON
        )
        parsed = resp.choices[0].message.content
        return json.loads(parsed)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Model returned invalid JSON: {e}")

    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))   



PROMPT_TEMPLATE = (
    "Analyze the provided text and identify a comprehensive list of unique, meaningful "
    "references useful for semantic search. Include proper names, locations, specialized terms, "
    "and distinctive contextual markers. "
    "If present, also include terms related to archaeological or historical features or phenomena, "
    "even if these are not explicitly named as such. "
    "Avoid generic or redundant words.\n\n"
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
        reasoning={"effort": effort},
        max_output_tokens=8196,
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



@app.post("/vector_search_v2", response_model=List[VectorSearchV2Result])
async def vector_search_v2(req: VectorSearchV2Request):
    # 1) Получаем embedding
    emb_resp = await embedding.create(input=req.query, model="text-embedding-3-small")
    raw_vector = emb_resp["data"][0]["embedding"]
    vector_str = "[" + ",".join(str(x) for x in raw_vector) + "]"

    # 2) query database
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        # увеличить число проб в ivfflat для поиска
        await conn.execute("SET ivfflat.probes = 128")
        rows = await conn.fetch(
            """
            SELECT
              document_id,
              doc_name,
              year,
              doc_type,
              chunk_index,
              cleaned_text   AS text,
              metadata_original,
              metadata_translated,
              embedding <=> $1 AS score
            FROM public.chunks_metadata_v2
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            vector_str,
            req.k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    finally:
        await conn.close()

    # 3) build response
    results: List[VectorSearchV2Result] = []
    for row in rows:
        # parse metadata fields if they are JSON strings
        orig = row["metadata_original"]
        if isinstance(orig, str):
            try:
                orig = json.loads(orig)
            except json.JSONDecodeError:
                orig = {}
        trans = row["metadata_translated"]
        if isinstance(trans, str):
            try:
                trans = json.loads(trans)
            except json.JSONDecodeError:
                trans = {}

        results.append(VectorSearchV2Result(
            document_id=row["document_id"],
            doc_name=row["doc_name"],
            year=row["year"],
            doc_type=row["doc_type"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            score=row["score"],
            metadata_original=orig,
            metadata_translated=trans,
        ))
    return results



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



def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))

# ------------------------ v1 (старый код — без изменений)
# class BGERerankFunction …  @app.post("/rerank_bge") …

# ------------------------ v2  (M3, 8 k токенов)
class BGERerankFunctionM3:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        num_threads: int | None = None,
    ):
        if num_threads:
            torch.set_num_threads(num_threads)

        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

        # CPU-batching даёт прирост, особенно с 16 threads
        self.executor = ThreadPoolExecutor()

    def _score_batch(self, query: str, docs: list[str]) -> list[float]:
        with torch.inference_mode():
            inputs = self.tokenizer(
                [query] * len(docs),
                docs,
                padding=True,
                truncation=True,            # теперь обрежет только >8192 токенов
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**inputs).logits.squeeze(-1)
            return sigmoid(logits).cpu().tolist()

    async def __call__(
        self, query: str, docs: list[str], batch_size: int = 4
    ) -> list[float]:
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                self.executor,
                self._score_batch,
                query,
                docs[i : i + batch_size],
            )
            for i in range(0, len(docs), batch_size)
        ]
        results = await asyncio.gather(*tasks)
        return [score for batch in results for score in batch]


# инициализируем при старте приложения
bge_reranker_v2 = BGERerankFunctionM3(num_threads=16)


class RerankRequest(BaseModel):
    question: str
    answers: list[str]
    threshold: float = 0.25
    # batch_size: int | None = None   # опционно переопределить

class RerankResult(BaseModel):
    index: int
    score: float
    text: str

@app.post("/rerank_bge_v2")
async def rerank_bge_v2_endpoint(req: RerankRequest):
    try:
        scores  = await bge_reranker_v2(req.question, req.answers, batch_size=4)
        passed  = [
            RerankResult(index=i, score=float(s), text=req.answers[i])
            for i, s in enumerate(scores)
            if s >= req.threshold
        ]
        # ⬇️  приводим к dict точно так же, как в v1
        return {"results": [r.dict() for r in sorted(passed, key=lambda r: r.score, reverse=True)]}
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

    
# Семофор для параллельной факт-экстракции
fact_semaphore = asyncio.Semaphore(8)

    

class KnowledgeRequest(BaseModel):
    question: str

class KnowledgeResponse(BaseModel):
    knowledge_answer: str

@app.post("/general_knowledge", response_model=KnowledgeResponse)
async def general_knowledge(req: KnowledgeRequest):
    # System prompt: instruct model to use internal knowledge and limit length to ~3000 characters
    system_prompt = (
    "You are an LLM specialized in providing general factual answers based solely on your internal knowledge. "
    "Only answer questions that ask for widely known, context-independent facts. "
    "If a question depends on user-provided documents, images, or first-person context, immediately return "
    "{\"knowledge_answer\": \"\"}. "
    "Limit your reply to approximately 3000 characters. If your full answer would exceed that, compress it "
    "to fit while preserving all key points. "
    "If your natural answer is shorter, do not pad or expand it. "
    "Return exactly a JSON object with a single field \"knowledge_answer\" containing only the answer text—"
    "no explanations, formatting, or extra fields."
)

    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": req.question}
    ]

    # Call GPT-4.1 for general knowledge
    response = await chat_completion.create(
        model="gpt-4.1-2025-04-14",
        messages=messages,
        temperature=0,
        max_tokens=2048
    )

    # Extract and return the answer
    answer = response.choices[0].message.content.strip()
    return {"knowledge_answer": answer}

class ChunkCandidate(BaseModel):
    chunk_id: int
    text: str

class ExtractFactsRequest(BaseModel):
    question: str
    chunks: List[ChunkCandidate]

class ChunkFacts(BaseModel):
    chunk_id: int
    facts: List[str]

@app.post("/extract_facts", response_model=List[ChunkFacts])
async def extract_facts(req: ExtractFactsRequest):
    # System prompt for fact extraction
    system_prompt = (
        "You are a fact extraction assistant. "
        "Given a user question and a text fragment, extract only those facts from the fragment that directly answer or support the question. "
        "Return a JSON array of objects, each with fields 'chunk_id' and 'facts', where 'facts' is a list of concise fact strings. "
        "Do not include any additional commentary or formatting."
    )
    # system_prompt = (
    #     "You are a fact extraction assistant. "
    #     "Given a user question and a text fragment, extract every fact that could help answer or clarify the question, whether the connection is direct or indirect. "
    #     "Return a JSON array of objects, each with fields 'chunk_id' and 'facts', where 'facts' is a list of clear, concise fact strings. "
    #     "Do not include any commentary or formatting outside the JSON."
    # )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps({"question": req.question, "chunks": [{"chunk_id": c.chunk_id, "text": c.text} for c in req.chunks]})}
    ]

    try:
        resp = await chat_completion.create(
            model="gpt-4.1-2025-04-14",
            messages=messages,
            temperature=0
        )
        content = resp.choices[0].message.content.strip()
        # Parse the response as JSON
        parsed = json.loads(content)
        # Validate and convert to output model
        results: List[ChunkFacts] = []
        for entry in parsed:
            results.append(ChunkFacts(
                chunk_id=int(entry.get("chunk_id")),
                facts=entry.get("facts", [])
            ))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fact extraction failed: {e}")
    

# Initialize OpenAI SDK client (key from environment)
client = OpenAI()

class WebSearchRequest(BaseModel):
    query: str
    search_context_size: str = "low"  # Options: 'low', 'medium', 'high'

class WebSearchResponse(BaseModel):
    answer: str

@app.post("/web_search", response_model=WebSearchResponse)
async def web_search_endpoint(req: WebSearchRequest):
    """
    Perform a web search using the OpenAI SDK with the web_search_preview tool.
    Model: gpt-4.1 with preview, tools enabled.
    """
    try:
        # Call OpenAI's responses.create in threadpool to avoid blocking
        response = await run_in_threadpool(
            client.responses.create,
            model="gpt-4.1",
            tools=[{"type": "web_search_preview", "search_context_size": req.search_context_size}],
            input=req.query
        )
        # Extract and return answer text
        return {"answer": response.output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Web search failed: {e}")



class Box(BaseModel):
    id: str
    type: Literal["text", "table", "column"]
    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)

class Complexity(BaseModel):
    score: float = Field(ge=0, le=1)
    text_density: float = Field(ge=0, le=1)
    noise_index: float = Field(ge=0, le=1)
    old_print_prob: float = Field(ge=0, le=1)

class PageAssessment(BaseModel):
    status: Literal["ok", "error"]
    message: Optional[str] = None
    rotation: Optional[int] = None  # 0 / 90 / 180 / 270
    has_text: bool
    complexity: Optional[Complexity] = None
    layout: Optional[Dict[str, List[Box]]] = None

# ───────────────────────────── setup ─────────────────────────────


# helper

def _file_to_data_uri(upload: UploadFile) -> str:
    data = upload.file.read()
    mime = mimetypes.guess_type(upload.filename)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"

# system‑prompt
_SYSTEM_PROMPT = (
    "You are an OCR-preflight assistant. Respond with a single valid JSON object and nothing else.\n\n"
    "JSON keys (no extras):\n"
    "  status: always 'ok'\n"
    "  rotation: one of 0, 90, 180, 270\n"
    "  has_text: true or false\n"
    "  complexity: object with keys: score, text_density, noise_index, old_print_prob (include only if has_text is true, otherwise omit or null)\n"
    "  layout: object with key 'boxes': array of box objects (include only if has_text is true, otherwise omit)\n\n"
    "If has_text is true:\n"
    "- Each box must represent a significant and logically distinct block of text suitable for separate OCR processing (such as a paragraph, main column, or long section). Do not create boxes for isolated lines, single captions, or labels unless they are substantial in size or content.\n"
    "- Avoid creating multiple tiny boxes. Do not create a box for every line, word, or small caption. Group text into the fewest meaningful blocks possible.\n"
    "- If the main text is interrupted by images, create one box for each uninterrupted main text region, but ignore tiny fragments or insignificant labels.\n"
    "If has_text is false, do not include 'layout' or 'boxes', and omit or set complexity to null.\n"
    "Never include any fields except those listed above. Never output comments, markdown, or explanations—only the JSON object.\n\n"
    "Example (main text only):\n"
    "{\"status\": \"ok\", \"rotation\": 0, \"has_text\": true, \"complexity\": {\"score\": 0.41, \"text_density\": 0.77, \"noise_index\": 0.09, \"old_print_prob\": 0.97}, \"layout\": {\"boxes\": [{\"id\": \"b1\", \"type\": \"text\", \"x0\": 0.10, \"y0\": 0.05, \"x1\": 0.90, \"y1\": 0.98}]}}\n"
    "Example (no text):\n"
    "{\"status\": \"ok\", \"rotation\": 0, \"has_text\": false}\n"
)

# ─────────────────────────── endpoint ───────────────────────────
@app.post("/page_assess", response_model=PageAssessment)
async def page_assess(
    file: UploadFile = File(...),
    model: str = "gpt-4.1-2025-04-14",
):
    img_uri = _file_to_data_uri(file)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe page strictly as JSON."},
                {"type": "image_url", "image_url": {"url": img_uri, "detail": "low"}},
            ],
        },
    ]

    resp = await chat_completion.create(
        model=model,
        temperature=0,
        max_tokens=512,
        messages=messages,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return PageAssessment(status="error", message="invalid_json", has_text=False)

    data.setdefault("status", "ok")

    try:
        assessment = PageAssessment.model_validate(data)
    except ValidationError as err:
        return PageAssessment(status="error", message=f"validation_error: {err}", has_text=False)

    if not assessment.has_text:
        assessment.complexity = None
        assessment.layout = None

    return assessment

SYSTEM_PROMPT_ =  (
    "Read the text below. "
    "Return a JSON object: {\"score\": N}, where N is a floating-point value from 0 to 1 reflecting your confidence that the text meaningfully references a specific, local, or lesser-known place, locality, or site of real or historical geographic interest—especially those that could potentially be identified, rediscovered, or further researched (such as abandoned settlements, old missions, forts, obscure villages, lost towns, or similarly distinctive objects). "
    "Assign a high score (close to 1) only if the text clearly names or describes such a site, in a way that allows precise localization or further investigation (for example, includes a unique name, detailed location, specific event, or direct association with a lesser-known people, group, or event). "
    "Assign a low score (below 0.3) if the text consists only of general background, ethnographic, cultural, botanical, zoological, economic, or social information, or mentions only famous, well-known, or modern locations, without a specific, localizable object of interest. "
    "If the text primarily describes food, plants, cultural practices, society, nature, travel, or events with no link to a concrete, localizable, or obscure place, assign a low score. "
    "Use the full range between 0 and 1."
)

class ObjectCheckRequest(BaseModel):
    text: str

class ObjectCheckResponse(BaseModel):
    score: float

@app.post("/check_object", response_model=ObjectCheckResponse)
async def check_object(req: ObjectCheckRequest):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_},
        {"role": "user", "content": req.text.strip()},
    ]

    try:
        resp = await chat_completion.create(
            model="gpt-4.1-2025-04-14",
            temperature=0.2,
            messages=messages,
            max_tokens=32,
            response_format={"type": "json_object"},
        )
        parsed = resp.choices[0].message.content
        result = json.loads(parsed)
        if "score" not in result:
            raise ValueError("Key 'has_object' or 'score' missing in model response")
        # Приводим score к float с fallback на 0.0
        try:
            score = float(result["score"])
        except Exception:
            score = 0.0
        return ObjectCheckResponse(
            score=score
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Model returned invalid JSON: {e}")

    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))





SYSTEM_PROMPT_HYPOTHESIS = (
    "You are a historical research assistant. "
    "Given up to three text fragments: CONTEXT_ABOVE, MAIN_CHUNK, and CONTEXT_BELOW, "
    "generate a single, comprehensive hypothesis about any historical settlement, site, or object described in the MAIN_CHUNK, starting directly with the hypothesis itself, without any introductory phrases. "
    "Use CONTEXT_ABOVE and CONTEXT_BELOW only to clarify ambiguous details, names, or locations, but do not generate hypotheses about objects mentioned only in those contexts. "
    "The hypothesis should be as rich and detailed as possible, including all unique names, toponyms, clues, and relationships—even those that seem minor or only indirectly relevant. "
    "Your output should be a JSON object: "
    "{\"hypothesis\": <full hypothesis>, \"confidence\": <confidence_score>} "
    "where 'hypothesis' is a detailed English summary, and 'confidence' is a number between 0 and 1 reflecting your confidence that the hypothesis is both meaningful and complete, based primarily on the MAIN_CHUNK."
)

class HypothesisRequest(BaseModel):
    context_above: str | None = None
    main_chunk: str
    context_below: str | None = None

class HypothesisResponse(BaseModel):
    hypothesis: str
    confidence: float

@app.post("/generate_hypothesis", response_model=HypothesisResponse)
async def generate_hypothesis(req: HypothesisRequest):
    # Формируем вход для LLM
    user_content = (
        f"CONTEXT_ABOVE:\n{req.context_above or ''}\n\n"
        f"MAIN_CHUNK:\n{req.main_chunk}\n\n"
        f"CONTEXT_BELOW:\n{req.context_below or ''}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_HYPOTHESIS},
        {"role": "user", "content": user_content},
    ]

    try:
        resp = await chat_completion.create(
            model="gpt-4.1-2025-04-14",
            temperature=0.2,
            messages=messages,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        parsed = resp.choices[0].message.content
        result = json.loads(parsed)
        # Минимальная валидация
        hypothesis = result.get("hypothesis", "").strip()
        confidence = float(result.get("confidence", 0.0))
        return HypothesisResponse(hypothesis=hypothesis, confidence=confidence)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Model returned invalid JSON: {e}")

    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
