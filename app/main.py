# ====== Standard library imports ======
import os
import io
import json
import asyncio
import base64
import mimetypes
from concurrent.futures import ThreadPoolExecutor

# ====== Third-party imports ======
import asyncpg
import torch
import tiktoken
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List

# ====== Local application imports ======
from gateway.gateway_client import chat_completion, response_completion, embedding
from dotenv import load_dotenv
load_dotenv("/opt2/.env")

DATABASE_URL = env = os.getenv("DATABASE_URL")

app = FastAPI()

def _file_to_data_uri(upload: UploadFile) -> str:
    data = upload.file.read()
    size_kb = len(data) / 1024
    print(f"[DEBUG] file {upload.filename} → {size_kb:.1f} KB")
    mime = mimetypes.guess_type(upload.filename)[0] or "image/jpeg"
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


def _file_to_data_uri(upload: UploadFile) -> str:
    data = upload.file.read()
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
    user_content = request.text.strip()
    messages = [
        {"role": "system", "content": system_msg_x},
        {"role": "user", "content": user_content}
    ]
    try:
        resp = await chat_completion.create(
            model="gpt-4.1-2025-04-14", #  "o4-mini-2025-04-16"    "gpt-4o-2024-11-20"
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



# Semaphore to limit concurrent fact extraction calls
data_fetch_semaphore = asyncio.Semaphore(8)

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
    facts: List[str]

@app.post("/search_pipeline", response_model=List[ChunkPipelineResult])
async def search_pipeline(req: PipelineRequest):
    try:
        # 1) Refine the query
        refine_resp = await refine_query(RefineQueryRequest(query=req.question))
        refined = refine_resp["refined_query"]

        # 1.1) Internal knowledge source
        kn_resp = await general_knowledge(KnowledgeRequest(question=req.question))
        knowledge_text = kn_resp["knowledge_answer"]
        knowledge_meta = {
            "year": "internal",
            "doc_name": "general_knowledge",
            "doc_type": "internal",
            "chunk_index": -1,
            "text": knowledge_text
        }

        # 2) Vector search
        if req.k >128:
            k_ = 128
        else:
            k_ = req.k
        vec_resp = await vector_search(VectorSearchRequest(query=refined, k=k_))
        vec_results = vec_resp["results"]

        # Combine internal knowledge and document chunks
        docs_meta = [knowledge_meta] + vec_results

        # 3) BGE rerank (pre-filter)
        bge_req = RerankRequest(
            question=refined,
            answers=[d["text"] for d in docs_meta],
            threshold=req.bge_threshold
        )
        bge_resp = await rerank_bge_endpoint(bge_req)
        bge_results = bge_resp["results"]

        # 4) LLM semantic rerank
        semantic_cands = [
            {"block_id": r["index"], "text": r["text"]}
            for r in bge_results
        ]
        sem_req = RerankSemanticV5Request(
            question=refined,
            candidates=semantic_cands,
            threshold=req.semantic_threshold
        )
        sem_resp = await rerank_semantic_v5(sem_req)

        # 5) Assemble final chunks with scores
        final = []
        for item in sem_resp:
            bid = item["block_id"]
            sem_score = item["score"]
            bge_entry = next(r for r in bge_results if r["index"] == bid)
            meta = docs_meta[bid]
            final.append({
                "year": meta["year"],
                "doc_name": meta["doc_name"],
                "doc_type": meta["doc_type"],
                "chunk_index": meta["chunk_index"],
                "text": meta["text"],
                "bge_score": bge_entry["score"],
                "semantic_score": sem_score
            })

        # 6) Extract facts per chunk in parallel
        async def fetch_facts(idx, chunk_text):
            async with data_fetch_semaphore:
                fact_req = ExtractFactsRequest(
                    question=req.question,
                    chunks=[ChunkCandidate(chunk_id=idx, text=chunk_text)]
                )
                fr = await extract_facts(fact_req)
                return (idx, fr[0].facts if fr else [])

        # Launch concurrent fact extraction tasks
        tasks = [fetch_facts(idx, chunk["text"]) for idx, chunk in enumerate(final)]
        facts_results = await asyncio.gather(*tasks)

        # 7) Merge facts into final
        enriched = []
        for idx, chunk in enumerate(final):
            # find facts for this chunk
            facts = next((facts for i, facts in facts_results if i == idx), [])
            chunk["facts"] = facts
            enriched.append(chunk)

        # Return enriched results as Pydantic models
        return [ChunkPipelineResult(**c) for c in enriched]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    

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