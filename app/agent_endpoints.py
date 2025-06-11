# app/reformulate_question.py
import json
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from openai import OpenAI, AsyncOpenAI
import asyncpg
import os
import asyncio
import traceback
import structlog
from starlette.concurrency import run_in_threadpool
from gateway.gateway_client import chat_completion, response_completion
from fastapi import APIRouter, HTTPException, Query
from pydantic import ValidationError
from app.classes import ReasoningRequest, ReasoningResponse
from app.classes import ReformulateRequest, ReformulateResponse, ReasoningRequest, ReasoningResponse, Action, Evidence
from app.classes import WebSearchRequest, GeneralKnowledgeRequest, EntitySearchRequest, ChunkSummaryRequest, ChunkSummaryResponse
from app.classes import VectorSearchRequest, VectorSearchV2Request, VectorSearchV2Result, RerankRequest, RerankResult
from app.classes import RerankSemanticV5Request, ExtractFactsRequest, ChunkCandidate, ChunkFacts, GetVerdictRequest, GetVerdictResponse
from app.func import build_reformulate_prompt, build_reasoning_prompt_v2
from app.main import vector_search_v2, rerank_bge_endpoint, rerank_semantic_v5, extract_facts
from dotenv import load_dotenv
load_dotenv("/opt2/.env")
DATABASE_URL = env = os.getenv("DATABASE_URL")
client = OpenAI()
openai_client = AsyncOpenAI() 
app = FastAPI()
log = structlog.get_logger(__name__)
CUTTING = 5

@app.post("/reformulate_question", response_model=List[Evidence])
async def reformulate_question(req: ReformulateRequest):
    prompt = build_reformulate_prompt(req)
    try:
        resp = await chat_completion.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": "You are a research assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.6
        )
        import json
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)
        evidence = Evidence(
            source="reformulate",
            value=data.get("reformulated_question", ""),
            details={
                "alternatives": data.get("alternatives", []),
                "reason": data.get("reason", "")
            },
            meta={}
        )
        return [evidence]
    except Exception as e:
        evidence = Evidence(
            source="reformulate",
            value=req.active_question,
            details={
                "alternatives": [],
                "reason": f"LLM output parsing error: {e} | Content: {content[:200]}"
            },
            meta={}
        )
        return [evidence]

@app.post("/llm_reasoning", response_model=ReasoningResponse)
async def llm_reasoning(
    req: ReasoningRequest,
    model: str = Query("o3-2025-04-16", description="LLM model"),  # 'gpt-4.1-2025-04-14'
    temperature: float = Query(0.25, description="Sampling temperature"),
    max_tokens: int = Query(4096, description="Maximum output tokens"),
) -> ReasoningResponse:
    """
    Один цикл глубокого рассуждения: отдаём промпт —
    получаем строго JSON-объект с действиями, гипотезой и т.д.
    """
    prompt = build_reasoning_prompt_v2(req)
    log.info("llm_reasoning.start", iteration=req.iteration, model=model)

    try:
        response = await chat_completion.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"},
            #temperature=temperature,
            #max_tokens=max_tokens,
        )
        raw_json_str = response.choices[0].message.content
        log.debug("llm_reasoning.raw", json=raw_json_str)

        # Валидация и возврат через Pydantic v2
        result = ReasoningResponse.model_validate_json(raw_json_str)
        log.info("llm_reasoning.ok")
        return result

    except ValidationError as e:
        # response.choices[0].message.content может не существовать в случае раннего сбоя
        raw_content = None
        try:
            raw_content = response.choices[0].message.content
        except Exception:
            pass
        log.error("llm_reasoning.validation_error", error=str(e), raw=raw_content)
        raise HTTPException(status_code=422, detail=f"Pydantic validation failed: {e}")

    except Exception as e:
        log.exception("llm_reasoning.fail")
        raise HTTPException(status_code=500, detail=f"LLM reasoning failed: {e}")

# @app.post("/llm_reasoning", response_model=ReasoningResponse)
# async def llm_reasoning(
#     req: ReasoningRequest,
#     model: str = Query("o3-2025-04-16" ,  description="LLM model"), #  'gpt-4.1-2025-04-14'
#     temperature: float = Query(0.25,     description="Sampling temperature"),
#     max_tokens: int = Query(4096,        description="Maximum output tokens"),
# ) -> ReasoningResponse:
#     """
#     Один цикл глубокого рассуждения: отдаём промпт —
#     получаем строго JSON-объект с действиями, гипотезой и т.д.
#     """
#     prompt = build_reasoning_prompt(req)
#     log.info("llm_reasoning.start", iteration=req.iteration, model=model)

#     try:
#         response = await openai_client.chat.completions.create(
#             model=model,
#             messages=[{"role": "system", "content": prompt}],
#             response_format={"type": "json_object"},
#             reasoning_effort="high",
#             #temperature=temperature,
#             #max_tokens=max_tokens,
#         )
#         raw_json_str = response.choices[0].message.content
#         log.debug("llm_reasoning.raw", json=raw_json_str)

#         # Pydantic v2: валидируем сразу из строки, без промежуточного json.loads
#         result = ReasoningResponse.model_validate_json(raw_json_str)
#         log.info("llm_reasoning.ok")
#         return result

#     except ValidationError as e:
#         log.error("llm_reasoning.validation_error", error=str(e), raw=response.choices[0].message.content)
#         raise HTTPException(status_code=422, detail=f"Pydantic validation failed: {e}")

#     except Exception as e:
#         log.exception("llm_reasoning.fail")
#         raise HTTPException(status_code=500, detail=f"LLM reasoning failed: {e}")

# tools = [{
#     "type": "function",
#     "function": {
#         "name": "reasoning_response",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "actions": {           # <-- здесь был set
#                     "type": "array",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "type":    {"type": "string", "enum": [
#                                 "vector_search", "web_search",
#                                 "entity_search", "general_knowledge",
#                                 "reformulate_question"
#                             ]},
#                             "query":   {"type": "string"},
#                             "reason":  {"type": "string"}
#                         },
#                         "required": ["type", "query", "reason"]
#                     },
#                     "minItems": 1
#                 },
#                 "finalize": {"type": "boolean"},
#                 "active_question": {"type": "string"},
#                 "hypothesis": {"type": "string"},
#                 "supporting_evidence": {
#                     "type": "array",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "source":  {"type": "string"},
#                             "value":   {"type": "string"},
#                             "details": {"type": "object"},   # строго объект!
#                             "meta":    {"type": "object"}
#                         },
#                         "required": ["source", "value", "details", "meta"]
#                     }
#                 },
#                 "confidence": {"type": ["number", "null"]}
#             },
#             "required": ["actions", "finalize", "supporting_evidence"]
#         }
#     }
# }]


# @app.post("/llm_reasoning", response_model=ReasoningResponse)
# async def llm_reasoning(
#     req: ReasoningRequest,
#     model: str = Query("o3-2025-04-16", description="LLM model (e.g., 'gpt-4.1-2025-04-14', 'o3')"), # 'gpt-4.1-2025-04-14'   
#     temperature: float = Query(0.25, description="Sampling temperature"),
#     max_tokens: int = Query(4096, description="Maximum output tokens"),
# ):
#     prompt = build_reasoning_prompt(req)
#     try:
#         resp =  client.chat.completions.create( #await
#             model=model,
#             response_format={"type": "json_object"},
#             messages=[
#                 {"role": "system", "content": prompt}
#             ],
#             #temperature=temperature,
#             tools = tools,
#             #top_p = 0.9,
#             #max_tokens=max_tokens,

#         )
#         #content = resp.choices[0].message.content.strip()
#         choice = resp.choices[0]

#         # ➟ правильный путь к строке-JSON с аргументами
#         args_json = choice.message.tool_calls[0].function.arguments
#         data = json.loads(args_json)
#         content = data
#         print("==== RAW LLM OUTPUT ====")
#         print(repr(content))

#         try:
#             #data = json.loads(content)
#             data = content
#             print("==== DATA LOADED FROM JSON ====")
#             print(data)
#             # Временно выводим структуру для дебага
#             print("==== KEYS AT TOP LEVEL ====")
#             print(list(data.keys()))
#             if "supporting_evidence" in data:
#                 print(f"supporting_evidence type: {type(data['supporting_evidence'])}, len: {len(data['supporting_evidence'])}")
#                 for i, ev in enumerate(data["supporting_evidence"]):
#                     print(f"  evidence[{i}] type: {type(ev)}, keys: {ev.keys() if isinstance(ev, dict) else 'not a dict'}")
#             if "actions" in data:
#                 print(f"actions type: {type(data['actions'])}, len: {len(data['actions'])}")
#                 for i, act in enumerate(data["actions"]):
#                     print(f"  action[{i}] type: {type(act)}, keys: {act.keys() if isinstance(act, dict) else 'not a dict'}")
#         except Exception as e:
#             print("==== LLM PARSE ERROR ====")
#             print("RAW OUTPUT:", repr(content))
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"LLM reasoning failed: {e}\nRAW OUTPUT:\n{content[:1000]}"
#             )

#         # Вот тут дебажим падение на ReasoningResponse
#         try:
#             result = ReasoningResponse(**data)
#         except Exception as e:
#             print("==== PYDANTIC PARSE ERROR ====")
#             print("TRACEBACK:")
#             print(traceback.format_exc())
#             print("==== DATA THAT CAUSED ERROR ====")
#             print(data)
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Pydantic ReasoningResponse parse failed: {e}\nTRACEBACK:\n{traceback.format_exc()}\nDATA:\n{str(data)[:1000]}"
#             )

#         return result

#     except Exception as e:
#         print("==== OUTER ERROR IN /llm_reasoning ====")
#         print(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"LLM reasoning failed (outer): {e}")


    
@app.post("/get_verdict", response_model=GetVerdictResponse)
async def get_verdict(req: GetVerdictRequest):
    # 1. general_knowledge
    gk_evidence = await general_knowledge_endpoint(GeneralKnowledgeRequest(query=req.hypothesis))
    gk_answer = gk_evidence[0].value.strip() if gk_evidence and gk_evidence[0].value else ""

    # 2. web_search (high)
    ws_evidence = await web_search_endpoint(WebSearchRequest(query=req.hypothesis, search_context_size="high"))
    ws_answer = ws_evidence[0].value.strip() if ws_evidence and ws_evidence[0].value else ""

    # 3. Verdict logic
    if gk_answer:
        verdict = "trivial"
        details = "The hypothesis is confirmed in general_knowledge."
    elif ws_answer:
        verdict = "publicly_known"
        details = "The hypothesis is found in web_search but not in general_knowledge."
    else:
        verdict = "not_found"
        details = "The hypothesis is not found in general_knowledge or web_search."

    return GetVerdictResponse(
        verdict=verdict,
        details=details,
        general_knowledge_answer=gk_answer,
        web_search_answer=ws_answer
    )

@app.post("/web_search", response_model=List[Evidence])
async def web_search_endpoint(req: WebSearchRequest):
    """
    Perform a web search using OpenAI SDK's web_search_preview tool,
    return answer(s) as a list of Evidence.
    """
    try:
        response = await run_in_threadpool(
            client.responses.create,
            model="gpt-4.1",
            tools=[{"type": "web_search_preview", "search_context_size": req.search_context_size}],
            input=req.query
        )
        # Достаем текст ответа — пример, адаптируй под реальный SDK
        answer_text = getattr(response, "output_text", None)
        if not answer_text:
            raise Exception("No output_text in response")
        # Evidence-формат (source, value, details, meta)
        ev = Evidence(
            source="web",
            value=answer_text,
            details={"search_context_size": req.search_context_size},
            meta={}
        )
        return [ev]  # всегда список!
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Web search failed: {e}")
    

@app.post("/general_knowledge", response_model=List[Evidence])
async def general_knowledge_endpoint(req: GeneralKnowledgeRequest):
    """
    Answer a general knowledge query using LLM (or other knowledge source),
    return result(s) as a list of Evidence.
    """
    try:
        # Вызов LLM через gateway, пример:
        resp = await chat_completion.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": "You are a historical research assistant. Return a concise, factual answer to the user's question."},
                {"role": "user", "content": req.query}
            ],
            max_tokens=256,
            temperature=0.0  # только факт, без фантазии!
        )
        answer_text = resp.choices[0].message.content.strip()
        evidence = Evidence(
            source="general_knowledge",
            value=answer_text,
            details={},
            meta={}
        )
        return [evidence]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"General knowledge search failed: {e}")
    

@app.post("/entity_search", response_model=List[Evidence])
async def entity_search(req: EntitySearchRequest):
    conn = await asyncpg.connect(DATABASE_URL)
    result = []
    try:
        for q in req.entities:
            if req.mode == "exact":
                sql = """
                SELECT
                    ent.entity,
                    COUNT(*) AS count
                FROM (
                    SELECT jsonb_array_elements_text(metadata_original->'entities') AS entity
                    FROM chunks_metadata_v2
                    UNION ALL
                    SELECT jsonb_array_elements_text(metadata_translated->'entities') AS entity
                    FROM chunks_metadata_v2
                ) AS ent
                WHERE LOWER(ent.entity) = LOWER($1)
                GROUP BY ent.entity
                """
                param = q
            else:  # substring (default)
                sql = """
                SELECT
                    ent.entity,
                    COUNT(*) AS count
                FROM (
                    SELECT jsonb_array_elements_text(metadata_original->'entities') AS entity
                    FROM chunks_metadata_v2
                    UNION ALL
                    SELECT jsonb_array_elements_text(metadata_translated->'entities') AS entity
                    FROM chunks_metadata_v2
                ) AS ent
                WHERE ent.entity ILIKE $1
                GROUP BY ent.entity
                """
                param = f"%{q}%"
            rows = await conn.fetch(sql, param)
            # Если найдено много вариантов (подстрочный режим) — ищем только те, что максимально похожи
            # Оставляем только ровно то, что искали, если режим "exact"
            # В любом случае: агрегируем count по entity (может быть несколько строк если подстрочный режим)
            total_count = sum([row["count"] for row in rows]) if rows else 0
            result.append(
                Evidence(
                    source="entity",
                    value=q,
                    details={"count": total_count},
                    meta={"mode": req.mode}
                )
            )
        await conn.close()
        return result
    except Exception as e:
        await conn.close()
        raise HTTPException(status_code=500, detail=f"Entity search failed: {e}")
    

@app.post("/entity_hybrid", response_model=List[Evidence])
async def entity_hybrid_endpoint(req: VectorSearchRequest):
    """
    Entity-based search: вытаскиваем чанки с entity, ранжируем через BGE и LLM.
    req.query — это entity/топоним/имя.
    """
    try:
        print(f"\n[entity_hybrid] Called with entity: {req.query}, k={req.k}")

        # 1. --- Entity SQL: ищем чанки с этой entity ---
        conn = await asyncpg.connect(DATABASE_URL)
        sql = """
            SELECT id, doc_name, doc_type, chunk_index, cleaned_text AS text, year
            FROM chunks_metadata_v2
            WHERE EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(metadata_original->'entities') ent
                WHERE LOWER(ent) = LOWER($1)
            )
            OR EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(metadata_translated->'entities') ent
                WHERE LOWER(ent) = LOWER($1)
            )
            LIMIT $2
        """
        rows = await conn.fetch(sql, req.query, req.k * 15)
        await conn.close()
        print(f"    ↳ {len(rows)} candidate chunks selected by entity")

        if not rows:
            return []

        # 2. --- Mapping for rerankers ---
        answers = [row["text"] for row in rows]
        idx2chunk = {i: row for i, row in enumerate(rows)}

        # 3. --- BGE rerank ---
        print("  → rerank_bge_endpoint ...")
        bge_out = await rerank_bge_endpoint(
            RerankRequest(
                question=req.query,
                answers=answers,
                threshold=getattr(req, "bge_threshold", 0.0),
            )
        )
        print(f"    ↳ BGE rerank: {len(bge_out.get('results', []))} candidates")

        bge_filtered = [
            r for r in bge_out.get("results", []) if r.get("score", 0) >= getattr(req, "bge_threshold", 0.0)
        ]
        print(f"    ↳ BGE filtered: {len(bge_filtered)} remain")
        bge_filtered.sort(key=lambda x: x.get("score", 0), reverse=True)

        # 4. --- LLM (semantic) rerank ---
        top_bge = bge_filtered[: req.k * 2]
        sem_candidates = [
            {"block_id": r["index"], "text": r["text"]} for r in top_bge
        ]
        print("  → rerank_semantic_v5 ...")
        sem_out = await rerank_semantic_v5(
            RerankSemanticV5Request(
                question=req.query,
                candidates=sem_candidates,
                threshold=getattr(req, "semantic_threshold", 0.0),
            )
        )
        print(f"    ↳ semantic rerank: {len(sem_out)} blocks after rerank")
        sem_out = sem_out[:CUTTING] 
        print(f"    ↳ semantic rerank: {len(sem_out)} blocks after cutting")
        # 5. --- Собираем shortlist ---
        results = []
        for entry in sem_out:
            list_idx = entry["block_id"]
            src = idx2chunk.get(list_idx)
            if src is None:
                print(f"    [!] List-idx {list_idx} not found! (should not happen)")
                continue

            bge_score = next(
                (r["score"] for r in bge_filtered if r["index"] == list_idx), None
            )

            results.append(
                {
                    "year": src["year"],
                    "doc_name": src["doc_name"],
                    "doc_type": src["doc_type"],
                    "chunk_index": src["chunk_index"],
                    "text": src["text"],
                    "bge_score": bge_score,
                    "semantic_score": entry["score"],
                    "facts": [],
                }
            )
        print(f"    ↳ shortlist after rerank: {len(results)} chunks")

        # 6. --- Факты (параллельно) ---
        print("  → extract_facts (parallel) ...")

        async def fetch_facts(idx: int, txt: str):
            async with fact_semaphore:
                fr = await extract_facts(
                    ExtractFactsRequest(
                        question=req.query,
                        chunks=[ChunkCandidate(chunk_id=idx, text=txt)],
                    )
                )
                return idx, (fr[0].facts[:3] if fr else [])

        fact_tasks = [fetch_facts(r["chunk_index"], r["text"]) for r in results]
        facts_output = await asyncio.gather(*fact_tasks)

        for idx, facts in facts_output:
            for r in results:
                if r["chunk_index"] == idx:
                    r["facts"] = facts
                    break
        print("    ↳ facts extraction complete")

        # 7. --- Summary (параллельно) ---
        print("  → summarize (parallel) ...")

        async def fetch_summary(idx: int, txt: str):
            if len(txt.split()) <= 256:
                return idx, txt
            async with fact_semaphore:
                sr = await chunk_summary(ChunkSummaryRequest(text=txt))
                return idx, sr.get("summary", txt)

        summary_tasks = [fetch_summary(r["chunk_index"], r["text"]) for r in results]
        summary_output = await asyncio.gather(*summary_tasks)

        for idx, summ in summary_output:
            for r in results:
                if r["chunk_index"] == idx:
                    r["summary"] = summ
                    break
        print("    ↳ summary complete")

        # 8. --- Сборка Evidence ---
        evidences = []
        for r in results:
            evidences.append(
                Evidence(
                    source="entity_hybrid",
                    value=r.get("summary", r["text"]),
                    details={
                        "facts": r["facts"],
                        "year": r["year"],
                        "doc_name": r["doc_name"],
                    },
                    meta={},
                )
            )
        print(f"  → returning {len(evidences)} Evidence items")
        return evidences

    except Exception as e:
        print("  [ERROR]:", str(e))
        raise HTTPException(status_code=500, detail=f"entity_hybrid failed: {e}")


@app.post("/chunk_summary", response_model=ChunkSummaryResponse)
async def chunk_summary(req: ChunkSummaryRequest):
    """
    Сжимает чанк до 3–5 ёмких предложений (<200 токенов). 
    В резюме только факты: названия, даты, события, описания мест. 
    Никаких вводных фраз («В этом документе…»).
    """
    system_msg = (
        "You are a historical research assistant. "
        "Return only the distilled facts — no preambles, no framing."
    )
    user_prompt = (
        "Extract the most relevant archaeological-historical information "
        "from the text below. Write 3–5 crisp sentences (≤200 tokens). "
        "Keep every proper noun (people, settlements, rivers, missions, "
        "tribes) and any clear dates or distances. Omit commentary, "
        "source references, and meta-phrases.\n\n"
        "Text:\n" + req.text
    )

    try:
        resp = await chat_completion.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=req.max_tokens or 400,   # 200-250 токенов → запас 400
            temperature=0.2                    # чуть ниже для меньшей «воды»
        )
        summary = resp.choices[0].message.content.strip()
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunk summary failed: {e}")

    
fact_semaphore = asyncio.Semaphore(8)


@app.post("/vector_search", response_model=List[Evidence])
async def vector_search_endpoint(req: VectorSearchRequest):
    try:
        print(f"\n[vector_search] Called with query: {req.query}, k={req.k}")

        # 1. Векторный поиск
        print("  → vector_search_v2 ...")
        vec_resp = await vector_search_v2(VectorSearchV2Request(query=req.query, k=req.k))
        print(f"    ↳ vec_resp: {len(vec_resp)} chunks found")

        # --------------------------------------------------------------
        #   Сохраняем список текстов и СРАЗУ строим маппинг
        #   list-index  →  объект vec_resp (в котором есть .chunk_index)
        # --------------------------------------------------------------
        answers     = [item.text for item in vec_resp]          # для BGE
        idx2chunk   = {i: ch for i, ch in enumerate(vec_resp)} # новый маппинг
        # --------------------------------------------------------------

        # 2. BGE rerank
        print("  → rerank_bge_endpoint ...")
        bge_out = await rerank_bge_endpoint(
            RerankRequest(
                question=req.query,
                answers=answers,
                threshold=req.bge_threshold,
            )
        )
        print(f"    ↳ BGE rerank: {len(bge_out.get('results', []))} candidates")

        bge_filtered = [
            r for r in bge_out.get("results", []) if r.get("score", 0) >= req.bge_threshold
        ]
        print(f"    ↳ BGE filtered: {len(bge_filtered)} remain")
        bge_filtered.sort(key=lambda x: x.get("score", 0), reverse=True)

        # 3. Semantic rerank
        top_bge = bge_filtered[: req.k]
        # block_id теперь — ЭТО list-index; мы не трогаем его дальше
        sem_candidates = [
            {"block_id": r["index"], "text": r["text"]} for r in top_bge
        ]
        print("  → rerank_semantic_v5 ...")
        sem_out = await rerank_semantic_v5(
            RerankSemanticV5Request(
                question=req.query,
                candidates=sem_candidates,
                threshold=req.semantic_threshold,
            )
        )
        print(f"    ↳ semantic rerank: {len(sem_out)} blocks after rerank")
        sem_out = sem_out[:CUTTING] 
        print(f"    ↳ semantic rerank: {len(sem_out)} blocks after cutting")
        # 4. Собираем shortlist
        results = []
        for entry in sem_out:
            list_idx = entry["block_id"]                  # ← это индекс в answers
            src       = idx2chunk.get(list_idx)           # ← вытаскиваем объект напрямую
            if src is None:
                print(f"    [!] List-idx {list_idx} not found! (should not happen)")
                continue

            bge_score = next(
                (r["score"] for r in bge_filtered if r["index"] == list_idx), None
            )

            results.append(
                {
                    "year": src.year,
                    "doc_name": src.doc_name,
                    "doc_type": src.doc_type,
                    "chunk_index": src.chunk_index,       # ← настоящий chunk_index
                    "text": src.text,
                    "bge_score": bge_score,
                    "semantic_score": entry["score"],
                    "facts": [],
                }
            )
        print(f"    ↳ shortlist after rerank: {len(results)} chunks")

        # --------------------------------------------------------------
        # 5. Факты — экстракция параллельно (как было)
        # --------------------------------------------------------------
        print("  → extract_facts (parallel) ...")

        async def fetch_facts(idx: int, txt: str):
            async with fact_semaphore:
                fr = await extract_facts(
                    ExtractFactsRequest(
                        question=req.query,
                        chunks=[ChunkCandidate(chunk_id=idx, text=txt)],
                    )
                )
                return idx, (fr[0].facts[:3] if fr else [])

        fact_tasks   = [fetch_facts(r["chunk_index"], r["text"]) for r in results]
        facts_output = await asyncio.gather(*fact_tasks)

        for idx, facts in facts_output:
            for r in results:
                if r["chunk_index"] == idx:
                    r["facts"] = facts
                    break
        print("    ↳ facts extraction complete")

        # --------------------------------------------------------------
        # 6. Summary — ПАРАЛЛЕЛЬНО через тот же семафор
        # --------------------------------------------------------------
        print("  → summarize (parallel) ...")

        async def fetch_summary(idx: int, txt: str):
            # короткие тексты оставляем как есть, длинные режем
            if len(txt.split()) <= 256:
                return idx, txt
            async with fact_semaphore:               # ← лимитируем параллелизм
                sr = await chunk_summary(ChunkSummaryRequest(text=txt))
                return idx, sr.get("summary", txt)

        summary_tasks   = [fetch_summary(r["chunk_index"], r["text"]) for r in results]
        summary_output  = await asyncio.gather(*summary_tasks)

        for idx, summ in summary_output:
            for r in results:
                if r["chunk_index"] == idx:
                    r["summary"] = summ
                    break
        print("    ↳ summary complete")

        # --------------------------------------------------------------
        # 7. Собрать Evidence (используем r["summary"] если есть)
        # --------------------------------------------------------------
        evidences = []
        for r in results:
            evidences.append(
                Evidence(
                    source="vector_search",
                    value=r.get("summary", r["text"]),
                    details={
                        "facts": r["facts"],
                        "year": r["year"],
                        "doc_name": r["doc_name"],
                    },
                    meta={},
                )
            )
        print(f"  → returning {len(evidences)} Evidence items")

        return evidences

    except Exception as e:
        print("  [ERROR]:", str(e))
        raise HTTPException(status_code=500, detail=f"vector_search failed: {e}")


