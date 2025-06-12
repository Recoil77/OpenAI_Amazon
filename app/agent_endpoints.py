# app/reformulate_question.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

import asyncpg
import os
import asyncio
import structlog
import tiktoken
from starlette.concurrency import run_in_threadpool
from gateway.gateway_client import chat_completion, response_completion
from fastapi import  HTTPException, Query
from pydantic import ValidationError
from app.classes import ReasoningRequest, ReasoningResponse, Evidence
from app.classes import WebSearchRequest, GeneralKnowledgeRequest, EntitySearchRequest, ChunkSummaryRequest, ChunkSummaryResponse
from app.classes import VectorSearchRequest, VectorSearchV2Request, VectorSearchV2Result, RerankRequest, RerankResult
from app.classes import RerankSemanticV5Request, ExtractFactsRequest, ChunkCandidate
from app.func import build_reasoning_prompt_v2
from app.main import vector_search_v2, rerank_bge_endpoint, rerank_semantic_v5, extract_facts
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = env = os.getenv("DATABASE_URL")

app = FastAPI()

log = structlog.get_logger(__name__)
encoding = tiktoken.get_encoding("cl100k_base")

CUTTING_BGE = 32
CUTTING_LLM = 8

@app.post("/llm_reasoning", response_model=ReasoningResponse)
async def llm_reasoning(
    req: ReasoningRequest,
    model: str = Query('gpt-4.1-2025-04-14', description="LLM model"),  #"o3-2025-04-16" 
    temperature: float = Query(0.25, description="Sampling temperature"),
    max_tokens: int = Query(4096, description="Maximum output tokens"),
) -> ReasoningResponse:
    """
    Один цикл глубокого рассуждения: отдаём промпт —
    получаем строго JSON-объект с действиями, гипотезой и т.д.
    """
    prompt = build_reasoning_prompt_v2(req)
    log.info("llm_reasoning.start", iteration=req.iteration, model=model)

    #num_tokens = len(encoding.encode(prompt))
    #log.info(f"Prompt tokens: {num_tokens}")
    try:
        #log.debug("llm_reasoning.raw", prompt=prompt )
        response = await chat_completion.create(
            model=model,
            messages=prompt,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw_json_str = response.choices[0].message.content
        log.debug("llm_reasoning.raw", json=raw_json_str)

        # Валидация и возврат через Pydantic v2
        result = ReasoningResponse.model_validate_json(raw_json_str)
        #log.info("llm_reasoning.ok")
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


@app.post("/web_search", response_model=List[Evidence])
async def web_search_endpoint(req: WebSearchRequest):
    """
    Perform a web search using OpenAI SDK's web_search_preview tool,
    return answer(s) as a list of Evidence.
    """
    try:
        response = await response_completion.create(
            model="gpt-4.1",
            tools=[{"type": "web_search_preview", "search_context_size": req.search_context_size}],
            input=req.query
        )
        
        answer_text = getattr(response, "output_text", None)
        if not answer_text:
            raise Exception("No output_text in response")
        
        ev = Evidence(
            source="web",
            value=answer_text,
            details={"search_context_size": req.search_context_size},
            meta={}
        )
        return [ev]  
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
                    source="entity_search",
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
        rows = await conn.fetch(sql, req.query, req.k)
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
        top_bge = bge_filtered[:CUTTING_BGE]
        print(f"    ↳ BGE after cuttingd: {len(top_bge)} remain")
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
        sem_out = sem_out[:CUTTING_LLM] 
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
        top_bge = bge_filtered[:CUTTING_BGE] 
        print(f"    ↳ BGE after cuttingd: {len(top_bge)} remain")        
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
        sem_out = sem_out[:CUTTING_LLM] 
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


@app.post("/vector_search_v2", response_model=List[Evidence])
async def vector_search_v2_endpoint(req: VectorSearchRequest):
    try:
        print(f"\n[vector_search_v2] Called with query: {req.query}, k={req.k}")

        # 1. Векторный поиск
        print("  → vector_search_v2 ...")
        vec_resp = await vector_search_v2(VectorSearchV2Request(query=req.query, k=req.k))
        print(f"    ↳ {len(vec_resp)} chunks found")

        # --------------------------------------------------------------
        #   Готовим ответы для BGE + маппинг list-index → vec_resp obj
        # --------------------------------------------------------------
        answers     = [item.text for item in vec_resp]
        idx2chunk   = {i: ch for i, ch in enumerate(vec_resp)}

        # 2. BGE rerank
        print("  → rerank_bge_endpoint ...")
        bge_out = await rerank_bge_endpoint(
            RerankRequest(
                question=req.query,
                answers=answers,
                threshold=req.bge_threshold,
            )
        )
        bge_filtered = [
            r for r in bge_out.get("results", []) if r.get("score", 0) >= req.bge_threshold
        ]
        bge_filtered.sort(key=lambda x: x.get("score", 0), reverse=True)

        # 3. Semantic rerank
        top_bge = bge_filtered[:CUTTING_BGE]
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
        sem_out = sem_out[:CUTTING_LLM]

        # 4. Собираем shortlist
        results = []
        for entry in sem_out:
            list_idx  = entry["block_id"]
            src       = idx2chunk.get(list_idx)
            if src is None:
                continue

            bge_score = next(
                (r["score"] for r in bge_filtered if r["index"] == list_idx), None
            )

            results.append(
                {
                    "year": src.year,
                    "doc_name": src.doc_name,
                    "chunk_index": src.chunk_index,
                    "text": src.text,
                    "bge_score": bge_score,
                    "semantic_score": entry["score"],
                }
            )
        print(f"    ↳ shortlist after rerank: {len(results)} chunks")

        # --------------------------------------------------------------
        # 5. Формируем Evidence (без facts и summary)
        # --------------------------------------------------------------
        evidences = [
            Evidence(
                source="vector_search_v2",
                value=r["text"],                      # полный текст чанка
                details={
                    "doc_name": r["doc_name"],
                    "year":     r["year"],
                },
                meta={},
            )
            for r in results
        ]
        print(f"  → returning {len(evidences)} Evidence items")

        return evidences

    except Exception as e:
        print("  [ERROR]:", str(e))
        raise HTTPException(status_code=500, detail=f"vector_search_v2 failed: {e}")