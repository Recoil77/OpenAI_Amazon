# app/reformulate_question.py
import json
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from openai import OpenAI
import asyncpg
import os
import asyncio
from starlette.concurrency import run_in_threadpool
from gateway.gateway_client import chat_completion
from app.classes import ReformulateRequest, ReformulateResponse, ReasoningRequest, ReasoningResponse, Action, Evidence
from app.classes import WebSearchRequest, GeneralKnowledgeRequest, EntitySearchRequest, ChunkSummaryRequest, ChunkSummaryResponse
from app.classes import VectorSearchRequest, VectorSearchV2Request, VectorSearchV2Result, RerankRequest, RerankResult
from app.classes import RerankSemanticV5Request, ExtractFactsRequest, ChunkCandidate, ChunkFacts
from app.func import build_reformulate_prompt, build_reasoning_prompt
from app.main import vector_search_v2, rerank_bge_endpoint, rerank_semantic_v5, extract_facts
from dotenv import load_dotenv
load_dotenv("/opt2/.env")
DATABASE_URL = env = os.getenv("DATABASE_URL")
client = OpenAI()

app = FastAPI()



# --- Endpoint implementation ---
@app.post("/reformulate_question", response_model=ReformulateResponse)
async def reformulate_question(req: ReformulateRequest):
    prompt = build_reformulate_prompt(req)
    # --- Вызов LLM (здесь пример на openai, замени на свой gateway если надо) ---
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
    try:
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)
        return ReformulateResponse(**data)
    except Exception as e:
        return ReformulateResponse(
            reformulated_question=req.active_question,
            alternatives=[],
            reason=f"LLM output parsing error: {e} | Content: {content[:200]}"
        )

@app.post("/llm_reasoning", response_model=ReasoningResponse)
async def llm_reasoning(req: ReasoningRequest):
    prompt = build_reasoning_prompt(req)
    try:
        resp = await chat_completion.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": "You are a deep reasoning engine for historical research. Only output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.2
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)
        return ReasoningResponse(
            actions=[Action(**a) for a in data.get("actions", [])],
            finalize=data.get("finalize", False),
            active_question=data.get("active_question", req.active_question),
            hypothesis=data.get("hypothesis", ""),
            supporting_evidence=[Evidence(**e) for e in data.get("supporting_evidence", [])],
            confidence=float(data["confidence"]) if "confidence" in data else None
        )
    except Exception as e:
        return ReasoningResponse(
            actions=[],
            finalize=False,
            active_question=req.active_question,
            hypothesis="",
            supporting_evidence=[],
            confidence=None
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
    

@app.post("/chunk_summary", response_model=ChunkSummaryResponse)
async def chunk_summary(req: ChunkSummaryRequest):
    """
    Сжимает чанк до короткого смыслового summary (1-3 предложения, ключевые имена, даты, факты).
    """
    prompt = (
        "Summarize the following historical text in 5-7 clear sentences, "
        "focusing on the most important people, events, dates, and places. "
        "Keep the summary under 256 tokens and omit redundant details. "
        "Text:\n" + req.text
    )
    try:
        resp = await chat_completion.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": "You are a historical research assistant. Return only the summary."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=req.max_tokens or 512,
            temperature=0.3
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
