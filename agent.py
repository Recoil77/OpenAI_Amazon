import os
import asyncio
from enum import Enum
from collections import OrderedDict
import httpx
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------------------------
SERVER_HOST = os.getenv("SERVER_ADDRESS") or "127.0.0.1"
SERVER = f"http://{SERVER_HOST}:8100"
CTX_LIMIT = 128           # max evidence items to carry
PARALLEL_LIMIT = 6        # simultaneous endpoint requests
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", 16))
KEEPALIVE = 120            # max keep‑alive connections

# ---------------------------------------------------------------------------
#  ENUMS & ENDPOINTS
# ---------------------------------------------------------------------------
class Act(str, Enum):
    VECTOR   = "vector_search"
    HYBRID   = "entity_hybrid"
    ENTITY   = "entity_search"
    GKNOW    = "general_knowledge"
    WEB      = "web_search"
    REFORM   = "reformulate_question"

ENDPOINTS = {
    Act.VECTOR: (f"{SERVER}/vector_search",  {"k": 128, "bge_threshold": 0.2, "semantic_threshold": 0.5}),
    Act.HYBRID: (f"{SERVER}/entity_hybrid",  {"k": 128, "bge_threshold": 0.2, "semantic_threshold": 0.5}),
    Act.ENTITY: (f"{SERVER}/entity_search",  {"mode": "substring"}),
    Act.GKNOW:  (f"{SERVER}/general_knowledge", {}),
    Act.WEB:    (f"{SERVER}/web_search", {"search_context_size": "medium"}),
    Act.REFORM: (f"{SERVER}/reformulate_question", {}),
    "LLM":      f"{SERVER}/llm_reasoning",
    "VERDICT":  f"{SERVER}/get_verdict",
}

# ---------------------------------------------------------------------------
#  UTILS
# ---------------------------------------------------------------------------
ALLOWED_KEYS = {"source", "value", "details", "meta"}

def sanitize_evidence(ev: dict) -> dict:
    """Keep only allowed keys and guarantee dict types."""
    ev = {k: v for k, v in ev.items() if k in ALLOWED_KEYS}
    ev.setdefault("details", {})
    ev.setdefault("meta", {})
    if not isinstance(ev["details"], dict):
        ev["details"] = {"raw": str(ev["details"])}
    if not isinstance(ev["meta"], dict):
        ev["meta"] = {"raw": str(ev["meta"])}
    return ev


def dedup_log(log: list[dict], limit: int = 8) -> list[dict]:
    out = OrderedDict()
    for item in reversed(log):
        key = (item["action"], item["query"])
        if key not in out:
            out[key] = item.copy()
        else:
            out[key]["result_count"] += item["result_count"]
        if len(out) >= limit:
            break
    return list(reversed(out.values()))


def pretty_evidence(ev, maxlen: int = 120) -> str:
    if not isinstance(ev, dict):
        return f"[BAD TYPE] {str(ev)[:maxlen]}"
    src   = ev.get("source", "?")
    val   = ev.get("value", "")[:maxlen]
    det   = ev.get("details", {})
    meta  = ev.get("meta", {})
    if src == "entity":
        return f"[entity_search] {val} — {det.get('count')} matches (mode: {meta.get('mode')})"
    return f"[{src}] {val} ({det.get('year')}, {det.get('doc_name')})"

# ---------------------------------------------------------------------------
#  HTTP WRAPPER (shared client)
# ---------------------------------------------------------------------------
async def _post(client: httpx.AsyncClient, url: str, payload: dict):
    try:
        r = await client.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        print("SERVER RESP:", e.response.text[:800])
        raise

# ---------------------------------------------------------------------------
#  LOGGING HELPERS
# ---------------------------------------------------------------------------

def log_reasoning(iter_idx: int, data: dict):
    print(f"\nITERATION {iter_idx}")
    print(" → actions       :", [a["type"] for a in data.get("actions", [])])
    print(" → hypothesis    :", data.get("hypothesis"))
    print(" → confidence    :", data.get("confidence"))
    print(" → active_q      :", data.get("active_question"))
    thoughts = data.get("agent_thoughts", "")
    print(" → thoughts      :", thoughts[:200].replace("\n", " ") + ("…" if len(thoughts) > 200 else ""))
    print(" → supporting_evd:", len(data.get("supporting_evidence", [])))

# ---------------------------------------------------------------------------
#  AGENT LOOP
# ---------------------------------------------------------------------------
async def agent_loop(user_query: str):
    context, log, prev_hyp, support = [], [], [], []
    thoughts = ""
    active_q = user_query
    stuck_rounds = 0  # early‑stop guard

    async with httpx.AsyncClient(base_url=SERVER, limits=httpx.Limits(max_keepalive_connections=KEEPALIVE)) as client:
        for it in range(MAX_ITERATIONS):
            # --- prepare request -------------------------------------------------
            reasoning_req = {
                "user_query": user_query,
                "active_question": active_q,
                "agent_thoughts": thoughts,
                "context": [sanitize_evidence(e) for e in context][-CTX_LIMIT:],
                "previous_hypotheses": prev_hyp,
                "supporting_evidence": support,
                "reasoning_log": dedup_log(log, 8),
                "iteration": it,
            }

            # --- call LLM reasoning --------------------------------------------
            reasoning = await _post(client, ENDPOINTS["LLM"], reasoning_req)
            log_reasoning(it, reasoning)  # вывели ход рассуждений

            actions = reasoning.get("actions", [])
            if not actions:
                print("No actions returned → stop.")
                break

            # --- parallel fetch -------------------------------------------------
            sem = asyncio.Semaphore(PARALLEL_LIMIT)

            async def fetch(act):
                url, base_payload = ENDPOINTS[Act(act["type"]).value]
                payload = base_payload.copy()
                if act["type"] == Act.ENTITY.value:
                    payload["entities"] = [e.strip() for e in act["query"].split(",")]
                else:
                    payload["query"] = act["query"]
                async with sem:
                    return act, await _post(client, url, payload)

            fetched = await asyncio.gather(*(fetch(a) for a in actions))

            # --- process evidence ----------------------------------------------
            new_evidence = []
            for act, evs in fetched:
                print(f"  [{act['type']}] '{act['query']}' → {len(evs)} evid")
                for ev in evs:
                    print("    ", pretty_evidence(ev))
                log.append({"iteration": it, "action": act["type"], "query": act["query"], "result_count": len(evs)})
                new_evidence.extend(evs)

            context.extend(new_evidence)
            support  = reasoning.get("supporting_evidence", support)
            thoughts = reasoning.get("agent_thoughts", thoughts)
            active_q = reasoning.get("active_question", active_q)
            prev_hyp = reasoning.get("previous_hypotheses", prev_hyp)

            # --- finalize / early‑stop -----------------------------------------
            if reasoning.get("finalize"):
                print("\n=== FINALIZED ===")
                break
            if not new_evidence:
                stuck_rounds += 1
                if stuck_rounds >= 4:
                    print("Stuck 4 rounds with no new evidence → stop.")
                    break
            else:
                stuck_rounds = 0

        # verdict --------------------------------------------------------------
        if reasoning.get("hypothesis"):
            verdict_req = {
                "hypothesis": reasoning["hypothesis"],
                "supporting_evidence": support,
            }
            verdict = await _post(client, ENDPOINTS["VERDICT"], verdict_req)
            print("\n=== VERDICT ===")
            print(" verdict  :", verdict.get("verdict"))
            print(" details  :", verdict.get("details"))

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(agent_loop("Find all information and facts related to the object or place named Bararoá."))
