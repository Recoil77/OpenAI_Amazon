import os
import asyncio
import json
from datetime import datetime
from utils.calls import call_fastapi_async
from dotenv import load_dotenv
load_dotenv()

SERVER_IP = os.getenv("SERVER_ADDRESS")
SERVER_ADDRESS = f"http://{SERVER_IP}:8100"

LLM_REASONING_URL =     f"{SERVER_ADDRESS}/llm_reasoning"
VECTOR_SEARCH_URL =     f"{SERVER_ADDRESS}/vector_search"
ENTITY_SEARCH_URL =     f"{SERVER_ADDRESS}/entity_search"
HYBRID_SEARCH_URL =     f"{SERVER_ADDRESS}/entity_hybrid"
GENERAL_KNOWLEDGE_URL = f"{SERVER_ADDRESS}/general_knowledge"
WEB_SEARCH_URL =        f"{SERVER_ADDRESS}/web_search"
MAX_ITERATIONS = 8


def get_endpoint_and_payload(action, context):
    action_type = action['type']
    query = action.get('query', '')
    if action_type == "vector_search":
        return VECTOR_SEARCH_URL, {"query": query, "k": 128, "bge_threshold": 0.25, "semantic_threshold": 0.5 }
    elif action_type == "entity_hybrid":
        return HYBRID_SEARCH_URL, {"query": query, "k": 128, "bge_threshold": 0.25, "semantic_threshold": 0.5 }
    elif action_type == "entity_search":
        if isinstance(query, str):
            entities = [e.strip() for e in query.split(",")]
        return ENTITY_SEARCH_URL, {"entities": entities, "mode": "substring"}
    elif action_type == "general_knowledge":
        return GENERAL_KNOWLEDGE_URL, {"query": query}
    elif action_type == "web_search":
        return WEB_SEARCH_URL, {"query": query, "search_context_size": "medium"}
    else:
        raise ValueError(f"Unknown action type: {action_type}")


def dedup_log(log: list[dict], limit: int = 16) -> list[dict]:
    reversed_log = list(reversed(log))
    aggregated: dict[tuple[str, str], dict] = {}
    for item in reversed_log:
        key = (item["action"], item["query"])
        if key not in aggregated:
            aggregated[key] = item.copy()
        else:
            aggregated[key]["result_count"] += item["result_count"]
    unique_recent = list(reversed(list(aggregated.values())))[:limit]
    return unique_recent

def dedup_supporting_evidence(evidences: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for ev in evidences:
        
        key = (
            ev.get("source"),
            ev.get("value", "").strip(),
            json.dumps(ev.get("details", {}), sort_keys=True)
        )
        if key not in seen:
            unique.append(ev)
            seen.add(key)
    return unique


def pretty_evidence(ev, maxlen=75):
    if isinstance(ev, dict):
        src = ev.get('source', '?')
        val = ev.get('value', '')[:maxlen]
        details = ev.get('details', {})
        meta = ev.get('meta', {})
        if src == "entity_search":
            count = details.get('count')
            mode = meta.get('mode')
            return f"[entity_search] {val} — {count} matches (mode: {mode})"
        year = details.get('year')
        doc = details.get('doc_name')
        return f"[{src}] {val} ({year}, {doc})"
    return f"[BAD TYPE: {type(ev)}] {str(ev)[:maxlen]}"


async def agent_loop(user_query):
    context = []
    reasoning_log = []
    previous_hypotheses = []
    supporting_evidence = []
    agent_thoughts: str = ""
    active_question = user_query

    for iteration in range(MAX_ITERATIONS):
        reasoning_request = {
            "user_query": user_query,
            "active_question": active_question,
            "agent_thoughts": agent_thoughts,
            "context": context,
            "previous_hypotheses": previous_hypotheses,
            "supporting_evidence": supporting_evidence,
            "reasoning_log": dedup_log(reasoning_log, 16),
            "iteration": iteration,
        }

        reasoning = await call_fastapi_async(LLM_REASONING_URL, reasoning_request)
        new_evidence = reasoning.get("new_facts", [])
        supporting_evidence.extend(new_evidence)
        supporting_evidence[:] = dedup_supporting_evidence(supporting_evidence)  

        print(f"ITERATION:   {iteration}")
        print(f"→ Actions: {[a['type'] for a in reasoning['actions']]}")
        print(f"→ Hypothesis: {reasoning.get('hypothesis')}")
        print(f"→ Memory: {reasoning.get('agent_thoughts')}")
        print(f"→ Supporting evidence new: {len(reasoning.get('new_facts', []))}")
        print(f"→ Confidence: {reasoning.get('confidence')}")
        print(f"→ Active question: {reasoning.get('active_question')}")
        print(f"→ Total supporting_evidences : {len(supporting_evidence)}")

        # print("*"*150)        
        # print(supporting_evidence)
        # print("*"*150)
        # print(dedup_log(reasoning_log, 8)) 
        # print("*"*150) 

        all_new_evidence = []
        for action in reasoning["actions"]:
            endpoint_url, payload = get_endpoint_and_payload(action, context)
            print(f"  [{action['type']}] Query: {action['query']}")
            try:
                evidence = await call_fastapi_async(endpoint_url, payload)
            except Exception as e:
                print(f"    !!! Endpoint error: {e}")
                evidence = []
            print(f"    ↳ {len(evidence)} evidence found")
            for ev in evidence:
                print("    ", pretty_evidence(ev))

            reasoning_log.append({
                "iteration": iteration,
                "action": action['type'],
                "query": action['query'],
                "result_count": len(evidence),
            })
            all_new_evidence.extend(evidence)
        
        context.extend(all_new_evidence)
        active_question = reasoning.get("active_question", active_question)
        previous_hypotheses = reasoning.get("previous_hypotheses", previous_hypotheses)
        agent_thoughts = reasoning.get("agent_thoughts", agent_thoughts)
        if reasoning.get("finalize"):
            print("\n=== FINALIZED ===")
            print(f"Final hypothesis: {reasoning.get('hypothesis')}")
            print(f"Supporting evidence: {len(reasoning.get('new_facts', []))}")
            break

    print("\n--- Reasoning complete ---\nFull log:")
    for step in reasoning_log:
        print(f"{step['iteration']}: [{step['action']}] {step['query']} ({step['result_count']})")
    print("\nContext size:", len(context))

    print("\n--- Supporting Evidence ---")
    for xxx in supporting_evidence:
        print(f"Supp: {xxx}")


    
    output = {
        "user_query": user_query,
        "final_hypothesis": reasoning.get("hypothesis"),
        "supporting_evidence": supporting_evidence,
        "reasoning_log": reasoning_log,
        "context": context,
        "agent_thoughts": agent_thoughts,
        "previous_hypotheses": previous_hypotheses,
    }
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"agent_run_{dt}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n=== Saved agent reasoning to: {fname} ===\n")


if __name__ == "__main__":
    user_query = """Find all information and facts related to the object or place named Antipas."""
    
    asyncio.run(agent_loop(user_query))


    # user_query = """
    # Find all information and facts related to the object or place named Bararoá.
    # Note: The name may have alternative spellings or variants in the documents (e.g., Bararoa, Bararoã, Bararóa, Bararua, Bararoá parish, etc.). 
    # Please check for different spellings and similar-looking names, and include relevant facts for any such variants referring to the same place.
    # """

#   user_query = """Find all information and facts related to the object or place named Thomar. Please check for different spellings and similar-looking names, and include relevant facts for any such variants referring to the same place."""
#   user_query = """Find and extract all information and facts related to abandoned settlements, remains of fortifications, artificial mounds, earthworks, old or ruined structures, and any other traces of former human activity (e.g., deserted villages, ruins, old roads, ancient clearings, embankments, etc.). Focus on any evidence or descriptions of abandoned or ancient sites that could serve as a basis for further investigation."""
 