import asyncio
from utils.calls import call_fastapi_async

# Настройки
LLM_REASONING_URL = "http://192.168.168.5:8100/llm_reasoning"
VECTOR_SEARCH_URL = "http://192.168.168.5:8100/vector_search"
ENTITY_SEARCH_URL = "http://192.168.168.5:8100/entity_search"
GENERAL_KNOWLEDGE_URL = "http://192.168.168.5:8100/general_knowledge"
WEB_SEARCH_URL = "http://192.168.168.5:8100/web_search"
REFORMULATE_URL = "http://192.168.168.5:8100/reformulate_question"
GET_VERDICT_URL = "http://192.168.168.5:8100/get_verdict"

MAX_ITERATIONS = 10

def get_endpoint_and_payload(action, context):
    """Маппинг типа action к endpoint и payload"""
    action_type = action['type']
    query = action.get('query', '')
    if action_type == "vector_search":
        return VECTOR_SEARCH_URL, {"query": query, "k": 64, "bge_threshold": 0.2, "semantic_threshold": 0.25 }
    elif action_type == "entity_search":
        if isinstance(query, str):
            # Разбиваем строку по запятым и убираем пробелы вокруг
            entities = [e.strip() for e in query.split(",")]
        return ENTITY_SEARCH_URL, {"entities": entities, "mode": "substring"}
    elif action_type == "general_knowledge":
        return GENERAL_KNOWLEDGE_URL, {"query": query}
    elif action_type == "web_search":
        return WEB_SEARCH_URL, {"query": query, "search_context_size": "medium"}
    elif action_type == "reformulate_question":
        return REFORMULATE_URL, {
            "user_query": action["query"],                # или user_query/active_question
            "active_question": action["query"],           # можно дублировать
            "context": context,                           # передаем context (список evidence)
            "reasoning_log": []                           # можно пустой или с шагами
        }
    else:
        raise ValueError(f"Unknown action type: {action_type}")

def pretty_evidence(ev, maxlen=120):
    if isinstance(ev, dict):
        src = ev.get('source', '?')
        val = ev.get('value', '')[:maxlen]
        details = ev.get('details', {})
        meta = ev.get('meta', {})
        # Для entity_search выводим особым образом
        if src == "entity":
            count = details.get('count')
            mode = meta.get('mode')
            return f"[entity_search] {val} — {count} matches (mode: {mode})"
        # Для обычных evidence — стандартно
        year = details.get('year')
        doc = details.get('doc_name')
        return f"[{src}] {val} ({year}, {doc})"
    return f"[BAD TYPE: {type(ev)}] {str(ev)[:maxlen]}"

async def agent_loop(user_query):
    context = []
    reasoning_log = []
    previous_hypotheses = []
    accumulated_supporting = []          # <-- новый список
    active_question = user_query

    for iteration in range(MAX_ITERATIONS):
        reasoning_request = {
            "user_query": user_query,
            "active_question": active_question,
            "context": context,
            "previous_hypotheses": previous_hypotheses,
            "reasoning_log": reasoning_log,
            "supporting_evidence": accumulated_supporting,   # <-- отправляем в LLM
            "iteration": iteration,
        }

        reasoning = await call_fastapi_async(LLM_REASONING_URL, reasoning_request)
        print(f"ITERATION:   {iteration}")
        print(f"→ Actions: {[a['type'] for a in reasoning['actions']]}")
        print(f"→ Hypothesis: {reasoning.get('hypothesis')}")
        print(f"→ Supporting evidence: {len(reasoning.get('supporting_evidence', []))}")
        print(f"→ Confidence: {reasoning.get('confidence')}")
        print(f"→ Active question: {reasoning.get('active_question')}")

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

            # reasoning_log update
            reasoning_log.append({
                "iteration": iteration,
                "action": action['type'],
                "query": action['query'],
                "result_count": len(evidence),
                #"evidence_summary": [ev['value'][:80] for ev in evidence]
            })
            all_new_evidence.extend(evidence)
        # Обновляем context и поля
        accumulated_supporting = reasoning.get("supporting_evidence", [])
        context.extend(all_new_evidence)
        active_question = reasoning.get("active_question", active_question)
        previous_hypotheses = reasoning.get("previous_hypotheses", previous_hypotheses)

        # Проверка на завершение reasoning
        if reasoning.get("finalize"):
            print("\n=== FINALIZED ===")
            print(f"Final hypothesis: {reasoning.get('hypothesis')}")
            print(f"Supporting evidence: {len(reasoning.get('supporting_evidence', []))}")
            break

    print("\n--- Reasoning complete ---\nFull log:")
    for step in reasoning_log:
        print(f"{step['iteration']}: [{step['action']}] {step['query']} ({step['result_count']})")
        for s in step.get('evidence_summary', []):
            print("    ", s)
    print("\nContext size:", len(context))

    print("\n--- Supporing Evidence ---")
    for xxx in (reasoning.get('supporting_evidence', [])):
        print(f"Supp: {xxx}")

    # Финальный вердикт (если нужно)
    try:
        if reasoning.get("hypothesis"):
            verdict_req = {
                "hypothesis": reasoning["hypothesis"],
                "supporting_evidence": reasoning.get("supporting_evidence", [])
            }
            verdict = await call_fastapi_async(GET_VERDICT_URL, verdict_req)
            print("\n=== FINAL VERDICT ===")
            print(f"Verdict: {verdict.get('verdict')}")
            print(f"Details: {verdict.get('details')}")
            print(f"GN answer: {verdict.get('general_knowledge_answer', 'general_knowledge_answer')}")
            print(f"Web search: {verdict.get('web_search_answer', 'general_knowledge_answer')}")
    except Exception as e:
        print(f"[Verdict error] {e}")

if __name__ == "__main__":
    user_query = "Find and identify a specific abandoned settlement, village, or mission described in the documents. Focus on one such object and collect all possible indirect clues about its location—such as distances, nearby places, and geographical features. Summarize all findings about this object."
    asyncio.run(agent_loop(user_query))

#"Find a description of an abandoned or lost objects in the texts and try to determine its location using any indirect information—such as distances, neighboring places, or geographical clues."
#"Find a settlement or place name in the texts. Determine if it is now abandoned or still known, using any supporting details from context."


