from app.classes import ReasoningRequest, ReformulateRequest

# --- PROMPT builder ---
def build_reformulate_prompt(data: ReformulateRequest) -> str:
    prompt = f"""You are a research assistant helping to reformulate historical questions for a deep reasoning system.

# Original user question:
{data.user_query}

# Current active question:
{data.active_question}

# Context (short facts):
"""
    # Собираем контекст для промпта (максимум 5 фактов, чтобы не раздувать prompt)
    facts = [f"- [{ev.source}] {ev.value}" for ev in data.context[:5]]
    if facts:
        prompt += "\n".join(facts)
    else:
        prompt += "(No new evidence yet.)"
    prompt += """

# Instructions:
You are stuck: recent reasoning steps did not produce new evidence.
Propose a more specific or alternative formulation of the active question using concrete details from context (if any).
If relevant, suggest 1-2 alternative phrasings that might yield new evidence. 
Return ONLY JSON:
{
  "reformulated_question": "...",
  "alternatives": ["...", "..."],
  "reason": "..."
}
"""
    return prompt


# --- Prompt builder ---
def build_reasoning_prompt(data: ReasoningRequest) -> str:
    facts = [f"- [{ev.source}] {ev.value}" for ev in data.context[:8]]
    log_lines = [
        f"{item.get('iteration', '?')}. [{item.get('action', '')}] {item.get('query', '')} ({item.get('result_count', '?')})"
        for item in data.reasoning_log[-5:]
    ]
    prompt = f"""You are a deep reasoning engine for historical research.

# Main user question:
{data.user_query}

# Current active question:
{data.active_question}

# Context (facts so far):
{chr(10).join(facts) if facts else '(none yet)'}

# Reasoning log (last steps):
{chr(10).join(log_lines) if log_lines else '(empty)'}

# Previous hypotheses:
{chr(10).join(data.previous_hypotheses) if data.previous_hypotheses else '(none yet)'}

# Instructions:
Based on the current context, propose the next best actions for finding the answer. For each action, specify the type ("vector_search", "entity_search", "web_search", etc), a query (short text), and a reason (why this is the best next step).
If you have enough evidence to finalize the answer, return "finalize": true, along with "hypothesis", "supporting_evidence", and "confidence".
Return ONLY valid JSON. Example:
{{
"actions": [{{"type": "...", "query": "...", "reason": "..."}}],
"finalize": false,
"active_question": "...",
"hypothesis": "...",
"supporting_evidence": [{{...}}],
"confidence": 0.8
}}
"""
    return prompt