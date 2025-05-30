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


# # --- Prompt builder ---
# def build_reasoning_prompt(data: ReasoningRequest) -> str:
#     facts = [f"- [{ev.source}] {ev.value}" for ev in data.context[:8]]
#     log_lines = [
#         f"{item.get('iteration', '?')}. [{item.get('action', '')}] {item.get('query', '')} ({item.get('result_count', '?')})"
#         for item in data.reasoning_log[-5:]
#     ]
#     prompt = f"""You are a deep reasoning engine for historical research.

# # Main user question:
# {data.user_query}

# # Current active question:
# {data.active_question}

# # Context (facts so far):
# {chr(10).join(facts) if facts else '(none yet)'}

# # Reasoning log (last steps):
# {chr(10).join(log_lines) if log_lines else '(empty)'}

# # Previous hypotheses:
# {chr(10).join(data.previous_hypotheses) if data.previous_hypotheses else '(none yet)'}

# # Instructions:
# Based on the current context, propose the next best actions for finding the answer. For each action, specify the type ("vector_search", "entity_search", "web_search", etc), a query (short text), and a reason (why this is the best next step).
# If you have enough evidence to finalize the answer, return "finalize": true, along with "hypothesis", "supporting_evidence", and "confidence".
# Return ONLY valid JSON. Example:
# {{
# "actions": [{{"type": "...", "query": "...", "reason": "..."}}],
# "finalize": false,
# "active_question": "...",
# "hypothesis": "...",
# "supporting_evidence": [{{...}}],
# "confidence": 0.8
# }}
# """
#     return prompt


# app/func.py

# def build_reasoning_prompt(data):
#     # data: ReasoningRequest

#     facts = [
#         f"- [{ev.source}] {ev.value}" 
#         for ev in getattr(data, "context", [])[:8]
#     ]
#     log_lines = [
#         f"{item.get('iteration', '?')}. [{item.get('action', '')}] {item.get('query', '')} ({item.get('result_count', '?')})"
#         for item in getattr(data, "reasoning_log", [])[-5:]
#     ]

#     preferred_order = """
# Preferred order of actions:
# 1. Reformulate the question if unclear or ambiguous.
# 2. Use vector_search for internal document search.
# 3. Use entity_search for entity statistics.
# 4. Use general_knowledge and web_search for reference/background, but never for reasoning completion.
# 5. Only finish reasoning when all reasonable actions are exhausted and a confident hypothesis is formed.
# """

#     prompt = f"""
# You are a deep reasoning engine for historical research.

# Sources available:
# - vector_search: returns detailed fragments from internal documents (summary, facts, year, etc).
# - web_search: returns relevant fragments from external sources.
# - entity_search: returns statistics on specified entities.
# - general_knowledge: provides brief reference facts.
# - reformulate_question: helps to clarify or rephrase the current question.

# {preferred_order}

# # Main user question:
# {data.user_query}

# # Current active question:
# {data.active_question}

# # Context (evidence found so far):
# {chr(10).join(facts) if facts else '[none yet]'}

# # Reasoning log (last steps):
# {chr(10).join(log_lines) if log_lines else '[no previous actions]'}

# Respond strictly with a valid JSON object:
# {{
#   "actions": [
#     {{"type": "vector_search"|"web_search"|"entity_search"|"general_knowledge"|"reformulate_question", "query": "<query>", "reason": "<why this action>"}}
#   ],
#   "finalize": false,
#   "active_question": "<updated question if any>",
#   "hypothesis": "<current hypothesis if any>",
#   "supporting_evidence": [],
#   "confidence": null
# }}
# Do not include explanations, markdown, or text outside JSON. Do not use real names, facts, or terms—use only abstract reasoning and structure.
# """

# #     return prompt
# from textwrap import dedent
# from app.classes import ReasoningRequest


# def build_reasoning_prompt(data: ReasoningRequest) -> str:
#     """Return a single prompt string that guides the LLM through one
#     step of the reasoning loop.

#     The prompt is intentionally verbose so the model receives all
#     necessary state (user question, active sub-question, context
#     evidence, reasoning log, previous hypotheses) **and** a compact
#     but precise set of rules that it must follow when producing JSON
#     output.

#     Args:
#         data: A `ReasoningRequest` carrying the current reasoning state.

#     Returns:
#         A fully-formed prompt string ready to be sent as the *user* part
#         of a ChatCompletion call.
#     """

#     # ---------- Helper blocks ----------
#     facts_block = "\n".join(
#         f"- [{ev.source}] {ev.value}" for ev in (data.context or [])[:8]
#     ) or "[none yet]"

#     log_block = "\n".join(
#         f"{item.get('iteration', '?')}. [{item.get('action', '')}] "
#         f"{item.get('query', '')} ({item.get('result_count', '?')})"
#         for item in (data.reasoning_log or [])[-5:]
#     ) or "[no previous actions]"

#     hypotheses_block = "\n".join(data.previous_hypotheses) if data.previous_hypotheses else "[none]"

#     evidence_block = "\n".join(
#         f"- [{ev.source}] {ev.value}" for ev in (data.supporting_evidence or [])
#     ) or "[none yet]"

#     preferred_order = dedent(
#         """
#         Preferred order of actions:
#         1. Reformulate the question if unclear or ambiguous.
#         2. Use vector_search for internal document search.
#         3. Use entity_search for entity statistics.
#         4. Use general_knowledge and web_search for reference/background, but **never** for reasoning completion.
#         5. Call finalize only when no further useful actions remain **and** a confident hypothesis is formed.
#         """
#     ).strip()

#     # ---------- Prompt template ----------
#     prompt = dedent(
#         f"""
#         You are a deep reasoning engine for historical research.

#         Sources available:
#         - vector_search   → returns detailed fragments from internal documents (summary, facts, year, etc.).
#         - web_search      → returns relevant fragments from external sources.
#         - entity_search   → returns statistics on specified entities.
#         - general_knowledge → returns concise reference facts.
#         - reformulate_question → helps clarify or rephrase the current question.

#         For each reasoning step you may issue **at most one** action of each type.  
#         If additional calls of the same type are required, ask for them in a subsequent reasoning cycle.

#         {preferred_order}

#         # Main user question:
#         {data.user_query}

#         # Current active question:
#         {data.active_question}

#         # Current supporting evidence:
#         {evidence_block}

#         # Context (evidence found so far):
#         {facts_block}

#         # Reasoning log (last steps):
#         {log_block}

#         # Previous hypotheses:
#         {hypotheses_block}

#         # Instructions for investigating abandoned / lost settlements:
#         - Do **not** rely solely on proper names.  
#         - Look for descriptions mentioning deserted villages, lost towns, abandoned missions, ruins, etc.  
#         - Use indirect clues—distances, neighbouring places, geographical features or historical events—to locate the site.  
#         - State the level of certainty and any limitations in your hypothesis.

#         # Instructions for entity_search & metadata:
#         - Use entity_search to gauge the prevalence or ambiguity of names / terms in the corpus before committing to a hypothesis.
#         - Combine entity statistics with context evidence; do **not** rely on counts alone.

#         # Instructions for supporting_evidence:
#         - Each item **must** be an object with keys: `source`, `value`, `details`, `meta`.  
#         - The **details** field **must** be a dictionary/JSON object (e.g., {{"summary": "…"}}); **never** a plain string.  
#         - Include only evidence that actually exists in *context*; never invent.
#         - Keep relevant evidence from previous steps unless clearly irrelevant.

#         # Instructions for search‑query reformulation:
#         - Queries must be short, natural phrases typical of historical texts—no Boolean logic or meta-language.

#         Respond **strictly** with a valid JSON object:
#         {{
#           "actions": [
#             {{"type": "vector_search"|"web_search"|"entity_search"|"general_knowledge"|"reformulate_question", "query": "<query>", "reason": "<why>"}}
#           ],
#           "finalize": false,
#           "active_question": "<updated question if any>",
#           "hypothesis": "<current hypothesis if any>",
#           "supporting_evidence": [],
#           "confidence": null
#         }}
#         Do **not** output anything outside this JSON—no markdown, headings, or explanations.
#         """
#     ).strip()

#     return prompt

from textwrap import dedent
from app.classes import ReasoningRequest


def build_reasoning_prompt(data: ReasoningRequest) -> str:
    """Build the reasoning‑step prompt.
    Provides all state plus strict instructions for JSON output.
    """

    # ---------- Helper blocks ----------
    facts_block = "\n".join(
        f"- [{ev.source}] {ev.value}" for ev in (data.context or [])[:8]
    ) or "[none yet]"

    log_block = "\n".join(
        f"{item.get('iteration', '?')}. [{item.get('action', '')}] "
        f"{item.get('query', '')} ({item.get('result_count', '?')})"
        for item in (data.reasoning_log or [])[-5:]
    ) or "[no previous actions]"

    hypotheses_block = "\n".join(data.previous_hypotheses) if data.previous_hypotheses else "[none]"

    preferred_order = dedent(
        """
        Preferred order of actions:
        1. Use vector_search for internal document search.
        2. Use entity_search for entity statistics.
        3. Use general_knowledge and web_search for reference/background, but **never** for reasoning completion.
        4. Call finalize only when no further useful actions remain **and** a confident hypothesis is formed.
        """
    ).strip()

    prompt = dedent(
        f"""
        You are a deep reasoning engine for historical research.

        Sources available:
        - vector_search   → returns detailed fragments from internal documents (summary, facts, year, etc.).
        - web_search      → returns relevant fragments from external sources.
        - entity_search   → returns statistics on specified entities.
        - general_knowledge → returns concise reference facts.

        For each reasoning step you may issue **at most one** action of each type.  
        If additional calls of the same type are required, ask for them in a subsequent reasoning cycle.

        {preferred_order}

        # Main user question:
        {data.user_query}

        # Current active question:
        {data.active_question}

        # Context (evidence found so far):
        {facts_block}

        # Reasoning log (last steps):
        {log_block}

        # Previous hypotheses:
        {hypotheses_block}

        # Instructions for investigating abandoned / lost settlements:
        - Do **not** rely solely on proper names.  
        - Look for descriptions mentioning deserted villages, lost towns, abandoned missions, ruins, etc.  
        - Use indirect clues—distances, neighbouring places, geographical features or historical events—to locate the site.  
        - State the level of certainty and any limitations in your hypothesis.

        # Instructions for entity_search & metadata:
        - Use entity_search to gauge the prevalence or ambiguity of names / terms in the corpus before committing to a hypothesis.
        - Combine entity statistics with context evidence; do **not** rely on counts alone.

        # Instructions for supporting_evidence:
        - **Condense all evidence into exactly one object** and place it as the sole element of the `supporting_evidence` array.  
        - Preserve every fact by aggregating multiple citations inside that object's `details` or `meta` fields (e.g., `meta.sources`).  
        - The object must include keys: `source`, `value`, `details`, `meta`. Use a generic `source` such as "aggregated" if necessary.  
        - The **details** field **must** be a dictionary/JSON object (e.g., {{"summary": "…", "highlights": ["…", "…"]}}); **never** a plain string.  
        - Include only evidence that actually exists in *context*; never invent.  
        - On subsequent steps, merge new relevant evidence into this same object instead of adding additional elements.  
        - Remove data from the aggregate only if it is proven irrelevant.

        # Instructions for search‑query reformulation (embedding search):
        - Generate **exactly one** `vector_search` action per reasoning step.  
        - The `query` string **must** be a single short phrase of **3 – 6 meaningful words** (≈5 – 7 tokens).  
        - Avoid comma‑separated lists, multiple synonyms, Boolean operators, or meta‑language ("find", "ruins OR mission").  
        - If you need a substantially different formulation, use a new reasoning step.

        Respond **strictly** with a valid JSON object:
        {{
          "actions": [
            {{"type": "vector_search"|"web_search"|"entity_search"|"general_knowledge", "query": "<query>", "reason": "<why>"}}
          ],
          "finalize": false,
          "active_question": "<updated question if any>",
          "hypothesis": "<current hypothesis if any>",
          "supporting_evidence": [],
          "confidence": null
        }}
        Do **not** output anything outside this JSON—no markdown, headings, or explanations.
        """
    ).strip()

    return prompt


