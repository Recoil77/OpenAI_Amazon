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






# from textwrap import dedent
# from app.classes import ReasoningRequest


# def build_reasoning_prompt(data: ReasoningRequest) -> str:
#     """Construct the prompt for one reasoning step.
#     Provides the model with current state plus strict rules for JSON output.
#     """

#     # ---------- Helper blocks ----------
#     facts_block = "\n".join(
#         f"- [{ev.source}] {ev.value}" for ev in (data.context or [])[:8]
#     ) or "[none yet]"

#     log_block = "\n".join(
#         f"{item.get('iteration')}. [{item.get('action')}] "
#         f"{item.get('query')} ({item.get('result_count')})"
#         for item in (data.reasoning_log or [])
#     ) or "[no previous actions]"

#     hypotheses_block = "\n".join(data.previous_hypotheses) if data.previous_hypotheses else "[none]"

#     preferred_order = dedent(
#         """
#         Preferred order of actions:
#         1. Use vector_search for internal document search.
#         2. Use entity_search for entity statistics.
#         3. Use general_knowledge and web_search for reference/background, but **never** for reasoning completion.
#         4. Call finalize only when no further useful actions remain **and** a confident hypothesis is formed.
#         """
#     ).strip()

#     prompt = dedent(
#         f"""
#         You are a deep reasoning engine for historical research.

#         Sources available:
#         - vector_search   → returns detailed fragments from internal documents (summary, facts, year, etc.).
#         - web_search      → returns relevant fragments from external sources.
#         - entity_search   → returns statistics on specified entities.
#         - general_knowledge → returns concise reference facts.

#         For each reasoning step you may issue **at most one** action of each type.  
#         If additional calls of the same type are required, ask for them in a subsequent reasoning cycle.

#         {preferred_order}

#         # Main user question:
#         {data.user_query}

#         # Current active question:
#         {data.active_question}

#         # Context (evidence found so far):
#         {facts_block}

#         # Recent actions (iteration · type · query → hits):
#         {log_block}

#         # Previous hypotheses:
#         {hypotheses_block}

#         # Instructions for investigating abandoned / lost settlements:
#         - **Select a single candidate object** (abandoned village / mission / settlement) and keep focus on it through the reasoning cycle.
#         - Do **not** hop between multiple objects unless the current candidate is definitively ruled out.
#         - Gather every indirect clue to its location — distances, neighbouring places, rivers, landmarks, travel directions, historical events.
#         - Cross‑check descriptions across sources to strengthen or weaken the hypothesis.
#         - Clearly state the level of certainty and note limitations if the exact location remains ambiguous.

#         # Object‑focus strategy:
#         - After choosing a candidate, rewrite `active_question` so it explicitly names that object (e.g., *"Locate the abandoned Jesuit mission of Exaltación"*).
#         - Accumulate clues step by step and merge them into the aggregated `supporting_evidence` object (e.g., add `details.clues`).
#         - Switch to a different candidate **only** if new evidence contradicts the current hypothesis or proves the object irrelevant.

#         # Instructions for entity_search & metadata:
#         - `entity_search` is the **first‑line, low‑cost tool** to see whether a specific **named entity** is actually present in our corpus metadata (settlements, rivers, missions, explorers, tribes, years).  
#         - Always send a **comma‑separated list of 1‑5 concise names** — *never* descriptive phrases.  Good examples:  
#           · `Villarico`, `River Moraji`, `Jesuit mission`  
#           · `Aguirre`, `Mission Exaltación`, `Santo Tomé`  
#         - Use it aggressively to narrow candidates **before** investing in additional `vector_search`.  Typical loop:
#             1. Extract 2‑5 plausible names or spellings from the current clue (older texts may vary orthography).  
#             2. Run `entity_search`; note hit counts.  
#             3. Keep the top‑hit names and discard zero‑hit variants.
#         - A rare entity may still be correct — cross‑check with indirect clues (distances, rivers, landmarks).
#         - Do **not** query phrases like "abandoned village across river"; they yield zero hits.

#         # Guidance for web_search:
#         - Use web_search **only after** you have at least **one concrete name** *and* a regional pointer (e.g., country, province, nearby river).  
#         - Form the query as `<name> <region or river>` — e.g., `"Villarico" "Rio Moraji" Brazil`.  
#         - Avoid generic or overly broad phrases; they add noise and rarely help locate forgotten sites from 17‑18th‑century texts.
#         - Use web_search **only after** you have at least one well‑defined name or coordinate clue.  
#         - Query should include that concrete name plus one distinctive qualifier (e.g., `"Exaltación Bolivia" river`).  
#         - Avoid generic phrases that will return broad, noisy results.

#         # Instructions for supporting_evidence:
#         - **Condense all evidence into exactly one object** and place it as the sole element of the `supporting_evidence` array.  
#         - Aggregate multiple facts in this object's `details`/`meta` (e.g., `meta.sources`).  
#         - Keys required: `source`, `value`, `details`, `meta`.  Details **must** be a dictionary, not a string.  
#         - Add new facts to this same object; remove only if proven irrelevant.

#         # Instructions for search‑query reformulation (embedding search):
#         - Generate **exactly one** `vector_search` action per reasoning step.  
#         - The `query` string **must** be a single phrase of **3–6 meaningful words** (≈5–7 tokens).  
#         - Avoid lists, commas, Boolean operators, meta‑language.  
#         - Use a new reasoning step for any substantially different formulation.

#         # Anti‑stall rules:
#         - Track your last actions (see *Recent actions* above).  
#         - **Do not** issue a `vector_search` whose query is semantically similar to any query already tried.  
#         - If two consecutive `vector_search` attempts each returned fewer than **3** new evidence items, your **next** action must be either `entity_search`, `web_search`, or `general_knowledge` instead of another `vector_search`.  
#         - Aim to collect evidence from **at least two different action types** before finalizing.

#         Respond **strictly** with a valid JSON object:
#         {{
#           "actions": [
#             {{"type": "vector_search"|"web_search"|"entity_search"|"general_knowledge", "query": "<query>", "reason": "<why>"}}
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
    """Construct the prompt for one reasoning step.
    Provides the model with current state plus strict rules for JSON output.
    """

    # ---------- Helper blocks ----------
    facts_block = "\n".join(
        f"- [{ev.source}] {ev.value}" for ev in (data.context or [])[:8]
    ) or "[none yet]"

    log_block = "\n".join(
        f"{item.get('iteration')}. [{item.get('action')}] "
        f"{item.get('query')} ({item.get('result_count')})"
        for item in (data.reasoning_log or [])
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

        # Recent actions (iteration · type · query → hits):
        {log_block}

        # Previous hypotheses:
        {hypotheses_block}

        # Instructions for investigating abandoned / lost settlements:
        - **Select a single candidate object** (abandoned village / mission / settlement) and keep focus on it through the reasoning cycle.
        - Do **not** hop between multiple objects unless the current candidate is definitively ruled out.
        - Gather every indirect clue to its location — distances, neighbouring places, rivers, landmarks, travel directions, historical events.
        - Cross‑check descriptions across sources to strengthen or weaken the hypothesis.
        - Clearly state the level of certainty and note limitations if the exact location remains ambiguous.

        # Object‑focus strategy:
        - After choosing a candidate, rewrite `active_question` so it explicitly names that object (e.g., *"Locate the abandoned Jesuit mission of Exaltación"*).
        - Accumulate clues step by step and merge them into the aggregated `supporting_evidence` object (e.g., add `details.clues`).
        - Switch to a different candidate **only** if new evidence contradicts the current hypothesis or proves the object irrelevant.

        # Instructions for entity_search & metadata:
        - `entity_search` is the **first‑line, low‑cost tool** to see whether a specific **named entity** is actually present in our corpus metadata (settlements, rivers, missions, explorers, tribes, years).  
        - Always send a **comma‑separated list of 1‑5 concise names** — *never* descriptive phrases.  Good examples:  
          · `Villarico`, `River Moraji`, `Jesuit mission`  
          · `Aguirre`, `Mission Exaltación`, `Santo Tomé`  
        - Use it aggressively to narrow candidates **before** investing in additional `vector_search`.  Typical loop:
            1. Extract 2‑5 plausible names or spellings from the current clue (older texts may vary orthography).  
            2. Run `entity_search`; note hit counts.  
            3. Keep the top‑hit names and discard zero‑hit variants.
        - A rare entity may still be correct — cross‑check with indirect clues (distances, rivers, landmarks).
        - Do **not** query phrases like "abandoned village across river"; they yield zero hits.

        # Guidance for web_search:
        - Use web_search **only after** you have at least **one concrete name** *and* a regional pointer (e.g., country, province, nearby river).  
        - Form the query as `<name> <region or river>` — e.g., `"Villarico" "Rio Moraji" Brazil`.  
        - Avoid generic or overly broad phrases; they add noise and rarely help locate forgotten sites from 17‑18th‑century texts.

        # Variant‑spelling strategy:
        - Colonial texts often vary orthography (`Muctira`, `Muctirá`, `Muctirae`; `Xiruma` vs. `Jiruma`).  
        - Before launching a new `vector_search`, consider running `entity_search` with **2‑4 plausible variants** of the same name.  
        - Pick the variant with the highest hit count, then proceed.

        # Handling early‑modern distance clues:
        - When you encounter units like *league/legua*, *jornada*, or vague travel times ("two days by canoe"), convert them to approximate kilometres (1 Spanish league ≈ 5.6 km).  
        - Add these conversions to `supporting_evidence.meta.distances` and use them to bound the search area.

        # Instructions for supporting_evidence:
        - **Condense all evidence into exactly one object** and place it as the sole element of the `supporting_evidence` array.  
        - Aggregate multiple facts in this object's `details`/`meta` (e.g., `meta.sources`).  
        - Keys required: `source`, `value`, `details`, `meta`.  Details **must** be a dictionary, not a string.  
        - Add new facts to this same object; remove only if proven irrelevant.

        # Instructions for search‑query reformulation (embedding search):
        - Generate **exactly one** `vector_search` action per reasoning step.  
        - The `query` string **must** be a single phrase of **3–6 meaningful words** (≈5–7 tokens).  
        - Avoid lists, commas, Boolean operators, meta‑language.  
        - Use a new reasoning step for any substantially different formulation.

        # Anti‑stall rules:
        - Track your last actions (see *Recent actions* above).  
        - **Do not** issue a `vector_search` whose query is semantically similar to any query already tried.  
        - If two consecutive `vector_search` attempts each returned fewer than **3** new evidence items, your **next** action must be either `entity_search`, `web_search`, or `general_knowledge` instead of another `vector_search`.  
        - Aim to collect evidence from **at least two different action types** before finalizing.

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



