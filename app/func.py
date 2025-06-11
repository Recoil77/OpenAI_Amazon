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

    agent_thoughts_block = data.agent_thoughts.strip() if getattr(data, "agent_thoughts", None) else "[none]"

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
        - entity_hybrid   → returns detailed fragments from internal documents (summary, facts, year, etc.) on specified entities.
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

        # Agent's internal reasoning (carry this over every step, reformulate and expand as needed):
        {agent_thoughts_block}

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

        # Instructions for internal reasoning:
        - Continuously read and update the `agent_thoughts` field as your internal monologue. Write down your current reasoning, doubts, partial insights, evolving hypotheses, and strategies in free-form text.
        - Each step, reformulate and expand this monologue with new information and analysis, building upon all prior thoughts.
        - Treat `agent_thoughts` as your personal memory and ongoing thought process. Use it to help yourself (the agent) reason more deeply, keep track of unresolved questions, and avoid repeating mistakes.
        - Do not simply restate logs or copy hypotheses; instead, explain your actual thinking and planning for the next steps.
        - Always carry over and expand your previous agent_thoughts. Do not reset or clear this field unless the content is clearly obsolete or you are explicitly instructed to summarize and condense.
        - Keep this field concise, but rich in context and insight. Summarize earlier content only if it becomes too lengthy or redundant.

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

        # Instructions for entity_search & metadata:
        - Same as entity_search but pick only one variant then proceed.  
                
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
        - **Condense all supporting evidence into exactly one object**, placed as the sole element of the `supporting_evidence` array.
        - This object must **accumulate all evidence found across all reasoning steps so far** — never reset or clear this evidence between steps.
        - **Always** carry over all previously found evidence into the current step, adding new facts and clues as they are discovered.
        - Only remove or revise evidence if you have found direct proof that it is incorrect or irrelevant; otherwise, keep all prior evidence.
        - Aggregate multiple facts, clues, distances, dates, and contextual details in the `details` and `meta` fields of this object (e.g., `meta.sources`, `details.clues`, `meta.distances`, etc.).
        - The required keys for this object are: `source`, `value`, `details`, `meta`. `details` **must** be a dictionary, not a string.
        - Your hypotheses, actions, and reasoning should **always reference the full, cumulative supporting_evidence**.
        - If no new evidence is found in this step, **return all previously accumulated evidence unchanged**.


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
            {{"type": "vector_search"|"web_search"|"entity_search"|"entity_hybrid"|"general_knowledge", "query": "<query>", "reason": "<why>"}}
          ],
          "agent_thoughts": "<update and expand your internal reasoning here — free-form text, in the agent's own words>",
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


def build_reasoning_prompt_v2(data: "ReasoningRequest") -> str:
    """Return an improved reasoning‑step prompt.

    Key **changes** vs the original `build_reasoning_prompt`:
    1. **Removed duplicates / typos** (second "Instructions for entity_search & metadata" block now correctly refers to *entity_hybrid*).
    2. **Unified formatting** – long blocks are collapsed behind `---` separators so the prompt stays readable yet dense.
    3. **Explicit schema** – the required/optional JSON keys are spelled out and their expected types indicated, including the previously missing
       `previous_hypotheses` and `confidence ∈ [0,1]` (float).
    4. **Stronger safety rails** – added a deterministic *fail‑safe* asking the model to output `{"actions":[],"finalize":false}` when unsure.
    5. **Token discipline** – reminds the model to keep every `query` ≤ 6 words and to cap `agent_thoughts` at ~120 tokens.
    6. **Minor tweaks** – clarified anti‑stall rule and emphasised reuse of `supporting_evidence` in every step.

    This function is drop‑in compatible with the rest of the agent – only the name changed.  You can either rename the old one or re‑export
    this as `build_reasoning_prompt`.
    """

    # ── Helper blocks ────────────────────────────────────────────────────────────
    facts_block = "\n".join(
        f"- [{ev.source}] {ev.value}" for ev in (data.context or [])[:8]
    ) or "[none yet]"

    log_block = "\n".join(
        f"{item.get('iteration')}. [{item.get('action')}] {item.get('query')} (hits:{item.get('result_count')})"
        for item in (data.reasoning_log or [])
    ) or "[no previous actions]"

    hypotheses_block = "\n".join(data.previous_hypotheses) if data.previous_hypotheses else "[none]"

    agent_thoughts_block = (
        data.agent_thoughts.strip() if getattr(data, "agent_thoughts", None) else "[none]"
    )

    preferred_order = (
        "1. vector_search  – internal archive (main)."  # noqa: E501
        "2. entity_search – stats / spelling variants."
        "3. entity_hybrid  – deep dive on a chosen entity (costly)."
        "4. general_knowledge or web_search  – external check (lowest priority)."
        "5. finalize        – only when hypothesis is well‑supported and no new actions help."
    )

    # ── Prompt body ──────────────────────────────────────────────────────────────
    prompt = dedent(
        f"""
        === ROLE ===
        You are **Deep‑Reason** – a chain‑of‑thought engine that solves historical location puzzles by iteratively querying tools and
        accumulating evidence.  Work step‑by‑step, keep internal thoughts in *agent_thoughts*, and reply **strictly** with JSON.
        -----------------------------------------------------------------------------
        # Main user request
        {data.user_query}
        # Focus question
        {data.active_question}
        -----------------------------------------------------------------------------
        ## Context so far (max 8 items)
        {facts_block}
        -----------------------------------------------------------------------------
        ## Recent actions (deduplicated)
        {log_block}
        -----------------------------------------------------------------------------
        ## Previous hypotheses
        {hypotheses_block}
        -----------------------------------------------------------------------------
        ## Agent thoughts (≈ keep ≤120 tokens; extend, never overwrite)
        {agent_thoughts_block}
        -----------------------------------------------------------------------------
        ## Preferred tool order
        {preferred_order}
        -----------------------------------------------------------------------------
        ### Object‑focus policy (abandoned settlements)
        • Lock on one candidate object and gather clues (distances, rivers, neighbours).
        • Update *active_question* to name that object explicitly.
        • Switch only if the hypothesis is falsified.
        -----------------------------------------------------------------------------
        ### Tool instructions
        — *vector_search*: exactly **one** per step, 3‑6 meaningful words.
        — *entity_search*: comma‑separated list (1‑5 names), no descriptive phrases.
        — *entity_hybrid*: use only after entity_search shows hits >0; pass exactly **one** name variant.
        — *web_search* / *general_knowledge*: only after you have a concrete name **and** region.
        -----------------------------------------------------------------------------
        ### Anti‑stall
        • Do **not** repeat a semantically similar vector_search query.
        • After two low‑yield vector_search (<3 new evidence each) you **must** switch tool type.
        • Collect evidence from ≥2 different tool types before *finalize*.
        -----------------------------------------------------------------------------
        ### Evidence accumulation
        • Maintain **one** cumulative object in *supporting_evidence[0]* (keys: source, value, details{{dict}}, meta{{dict}}).
        • Always append, never drop, unless disproven.
        -----------------------------------------------------------------------------
        ### Required JSON schema
        {{
          "actions"           : [{{"type":"vector_search|entity_search|entity_hybrid|web_search|general_knowledge",
                                  "query":"<≤6‑word phrase>",
                                  "reason":"<short justification>"}}],
          "agent_thoughts"    : "<internal chain‑of‑thought>",
          "active_question"   : "<possibly rewritten question>",
          "hypothesis"        : "<current hypothesis or empty>",
          "previous_hypotheses": ["..."]  ,  # append when you discard / supersede one
          "supporting_evidence": [],            # see accumulation rule above
          "confidence"        : null,          # float 0‑1 when confident, else null
          "finalize"          : false
        }}
        • If unsure or no useful action, respond with the same JSON but an empty *actions* list.
        -----------------------------------------------------------------------------
        OUTPUT **ONLY** THE JSON OBJECT – no markdown, no extra text.
        """
    ).strip()

    return prompt

