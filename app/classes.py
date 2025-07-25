from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from uuid import UUID


from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class Evidence(BaseModel):
    source: str = ""
    value: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)
    meta:   Dict[str, Any] = Field(default_factory=dict)

class Action(BaseModel):
    type: str
    query: Optional[str] = None

class ReasoningResponse(BaseModel):
    actions:         List[Action]      = Field(default_factory=list)
    finalize:        bool              = False
    active_question: str               = ""
    hypothesis:      Optional[str]     = ""
    confidence:      Optional[float]   = None
    agent_thoughts:  Optional[str]     = ""
    new_facts:       List[Evidence]    = Field(default_factory=list)

    # ── ключевая строка ─────────────────────────────
    model_config = ConfigDict(extra='ignore')
    # ────────────────────────────────────────────────
class ReasoningRequest(BaseModel):
    user_query: str
    active_question: str
    agent_thoughts: Optional[str]          = ""
    context: List[Evidence]
    previous_hypotheses: List[str] = []
    supporting_evidence: List[Evidence] = []   # <-- добавить
    reasoning_log: List[Dict[str, Any]] = []
    iteration: int = 0
    

class WebSearchRequest(BaseModel):
    query: str
    search_context_size: Literal["low", "medium", "high"] = "low"

class GeneralKnowledgeRequest(BaseModel):
    query: str


class EntitySearchRequest(BaseModel):
    entities: List[str]
    mode: Literal["substring", "exact"] = "substring"


class ChunkSummaryRequest(BaseModel):
    text: str
    max_tokens: Optional[int] = 256  # лимитируем длину summary

class ChunkSummaryResponse(BaseModel):
    summary: str


class VectorSearchRequest(BaseModel):
    query: str
    k: int = 32
    bge_threshold: float = 0.25
    semantic_threshold: float = 0.25


class VectorSearchV2Request(BaseModel):
    query: str
    k: Optional[int] = 10

class VectorSearchV2Result(BaseModel):
    document_id: UUID
    doc_name: str
    year: int
    doc_type: str
    chunk_index: int
    text: str
    score: float  # cosine distance
    metadata_original: Dict
    metadata_translated: Dict


class RerankRequest(BaseModel):
    question: str
    answers: list[str]
    threshold: float = 0.25

class RerankResult(BaseModel):
    index: int
    score: float
    text: str    


typing_list = list
class RerankBlockCandidate(BaseModel):
    block_id: int
    text: str

class RerankSemanticV5Request(BaseModel):
    question: str
    candidates: typing_list[RerankBlockCandidate]
    threshold: float = 0.25

class RerankBlockCandidate(BaseModel):
    block_id: int
    text: str


class RerankMinimalResult(BaseModel):
    block_id: int
    score: float


class ChunkCandidate(BaseModel):
    chunk_id: int
    text: str

class ExtractFactsRequest(BaseModel):
    question: str
    chunks: List[ChunkCandidate]

class ChunkFacts(BaseModel):
    chunk_id: int
    facts: List[str]


class GetVerdictRequest(BaseModel):
    hypothesis: str
    supporting_evidence: Optional[List[Evidence]] = None

class GetVerdictResponse(BaseModel):
    verdict: str
    details: str
    general_knowledge_answer: str
    web_search_answer: str 

class MetadataRequest(BaseModel):
    document_id: str
    year: Optional[int] = None
    doc_type: Optional[str] = None
    text: str

class MetadataResponse(BaseModel):
    document_id: str
    year: Optional[int] = None
    doc_type: Optional[str] = None
    entities: List[str]
    text: str
