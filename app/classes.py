from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from uuid import UUID


class Evidence(BaseModel):
    source: str
    value: str
    details: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}

class Action(BaseModel):
    type: str
    query: str
    reason: Optional[str] = ""

class ReasoningResponse(BaseModel):
    actions: List[Action] = []
    finalize: bool = False
    active_question: str = ""
    hypothesis: Optional[str] = ""
    supporting_evidence: List[Evidence] = []
    confidence: Optional[float] = None

class ReasoningRequest(BaseModel):
    user_query: str
    active_question: str
    context: List[Evidence]
    previous_hypotheses: List[str] = []
    supporting_evidence: List[Evidence] = []   # <-- добавить
    reasoning_log: List[Dict[str, Any]] = []
    iteration: int = 0


# --- Request/Response schemas ---
class ReformulateRequest(BaseModel):
    user_query: str
    active_question: str
    context: List[Evidence]
    reasoning_log: Optional[List[Dict[str, Any]]] = []

class ReformulateResponse(BaseModel):
    reformulated_question: str
    alternatives: List[str] = []
    reason: str = ""





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