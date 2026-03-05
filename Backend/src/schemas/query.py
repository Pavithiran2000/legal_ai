"""Query-related Pydantic schemas matching finetuned model output."""
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


# ── Input ────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=2000, description="User's legal scenario or question")
    top_k: Optional[int] = Field(default=None, description="Number of context chunks to retrieve")
    temperature: Optional[float] = Field(default=None, description="LLM temperature")


class FeedbackRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


# ── Finetuned Model Output Schema ───────────────────────────────

class PrimaryViolation(BaseModel):
    violation_type: str = ""
    act_name: str = ""
    act_year: Any = ""
    act_section_number: str = ""
    act_section_text: str = ""
    why_relevant: str = ""


class SupportingCase(BaseModel):
    case_name: str = ""
    case_year: Any = ""
    case_citation: str = ""
    case_summary: str = ""
    why_relevant: str = ""


class OutputSummary(BaseModel):
    primary_issue: str = ""
    violation_count: int = 0
    acts_count: int = 0
    cases_count: int = 0


class LegalOutput(BaseModel):
    """Matches the finetuned model's JSON output schema exactly."""
    out_of_scope: bool = False
    scope_category: str = "labour_employment_law"
    summary: OutputSummary = Field(default_factory=OutputSummary)
    primary_violations: List[PrimaryViolation] = Field(default_factory=list)
    supporting_cases: List[SupportingCase] = Field(default_factory=list)
    legal_reasoning: str = ""
    recommended_action: Any = Field(default_factory=list)  # list[str] or str
    limits: Any = Field(default_factory=list)  # list[str] or str
    confidence: float = 0.0


# ── API Response ─────────────────────────────────────────────────

class QueryResponse(BaseModel):
    success: bool = True
    query_id: str = ""
    query: str = ""
    recommendation: LegalOutput = Field(default_factory=LegalOutput)
    model_used: str = ""
    generation_time_ms: int = 0
    context_chunks_used: int = 0
    timestamp: str = ""


class OutOfScopeResponse(BaseModel):
    success: bool = True
    query_id: str = ""
    out_of_scope: bool = True
    scope_category: str = ""
    message: str = ""
    confidence: float = 0.0


class QueryHistoryItem(BaseModel):
    id: str
    query_text: str
    out_of_scope: bool = False
    scope_category: Optional[str] = None
    confidence: Optional[float] = None
    model_used: Optional[str] = None
    created_at: Optional[datetime] = None


class QueryDetailResponse(BaseModel):
    id: str
    query_text: str
    response_json: Optional[dict] = None
    out_of_scope: bool = False
    scope_category: Optional[str] = None
    confidence: Optional[float] = None
    model_used: Optional[str] = None
    generation_time_ms: Optional[int] = None
    feedback_rating: Optional[int] = None
    feedback_comment: Optional[str] = None
    created_at: Optional[datetime] = None
