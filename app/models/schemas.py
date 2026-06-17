from enum import Enum
from typing import Any, List, Optional, TypedDict
from pydantic import BaseModel, Field, field_validator

class UrgencyLevel(str, Enum):
    EMERGENCY = "Emergency"
    CRITICAL = "Critical"
    URGENT = "Urgent"
    NON_URGENT = "Non-Urgent"
    SELF_CARE = "Self-Care"

class TriageRequest(BaseModel):
    patient_id: str = Field(
        ...,
        description="Unique identifier for the patient.",
        examples=["pat_12345"]
    )
    message: str = Field(
        ..., 
        description="The symptom description or message provided by the patient.",
        examples=["I have a severe headache and stiff neck that started suddenly."]
    )

class TriageAssessment(BaseModel):
    urgency: UrgencyLevel = Field(
        ..., 
        description="Assessed urgency level for the patient's symptoms."
    )
    condition: str = Field(
        ..., 
        description="The primary suspected condition or diagnostic category to discuss with a doctor."
    )
    red_flags: List[str] = Field(
        default_factory=list, 
        description="Key warning signs/symptoms that would elevate this to an emergency."
    )
    confidence: float = Field(
        ..., 
        description="Confidence score in the triage assessment, between 0.0 and 1.0."
    )
    disclaimer: str = Field(
        ..., 
        description="Standard medical disclaimer clarifying that this is an AI assistant, not a doctor, and outlining safety precautions."
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0 inclusive.")
        return round(v, 2)

class TriageResponse(TriageAssessment):
    patient_id: str = Field(
        ...,
        description="Unique identifier for the patient matching the request."
    )
    local_triage_used: Optional[bool] = None


class SymptomAnalysis(BaseModel):
    is_critical: bool = Field(
        ...,
        description="True if the symptoms suggest an immediate life-threatening emergency."
    )
    search_query: str = Field(
        ...,
        description="A 2-4 keyword search query optimized for looking up medical databases."
    )

class TriageState(TypedDict):
    """
    State representing the context carried through the LangGraph workflow nodes.
    """
    patient_id: str
    message: str
    is_critical: Optional[bool]
    search_query: Optional[str]
    search_results: Optional[str]
    triage_response: Optional[TriageResponse]
    errors: List[str]
    features: Optional[Any]
    local_triage_used: Optional[bool]


class ProcessingMetrics(BaseModel):
    total_cases: int
    emergency_count: int = Field(..., alias="emergency_count")
    urgent_count: int = Field(..., alias="urgent_count")
    non_urgent_count: int = Field(..., alias="non_urgent_count")
    self_care_count: int = Field(..., alias="self_care_count")

class BatchTriageResponse(BaseModel):
    total_cases: int
    processed_cases: int
    results: List[TriageResponse]
    metrics: ProcessingMetrics
    groq_calls_used: Optional[int] = None
    groq_calls_saved: Optional[int] = None
    llm_budget_exhausted: Optional[bool] = None
    local_triage_used: Optional[int] = None
    llm_triage_used: Optional[int] = None
    local_triage_saved_calls: Optional[int] = None


