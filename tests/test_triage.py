import os
# Force Mock Mode for testing to prevent slow timeouts on offline API requests
os.environ["GROQ_API_KEY"] = "mock"

import sys
import pytest
from fastapi.testclient import TestClient

# Ensure app is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.models.schemas import UrgencyLevel

client = TestClient(app)

def test_health_check():
    """
    Verifies that the /health endpoint is active and returns status.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "mock_mode" in data
    assert "model" in data

def test_triage_critical_emergency():
    """
    Tests that chest pain triggers a rule-based Emergency triage response.
    """
    payload = {
        "patient_id": "pat_critical_999",
        "message": "I have severe crushing chest pain radiating to my neck and left arm. It started suddenly and I feel dizzy."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["patient_id"] == "pat_critical_999"
    assert data["urgency"] == UrgencyLevel.EMERGENCY.value
    
    # Flexible check for cardiac conditions in description
    condition_lower = data["condition"].lower()
    assert any(term in condition_lower for term in ["cardiac", "myocardial", "infarction", "angina", "heart", "unspecified"])
    assert len(data["red_flags"]) > 0
    assert data["confidence"] >= 0.5
    assert any(term in data["disclaimer"].lower() for term in ["911", "er", "emergency", "doctor", "medical"])

def test_triage_urgent_care():
    """
    Tests that localized pain and high fever triggers an Urgent or Emergency triage response.
    """
    payload = {
        "patient_id": "pat_urgent_777",
        "message": "My stomach is hurting severely in the lower right area and I have a high fever of 103F. I cannot keep any fluids down."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["patient_id"] == "pat_urgent_777"
    # Urgent stomach pain can be categorized as Urgent or Emergency depending on live LLM evaluation
    assert data["urgency"] in [UrgencyLevel.URGENT.value, UrgencyLevel.EMERGENCY.value]
    assert len(data["red_flags"]) > 0
    assert 0.0 <= data["confidence"] <= 1.0

def test_triage_self_care():
    """
    Tests that mild muscle soreness triggers a Self-Care or Non-Urgent triage response.
    """
    payload = {
        "patient_id": "pat_selfcare_111",
        "message": "My calves and thighs are a bit sore and tired after completing a 5k run yesterday. No other issues."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["patient_id"] == "pat_selfcare_111"
    # Standard sore muscles is Self-Care or Non-Urgent
    assert data["urgency"] in [UrgencyLevel.SELF_CARE.value, UrgencyLevel.NON_URGENT.value]
    assert len(data["red_flags"]) > 0
    assert any(term in data["disclaimer"].lower() for term in ["home", "rest", "monitor", "healthcare", "provider", "doctor"])

def test_validation_error_invalid_payload():
    """
    Verifies that the custom validation handler intercepts invalid field values.
    """
    # message field is missing
    payload = {"patient_id": "pat_unknown"}
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 422
    
    data = response.json()
    assert data["error"] == "Unprocessable Entity"
    assert "details" in data

def test_batch_triage(monkeypatch):
    """
    Verifies the rule-first batch triage pipeline:
    - Rule-based matches (chest pain, paper cut) are bypassed.
    - Standard symptoms fall back to the LLM.
    - Aggregates counters and outputs proper metrics structure.
    """
    mock_cases = [
        {"patient_id": "case_rule_emergency", "message": "I have crushing chest pain that radiates to my left arm."},
        {"patient_id": "case_rule_selfcare", "message": "I got a small paper cut on my thumb."},
        {"patient_id": "case_llm_urgent", "message": "My stomach is hurting severely in the lower right area and I have a high fever of 103F."}
    ]
    
    # Mock CasesService.fetch_cases to return mock_cases
    from app.services.cases import CasesService
    async def mock_fetch_cases(self):
        return mock_cases
        
    monkeypatch.setattr(CasesService, "fetch_cases", mock_fetch_cases)
    
    response = client.post("/api/v1/batch-triage")
    assert response.status_code == 200
    
    data = response.json()
    assert data["total_cases"] == 3
    assert data["processed_cases"] == 3
    assert len(data["results"]) == 3
    
    # Verify patient_id matches
    ids = [res["patient_id"] for res in data["results"]]
    assert "case_rule_emergency" in ids
    assert "case_rule_selfcare" in ids
    assert "case_llm_urgent" in ids
    
    # Verify metrics structure and basic counts
    metrics = data["metrics"]
    assert metrics["total_cases"] == 3
    assert metrics["emergency_count"] >= 1  # case_rule_emergency matches emergency
    assert metrics["self_care_count"] >= 1  # case_rule_selfcare matches self-care
    assert metrics["urgent_count"] + metrics["emergency_count"] >= 2

def test_batch_triage_rate_limiting(monkeypatch):
    """
    Verifies that when BATCH_MAX_LLM_CASES is set, the LLM calls are limited,
    fallback responses are returned after the limit is reached, and correct metrics are returned.
    """
    # 3 cases:
    # 1. "I have crushing chest pain..." -> Rule engine (emergency rule, bypasses LLM)
    # 2. "I feel a bit dizzy" -> Needs LLM (first LLM call, budget used = 1)
    # 3. "My throat is ticklish" -> Needs LLM (budget exhausted fallback)
    mock_cases = [
        {"patient_id": "case_rule", "message": "I have crushing chest pain that radiates to my left arm."},
        {"patient_id": "case_llm_1", "message": "I feel a bit dizzy and tired."},
        {"patient_id": "case_fallback", "message": "I feel a strange buzzing in my elbow when I snap my fingers."}
    ]
    
    from app.services.cases import CasesService
    async def mock_fetch_cases(self):
        return mock_cases
        
    monkeypatch.setattr(CasesService, "fetch_cases", mock_fetch_cases)
    monkeypatch.setenv("BATCH_MAX_LLM_CASES", "1")
    
    response = client.post("/api/v1/batch-triage")
    assert response.status_code == 200
    
    data = response.json()
    assert data["total_cases"] == 3
    assert data["processed_cases"] == 3
    assert len(data["results"]) == 3
    
    # Verify new metrics
    assert data["groq_calls_used"] == 2
    assert data["groq_calls_saved"] == 4
    assert data["llm_budget_exhausted"] is True
    
    # Check individual results
    results_map = {res["patient_id"]: res for res in data["results"]}
    
    # Rule case should bypass LLM and succeed
    assert results_map["case_rule"]["urgency"] == UrgencyLevel.EMERGENCY.value
    assert results_map["case_rule"]["confidence"] >= 0.8
    
    # LLM case (should be processed by LLM/MockLLM, not fallback)
    assert results_map["case_llm_1"]["urgency"] in [UrgencyLevel.SELF_CARE.value, UrgencyLevel.NON_URGENT.value]
    
    # Fallback case (budget exhausted)
    fb = results_map["case_fallback"]
    assert fb["urgency"] == UrgencyLevel.URGENT.value
    assert fb["condition"] == "Needs Clinical Review"
    assert fb["red_flags"] == []
    assert fb["confidence"] == 0.50
    assert "screening" in fb["disclaimer"].lower() or "clinical" in fb["disclaimer"].lower()

def test_hybrid_triage_routing():
    """
    Verifies Phase 1 Hybrid Triage Engine routing:
    - Emergency chest pain matches local rules and skips LLM.
    - Severe bleeding matches local rules/indicators and skips LLM.
    - Simple fever matches local rules and skips LLM.
    - Unknown symptom does not match rules and calls the LLM.
    """
    # 1. Emergency chest pain
    payload = {
        "patient_id": "pat_chest_pain",
        "message": "I have crushing chest pain and it radiates to my jaw."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["local_triage_used"] is True
    assert data["urgency"] == UrgencyLevel.EMERGENCY.value

    # 2. Severe bleeding
    payload = {
        "patient_id": "pat_severe_bleeding",
        "message": "I am bleeding heavily from a wound and it won't stop."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["local_triage_used"] is True
    assert data["urgency"] == UrgencyLevel.EMERGENCY.value

    # 3. Simple fever
    payload = {
        "patient_id": "pat_simple_fever",
        "message": "I have a fever of 100F since yesterday."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["local_triage_used"] is True
    assert data["urgency"] in [UrgencyLevel.NON_URGENT.value, UrgencyLevel.URGENT.value]

    # 4. Unknown symptom
    payload = {
        "patient_id": "pat_unknown_symptom",
        "message": "I feel a strange buzzing in my elbow when I snap my fingers."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["local_triage_used"] is False
