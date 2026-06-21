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

@pytest.fixture(autouse=True)
def clear_cache():
    from app.services.cache import cache_service
    cache_service.clear()


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
        "message": "I have crushing chest pain and it radiates to my jaw. I am also sweating a lot."
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

def test_rag_retrieval_and_metrics(monkeypatch):
    """
    Verifies that standard LLM triage pathways (which do not match local rules)
    correctly trigger RAG retrieval, populate sources, document names, scores,
    and return retrieval performance latency metrics.
    """
    payload = {
        "patient_id": "pat_rag_test",
        "message": "I feel a strange buzzing in my elbow when I snap my fingers."
    }
    
    # Set configurable retrieval top_k limit
    monkeypatch.setenv("RAG_TOP_K", "2")
    
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["patient_id"] == "pat_rag_test"
    assert data["local_triage_used"] is False
    assert data["rag_used"] is True
    assert data["retrieved_chunks"] == 2
    assert data["retrieval_latency_ms"] >= 0
    
    # Check sources array validation
    sources = data["sources"]
    assert isinstance(sources, list)
    assert len(sources) == 2
    for src in sources:
        assert "source" in src
        assert "page" in src
        assert "document_type" in src
        assert "score" in src
        assert isinstance(src["score"], float)
        assert 0.0 <= src["score"] <= 1.0


def test_rag_integration_local_bypass(monkeypatch):
    """
    Verify that when local rule engine threshold (>= 0.8) is met,
    the retriever is bypassed completely, and metrics default correctly.
    """
    from app.services.rag import FAISSRetriever
    retrieval_called = False
    
    original_retrieve = FAISSRetriever.retrieve
    def mock_retrieve(self, query, top_k=3):
        nonlocal retrieval_called
        retrieval_called = True
        return original_retrieve(self, query, top_k)
        
    monkeypatch.setattr(FAISSRetriever, "retrieve", mock_retrieve)
    
    # Crushing chest pain should trigger local bypass
    payload = {
        "patient_id": "pat_bypass_integration",
        "message": "I have crushing chest pain and it radiates to my jaw. I am also sweating a lot."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["local_triage_used"] is True
    assert retrieval_called is False
    assert data["rag_used"] is False
    assert data["retrieved_chunks"] == 0
    assert data["retrieval_latency_ms"] == 0


def test_live_retriever_filtering(monkeypatch):
    """
    Verify that retrieve() correctly filters out chunks with similarity scores below MIN_RELEVANCE_SCORE (0.25).
    """
    from app.services.rag import FAISSRetriever
    from langchain_core.documents import Document
    
    retriever = FAISSRetriever()
    
    class MockIndex:
        def similarity_search_with_relevance_scores(self, query, k):
            return [
                (Document(page_content="High match context", metadata={"source": "good_guide.pdf", "page": 5, "document_type": "emergency"}), 0.85),
                (Document(page_content="Low match context", metadata={"source": "bad_guide.pdf", "page": 2, "document_type": "chronic"}), 0.12)
            ]
            
    retriever.index = MockIndex()
    monkeypatch.setenv("GROQ_API_KEY", "live_test_key")
    
    res = retriever.retrieve("test query", top_k=2)
    sources = res["sources"]
    
    # Should filter out the one with score 0.12 (below 0.25)
    assert len(sources) == 1
    assert sources[0]["source"] == "good_guide.pdf"
    assert sources[0]["score"] == 0.68
    assert sources[0]["document_type"] == "emergency"
    assert sources[0]["page"] == 5


def test_critical_emergency_bypasses_rag_and_search(monkeypatch):
    """
    Verify that when analyze_node detects emergency keywords and flags is_critical=True,
    the retriever and search nodes are bypassed in LangGraph.
    """
    # Force evaluate_rules to return None to bypass local rule engine
    monkeypatch.setattr("app.graph.triage_graph.evaluate_rules", lambda features: None)
    
    from app.services.rag import FAISSRetriever
    retrieval_called = False
    
    original_retrieve = FAISSRetriever.retrieve
    def mock_retrieve(self, query, top_k=3):
        nonlocal retrieval_called
        retrieval_called = True
        return original_retrieve(self, query, top_k)
        
    monkeypatch.setattr(FAISSRetriever, "retrieve", mock_retrieve)
    
    # "severe bleeding" matches EMERGENCY_KEYWORDS list in triage_graph.py
    payload = {
        "patient_id": "pat_critical_bypass",
        "message": "I got cut and have severe bleeding."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Verify RAG and search nodes were bypassed
    assert data["local_triage_used"] is False
    assert retrieval_called is False
    assert data["rag_used"] is False
    assert data["retrieved_chunks"] == 0
    assert data["retrieval_latency_ms"] == 0


def test_pregnancy_bleeding_safety_escalation():
    """
    1. Verify pregnancy + bleeding is escalated to Emergency
    """
    payload = {
        "patient_id": "pat_preg_bleed",
        "message": "I am pregnant and noticed some bleeding today."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] == UrgencyLevel.EMERGENCY.value
    assert data["condition"] == "Possible obstetric emergency"
    assert data["local_triage_used"] is True


def test_stroke_combination_escalation():
    """
    2. Verify weakness + slurred speech combination matrix rule (Possible stroke)
    """
    payload = {
        "patient_id": "pat_stroke",
        "message": "My arm feels weak and I am having difficulty talking."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] == UrgencyLevel.EMERGENCY.value
    assert "stroke" in data["condition"].lower()
    assert data["local_triage_used"] is True


def test_cardiac_sweating_escalation():
    """
    3. Verify chest pain + sweating combination (Possible MI)
    """
    payload = {
        "patient_id": "pat_cardiac_sweat",
        "message": "I have a tight chest pain and I am sweating a lot."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] == UrgencyLevel.EMERGENCY.value
    assert "infarction" in data["condition"].lower() or "cardiac" in data["condition"].lower()
    assert data["local_triage_used"] is True


def test_meningitis_fever_stiff_neck_escalation():
    """
    4. Verify fever + stiff neck combination (Possible Meningitis)
    """
    payload = {
        "patient_id": "pat_meningitis",
        "message": "I have a high fever and my neck is very stiff."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] == UrgencyLevel.EMERGENCY.value
    assert "meningitis" in data["condition"].lower()
    assert data["local_triage_used"] is True


def test_sepsis_fever_dehydration_escalation():
    """
    5. Verify fever + severe dehydration combination (Possible Sepsis)
    """
    payload = {
        "patient_id": "pat_sepsis",
        "message": "I have a high fever and feel severely dehydrated."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] == UrgencyLevel.EMERGENCY.value
    assert "sepsis" in data["condition"].lower()
    assert data["local_triage_used"] is True


def test_pe_combination_escalation():
    """
    6. Verify chest pain + dizziness + shortness of breath combination (Possible PE)
    """
    payload = {
        "patient_id": "pat_pe",
        "message": "I have chest pain and dizziness and shortness of breath."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] == UrgencyLevel.EMERGENCY.value
    assert "pulmonary embolism" in data["condition"].lower()
    assert data["local_triage_used"] is True


def test_age_aware_escalations():
    """
    7. Verify age-aware rules:
       - child (age < 5) + fever -> Urgent
       - elderly (age > 65) + confusion -> Emergency
       - child + severe dehydration -> Emergency
    """
    # Child + Fever -> Urgent
    payload1 = {
        "patient_id": "pat_child_fever",
        "message": "My 3-year-old child has a fever."
    }
    response1 = client.post("/api/v1/triage", json=payload1)
    assert response1.status_code == 200
    assert response1.json()["urgency"] == UrgencyLevel.URGENT.value
    assert "fever" in response1.json()["condition"].lower()

    # Elderly + Confusion -> Emergency
    payload2 = {
        "patient_id": "pat_elderly_conf",
        "message": "My 75-year-old grandfather is suddenly confused."
    }
    response2 = client.post("/api/v1/triage", json=payload2)
    assert response2.status_code == 200
    assert response2.json()["urgency"] == UrgencyLevel.EMERGENCY.value
    assert "confusion" in response2.json()["condition"].lower() or "mental status" in response2.json()["condition"].lower()

    # Child + Dehydration -> Emergency
    payload3 = {
        "patient_id": "pat_child_dehydr",
        "message": "The baby is severely dehydrated."
    }
    response3 = client.post("/api/v1/triage", json=payload3)
    assert response3.status_code == 200
    assert response3.json()["urgency"] == UrgencyLevel.EMERGENCY.value
    assert "dehydration" in response3.json()["condition"].lower()


def test_clinical_validator_confidence_guardrail(monkeypatch):
    """
    8. Verify that if final LLM confidence < 0.50 and urgency is low,
       the validator escalates to Urgent / Requires Physician Review.
    """
    # Bypass local rule engine
    monkeypatch.setattr("app.graph.triage_graph.evaluate_rules", lambda features: None)
    
    from app.services.llm import MockStructuredRunnable
    from app.models.schemas import TriageAssessment, UrgencyLevel
    
    original_invoke = MockStructuredRunnable.invoke
    def mock_invoke(self, input_messages, config=None):
        if self.schema.__name__ == "TriageAssessment":
            return TriageAssessment(
                urgency=UrgencyLevel.NON_URGENT,
                condition="Mild Headache",
                red_flags=[],
                confidence=0.40,
                disclaimer="Monitor at home."
            )
        return original_invoke(self, input_messages, config)
        
    monkeypatch.setattr(MockStructuredRunnable, "invoke", mock_invoke)
    
    payload = {
        "patient_id": "pat_low_conf",
        "message": "I have a mild headache."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] == UrgencyLevel.URGENT.value
    assert data["condition"] == "Requires Physician Review"


def test_gi_bleed_vomiting_blood():
    """
    9. Verify vomiting blood triggers GI Bleed escalation (Emergency)
    """
    payload = {
        "patient_id": "pat_gi_bleed",
        "message": "I started vomiting blood."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] == UrgencyLevel.EMERGENCY.value
    assert "gastrointestinal" in data["condition"].lower() or "gi" in data["condition"].lower()
    assert data["local_triage_used"] is True


def test_uti_synonym_normalization():
    """
    10. Verify synonym normalization works: "pain while peeing" -> dysuria.
    """
    from app.services.nlp_processor import extract_features
    features = extract_features("I have pain while peeing")
    assert "dysuria" in features.symptoms


def test_chest_pain_alone_is_urgent():
    """
    11. Verify that chest pain alone is classified as Urgent, not Emergency (Fix 1).
    """
    payload = {
        "patient_id": "pat_cp_alone",
        "message": "I have some chest pain."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] == UrgencyLevel.URGENT.value


def test_oxygen_respiratory_risks():
    """
    12. Verify shortness of breath + blue lips/confusion triggers Emergency (Fix 2).
    """
    # Blue lips / cyanosis
    payload1 = {
        "patient_id": "pat_resp_cyanosis",
        "message": "I have shortness of breath and my skin has cyanosis with blue lips."
    }
    response1 = client.post("/api/v1/triage", json=payload1)
    assert response1.status_code == 200
    assert response1.json()["urgency"] == UrgencyLevel.EMERGENCY.value
    assert "hypoxia" in response1.json()["condition"].lower() or "respiratory" in response1.json()["condition"].lower()

    # Confusion
    payload2 = {
        "patient_id": "pat_resp_conf",
        "message": "I have shortness of breath and feel very confused."
    }
    response2 = client.post("/api/v1/triage", json=payload2)
    assert response2.status_code == 200
    assert response2.json()["urgency"] == UrgencyLevel.EMERGENCY.value


def test_escalation_reason_and_disclaimer():
    """
    13. Verify escalation_reason is populated and legal disclaimer is prepended (Fix 5 & Fix 6).
    """
    payload = {
        "patient_id": "pat_disclaimer_test",
        "message": "I am pregnant and have spotting."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["escalation_reason"] is not None
    assert "Pregnancy" in data["escalation_reason"] or "rule" in data["escalation_reason"]
    assert data["disclaimer"].startswith("This system is an AI-assisted symptom triage support tool and not a medical diagnosis system.")


def test_noncritical_chest_pain_not_bypassed(monkeypatch):
    """
    14. Verify that non-critical chest pain does not trigger the emergency keyword bypass
    and proceeds through the normal pipeline (RAG & search are not bypassed).
    """
    from app.services.rag import FAISSRetriever
    retrieval_called = False
    original_retrieve = FAISSRetriever.retrieve
    def mock_retrieve(self, query, top_k=3):
        nonlocal retrieval_called
        retrieval_called = True
        return original_retrieve(self, query, top_k)
    monkeypatch.setattr(FAISSRetriever, "retrieve", mock_retrieve)

    payload = {
        "patient_id": "pat_noncrit_cp",
        "message": "I have mild chest pain after gym"
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["local_triage_used"] is False
    assert retrieval_called is True
    assert data["rag_used"] is True
    assert data["urgency"] == UrgencyLevel.URGENT.value
    assert data["condition"] == "Chest Pain Requiring Medical Evaluation"


def test_seizure_emergency_bypass(monkeypatch):
    """
    15. Verify that true immediate critical keyword 'seizure' triggers the emergency bypass.
    """
    # Bypass local rule engine so we route to analyze_node
    monkeypatch.setattr("app.graph.triage_graph.evaluate_rules", lambda features: None)
    
    from app.services.rag import FAISSRetriever
    retrieval_called = False
    original_retrieve = FAISSRetriever.retrieve
    def mock_retrieve(self, query, top_k=3):
        nonlocal retrieval_called
        retrieval_called = True
        return original_retrieve(self, query, top_k)
    monkeypatch.setattr(FAISSRetriever, "retrieve", mock_retrieve)

    payload = {
        "patient_id": "pat_seizure_bypass",
        "message": "I had a seizure and lost consciousness."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["local_triage_used"] is False
    assert retrieval_called is False
    assert data["rag_used"] is False
    assert data["urgency"] == UrgencyLevel.EMERGENCY.value


def test_llm_disclaimer_enforced(monkeypatch):
    """
    16. Ensure all LLM triage responses contain the legal disclaimer prefix.
    """
    monkeypatch.setattr("app.graph.triage_graph.evaluate_rules", lambda features: None)
    
    payload = {
        "patient_id": "pat_disclaimer_triage_node",
        "message": "I feel a strange buzzing in my elbow when I snap my fingers."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    expected_prefix = (
        "This system is an AI-assisted symptom triage support tool and not a medical diagnosis system. "
        "It cannot replace physician evaluation, emergency services, or clinical judgment."
    )
    assert data["disclaimer"].startswith(expected_prefix)


def test_low_risk_chest_pain_label():
    """
    17. Verify that chest pain without high-risk symptoms sets the condition to neutral label.
    """
    payload = {
        "patient_id": "pat_low_risk_cp_label",
        "message": "I have mild chest pain"
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["urgency"] == UrgencyLevel.URGENT.value
    assert data["condition"] == "Chest Pain Requiring Medical Evaluation"


def test_priority_override():
    """
    18. Verify that when multiple safety rules trigger in the validator node, the highest priority condition is retained.
    """
    payload = {
        "patient_id": "pat_multi_risk_priority",
        "message": "I have a high fever and feel confused, and I also have sudden chest pain and shortness of breath."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # PE (95) > Sepsis (90), so PE condition and urgency are kept
    assert data["urgency"] == UrgencyLevel.EMERGENCY.value
    assert data["condition"] == "Possible pulmonary embolism"


def test_stroke_escalation_reason(monkeypatch):
    """
    19. Verify that a stroke emergency keyword bypass in analyze_node sets the correct escalation_reason.
    """
    # Bypass local rule engine to force routing to analyze_node
    monkeypatch.setattr("app.graph.triage_graph.evaluate_rules", lambda features: None)
    
    payload = {
        "patient_id": "pat_stroke_reason",
        "message": "My face is drooping and I have slurred speech"
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["urgency"] == UrgencyLevel.EMERGENCY.value
    assert data["local_triage_used"] is False
    assert data["escalation_reason"] is not None
    assert "Emergency keyword trigger: slurred speech" in data["escalation_reason"]


def test_red_flags_deduplication(monkeypatch):
    """
    20. Verify that duplicate red flags (including semantic duplicates like 'diaphoresis' references)
    are successfully normalized and deduplicated by the validator node.
    """
    # Bypass local rule engine to force routing to LLM and validator
    monkeypatch.setattr("app.graph.triage_graph.evaluate_rules", lambda features: None)
    
    # Mock LLM to return a response with 'Profuse sweating (diaphoresis)' in red flags
    from app.services.llm import MockStructuredRunnable
    from app.models.schemas import TriageAssessment, UrgencyLevel
    
    original_invoke = MockStructuredRunnable.invoke
    def mock_invoke(self, input_messages, config=None):
        if self.schema.__name__ == "TriageAssessment":
            return TriageAssessment(
                urgency=UrgencyLevel.URGENT,
                condition="Chest Pain Requiring Medical Evaluation",
                red_flags=[
                    "Pain radiating to left arm, neck, or jaw",
                    "Profuse sweating (diaphoresis)"
                ],
                confidence=0.85,
                disclaimer="Seek urgent care."
            )
        return original_invoke(self, input_messages, config)
        
    monkeypatch.setattr(MockStructuredRunnable, "invoke", mock_invoke)
    
    payload = {
        "patient_id": "pat_dedupe_rf",
        "message": "I have some mild chest pain"
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Validator appends ["Pain radiating to left arm, neck, or jaw", "Profuse sweating", "Shortness of breath"]
    # So raw list has:
    # 1. Pain radiating to left arm, neck, or jaw (from LLM)
    # 2. Profuse sweating (diaphoresis) (from LLM)
    # 3. Pain radiating to left arm, neck, or jaw (from low-risk CP rule)
    # 4. Profuse sweating (from low-risk CP rule)
    # 5. Shortness of breath (from low-risk CP rule)
    # After normalization and deduplication, we should get exactly 3 items:
    # - "Pain radiating to left arm, neck, or jaw"
    # - "Profuse sweating (diaphoresis)"
    # - "Shortness of breath"
    
    red_flags = data["red_flags"]
    assert len(red_flags) == 3
    # Check that they match the expected deduplicated elements
    assert "Pain radiating to left arm, neck, or jaw" in red_flags
    assert "Profuse sweating (diaphoresis)" in red_flags
    assert "Shortness of breath" in red_flags
    assert "Profuse sweating" not in red_flags  # Sweating without diaphoresis was ignored because the normalized keys matched


@pytest.mark.anyio
async def test_validator_node_confidence_calibration():
    """
    21. Verify that the clinical confidence calibration formula is calculated correctly
    inside the validator node based on rule, RAG, LLM, and validator reliability weights.
    """
    from app.graph.triage_graph import validator_node
    from app.models.schemas import TriageResponse, UrgencyLevel
    
    # Mock triage response to pass into validator
    triage_response = TriageResponse(
        patient_id="pat_test_calib",
        urgency=UrgencyLevel.URGENT,
        condition="Acute Inflammatory Process",
        red_flags=["Fever"],
        confidence=0.80, # LLM confidence
        disclaimer="Standard disclaimer"
    )
    
    state = {
        "patient_id": "pat_test_calib",
        "message": "I have a high fever and stomach pain",
        "triage_response": triage_response,
        "retrieved_chunks": 3,
        "errors": []
    }
    
    res = await validator_node(state)
    calibrated_response = res["triage_response"]
    
    # Trace calculation:
    # - fever + abdominal pain (stomach pain synonyms map to abdominal pain in nlp_processor)
    # - rule_res matches (e.g. abdominal pain rule with 0.85 confidence)
    #   rule_conf = 0.85
    # - retrieved_chunks is 3
    #   rag_conf = 3 / 3 = 1.0
    # - llm_conf = 0.80
    # - override_applied = True (due to combination rule matching abdominal pain + fever setting override_applied)
    #   validator_conf = 0.98
    # Calibrated confidence formula:
    # final = 0.35 * 0.85 + 0.25 * 1.0 + 0.25 * 0.80 + 0.15 * 0.75
    # final = 0.2975 + 0.25 + 0.20 + 0.1125 = 0.86
    
    assert calibrated_response.confidence == 0.86






