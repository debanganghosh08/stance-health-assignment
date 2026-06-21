import os
import logging
from typing import Literal, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from app.models.schemas import TriageState, SymptomAnalysis, TriageAssessment, TriageResponse, UrgencyLevel
from app.prompts.triage_prompt import ANALYSIS_SYSTEM_PROMPT, TRIAGE_SYSTEM_PROMPT
from app.services.llm import get_llm
from app.services.search import get_search_service
from app.services.nlp_processor import extract_features
from app.services.rule_engine import evaluate_rules
from app.services.rag import get_retriever
from app.constants import LEGAL_DISCLAIMER

logger = logging.getLogger("app.graph.triage_graph")

# List of critical symptoms for deterministic rule-based pre-check
EMERGENCY_KEYWORDS = [
    "stroke",
    "slurred speech",
    "seizure",
    "anaphylaxis",
    "unable to breathe",
    "loss of consciousness",
    "severe bleeding"
]

def normalize_red_flags(flags: List[str]) -> List[str]:
    seen = set()
    cleaned = []

    for flag in flags:
        normalized = flag.lower()
        normalized = normalized.replace("(diaphoresis)", "")
        normalized = normalized.strip()

        if normalized not in seen:
            seen.add(normalized)
            cleaned.append(flag)

    return cleaned

async def feature_extraction_node(state: TriageState) -> dict:
    """
    Extracts clinical features from patient message locally.
    """
    import uuid
    trace_id = state.get("trace_id") or str(uuid.uuid4())
    logger.info({"trace_id": trace_id, "node": "feature_extraction_node", "status": "started"})
    message = state.get("message", "")
    features = extract_features(message)
    return {"features": features, "trace_id": trace_id}

async def rule_engine_node(state: TriageState) -> dict:
    """
    Evaluates extracted symptoms using local rules. If confidence meets the threshold,
    bypasses LLM reasoning and populates the response immediately.
    """
    trace_id = state.get("trace_id")
    logger.info({"trace_id": trace_id, "node": "rule_engine_node", "status": "started"})
    features = state.get("features")
    errors = state.get("errors", [])
    patient_id = state.get("patient_id", "unknown")
    
    if not features:
        return {"errors": errors + ["No features extracted before rule engine."]}

    rule_res = evaluate_rules(features)
    threshold = float(os.getenv("LOCAL_TRIAGE_CONFIDENCE_THRESHOLD", "0.80"))

    if rule_res and rule_res.confidence >= threshold:
        logger.info({
            "trace_id": trace_id,
            "node": "rule_engine_node",
            "message": f"⚡ [LOCAL BYPASS] Rule engine match for '{rule_res.matched_rule}' with confidence {rule_res.confidence} >= threshold {threshold}. Skipping LLM."
        })
        
        urg = rule_res.urgency
        if urg == "Emergency":
            disclaimer = f"{LEGAL_DISCLAIMER} CRITICAL WARNING: These symptoms are potentially life-threatening. Seek immediate emergency care by calling 911 or visiting the nearest Emergency Room. Do not wait."
        elif urg == "Urgent":
            disclaimer = f"{LEGAL_DISCLAIMER} This is a screening triage tool, not a clinical diagnosis. You should contact a doctor or visit an urgent care center within 12-24 hours for evaluation."
        elif urg == "Non-Urgent":
            disclaimer = f"{LEGAL_DISCLAIMER} Please schedule an appointment with your primary care provider at your convenience. Seek urgent care if your condition changes or new red flags develop."
        else:
            disclaimer = f"{LEGAL_DISCLAIMER} These symptoms appear mild and manageable at home. Rest, stay hydrated, and monitor. If your symptoms worsen or do not improve in a few days, consult a healthcare professional."

        triage_response = TriageResponse(
            patient_id=patient_id,
            urgency=UrgencyLevel(rule_res.urgency),
            condition=rule_res.condition,
            red_flags=rule_res.red_flags,
            confidence=rule_res.confidence,
            disclaimer=disclaimer,
            local_triage_used=True,
            sources=None,
            rag_used=False,
            retrieval_latency_ms=0,
            retrieved_chunks=0,
            escalation_reason=f"Deterministic rule: {rule_res.matched_rule}"
        )
        return {
            "triage_response": triage_response,
            "local_triage_used": True,
            "errors": errors
        }
    
    logger.info("ℹ️ Local rule engine confidence below threshold. Proceeding to LLM flow.")
    return {"local_triage_used": False}

def local_triage_router(state: TriageState) -> Literal["analyze_node", "__end__"]:
    """
    Routes to END if local triage was successful (high confidence),
    otherwise routes to analyze_node.
    """
    if state.get("local_triage_used") or state.get("triage_response") is not None:
        logger.info("🏁 Local triage used successfully. Routing to END.")
        return "__end__"
    logger.info("🧠 Local triage not used. Routing to LLM Analyze Node.")
    return "analyze_node"

async def analyze_node(state: TriageState) -> dict:
    """
    Analyzes raw patient symptoms to detect critical life-threatening emergencies
    and formulate a relevant search query for the medical database.
    """
    trace_id = state.get("trace_id")
    logger.info({"trace_id": trace_id, "node": "analyze_node", "status": "started"})
    message = state.get("message", "")
    errors = state.get("errors", [])

    # Step 1: Rule-based deterministic emergency check
    message_lower = message.lower()
    detected_keywords = [kw for kw in EMERGENCY_KEYWORDS if kw in message_lower]
    
    if detected_keywords:
        logger.info({
            "trace_id": trace_id,
            "node": "analyze_node",
            "message": f"🚨 Rule-based check: Emergency detected via keywords {detected_keywords}. Bypassing LLM symptom classification."
        })
        return {
            "is_critical": True,
            "search_query": "emergency bypass",
            "escalation_reason": f"Emergency keyword trigger: {', '.join(detected_keywords)}",
            "errors": errors
        }

    # Otherwise, continue to Step 2: normal LLM reasoning
    llm = get_llm()
    
    messages = [
        SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
        HumanMessage(content=f"Analyze the following symptoms:\n{message}")
    ]

    try:
        # Request structured output matching the SymptomAnalysis schema
        llm_structured = llm.with_structured_output(SymptomAnalysis)
        analysis: SymptomAnalysis = await llm_structured.ainvoke(messages)
        
        logger.info({
            "trace_id": trace_id,
            "node": "analyze_node",
            "message": f"Analysis complete. Is critical: {analysis.is_critical}, Search query: '{analysis.search_query}'"
        })
        
        ret_dict = {
            "is_critical": analysis.is_critical,
            "search_query": analysis.search_query,
            "errors": errors
        }
        from app.services.llm import llm_cache_hit_var
        ret_dict["llm_cache_hit"] = llm_cache_hit_var.get()
        return ret_dict
    except Exception as e:
        logger.error(f"❌ Error in analyze_node: {e}", exc_info=True)
        # Fallback to safe default values to ensure robustness
        return {
            "is_critical": False,
            "search_query": message[:50],  # Fallback search query is first 50 chars of symptoms message
            "errors": errors + [f"analyze_node error: {str(e)}"]
        }

async def retrieve_node(state: TriageState) -> dict:
    """
    Queries the local FAISS retriever for medical document context.
    Bypassed on critical emergency.
    """
    trace_id = state.get("trace_id")
    logger.info({"trace_id": trace_id, "node": "retrieve_node", "status": "started"})
    is_critical = state.get("is_critical", False)
    search_query = state.get("search_query", "")
    errors = state.get("errors", [])
    
    if is_critical or not search_query or search_query == "emergency bypass":
        logger.info("🚨 Critical emergency or empty query: Bypassing retrieve_node.")
        return {
            "retrieved_context": "",
            "sources": [],
            "rag_used": False,
            "retrieval_latency_ms": 0,
            "retrieved_chunks": 0,
            "errors": errors
        }

    try:
        retriever = get_retriever()
        top_k = int(os.getenv("RAG_TOP_K", "3"))
        
        # Pass the original message for Level 2 (RAG) caching if supported (for mock compatibility in tests)
        import inspect
        sig = inspect.signature(retriever.retrieve)
        if "message" in sig.parameters:
            res = retriever.retrieve(search_query, top_k=top_k, message=state.get("message"))
        else:
            res = retriever.retrieve(search_query, top_k=top_k)
        
        context = res.get("context", "")
        sources = res.get("sources", [])
        latency = res.get("retrieval_latency_ms", 0)
        
        # Cap context content size to MAX_CONTEXT_CHARS (2000 chars)
        context_capped = context[:2000]
        
        retrieved_chunks = len(sources)
        
        # Check cache hit status
        from app.services.rag import rag_cache_hit_var
        is_hit = rag_cache_hit_var.get()
        
        ret_dict = {
            "retrieved_context": context_capped,
            "sources": sources,
            "rag_used": retrieved_chunks > 0,
            "retrieval_latency_ms": latency,
            "retrieved_chunks": retrieved_chunks,
            "errors": errors
        }
        
        ret_dict["rag_cache_hit"] = is_hit
        return ret_dict
        
    except Exception as e:
        logger.error(f"❌ Error in retrieve_node: {e}", exc_info=True)
        return {
            "retrieved_context": "",
            "sources": [],
            "rag_used": False,
            "retrieval_latency_ms": 0,
            "retrieved_chunks": 0,
            "errors": errors + [f"retrieve_node error: {str(e)}"]
        }


async def search_node(state: TriageState) -> dict:
    """
    Queries the offline medical reference database using the query formulated in the analysis stage.
    """
    trace_id = state.get("trace_id")
    logger.info({"trace_id": trace_id, "node": "search_node", "status": "started"})
    is_critical = state.get("is_critical", False)
    search_query = state.get("search_query", "")
    errors = state.get("errors", [])
    
    if is_critical or not search_query or search_query == "emergency bypass":
        logger.info("🚨 Critical emergency or empty query: Bypassing search_node.")
        return {
            "search_results": "Emergency bypass: Immediate emergency care required.",
            "errors": errors
        }
    
    search_service = get_search_service()
    
    try:
        results = await search_service.search(search_query)
        return {
            "search_results": results,
            "errors": errors
        }
    except Exception as e:
        logger.error(f"❌ Error in search_node: {e}", exc_info=True)
        return {
            "search_results": "Clinical search failure. Reverting to general clinical judgment guidelines.",
            "errors": errors + [f"search_node error: {str(e)}"]
        }

async def triage_node(state: TriageState) -> dict:
    """
    Synthesizes the patient symptoms, medical search context, retrieved context, and emergency status
    to output the final structured triage recommendation.
    """
    trace_id = state.get("trace_id")
    logger.info({"trace_id": trace_id, "node": "triage_node", "status": "started"})
    message = state.get("message", "")
    search_results = state.get("search_results") or "No search context available."
    retrieved_context = state.get("retrieved_context") or "No retrieved context available."
    is_critical = state.get("is_critical", False)
    errors = state.get("errors", [])
    patient_id = state.get("patient_id", "unknown")

    llm = get_llm()
    
    # Cap retrieved context to 2000 characters to prevent prompt context bloat
    retrieved_context_capped = retrieved_context[:2000]
    
    # Prepare comprehensive clinical context for triage
    clinical_context = (
        f"PATIENT REPORTED SYMPTOMS:\n{message}\n\n"
        f"EMERGENCY CLASSIFICATION:\nIs acute emergency suspected? {'YES' if is_critical else 'NO'}\n\n"
        f"RETRIEVED MEDICAL REFERENCES:\n{search_results}\n\n"
        f"MEDICAL CONTEXT:\n{retrieved_context_capped}"
    )

    messages = [
        SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
        HumanMessage(content=clinical_context)
    ]

    try:
        llm_structured = llm.with_structured_output(TriageAssessment)
        assessment: TriageAssessment = await llm_structured.ainvoke(messages)
        
        # Override urgency to Emergency if flagged critical by rule-based precheck or LLM analyze node
        final_urgency = UrgencyLevel.EMERGENCY if is_critical else assessment.urgency

        # Propagate RAG metrics and sources with explicit fallback defaults
        retrieved_chunks = state.get("retrieved_chunks")
        if retrieved_chunks is None:
            retrieved_chunks = 0
        rag_used = retrieved_chunks > 0
        retrieval_latency = state.get("retrieval_latency_ms")
        if retrieval_latency is None:
            retrieval_latency = 0

        final_disclaimer = f"{LEGAL_DISCLAIMER} {assessment.disclaimer}"

        triage_response = TriageResponse(
            patient_id=patient_id,
            urgency=final_urgency,
            condition=assessment.condition,
            red_flags=assessment.red_flags,
            confidence=assessment.confidence,
            disclaimer=final_disclaimer,
            local_triage_used=False,
            sources=state.get("sources"),
            rag_used=rag_used,
            retrieval_latency_ms=retrieval_latency,
            retrieved_chunks=retrieved_chunks,
            escalation_reason=state.get("escalation_reason")
        )
        
        logger.info(f"Triage complete. Level: {triage_response.urgency.value}, Suspected condition: {triage_response.condition}")
        
        ret_dict = {
            "triage_response": triage_response,
            "errors": errors
        }
        from app.services.llm import llm_cache_hit_var
        ret_dict["llm_cache_hit"] = llm_cache_hit_var.get() or state.get("llm_cache_hit", False)
        return ret_dict
    except Exception as e:
        logger.error(f"❌ Error in triage_node: {e}", exc_info=True)
        
        # Propagate RAG metrics and sources with explicit fallback defaults for error state
        retrieved_chunks = state.get("retrieved_chunks")
        if retrieved_chunks is None:
            retrieved_chunks = 0
        rag_used = retrieved_chunks > 0
        retrieval_latency = state.get("retrieval_latency_ms")
        if retrieval_latency is None:
            retrieval_latency = 0

        # Safe fallback logic in case structured generation fails
        fallback = TriageResponse(
            patient_id=patient_id,
            urgency=UrgencyLevel.EMERGENCY if is_critical else UrgencyLevel.URGENT,
            condition="Acute Unspecified Illness",
            red_flags=[
                "Difficulty breathing or swallowing",
                "Severe sudden onset of pain",
                "Confusion, slurred speech, or weakness"
            ],
            confidence=0.50,
            disclaimer=f"{LEGAL_DISCLAIMER} CRITICAL DISCLAIMER: An internal processing error occurred. If you are experiencing severe symptoms, chest pain, or breathing difficulties, please go to the nearest ER or contact emergency services immediately.",
            local_triage_used=False,
            sources=state.get("sources"),
            rag_used=rag_used,
            retrieval_latency_ms=retrieval_latency,
            retrieved_chunks=retrieved_chunks,
            escalation_reason="Fallback due to internal error"
        )
        return {
            "triage_response": fallback,
            "errors": errors + [f"triage_node error: {str(e)}"]
        }

def critical_router(state: TriageState) -> Literal["retrieve_node", "triage_node"]:
    """
    Router determining if we bypass retrieval and search nodes.
    Bypassing on critical symptoms ensures lower latency when immediate intervention is needed.
    """
    if state.get("is_critical", False):
        logger.info("🚨 Critical emergency detected. Bypassing retrieval and search nodes.")
        return "triage_node"
    logger.info("ℹ️ Non-critical symptoms. Routing to retrieve_node.")
    return "retrieve_node"

async def validator_node(state: TriageState) -> dict:
    """
    Validation node executing after triage_node.
    Re-runs feature extraction, detects clinical safety risks, under-triage issues,
    and applies confidence guardrails.
    """
    trace_id = state.get("trace_id")
    logger.info({"trace_id": trace_id, "node": "validator_node", "status": "started"})
    triage_response_orig = state.get("triage_response")
    if not triage_response_orig:
        logger.warning("No triage_response in state. Skipping validation.")
        return {}
    
    triage_response = triage_response_orig.model_copy(deep=True)

    message = state.get("message", "")
    features = extract_features(message)
    msg_lower = message.lower()
    
    from app.services.rule_engine import EMERGENCY_COMBINATION_MATRIX
    
    detected_risks = []
    override_applied = False
    original_urgency = triage_response.urgency.value
    final_urgency = triage_response.urgency.value

    # Helper: urgency hierarchy comparison
    urgency_ranks = {"Self-Care": 1, "Non-Urgent": 2, "Urgent": 3, "Critical": 4, "Emergency": 5}
    
    current_priority = 0

    def apply_override(priority: int, urgency: str, condition: str, red_flags: List[str], risk_name: str):
        nonlocal current_priority, override_applied, final_urgency
        # Append to detected_risks regardless of priority to log that it matched
        if risk_name not in detected_risks:
            detected_risks.append(risk_name)
            
        if priority > current_priority:
            current_priority = priority
            override_applied = True
            if urgency_ranks.get(triage_response.urgency.value, 0) < urgency_ranks.get(urgency, 0):
                final_urgency = urgency
                triage_response.urgency = UrgencyLevel(urgency)
            
            # Always update condition to match highest priority risk
            triage_response.condition = condition
            
            # Merge red flags
            for rf in red_flags:
                if rf not in triage_response.red_flags:
                    triage_response.red_flags.append(rf)

    # 1. Pregnancy Safety Escalation (Task 3)
    if "pregnancy" in features.symptoms:
        pregnancy_triggers = ["bleeding", "abdominal pain", "dizziness", "fainting", "blurred vision", "headache"]
        matched_trigger = None
        for trig in pregnancy_triggers:
            if trig in features.symptoms:
                if trig == "headache" and features.severity not in ["severe", "unbearable"] and "severe" not in msg_lower:
                    continue
                matched_trigger = trig
                break
        if matched_trigger:
            apply_override(
                85,
                "Emergency",
                "Possible obstetric emergency",
                [
                    "Vaginal bleeding or fluid leakage during pregnancy",
                    "Severe abdominal pain or cramping",
                    "Severe headache, visual changes, or blurred vision",
                    "Significant dizziness, fainting, or loss of consciousness"
                ],
                f"Pregnancy + {matched_trigger.capitalize()}"
            )

    # 2. Sepsis Escalation (Task Upgrade 1)
    if "fever" in features.symptoms:
        if "confusion" in features.symptoms:
            apply_override(
                90,
                "Emergency",
                "Possible sepsis",
                ["High fever with altered mental status", "Stiff neck", "Rapid breathing", "Cold or pale extremities"],
                "Sepsis Risk (Fever + Confusion)"
            )
        elif "severe dehydration" in features.symptoms:
            apply_override(
                90,
                "Emergency",
                "Possible sepsis",
                ["High fever with severe dehydration", "Rapid heart rate", "Decreased urination", "Confusion"],
                "Sepsis Risk (Fever + Severe Dehydration)"
            )
        elif "dizziness" in features.symptoms and "weakness" in features.symptoms:
            apply_override(
                90,
                "Emergency",
                "Possible sepsis",
                ["High fever with systemic weakness and dizziness", "Low blood pressure indicators", "Rapid pulse"],
                "Sepsis Risk (Fever + Dizziness + Weakness)"
            )

    # 3. Pulmonary Embolism Escalation (Task Upgrade 2)
    if all(x in features.symptoms for x in ["chest pain", "shortness of breath", "dizziness"]):
        apply_override(
            95,
            "Emergency",
            "Possible pulmonary embolism",
            ["Sharp chest pain", "Sudden shortness of breath", "Dizziness or fainting"],
            "PE Risk (Chest Pain + Dizziness + Shortness of Breath)"
        )
    elif "shortness of breath" in features.symptoms and "chest pain" in features.symptoms:
        apply_override(
            95,
            "Emergency",
            "Possible pulmonary embolism",
            ["Sharp chest pain worsening on breathing", "Sudden shortness of breath", "Coughing up blood"],
            "PE Risk (Shortness of Breath + Chest Pain)"
        )

    # 3.5 Oxygen-related Respiratory Risk (Fix 2)
    if "shortness of breath" in features.symptoms:
        if "blue lips" in features.symptoms:
            apply_override(
                90,
                "Emergency",
                "Potential Hypoxia / Respiratory Emergency",
                ["Shortness of breath with blue/pale lips or skin (cyanosis)", "Low oxygen levels", "Confusion"],
                "Respiratory Risk (Shortness of Breath + Blue Lips)"
            )
        elif "confusion" in features.symptoms:
            apply_override(
                90,
                "Emergency",
                "Potential Hypoxia / Respiratory Emergency",
                ["Shortness of breath with altered mental status", "Lethargy", "Blue/pale lips or skin"],
                "Respiratory Risk (Shortness of Breath + Confusion)"
            )

    # 4. Age-Aware Safety Escalation (Task Upgrade 3)
    is_pediatric = (features.numeric_age is not None and features.numeric_age < 5) or features.age_category in ["child", "infant"]
    is_elderly = (features.numeric_age is not None and features.numeric_age > 65) or features.age_category == "elderly"
    
    if is_pediatric and "fever" in features.symptoms:
        apply_override(
            70,
            "Urgent",
            "Febrile Child / Pediatric Fever",
            ["Temperature > 100.4°F in child < 5 years old", "Lethargy or irritability", "Poor feeding or hydration"],
            "Pediatric Fever Risk"
        )
        
    if is_elderly and "confusion" in features.symptoms:
        apply_override(
            90,
            "Emergency",
            "Altered Mental Status in Elderly",
            ["Acute confusion in elderly patient", "Stiff neck or fever", "New-onset weakness or slurred speech"],
            "Elderly Confusion Risk"
        )

    if is_pediatric and "severe dehydration" in features.symptoms:
        apply_override(
            80,
            "Emergency",
            "Pediatric Dehydration Emergency",
            ["Severe dehydration in a child", "Lethargy or listlessness", "Sunken eyes or dry mouth"],
            "Pediatric Dehydration Risk"
        )

    # 5. Combination Matrix Overrides (Task 2)
    for entry in EMERGENCY_COMBINATION_MATRIX:
        if all(s in features.symptoms for s in entry["symptoms"]):
            if "abdominal pain" in entry["symptoms"]:
                if features.severity not in ["severe", "unbearable"]:
                    continue
            
            # Determine priority dynamically
            cond_lower = entry["condition"].lower()
            priority = 75
            if "myocardial" in cond_lower or "cardiac" in cond_lower:
                priority = 100
            elif "stroke" in cond_lower:
                priority = 100
            elif "meningitis" in cond_lower:
                priority = 90
            elif "hemorrhage" in cond_lower or "gi bleed" in cond_lower or "gastrointestinal" in cond_lower:
                priority = 85
                
            apply_override(
                priority,
                entry["urgency"],
                entry["condition"],
                entry["red_flags"],
                f"Combination: {' + '.join(entry['symptoms'])}"
            )

    # 6. Single symptom: Chest Pain (Fix 1 / Fix 3: Improved Chest Pain Logic)
    if "chest pain" in features.symptoms:
        high_risk = any([
            "shortness of breath" in features.symptoms,
            "sweating" in features.symptoms,
            "arm pain" in features.symptoms,
            "dizziness" in features.symptoms
        ])
        if high_risk:
            apply_override(
                100,
                "Emergency",
                "Potential Acute Coronary Syndrome (Heart Attack)",
                ["Pain radiating to left arm, neck, or jaw", "Profuse sweating", "Shortness of breath"],
                "Chest Pain (High Risk)"
            )
        else:
            apply_override(
                65,
                "Urgent",
                "Chest Pain Requiring Medical Evaluation",
                ["Pain radiating to left arm, neck, or jaw", "Profuse sweating", "Shortness of breath"],
                "Chest Pain (Low Risk)"
            )

    # 7. Confidence Guardrail (Task Upgrade 4)
    # If LLM confidence < 0.50 and urgency is low, escalate to at least Urgent or mark as physician review.
    if triage_response.confidence < 0.50 and triage_response.urgency.value in ["Self-Care", "Non-Urgent"]:
        apply_override(
            60,
            "Urgent",
            "Requires Physician Review",
            [],
            "Low Confidence Guardrail (<0.50)"
        )

    # 8. Logs to match TASK 5 requirements
    risk_str = ", ".join(detected_risks) if detected_risks else "None"
    override_str = "Yes" if override_applied else "No"
    
    # Normalize red flags to prevent duplicates
    triage_response.red_flags = normalize_red_flags(triage_response.red_flags)

    # Set escalation reason if risks were detected by validator
    if detected_risks:
        triage_response.escalation_reason = f"{risk_str} detected by validator"

    # Enforce legal disclaimer prefix (Fix 6)
    if LEGAL_DISCLAIMER not in triage_response.disclaimer:
        triage_response.disclaimer = f"{LEGAL_DISCLAIMER} {triage_response.disclaimer}"

    # Clinical Confidence Calibration (Issue 1)
    rule_res = evaluate_rules(features)
    rule_conf = rule_res.confidence if rule_res else 0.0
    rag_conf = min(state.get("retrieved_chunks", 0) / 3, 1.0) if state.get("retrieved_chunks") is not None else 0.0
    llm_conf = triage_response_orig.confidence if triage_response_orig else triage_response.confidence
    validator_conf = 0.98 if override_applied else 0.75

    final_conf = (
        0.35 * rule_conf +
        0.25 * rag_conf +
        0.25 * llm_conf +
        0.15 * validator_conf
    )
    triage_response.confidence = round(final_conf, 2)

    logger.info({
        "trace_id": trace_id,
        "node": "validator",
        "risk": risk_str,
        "original_urgency": original_urgency,
        "final_urgency": final_urgency,
        "safety_status": "SECURE",
        "override_applied": override_str,
        "calibrated_confidence": triage_response.confidence
    })

    llm_hit = bool(state.get("llm_cache_hit"))
    rag_hit = bool(state.get("rag_cache_hit"))
    
    if llm_hit:
        triage_response.cache_hit = True
        triage_response.cache_layer = "llm"
    elif rag_hit:
        triage_response.cache_hit = True
        triage_response.cache_layer = "rag"
    else:
        triage_response.cache_hit = False
        triage_response.cache_layer = None

    return {
        "triage_response": triage_response,
        "cache_hit": triage_response.cache_hit,
        "cache_layer": triage_response.cache_layer
    }


# Construct the StateGraph
workflow = StateGraph(TriageState)

# Add all execution nodes
workflow.add_node("feature_extraction_node", feature_extraction_node)
workflow.add_node("rule_engine_node", rule_engine_node)
workflow.add_node("analyze_node", analyze_node)
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("search_node", search_node)
workflow.add_node("triage_node", triage_node)
workflow.add_node("validator_node", validator_node)

# Define edges
workflow.add_edge(START, "feature_extraction_node")
workflow.add_edge("feature_extraction_node", "rule_engine_node")

# Conditional router from rule engine
workflow.add_conditional_edges(
    "rule_engine_node",
    local_triage_router,
    {
        "__end__": END,
        "analyze_node": "analyze_node"
    }
)

# Route conditionally based on critical check
workflow.add_conditional_edges(
    "analyze_node",
    critical_router,
    {
        "retrieve_node": "retrieve_node",
        "triage_node": "triage_node"
    }
)

# RAG retrieve node always routes to local DB search node
workflow.add_edge("retrieve_node", "search_node")

# Search node always advances to final triage node
workflow.add_edge("search_node", "triage_node")
workflow.add_edge("triage_node", "validator_node")
workflow.add_edge("validator_node", END)

# Compile the graph
triage_graph = workflow.compile()
