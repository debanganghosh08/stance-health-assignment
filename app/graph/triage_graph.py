import logging
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from app.models.schemas import TriageState, SymptomAnalysis, TriageAssessment, TriageResponse, UrgencyLevel
from app.prompts.triage_prompt import ANALYSIS_SYSTEM_PROMPT, TRIAGE_SYSTEM_PROMPT
from app.services.llm import get_llm
from app.services.search import SearchService

logger = logging.getLogger("app.graph.triage_graph")

# List of critical symptoms for deterministic rule-based pre-check
EMERGENCY_KEYWORDS = [
    "chest pain",
    "stroke",
    "slurred speech",
    "seizure",
    "anaphylaxis",
    "unable to breathe",
    "shortness of breath",
    "loss of consciousness",
    "severe bleeding"
]

async def analyze_node(state: TriageState) -> dict:
    """
    Analyzes raw patient symptoms to detect critical life-threatening emergencies
    and formulate a relevant search query for the medical database.
    """
    logger.info("🟢 Starting analyze_node")
    message = state.get("message", "")
    errors = state.get("errors", [])

    # Step 1: Rule-based deterministic emergency check
    message_lower = message.lower()
    detected_keywords = [kw for kw in EMERGENCY_KEYWORDS if kw in message_lower]
    
    if detected_keywords:
        logger.info(f"🚨 Rule-based check: Emergency detected via keywords {detected_keywords}. Bypassing LLM symptom classification.")
        return {
            "is_critical": True,
            "search_query": "emergency bypass",
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
        
        logger.info(f"Analysis complete. Is critical: {analysis.is_critical}, Search query: '{analysis.search_query}'")
        
        return {
            "is_critical": analysis.is_critical,
            "search_query": analysis.search_query,
            "errors": errors
        }
    except Exception as e:
        logger.error(f"❌ Error in analyze_node: {e}", exc_info=True)
        # Fallback to safe default values to ensure robustness
        return {
            "is_critical": False,
            "search_query": message[:50],  # Fallback search query is first 50 chars of symptoms message
            "errors": errors + [f"analyze_node error: {str(e)}"]
        }

async def search_node(state: TriageState) -> dict:
    """
    Queries the offline medical reference database using the query formulated in the analysis stage.
    """
    logger.info("🟢 Starting search_node")
    search_query = state.get("search_query", "")
    errors = state.get("errors", [])
    
    search_service = SearchService()
    
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
    Synthesizes the patient symptoms, medical search context, and emergency status
    to output the final structured triage recommendation.
    """
    logger.info("🟢 Starting triage_node")
    message = state.get("message", "")
    search_results = state.get("search_results") or "No search context available."
    is_critical = state.get("is_critical", False)
    errors = state.get("errors", [])
    patient_id = state.get("patient_id", "unknown")

    llm = get_llm()
    
    # Prepare comprehensive clinical context for triage
    clinical_context = (
        f"PATIENT REPORTED SYMPTOMS:\n{message}\n\n"
        f"EMERGENCY CLASSIFICATION:\nIs acute emergency suspected? {'YES' if is_critical else 'NO'}\n\n"
        f"RETRIEVED MEDICAL REFERENCES:\n{search_results}"
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

        # Propagate patient_id into the final TriageResponse model
        triage_response = TriageResponse(
            patient_id=patient_id,
            urgency=final_urgency,
            condition=assessment.condition,
            red_flags=assessment.red_flags,
            confidence=assessment.confidence,
            disclaimer=assessment.disclaimer
        )
        
        logger.info(f"Triage complete. Level: {triage_response.urgency.value}, Suspected condition: {triage_response.condition}")
        
        return {
            "triage_response": triage_response,
            "errors": errors
        }
    except Exception as e:
        logger.error(f"❌ Error in triage_node: {e}", exc_info=True)
        
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
            disclaimer="CRITICAL DISCLAIMER: An internal processing error occurred. If you are experiencing severe symptoms, chest pain, or breathing difficulties, please go to the nearest ER or contact emergency services immediately."
        )
        return {
            "triage_response": fallback,
            "errors": errors + [f"triage_node error: {str(e)}"]
        }

def critical_router(state: TriageState) -> Literal["search_node", "triage_node"]:
    """
    Router determining if we bypass the clinical knowledge lookup.
    Bypassing search on critical symptoms ensures lower latency when immediate intervention is needed.
    """
    if state.get("is_critical", False):
        logger.info("🚨 Critical emergency detected. Bypassing knowledge search node.")
        return "triage_node"
    logger.info("ℹ️ Non-critical symptoms. Routing to knowledge search node.")
    return "search_node"

# Construct the StateGraph
workflow = StateGraph(TriageState)

# Add all execution nodes
workflow.add_node("analyze_node", analyze_node)
workflow.add_node("search_node", search_node)
workflow.add_node("triage_node", triage_node)

# Define edges
workflow.add_edge(START, "analyze_node")

# Route conditionally based on critical check
workflow.add_conditional_edges(
    "analyze_node",
    critical_router,
    {
        "search_node": "search_node",
        "triage_node": "triage_node"
    }
)

# Search node always advances to final triage node
workflow.add_edge("search_node", "triage_node")
workflow.add_edge("triage_node", END)

# Compile the graph
triage_graph = workflow.compile()
