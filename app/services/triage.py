import logging
from typing import Optional
from app.graph.triage_graph import triage_graph
from app.models.schemas import TriageResponse, TriageState

logger = logging.getLogger("app.services.triage")

class TriageService:
    """
    Orchestrates the symptom triage process by invoking the compiled LangGraph workflow.
    """
    async def triage_symptoms(self, message: str, patient_id: str) -> TriageResponse:
        logger.info(f"TriageService starting triage workflow for patient '{patient_id}': '{message[:60]}...'")
        
        # Initialize the LangGraph State Dict
        initial_state: TriageState = {
            "patient_id": patient_id,
            "message": message,
            "is_critical": None,
            "search_query": None,
            "search_results": None,
            "triage_response": None,
            "errors": []
        }
        
        try:
            # Run the compiled StateGraph workflow asynchronously
            final_state = await triage_graph.ainvoke(initial_state)
            
            # Retrieve the structured response from state
            triage_response: Optional[TriageResponse] = final_state.get("triage_response")
            
            # Log any internal pipeline errors that occurred during execution
            pipeline_errors = final_state.get("errors", [])
            if pipeline_errors:
                logger.warning(f"⚠️ Symptoms triage pipeline encountered non-fatal errors: {pipeline_errors}")
            
            if not triage_response:
                raise RuntimeError("Symptoms triage graph execution completed but failed to yield a triage response.")
                
            return triage_response
            
        except Exception as e:
            logger.error(f"❌ Critical exception inside TriageService: {e}", exc_info=True)
            raise e
