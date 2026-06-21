import logging
from typing import Optional
from app.graph.triage_graph import triage_graph
from app.models.schemas import TriageResponse, TriageState
from app.services.logging import request_id_var, log_audit_event
from app.services.metrics import metrics_registry

logger = logging.getLogger("app.services.triage")

class TriageService:
    """
    Orchestrates the symptom triage process by invoking the compiled LangGraph workflow.
    """
    async def triage_symptoms(self, message: str, patient_id: str) -> TriageResponse:
        import time
        from app.services.cache import get_normalized_hash, cache_service
        
        req_id = request_id_var.get("")
        logger.info(f"TriageService starting triage workflow for patient '{patient_id}': '{message[:60]}...'")
        
        start_time = time.time()
        
        # Calculate Level 1 cache key
        h = get_normalized_hash(message)
        cache_key = f"triage:{h}"
        
        # Check Level 1 Cache (Full Response Cache)
        cached_res = cache_service.get(cache_key)
        if cached_res is not None:
            try:
                triage_response = TriageResponse.model_validate(cached_res)
                triage_response.patient_id = patient_id
                triage_response.cache_hit = True
                triage_response.cache_layer = "triage_response"
                
                delta_ms = int((time.time() - start_time) * 1000)
                triage_response.processing_time_ms = max(delta_ms, 0)
                
                # Replace prints with logger
                logger.info(f"CACHE: Layer: triage_response | Key: {cache_key} | Hit: True")
                logger.info(f"PERFORMANCE: Processing Time: {triage_response.processing_time_ms}ms")
                
                # Record metrics
                metrics_registry.increment("cache_hits")
                
                # Secure Clinical Audit completion event
                log_audit_event(
                    "triage_completed",
                    req_id,
                    patient_id,
                    {
                        "urgency": triage_response.urgency.value,
                        "condition": triage_response.condition,
                        "override_applied": False,
                        "cache_layer": "triage_response",
                        "latency_ms": triage_response.processing_time_ms
                    }
                )
                
                return triage_response
            except Exception as e:
                logger.warning(f"⚠️ Failed to parse cached triage response: {e}. Re-executing pipeline.")
        
        # Initialize the LangGraph State Dict
        initial_state: TriageState = {
            "patient_id": patient_id,
            "message": message,
            "is_critical": None,
            "search_query": None,
            "search_results": None,
            "triage_response": None,
            "errors": [],
            "cache_hit": False,
            "cache_layer": None,
            "llm_cache_hit": False,
            "rag_cache_hit": False,
            "trace_id": req_id
        }
        
        try:
            # Run the compiled StateGraph workflow asynchronously
            final_state = await triage_graph.ainvoke(initial_state)
            
            # Print Triage Engine diagnostics logs (Task 7) -> Replace prints with logger
            local_bypass = final_state.get("local_triage_used", False)
            rag_used = final_state.get("rag_used", False)
            llm_used = not local_bypass
            
            logger.info(
                f"[TRIAGE ENGINE] Local Bypass: {'Yes' if local_bypass else 'No'} | "
                f"LLM Used: {'Yes' if llm_used else 'No'} | "
                f"RAG Used: {'Yes' if rag_used else 'No'}"
            )
            
            # Retrieve the structured response from state
            triage_response: Optional[TriageResponse] = final_state.get("triage_response")
            
            # Log any internal pipeline errors that occurred during execution
            pipeline_errors = final_state.get("errors", [])
            if pipeline_errors:
                logger.warning(f"⚠️ Symptoms triage pipeline encountered non-fatal errors: {pipeline_errors}")
            
            if not triage_response:
                raise RuntimeError("Symptoms triage graph execution completed but failed to yield a triage response.")
            
            # Calculate processing metrics
            delta_ms = int((time.time() - start_time) * 1000)
            triage_response.processing_time_ms = max(delta_ms, 1)
            
            # Copy cache status from graph execution state (if any inner cache was hit)
            triage_response.cache_hit = final_state.get("cache_hit", False) or False
            triage_response.cache_layer = final_state.get("cache_layer")
            
            # Conditional Caching Constraint: Cache only if RAG was used or LLM was used (local_triage_used is False)
            expensive = triage_response.rag_used or (not triage_response.local_triage_used)
            if expensive:
                cache_service.set(cache_key, triage_response.model_dump())
                logger.info(f"CACHE: Layer: triage_response | Key: {cache_key} | Hit: False")
            else:
                logger.info(f"ℹ️ Request is not expensive (local rules match, no RAG/LLM). Skipping cache storage.")
                
            logger.info(f"PERFORMANCE: Processing Time: {triage_response.processing_time_ms}ms")
            
            # Record metrics
            if triage_response.cache_hit:
                metrics_registry.increment("cache_hits")
            else:
                metrics_registry.increment("cache_misses")
                
            if local_bypass:
                metrics_registry.increment("local_bypass_count")
                
            from app.models.schemas import UrgencyLevel
            if triage_response.urgency in (UrgencyLevel.EMERGENCY, UrgencyLevel.CRITICAL):
                metrics_registry.increment("emergency_cases")
                
            # Secure Clinical Audit completion event
            log_audit_event(
                "triage_completed",
                req_id,
                patient_id,
                {
                    "urgency": triage_response.urgency.value,
                    "condition": triage_response.condition,
                    "override_applied": final_state.get("override_applied", False) or False,
                    "cache_layer": triage_response.cache_layer,
                    "latency_ms": triage_response.processing_time_ms
                }
            )
            
            return triage_response
            
        except Exception as e:
            logger.error(f"❌ Critical exception inside TriageService: {e}", exc_info=True)
            raise e
