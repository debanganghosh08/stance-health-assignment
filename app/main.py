import os
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv

from app.models.schemas import TriageRequest, TriageResponse, BatchTriageResponse
from app.services.triage import TriageService
from app.services.cases import CasesService
from app.services.logging import setup_logging, request_id_var
from app.services.metrics import metrics_registry

# Load environment variables
load_dotenv()

# Logger instance
logger = logging.getLogger("app.main")

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that manages trace tracking via X-Request-ID headers and thread-safe ContextVars.
    """
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID")
        if not req_id:
            req_id = str(uuid.uuid4())
            
        token = request_id_var.set(req_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
            
        response.headers["X-Request-ID"] = req_id
        return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize unified logging
    setup_logging()
    
    from app.services.llm import GROQ_AVAILABLE
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    # Check if dependencies are missing on startup
    missing_deps = []
    for dep in ["langchain", "langchain_groq", "groq"]:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)
            
    if missing_deps:
        logger.error(
            f"❌ STARTUP DEPENDENCY VALIDATION FAILED: Missing dependencies: {missing_deps}. "
            "Please ensure requirements.txt is installed in the active python environment. "
            "Server will continue running in MockLLM mode."
        )
    else:
        logger.info("✅ STARTUP DEPENDENCY VALIDATION PASSED: All Groq modules are importable.")
 
    is_mock = (
        not GROQ_AVAILABLE or
        not api_key or
        api_key.strip() == "" or
        "your_groq_api_key" in api_key.lower() or
        api_key == "mock"
    )
 
    # Pre-initialize FAISS RAG retriever
    try:
        from app.services.rag import get_retriever
        logger.info("📂 Lifespan startup: Pre-initializing RAG retriever index...")
        get_retriever()
        logger.info("📂 Lifespan startup: RAG retriever index ready.")
    except Exception as startup_err:
        logger.error(f"❌ Failed to initialize RAG retriever on startup: {startup_err}", exc_info=True)
 
    logger.info("=========================================")
    logger.info("🚀 Symptom Triage Agent App Initializing")
    logger.info(f"🤖 Configured LLM Model: {model}")
    logger.info(f"🔑 Mock LLM Mode: {'ACTIVE (Offline Sandbox)' if is_mock else 'INACTIVE (Live Groq)'}")
    logger.info("=========================================")
    yield

# Instantiate FastAPI application
app = FastAPI(
    title="Symptom Triage Agent API",
    description="Production-style clinical symptom triage screening API powered by LangGraph and Groq LLM.",
    version="1.0.0",
    lifespan=lifespan
)

# Register Request ID Middleware first to establish correlation context early
app.add_middleware(RequestIDMiddleware)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handler: Request Validation Errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Request validation failure on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Unprocessable Entity",
            "message": "Input validation failed. Please check the structure of your payload.",
            "details": exc.errors()
        }
    )

# Exception Handler: General Exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception occurred on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while executing the symptoms triage pipeline.",
            "details": str(exc)
        }
    )

# Endpoints
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Check the health of the service, environment variables, and Mock LLM state.
    """
    from app.services.llm import GROQ_AVAILABLE
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    is_mock = (
        not GROQ_AVAILABLE or
        not api_key or
        api_key.strip() == "" or
        "your_groq_api_key" in api_key.lower() or
        api_key == "mock"
    )
    
    return {
        "status": "healthy",
        "groq_available": GROQ_AVAILABLE,
        "mock_mode": is_mock,
        "model": model
    }

@app.get("/system/health")
async def system_health():
    """
    Detailed health check including subsystems api, groq, faiss, cache, and audit logger.
    Supports healthy, degraded, and unhealthy modes.
    """
    # 1. API status
    api_status = "healthy"
    
    # 2. Groq status
    from app.services.llm import GROQ_AVAILABLE
    api_key = os.getenv("GROQ_API_KEY")
    is_mock = (
        not GROQ_AVAILABLE or
        not api_key or
        api_key.strip() == "" or
        api_key == "mock" or
        "your_groq_api_key" in api_key.lower()
    )
    groq_status = "degraded" if is_mock else "healthy"
    
    # 3. FAISS status
    from app.services.rag import get_retriever
    try:
        retriever = get_retriever()
        if retriever.index is not None:
            faiss_status = "healthy"
        elif getattr(retriever, "mock_chunks", None) is not None:
            faiss_status = "degraded"
        else:
            faiss_status = "unhealthy"
    except Exception:
        faiss_status = "unhealthy"
        
    # 4. Cache status
    from app.services.cache import cache_service
    try:
        if cache_service.redis_cache.is_active:
            cache_status = "healthy"
        else:
            cache_status = "degraded"
    except Exception:
        cache_status = "unhealthy"
        
    # 5. Audit logger status
    audit_status = "healthy"
    try:
        os.makedirs("logs", exist_ok=True)
        test_file = "logs/.audit_health_check"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("test")
        os.remove(test_file)
    except Exception:
        audit_status = "unhealthy"
        
    overall_status = "healthy"
    statuses = [api_status, groq_status, faiss_status, cache_status, audit_status]
    if "unhealthy" in statuses:
        overall_status = "unhealthy"
    elif "degraded" in statuses:
        overall_status = "degraded"
        
    return JSONResponse(
        status_code=status.HTTP_200_OK if overall_status != "unhealthy" else status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": overall_status,
            "details": {
                "api": api_status,
                "groq": groq_status,
                "faiss": faiss_status,
                "cache": cache_status,
                "audit_logger": audit_status
            }
        }
    )

@app.get("/metrics")
async def get_metrics(request: Request):
    """
    Exposes registry metrics in JSON or Prometheus text format.
    """
    accept_header = request.headers.get("Accept", "")
    if "text/plain" in accept_header or "prometheus" in accept_header:
        from fastapi.responses import Response
        return Response(content=metrics_registry.export_prometheus(), media_type="text/plain")
    return metrics_registry.export_json()

@app.post("/api/v1/triage", response_model=TriageResponse, status_code=status.HTTP_200_OK)
async def triage_symptoms(request: TriageRequest):
    """
    Triage patient symptom message to evaluate urgency, suspect condition, red flags, confidence, and disclaimer.
    """
    import time
    start = time.time()
    metrics_registry.increment("total_requests")
    triage_service = TriageService()
    try:
        response = await triage_service.triage_symptoms(
            message=request.message,
            patient_id=request.patient_id
        )
        latency = (time.time() - start) * 1000
        metrics_registry.observe_latency("overall", latency)
        metrics_registry.increment("successful_requests")
        return response
    except Exception as e:
        metrics_registry.increment("failed_requests")
        logger.error(f"Failed to triage symptoms: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Triage process failed: {str(e)}"
        )

@app.post("/api/v1/batch-triage", response_model=BatchTriageResponse, status_code=status.HTTP_200_OK)
async def batch_triage_symptoms():
    """
    Fetch patient cases from remote URL and perform batch triage evaluations using a rule-first approach.
    """
    import time
    start = time.time()
    metrics_registry.increment("total_requests")
    cases_service = CasesService()
    try:
        response = await cases_service.run_batch_triage()
        latency = (time.time() - start) * 1000
        metrics_registry.observe_latency("overall", latency)
        metrics_registry.increment("successful_requests")
        return response
    except Exception as e:
        metrics_registry.increment("failed_requests")
        logger.error(f"Failed to execute batch triage: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch triage process failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)
