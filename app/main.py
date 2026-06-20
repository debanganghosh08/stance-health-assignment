import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.models.schemas import TriageRequest, TriageResponse, BatchTriageResponse
from app.services.triage import TriageService
from app.services.cases import CasesService

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
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


# Instantiate FastAPI application with lifespan context manager
app = FastAPI(
    title="Symptom Triage Agent API",
    description="Production-style clinical symptom triage screening API powered by LangGraph and Groq LLM.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend flexibility (if integrated later)
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

@app.post("/api/v1/triage", response_model=TriageResponse, status_code=status.HTTP_200_OK)
async def triage_symptoms(request: TriageRequest):
    """
    Triage patient symptom message to evaluate urgency, suspect condition, red flags, confidence, and disclaimer.
    """
    triage_service = TriageService()
    try:
        response = await triage_service.triage_symptoms(
            message=request.message,
            patient_id=request.patient_id
        )
        return response
    except Exception as e:
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
    cases_service = CasesService()
    try:
        response = await cases_service.run_batch_triage()
        return response
    except Exception as e:
        logger.error(f"Failed to execute batch triage: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch triage process failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    # Allow running main.py directly for development ease
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)
