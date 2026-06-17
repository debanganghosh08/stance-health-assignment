import os
import logging
from typing import Any, Dict, List, Type, Union
from pydantic import BaseModel
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger("app.services.llm")
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Startup Dependency Validation Check
GROQ_AVAILABLE = False
try:
    import langchain
    import langchain_groq
    import groq
    GROQ_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Groq dependencies are missing ({e}). App will run in MockLLM fallback mode.")

class MockStructuredRunnable:
    """
    Mock structured LLM runnable that returns mock Pydantic models
    based on symptom keywords. Useful for offline testing and demoing without API keys.
    """
    def __init__(self, schema: Type[BaseModel]):
        self.schema = schema

    def _extract_text(self, input_messages: Any) -> str:
        """Helper to extract raw text content from LangChain messages format, ignoring system prompts."""
        if isinstance(input_messages, str):
            return input_messages
        elif isinstance(input_messages, list):
            text_parts = []
            for msg in input_messages:
                # Ignore system messages to avoid matching formatting/system keywords
                if hasattr(msg, "type") and msg.type == "system":
                    continue
                if isinstance(msg, dict) and msg.get("role") == "system":
                    continue
                
                if hasattr(msg, "content"):
                    text_parts.append(str(msg.content))
                elif isinstance(msg, dict) and "content" in msg:
                    text_parts.append(str(msg["content"]))
                else:
                    text_parts.append(str(msg))
            return " ".join(text_parts)
        elif hasattr(input_messages, "content"):
            if hasattr(input_messages, "type") and input_messages.type == "system":
                return ""
            return str(input_messages.content)
        return str(input_messages)

    def invoke(self, input_messages: Any, config: Any = None) -> BaseModel:
        text = self._extract_text(input_messages).lower()
        logger.info(f"[MOCK LLM] Simulating response for schema: {self.schema.__name__}")

        # Schema 1: SymptomAnalysis
        if self.schema.__name__ == "SymptomAnalysis":
            is_critical = any(kw in text for kw in [
                "chest pain", "heart", "stroke", "paralysis", "breathing", 
                "short of breath", "unconscious", "slurred", "bleeding", "anaphylaxis"
            ])
            
            # Simple keyword extraction for search query
            words = [w for w in text.split() if len(w) > 3 and w not in ["have", "with", "from", "about", "that", "this", "pain"]]
            query = " ".join(words[:3]) if words else "general symptoms"
            
            return self.schema(
                is_critical=is_critical,
                search_query=query if not is_critical else "emergency chest pain signs"
            )
            
        # Schema 2: TriageResponse or TriageAssessment
        elif self.schema.__name__ in ("TriageResponse", "TriageAssessment"):
            from app.models.schemas import UrgencyLevel
            
            # Extract just the patient reported symptoms block to avoid false matches on search references
            symptoms_text = text
            if "patient reported symptoms:" in text:
                try:
                    symptoms_text = text.split("patient reported symptoms:")[1].split("emergency classification:")[0]
                except Exception:
                    pass
            
            is_critical = "acute emergency suspected? yes" in text or any(kw in symptoms_text for kw in [
                "chest pain", "heart attack", "stroke", "short of breath", "anaphylaxis"
            ])
            is_urgent = any(kw in symptoms_text for kw in [
                "fever", "abdominal", "stomach", "vomit", "infection", "fracture", "severe pain"
            ])
            
            if is_critical:
                return self.schema(
                    urgency=UrgencyLevel.CRITICAL,
                    condition="Potential Acute Cardiac or Respiratory Event",
                    red_flags=[
                        "Crushing or radiating chest pain",
                        "Severe sudden shortness of breath",
                        "Confusion or loss of consciousness"
                    ],
                    confidence=0.95,
                    disclaimer="CRITICAL WARNING: These symptoms are potentially life-threatening. Seek immediate emergency care by calling 911 or visiting the nearest Emergency Room. Do not wait."
                )
            elif is_urgent:
                return self.schema(
                    urgency=UrgencyLevel.URGENT,
                    condition="Acute Inflammatory or Infectious Process",
                    red_flags=[
                        "Fever rising above 103°F (39.4°C)",
                        "Severe localized abdominal tenderness",
                        "Inability to keep liquids down for 24 hours"
                    ],
                    confidence=0.85,
                    disclaimer="This is a screening triage tool, not a clinical diagnosis. You should contact a doctor or visit an urgent care center within 12-24 hours for evaluation."
                )
            else:
                # Self-care vs Non-urgent
                is_self_care = any(kw in text for kw in ["cold", "sore muscle", "minor cut", "mild headache", "tired", "run yesterday"])
                if is_self_care:
                    return self.schema(
                        urgency=UrgencyLevel.SELF_CARE,
                        condition="Minor Self-Limiting Symptomatology",
                        red_flags=[
                            "Fever that persists longer than 3 days",
                            "Inability to tolerate fluids",
                            "Sudden worsening of symptoms"
                        ],
                        confidence=0.90,
                        disclaimer="These symptoms appear mild and manageable at home. Rest, stay hydrated, and monitor. If your symptoms worsen or do not improve in a few days, consult a healthcare professional."
                    )
                else:
                    return self.schema(
                        urgency=UrgencyLevel.NON_URGENT,
                        condition="Mild/Chronic Subacute Medical Condition",
                        red_flags=[
                            "Symptoms worsening over several days",
                            "Development of localized swelling or heat",
                            "New unexplained symptoms appearing"
                        ],
                        confidence=0.75,
                        disclaimer="Please schedule an appointment with your primary care provider at your convenience. Seek urgent care if your condition changes or new red flags develop."
                    )
                    
        # Fallback for any other schema
        return self.schema()

    async def ainvoke(self, input_messages: Any, config: Any = None) -> BaseModel:
        """Async variant of invoke to satisfy LangGraph's async execution path."""
        return self.invoke(input_messages, config)

class MockLLM:
    """
    Mock LLM wrapper that mimics the ChatGroq client class.
    """
    def with_structured_output(self, schema: Type[BaseModel]) -> MockStructuredRunnable:
        return MockStructuredRunnable(schema)

class ThrottledStructuredRunnable:
    """
    Wraps a structured runnable (e.g. ChatGroq.with_structured_output())
    and throttles execution by sleeping 1 second before calling the API.
    """
    def __init__(self, original_runnable: Any):
        self.original_runnable = original_runnable

    async def ainvoke(self, input_messages: Any, config: Any = None) -> BaseModel:
        import asyncio
        logger.info("Throttling Groq request: sleeping 1 second before ainvoke...")
        await asyncio.sleep(1)
        return await self.original_runnable.ainvoke(input_messages, config)

    def invoke(self, input_messages: Any, config: Any = None) -> BaseModel:
        import time
        logger.info("Throttling Groq request: sleeping 1 second before invoke...")
        time.sleep(1)
        return self.original_runnable.invoke(input_messages, config)

class ThrottledChatGroq:
    """
    Wraps the ChatGroq model instance to intercept `with_structured_output`
    calls and apply throttling to the returned runnables.
    """
    def __init__(self, client: Any):
        self.client = client

    def with_structured_output(self, schema: Type[BaseModel]) -> ThrottledStructuredRunnable:
        original_runnable = self.client.with_structured_output(schema)
        return ThrottledStructuredRunnable(original_runnable)

_llm_instance = None

def get_llm() -> Union[Any, MockLLM]:
    """
    Instantiates the ChatGroq model using the Groq API.
    If the GROQ_API_KEY is missing, placeholder or invalid, or if dependencies
    are missing or initialization fails, it returns a MockLLM client to allow
    complete local verification and offline demos.
    Reuses the initialized instance (singleton pattern).
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    _llm_instance = _initialize_llm()
    return _llm_instance

def _initialize_llm() -> Union[Any, MockLLM]:
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    timeout = float(os.getenv("GROQ_TIMEOUT", "30.0"))

    # Check dependency availability
    if not GROQ_AVAILABLE:
        logger.warning(
            "⚠️ Groq dependencies are unavailable. "
            "Mock mode enabled: using internal MockLLM service."
        )
        return MockLLM()

    # Check for empty, missing or default values
    is_mock = (
        not api_key or
        api_key.strip() == "" or
        "your_groq_api_key" in api_key.lower() or
        api_key == "mock"
    )

    if is_mock:
        logger.warning(
            "⚠️ GROQ_API_KEY is not set or contains default placeholder. "
            "Mock mode enabled: using internal MockLLM service."
        )
        return MockLLM()

    # Otherwise, return actual Groq ChatGroq client
    try:
        from langchain_groq import ChatGroq
        logger.info(f"🔌 Groq initialized: Initializing ChatGroq client with model: {model_name} (timeout: {timeout}s)")
        raw_client = ChatGroq(
            model_name=model_name,
            groq_api_key=api_key,
            temperature=0.1,  # Low temperature for clinical triage consistency
            request_timeout=timeout,
            max_retries=2
        )
        return ThrottledChatGroq(raw_client)
    except Exception as e:
        logger.error(f"❌ Failed to initialize ChatGroq: {e}. Groq unavailable. Falling back to MockLLM.", exc_info=True)
        return MockLLM()
