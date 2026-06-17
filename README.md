# 🩺 Symptom Triage Agent

An advanced, production-grade clinical symptom triage screening API built with **FastAPI**, **LangGraph**, **Pydantic v2**, and **Groq LLM**.

This system is designed as a secure, fast, and highly reliable patient symptom screening service. It evaluates symptom descriptions, determines clinical urgency, identifies suspected medical conditions, extracts red flag warnings, generates standardized clinical disclaimers, and outputs structured, validated JSON payloads. 

### 🎯 The Problem & Context
In clinical operations and digital health, processing incoming patient messages at scale poses multiple challenges:
1. **Clinical Safety & Speed**: Critical emergencies must bypass standard pipelines immediately (low latency, zero LLM hallucinations).
2. **API Cost & Rate Limits**: Large-scale batch runs (e.g., triage processing on 100+ patient records) trigger `HTTP 429 Too Many Requests` due to high concurrency.
3. **Structured Integration**: Downstream EHR (Electronic Health Record) systems require strict, deterministic data schemas.

This repository implements a **Rule-First Architecture** with **Groq Rate-Limiting Mitigation** (throttling, singleton client caching, LLM budget limit, and graceful fallbacks) to achieve high safety, ultra-low cost, and robust reliability under heavy load.

---

## 🚀 Key Features

*   ⚡ **FastAPI Backend**: Fully asynchronous, lightweight API server featuring unified global exception handlers and robust request validation.
*   🕸️ **LangGraph Workflow**: Orchestrates analysis, medical database search, and final clinical triage as a state machine.
*   🧠 **Groq LLM Integration**: Uses high-performance models (e.g., `llama-3.1-8b-instant`) for natural language symptom understanding.
*   🛡️ **Rule-First Architecture**: Bypasses LLM evaluation for common patterns to maximize speed and eliminate API costs.
*   🚨 **Emergency Pre-Check Detection**: Scans user inputs for critical cardiovascular or neurological keywords (e.g., *chest pain*, *stroke*) to bypass knowledge lookup and instantly route to emergency status.
*   📦 **Structured Outputs**: Direct integration of Pydantic models with Groq structured JSON parser to guarantee EHR compatibility.
*   📊 **Batch Triage Endpoint**: Process datasets of patient cases concurrently with budget limits (`BATCH_MAX_LLM_CASES`) and real-time efficiency metrics.
*   🐢 **Request Throttling**: Automatic 1-second delay pre-Groq call in live mode to avoid burst limit exhaustion.
*   🔄 **Graceful Fallbacks & Mock Mode**: Falls back to offline rule engines and mock structures if Groq API keys are absent or network endpoints are offline.
*   🧪 **Robust Test Suite**: Pytest coverage with mocked API keys running in under a second.

---

## 🏗️ System Architecture

The workflow is built on top of a LangGraph `StateGraph`. Incoming user symptom messages are analyzed, checked for acute conditions, routed dynamically, enhanced with local clinical guidance, and compiled into a structured clinical output.

### Data Flow Diagram
```mermaid
graph TD
    Start([POST /api/v1/triage]) --> RuleCheck{1. Rule Engine Check}
    
    RuleCheck -- "Rule Match (Conf >= 0.8)" --> ReturnRule[Return Deterministic Response]
    RuleCheck -- "No Rule Match" --> AnalyzeNode[2. Analyze Node]
    
    AnalyzeNode --> EmergencyCheck{3. Emergency Keyword / LLM Check?}
    
    EmergencyCheck -- "Yes (Critical Emergency)" --> TriageNode[5. Triage Node (Bypasses Search)]
    EmergencyCheck -- "No (Standard)" --> SearchNode[4. Medical DB Search Node]
    
    SearchNode --> TriageNode
    TriageNode --> ReturnLLM[Return LLM Response]
```

### LangGraph Node Definition
1.  **Symptom Analyzer (`analyze_node`)**: Inspects symptoms to see if they constitute an emergency (e.g., crushing chest pain) and builds a 2-4 keyword database search query.
2.  **Conditional Router (`critical_router`)**: Bypasses database searches on critical emergencies to reduce latency. Non-urgent cases are routed to the knowledge reference search node.
3.  **Knowledge Search (`search_node`)**: Queries a local clinical guide reference database to find relevant medical context.
4.  **Triage Generator (`triage_node`)**: Synthesizes symptom details, search references, and urgency parameters to output structured output parameters.

---

## 🛠️ Tech Stack

| Technology | Role / Purpose | Why Chosen? |
| :--- | :--- | :--- |
| **FastAPI** | REST API Framework | Asynchronous capabilities, automatic OpenAPI/Swagger documentation, and native Pydantic integration. |
| **LangGraph** | Workflow Orchestrator | Graph-based state machines that allow cycles, conditional routing, and deterministic control of agentic loops. |
| **LangChain Groq** | LLM Integration | High-level abstraction for calling Groq APIs with native structured output parsing. |
| **Groq API** | Inference Provider | Extremely high throughput and low-latency responses using Llama 3 models. |
| **Pydantic v2** | Data Validation | Core validation tool ensuring incoming and outgoing payloads match strict clinical EHR schemas. |
| **Pytest** | Testing Framework | Powerful local testing with assertions, fixtures, and execution speed. |
| **Uvicorn** | ASGI Server | High-performance server runner for FastAPI. |

---

## 📂 Project Structure

```text
app/
├── graph/
│   └── triage_graph.py       # LangGraph definition, nodes, conditional routing
├── models/
│   └── schemas.py            # Pydantic v2 schemas for API requests, responses, and Graph state
├── prompts/
│   └── triage_prompt.py      # System prompts for triage and classification nodes
├── resources/
│   └── cases_fallback.json   # 100 sample cases used for offline batch triage testing
├── services/
│   ├── cases.py              # Batch processing service, rule engine, fallback logic
│   ├── llm.py                # ChatGroq client creation, singleton caching, and sleep throttling
│   ├── search.py             # Local Medical reference keyword indexing service
│   └── triage.py             # Orchestrator running the compiled LangGraph workflow
├── main.py                   # FastAPI Application setup and route definitions
```

### Folder Responsibilities:
*   `graph/`: Houses the logical definition of nodes, edges, state transitions, and conditional routing.
*   `models/`: Maintains the integrity of all input/output data contracts.
*   `prompts/`: Version-controlled clinical context instruction prompts.
*   `services/`: Business logic, client clients, search algorithms, and batch processing controls.

---

## 🔌 API Endpoints

### 1. Health Status Check
*   **Path**: `GET /health`
*   **Purpose**: Returns the service status, configured model, and whether the API is running in Live Groq or Mock LLM mode.
*   **Response Schema**:
    ```json
    {
      "status": "healthy",
      "groq_available": true,
      "mock_mode": false,
      "model": "llama-3.1-8b-instant"
    }
    ```

---

### 2. Single Patient Triage
*   **Path**: `POST /api/v1/triage`
*   **Purpose**: Runs the LangGraph screening pipeline on a single patient symptom message.
*   **Request Payload**:
    ```json
    {
      "patient_id": "pat_102",
      "message": "I have crushing chest pain radiating to my left arm."
    }
    ```
*   **Response Payload**:
    ```json
    {
      "patient_id": "pat_102",
      "urgency": "Emergency",
      "condition": "Potential Acute Coronary Syndrome (Heart Attack)",
      "red_flags": [
        "Pain radiating to left arm, neck, or jaw",
        "Profuse sweating (diaphoresis)",
        "Shortness of breath or nausea"
      ],
      "confidence": 0.95,
      "disclaimer": "CRITICAL WARNING: These symptoms are potentially life-threatening. Seek immediate emergency care by calling 911 or visiting the nearest ER."
    }
    ```

---

### 3. Batch Triage
*   **Path**: `POST /api/v1/batch-triage`
*   **Purpose**: Fetches patient cases, triages them sequentially using a rule-first strategy, enforces budget limits, throttles requests, and returns results along with efficiency metrics.
*   **Request Payload**: None (fetches automatically from case database).
*   **Response Payload**:
    ```json
    {
      "total_cases": 100,
      "processed_cases": 100,
      "results": [ ... ],
      "metrics": {
        "total_cases": 100,
        "emergency_count": 8,
        "urgent_count": 76,
        "non_urgent_count": 10,
        "self_care_count": 6
      },
      "groq_calls_used": 30,
      "groq_calls_saved": 170,
      "llm_budget_exhausted": true
    }
    ```

---

## 📊 Batch Processing & Rate-Limiting Controls

When processing batch loads of 100 cases, rate-limit (429) mitigation is handled via five distinct layers:
1.  **LLM Budget Cap (`BATCH_MAX_LLM_CASES=15`)**: Restricts the maximum number of cases that can utilize LLM resources. Once reached, subsequent cases that do not match the rule engine default immediately to a clinical fallback response.
2.  **Rule-First Engine**: Compares symptom descriptions against highly optimized deterministic patterns. If matched with high confidence ($\ge 0.8$), the case bypasses LLM entirely.
3.  **Singleton Client**: Reuses a single cached client wrapper (`ThrottledChatGroq`) to avoid the overhead of repeated client initializations.
4.  **Request Throttling**: Live API calls sleep for 1 second (`sleep(1)`) to avoid exhausting Groq's token-per-minute (TPM) and request-per-minute (RPM) limits.
5.  **Offline Fallback JSON**: If the external cases API is offline, the system automatically falls back to reading [`app/resources/cases_fallback.json`](file:///c:/stance-agent/app/resources/cases_fallback.json) to prevent service interruptions.

---

## 🚨 Emergency Detection & Clinical Safety

### Keyword Detection List:
`chest pain`, `stroke`, `slurred speech`, `seizure`, `anaphylaxis`, `unable to breathe`, `shortness of breath`, `loss of consciousness`, `severe bleeding`.

*   **Safety Implementation**: The triage graph interceptor detects these emergencies, assigns `is_critical=True`, and bypasses database search.
*   **Standardized Disclaimers**: Every response outputs a validated medical warning clarifying that the tool is a screen, not a diagnosis, advising appropriate next steps.
*   **Fallback Response**: If the LLM budget is exhausted during batch processing, the agent safely falls back to a standardized triage assessment containing:
    *   **Urgency**: `Urgent`
    *   **Condition**: `Needs Clinical Review`
    *   **Red Flags**: `[]`
    *   **Confidence**: `0.50`
    *   **Disclaimer**: Standard clinical safety disclaimer.

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable Name | Purpose | Example / Default |
| :--- | :--- | :--- |
| `PORT` | Local FastAPI server port | `8000` |
| `HOST` | Local binding interface | `0.0.0.0` |
| `GROQ_API_KEY` | Groq authorization key | `gsk_...` or `mock` |
| `GROQ_MODEL` | LLM model identifier | `llama-3.1-8b-instant` |
| `BATCH_MAX_LLM_CASES` | Max LLM queries in a batch run | `15` |

---

## 💻 Installation & Setup

### 1. Clone & Navigate
```bash
git clone https://github.com/Chris-healthflex/ai-intern.git
cd ai-intern
```

### 2. Configure Virtual Environment
```bash
# Create environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration & Execution

Create a `.env` file in the root directory:
```env
PORT=8000
HOST=0.0.0.0
GROQ_API_KEY=your_actual_groq_key_here
GROQ_MODEL=llama-3.1-8b-instant
BATCH_MAX_LLM_CASES=15
```

### Run Server Locally:
```bash
uvicorn app.main:app --reload
```
Once started, explore the interactive documentation:
*   Interactive Swagger API: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
*   Alternative ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🧪 Testing

The repository contains automated unit tests verifying health checks, request validations, rule bypass engines, mock fallbacks, and batch limits.

```bash
# Run pytest tests
pytest -v tests/test_triage.py
```

*Note: The test file automatically overrides `GROQ_API_KEY="mock"` to run in instant, zero-cost mock mode (finishing in under `0.6s`).*

---

## 🛡️ Performance Optimizations Summary
*   **LLM Bypass**: Rules resolve known symptoms in $< 1\text{ms}$ with zero network queries.
*   **Rate-Limit Throttling**: The 1-second delay and cached singleton protect API limits.
*   **Fallback Triage**: Graceful recovery with static responses when LLM rates are exhausted.

---

## 🔮 Future Improvements
1.  **RAG-Enhanced Search**: Replace the mock keyword database with a Vector Database containing clinical literature (e.g. UpToDate).
2.  **Multi-Model Fallbacks**: Auto-switch to secondary endpoints (e.g. OpenRouter or local Ollama) if Groq returns HTTP status errors.
3.  **Analytics & Logging Dashboard**: Implement LangSmith telemetry tracing alongside a React-based monitoring dashboard.
4.  **EHR Sync Integration**: Direct endpoints connecting triage outputs into FHIR/HL7 profiles.

---

## ✍️ Author
**Debangan Ghosh**  
*   **GitHub**: [@debanganghosh](https://github.com/debanganghosh)  
*   **LinkedIn**: [Profile Link](https://www.linkedin.com/in/debanganghosh)

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
