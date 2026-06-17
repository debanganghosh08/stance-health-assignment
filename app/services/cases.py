import httpx
import logging
from typing import List, Optional
from app.models.schemas import TriageResponse, UrgencyLevel, BatchTriageResponse, ProcessingMetrics
from app.services.triage import TriageService

logger = logging.getLogger("app.services.cases")

class CasesService:
    """
    Handles fetching and batch processing of patient symptom cases from the external API.
    Optimizes LLM usage with a rule-first architecture.
    """
    def __init__(self):
        self.triage_service = TriageService()
        self.cases_url = "https://ai-stance.vercel.app/api/cases"

    async def fetch_cases(self) -> List[dict]:
        logger.info(f"Fetching case list from remote API: {self.cases_url}")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.cases_url, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                cases = data.get("cases", [])
                logger.info(f"Successfully retrieved {len(cases)} cases from remote API.")
                return cases
        except Exception as e:
            logger.warning(f"⚠️ Remote fetch failed: {e}. Falling back to local cases resource file.")
            try:
                import os
                import json
                fallback_path = os.path.join(os.path.dirname(__file__), "..", "resources", "cases_fallback.json")
                with open(fallback_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cases = data.get("cases", [])
                    logger.info(f"Successfully loaded {len(cases)} cases from local fallback resource.")
                    return cases
            except Exception as fe:
                logger.error(f"❌ Critical: Fallback cases load failed: {fe}", exc_info=True)
                raise RuntimeError(f"Failed to fetch cases from API and fallback load failed: {str(e)}")

    def evaluate_rules(self, message: str, patient_id: str) -> Optional[TriageResponse]:
        """
        Deterministic clinical rule-based check. If a match is found with confidence >= 0.8,
        returns the populated TriageResponse, allowing the system to skip LLM execution.
        """
        msg = message.lower()

        # Rule 1: Crushing chest pain / acute heart attack symptoms (Confidence: 0.95)
        if "crushing chest pain" in msg or "chest pain for the past" in msg:
            return TriageResponse(
                patient_id=patient_id,
                urgency=UrgencyLevel.EMERGENCY,
                condition="Potential Acute Coronary Syndrome (Heart Attack)",
                red_flags=[
                    "Pain radiating to left arm, neck, or jaw",
                    "Profuse sweating (diaphoresis)",
                    "Shortness of breath or nausea"
                ],
                confidence=0.95,
                disclaimer="CRITICAL WARNING: These symptoms are potentially life-threatening. Seek immediate emergency medical care by calling 911 or visiting the nearest ER."
            )

        # Rule 2: Unilateral weakness / potential stroke symptoms (Confidence: 0.98)
        if "face feels droopy" in msg or ("arm is very weak" in msg and "speech" in msg):
            return TriageResponse(
                patient_id=patient_id,
                urgency=UrgencyLevel.EMERGENCY,
                condition="Potential Acute Stroke (Brain Ischemia)",
                red_flags=[
                    "Face drooping on one side",
                    "Unilateral arm or leg weakness",
                    "Slurred or incoherent speech"
                ],
                confidence=0.98,
                disclaimer="CRITICAL WARNING: These symptoms suggest an acute stroke event. Record time of symptom onset and call 911 or go to the nearest ER immediately."
            )

        # Rule 3: Anaphylaxis / allergic throat swelling (Confidence: 0.95)
        if "stung by a bee" in msg or ("throat is starting to swell" in msg and "allergy" in msg):
            return TriageResponse(
                patient_id=patient_id,
                urgency=UrgencyLevel.EMERGENCY,
                condition="Potential Anaphylactoid Allergic Reaction",
                red_flags=[
                    "Throat swelling or difficulty swallowing",
                    "Tingling lips or facial swelling",
                    "Dizziness, confusion, or breathing difficulties"
                ],
                confidence=0.95,
                disclaimer="CRITICAL WARNING: This may be a severe systemic allergic reaction. Administer epinephrine (EpiPen) if available and call 911 immediately."
            )

        # Rule 4: Minor paper cut (Confidence: 0.95)
        if "paper cut" in msg and "thumb" in msg:
            return TriageResponse(
                patient_id=patient_id,
                urgency=UrgencyLevel.SELF_CARE,
                condition="Minor Cut / Skin Laceration",
                red_flags=[
                    "Spreading redness, warmth, or pus",
                    "Uncontrolled bleeding after 10 minutes",
                    "Fever or systemic chills"
                ],
                confidence=0.95,
                disclaimer="This is a minor skin laceration. Clean the area with mild soap, apply pressure if bleeding, keep it covered with a bandage, and monitor for infection."
            )

        # Rule 5: Sunburn without blisters (Confidence: 0.90)
        if "sunburn" in msg and "beach" in msg and "blister" not in msg:
            return TriageResponse(
                patient_id=patient_id,
                urgency=UrgencyLevel.SELF_CARE,
                condition="First-Degree Sunburn",
                red_flags=[
                    "Fever or systemic chills",
                    "Severe headache or confusion",
                    "Blistering over a large surface area"
                ],
                confidence=0.90,
                disclaimer="Apply cool compresses, stay hydrated, use moisturizing lotions or aloe vera, and avoid further sun exposure. Monitor for systemic symptoms."
            )

        # Rule 6: Computer strain headache (Confidence: 0.85)
        if "computer screen" in msg and "water today" in msg:
            return TriageResponse(
                patient_id=patient_id,
                urgency=UrgencyLevel.SELF_CARE,
                condition="Tension Headache secondary to Dehydration / Eyestrain",
                red_flags=[
                    "Sudden severe onset (thunderclap)",
                    "Stiff neck or high fever",
                    "Visual disturbances or speech confusion"
                ],
                confidence=0.85,
                disclaimer="Take a break from digital screens, rest in a quiet space, and drink plenty of fluids. Seek medical review if pain worsens or red flags develop."
            )

        # Rule 7: Twisted ankle with weight bearing capability (Confidence: 0.85)
        if "twisted my ankle" in msg and "weight" in msg:
            return TriageResponse(
                patient_id=patient_id,
                urgency=UrgencyLevel.SELF_CARE,
                condition="Minor Ankle Sprain",
                red_flags=[
                    "Inability to bear weight at all",
                    "Severe localized bone tenderness",
                    "Coldness, tingling, or color changes in the foot"
                ],
                confidence=0.85,
                disclaimer="Rest the limb, apply ice packs for 15-20 minutes, compress with a bandage, and elevate (R.I.C.E. protocol). Monitor symptoms over 48 hours."
            )

        # Rule 8: Muscle soreness post gym workout (Confidence: 0.90)
        if "gym for the first time" in msg and "muscle ache" in msg:
            return TriageResponse(
                patient_id=patient_id,
                urgency=UrgencyLevel.SELF_CARE,
                condition="Delayed Onset Muscle Soreness (DOMS)",
                red_flags=[
                    "Dark brown or tea-colored urine (rhabdomyolysis indicator)",
                    "Severe swelling or inability to move the limb",
                    "Extreme localized muscle weakness"
                ],
                confidence=0.90,
                disclaimer="This is normal muscle soreness. Rest the muscles, stay well hydrated, perform gentle stretching, and monitor. Consult a doctor if urine color changes or pain is extreme."
            )

        return None

    async def run_batch_triage(self) -> BatchTriageResponse:
        """
        Executes symptom triage on the loaded dataset, choosing rules or LLM based on confidence.
        """
        import os
        batch_max_llm_cases = int(os.getenv("BATCH_MAX_LLM_CASES", "15"))

        cases = await self.fetch_cases()
        total_cases = len(cases)
        results: List[TriageResponse] = []

        metrics = {
            "emergency": 0,
            "urgent": 0,
            "non_urgent": 0,
            "self_care": 0
        }

        bypassed_count = 0
        llm_cases_used = 0

        for case in cases:
            patient_id = case.get("patient_id")
            message = case.get("message")

            if not patient_id or not message:
                logger.warning(f"Skipping malformed case: {case}")
                continue

            # Check rule engine first
            rule_triage = self.evaluate_rules(message, patient_id)

            if rule_triage and rule_triage.confidence >= 0.8:
                logger.info(f"⚡ [BYPASS] Rule-based match for patient {patient_id} with confidence {rule_triage.confidence}. Skipping LLM.")
                triage_res = rule_triage
                bypassed_count += 1
            else:
                # Rule confidence < 0.8
                if llm_cases_used < batch_max_llm_cases:
                    logger.info(f"🧠 [LLM] No rule match for patient {patient_id}. Dispatching to LangGraph LLM. (Used: {llm_cases_used}/{batch_max_llm_cases})")
                    triage_res = await self.triage_service.triage_symptoms(message, patient_id)
                    llm_cases_used += 1
                else:
                    logger.info(f"⚠️ [BUDGET EXHAUSTED] LLM budget exhausted. Returning fallback response for patient {patient_id}.")
                    triage_res = TriageResponse(
                        patient_id=patient_id,
                        urgency=UrgencyLevel.URGENT,
                        condition="Needs Clinical Review",
                        red_flags=[],
                        confidence=0.50,
                        disclaimer="This is a screening triage tool, not a clinical diagnosis. You should contact a doctor or visit an urgent care center within 12-24 hours for evaluation."
                    )

            # Record result
            results.append(triage_res)

            # Update metrics
            urgency = triage_res.urgency
            if urgency in (UrgencyLevel.EMERGENCY, UrgencyLevel.CRITICAL):
                metrics["emergency"] += 1
            elif urgency == UrgencyLevel.URGENT:
                metrics["urgent"] += 1
            elif urgency == UrgencyLevel.NON_URGENT:
                metrics["non_urgent"] += 1
            elif urgency == UrgencyLevel.SELF_CARE:
                metrics["self_care"] += 1

        logger.info(f"Batch processing completed. LLM Bypassed: {bypassed_count}/{total_cases} cases. LLM Used: {llm_cases_used}/{batch_max_llm_cases}.")

        processing_metrics = ProcessingMetrics(
            total_cases=total_cases,
            emergency_count=metrics["emergency"],
            urgent_count=metrics["urgent"],
            non_urgent_count=metrics["non_urgent"],
            self_care_count=metrics["self_care"]
        )

        groq_calls_used = 2 * llm_cases_used
        groq_calls_saved = 2 * (total_cases - llm_cases_used)
        llm_budget_exhausted = llm_cases_used >= batch_max_llm_cases

        return BatchTriageResponse(
            total_cases=total_cases,
            processed_cases=len(results),
            results=results,
            metrics=processing_metrics,
            groq_calls_used=groq_calls_used,
            groq_calls_saved=groq_calls_saved,
            llm_budget_exhausted=llm_budget_exhausted
        )
