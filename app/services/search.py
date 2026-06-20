import logging
from typing import Dict

logger = logging.getLogger("app.services.search")

# Local Mock Medical Knowledge Base containing standard triage guidance
# to simulate an external medical index query (e.g. PubMed/UpToDate)
MEDICAL_KNOWLEDGE_BASE: Dict[str, str] = {
    "chest pain": (
        "CLINICAL REFERENCE: Acute Chest Pain\n"
        "- Potential conditions: Myocardial infarction (heart attack), angina, pulmonary embolism, aortic dissection, GERD, panic attack.\n"
        "- Red flags: Pain radiating to left arm, shoulder, neck or jaw; chest pressure or squeezing; shortness of breath; profuse sweating (diaphoresis); lightheadedness.\n"
        "- Triage action: Immediate emergency service dispatch (911) required. Do not drive to the ER."
    ),
    "breathing": (
        "CLINICAL REFERENCE: Dyspnea (Shortness of Breath)\n"
        "- Potential conditions: Asthma exacerbation, anaphylaxis, pneumonia, heart failure, COPD flare.\n"
        "- Red flags: Inability to speak in full sentences, blue lips or fingernails (cyanosis), stridor or severe wheezing, accessory muscle use.\n"
        "- Triage action: Critical urgency. Emergency room evaluation recommended immediately."
    ),
    "stroke": (
        "CLINICAL REFERENCE: Acute Neurological Deficits / Stroke (FAST)\n"
        "- Potential conditions: Ischemic stroke, hemorrhagic stroke, transient ischemic attack (TIA).\n"
        "- Red flags: Face drooping, arm weakness, speech difficulty (slurred or incoherent), sudden vision changes, loss of balance.\n"
        "- Triage action: Critical emergency. Record time of onset and call 911 immediately."
    ),
    "fever": (
        "CLINICAL REFERENCE: Pyrexia (Fever) in Adults\n"
        "- Potential conditions: Viral syndrome, bacterial infection (UTI, pneumonia, strep throat), inflammatory response.\n"
        "- Red flags: Temperature > 103°F (39.4°C) unresponsive to antipyretics, stiff neck, severe headache, confusion, rash, photophobia.\n"
        "- Triage action: Urgent if red flags are present or if fever persists > 3 days. Otherwise, non-urgent or self-care."
    ),
    "headache": (
        "CLINICAL REFERENCE: Cephalea (Headache)\n"
        "- Potential conditions: Tension headache, migraine, dehydration, sinus congestion, subarachnoid hemorrhage (thunderclap headache).\n"
        "- Red flags: Sudden onset 'worst headache of life', fever, stiff neck, confusion, seizure, double vision, headache following head trauma.\n"
        "- Triage action: Critical if sudden/severe with stiff neck. Otherwise, non-urgent or self-care."
    ),
    "abdominal": (
        "CLINICAL REFERENCE: Acute Abdominal Pain\n"
        "- Potential conditions: Appendicitis, cholecystitis, gastroenteritis, bowel obstruction, kidney stones.\n"
        "- Red flags: Severe pain localizing to the Right Lower Quadrant, rigid abdomen, high fever, blood in vomit or stool, inability to keep down liquids.\n"
        "- Triage action: Urgent to Critical depending on severity. Requires prompt physician evaluation."
    ),
    "stomach": (
        "CLINICAL REFERENCE: Gastrointestinal Distress / Stomach Pain\n"
        "- Potential conditions: Gastroenteritis, gastritis, food poisoning, GERD.\n"
        "- Red flags: Persistent vomiting for over 24 hours, blood in stool, severe sudden sharp pain.\n"
        "- Triage action: Self-care if mild and vomiting is controlled. Urgent if severe localized pain or dehydration signs occur."
    ),
    "throat": (
        "CLINICAL REFERENCE: Sore Throat / Pharyngitis\n"
        "- Potential conditions: Viral pharyngitis (cold), streptococcal pharyngitis (strep throat), tonsillitis.\n"
        "- Red flags: Difficulty swallowing saliva (drooling), inability to open mouth (trismus), stridor/difficulty breathing.\n"
        "- Triage action: Urgent if difficulty swallowing or breathing. Otherwise, self-care with hydration and lozenges."
    )
}

class SearchService:
    """
    Simulates a clinical search engine querying a medical reference database.
    Can be replaced in production with Tavily API, PubMed search, or a Vector DB index of clinical documents.
    """
    async def search(self, query: str) -> str:
        if not query:
            return "No search query formulated. Use general clinical evaluation standards."

        normalized_query = query.lower()
        logger.info(f"🔍 Searching medical reference database for: '{normalized_query}'")

        matched_references = []
        for keyword, clinical_note in MEDICAL_KNOWLEDGE_BASE.items():
            if keyword in normalized_query or any(word in keyword for word in normalized_query.split() if len(word) > 3):
                matched_references.append(clinical_note)

        if not matched_references:
            # Fallback clinical guidance if query doesn't match specific keys
            logger.info("⚠️ No specific clinical matches. Returning general triage reference guidelines.")
            return (
                "CLINICAL REFERENCE: General Symptom Triage Guidelines\n"
                "- Assess basic vital signs: breathing rate, conscious status, and skin temperature/color.\n"
                "- Monitor for systemic signs of infection (fever, chills, weakness) or acute tissue ischemia (severe sharp localized pain).\n"
                "- Instruct patient: if symptoms worsen rapidly or new neurological/cardiopulmonary symptoms develop, seek emergency care immediately."
            )

        return "\n\n---\n\n".join(matched_references)

_search_service = None

def get_search_service() -> SearchService:
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service
