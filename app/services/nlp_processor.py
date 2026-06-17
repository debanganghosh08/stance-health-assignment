import re
from typing import List, Optional
from pydantic import BaseModel

class SymptomFeatures(BaseModel):
    symptoms: List[str]
    severity: Optional[str] = None
    duration: Optional[str] = None
    body_parts: List[str]
    emergency_indicators: List[str]
    age_reference: Optional[str] = None

# Predefined keywords for local matching
SYMPTOMS_LIST = [
    "chest pain", "headache", "fever", "cough", "dizziness",
    "vomiting", "nausea", "shortness of breath", "seizure",
    "bleeding", "abdominal pain", "fatigue"
]

SEVERITIES = ["mild", "moderate", "severe", "unbearable"]

BODY_PARTS = ["chest", "head", "stomach", "abdomen", "arm", "leg", "neck", "back"]

EMERGENCY_INDICATORS = [
    "loss of consciousness", "unable to breathe", "severe bleeding",
    "seizure", "stroke symptoms", "slurred speech"
]

# Regex patterns
DURATION_PATTERN = re.compile(
    r'\b(?:\d+|several|a few|a|one|two|three|four|five|six|seven)\s*(?:hour|day|week|month|year)s?\b',
    re.IGNORECASE
)

AGE_PATTERN = re.compile(
    r'\b(?:\d+\s*(?:years?\s*old|yo|y/o|yo|year-old)|child|infant|elderly|adult|baby|teenager|toddler)\b',
    re.IGNORECASE
)

def extract_features(message: str) -> SymptomFeatures:
    """
    Extracts clinical features from a raw symptom description using regex and keywords.
    """
    msg_lower = message.lower()
    
    # Extract symptoms
    extracted_symptoms = []
    for symptom in SYMPTOMS_LIST:
        if symptom in msg_lower:
            extracted_symptoms.append(symptom)
        elif symptom == "fever" and ("temperature" in msg_lower or "feverish" in msg_lower):
            extracted_symptoms.append("fever")
        elif symptom == "abdominal pain" and ("stomach" in msg_lower or "abdomen" in msg_lower or "belly" in msg_lower):
            extracted_symptoms.append("abdominal pain")
        elif symptom == "shortness of breath" and ("trouble breathing" in msg_lower or "hard to breathe" in msg_lower or "difficulty breathing" in msg_lower):
            extracted_symptoms.append("shortness of breath")
            
    # Remove duplicate/overlapping symptom definitions (e.g. if we have both headache and ache)
    # Deduplicate
    extracted_symptoms = list(dict.fromkeys(extracted_symptoms))

    # Extract severity
    extracted_severity = None
    for sev in SEVERITIES:
        if re.search(r'\b' + re.escape(sev) + r'\b', msg_lower):
            extracted_severity = sev
            break

    # Extract duration
    duration_match = DURATION_PATTERN.search(message)
    extracted_duration = duration_match.group(0) if duration_match else None

    # Extract body parts
    extracted_body_parts = []
    for bp in BODY_PARTS:
        if re.search(r'\b' + re.escape(bp) + r'\b', msg_lower):
            extracted_body_parts.append(bp)

    # Extract emergency indicators
    extracted_emergencies = []
    for em in EMERGENCY_INDICATORS:
        if em in msg_lower:
            extracted_emergencies.append(em)
        elif em == "unable to breathe" and ("can't breathe" in msg_lower or "cannot breathe" in msg_lower or "unable to breathe" in msg_lower):
            extracted_emergencies.append("unable to breathe")
        elif em == "severe bleeding" and ("bleeding heavily" in msg_lower or "blood gushing" in msg_lower):
            extracted_emergencies.append("severe bleeding")
        elif em == "stroke symptoms" and ("face droop" in msg_lower or "drooping face" in msg_lower or "weakness on one side" in msg_lower):
            extracted_emergencies.append("stroke symptoms")
            
    # Extract age reference
    age_match = AGE_PATTERN.search(message)
    extracted_age = age_match.group(0) if age_match else None

    return SymptomFeatures(
        symptoms=extracted_symptoms,
        severity=extracted_severity,
        duration=extracted_duration,
        body_parts=extracted_body_parts,
        emergency_indicators=extracted_emergencies,
        age_reference=extracted_age
    )
