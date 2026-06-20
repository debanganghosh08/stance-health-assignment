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
    numeric_age: Optional[int] = None
    age_category: Optional[str] = None  # "infant", "child", "elderly", "adult"

# Predefined keywords for local matching
SYMPTOMS_LIST = [
    "chest pain", "headache", "fever", "cough", "dizziness",
    "vomiting", "nausea", "shortness of breath", "seizure",
    "bleeding", "abdominal pain", "fatigue",
    "pregnancy", "fainting", "blurred vision", "confusion",
    "sweating", "stiff neck", "weakness", "facial droop",
    "slurred speech", "arm pain", "palpitations", "dysuria",
    "black stool", "vomiting blood", "severe dehydration",
    "cyanosis", "blue lips", "low oxygen", "oxygen saturation"
]

SYNONYMS = {
    # Heart Attack / Chest Pain synonyms
    "chest tightness": "chest pain",
    "chest pressure": "chest pain",
    "crushing chest pain": "chest pain",
    "tight chest": "chest pain",
    "pressure in chest": "chest pain",
    "heavy chest": "chest pain",
    "heart attack": "chest pain",
    "chest pain": "chest pain",

    # Stroke synonyms
    "face feels numb": "facial droop",
    "one side numb": "weakness",
    "cannot lift arm": "weakness",
    "cant lift arm": "weakness",
    "numbness on one side": "weakness",
    "numbness in face": "facial droop",
    "numb face": "facial droop",
    "slurred": "slurred speech",
    "slurred speech": "slurred speech",
    "difficulty talking": "slurred speech",
    "trouble talking": "slurred speech",
    "hard to talk": "slurred speech",
    "difficulty speaking": "slurred speech",
    "trouble speaking": "slurred speech",
    "can't move one side": "weakness",
    "cannot move one side": "weakness",
    "weakness on one side": "weakness",
    "arm weakness": "weakness",
    "leg weakness": "weakness",
    "feeling weak": "weakness",
    "weak": "weakness",
    "crooked face": "facial droop",
    "face droop": "facial droop",
    "drooping face": "facial droop",

    # Pulmonary Embolism synonyms
    "pain while breathing": "chest pain",
    "sharp pain on breathing": "chest pain",
    "pain when breathing": "chest pain",
    "sharp pain when breathing": "chest pain",
    "pain on breathing": "chest pain",

    # UTI / Dysuria synonyms
    "burning pee": "dysuria",
    "frequent urination": "dysuria",
    "cloudy urine": "dysuria",
    "pee frequently": "dysuria",
    "urinating a lot": "dysuria",
    "frequent pee": "dysuria",
    "pain while peeing": "dysuria",
    "burning urination": "dysuria",
    "burning while peeing": "dysuria",
    "pain when peeing": "dysuria",
    "burning when peeing": "dysuria",

    # Pregnancy / Bleeding synonyms
    "spotting": "bleeding",
    "vaginal bleeding": "bleeding",
    "pregnant": "pregnancy",
    "pregnancy": "pregnancy",
    "expecting": "pregnancy",
    "bleeding": "bleeding",
    "blood": "bleeding",

    # Palpitations
    "heart pounding": "palpitations",
    "pounding heart": "palpitations",
    "racing heart": "palpitations",
    "fluttering heart": "palpitations",
    "heart fluttering": "palpitations",

    # GI Bleed synonyms
    "dark stool": "black stool",
    "black stool": "black stool",
    "tarry stool": "black stool",
    "black tarry stool": "black stool",
    "blood vomiting": "vomiting blood",
    "vomiting blood": "vomiting blood",
    "throw up blood": "vomiting blood",
    "throwing up blood": "vomiting blood",
    "hematemesis": "vomiting blood",

    # Dehydration
    "dehydration": "severe dehydration",
    "severely dehydrated": "severe dehydration",
    "severe dehydration": "severe dehydration",
    "dehydr": "severe dehydration",

    # Oxygen / Cyanosis synonyms
    "cyanosis": "blue lips",
    "blue skin": "blue lips",
    "blue lips": "blue lips",
    "low oxygen": "low oxygen",
    "oxygen saturation": "low oxygen",
    "low o2": "low oxygen",
    "hypoxia": "low oxygen",

    # Miscellaneous general symptoms
    "headache": "headache",
    "fever": "fever",
    "cough": "cough",
    "nausea": "nausea",
    "vomiting": "vomiting",
    "fatigue": "fatigue",
    "dizzy": "dizziness",
    "dizziness": "dizziness",
    "faint": "fainting",
    "fainting": "fainting",
    "blurred vision": "blurred vision",
    "blurry vision": "blurred vision",
    "confusion": "confusion",
    "confused": "confusion",
    "sweating": "sweating",
    "sweaty": "sweating",
    "stiff neck": "stiff neck",
    "arm pain": "arm pain",
    "left arm pain": "arm pain"
}

SEVERITIES = ["mild", "moderate", "severe", "unbearable"]

BODY_PARTS = ["chest", "head", "stomach", "abdomen", "arm", "leg", "neck", "back"]

EMERGENCY_INDICATORS = [
    "loss of consciousness", "unable to breathe", "severe bleeding",
    "seizure", "stroke symptoms", "slurred speech", "weakness"
]

# Regex patterns
DURATION_PATTERN = re.compile(
    r'\b(?:\d+|several|a few|a|one|two|three|four|five|six|seven)\s*(?:hour|day|week|month|year)s?\b',
    re.IGNORECASE
)

AGE_PATTERN = re.compile(
    r'\b(?:\d+[\s-]*(?:years?\s*old|yo|y/o|year-old)|child|infant|elderly|adult|baby|teenager|toddler)\b',
    re.IGNORECASE
)

def parse_age(age_str: str) -> Optional[int]:
    if not age_str:
        return None
    match = re.search(r'\d+', age_str)
    if match:
        return int(match.group(0))
    return None

def extract_features(message: str) -> SymptomFeatures:
    """
    Extracts clinical features from a raw symptom description using regex and keywords.
    """
    msg_lower = message.lower()
    
    # Extract symptoms using Synonym Normalization Map first
    extracted_symptoms = []
    for phrase, standard in SYNONYMS.items():
        if phrase in msg_lower:
            extracted_symptoms.append(standard)
            
    # Also scan predefined keywords
    for symptom in SYMPTOMS_LIST:
        if symptom in msg_lower:
            extracted_symptoms.append(symptom)
        elif symptom == "fever" and ("temperature" in msg_lower or "feverish" in msg_lower or "high temp" in msg_lower):
            extracted_symptoms.append("fever")
        elif symptom == "abdominal pain" and ("stomach" in msg_lower or "abdomen" in msg_lower or "belly" in msg_lower or "abdominal" in msg_lower):
            extracted_symptoms.append("abdominal pain")
        elif symptom == "shortness of breath" and ("trouble breathing" in msg_lower or "hard to breathe" in msg_lower or "difficulty breathing" in msg_lower or "can't breathe" in msg_lower):
            extracted_symptoms.append("shortness of breath")
        elif symptom == "stiff neck" and ("neck" in msg_lower and "stiff" in msg_lower):
            extracted_symptoms.append("stiff neck")
        elif symptom == "weakness" and ("weak" in msg_lower):
            extracted_symptoms.append("weakness")
            
    # Deduplicate symptoms
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
            
    # Extract age reference, numeric age and age category
    extracted_age = None
    numeric_age = None
    age_category = None
    
    age_match = AGE_PATTERN.search(message)
    if age_match:
        extracted_age = age_match.group(0)
        numeric = parse_age(extracted_age)
        if numeric is not None:
            numeric_age = numeric
            if numeric < 5:
                age_category = "child"
            elif numeric > 65:
                age_category = "elderly"
            else:
                age_category = "adult"
        else:
            age_lower = extracted_age.lower()
            if any(x in age_lower for x in ["infant", "baby", "toddler"]):
                age_category = "infant"
            elif "child" in age_lower:
                age_category = "child"
            elif "elderly" in age_lower:
                age_category = "elderly"

    return SymptomFeatures(
        symptoms=extracted_symptoms,
        severity=extracted_severity,
        duration=extracted_duration,
        body_parts=extracted_body_parts,
        emergency_indicators=extracted_emergencies,
        age_reference=extracted_age,
        numeric_age=numeric_age,
        age_category=age_category
    )
