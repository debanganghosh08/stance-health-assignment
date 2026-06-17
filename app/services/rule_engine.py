from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.services.nlp_processor import SymptomFeatures

class RuleEngineResult(BaseModel):
    matched_rule: Optional[str] = None
    urgency: str
    condition: str
    confidence: float
    red_flags: List[str]

# Standard medical rule configurations
SYMPTOM_RULES: Dict[str, Dict[str, Any]] = {
    "chest pain": {
        "urgency": "Emergency",
        "condition": "Potential Acute Coronary Syndrome (Heart Attack)",
        "confidence": 0.95,
        "red_flags": ["Pain radiating to left arm, neck, or jaw", "Profuse sweating", "Shortness of breath"]
    },
    "seizure": {
        "urgency": "Emergency",
        "condition": "Acute Seizure Activity",
        "confidence": 0.95,
        "red_flags": ["Loss of consciousness", "Biting tongue or cheek", "Incontinence or post-ictal confusion"]
    },
    "stroke": {
        "urgency": "Emergency",
        "condition": "Potential Acute Stroke (Brain Ischemia)",
        "confidence": 0.98,
        "red_flags": ["Face drooping on one side", "Unilateral arm weakness", "Slurred or incoherent speech"]
    },
    "shortness of breath": {
        "urgency": "Emergency",
        "condition": "Acute Respiratory Distress",
        "confidence": 0.95,
        "red_flags": ["Inability to speak in full sentences", "Blue lips or fingernails", "Accessory muscle use"]
    },
    "severe bleeding": {
        "urgency": "Emergency",
        "condition": "Active Severe Hemorrhage",
        "confidence": 0.95,
        "red_flags": ["Uncontrolled bleeding after 10 minutes", "Dizziness, cold sweat, or confusion"]
    },
    "abdominal pain": {
        "urgency": "Urgent",
        "condition": "Acute Abdominal Inflammation",
        "confidence": 0.85,
        "red_flags": ["Severe right lower quadrant tenderness", "Rigid or swollen abdomen", "High fever"]
    },
    "headache": {
        "urgency": "Non-Urgent",
        "condition": "Tension-type Headache",
        "confidence": 0.85,
        "red_flags": ["Sudden thunderclap onset", "Stiff neck or fever", "Visual/neurological changes"]
    },
    "fever": {
        "urgency": "Non-Urgent",
        "condition": "Acute Febrile Illness",
        "confidence": 0.85,
        "red_flags": ["Temperature above 103°F", "Stiff neck", "Confusion or lethargy"]
    },
    "cough": {
        "urgency": "Self-Care",
        "condition": "Mild Upper Respiratory Tract Infection",
        "confidence": 0.85,
        "red_flags": ["Shortness of breath", "Hemoptysis (coughing blood)", "Stridor or high fever"]
    },
    "fatigue": {
        "urgency": "Self-Care",
        "condition": "Mild General Fatigue",
        "confidence": 0.85,
        "red_flags": ["Chest pain or shortness of breath", "Sudden severe weakness"]
    },
    "nausea": {
        "urgency": "Self-Care",
        "condition": "Mild Gastrointestinal Distress",
        "confidence": 0.85,
        "red_flags": ["Persistent vomiting > 24h", "Inability to keep liquids down", "Severe abdominal pain"]
    }
}

def evaluate_rules(features: SymptomFeatures) -> Optional[RuleEngineResult]:
    """
    Evaluates extracted NLP features against clinical rules.
    If a matched rule's confidence is >= local triage threshold, returns a RuleEngineResult.
    """
    # 1. Prioritize critical emergency indicators
    if features.emergency_indicators:
        for indicator in features.emergency_indicators:
            if "seizure" in indicator:
                rule = SYMPTOM_RULES["seizure"]
                return RuleEngineResult(matched_rule="seizure", **rule)
            if "stroke" in indicator or "slurred speech" in indicator:
                rule = SYMPTOM_RULES["stroke"]
                return RuleEngineResult(matched_rule="stroke", **rule)
            if "unable to breathe" in indicator or "shortness of breath" in indicator:
                rule = SYMPTOM_RULES["shortness of breath"]
                return RuleEngineResult(matched_rule="shortness of breath", **rule)
            if "severe bleeding" in indicator:
                rule = SYMPTOM_RULES["severe bleeding"]
                return RuleEngineResult(matched_rule="severe bleeding", **rule)
            if "loss of consciousness" in indicator:
                rule = SYMPTOM_RULES["seizure"]  # Map general loss of consciousness to emergency seizure/neurological template
                return RuleEngineResult(matched_rule="loss of consciousness", **rule)

    # 2. Match symptoms in order of clinical priority
    priority_order = [
        "chest pain", "seizure", "stroke", "shortness of breath", "severe bleeding",
        "abdominal pain", "headache", "fever", "cough", "fatigue", "nausea"
    ]

    for symptom in priority_order:
        if symptom in features.symptoms:
            rule_data = SYMPTOM_RULES[symptom]
            urgency = rule_data["urgency"]
            confidence = rule_data["confidence"]
            
            # Elevate severity if "severe" or "unbearable" is noted in the features
            if features.severity in ["severe", "unbearable"]:
                if urgency == "Self-Care":
                    urgency = "Non-Urgent"
                elif urgency == "Non-Urgent":
                    urgency = "Urgent"
                elif urgency == "Urgent":
                    urgency = "Emergency"
                confidence = min(confidence + 0.05, 1.0)

            return RuleEngineResult(
                matched_rule=symptom,
                urgency=urgency,
                condition=rule_data["condition"],
                confidence=round(confidence, 2),
                red_flags=rule_data["red_flags"]
            )

    return None
