from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.services.nlp_processor import SymptomFeatures

class RuleEngineResult(BaseModel):
    matched_rule: Optional[str] = None
    urgency: str
    condition: str
    confidence: float
    red_flags: List[str]

# Reusable configuration for high-risk combinations
EMERGENCY_COMBINATION_MATRIX = [
    # CARDIAC
    {
        "symptoms": ["chest pain", "sweating"],
        "urgency": "Emergency",
        "condition": "Possible Myocardial Infarction (Heart Attack)",
        "red_flags": ["Chest pain/pressure radiating to left arm or jaw", "Profuse sweating (diaphoresis)", "Shortness of breath"],
        "confidence": 0.98
    },
    {
        "symptoms": ["chest pain", "arm pain"],
        "urgency": "Emergency",
        "condition": "Possible Myocardial Infarction (Heart Attack)",
        "red_flags": ["Chest pain radiating to left arm or jaw", "Profuse sweating", "Shortness of breath"],
        "confidence": 0.98
    },
    {
        "symptoms": ["chest pain", "shortness of breath"],
        "urgency": "Emergency",
        "condition": "Potential Cardiorespiratory Emergency",
        "red_flags": ["Chest pain radiating to left arm or jaw", "Difficulty breathing"],
        "confidence": 0.95
    },
    {
        "symptoms": ["chest pain", "dizziness"],
        "urgency": "Emergency",
        "condition": "Potential Cardiorespiratory Emergency / Cardiac Risk",
        "red_flags": ["Chest pain radiating to left arm or jaw", "Severe dizziness or fainting"],
        "confidence": 0.95
    },
    # STROKE
    {
        "symptoms": ["weakness", "slurred speech"],
        "urgency": "Emergency",
        "condition": "Possible Acute Stroke (Brain Ischemia)",
        "red_flags": ["Unilateral body weakness or numbness", "Slurred or incoherent speech", "Facial drooping"],
        "confidence": 0.98
    },
    {
        "symptoms": ["facial droop", "weakness"],
        "urgency": "Emergency",
        "condition": "Possible Acute Stroke (Brain Ischemia)",
        "red_flags": ["Unilateral body weakness or numbness", "Slurred or incoherent speech", "Facial drooping"],
        "confidence": 0.98
    },
    {
        "symptoms": ["confusion", "weakness"],
        "urgency": "Emergency",
        "condition": "Possible Acute Stroke (Brain Ischemia)",
        "red_flags": ["Unilateral body weakness or numbness", "Altered mental status or confusion"],
        "confidence": 0.98
    },
    # MENINGITIS
    {
        "symptoms": ["fever", "stiff neck"],
        "urgency": "Emergency",
        "condition": "Possible Meningitis Infection",
        "red_flags": ["Stiff neck with high fever", "Confusion or altered mental status", "Severe headache"],
        "confidence": 0.98
    },
    # GI BLEED
    {
        "symptoms": ["black stool", "dizziness"],
        "urgency": "Emergency",
        "condition": "Possible Gastrointestinal Hemorrhage (GI Bleed)",
        "red_flags": ["Black, tarry, or bloody stools", "Severe dizziness or fainting", "Vomiting blood"],
        "confidence": 0.95
    },
    {
        "symptoms": ["vomiting blood"],
        "urgency": "Emergency",
        "condition": "Possible Upper Gastrointestinal Hemorrhage",
        "red_flags": ["Vomiting blood or coffee-ground material", "Severe dizziness or fainting", "Black tarry stools"],
        "confidence": 0.98
    },
    # ABDOMINAL
    {
        "symptoms": ["abdominal pain", "vomiting"],
        "urgency": "Urgent",
        "condition": "Acute Abdominal Distress",
        "red_flags": ["Severe abdominal pain", "Inability to keep liquids down", "Fever"],
        "confidence": 0.90
    },
    {
        "symptoms": ["abdominal pain", "fever"],
        "urgency": "Urgent",
        "condition": "Acute Abdominal Distress / Possible Appendicitis",
        "red_flags": ["Severe abdominal pain", "High fever", "Persistent vomiting"],
        "confidence": 0.90
    }
]

# Standard medical rule configurations
SYMPTOM_RULES: Dict[str, Dict[str, Any]] = {
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
    Collects all matching rules, ranks them by priority, and returns the highest priority match.
    """
    msg_lower = ""
    if features.symptoms:
        # Fallback query content lowercasing check
        msg_lower = " ".join(features.symptoms).lower()

    matches = []

    # 1. Pregnancy Safety Rules (Task 3)
    if "pregnancy" in features.symptoms:
        pregnancy_triggers = ["bleeding", "abdominal pain", "dizziness", "fainting", "blurred vision", "headache"]
        matched_trigger = None
        for trig in pregnancy_triggers:
            if trig in features.symptoms:
                matched_trigger = trig
                break
        if matched_trigger:
            matches.append(RuleEngineResult(
                matched_rule=f"Pregnancy Safety Escalation ({matched_trigger})",
                urgency="Emergency",
                condition="Possible obstetric emergency",
                confidence=0.98,
                red_flags=[
                    "Vaginal bleeding or fluid leakage during pregnancy",
                    "Severe abdominal pain or cramping",
                    "Severe headache, visual changes, or blurred vision",
                    "Significant dizziness, fainting, or loss of consciousness"
                ]
            ))

    # 2. Sepsis Escalation Rules (Task Upgrade 1)
    if "fever" in features.symptoms:
        if "confusion" in features.symptoms:
            matches.append(RuleEngineResult(
                matched_rule="Sepsis (Fever + Confusion)",
                urgency="Emergency",
                condition="Possible sepsis",
                confidence=0.98,
                red_flags=["High fever with altered mental status", "Stiff neck", "Rapid breathing", "Cold or pale extremities"]
            ))
        if "severe dehydration" in features.symptoms:
            matches.append(RuleEngineResult(
                matched_rule="Sepsis (Fever + Severe Dehydration)",
                urgency="Emergency",
                condition="Possible sepsis",
                confidence=0.98,
                red_flags=["High fever with severe dehydration", "Rapid heart rate", "Decreased urination", "Confusion"]
            ))
        if "dizziness" in features.symptoms and "weakness" in features.symptoms:
            matches.append(RuleEngineResult(
                matched_rule="Sepsis (Fever + Dizziness + Weakness)",
                urgency="Emergency",
                condition="Possible sepsis",
                confidence=0.98,
                red_flags=["High fever with systemic weakness and dizziness", "Low blood pressure indicators", "Rapid pulse"]
            ))

    # 3. Pulmonary Embolism Escalation Rules (Task Upgrade 2)
    if "shortness of breath" in features.symptoms and "chest pain" in features.symptoms:
        matches.append(RuleEngineResult(
            matched_rule="PE (Shortness of Breath + Chest Pain)",
            urgency="Emergency",
            condition="Possible pulmonary embolism",
            confidence=0.98,
            red_flags=["Sharp chest pain worsening on breathing", "Sudden shortness of breath", "Coughing up blood"]
        ))
    if "chest pain" in features.symptoms and "dizziness" in features.symptoms and "shortness of breath" in features.symptoms:
        matches.append(RuleEngineResult(
            matched_rule="PE (Chest Pain + Dizziness + Shortness of Breath)",
            urgency="Emergency",
            condition="Possible pulmonary embolism",
            confidence=0.98,
            red_flags=["Sharp chest pain", "Sudden shortness of breath", "Dizziness or fainting"]
        ))

    # 3.5 Respiratory Oxygen-Related Safety Rules (Task Upgrade 5)
    if "shortness of breath" in features.symptoms:
        if "blue lips" in features.symptoms:
            matches.append(RuleEngineResult(
                matched_rule="Respiratory Risk (Shortness of Breath + Blue Lips)",
                urgency="Emergency",
                condition="Potential Hypoxia / Respiratory Emergency",
                confidence=0.98,
                red_flags=["Shortness of breath with blue/pale lips or skin (cyanosis)", "Low oxygen levels", "Confusion"]
            ))
        if "confusion" in features.symptoms:
            matches.append(RuleEngineResult(
                matched_rule="Respiratory Risk (Shortness of Breath + Confusion)",
                urgency="Emergency",
                condition="Potential Hypoxia / Respiratory Emergency",
                confidence=0.98,
                red_flags=["Shortness of breath with altered mental status", "Lethargy", "Blue/pale lips or skin"]
            ))

    # 4. Age-Aware Safety Escalation Rules (Task Upgrade 3)
    is_pediatric = (features.numeric_age is not None and features.numeric_age < 5) or features.age_category in ["child", "infant"]
    is_elderly = (features.numeric_age is not None and features.numeric_age > 65) or features.age_category == "elderly"
    
    if is_pediatric and "fever" in features.symptoms:
        matches.append(RuleEngineResult(
            matched_rule="Pediatric Fever Escalation",
            urgency="Urgent",
            condition="Febrile Child / Pediatric Fever",
            confidence=0.95,
            red_flags=["Temperature > 100.4°F in child < 5 years old", "Lethargy or irritability", "Poor feeding or hydration"]
        ))
    if is_elderly and "confusion" in features.symptoms:
        matches.append(RuleEngineResult(
            matched_rule="Elderly Confusion Escalation",
            urgency="Emergency",
            condition="Altered Mental Status in Elderly",
            confidence=0.95,
            red_flags=["Acute confusion in elderly patient", "Stiff neck or fever", "New-onset weakness or slurred speech"]
        ))
    if is_pediatric and "severe dehydration" in features.symptoms:
        matches.append(RuleEngineResult(
            matched_rule="Pediatric Dehydration Escalation",
            urgency="Emergency",
            condition="Pediatric Dehydration Emergency",
            confidence=0.98,
            red_flags=["Severe dehydration in a child", "Lethargy or listlessness", "Sunken eyes or dry mouth", "No wet diapers for 8+ hours"]
        ))

    # 5. Combination Matrix Rules (Task 2)
    for entry in EMERGENCY_COMBINATION_MATRIX:
        if all(s in features.symptoms for s in entry["symptoms"]):
            if "abdominal pain" in entry["symptoms"]:
                # Abdominal pain combinations require severe/unbearable severity
                if features.severity not in ["severe", "unbearable"]:
                    continue
            matches.append(RuleEngineResult(
                matched_rule=f"Combination Matrix: {' + '.join(entry['symptoms'])}",
                urgency=entry["urgency"],
                condition=entry["condition"],
                confidence=entry["confidence"],
                red_flags=entry["red_flags"]
            ))

    # 6. Prioritize critical emergency indicators
    if features.emergency_indicators:
        for indicator in features.emergency_indicators:
            if "seizure" in indicator:
                rule = SYMPTOM_RULES["seizure"]
                matches.append(RuleEngineResult(matched_rule="seizure", **rule))
            if "stroke" in indicator or "slurred speech" in indicator:
                rule = SYMPTOM_RULES["stroke"]
                matches.append(RuleEngineResult(matched_rule="stroke", **rule))
            if "unable to breathe" in indicator or "shortness of breath" in indicator:
                rule = SYMPTOM_RULES["shortness of breath"]
                matches.append(RuleEngineResult(matched_rule="shortness of breath", **rule))
            if "severe bleeding" in indicator:
                rule = SYMPTOM_RULES["severe bleeding"]
                matches.append(RuleEngineResult(matched_rule="severe bleeding", **rule))
            if "loss of consciousness" in indicator:
                rule = SYMPTOM_RULES["seizure"]  # Map to emergency seizure/neurological template
                matches.append(RuleEngineResult(matched_rule="loss of consciousness", **rule))

    # 7. Match symptoms in order of clinical priority
    priority_order = [
        "seizure", "stroke", "shortness of breath", "severe bleeding",
        "abdominal pain", "headache", "fever", "cough", "fatigue", "nausea"
    ]

    for symptom in priority_order:
        if symptom in features.symptoms and symptom in SYMPTOM_RULES:
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

            matches.append(RuleEngineResult(
                matched_rule=symptom,
                urgency=urgency,
                condition=rule_data["condition"],
                confidence=round(confidence, 2),
                red_flags=rule_data["red_flags"]
            ))

    if not matches:
        return None

    # Helper function to compute priority dynamically
    def get_rule_priority(matched_rule: str) -> int:
        mr_lower = matched_rule.lower()
        if "stroke" in mr_lower or "facial droop" in mr_lower or "slurred speech" in mr_lower:
            return 100
        if "cardiac" in mr_lower or "myocardial" in mr_lower or "chest pain" in mr_lower:
            return 100
        if "pe" in mr_lower or "pulmonary embolism" in mr_lower or "seizure" in mr_lower:
            return 95
        if "sepsis" in mr_lower or "meningitis" in mr_lower or "elderly confusion" in mr_lower:
            return 90
        if "pregnancy" in mr_lower:
            return 85
        if "gi bleed" in mr_lower or "dehydration" in mr_lower or "hemorrhage" in mr_lower:
            return 85
        if "pediatric fever" in mr_lower or "febrile child" in mr_lower:
            return 70
        
        # Defaults
        for sym, prio in [
            ("shortness of breath", 90), ("severe bleeding", 90),
            ("abdominal pain", 70), ("headache", 60), ("fever", 60)
        ]:
            if sym in mr_lower:
                return prio
        return 50

    # Sort matches by priority descending, keeping first on tie
    matches.sort(key=lambda x: get_rule_priority(x.matched_rule), reverse=True)
    return matches[0]
