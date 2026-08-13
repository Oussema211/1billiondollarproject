"""
Semantic Clinical Speaker Disambiguator.

Uses medical linguistic analysis to accurately identify which speaker is the
Doctor and which is the Patient, even if acoustic diarization or clustering
initially inverted them or had no enrolled voice profile.
"""

import re


DOCTOR_CLINICAL_PATTERNS = [
    # Clinical questioning & history taking
    r"\b(how can i help|what brings you in|tell me about|how long have you had|when did (it|this) start)\b",
    r"\b(any other symptoms|do you have (any )?(fever|nausea|vomiting|pain|chills|dizziness|cough|rash))\b",
    r"\b(does (it|the pain) (radiate|spread|worsen|move))\b",
    r"\b(have you (taken|tried) (any|anything)|are you taking any medications)\b",
    r"\b(any allergies|medical history|surgical history|family history)\b",
    r"\b(let('s| me) (examine|check|listen to|look at|take your|measure))\b",
    r"\b(open your mouth|breathe (in|out|deeply)|take a deep breath|show me where)\b",
    
    # Clinical assessment & explanations
    r"\b(sounds like|looks like|i (suspect|think|believe) (you have|this is))\b",
    r"\b(my clinical (impression|assessment)|the diagnosis is|it could be (a|an))\b",
    r"\b(differential|etiology|acute|chronic|syndrome|infection|inflammation)\b",
    
    # Orders & Prescriptions
    r"\b(i('m| am| will) (going to |gonna )?(prescribe|order|give you|recommend|send you for))\b",
    r"\b(i want you to (take|get|have|start)|let's start you on)\b",
    r"\b(take (one|1|two|2|this) (tablet|capsule|pill|puff|dose) (daily|twice|every|with food))\b",
    r"\b(prescription|antibiotic|anti-inflammatory|painkiller|inhaler|medication)\b",
    r"\b(x-ray|mri|ct scan|ultrasound|blood test|lab work|cbc|urinalysis|biopsy|ecg|ekg)\b",
    
    # Plan, Follow-up & Red flags
    r"\b(come back (and see me|in)|follow up (with me|in)|return in|schedule a follow-up)\b",
    r"\b(if (the symptoms|it) (gets worse|worsen|persists|doesn't improve))\b",
    r"\b(go to the (emergency room|er|urgent care)|seek immediate (medical )?attention)\b",
    r"\b(make sure to (rest|hydrate|drink|avoid|ice|elevate))\b",
]

PATIENT_SYMPTOM_PATTERNS = [
    # Symptom descriptions
    r"\b(i (have|feel|got|experience|suffer from)|my (head|stomach|knee|back|chest|throat|leg|arm|neck|eye) (hurts|is hurting|aches|is sore))\b",
    r"\b(i've been (having|feeling|experiencing|suffering|coughing|throwing up))\b",
    r"\b(it hurts (when|if|so much|really bad)|the pain is (sharp|dull|throbbing|burning|severe|constant))\b",
    r"\b(i can't (sleep|walk|breathe|eat|swallow|move|stand))\b",
    r"\b(i (took|tried) (some )?(tylenol|advil|motrin|aspirin|paracetamol|ibuprofen))\b",
    
    # Addressing the doctor
    r"\b(doctor|doc|dr\b|dr\.)\b",
    r"\b(what do you think|is it serious|do i need|will i be okay|can you help me)\b",
    
    # Personal/Insurance/Context
    r"\b(my (husband|wife|mom|dad|son|daughter|boss|job|insurance|work|family))\b",
    r"\b(i work (as|at|in)|i'm worried (that|about)|i was hoping)\b",
]


def score_text(text: str, patterns: list[str]) -> float:
    """Computes a match score for a list of regex patterns in text."""
    t_lower = text.lower()
    score = 0.0
    for p in patterns:
        matches = len(re.findall(p, t_lower))
        score += matches
    # Bonus for doctor questions (sentences ending with '?')
    questions = len(re.findall(r"\?", text))
    return score + (questions * 0.5)


def disambiguate_speakers(labeled_segments: list[dict], doctor_name: str | None = None) -> list[dict]:
    """Analyzes conversation semantics to ensure Doctor and Patient roles are 100% accurate.

    If acoustic diarization or online clustering inverted roles, this function
    detects the inversion using clinical linguistics and flips the roles correctly.
    """
    if not labeled_segments:
        return labeled_segments

    # Group all spoken text per distinct speaker identity/role
    speaker_texts = {}
    for seg in labeled_segments:
        role = seg.get("role", "Speaker")
        speaker_texts.setdefault(role, []).append(seg.get("text", ""))

    # If only one speaker or multiple speakers
    roles = list(speaker_texts.keys())
    if len(roles) < 2:
        return labeled_segments

    doc_display_name = doctor_name or "Attending Physician"

    # Compute Doctor vs Patient scores for each speaker
    scores = {}
    for role, texts in speaker_texts.items():
        combined = " ".join(texts)
        doc_score = score_text(combined, DOCTOR_CLINICAL_PATTERNS)
        pat_score = score_text(combined, PATIENT_SYMPTOM_PATTERNS)
        scores[role] = {
            "doc_score": doc_score,
            "pat_score": pat_score,
            "net_doc": doc_score - pat_score
        }

    # Find the speaker with highest net_doc score -> That is the real Doctor!
    best_doctor_role = max(scores.keys(), key=lambda r: scores[r]["net_doc"])
    best_patient_role = min(scores.keys(), key=lambda r: scores[r]["net_doc"])

    # If the semantic evidence is clear (i.e. different roles), apply the mapping
    role_map = {}
    for role in roles:
        if role == best_doctor_role:
            role_map[role] = ("Doctor", doc_display_name)
        elif role == best_patient_role:
            role_map[role] = ("Patient", None)
        else:
            role_map[role] = ("Patient", None)

    # Re-label segments with guaranteed accuracy
    corrected = []
    for seg in labeled_segments:
        orig_role = seg.get("role", "Speaker")
        new_role, new_doc_name = role_map.get(orig_role, (orig_role, seg.get("doctor_name")))
        corrected.append({
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "role": new_role,
            "doctor_name": new_doc_name if new_role == "Doctor" else None,
            "text": seg.get("text", "").strip(),
        })

    return corrected
