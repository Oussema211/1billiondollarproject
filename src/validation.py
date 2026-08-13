"""Phase 1 automated validation layer.

Runs after ReportGenerator._repair_json() produces a parsed report dict and
before that report reaches the GUI/export path. Purely advisory — it never
raises, blocks, or modifies report content. Every issue it finds becomes a
structured flag for a human to review (Phase 2 consumes these; this phase
does not display them anywhere).

Severity taxonomy:
  "error"   — structurally broken: required top-level key missing, wrong
              type, a prescriptions item that isn't an object, or a
              prescriptions item with no (or empty) medication.
  "warning" — soft/content issue: empty string where the schema expects
              null, a prescriptions item missing dosage or instructions, or
              a groundedness mismatch (heuristic, not a certainty).
"""
import difflib
import re

SCHEMA_REQUIRED_KEYS = [
    "chief_complaint", "history_notes", "examination_findings", "diagnosis",
    "tests_ordered", "prescriptions", "follow_up", "other_instructions",
]

# Fields that must be a string or null (i.e. not a list/dict/number).
NULLABLE_STRING_FIELDS = [
    "chief_complaint", "history_notes", "examination_findings", "diagnosis",
    "follow_up", "other_instructions",
]

PRESCRIPTION_REQUIRED_FIELDS = ["medication", "dosage", "instructions"]

# Untuned — see Phase 1 report-back for the deferred-decision note.
FUZZY_MATCH_THRESHOLD = 0.8


def _flag(field: str, issue: str, severity: str) -> dict:
    return {"type": "schema", "field": field, "issue": issue, "severity": severity}


def validate_schema(report: dict) -> list:
    """Check the parsed report dict against the shape ReportGenerator's
    SYSTEM_PROMPT asks the LLM for. Never raises — malformed input just
    produces flags.
    """
    flags = []

    for key in SCHEMA_REQUIRED_KEYS:
        if key not in report:
            flags.append(_flag(key, "required field missing", "error"))

    for key in NULLABLE_STRING_FIELDS:
        if key not in report:
            continue
        val = report[key]
        if val is None:
            continue
        if not isinstance(val, str):
            flags.append(_flag(key, f"expected string or null, got {type(val).__name__}", "error"))
        elif val.strip() == "":
            flags.append(_flag(key, "empty string returned instead of null", "warning"))

    for key in ("tests_ordered", "prescriptions"):
        if key not in report:
            continue
        val = report[key]
        if not isinstance(val, list):
            flags.append(_flag(key, f"expected a list, got {type(val).__name__}", "error"))

    prescriptions = report.get("prescriptions")
    if isinstance(prescriptions, list):
        for i, item in enumerate(prescriptions):
            prefix = f"prescriptions[{i}]"
            if not isinstance(item, dict):
                flags.append(_flag(prefix, "prescription item is not an object", "error"))
                continue
            for field in PRESCRIPTION_REQUIRED_FIELDS:
                val = item.get(field)
                if not (isinstance(val, str) and val.strip()):
                    severity = "error" if field == "medication" else "warning"
                    flags.append(_flag(f"{prefix}.{field}", "missing or empty", severity))

    return flags


def _normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", s.lower()).strip()


def _is_grounded(term: str, transcript_norm: str, threshold: float = FUZZY_MATCH_THRESHOLD) -> bool:
    """True if `term` has a reasonable match somewhere in the transcript.

    Fast path: normalized substring match. Fallback: slide a word-window of
    the same length as `term` across the transcript and compare similarity
    with difflib — catches paraphrases/near-misses without any NLP
    dependency. Heuristic, not a certainty (by design).
    """
    term_norm = _normalize_text(term)
    if not term_norm:
        return True  # nothing to check

    if term_norm in transcript_norm:
        return True

    words = transcript_norm.split()
    term_words = term_norm.split()
    window = max(len(term_words), 1)
    if not words:
        return False

    best_ratio = 0.0
    for i in range(len(words) - window + 1):
        chunk = " ".join(words[i:i + window])
        ratio = difflib.SequenceMatcher(None, term_norm, chunk).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
        if best_ratio >= threshold:
            return True
    return best_ratio >= threshold


def _groundedness_flag(field: str, value: str) -> dict:
    return {
        "type": "groundedness",
        "field": field,
        "value": value,
        "issue": "not found in transcript, review recommended",
        "severity": "warning",
    }


def check_groundedness(report: dict, transcript_text: str) -> list:
    """For each prescription medication and the diagnosis, check whether it
    has a reasonable match in the source transcript text. Flags possible
    hallucinations — a heuristic, not a certainty. Never auto-corrects.
    """
    flags = []
    transcript_norm = _normalize_text(transcript_text or "")

    diagnosis = report.get("diagnosis")
    if isinstance(diagnosis, str) and diagnosis.strip():
        if not _is_grounded(diagnosis, transcript_norm):
            flags.append(_groundedness_flag("diagnosis", diagnosis))

    prescriptions = report.get("prescriptions")
    if isinstance(prescriptions, list):
        for i, item in enumerate(prescriptions):
            if not isinstance(item, dict):
                continue
            med = item.get("medication")
            if isinstance(med, str) and med.strip():
                if not _is_grounded(med, transcript_norm):
                    flags.append(_groundedness_flag(f"prescriptions[{i}].medication", med))

    return flags
