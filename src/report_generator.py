import json
import re
from .llm_backend import build_llm_backend
from .validation import validate_schema, check_groundedness

# Used when extract() is called without an explicit speaker_confidence (e.g.
# direct/legacy callers) so report["_review"]["speaker_confidence"] always
# has a consistent shape rather than being absent.
DEFAULT_SPEAKER_CONFIDENCE = {
    "tier": "unresolved",
    "score": None,
    "manually_overridden": False,
    "override_name": None,
}

SYSTEM_PROMPT = """You are a clinical scribe AI. Extract a medical report from this doctor-patient transcript.
Return ONLY valid JSON, no markdown:
{
  "chief_complaint": "string or null",
  "history_notes": "string or null",
  "examination_findings": "string or null",
  "diagnosis": "string or null",
  "tests_ordered": ["string"],
  "prescriptions": [{"medication":"string","dosage":"string or null","instructions":"string or null"}],
  "follow_up": "string or null",
  "other_instructions": "string or null"
}
Rules: Only extract what is explicitly mentioned. Use null for missing fields. No invented details."""

# Maximum characters of transcript text sent to the LLM. Assumes a ~4096
# token context; system prompt ~200 tokens, leaving ~3800 tokens for
# transcript (~5000-5500 chars at ~1.4 chars/token).
MAX_TRANSCRIPT_CHARS = 5_500


class ReportGenerator:
    def __init__(self):
        self.backend = build_llm_backend()

    # ── Transcript processing ────────────────────────────────────────────────

    @staticmethod
    def _group_turns(segments: list) -> str:
        """Collapse consecutive same-speaker segments into turn blocks.

        This reduces token count significantly (no repeated timestamps / role
        lines) and makes the transcript easier for the LLM to parse.
        """
        if not segments:
            return ""
        turns = []
        cur_role = segments[0]["role"]
        cur_name = segments[0].get("doctor_name") or ""
        cur_texts = [segments[0]["text"]]
        cur_start = segments[0]["start"]

        for seg in segments[1:]:
            role = seg["role"]
            name = seg.get("doctor_name") or ""
            if role == cur_role and name == cur_name:
                cur_texts.append(seg["text"])
            else:
                speaker = cur_role + (f" ({cur_name})" if cur_name else "")
                turns.append(f"[{cur_start:.0f}s] {speaker}: {' '.join(cur_texts)}")
                cur_role, cur_name = role, name
                cur_texts = [seg["text"]]
                cur_start = seg["start"]

        speaker = cur_role + (f" ({cur_name})" if cur_name else "")
        turns.append(f"[{cur_start:.0f}s] {speaker}: {' '.join(cur_texts)}")
        return "\n".join(turns)

    def _truncate_transcript(self, text: str, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
        """Keep the first 60 % + last 40 % of the transcript text.

        This preserves the chief complaint (usually stated at the start) and
        the clinical plan / prescriptions (usually at the end) while dropping
        the middle portion of long consultations that tend to be conversational
        filler.  A separator line is inserted so the LLM knows text was cut.
        """
        if len(text) <= max_chars:
            return text

        head_len = int(max_chars * 0.60)
        tail_len = max_chars - head_len

        head = text[:head_len]
        tail = text[-tail_len:]

        # Try to cut on a newline boundary so we don't break mid-sentence
        head_cut = head.rfind("\n")
        if head_cut > head_len // 2:
            head = head[:head_cut]

        tail_cut = tail.find("\n")
        if 0 < tail_cut < tail_len // 2:
            tail = tail[tail_cut + 1:]

        return head + "\n[...transcript truncated for length...]\n" + tail

    # ── JSON repair ──────────────────────────────────────────────────────────

    @staticmethod
    def _repair_json(raw: str) -> dict:
        """Attempt several strategies to extract valid JSON from a messy LLM output."""
        raw = raw.strip()

        # Strategy 1 — direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Strategy 2 — grab first {...} block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

        # Strategy 3 — strip markdown fences and retry
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 4 — line-by-line: find first line starting with '{' and
        # accumulate until the braces balance
        lines = raw.splitlines()
        start = next((i for i, l in enumerate(lines) if l.strip().startswith("{")), None)
        if start is not None:
            depth, buf = 0, []
            for line in lines[start:]:
                depth += line.count("{") - line.count("}")
                buf.append(line)
                if depth <= 0:
                    break
            candidate = "\n".join(buf)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        raise ValueError(f"LLM did not return valid JSON:\n{raw[:500]}")

    # ── Public API ───────────────────────────────────────────────────────────

    def extract(self, labeled_transcript: list, speaker_confidence: dict | None = None) -> dict:
        # Group consecutive same-speaker segments → cleaner, token-efficient text.
        # Kept full/untruncated here — this is also the source text used for
        # the groundedness check below, before it gets cut down for the LLM.
        text = self._group_turns(labeled_transcript)
        if not text.strip():
            raise ValueError(
                "No speech was detected in this recording — there's nothing to "
                "summarize. Check the microphone input (wrong device, muted, or "
                "no permission granted) and try again."
            )
        truncated = self._truncate_transcript(text)

        raw = self.backend.generate(SYSTEM_PROMPT, truncated, temperature=0.1, max_tokens=512)
        report = self._repair_json(raw)

        flags = validate_schema(report) + check_groundedness(report, text)
        report["_review"] = {
            "flags": flags,
            "speaker_confidence": speaker_confidence or DEFAULT_SPEAKER_CONFIDENCE,
        }
        return report
