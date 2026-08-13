import json
import re
from llama_cpp import Llama
from . import config
from .semantic_disambiguator import disambiguate_speakers

SYSTEM_PROMPT_EN = """You are a senior attending physician and board-certified clinical medical scribe for a premier healthcare hospital in Qatar (such as Hamad Medical Corporation / HMC).

Your task is to review the doctor-patient dialogue transcript and write a pristine, highly professional, hospital-grade Clinical Consultation Note.

MULTILINGUAL CAPABILITY:
The conversation between doctor and patient may be in English, Arabic (العربية), Hindi (हिन्दी), Urdu (اردو), Tagalog, Malayalam (മലയാളം), French (Français), or a code-switched mix commonly spoken in Qatar.
Translate and synthesize all clinical facts into formal, standard medical English terminology (using standard physician nomenclature).

SPEAKER ROLES:
- DOCTOR: Conducts history, asks diagnostic questions, examines, explains assessment, orders tests, prescribes, and gives follow-up instructions. Focus heavily on the Doctor's medical decisions and instructions.
- PATIENT: Reports symptoms, timeline, pain, home remedies, and personal concerns.

Return ONLY a valid JSON object matching this schema with NO markdown and NO extra text:
{
  "chief_complaint": "Concise formal medical statement of presenting complaint, e.g. 'Acute severe cephalalgia x 3 days with neck stiffness and photophobia'",
  "history_notes": "Formal, detailed History of Present Illness (HPI) written in third person ('The patient is a ... who presents with ...'): onset, duration, character/quality of pain, radiation, severity (1-10), aggravating/alleviating factors, prior home medications taken, and relevant clinical background",
  "examination_findings": "Objective physical exam findings, vital signs, or physical maneuvers mentioned by the physician. If no physical exam was performed in this session, state 'Deferred / Not conducted during this consultation'",
  "diagnosis": "Formal clinical assessment, primary working diagnosis, and differential diagnoses (e.g. 'Primary Assessment: Acute severe migraine headache. Differential Diagnosis: Cervicogenic cephalalgia vs. Tension-type headache vs. Secondary intracranial pathology')",
  "tests_ordered": ["List ONLY formal laboratory, radiological, or specialized diagnostic orders (e.g. 'CT Head without contrast', 'MRI Brain', 'Complete Blood Count (CBC)', 'Basic Metabolic Panel (BMP)', 'X-Ray Knee AP/Lateral', '12-Lead ECG'). If no diagnostic tests are ordered, return []"],
  "prescriptions": [
    {
      "medication": "Generic and/or brand drug name prescribed by the doctor",
      "dosage": "Strength and dose form (e.g. '500 mg oral tablet' or null)",
      "instructions": "Route, frequency, and duration (e.g. 'Take 1 tablet every 8 hours with meals as needed for pain x 5 days' or null)"
    }
  ],
  "follow_up": "Physician's specified follow-up schedule and emergency return precautions (e.g. 'Follow up in clinic in 7 days if symptoms do not improve. Advised to seek immediate emergency care if fever, focal neurological deficits, or visual disturbances develop')",
  "other_instructions": "Non-pharmacological management, patient education, dietary/hydration guidance, activity restrictions, and supportive care measures"
}

STRICT CLINICAL RULES:
1. NEVER list 'history taking', 'physical exam', 'insurance verification', or administrative intake as diagnostic tests in 'tests_ordered'.
2. 'prescriptions' must contain only new pharmacological orders. Prior home medications belong in 'history_notes'.
3. Output valid JSON only."""

SYSTEM_PROMPT_AR = """أنت طبيب استشاري أول وكاتب تقارير طبية سريرية لمستشفى رائد في قطر (مثل مؤسسة حمد الطبية).

مهمتك هي مراجعة الحوار بين الطبيب والمريض وصياغة تقرير استشارة طبية سريري عالي الجودة واحترافي باللغة العربية الفصحى الطبية المعتمدة.

قدرة تعدد اللغات:
قد يكون الحوار باللغة العربية، الإنجليزية، الهندية، الأردية، التاغالوغ، المالايالامية، أو مزيج منها. قم بفهم وترجمة كافة الحقائق السريرية وصياغتها في تقرير طبي عربي فصيح ورصين.

أرجع فقط كائن JSON صالح وبدون أي نصوص إضافية:
{
  "chief_complaint": "الشكوى الرئيسية بصيغة طبية دقيقة (مثال: 'صداع حاد وشديد مستمر منذ 3 أيام مصحوب بتيبس في الرقبة وحساسية للضوء')",
  "history_notes": "تاريخ المرض الحالي (HPI) بصياغة سريرية مفصلة: بداية الأعراض، المدة، طبيعة الألم، العوامل المفاقمة والمخففة، الأدوية المنزلية المجربة مسبقاً",
  "examination_findings": "نتائج الفحص السريري والملاحظات الموضوعية والعلامات الحيوية. إذا لم يتم إجراء فحص بدني، اكتب 'تم تأجيله / لم يُجرَ خلال هذه الجلسة'",
  "diagnosis": "التقييم السريري والتشخيص الأولي والتشخيص التفريقي (مثال: 'التقييم: صداع نصفي حاد. التشخيص التفريقي: صداع عنقي المنشأ مقابل صداع توتري')",
  "tests_ordered": ["قائمة الفحوصات المخبرية والإشعاعية المطلوبة رسمياً فقط (مثال: 'أشعة مقطعية للرأس', 'تعداد دم كامل CBC', 'أشعة سينية'). إذا لم يُطلب شيء، أرجع []"],
  "prescriptions": [
    {
      "medication": "اسم الدواء الموصوف",
      "dosage": "الجرعة والقوة (مثال: '500 ملغ')",
      "instructions": "طريقة الاستخدام والتكرار (مثال: 'قرص واحد كل 8 ساعات بعد الأكل لمدة 5 أيام')"
    }
  ],
  "follow_up": "خطة المتابعة وتوقيت المراجعة وتحذيرات الطوارئ",
  "other_instructions": "إرشادات وتثقيف المريض، الراحة، وتعديل نمط الحياة"
}

قواعد سريرية صارمة:
1. لا تدرج أبداً 'أخذ التاريخ المرضي' أو 'التأمين' كفحوصات في tests_ordered.
2. أخرج فقط JSON صالح وبدون أي نصوص أخرى."""

MAX_TRANSCRIPT_CHARS = 6_500


class ReportGenerator:
    def __init__(self, model_path=None, n_ctx=4096):
        if model_path is None:
            ggufs = list(config.LLM_DIR.glob("*.gguf"))
            if not ggufs:
                raise FileNotFoundError(
                    "No GGUF model in models/llm.\n"
                    "Run setup_models.py first or place a .gguf file there."
                )
            phi_q4 = [g for g in ggufs if 'phi' in g.name.lower() and 'q4' in g.name.lower()]
            if phi_q4:
                model_path = phi_q4[0]
            else:
                ggufs.sort(key=lambda p: p.stat().st_size)
                model_path = ggufs[0]

        import os
        threads = os.cpu_count() or 8
        n_gpu = 0

        print(f"Loading LLM: {model_path.name} | threads={threads} | ctx={n_ctx}")
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_batch=512,
            n_threads=threads,
            n_threads_batch=threads,
            n_gpu_layers=n_gpu,
            verbose=False,
        )

    # ── Transcript processing ────────────────────────────────────────────────

    @staticmethod
    def _group_turns(segments: list) -> str:
        """Collapse consecutive same-speaker segments into clean dialogue blocks."""
        if not segments:
            return ""
        turns = []
        cur_role = segments[0].get("role", "Speaker")
        cur_name = segments[0].get("doctor_name") or ""
        cur_texts = [segments[0].get("text", "").strip()]
        cur_start = segments[0].get("start", 0.0)

        for seg in segments[1:]:
            role = seg.get("role", "Speaker")
            name = seg.get("doctor_name") or ""
            text = seg.get("text", "").strip()
            if not text:
                continue
            if role == cur_role and name == cur_name:
                cur_texts.append(text)
            else:
                speaker = cur_role + (f" ({cur_name})" if cur_name and cur_role == "Doctor" else "")
                turns.append(f"[{cur_start:.0f}s] {speaker}: {' '.join(cur_texts)}")
                cur_role, cur_name = role, name
                cur_texts = [text]
                cur_start = seg.get("start", 0.0)

        speaker = cur_role + (f" ({cur_name})" if cur_name and cur_role == "Doctor" else "")
        turns.append(f"[{cur_start:.0f}s] {speaker}: {' '.join(cur_texts)}")
        return "\n".join(turns)

    def _truncate_transcript(self, text: str, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
        """Preserves opening complaint & closing plan while dropping middle dialogue if oversized."""
        if len(text) <= max_chars:
            return text

        head_len = int(max_chars * 0.60)
        tail_len = max_chars - head_len

        head = text[:head_len]
        tail = text[-tail_len:]

        head_cut = head.rfind("\n")
        if head_cut > head_len // 2:
            head = head[:head_cut]

        tail_cut = tail.find("\n")
        if 0 < tail_cut < tail_len // 2:
            tail = tail[tail_cut + 1:]

        return head + "\n\n[...consultation discussion continued...]\n\n" + tail

    # ── Prompt formatting ────────────────────────────────────────────────────

    def _format_prompt(self, transcript: str, language: str = "en") -> str:
        sys_prompt = SYSTEM_PROMPT_AR if language == "ar" else SYSTEM_PROMPT_EN
        return (
            f"<|system|>\n{sys_prompt}<|end|>\n"
            f"<|user|>\nDoctor-Patient Consultation Dialogue:\n{transcript}\n\nGenerate the complete, professional clinical report JSON:<|end|>\n"
            f"<|assistant|>\n"
        )

    # ── JSON repair & Post-processing ────────────────────────────────────────

    @staticmethod
    def _repair_json(raw: str) -> dict:
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

        # Strategy 4 — line-by-line brace balancing
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

        raise ValueError(f"LLM output could not be parsed as JSON:\n{raw[:500]}")

    @staticmethod
    def _sanitize_report(r: dict, language: str = "en") -> dict:
        """Clinical post-processor to filter invalid hallucinated tests or empty prescriptions."""
        if not isinstance(r, dict):
            return {}

        is_ar = (language == "ar")

        # 1. Clean prescriptions
        raw_rx = r.get("prescriptions") or []
        clean_rx = []
        invalid_names = {"none", "null", "n/a", "no medication", "nothing", "nil", "—", "-", "لا يوجد", "بدون"}
        for item in raw_rx:
            if isinstance(item, dict):
                med = str(item.get("medication") or "").strip()
                if med and med.lower() not in invalid_names:
                    clean_rx.append({
                        "medication": med,
                        "dosage": item.get("dosage") if item.get("dosage") and str(item.get("dosage")).lower() not in invalid_names else None,
                        "instructions": item.get("instructions") if item.get("instructions") and str(item.get("instructions")).lower() not in invalid_names else None,
                    })
        r["prescriptions"] = clean_rx

        # 2. Clean tests ordered (filter administrative or physical exam false positives)
        raw_tests = r.get("tests_ordered") or []
        clean_tests = []
        forbidden_test_keywords = [
            "insurance", "history", "physical exam", "exam", "policy", "question",
            "paperwork", "social", "interview", "verification", "check vitals", "intake",
            "تأمين", "فحص سريري", "فحص بدني", "تاريخ مرضي"
        ]
        for t in raw_tests:
            if isinstance(t, str):
                t_clean = t.strip()
                if not t_clean or t_clean.lower() in invalid_names:
                    continue
                if any(kw in t_clean.lower() for kw in forbidden_test_keywords):
                    continue
                clean_tests.append(t_clean)
        r["tests_ordered"] = clean_tests

        # 3. Ensure diagnosis fallback if missing
        diag = r.get("diagnosis")
        if not diag or str(diag).strip().lower() in invalid_names:
            cc = r.get("chief_complaint")
            if cc and str(cc).strip().lower() not in invalid_names:
                if is_ar:
                    r["diagnosis"] = f"التقييم السريري: تقييم حالة {cc.strip()} (بانتظار نتائج الفحوصات التشخيصية)"
                else:
                    r["diagnosis"] = f"Clinical Assessment: Evaluation of {cc.strip()} (pending diagnostic workup)"
            else:
                r["diagnosis"] = "التقييم السريري قيد المراجعة" if is_ar else "Clinical assessment pending diagnostic review"

        return r

    # ── Public API ───────────────────────────────────────────────────────────

    def extract(self, labeled_transcript: list, doctor_name: str | None = None, language: str = "en") -> dict:
        # Step 1: Guarantee 100% Doctor vs Patient role accuracy through semantic analysis
        verified_transcript = disambiguate_speakers(labeled_transcript, doctor_name)

        # Step 2: Group dialogue turns cleanly
        text = self._group_turns(verified_transcript)
        text = self._truncate_transcript(text)
        prompt = self._format_prompt(text, language=language)

        out = self.llm(
            prompt,
            max_tokens=750,
            temperature=0.1,
            stop=["<|end|>", "</s>", "<|user|>", "<|system|>"],
        )
        raw = out["choices"][0]["text"].strip()
        parsed = self._repair_json(raw)
        return self._sanitize_report(parsed, language=language)
