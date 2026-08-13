import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import queue
from pathlib import Path
from datetime import datetime
from docx import Document
import sounddevice as sd
import numpy as np
import json
import wave
import tempfile
import shutil
import os
import platform

from .pipeline import AudioPipeline
from .report_generator import ReportGenerator
from .llm_backend import _resolve_gguf_path
from . import config


class MedicalScribeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Local Medical Scribe")
        self.geometry("1400x900")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.pipeline: AudioPipeline | None = None
        self.report_gen: ReportGenerator | None = None

        # ── Recording state ──────────────────────────────────────────────────
        self.is_recording = False
        self._audio_queue: queue.Queue = queue.Queue()   # raw chunks from InputStream
        self._recorded_chunks: list = []                 # accumulated float32 arrays
        self._recording_stream = None
        self.sample_rate = 16000

        # ── Current results ──────────────────────────────────────────────────
        self.current_report = None
        self.current_report_stem: str | None = None
        self.current_transcript: list = []
        self.current_attending = ""
        self.current_patient = ""

        # ── Review/export gate (Phase 2) ────────────────────────────────────
        self.report_confirmed = False

        self._build_sidebar()
        self._build_content()
        self._show_tab("process")

        # Load models in background so GUI opens instantly
        threading.Thread(target=self._init_backend, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    # Sidebar
    # ══════════════════════════════════════════════════════════════════════════

    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=220, corner_radius=0)
        sb.pack(side="left", fill="y", padx=0, pady=0)

        ctk.CTkLabel(sb, text="Medical Scribe",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20, 10))

        self.nav_buttons = {}
        for key, label in [
            ("process", "Process Visit"),
            ("enroll",  "Enroll Doctor"),
            ("reports", "Reports"),
        ]:
            btn = ctk.CTkButton(sb, text=label, command=lambda k=key: self._show_tab(k))
            btn.pack(pady=8, padx=15, fill="x")
            self.nav_buttons[key] = btn

        self.status = ctk.CTkLabel(sb, text="Initializing...", wraplength=180)
        self.status.pack(side="bottom", pady=(5, 20))

        # ── Backend selector + indicator (Phase 4) ──────────────────────────
        backend_frame = ctk.CTkFrame(sb, fg_color="transparent")
        backend_frame.pack(side="bottom", fill="x", padx=15, pady=(10, 0))

        ctk.CTkLabel(backend_frame, text="LLM Backend",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w")

        self.backend_menu = ctk.CTkOptionMenu(
            backend_frame, values=["local_gguf", "api", "phi3_legacy"],
            command=self._on_backend_selected, width=190, state="disabled",
        )
        self.backend_menu.set(config.LLM_BACKEND)
        self.backend_menu.pack(fill="x", pady=(2, 5))

        self.backend_indicator = ctk.CTkLabel(
            backend_frame, text="Backend not loaded yet", text_color="gray",
            wraplength=190, justify="left", anchor="w", font=ctk.CTkFont(size=11),
        )
        self.backend_indicator.pack(fill="x")

    # ══════════════════════════════════════════════════════════════════════════
    # Backend indicator / live switching (Phase 4)
    # ══════════════════════════════════════════════════════════════════════════

    def _describe_backend(self, backend_obj) -> tuple[str, str]:
        """Human-readable (text, color) for the currently active backend
        instance — reads off the real constructed object plus config.py
        (the Phase 0 single source of truth), never re-derives/duplicates
        the selection logic itself.
        """
        kind = type(backend_obj).__name__
        if kind in ("LocalGGUFBackend", "Phi3LegacyBackend"):
            label = "Local GGUF" if kind == "LocalGGUFBackend" else "Phi-3 Legacy"
            try:
                path = _resolve_gguf_path()
                return f"Active backend: {label}\n({path.name})", "#27ae60"
            except Exception as e:
                return f"Active backend: {label} (file lookup failed: {e})", "#e67e22"
        if kind == "APIBackend":
            host = (config.LLM_API_BASE_URL or "?").split("//")[-1].split("/")[0]
            return f"Active backend: API\n({config.LLM_API_MODEL or '?'} @ {host})", "#27ae60"
        return f"Active backend: {kind}", "#27ae60"

    def _set_backend_indicator(self, text: str, color: str):
        self.backend_indicator.configure(text=text, text_color=color)

    def _on_backend_selected(self, choice: str):
        if choice == config.LLM_BACKEND:
            return
        config.LLM_BACKEND = choice
        self._log(f"Switching LLM backend to '{choice}'...")
        self._set_backend_indicator(f"Loading backend: {choice}...", "#2980b9")
        self.backend_menu.configure(state="disabled")
        threading.Thread(target=self._reload_backend, daemon=True).start()

    def _reload_backend(self):
        try:
            new_report_gen = ReportGenerator()
            self.report_gen = new_report_gen
            self._log(f"Backend switched to '{config.LLM_BACKEND}'.")
            text, color = self._describe_backend(self.report_gen.backend)
            self._set_backend_indicator(text, color)
        except Exception as e:
            self._log(f"ERROR: backend switch failed: {e}")
            self._set_backend_indicator(f"Backend '{config.LLM_BACKEND}' FAILED to load: {e}", "#c0392b")
        finally:
            self.backend_menu.configure(state="normal")

    # ══════════════════════════════════════════════════════════════════════════
    # Content area
    # ══════════════════════════════════════════════════════════════════════════

    def _build_content(self):
        self.content = ctk.CTkFrame(self, corner_radius=0)
        self.content.pack(side="right", fill="both", expand=True)

        self._build_process_tab()
        self._build_enroll_tab()
        self._build_reports_tab()

    # ── Process tab ──────────────────────────────────────────────────────────

    def _build_process_tab(self):
        self.tab_process = ctk.CTkFrame(self.content)

        # ── Top controls ─────────────────────────────────────────────────────
        ctrl = ctk.CTkFrame(self.tab_process)
        ctrl.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=10)

        ctk.CTkLabel(ctrl, text="Patient Name").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.entry_patient = ctk.CTkEntry(ctrl, width=250, placeholder_text="Patient name")
        self.entry_patient.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        ctk.CTkLabel(ctrl, text="Doctor Name").grid(row=0, column=2, padx=(20, 5), pady=10, sticky="w")
        self.entry_doctor = ctk.CTkEntry(ctrl, width=200, placeholder_text="Your name (optional)")
        self.entry_doctor.grid(row=0, column=3, padx=5, pady=10, sticky="w")

        self.btn_record = ctk.CTkButton(
            ctrl, text="🎙 Start Recording",
            command=self._toggle_recording, fg_color="#c0392b", width=160,
        )
        self.btn_record.grid(row=0, column=4, padx=10)

        self.recording_status = ctk.CTkLabel(ctrl, text="Ready to record", text_color="gray")
        self.recording_status.grid(row=1, column=0, columnspan=5, pady=5)

        # ── No-profile warning banner ─────────────────────────────────────
        self._no_profile_banner = ctk.CTkLabel(
            ctrl,
            text="⚠  No doctor voice profile enrolled. Speaker roles may be incorrect."
                 "  →  Use 'Enroll Doctor' tab to fix this, or enter Doctor Name above.",
            text_color="#e67e22",
            wraplength=700,
        )
        self._no_profile_banner.grid(row=2, column=0, columnspan=5, pady=(0, 5))
        self._update_profile_banner()

        # ── Load from file ────────────────────────────────────────────────────
        ctk.CTkLabel(self.tab_process, text="Or load audio file:").grid(
            row=1, column=0, padx=10, pady=10, sticky="w")
        self.entry_audio = ctk.CTkEntry(self.tab_process, width=400)
        self.entry_audio.grid(row=1, column=1, padx=10, pady=10)
        ctk.CTkButton(self.tab_process, text="Browse",
                      command=self._browse_audio).grid(row=1, column=2, padx=10)

        ctk.CTkButton(
            self.tab_process, text="▶  Process Audio File",
            command=self._run_pipeline, fg_color="#27ae60",
        ).grid(row=2, column=1, pady=10)

        # ── Progress bar & log ────────────────────────────────────────────────
        self.progress = ctk.CTkProgressBar(self.tab_process, mode="indeterminate")
        self.progress.grid(row=3, column=0, columnspan=3, sticky="ew", padx=20, pady=5)
        self.progress.stop()
        self.progress.set(0)

        self.log_box = ctk.CTkTextbox(self.tab_process, height=140)
        self.log_box.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=20, pady=5)

        # ── Tabview: Transcript | Review ──────────────────────────────────────
        self.result_tabs = ctk.CTkTabview(self.tab_process)
        self.result_tabs.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=20, pady=5)

        self.result_tabs.add("Transcript")
        self.result_tabs.add("Review")

        # Transcript tab
        tx_frame = self.result_tabs.tab("Transcript")
        tx_frame.grid_columnconfigure(0, weight=1)
        tx_frame.grid_rowconfigure(0, weight=1)

        self.transcript_box = ctk.CTkTextbox(tx_frame)
        self.transcript_box.grid(row=0, column=0, sticky="nsew")

        ctk.CTkButton(
            tx_frame, text="Copy Transcript",
            command=self._copy_transcript, width=140,
        ).grid(row=1, column=0, pady=5)

        # Review tab — editable draft + flags + confirm/export gate
        rp_frame = self.result_tabs.tab("Review")
        self._build_review_widgets(rp_frame)

        # Row/column weights
        self.tab_process.grid_columnconfigure(1, weight=1)
        self.tab_process.grid_rowconfigure(5, weight=1)

    # ── Review tab (Phase 2: editable draft + flags + confirm/export gate) ────

    REVIEW_TEXT_FIELDS = [
        ("chief_complaint", "Chief Complaint", "entry"),
        ("history_notes", "History & Subjective Notes", "text"),
        ("examination_findings", "Physical Examination & Vitals", "text"),
        ("diagnosis", "Diagnosis & Assessment", "entry"),
        ("tests_ordered", "Diagnostic Tests Ordered (one per line)", "text"),
        ("follow_up", "Follow-up & Care Plan", "entry"),
        ("other_instructions", "Other Instructions", "text"),
    ]

    SEVERITY_STYLE = {
        "error":   ("#c0392b", "✕"),   # red  ✕
        "warning": ("#e67e22", "⚠"),   # orange ⚠
    }

    TIER_STYLE = {
        "threshold_match":           ("#27ae60", "High confidence match"),
        "best_score_promoted":       ("#e67e22", "Low confidence — best available match, below threshold"),
        "no_profiles_speaking_time": ("#e67e22", "No voice profiles enrolled — guessed from speaking time"),
        "speaking_time_fallback":    ("#c0392b", "Very low confidence — guessed from speaking time"),
        "unresolved":                ("#c0392b", "Could not be resolved automatically"),
    }

    def _build_review_widgets(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        # ── Speaker confidence panel ────────────────────────────────────────
        self.speaker_conf_frame = ctk.CTkFrame(parent)
        self.speaker_conf_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        self.speaker_conf_label = ctk.CTkLabel(
            self.speaker_conf_frame, text="No report generated yet.", justify="left", anchor="w")
        self.speaker_conf_label.pack(fill="x", padx=10, pady=5)
        self.speaker_override_label = ctk.CTkLabel(
            self.speaker_conf_frame, text="", text_color="#2980b9", justify="left", anchor="w")
        self.speaker_override_label.pack(fill="x", padx=10, pady=(0, 5))

        # ── Scrollable editable form ─────────────────────────────────────────
        form = ctk.CTkScrollableFrame(parent)
        form.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        form.grid_columnconfigure(0, weight=1)

        self.review_fields = {}
        self.review_flag_frames = {}

        for key, label, kind in self.REVIEW_TEXT_FIELDS:
            ctk.CTkLabel(form, text=label, font=ctk.CTkFont(weight="bold")).pack(
                anchor="w", padx=5, pady=(10, 0))
            if kind == "entry":
                widget = ctk.CTkEntry(form)
                widget.pack(fill="x", padx=5, pady=2)
            else:
                widget = ctk.CTkTextbox(form, height=70)
                widget.pack(fill="x", padx=5, pady=2)
            widget.bind("<KeyRelease>", self._on_review_field_changed)
            self.review_fields[key] = widget

            flag_frame = ctk.CTkFrame(form, fg_color="transparent")
            flag_frame.pack(fill="x", padx=5, pady=(0, 2))
            self.review_flag_frames[key] = flag_frame

        # ── Prescriptions editor ────────────────────────────────────────────
        ctk.CTkLabel(form, text="Prescriptions", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=5, pady=(10, 0))
        self.prescriptions_frame = ctk.CTkFrame(form, fg_color="transparent")
        self.prescriptions_frame.pack(fill="x", padx=5, pady=2)
        self.prescription_rows: list = []

        ctk.CTkButton(
            form, text="+ Add Prescription", width=140,
            command=lambda: self._add_prescription_row(),
        ).pack(anchor="w", padx=5, pady=(0, 10))

        # ── Confirm / export gate ────────────────────────────────────────────
        gate = ctk.CTkFrame(parent)
        gate.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

        self.review_status_label = ctk.CTkLabel(gate, text="", text_color="gray", justify="left", anchor="w")
        self.review_status_label.pack(side="top", fill="x", padx=10, pady=(5, 0))

        btn_row = ctk.CTkFrame(gate, fg_color="transparent")
        btn_row.pack(side="top", fill="x", padx=10, pady=5)

        self.btn_confirm = ctk.CTkButton(
            btn_row, text="✓ Approve Report",
            command=self._confirm_report, fg_color="#27ae60", width=180,
        )
        self.btn_confirm.pack(side="left", padx=(0, 10))

        self.btn_export = ctk.CTkButton(
            btn_row, text="Export DOCX",
            command=self._export_docx, width=140, state="disabled",
        )
        self.btn_export.pack(side="left")

    def _add_prescription_row(self, medication: str = "", dosage: str = "", instructions: str = ""):
        row_frame = ctk.CTkFrame(self.prescriptions_frame)
        row_frame.pack(fill="x", pady=2)
        row_frame.grid_columnconfigure(0, weight=2)
        row_frame.grid_columnconfigure(1, weight=1)
        row_frame.grid_columnconfigure(2, weight=2)

        med_entry = ctk.CTkEntry(row_frame, placeholder_text="Medication")
        med_entry.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        med_entry.insert(0, medication or "")

        dose_entry = ctk.CTkEntry(row_frame, placeholder_text="Dosage")
        dose_entry.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        dose_entry.insert(0, dosage or "")

        instr_entry = ctk.CTkEntry(row_frame, placeholder_text="Instructions")
        instr_entry.grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        instr_entry.insert(0, instructions or "")

        flag_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
        flag_frame.grid(row=1, column=0, columnspan=4, sticky="ew", padx=2)

        row_data = {
            "frame": row_frame, "medication": med_entry, "dosage": dose_entry,
            "instructions": instr_entry, "flag_frame": flag_frame,
        }

        remove_btn = ctk.CTkButton(
            row_frame, text="✕", width=28, fg_color="#c0392b",
            command=lambda: self._remove_prescription_row(row_data),
        )
        remove_btn.grid(row=0, column=3, padx=2, pady=2)

        for entry in (med_entry, dose_entry, instr_entry):
            entry.bind("<KeyRelease>", self._on_review_field_changed)

        self.prescription_rows.append(row_data)
        self._on_review_field_changed()
        return row_data

    def _remove_prescription_row(self, row_data: dict):
        row_data["frame"].destroy()
        self.prescription_rows.remove(row_data)
        self._on_review_field_changed()

    def _clear_flag_frame(self, frame):
        for w in frame.winfo_children():
            w.destroy()

    def _render_flag_label(self, parent, flag: dict):
        color, prefix = self.SEVERITY_STYLE.get(flag.get("severity"), ("gray", "•"))
        text = f"{prefix} {flag.get('issue', '')}"
        if flag.get("type") == "groundedness" and flag.get("value"):
            text += f"  (“{flag['value']}”)"
        ctk.CTkLabel(parent, text=text, text_color=color, justify="left", anchor="w").pack(
            fill="x", padx=4, pady=1)

    def _populate_review(self, report: dict):
        """Fill the Review tab's editable widgets from a freshly generated
        report and render its _review flags / speaker_confidence. Resets the
        confirm/export gate — every new report starts unconfirmed, even if
        it has zero flags.
        """
        self.report_confirmed = False
        self.btn_export.configure(state="disabled")

        review = report.get("_review") or {}
        flags = review.get("flags") or []
        flags_by_field: dict = {}
        for f in flags:
            flags_by_field.setdefault(f.get("field"), []).append(f)

        for key, widget in self.review_fields.items():
            value = report.get(key)
            if key == "tests_ordered":
                text = "\n".join(value) if isinstance(value, list) else (value or "")
            else:
                text = value if isinstance(value, str) else ("" if value is None else str(value))

            if isinstance(widget, ctk.CTkTextbox):
                widget.delete("1.0", "end")
                widget.insert("1.0", text)
            else:
                widget.delete(0, "end")
                widget.insert(0, text)

            flag_frame = self.review_flag_frames[key]
            self._clear_flag_frame(flag_frame)
            for f in flags_by_field.get(key, []):
                self._render_flag_label(flag_frame, f)

        # Prescriptions — rebuild rows from scratch
        for row in list(self.prescription_rows):
            row["frame"].destroy()
        self.prescription_rows = []
        prescriptions = report.get("prescriptions") or []
        if isinstance(prescriptions, list):
            for i, item in enumerate(prescriptions):
                if not isinstance(item, dict):
                    continue
                row = self._add_prescription_row(
                    item.get("medication") or "", item.get("dosage") or "", item.get("instructions") or "")
                row_flags = (
                    flags_by_field.get(f"prescriptions[{i}]", [])
                    + flags_by_field.get(f"prescriptions[{i}].medication", [])
                    + flags_by_field.get(f"prescriptions[{i}].dosage", [])
                    + flags_by_field.get(f"prescriptions[{i}].instructions", [])
                )
                for f in row_flags:
                    self._render_flag_label(row["flag_frame"], f)

        # Speaker confidence panel
        sc = review.get("speaker_confidence") or {}
        tier = sc.get("tier")
        color, desc = self.TIER_STYLE.get(tier, ("gray", "Unknown"))
        score = sc.get("score")
        score_text = f" (similarity {score:.2f})" if isinstance(score, (int, float)) else ""
        self.speaker_conf_label.configure(
            text=f"Speaker identification: {desc}{score_text}", text_color=color)
        if sc.get("manually_overridden"):
            self.speaker_override_label.configure(
                text=f"ⓘ Doctor name manually entered by user: {sc.get('override_name') or '—'} "
                     "(overrides automated identification)")
        else:
            self.speaker_override_label.configure(text="")

        if not flags:
            self.review_status_label.configure(
                text="No issues flagged by automated review — please still confirm before export.",
                text_color="gray")
        else:
            n_err = sum(1 for f in flags if f.get("severity") == "error")
            n_warn = sum(1 for f in flags if f.get("severity") == "warning")
            self.review_status_label.configure(
                text=f"{n_err} error(s), {n_warn} warning(s) flagged — review before confirming.",
                text_color="#c0392b" if n_err else "#e67e22")

    def _collect_review_form_values(self) -> dict:
        values = {}
        for key, widget in self.review_fields.items():
            if isinstance(widget, ctk.CTkTextbox):
                text = widget.get("1.0", "end").rstrip("\n")
            else:
                text = widget.get().strip()

            if key == "tests_ordered":
                values[key] = [line.strip() for line in text.splitlines() if line.strip()]
            else:
                values[key] = text if text else None

        prescriptions = []
        for row in self.prescription_rows:
            med = row["medication"].get().strip()
            dose = row["dosage"].get().strip()
            instr = row["instructions"].get().strip()
            if med or dose or instr:
                prescriptions.append({
                    "medication": med or None,
                    "dosage": dose or None,
                    "instructions": instr or None,
                })
        values["prescriptions"] = prescriptions
        return values

    def _on_review_field_changed(self, event=None):
        """Any edit after approval invalidates it — export must be re-approved
        against the current (edited) content, never a stale approved snapshot.
        """
        if self.report_confirmed:
            self.report_confirmed = False
            self.btn_export.configure(state="disabled")
            self._log("Report edited after approval — please re-approve before exporting.")

    def _confirm_report(self):
        if not self.current_report_stem:
            messagebox.showwarning("Warning", "No report available to approve.")
            return

        final_report = self._collect_review_form_values()
        self.current_report = final_report
        self.report_confirmed = True
        self.btn_export.configure(state="normal")

        out_path = config.OUTPUT_DIR / f"{self.current_report_stem}_report.confirmed.json"
        out_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
        self._log(f"Report approved. Saved confirmed record → {out_path}")
        self._refresh_reports_list()

    # ── Enroll tab ───────────────────────────────────────────────────────────

    def _build_enroll_tab(self):
        self.tab_enroll = ctk.CTkFrame(self.content)
        ctk.CTkLabel(self.tab_enroll, text="Doctor Name").grid(
            row=0, column=0, padx=10, pady=10, sticky="w")
        self.entry_doc_name = ctk.CTkEntry(self.tab_enroll, width=300)
        self.entry_doc_name.grid(row=0, column=1, padx=10)
        ctk.CTkButton(self.tab_enroll, text="Browse Audio",
                      command=self._browse_enroll).grid(row=0, column=2, padx=10)
        ctk.CTkButton(self.tab_enroll, text="Enroll",
                      command=self._run_enroll, fg_color="#27ae60").grid(row=1, column=1, pady=20)
        self.enroll_log = ctk.CTkTextbox(self.tab_enroll, height=400)
        self.enroll_log.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=20, pady=10)
        self.tab_enroll.grid_columnconfigure(1, weight=1)
        self.tab_enroll.grid_rowconfigure(2, weight=1)

    # ── Reports tab ───────────────────────────────────────────────────────────

    def _build_reports_tab(self):
        self.tab_reports = ctk.CTkFrame(self.content)
        ctk.CTkButton(
            self.tab_reports, text="Open Output Folder",
            command=lambda: self._open_path(config.OUTPUT_DIR),
        ).pack(pady=20)
        self.reports_scroll = ctk.CTkScrollableFrame(self.tab_reports)
        self.reports_scroll.pack(fill="both", expand=True, padx=20, pady=10)
        self._refresh_reports_list()

    # ══════════════════════════════════════════════════════════════════════════
    # Navigation
    # ══════════════════════════════════════════════════════════════════════════

    def _show_tab(self, key: str):
        for w in [self.tab_process, self.tab_enroll, self.tab_reports]:
            w.pack_forget()
        if key == "process":
            self.tab_process.pack(fill="both", expand=True)
        elif key == "enroll":
            self.tab_enroll.pack(fill="both", expand=True)
        else:
            self.tab_reports.pack(fill="both", expand=True)
            self._refresh_reports_list()

    # ══════════════════════════════════════════════════════════════════════════
    # Backend init
    # ══════════════════════════════════════════════════════════════════════════

    def _init_backend(self):
        try:
            self._log("Loading AI models... (this takes ~30-60 s on first run)")
            self.pipeline = AudioPipeline(progress_callback=self._log)

            self._set_backend_indicator(f"Loading backend: {config.LLM_BACKEND}...", "#2980b9")
            self.report_gen = ReportGenerator()
            text, color = self._describe_backend(self.report_gen.backend)
            self._set_backend_indicator(text, color)

            self._set_status(f"Ready | Device: {config.DEVICE}")
            self._log("All models loaded. Ready.")
        except Exception as e:
            self._set_status(f"Error: {e}")
            if self.report_gen is None:
                self._set_backend_indicator(
                    f"Backend '{config.LLM_BACKEND}' FAILED to load: {e}", "#c0392b")
            messagebox.showerror("Model Load Failed", str(e))
        finally:
            self.backend_menu.configure(state="normal")

    # ══════════════════════════════════════════════════════════════════════════
    # Recording — streaming InputStream (no pre-allocated buffer)
    # ══════════════════════════════════════════════════════════════════════════

    def _toggle_recording(self):
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _update_profile_banner(self):
        """Show/hide the no-profile warning based on enrolled profiles."""
        has_profiles = any(config.PROFILES_DIR.glob("*.npy"))
        if has_profiles:
            self._no_profile_banner.grid_remove()
        else:
            self._no_profile_banner.grid()

    def _start_recording(self):
        if not self.pipeline:
            messagebox.showerror("Error", "Models still loading. Please wait.")
            return

        patient_name = self.entry_patient.get().strip()
        if not patient_name:
            messagebox.showerror("Error", "Please enter patient name first.")
            return

        self._recorded_chunks.clear()
        self.is_recording = True
        self.btn_record.configure(text="⏹ Stop Recording", fg_color="#e67e22")
        self.recording_status.configure(text="● Recording…", text_color="red")
        self._log("Recording started (streaming — no time limit).")

        def _audio_callback(indata, frames, time_info, status):
            if status:
                self._log(f"[Audio stream] {status}")
            if self.is_recording:
                self._recorded_chunks.append(indata.copy())

        self._recording_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=_audio_callback,
        )
        self._recording_stream.start()

    def _stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        if self._recording_stream:
            self._recording_stream.stop()
            self._recording_stream.close()
            self._recording_stream = None

        self.btn_record.configure(text="🎙 Start Recording", fg_color="#c0392b")
        self.recording_status.configure(text="Processing recording…", text_color="#2980b9")
        self._log("Recording stopped. Saving & processing…")

        threading.Thread(target=self._process_recording, daemon=True).start()

    def _process_recording(self):
        try:
            if not self._recorded_chunks:
                self._log("No audio captured.")
                self.recording_status.configure(text="No audio captured", text_color="gray")
                return

            audio_data = np.concatenate(self._recorded_chunks, axis=0).squeeze()
            duration_s = len(audio_data) / self.sample_rate
            self._log(f"Captured {duration_s:.1f} s of audio.")

            # Save to temp WAV
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, dir=config.TEMP_DIR)
            tmp_path = tmp.name
            tmp.close()

            audio_int16 = (audio_data * 32767).astype(np.int16)
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())

            self._log(f"Saved recording → {tmp_path}")
            self._run_full_pipeline(
                tmp_path,
                self.entry_patient.get() or "N/A",
                doctor_name_override=self.entry_doctor.get().strip() or None,
                cleanup_path=tmp_path,
            )

            self.recording_status.configure(text="Report generated ✓", text_color="#27ae60")

        except Exception as e:
            self._log(f"ERROR: {e}")
            messagebox.showerror("Processing Error", str(e))
            self.recording_status.configure(text="Processing failed", text_color="gray")

    # ══════════════════════════════════════════════════════════════════════════
    # File pipeline
    # ══════════════════════════════════════════════════════════════════════════

    def _browse_audio(self):
        f = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac")])
        if f:
            self.entry_audio.delete(0, "end")
            self.entry_audio.insert(0, f)

    def _run_pipeline(self):
        path = self.entry_audio.get()
        if not Path(path).exists():
            messagebox.showerror("Error", "Select a valid audio file.")
            return
        self.progress.start()
        threading.Thread(
            target=self._run_full_pipeline,
            args=(path, self.entry_patient.get() or "N/A"),
            kwargs={"doctor_name_override": self.entry_doctor.get().strip() or None},
            daemon=True,
        ).start()

    def _run_full_pipeline(
        self,
        audio_path: str,
        patient_name: str,
        doctor_name_override: str | None = None,
        cleanup_path: str | None = None,
    ):
        """Core processing: STT → transcript preview → LLM report."""
        try:
            self.progress.start()
            self.transcript_box.delete("1.0", "end")

            # Timestamp suffix avoids two visits colliding on the same output
            # filenames (e.g. two uploaded files both literally named
            # "recording.wav" — mic recordings were already effectively
            # collision-free via tempfile's own uniqueness guarantee).
            stem = f"{Path(audio_path).stem}_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
            self.current_report_stem = stem

            # ── 0. Raw audio retention (Phase 3) — off by default ──────────
            if config.RETAIN_RAW_AUDIO:
                try:
                    ext = Path(audio_path).suffix or ".wav"
                    audio_dest = config.AUDIO_DIR / f"{stem}{ext}"
                    shutil.copy2(audio_path, audio_dest)
                    self._log(f"Retained raw audio → {audio_dest}")
                except Exception as e:
                    self._log(f"WARNING: could not retain raw audio: {e}")

            # ── 1. Audio pipeline (diarization + Whisper) ─────────────────
            result = self.pipeline.process(audio_path, patient_name)
            labeled = result["labeled_transcript"]

            self.current_transcript = labeled
            # If the user typed a doctor name, trust it over speaker-ID result
            attending = result["attending_doctor"]
            if doctor_name_override:
                attending = doctor_name_override
                # Also back-fill doctor_name in transcript segments for LLM context
                for seg in labeled:
                    if seg["role"] == "Doctor":
                        seg["doctor_name"] = doctor_name_override
            elif attending == "Unknown" and not any(config.PROFILES_DIR.glob("*.npy")):
                attending = "Attending Physician (unverified)"
            self.current_attending = attending
            self.current_patient = result["patient_name"]

            # ── 2. Show transcript immediately, and persist it (Phase 3) ───
            self._populate_transcript(labeled)
            self.result_tabs.set("Transcript")   # switch to transcript view

            dur_min = result.get("audio_duration_s", 0) / 60
            wc = result.get("word_count", 0)
            self._log(f"Transcript complete — {dur_min:.1f} min audio, ~{wc} words")

            # Persisted unconditionally for every processed visit, before
            # report generation, so it's on disk even if that step fails.
            transcript_record = {
                "stem": stem,
                "patient_name": result["patient_name"],
                "audio_duration_s": result.get("audio_duration_s"),
                "word_count": result.get("word_count"),
                "segments": labeled,
            }
            transcript_path = config.OUTPUT_DIR / f"{stem}_transcript.json"
            transcript_path.write_text(json.dumps(transcript_record, indent=2), encoding="utf-8")
            self._log(f"Saved transcript → {transcript_path}")

            # ── 3. Generate report with local LLM ─────────────────────────
            self._log("Generating medical report with local LLM…")
            speaker_confidence = dict(result.get("speaker_resolution") or {})
            speaker_confidence["manually_overridden"] = bool(doctor_name_override)
            speaker_confidence["override_name"] = doctor_name_override or None
            report = self.report_gen.extract(labeled, speaker_confidence=speaker_confidence)
            self.current_report = report

            # This is the AI draft only — not yet reviewed/approved. Export
            # stays locked until the clinician confirms it from this tab.
            self._populate_review(report)
            self.result_tabs.set("Review")       # switch to review view

            # ── 4. Save JSON (draft, including _review — unchanged from Phase 1) ──
            out_json = config.OUTPUT_DIR / f"{stem}_report.json"
            out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
            self._log(f"Saved draft JSON → {out_json}")

        except Exception as e:
            self._log(f"ERROR: {e}")
            messagebox.showerror("Pipeline Error", str(e))
        finally:
            self.progress.stop()
            if cleanup_path:
                Path(cleanup_path).unlink(missing_ok=True)

    # ── Transcript display ────────────────────────────────────────────────────

    def _populate_transcript(self, labeled: list):
        """Render the labeled transcript into the transcript textbox."""
        self.transcript_box.delete("1.0", "end")
        for seg in labeled:
            ts = f"[{seg['start']:.1f}s]"
            role = seg["role"]
            name = seg.get("doctor_name") or ""
            speaker = f"{role}" + (f" ({name})" if name else "")
            self.transcript_box.insert("end", f"{ts} {speaker}:\n  {seg['text']}\n\n")
        self.transcript_box.see("1.0")

    def _copy_transcript(self):
        if not self.current_transcript:
            messagebox.showinfo("Info", "No transcript available yet.")
            return
        lines = []
        for seg in self.current_transcript:
            role = seg["role"]
            name = seg.get("doctor_name") or ""
            speaker = role + (f" ({name})" if name else "")
            lines.append(f"[{seg['start']:.1f}s] {speaker}: {seg['text']}")
        text = "\n".join(lines)
        self.clipboard_clear()
        self.clipboard_append(text)
        self._log("Transcript copied to clipboard.")

    # ══════════════════════════════════════════════════════════════════════════
    # Enroll
    # ══════════════════════════════════════════════════════════════════════════

    def _browse_enroll(self):
        f = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.m4a")])
        if f:
            self.entry_enroll_path = f
            self.enroll_log.insert("end", f"Selected: {f}\n")

    def _run_enroll(self):
        name = self.entry_doc_name.get().strip()
        if not name or not hasattr(self, "entry_enroll_path"):
            messagebox.showerror("Error", "Enter doctor name and select audio.")
            return
        threading.Thread(
            target=self._enroll_thread, args=(name, self.entry_enroll_path), daemon=True
        ).start()

    def _enroll_thread(self, name: str, path: str):
        try:
            safe = self.pipeline.enroll_doctor(name, path)
            self.enroll_log.insert("end", f"Enrolled: {name} ({safe})\n")
            self._update_profile_banner()   # hide warning once a profile exists
            self._refresh_reports_list()
        except Exception as e:
            self.enroll_log.insert("end", f"ERROR: {e}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # Export
    # ══════════════════════════════════════════════════════════════════════════

    def _export_docx(self):
        if not self.report_confirmed:
            messagebox.showerror(
                "Not approved",
                "Approve the report on the Review tab before exporting.")
            return
        if not self.current_report:
            messagebox.showwarning("Warning", "No report available to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".docx", filetypes=[("Word Document", "*.docx")])
        if not path:
            return

        doc = Document()

        title = doc.add_heading("CLINICAL CONSULTATION REPORT", level=0)
        title.alignment = 1  # Centered

        meta_table = doc.add_table(rows=3, cols=2)
        meta_table.style = "Table Grid"

        hdr_cells = meta_table.rows[0].cells
        hdr_cells[0].text = f"Patient Name: {self.current_patient}"
        hdr_cells[1].text = f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        row1_cells = meta_table.rows[1].cells
        row1_cells[0].text = f"Attending Doctor: {self.current_attending}"
        row1_cells[1].text = "Facility: Local Clinical Scribe"

        row2_cells = meta_table.rows[2].cells
        row2_cells[0].text = "Consultation Type: In-Person Scribe Session"
        row2_cells[1].text = "Status: Completed"

        doc.add_paragraph().paragraph_format.space_after = 12

        r = self.current_report

        doc.add_heading("1. Chief Complaint", level=2)
        doc.add_paragraph(r.get("chief_complaint") or "Not specified.")

        doc.add_heading("2. History & Subjective Notes", level=2)
        doc.add_paragraph(r.get("history_notes") or "No history details recorded.")

        doc.add_heading("3. Physical Examination & Vitals", level=2)
        doc.add_paragraph(r.get("examination_findings") or "Unremarkable / None noted.")

        doc.add_heading("4. Assessment & Diagnosis", level=2)
        doc.add_paragraph(r.get("diagnosis") or "Pending clinical review.")

        doc.add_heading("5. Prescriptions & Medications", level=2)
        prescriptions = r.get("prescriptions") or []
        if prescriptions:
            rx_table = doc.add_table(rows=1, cols=3)
            rx_table.style = "Table Grid"
            hdr = rx_table.rows[0].cells
            hdr[0].text = "Medication"
            hdr[1].text = "Dosage"
            hdr[2].text = "Instructions / Frequency"
            for item in prescriptions:
                row = rx_table.add_row().cells
                row[0].text = str(item.get("medication") or "—")
                row[1].text = str(item.get("dosage") or "—")
                row[2].text = str(item.get("instructions") or "—")
        else:
            doc.add_paragraph("No medications prescribed.")

        doc.add_heading("6. Diagnostic Tests & Labs Ordered", level=2)
        tests = r.get("tests_ordered") or []
        if tests:
            for test in tests:
                doc.add_paragraph(str(test), style="List Bullet")
        else:
            doc.add_paragraph("No additional diagnostic tests ordered.")

        doc.add_heading("7. Follow-up & Care Plan", level=2)
        doc.add_paragraph(r.get("follow_up") or "As needed.")

        if r.get("other_instructions"):
            doc.add_heading("8. Patient Care Instructions", level=2)
            doc.add_paragraph(r.get("other_instructions"))

        doc.save(path)
        self._log(f"Exported DOCX → {path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_reports_list(self):
        """Per-visit rows showing which artifacts exist (draft report always;
        confirmed/transcript/retained-audio only if present) with an Open
        button for each one that's actually on disk.
        """
        for w in self.reports_scroll.winfo_children():
            w.destroy()

        report_files = sorted(config.OUTPUT_DIR.glob("*_report.json"), reverse=True)
        if not report_files:
            ctk.CTkLabel(self.reports_scroll, text="No processed visits yet.",
                         text_color="gray").pack(pady=10)
            return

        suffix = "_report.json"
        for f in report_files:
            stem = f.name[:-len(suffix)] if f.name.endswith(suffix) else f.stem
            confirmed_path = config.OUTPUT_DIR / f"{stem}_report.confirmed.json"
            transcript_path = config.OUTPUT_DIR / f"{stem}_transcript.json"
            audio_matches = list(config.AUDIO_DIR.glob(f"{stem}.*"))
            audio_path = audio_matches[0] if audio_matches else None

            row = ctk.CTkFrame(self.reports_scroll)
            row.pack(fill="x", pady=3)

            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            ctk.CTkLabel(
                row, text=f"{stem}   ({mtime})",
                font=ctk.CTkFont(weight="bold"), anchor="w",
            ).pack(fill="x", padx=10, pady=(5, 2))

            btn_row = ctk.CTkFrame(row, fg_color="transparent")
            btn_row.pack(fill="x", padx=10, pady=(0, 5))

            self._add_artifact_button(btn_row, "Draft", f, True)
            self._add_artifact_button(btn_row, "Confirmed", confirmed_path, confirmed_path.exists())
            self._add_artifact_button(btn_row, "Transcript", transcript_path, transcript_path.exists())
            self._add_artifact_button(btn_row, "Audio", audio_path, audio_path is not None)

    def _add_artifact_button(self, parent, label: str, path, available: bool):
        mark = "✓" if available else "✗"
        btn = ctk.CTkButton(
            parent, text=f"{mark} {label}", width=110,
            state="normal" if available else "disabled",
            fg_color="#27ae60" if available else "gray30",
            command=(lambda p=path: self._open_path(p)) if available else None,
        )
        btn.pack(side="left", padx=(0, 5))

    def _open_path(self, path: Path):
        """Open a file or folder with its OS default handler."""
        import subprocess
        system = platform.system()
        if system == "Windows":
            os.startfile(str(path))
        elif system == "Darwin":
            subprocess.run(["open", str(path)])
        else:
            subprocess.run(["xdg-open", str(path)])

    def _log(self, msg: str):
        self.log_box.insert("end", f"{msg}\n")
        self.log_box.see("end")

    def _set_status(self, text: str):
        self.status.configure(text=text)


def main():
    app = MedicalScribeApp()
    app.mainloop()
