import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import queue
from pathlib import Path
from docx import Document
import sounddevice as sd
import numpy as np
import json
import wave
import tempfile

from .pipeline import AudioPipeline
from .report_generator import ReportGenerator
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
        self.current_transcript: list = []
        self.current_attending = ""
        self.current_patient = ""

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
        self.status.pack(side="bottom", pady=20)

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

        # ── Tabview: Transcript | Report ─────────────────────────────────────
        self.result_tabs = ctk.CTkTabview(self.tab_process)
        self.result_tabs.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=20, pady=5)

        self.result_tabs.add("Transcript")
        self.result_tabs.add("Report")

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

        # Report tab
        rp_frame = self.result_tabs.tab("Report")
        rp_frame.grid_columnconfigure(0, weight=1)
        rp_frame.grid_rowconfigure(0, weight=1)

        self.report_box = ctk.CTkTextbox(rp_frame)
        self.report_box.grid(row=0, column=0, sticky="nsew")

        ctk.CTkButton(
            rp_frame, text="Export DOCX",
            command=self._export_docx, width=140,
        ).grid(row=1, column=0, pady=5)

        # Row/column weights
        self.tab_process.grid_columnconfigure(1, weight=1)
        self.tab_process.grid_rowconfigure(5, weight=1)

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
            command=lambda: self._open_folder(config.OUTPUT_DIR),
        ).pack(pady=20)
        self.reports_list = ctk.CTkTextbox(self.tab_reports)
        self.reports_list.pack(fill="both", expand=True, padx=20, pady=10)
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
            self.report_gen = ReportGenerator()
            self._set_status(f"Ready | Device: {config.DEVICE}")
            self._log("All models loaded. Ready.")
        except Exception as e:
            self._set_status(f"Error: {e}")
            messagebox.showerror("Model Load Failed", str(e))

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
            self.report_box.delete("1.0", "end")
            self.transcript_box.delete("1.0", "end")

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

            # ── 2. Show transcript immediately ────────────────────────────
            self._populate_transcript(labeled)
            self.result_tabs.set("Transcript")   # switch to transcript view

            dur_min = result.get("audio_duration_s", 0) / 60
            wc = result.get("word_count", 0)
            self._log(f"Transcript complete — {dur_min:.1f} min audio, ~{wc} words")

            # ── 3. Generate report with local LLM ─────────────────────────
            self._log("Generating medical report with local LLM…")
            report = self.report_gen.extract(labeled)
            self.current_report = report

            text = self._render_text(report, result["patient_name"], result["attending_doctor"])
            self.report_box.insert("1.0", text)
            self.result_tabs.set("Report")       # switch to report view

            # ── 4. Save JSON ──────────────────────────────────────────────
            stem = Path(audio_path).stem
            out_json = config.OUTPUT_DIR / f"{stem}_report.json"
            out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
            self._log(f"Saved JSON → {out_json}")

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
        if not self.current_report:
            messagebox.showwarning("Warning", "No report available to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".docx", filetypes=[("Word Document", "*.docx")])
        if not path:
            return

        from datetime import datetime
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

    def _render_text(self, r, patient, doctor) -> str:
        lines = [
            "=" * 60,
            "               CLINICAL CONSULTATION REPORT",
            "=" * 60,
            f"Patient Name:     {patient}",
            f"Attending Doctor: {doctor}",
            "-" * 60,
            "CHIEF COMPLAINT:",
            f"  {r.get('chief_complaint') or '—'}",
            "",
            "HISTORY & SUBJECTIVE NOTES:",
            f"  {r.get('history_notes') or '—'}",
            "",
            "PHYSICAL EXAMINATION & VITALS:",
            f"  {r.get('examination_findings') or '—'}",
            "",
            "DIAGNOSIS & ASSESSMENT:",
            f"  {r.get('diagnosis') or '—'}",
            "",
            "DIAGNOSTIC TESTS ORDERED:",
        ]
        tests = r.get("tests_ordered") or []
        lines += [f"  • {t}" for t in tests] if tests else ["  • None"]

        lines += ["", "PRESCRIPTIONS:"]
        prescriptions = r.get("prescriptions") or []
        if prescriptions:
            for p in prescriptions:
                lines.append(
                    f"  • {p.get('medication')} "
                    f"| Dosage: {p.get('dosage') or '—'} "
                    f"| Instructions: {p.get('instructions') or '—'}"
                )
        else:
            lines.append("  • None")

        lines += [
            "",
            "FOLLOW-UP & CARE PLAN:",
            f"  {r.get('follow_up') or '—'}",
        ]
        if r.get("other_instructions"):
            lines += ["", "OTHER INSTRUCTIONS:", f"  {r.get('other_instructions')}"]

        lines.append("=" * 60)
        return "\n".join(lines)

    def _refresh_reports_list(self):
        self.reports_list.delete("1.0", "end")
        files = sorted(config.OUTPUT_DIR.glob("*_report.json"))
        self.reports_list.insert("1.0", f"Generated Reports ({len(files)}):\n\n")
        for f in files:
            self.reports_list.insert("end", f"• {f.name}\n")

    def _open_folder(self, folder: Path):
        import platform
        import subprocess
        p = platform.system()
        if p == "Windows":
            subprocess.run(["explorer", str(folder)])
        elif p == "Darwin":
            subprocess.run(["open", str(folder)])
        else:
            subprocess.run(["xdg-open", str(folder)])

    def _log(self, msg: str):
        self.log_box.insert("end", f"{msg}\n")
        self.log_box.see("end")

    def _set_status(self, text: str):
        self.status.configure(text=text)


def main():
    app = MedicalScribeApp()
    app.mainloop()
