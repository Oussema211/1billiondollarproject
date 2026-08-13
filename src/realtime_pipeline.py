"""
Real-time transcription pipeline for live recordings & fast file processing.

Uses Whisper (STT) + SpeechBrain (speaker embeddings) processed on
audio chunks *during* the recording or in fast-pass for uploaded files.

Pyannote diarization is intentionally NOT used here — it requires heavy
neural segmentation and is far too slow for real-time / instant clinical use.

Speaker assignment:
  1. If an enrolled doctor profile exists  → cosine similarity lookup.
  2. Otherwise                             → online 2-cluster embedding
                                             tracker (first speaker = Doctor).
"""

import numpy as np
import wave
import tempfile
from pathlib import Path
import torch
import torchaudio


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


# ── Online speaker tracker ────────────────────────────────────────────────────

class OnlineSpeakerTracker:
    """Lightweight online 2-speaker tracker.

    Maintains a running mean embedding per speaker cluster and assigns
    each new chunk to the nearest cluster (or creates a second one).

    Works well for doctor–patient consultations where there are exactly
    two speakers and the doctor typically opens the consultation.
    """

    CLUSTER_SIM_THRESH = 0.60   # min cosine sim to assign to existing cluster
    PROFILE_SIM_THRESH = 0.65   # min cosine sim to accept an enrolled profile

    def __init__(self, doctor_name: str | None = None):
        self._doctor_name = doctor_name or "Attending Physician"
        self._clusters: dict[int, np.ndarray] = {}   # cid → mean embedding
        self._counts:   dict[int, int] = {}
        self._roles:    dict[int, dict] = {}          # cid → {role, name}
        self._next_cid = 0

    def _new_cluster(self, emb: np.ndarray, role: str, name: str | None) -> int:
        cid = self._next_cid
        self._next_cid += 1
        self._clusters[cid] = emb.copy()
        self._counts[cid] = 1
        self._roles[cid] = {"role": role, "name": name}
        return cid

    def _update(self, cid: int, emb: np.ndarray):
        n = self._counts[cid]
        self._clusters[cid] = (self._clusters[cid] * n + emb) / (n + 1)
        self._counts[cid] += 1

    def assign(self, emb: np.ndarray, profiles: dict) -> dict:
        """Return ``{"role": ..., "name": ...}`` for this chunk embedding."""
        # 1. Try enrolled profiles first
        if profiles:
            best_doc, best_score = None, -1.0
            for name, pemb in profiles.items():
                s = _cosine(emb, pemb)
                if s > best_score:
                    best_doc, best_score = name, s
            if best_score >= self.PROFILE_SIM_THRESH:
                if not self._clusters:
                    self._new_cluster(emb, "Doctor", best_doc)
                return {"role": "Doctor", "name": best_doc}

        # 2. Online clustering
        if not self._clusters:
            # First chunk → Doctor (doctors typically open the consultation)
            self._new_cluster(emb, "Doctor", self._doctor_name)
            return self._roles[0]

        sims = {cid: _cosine(emb, cemb) for cid, cemb in self._clusters.items()}
        best_cid = max(sims, key=sims.get)
        best_sim = sims[best_cid]

        if best_sim >= self.CLUSTER_SIM_THRESH:
            # Same speaker as an existing cluster
            self._update(best_cid, emb)
            return self._roles[best_cid]

        if len(self._clusters) < 2:
            # Second distinct voice → Patient
            cid = self._new_cluster(emb, "Patient", None)
            return self._roles[cid]

        # More than 2 speakers unexpected; assign to nearest
        self._update(best_cid, emb)
        return self._roles[best_cid]


# ── Chunk transcriber ─────────────────────────────────────────────────────────

class RealtimeTranscriber:
    """Transcribes audio chunks using pre-loaded Whisper + SpeechBrain models."""

    def __init__(self, whisper_model, embedder, temp_dir: Path,
                 doctor_name: str | None = None):
        self.whisper  = whisper_model
        self.embedder = embedder
        self.temp_dir = Path(temp_dir)
        self.tracker  = OnlineSpeakerTracker(doctor_name=doctor_name)

    def process_chunk(
        self,
        audio_np:    np.ndarray,  # float32 mono
        sample_rate: int,
        profiles:    dict,
        offset_s:    float = 0.0, # cumulative time so timestamps are absolute
    ) -> list:
        """Transcribe one chunk and return a list of labeled segment dicts."""
        if audio_np.size < int(sample_rate * 0.8):   # < 0.8s → skip
            return []

        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, dir=self.temp_dir)
        audio_i16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_i16.tobytes())
        tmp.close()

        try:
            segs, _ = self.whisper.transcribe(
                tmp.name,
                beam_size=1,
                best_of=1,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=250),
            )
            segs = list(segs)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

        if not segs:
            return []

        # Speaker embedding for the whole chunk
        waveform = torch.from_numpy(audio_np).float()
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        with torch.no_grad():
            emb = self.embedder.encode_batch(waveform.unsqueeze(0))
        emb = emb.squeeze().detach().cpu().numpy()

        role_info = self.tracker.assign(emb, profiles)

        return [
            {
                "start":       seg.start + offset_s,
                "end":         seg.end   + offset_s,
                "role":        role_info["role"],
                "doctor_name": role_info["name"],
                "text":        seg.text.strip(),
            }
            for seg in segs
            if seg.text.strip()
        ]

    def process_file_fast(
        self,
        audio_path: str,
        waveform: torch.Tensor,
        sample_rate: int,
        profiles: dict,
        progress_cb=None,
    ) -> list:
        """Fast file processor: Whisper full transcription + segment speaker tracking.
        
        Takes ~15-20s for a 15-minute file without slow Pyannote diarization.
        """
        cb = progress_cb or print
        cb("Fast transcribing with Whisper...")
        segs, info = self.whisper.transcribe(
            audio_path,
            beam_size=1,
            best_of=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
        )
        segs = list(segs)
        cb(f"Whisper finished: {len(segs)} segments found. Assigning speakers...")

        if sample_rate != 16000:
            w_16k = torchaudio.functional.resample(waveform, sample_rate, 16000)
        else:
            w_16k = waveform

        labeled = []
        for i, seg in enumerate(segs):
            text = seg.text.strip()
            if not text:
                continue
            start_f = int(seg.start * 16000)
            end_f   = int(seg.end * 16000)
            chunk   = w_16k[start_f:end_f]

            if chunk.numel() >= 8000:  # >= 0.5s for embedding
                with torch.no_grad():
                    emb = self.embedder.encode_batch(chunk.unsqueeze(0))
                emb_np = emb.squeeze().detach().cpu().numpy()
                role_info = self.tracker.assign(emb_np, profiles)
            else:
                # If too short, keep previous speaker role
                role_info = (
                    {"role": labeled[-1]["role"], "name": labeled[-1]["doctor_name"]}
                    if labeled
                    else {"role": "Doctor", "name": self.tracker._doctor_name}
                )

            labeled.append({
                "start": seg.start,
                "end": seg.end,
                "role": role_info["role"],
                "doctor_name": role_info["name"],
                "text": text,
            })

        return labeled
