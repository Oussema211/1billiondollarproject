import os
import sys
import json
import warnings
import numpy as np
import torch
import torchaudio

# Windows fix for SpeechBrain inspect recursion
try:
    import speechbrain.utils.importutils as _sb_iu
    _sb_orig_ensure = _sb_iu.LazyModule.ensure_module
    def _patched_sb_ensure_module(self, stacklevel=0):
        try:
            importer_frame = sys._getframe(stacklevel + 1)
            filename = importer_frame.f_code.co_filename
            if "inspect.py" in filename:
                raise AttributeError()
        except (ValueError, AttributeError):
            raise AttributeError()
        return _sb_orig_ensure(self, stacklevel)
    _sb_iu.LazyModule.ensure_module = _patched_sb_ensure_module
except Exception:
    pass

from speechbrain.utils.fetching import LocalStrategy

# Suppress all warnings from pyannote (torchcodec not used - we use torchaudio)
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.*")
# Suppress SpeechBrain deprecation warning
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*")

from pyannote.audio import Pipeline
from speechbrain.inference.speaker import EncoderClassifier
from faster_whisper import WhisperModel
from . import config

# ── Chunked diarization: process at most this many seconds at once to avoid
#    OOM on long recordings when the full waveform would not fit in RAM.
MAX_DIAR_CHUNK_S = 300  # 5 minutes per chunk


def load_audio(path: str):
    import soundfile as sf
    try:
        data, sr = sf.read(path, dtype='float32')
        waveform = torch.from_numpy(data)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=1, keepdim=True).T
        else:
            waveform = waveform.unsqueeze(0)
        return waveform.squeeze(0), sr
    except Exception:
        if str(path).lower().endswith('.wav'):
            import wave
            with wave.open(str(path), 'rb') as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                n_channels = wf.getnchannels()
                frames = wf.readframes(n_frames)
                audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                if n_channels > 1:
                    audio_np = audio_np.reshape(-1, n_channels).mean(axis=1)
                return torch.from_numpy(audio_np), sr
        raise


def get_embedding(waveform: torch.Tensor, sample_rate: int, model):
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    with torch.no_grad():
        embedding = model.encode_batch(waveform.unsqueeze(0))
    return embedding.squeeze().detach().cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def extract_speaker_audio(diarization, waveform: torch.Tensor, sample_rate: int) -> dict:
    segments = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = int(turn.start * sample_rate)
        end   = int(turn.end   * sample_rate)
        segments.setdefault(speaker, []).append(waveform[start:end])
    return {spk: torch.cat(chunks) for spk, chunks in segments.items()}


def _merge_diarizations(results: list, offsets: list):
    """Merge multiple pyannote Annotation objects with time offsets into one."""
    from pyannote.core import Annotation, Segment
    merged = Annotation()
    for ann, offset in zip(results, offsets):
        for turn, track, speaker in ann.itertracks(yield_label=True):
            shifted = Segment(turn.start + offset, turn.end + offset)
            # Use offset-prefixed track names to avoid collisions
            merged[shifted, f"{offset:.0f}_{track}"] = speaker
    return merged


class AudioPipeline:
    def __init__(self, progress_callback=None):
        self.progress = progress_callback or print
        self._load_models()

    def _load_models(self):
        self.progress("Loading diarization model (Pyannote)...")
        pyannote_cfg = config.MODELS_DIR / "pyannote" / "config.yaml"
        if pyannote_cfg.exists():
            self.diar = Pipeline.from_pretrained(str(pyannote_cfg))
        else:
            # Fallback: load from HF cache (populated during setup)
            self.diar = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1"
            )
        if config.DEVICE == "cuda":
            self.diar.to(torch.device("cuda"))
        self.progress("[OK] Diarization model loaded")

        self.progress("Loading speaker embedding model (SpeechBrain)...")
        self.embedder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(config.SPEECHBRAIN_DIR),
            run_opts={"device": config.DEVICE},
            local_strategy=LocalStrategy.COPY,
        )
        self.progress("[OK] Speaker embedding model loaded")

        self.progress("Loading Whisper transcription model (Base - faster)...")
        compute = "float16" if config.DEVICE == "cuda" else "int8"
        threads = min(8, os.cpu_count() or 4)
        self.whisper = WhisperModel(
            "base",
            device=config.DEVICE,
            compute_type=compute,
            cpu_threads=threads,
            num_workers=2,
            download_root=str(config.WHISPER_DIR),
        )
        self.progress("[OK] Whisper model loaded")

    def enroll_doctor(self, name: str, audio_path: str):
        w, sr = load_audio(audio_path)
        emb = get_embedding(w, sr, self.embedder)
        safe = name.strip().replace(" ", "_")
        np.save(config.PROFILES_DIR / f"{safe}.npy", emb)
        return safe

    def load_doctor_profiles(self) -> dict:
        profiles = {}
        for f in config.PROFILES_DIR.glob("*.npy"):
            profiles[f.stem.replace("_", " ")] = np.load(f)
        return profiles

    def identify_speakers(self, diarization, waveform, sample_rate, profiles):
        spk_audio = extract_speaker_audio(diarization, waveform, sample_rate)
        label_map = {}
        debug = {}

        if not profiles:
            # No profiles enrolled — assign Doctor to the speaker with the
            # most total audio (doctors typically dominate consultation time)
            most_spk = max(spk_audio, key=lambda s: spk_audio[s].numel(), default=None)
            for spk in spk_audio:
                if spk == most_spk:
                    label_map[spk] = {"role": "Doctor", "name": "Attending Physician"}
                else:
                    label_map[spk] = {"role": "Patient", "name": None}
            return label_map, {"info": "No enrolled profiles — role assigned by speaking time."}

        # Score every speaker against every enrolled profile
        scores = {}   # spk -> (best_doc, best_score)
        for spk, audio in spk_audio.items():
            if audio.numel() < sample_rate * 1:
                label_map[spk] = {"role": "Unknown", "name": None}
                continue
            emb = get_embedding(audio, sample_rate, self.embedder)
            best_doc, best_score = None, -1.0
            for doc_name, doc_emb in profiles.items():
                score = cosine_similarity(emb, doc_emb)
                if score > best_score:
                    best_doc, best_score = doc_name, score
            scores[spk] = (best_doc, best_score)
            debug[spk] = (best_doc, round(best_score, 3))

        # Assign roles for speakers that cleared the threshold
        matched_any = False
        for spk, (best_doc, best_score) in scores.items():
            if best_score >= config.SIMILARITY_THRESHOLD:
                label_map[spk] = {"role": "Doctor", "name": best_doc}
                matched_any = True
            else:
                label_map[spk] = {"role": "Patient", "name": None}

        # ── Fallback: no speaker cleared the threshold ────────────────────
        if not matched_any and scores:
            best_spk_by_score = max(scores, key=lambda s: scores[s][1])
            best_score_val = scores[best_spk_by_score][1]

            if best_score_val > 0.0:
                doc_name = scores[best_spk_by_score][0]
                label_map[best_spk_by_score] = {"role": "Doctor", "name": doc_name}
                debug["fallback"] = f"score below threshold ({best_score_val:.3f}), promoted best match"
            else:
                most_spk = max(spk_audio, key=lambda s: spk_audio[s].numel())
                label_map[most_spk] = {"role": "Doctor", "name": "Attending Physician"}
                debug["fallback"] = "all scores negative, used speaking-time heuristic"

        return label_map, debug

    def transcribe(self, audio_path: str):
        segments, info = self.whisper.transcribe(
            audio_path,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
        )
        return list(segments), info

    def merge_transcript(self, diarization, whisper_segments, label_map):
        turns = [(t.start, t.end, s) for t, _, s in diarization.itertracks(yield_label=True)]
        out = []
        for seg in whisper_segments:
            best_ov, best_spk = 0.0, "Unknown"
            for ts, te, spk in turns:
                ov = max(0.0, min(seg.end, te) - max(seg.start, ts))
                if ov > best_ov:
                    best_ov, best_spk = ov, spk
            info = label_map.get(best_spk, {"role": "Unknown", "name": None})
            out.append({
                "start": seg.start, "end": seg.end,
                "role": info["role"], "doctor_name": info["name"],
                "text": seg.text.strip()
            })
        return out

    def get_attending_doctor(self, labeled):
        names = [s["doctor_name"] for s in labeled if s["role"] == "Doctor" and s["doctor_name"]]
        return max(set(names), key=names.count) if names else "Unknown"

    def _diarize_chunked(self, waveform: torch.Tensor, sr: int):
        """Run pyannote diarization in MAX_DIAR_CHUNK_S chunks to keep RAM usage low.

        For audio shorter than the chunk size the original single-pass path is
        used so there is zero overhead for short recordings.
        """
        total_s = waveform.shape[0] / sr
        if total_s <= MAX_DIAR_CHUNK_S:
            # Short recording — single pass, no chunking overhead
            w_in = waveform.unsqueeze(0) if waveform.ndim == 1 else waveform
            result = self.diar({"waveform": w_in, "sample_rate": sr})
            return result.speaker_diarization if hasattr(result, "speaker_diarization") else result

        # Long recording — split, diarize each chunk, merge with time offsets
        self.progress(
            f"Audio is {total_s/60:.1f} min — running chunked diarization "
            f"({MAX_DIAR_CHUNK_S//60} min chunks)..."
        )
        chunk_frames = int(MAX_DIAR_CHUNK_S * sr)
        annotations = []
        offsets = []
        n_chunks = int(np.ceil(waveform.shape[0] / chunk_frames))

        for i in range(n_chunks):
            start_f = i * chunk_frames
            end_f   = min(start_f + chunk_frames, waveform.shape[0])
            chunk   = waveform[start_f:end_f]
            offset_s = start_f / sr

            self.progress(
                f"  Diarizing chunk {i+1}/{n_chunks} "
                f"({offset_s/60:.1f}–{end_f/sr/60:.1f} min)..."
            )
            w_in = chunk.unsqueeze(0) if chunk.ndim == 1 else chunk
            result = self.diar({"waveform": w_in, "sample_rate": sr})
            ann = result.speaker_diarization if hasattr(result, "speaker_diarization") else result
            annotations.append(ann)
            offsets.append(offset_s)

        self.progress("Merging diarization chunks...")
        return _merge_diarizations(annotations, offsets)

    def process(self, audio_path: str, patient_name: str = "N/A"):
        self.progress("Loading audio...")
        waveform, sr = load_audio(audio_path)
        audio_duration_s = waveform.shape[0] / sr

        self.progress(
            f"Audio loaded: {audio_duration_s/60:.1f} min "
            f"({waveform.shape[0]:,} samples @ {sr} Hz)"
        )

        self.progress("Running diarization...")
        diar = self._diarize_chunked(waveform, sr)

        self.progress("Identifying speakers...")
        profiles = self.load_doctor_profiles()
        label_map, debug = self.identify_speakers(diar, waveform, sr, profiles)
        self.progress(f"Speaker scores: {debug}")

        self.progress("Transcribing with Whisper...")
        segs, info = self.transcribe(audio_path)
        self.progress(f"Detected language: {info.language} ({info.language_probability:.2f})")

        self.progress("Aligning transcript...")
        labeled = self.merge_transcript(diar, segs, label_map)
        attending = self.get_attending_doctor(labeled)

        word_count = sum(len(s["text"].split()) for s in labeled)
        self.progress(f"Transcript ready: {len(labeled)} segments, ~{word_count} words")

        return {
            "diarization": diar,
            "labeled_transcript": labeled,
            "attending_doctor": attending,
            "patient_name": patient_name,
            "audio_duration_s": audio_duration_s,
            "word_count": word_count,
        }
