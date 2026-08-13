import sys
import os
from pathlib import Path

def get_base_dir():
    """Returns the folder where the EXE or script lives."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    return Path(__file__).parent.parent

BASE_DIR = get_base_dir()

# Local folders
MODELS_DIR      = BASE_DIR / "models"
PROFILES_DIR    = BASE_DIR / "doctor_profiles"
OUTPUT_DIR      = BASE_DIR / "output"
TEMP_DIR        = BASE_DIR / "temp"
AUDIO_DIR       = BASE_DIR / "audio"

for d in [MODELS_DIR, PROFILES_DIR, OUTPUT_DIR, TEMP_DIR, AUDIO_DIR]:
    d.mkdir(exist_ok=True)

# Force HuggingFace & Torch to use our local cache only
os.environ["HF_HOME"]               = str(MODELS_DIR / "hf_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR / "hf_cache")
os.environ["TORCH_HOME"]            = str(MODELS_DIR / "torch_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"]        = "1"
os.environ["TRANSFORMERS_OFFLINE"]  = "1"

# Sub-paths
WHISPER_DIR     = MODELS_DIR / "whisper"
SPEECHBRAIN_DIR = MODELS_DIR / "speechbrain"
LLM_DIR         = MODELS_DIR / "llm"

# Hardware
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIMILARITY_THRESHOLD = 0.70

# Whisper transcription model size — single source of truth read by both
# setup_models.py (download) and pipeline.py (runtime load). Must stay in
# sync between the two or setup fetches a model that's never used.
WHISPER_MODEL_SIZE = "base"

# ── LLM backend selection ────────────────────────────────────────────────
# "local_gguf"  — llama-cpp-python, chat template read from the loaded GGUF
# "api"         — OpenAI-compatible chat completions endpoint
# "phi3_legacy" — original hardcoded Phi-3 <|system|>/<|user|> tag format
LLM_BACKEND = "local_gguf"

# local_gguf backend options
LLM_GGUF_FILENAME = None       # exact filename in LLM_DIR to load; required if >1 .gguf present
LLM_LOCAL_CHAT_FORMAT = None   # optional llama-cpp-python built-in chat_format override (e.g. "chatml")

# api backend options
LLM_API_BASE_URL = None        # e.g. "https://api.example.com/v1"
LLM_API_KEY_ENV = "LLM_API_KEY"  # name of the env var holding the API key
LLM_API_MODEL = None           # model name to send in the request payload

# ── Traceability persistence (Phase 3) ──────────────────────────────────
# Whether to keep a permanent copy of processed audio in AUDIO_DIR, linked
# to its report/transcript by the same stem. Off by default — this matches
# today's behavior (mic recordings are deleted after processing, uploaded
# files are left wherever the user had them) and keeps the app's original
# fully-offline / no-retained-recordings posture unless a human explicitly
# opts in. Turning this on means actual patient conversations persist on
# disk indefinitely (no auto-delete/retention-policy exists) — a real
# privacy and storage-growth decision, not a default to flip casually.
RETAIN_RAW_AUDIO = False
