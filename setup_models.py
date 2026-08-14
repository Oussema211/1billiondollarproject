import os
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download, login

BASE = Path(__file__).parent
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

# WHISPER_MODEL_SIZE is read from src/config.py so this script and pipeline.py
# can never drift apart on which Whisper size is actually used at runtime.
from src.config import WHISPER_MODEL_SIZE
# config.py sets HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1 for runtime use, but this
# script's entire job is to reach the network — undo that for this process.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

# 1. HuggingFace login (one time) - skipped since already logged in
print("Already logged in to HuggingFace...")

# 2. Pyannote (downloads to local HF cache inside models/)
print("Downloading Pyannote diarization...")
os.environ["HF_HOME"] = str(MODELS / "hf_cache")
from pyannote.audio import Pipeline
_ = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")

# 3. SpeechBrain (auto-downloads on first use, but we pre-fetch)
print("Downloading SpeechBrain ECAPA model...")
snapshot_download(
    repo_id="speechbrain/spkrec-ecapa-voxceleb",
    local_dir=str(MODELS / "speechbrain"),
    local_dir_use_symlinks=False,
)

# 4. Whisper (size defined once in src/config.py — see WHISPER_MODEL_SIZE)
print(f"Downloading Whisper ({WHISPER_MODEL_SIZE})...")
from faster_whisper import WhisperModel
_ = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8", download_root=str(MODELS / "whisper"))

# 5. LLM — Qwen3-4B Instruct (Q4_K_M quantization, ~2.5 GB)
# The previous TheBloke/Llama-3.1-8B-Instruct-GGUF repo no longer exists
# (404 Repository Not Found as of 2026-08) and was never the app's actual
# target model anyway — LocalGGUFBackend was built around Qwen3's chat
# template. Repo/filename verified live against the HF API before pinning
# here, not guessed. 4B/Q4_K_M chosen over larger Qwen3 sizes as a
# reasonable fit for CPU-only, modest-RAM machines; bump to a larger size
# manually if running on stronger hardware.
print("Downloading local LLM (Qwen3-4B Q4_K_M)...")
LLM_DIR = MODELS / "llm"
LLM_DIR.mkdir(exist_ok=True)

hf_hub_download(
    repo_id="Qwen/Qwen3-4B-GGUF",
    filename="Qwen3-4B-Q4_K_M.gguf",
    local_dir=str(LLM_DIR),
    local_dir_use_symlinks=False,
)

print("\nAll models downloaded successfully!")
print(f"Total size: check {MODELS} folder (~3 GB expected with Qwen3-4B Q4_K_M)")
