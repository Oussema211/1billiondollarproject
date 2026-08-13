import os
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download, login

BASE = Path(__file__).parent
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

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

# 4. Whisper Large-V3
print("Downloading Whisper Large-V3...")
from faster_whisper import WhisperModel
_ = WhisperModel("large-v3", device="cpu", compute_type="int8", download_root=str(MODELS / "whisper"))

# 5. LLM — Llama 3.1 8B Instruct (Q4_K_M quantization, ~4.7 GB)
print("Downloading local LLM (Llama 3.1 8B Q4_K_M)...")
LLM_DIR = MODELS / "llm"
LLM_DIR.mkdir(exist_ok=True)

# Using TheBloke's GGUF mirror (standard, well-tested)
hf_hub_download(
    repo_id="TheBloke/Llama-3.1-8B-Instruct-GGUF",
    filename="llama-3.1-8b-instruct.Q4_K_M.gguf",
    local_dir=str(LLM_DIR),
    local_dir_use_symlinks=False,
)

print("\nAll models downloaded successfully!")
print(f"Total size: check {MODELS} folder (~10 GB expected)")
