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

for d in [MODELS_DIR, PROFILES_DIR, OUTPUT_DIR, TEMP_DIR]:
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
