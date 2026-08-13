"""LLM backend abstraction.

All report-generation call sites go through LLMBackend.generate() so the rest
of the pipeline (transcript grouping/truncation, JSON repair) never needs to
know which backend is active or what prompt format a given model expects.

Backend selection is a single config value (config.LLM_BACKEND), read by
build_llm_backend() — not a branch scattered across files.
"""
import os
from abc import ABC, abstractmethod
from pathlib import Path

from . import config


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.1, max_tokens: int = 512) -> str:
        """Run one completion and return the raw text response."""
        raise NotImplementedError


def _resolve_gguf_path() -> Path:
    ggufs = list(config.LLM_DIR.glob("*.gguf"))
    if not ggufs:
        raise FileNotFoundError(
            "No GGUF model in models/llm.\n"
            "Run setup_models.py first or place a .gguf file there."
        )
    if config.LLM_GGUF_FILENAME:
        match = config.LLM_DIR / config.LLM_GGUF_FILENAME
        if not match.exists():
            raise FileNotFoundError(
                f"config.LLM_GGUF_FILENAME={config.LLM_GGUF_FILENAME!r} "
                f"not found in {config.LLM_DIR}"
            )
        return match
    if len(ggufs) > 1:
        names = ", ".join(g.name for g in ggufs)
        raise RuntimeError(
            f"Multiple GGUF files found in models/llm ({names}) and "
            "config.LLM_GGUF_FILENAME is not set. Set it to the exact "
            "filename to load."
        )
    return ggufs[0]


class LocalGGUFBackend(LLMBackend):
    """Wraps llama-cpp-python. Uses the chat template embedded in the loaded
    GGUF itself (via create_chat_completion) rather than a hand-written
    prompt format — the model dictates its own tags, not this code.
    """

    def __init__(self, model_path: Path | None = None, n_ctx: int = 4096,
                 n_threads: int | None = None, n_gpu_layers: int = 0,
                 chat_format: str | None = None):
        from llama_cpp import Llama

        if model_path is None:
            model_path = _resolve_gguf_path()
        threads = n_threads or os.cpu_count() or 8
        chat_format = chat_format or config.LLM_LOCAL_CHAT_FORMAT

        print(f"Loading LLM: {model_path.name} | threads={threads} | ctx={n_ctx}")
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_batch=512,
            n_threads=threads,
            n_threads_batch=threads,
            n_gpu_layers=n_gpu_layers,
            chat_format=chat_format,  # None => llama-cpp-python auto-detects
            verbose=False,            #         from the GGUF's own metadata
        )

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.1, max_tokens: int = 512) -> str:
        resp = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"].strip()


class Phi3LegacyBackend(LLMBackend):
    """The original hardcoded Phi-3 chat-tag prompt path, preserved as-is for
    anyone who loads a Phi-3 GGUF. Not the default — see config.LLM_BACKEND.
    """

    def __init__(self, model_path: Path | None = None, n_ctx: int = 4096,
                 n_threads: int | None = None, n_gpu_layers: int = 0):
        from llama_cpp import Llama

        if model_path is None:
            model_path = _resolve_gguf_path()
        threads = n_threads or os.cpu_count() or 8

        print(f"Loading LLM (Phi-3 legacy format): {model_path.name} | threads={threads} | ctx={n_ctx}")
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_batch=512,
            n_threads=threads,
            n_threads_batch=threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.1, max_tokens: int = 512) -> str:
        prompt = (
            f"<|system|>\n{system_prompt}<|end|>\n"
            f"<|user|>\n{user_prompt}<|end|>\n"
            f"<|assistant|>\n"
        )
        out = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|end|>", "</s>", "<|user|>"],
        )
        return out["choices"][0]["text"].strip()


class APIBackend(LLMBackend):
    """Calls an OpenAI-compatible /chat/completions endpoint. Provider-agnostic:
    only base URL, an env var name for the API key, and a model name are
    configured — nothing here assumes a specific provider.
    """

    def __init__(self, base_url: str, api_key_env: str, model: str):
        if not base_url:
            raise ValueError("APIBackend requires config.LLM_API_BASE_URL to be set")
        if not model:
            raise ValueError("APIBackend requires config.LLM_API_MODEL to be set")
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.1, max_tokens: int = 512) -> str:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


def build_llm_backend() -> LLMBackend:
    """Single selection point for which backend is active — reads
    config.LLM_BACKEND. Nothing else in the codebase should branch on this.
    """
    backend = config.LLM_BACKEND

    if backend == "local_gguf":
        return LocalGGUFBackend()
    if backend == "phi3_legacy":
        return Phi3LegacyBackend()
    if backend == "api":
        return APIBackend(
            base_url=config.LLM_API_BASE_URL,
            api_key_env=config.LLM_API_KEY_ENV,
            model=config.LLM_API_MODEL,
        )
    raise ValueError(
        f"Unknown config.LLM_BACKEND={backend!r}; expected "
        "'local_gguf', 'phi3_legacy', or 'api'."
    )
