"""Ollama REST client.

Wraps the subset of the Ollama HTTP API we need:
  - GET  /api/tags             — list installed models / health check
  - POST /api/chat             — chat completion (streaming + non-streaming)
  - POST /api/generate         — single-prompt completion
  - POST /api/embeddings       — embed a single string

We always check Ollama is running before any LLM call and raise a friendly
`OllamaUnavailable` so the CLI can render a helpful message instead of a stack
trace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Iterator

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from bibwizard.utils.config import settings


class OllamaUnavailable(RuntimeError):
    """Raised when the Ollama server can't be reached."""


class OllamaModelMissing(RuntimeError):
    """Raised when a required model isn't installed on the Ollama server."""


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class OllamaClient:
    def __init__(
        self,
        host: str | None = None,
        llm_model: str | None = None,
        embed_model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.host = (host or settings.ollama_host).rstrip("/")
        self.llm_model = llm_model or settings.ollama_llm_model
        self.embed_model = embed_model or settings.ollama_embed_model
        self.timeout = timeout or settings.ollama_timeout

    # ---------- health ----------

    def is_running(self) -> bool:
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{self.host}/api/tags")
                return resp.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def list_models(self) -> list[str]:
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.host}/api/tags")
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            raise OllamaUnavailable(
                f"Could not reach Ollama at {self.host}: {e}"
            ) from e
        return [m.get("name", "") for m in data.get("models", [])]

    def ensure_ready(self, *, need_llm: bool = True, need_embed: bool = False) -> None:
        """Raise a helpful error if Ollama / required models are missing."""
        if not self.is_running():
            raise OllamaUnavailable(
                f"Ollama doesn't appear to be running at {self.host}.\n"
                f"  • Start it with: `ollama serve`\n"
                f"  • Or set OLLAMA_HOST in your .env to the right URL."
            )
        installed = {m.split(":")[0]: m for m in self.list_models()}
        full_installed = set(self.list_models())

        def _present(model: str) -> bool:
            return model in full_installed or model.split(":")[0] in installed

        missing = []
        if need_llm and not _present(self.llm_model):
            missing.append(self.llm_model)
        if need_embed and not _present(self.embed_model):
            missing.append(self.embed_model)
        if missing:
            pulls = "\n".join(f"  ollama pull {m}" for m in missing)
            raise OllamaModelMissing(
                "Required Ollama model(s) not installed:\n"
                + "\n".join(f"  • {m}" for m in missing)
                + f"\nPull them with:\n{pulls}"
            )

    # ---------- chat ----------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(httpx.HTTPError),
        reraise=True,
    )
    def chat(
        self,
        messages: Iterable[ChatMessage | dict],
        *,
        model: str | None = None,
        stream: bool = False,
        options: dict | None = None,
        format: str | None = None,
    ) -> str | Iterator[str]:
        payload: dict = {
            "model": model or self.llm_model,
            "messages": [
                m.to_dict() if isinstance(m, ChatMessage) else m for m in messages
            ],
            "stream": stream,
        }
        if options:
            payload["options"] = options
        if format:
            # Ollama's `format` field constrains generation. "json" forces
            # the response to be a single valid JSON document.
            payload["format"] = format

        if stream:
            return self._stream_chat(payload)

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.host}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data.get("message", {}).get("content", "")

    def _stream_chat(self, payload: dict) -> Iterator[str]:
        with httpx.stream(
            "POST", f"{self.host}/api/chat", json=payload, timeout=self.timeout
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if chunk.get("done"):
                    return
                token = chunk.get("message", {}).get("content")
                if token:
                    yield token

    # ---------- generate (single-shot) ----------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(httpx.HTTPError),
        reraise=True,
    )
    def generate(self, prompt: str, *, model: str | None = None, options: dict | None = None) -> str:
        payload = {
            "model": model or self.llm_model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.host}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")

    # ---------- embeddings ----------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(httpx.HTTPError),
        reraise=True,
    )
    def embed(self, text: str, *, model: str | None = None) -> list[float]:
        payload = {
            "model": model or self.embed_model,
            "prompt": text,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.host}/api/embeddings", json=payload)
            resp.raise_for_status()
            data = resp.json()
        emb = data.get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError(f"Bad embedding response from Ollama: {data}")
        return emb

    def embed_batch(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        # Ollama's /api/embeddings is single-text; we just loop.
        return [self.embed(t, model=model) for t in texts]


# Convenience module-level singleton
_default_client: OllamaClient | None = None


def get_client() -> OllamaClient:
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient()
    return _default_client
