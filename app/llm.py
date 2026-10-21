"""
Local LLM access via fox-services.

fox-services (http://localhost:8210) wraps the whole Ollama service and adds:
  * model routing  - "fox-managed" is resolved per request to a loaded model
  * concurrency control - per-model slots, single global large-model load lock
  * queue visibility + telemetry (token counts, duration)

We talk to it over its OpenAI-shaped / Ollama-shaped HTTP surface so that the
same code path works whether the gateway is up (preferred) or not (direct
Ollama fallback on :11434).

Two call shapes are used by the lab:
  chat()  - text in / text out, used for agent rollouts and evolver proposals
  embed() - vectors, used by the similarity-retrieval evolvers
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx

FOX_SERVICES_URL = os.environ.get("FOX_SERVICES_URL", "http://localhost:8210").rstrip("/")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("SLA_MODEL", "fox-managed")
DEFAULT_EMBED_MODEL = os.environ.get("SLA_EMBED_MODEL", "fox-managed")

# Request timeout. Local 27B-31B models on a shared GPU are slow; a full
# ReAct rollout with tool feedback can legitimately take minutes.
CHAT_TIMEOUT = float(os.environ.get("SLA_CHAT_TIMEOUT", "600"))
EMBED_TIMEOUT = float(os.environ.get("SLA_EMBED_TIMEOUT", "120"))


class LLMError(RuntimeError):
    """Raised when neither fox-services nor Ollama can serve a request."""


class LLMClient:
    """Thin async client over fox-services with a direct-Ollama fallback."""

    def __init__(self, fox_url: str = FOX_SERVICES_URL, ollama_url: str = OLLAMA_URL):
        self.fox_url = fox_url
        self.ollama_url = ollama_url
        self._client: Optional[httpx.AsyncClient] = None
        self._embed_cache: Dict[str, List[float]] = {}

    async def __aenter__(self) -> "LLMClient":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(CHAT_TIMEOUT, connect=10.0),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=8),
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(CHAT_TIMEOUT, connect=10.0),
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=8),
            )
        return self._client

    # ------------------------------------------------------------------ utils

    async def _post(self, base: str, path: str, payload: Dict[str, Any],
                    timeout: Optional[float] = None) -> Dict[str, Any]:
        resp = await self.client.post(
            f"{base}{path}", json=payload, timeout=timeout or CHAT_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()

    async def health(self) -> Dict[str, Any]:
        """Which backends are reachable, and what models they offer."""
        out: Dict[str, Any] = {"fox_services": False, "ollama": False, "models": []}
        try:
            r = await self.client.get(f"{self.fox_url}/api/ollama/models", timeout=8.0)
            if r.status_code == 200:
                out["fox_services"] = True
                data = r.json()
                out["models"] = [
                    {"name": m.get("name"), "size": m.get("size"),
                     "params": (m.get("details") or {}).get("parameter_size"),
                     "capabilities": m.get("capabilities", [])}
                    for m in data.get("models", [])
                ]
        except Exception:
            pass
        if not out["models"]:
            try:
                r = await self.client.get(f"{self.ollama_url}/api/tags", timeout=8.0)
                if r.status_code == 200:
                    out["ollama"] = True
                    data = r.json()
                    out["models"] = [
                        {"name": m.get("name"), "size": m.get("size"),
                         "params": (m.get("details") or {}).get("parameter_size"),
                         "capabilities": m.get("capabilities", [])}
                        for m in data.get("models", [])
                    ]
            except Exception:
                pass
        out["reachable"] = out["fox_services"] or out["ollama"]
        return out

    # ------------------------------------------------------------------- chat

    async def chat(self, messages: List[Dict[str, str]], model: str = DEFAULT_MODEL,
                   temperature: float = 0.0, format: Optional[str] = None,
                   timeout: Optional[float] = None) -> Dict[str, Any]:
        """One chat completion. Returns {content, model, tokens, duration_ms}."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if format:
            payload["format"] = format

        started = time.time()
        try:
            data = await self._post(self.fox_url, "/api/chat", payload, timeout)
        except Exception as fox_err:
            try:
                data = await self._post(self.ollama_url, "/api/chat", payload, timeout)
            except Exception as ollama_err:
                raise LLMError(
                    f"fox-services failed ({fox_err}); Ollama failed ({ollama_err})"
                ) from ollama_err

        message = data.get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):  # multimodal shapes
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        return {
            "content": content,
            "thinking": message.get("thinking", ""),
            "model": data.get("model", model),
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "duration_ms": int((time.time() - started) * 1000),
        }

    async def complete(self, prompt: str, model: str = DEFAULT_MODEL,
                       temperature: float = 0.0, format: Optional[str] = None,
                       timeout: Optional[float] = None) -> str:
        res = await self.chat(
            [{"role": "user", "content": prompt}],
            model=model, temperature=temperature, format=format, timeout=timeout,
        )
        return res["content"]

    async def json_complete(self, prompt: str, model: str = DEFAULT_MODEL,
                            temperature: float = 0.0, timeout: Optional[float] = None) -> Any:
        """Ask for JSON and parse it, tolerating markdown fences."""
        raw = await self.complete(
            prompt, model=model, temperature=temperature, format="json", timeout=timeout
        )
        return parse_json_loose(raw)

    # ------------------------------------------------------------------ embed

    async def embed(self, texts: List[str], model: str = DEFAULT_EMBED_MODEL) -> List[List[float]]:
        """Embed a batch of texts. Falls back to a hashed bag-of-words vector."""
        if not texts:
            return []
        try:
            data = await self._post(
                self.fox_url, "/api/embed",
                {"model": model, "input": texts}, EMBED_TIMEOUT,
            )
            vectors = data.get("embeddings")
            if vectors:
                return vectors
        except Exception:
            pass
        try:
            data = await self._post(
                self.ollama_url, "/api/embed",
                {"model": model, "input": texts}, EMBED_TIMEOUT,
            )
            vectors = data.get("embeddings")
            if vectors:
                return vectors
        except Exception:
            pass
        return [hashed_embedding(t) for t in texts]

    async def embed_one(self, text: str, model: str = DEFAULT_EMBED_MODEL) -> List[float]:
        key = f"{model}::{text}"
        if key in self._embed_cache:
            return self._embed_cache[key]
        vectors = await self.embed([text], model=model)
        vec = vectors[0] if vectors else hashed_embedding(text)
        self._embed_cache[key] = vec
        return vec


# --------------------------------------------------------------------- utils

def parse_json_loose(raw: str) -> Any:
    """Parse JSON out of a model response, tolerating fences and prose."""
    if raw is None:
        return None
    text = raw.strip()
    if text.startswith("```"):
        # ```json ... ``` or ``` ... ```
        if "\n" in text:
            text = text.split("\n", 1)[1]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fall back to the outermost {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        i = text.find(opener)
        if i == -1:
            continue
        j = text.rfind(closer)
        if j > i:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                continue
    return None


def hashed_embedding(text: str, dim: int = 256) -> List[float]:
    """
    Deterministic fallback embedding: hashed bag of words.

    Not semantic, but stable and dimension-consistent, so similarity retrieval
    still produces a usable (if topical-only) neighbourhood. Used only when no
    embedding model is reachable.
    """
    import hashlib
    vec = [0.0] * dim
    tokens = [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]
    for tok in tokens:
        h = hashlib.blake2b(tok.encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if h[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = sum(a[i] * a[i] for i in range(n)) ** 0.5
    nb = sum(b[i] * b[i] for i in range(n)) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


_client_singleton: Optional[LLMClient] = None


def get_client() -> LLMClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = LLMClient()
    return _client_singleton
