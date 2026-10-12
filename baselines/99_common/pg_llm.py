"""
pg_llm.py — the chat model behind PaperGuard's verifier and the re-rating reviewer.

Default: gpt-4o-mini through the OpenAI Chat Completions API. PaperGuard's released code
used GPT-4o on Azure for its verifier (AZURE_OPENAI_DEPLOYMENT default "gpt-4o"), so
gpt-4o-mini keeps the verifier in the same model family; only the size changes.

The call is the plain REST request, sent with httpx, so it behaves the same as ft_judge.py's
gpt-4o-mini calls, which already work on the cluster. It does not depend on the openai SDK
version installed in the conda environment.
    * OPENAI_API_KEY    from the environment (the PBS sources path/to/.keys; never hard-coded)
    * OPENAI_BASE_URL   optional; default https://api.openai.com/v1
    * retries           429 / 5xx / transport errors, exponential backoff, and
                        Retry-After is honoured. A 400 (bad request) is NOT retried: it
                        will not succeed on a second attempt, and retrying only hides it.
    * one client per thread, so the job can keep several requests in flight.

backend="vllm" sends the same request to a local OpenAI-compatible vLLM server instead
(--llm-base-url http://127.0.0.1:<port>/v1, --llm <served model name>). The PBS does not
use it; it is here so a local model can be swapped in without touching the method code.
"""

from __future__ import annotations

import os
import random
import threading
import time
from typing import Dict, List, Optional


class LLMError(RuntimeError):
    pass


class ChatLLM:
    def __init__(self, model: str = "gpt-4o-mini", *, backend: str = "openai",
                 base_url: str = "", temperature: float = 0.0, max_tokens: int = 200,
                 seed: Optional[int] = 0, timeout: float = 180.0, max_retries: int = 8):
        self.model = model
        self.backend = backend
        if backend == "openai":
            self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                             or "https://api.openai.com/v1").rstrip("/")
            self.key = os.environ.get("OPENAI_API_KEY", "")
            if not self.key:
                raise LLMError("OPENAI_API_KEY is not set (the PBS sources path/to/.keys)")
        elif backend == "vllm":
            if not base_url:
                raise LLMError("backend 'vllm' needs --llm-base-url http://host:port/v1")
            self.base_url = base_url.rstrip("/")
            self.key = os.environ.get("VLLM_API_KEY", "")
        else:
            raise LLMError(f"unknown backend {backend!r}")
        self.temperature, self.max_tokens, self.seed = temperature, max_tokens, seed
        self.timeout, self.max_retries = timeout, max_retries
        self._tl = threading.local()
        self._lock = threading.Lock()
        self.n_calls = 0
        self.n_retries = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # ------------------------------------------------------------------ internals
    def _client(self):
        c = getattr(self._tl, "c", None)
        if c is None:
            import httpx
            # trust_env: the OpenAI API is reached through the cluster's proxy settings;
            # a local vLLM server must NOT be (the same trap ft_vllm.py documents).
            c = httpx.Client(timeout=self.timeout, trust_env=(self.backend == "openai"))
            self._tl.c = c
        return c

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.key}"} if self.key else {}

    # ------------------------------------------------------------------ public
    def chat(self, messages: List[Dict[str, str]], *, temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature if temperature is None else temperature,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens}
        if self.seed is not None:
            body["seed"] = int(self.seed)
        last = ""
        for attempt in range(self.max_retries):
            try:
                r = self._client().post(f"{self.base_url}/chat/completions",
                                        headers=self._headers(), json=body)
            except Exception as e:                         # transport
                last = f"{type(e).__name__}: {e}"
                self._backoff(attempt, None)
                continue
            if r.status_code == 200:
                d = r.json()
                u = d.get("usage") or {}
                with self._lock:
                    self.n_calls += 1
                    self.prompt_tokens += int(u.get("prompt_tokens") or 0)
                    self.completion_tokens += int(u.get("completion_tokens") or 0)
                return d["choices"][0]["message"].get("content") or ""
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code == 429 or r.status_code >= 500:
                self._backoff(attempt, r.headers.get("retry-after"))
                continue
            raise LLMError(last)                           # 400/401/403/404: not retryable
        raise LLMError(f"{self.model}: failed after {self.max_retries} attempts: {last}")

    def __call__(self, prompt: str, **kw) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kw)

    def _backoff(self, attempt: int, retry_after: Optional[str]) -> None:
        with self._lock:
            self.n_retries += 1
        try:
            wait = float(retry_after) if retry_after else 0.0
        except ValueError:
            wait = 0.0
        wait = max(wait, min(60.0, 2.0 ** attempt)) + random.uniform(0, 1.0)
        time.sleep(wait)

    def check(self) -> str:
        """One tiny call before the run starts: a bad key or model name fails in seconds."""
        return self.chat([{"role": "user", "content": "Reply with the single word OK."}],
                         temperature=0.0, max_tokens=5)

    def stats(self) -> Dict:
        return {"model": self.model, "backend": self.backend, "calls": self.n_calls,
                "retries": self.n_retries, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}
