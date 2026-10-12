"""
v8_llm.py — one interface over the backends: a local vLLM server (with per-agent LoRA
routing) or the OpenAI API (gpt-4o-mini). Plus the reviewer.

    llm.chat(messages, model=..., temperature=..., max_tokens=..., n=...) -> List[str]

WHY THE ROUTING KEY IS (task, agent)
    v8 serves up to two adapter families per ablation: `detect` (LOCATOR, EVIDENCE,
    VERIFIER, LEADER) and `localize` (LEADER in repair mode). vLLM selects a LoRA through
    the `model` field, so every call names the adapter it needs. The REVIEWER is always the
    BASE model: it is the system under attack, and training it would change the thing
    whose manipulation is being measured.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import ft_config as C
import v8_agents as A

try:
    import httpx
except Exception:                                   # pragma: no cover
    httpx = None


class LLMError(RuntimeError):
    pass


class _Fatal(Exception):
    pass


# ======================================================================================
# backends
# ======================================================================================
class Backend:
    name = "base"

    def __init__(self):
        self._lock = threading.Lock()
        self.n_calls = 0

    def _bump(self, k: int = 1) -> None:
        with self._lock:
            self.n_calls += k

    def chat(self, messages: List[Dict], *, model: str = "", temperature: float = 0.0,
             max_tokens: int = 600, n: int = 1, seed: Optional[int] = None) -> List[str]:
        raise NotImplementedError

    def models(self) -> List[str]:
        return []


class VLLMBackend(Backend):
    """ft_vllm.VLLMClient underneath: trust_env=False, 404 names what IS served."""
    name = "vllm"

    def __init__(self, url: str, base_model: str, chat_kwargs: Optional[Dict] = None,
                 timeout: float = 900.0):
        super().__init__()
        import ft_vllm as V
        self.cl = V.VLLMClient(url, base_model, timeout=timeout, chat_kwargs=chat_kwargs)
        self.base = base_model

    def chat(self, messages, *, model="", temperature=0.0, max_tokens=600, n=1, seed=None):
        out = self.cl.chat(messages, model=model or self.base, temperature=temperature,
                           max_tokens=max_tokens, n=n, seed=seed)
        self._bump()
        return out

    def models(self) -> List[str]:
        return self.cl.models()

    def require(self, names: Sequence[str]) -> None:
        self.cl.require(list(names))


class OpenAIBackend(Backend):
    """
    gpt-4o-mini over HTTPS. The key comes from the environment (source path/to/.keys in the
    PBS); it is never written to a file this package produces.
    """
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", timeout: float = 180.0,
                 max_retries: int = 6):
        super().__init__()
        if httpx is None:
            raise LLMError("httpx is not installed")
        self.key = os.environ.get("OPENAI_API_KEY", "")
        if not self.key:
            raise LLMError("OPENAI_API_KEY is not set. Put it in path/to/.keys (chmod 600) and "
                           "source it from the PBS -- never on an export line in the PBS.")
        self.model = model
        self.base = model
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.max_retries = max_retries
        self._c = httpx.Client(timeout=timeout)

    def chat(self, messages, *, model="", temperature=0.0, max_tokens=600, n=1, seed=None):
        body = {"model": model or self.model, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens, "n": max(1, n)}
        if seed is not None:
            body["seed"] = int(seed)
        last = None
        for attempt in range(self.max_retries):
            try:
                r = self._c.post(f"{self.base_url}/chat/completions", json=body,
                                 headers={"Authorization": f"Bearer {self.key}"})
                if r.status_code in (429, 500, 502, 503, 504):
                    raise LLMError(f"HTTP {r.status_code}")
                if 400 <= r.status_code < 500:        # bad key / bad request: retrying won't help
                    raise _Fatal(f"HTTP {r.status_code}: {r.text[:300]}")
                r.raise_for_status()
                self._bump()
                return [c["message"]["content"] or "" for c in r.json()["choices"]]
            except _Fatal as e:
                raise LLMError(f"OpenAI rejected the request: {e}")
            except Exception as e:
                last = e
                time.sleep(min(60.0, 2.0 * (2 ** attempt)) * (0.5 + random.random()))
        raise LLMError(f"OpenAI call failed after {self.max_retries} attempts: {last}")

    def models(self) -> List[str]:
        return [self.model]


def make_backend(model_key: str, url: str = "") -> Backend:
    spec = C.MODELS[model_key]
    if spec.api:
        return OpenAIBackend(model_key)
    url = url or os.environ.get("VLLM_URL", "")
    if not url:
        raise LLMError("VLLM_URL is not set; the PBS starts the server and exports it")
    served = os.environ.get("VLLM_MODEL", "") or spec.path
    return VLLMBackend(url, served, chat_kwargs=spec.chat_kwargs)


# ======================================================================================
# routing: (task, agent) -> served model / adapter name
# ======================================================================================
@dataclass
class Routing:
    base: str
    table: Dict[str, str] = field(default_factory=dict)

    def name(self, task: str, agent: str) -> str:
        return (self.table.get(f"{task}:{agent}") or self.table.get(f"{task}:*")
                or self.base)

    def adapters(self) -> List[str]:
        return sorted({v for v in self.table.values() if v != self.base})

    def describe(self) -> str:
        if not self.table:
            return f"  every agent -> {self.base} (zero-shot)"
        keys = [f"detect:{a}" for a in C.TASK_AGENTS["detect"]] + ["localize:LEADER"]
        return "\n".join(f"  {k:<18} -> {self.name(*k.split(':'))}" for k in keys)


def build_routing(model_key: str, ablation: str, base_served: str,
                  adapter_root: str = "") -> Routing:
    ab = C.ABL.get(ablation)
    if ab is None:
        raise SystemExit(f"[FATAL] unknown ablation {ablation!r}; one of {list(C.ABL)}")
    if ab.stage is None or C.MODELS[model_key].api:
        return Routing(base_served)
    tab: Dict[str, str] = {}
    for task in C.TASKS:
        stage = f"{ab.stage}-{task}"
        if ab.per_agent:
            for ag in C.TASK_AGENTS[task]:
                tab[f"{task}:{ag}"] = C.lora_name(model_key, stage, ag)
        else:
            tab[f"{task}:*"] = C.lora_name(model_key, stage)
    return Routing(base_served, tab)


# ======================================================================================
# the reviewer — base model, benchmark prompt when available
# ======================================================================================
class Reviewer:
    def __init__(self, llm: Backend, served: str, *, system_role: bool = True,
                 max_tokens: int = 900, max_chars: int = 32000):
        self.llm = llm
        self.served = served
        self.system_role = system_role
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.system, self.instruction, self.source = A.reviewer_prompt()

    def _fit(self, doc: str) -> str:
        """Over-long: keep the head AND the tail. Insertions live in the tail, so cutting
        from the end would silently remove the attack being measured."""
        if len(doc) <= self.max_chars:
            return doc
        h = self.max_chars // 3
        return doc[:h] + "\n\n[...]\n\n" + doc[-(self.max_chars - h):]

    def review(self, doc: str) -> Dict:
        user = self._fit(doc or "") + "\n\n" + self.instruction
        msgs = A.messages(self.system, user, system_role=self.system_role)
        out = {"rating": None, "decision": "", "weaknesses": [], "summary": "", "raw": "",
               "calls": 0, "error": ""}
        for attempt in range(2):
            try:
                raw = self.llm.chat(msgs, model=self.served, temperature=0.0,
                                    max_tokens=self.max_tokens)[0]
            except Exception as e:
                out["error"] = f"{type(e).__name__}: {e}"[:300]
                out["calls"] += 1
                continue
            out["calls"] += 1
            rv = A.parse_review(raw)
            out.update({k: rv[k] for k in ("rating", "decision", "weaknesses", "summary",
                                           "raw")})
            if rv["rating"] is not None:
                break
            msgs = A.messages(self.system, user + "\n\nIMPORTANT: include a line "
                              "'RATING: <integer 1-10>' and a line 'DECISION: Accept' or "
                              "'DECISION: Reject'.", system_role=self.system_role)
        return out
