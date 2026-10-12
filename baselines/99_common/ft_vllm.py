"""
ft_vllm.py — the one HTTP client every stage uses to talk to vLLM.

WHY ONE CLIENT
=======================================================================================
Four programs here call a served model, and three details have already cost this project
a debugging session each:

  1. trust_env=False. httpx reads HTTP_PROXY/HTTPS_PROXY from the environment by default.
     On this cluster those are set, so a request to 127.0.0.1 is handed to the proxy,
     which cannot route it. The failure is a timeout, not a proxy error, so it reads like
     the server never came up.
  2. The `model` field must EXACTLY equal the server's --served-model-name. serve_vllm.sh
     serves under the model DIRECTORY path; a multi-LoRA server additionally answers to
     each adapter name. A near-miss is a 404 that looks like a model problem.
  3. LoRA selection IS the model field. There is no separate adapter parameter — you ask
     for `gemma3-27b-it-dpo-leader` and vLLM applies that adapter. Which means a typo
     silently... does not silently do anything, it 404s, and that is the good case. The
     bad case is asking for the BASE name when you meant an adapter: you get a perfectly
     valid answer from the untrained model and the ablation measures nothing. `require`
     exists for that: it fails if the name you want is not on the server.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence

try:
    import httpx
except Exception:                                    # pragma: no cover
    httpx = None


class VLLMError(RuntimeError):
    pass


class VLLMClient:
    def __init__(self, url: str, model: str, timeout: float = 600.0,
                 max_retries: int = 3, temperature: float = 0.0,
                 top_p: float = 1.0, max_tokens: int = 700, seed: Optional[int] = None,
                 chat_kwargs: Optional[Dict] = None):
        if httpx is None:
            raise VLLMError("httpx is not installed in this environment")
        self.url = url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        # chat_template_kwargs reaches the server's template renderer. For qwen3 this is
        # {"enable_thinking": False}: without it the model spends the whole token budget
        # inside <think>...</think>, the JSON is never emitted, and every completion scores
        # as a parse failure rather than as a wrong answer.
        self.chat_kwargs = dict(chat_kwargs or {})
        self.max_retries = max_retries
        # trust_env=False: see note 1 above.
        self._c = httpx.Client(timeout=timeout, trust_env=False)
        self.n_calls = 0
        self.n_tokens_out = 0

    # ---------------------------------------------------------------- introspection
    def models(self) -> List[str]:
        r = self._c.get(f"{self.url}/models")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    def require(self, names: Sequence[str]) -> None:
        """
        Fail NOW if an adapter the run depends on is not served.

        Asking for a name the server does not have 404s, which is recoverable. Asking for
        the base model when you meant an adapter returns a valid answer from an untrained
        model, and the ablation row then reports the baseline under a fine-tuned label.
        That one is not recoverable after the fact, so it is checked up front.
        """
        have = set(self.models())
        missing = [n for n in names if n not in have]
        if missing:
            raise VLLMError(
                f"these model/adapter names are NOT served: {missing}\n"
                f"  served: {sorted(have)}\n"
                f"  Start the server with --enable-lora and a --lora-modules entry for "
                f"each adapter (serve_vllm_lora.sh does this from the adapters directory).")

    # ---------------------------------------------------------------- generation
    def chat(self, messages: List[Dict[str, str]], *, model: str = "",
             temperature: Optional[float] = None, max_tokens: Optional[int] = None,
             top_p: Optional[float] = None, seed: Optional[int] = None,
             n: int = 1, stop: Optional[Sequence[str]] = None) -> List[str]:
        body = {
            "model": model or self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p if top_p is None else top_p,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "n": max(1, int(n)),
        }
        s = self.seed if seed is None else seed
        if s is not None:
            body["seed"] = int(s)
        if self.chat_kwargs:
            body["chat_template_kwargs"] = self.chat_kwargs
        if stop:
            body["stop"] = list(stop)

        last = None
        for attempt in range(self.max_retries):
            try:
                r = self._c.post(f"{self.url}/chat/completions", json=body)
                if r.status_code == 404:
                    raise VLLMError(
                        f"404 for model {body['model']!r}. Served: {self.models()}")
                r.raise_for_status()
                d = r.json()
                self.n_calls += 1
                u = d.get("usage") or {}
                self.n_tokens_out += int(u.get("completion_tokens") or 0)
                return [c["message"]["content"] or "" for c in d["choices"]]
            except VLLMError:
                raise
            except Exception as e:                    # transport / 5xx / decode
                last = e
                if attempt + 1 < self.max_retries:
                    time.sleep(2.0 * (attempt + 1))
        raise VLLMError(f"chat failed after {self.max_retries} attempts: "
                        f"{type(last).__name__}: {last}")

    def complete_one(self, prompt: str, system: str = "", **kw) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        return self.chat(msgs, **kw)[0]

    # A `gen(prompt) -> str` callable, which is the interface strategies.generate_payload
    # and agents.elicit_flaws both expect. Passing the bound method keeps their code
    # unchanged, so the attack text produced for training is produced by THE SAME code
    # that produced the attacks in the benchmark — not a reimplementation that drifts.
    def as_gen(self, system: str = "", **kw):
        def gen(prompt: str) -> str:
            return self.complete_one(prompt, system=system, **kw)
        return gen

    def close(self) -> None:
        try:
            self._c.close()
        except Exception:
            pass


def client_from_env(model: str = "", **kw) -> VLLMClient:
    url = os.environ.get("VLLM_URL", "")
    if not url:
        raise VLLMError("VLLM_URL is not set (serve_vllm.sh writes the port it chose to "
                        "<log>.port and the PBS files read it back)")
    return VLLMClient(url, model or os.environ.get("VLLM_MODEL", ""), **kw)


def extract_json(text: str) -> Optional[dict]:
    """
    Pull the first JSON object out of a completion.

    Models wrap JSON in prose and in ```json fences no matter how firmly the prompt says
    not to, and a brace-counting scan is the difference between a 3% and a 30% parse
    failure rate. A parse failure is not neutral here: it is scored as a non-answer, and a
    role whose answers cannot be read looks like a role that cannot detect.
    """
    if not text:
        return None
    t = text.strip()
    # Reasoning models wrap their scratchpad in <think>...</think>. An unclosed block means
    # the answer was cut off mid-thought; a closed one leaves the real answer after it.
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[-1].strip()
    elif t.startswith("<think>"):
        return None
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if "```" in t:
            t = t[: t.rindex("```")]
    start = t.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start: i + 1])
                except Exception:
                    return None
    return None
