#!/usr/bin/env python3
"""
sr_vllm.py — one OpenAI-compatible chat client for a local vLLM server (or the OpenAI API).

Why not V4's ft_vllm.VLLMClient: the GRPO attacker needs G=8 completions per prompt in
one request (`n`), and that client's sampling signature is not something this package can
rely on. The things that matter are kept identical to how your jobs talk to vLLM:

  * local servers are reached WITHOUT the proxy (your jobs export no_proxy=127.0.0.1 for
    the same reason; a proxied localhost call hangs);
  * `chat_template_kwargs` carries the model's chat kwargs (qwen3: enable_thinking=False,
    or every completion opens with <think> and nothing parses);
  * the served name IS the adapter selector, and `served_models()` lets every stage refuse
    to run when the adapter it was told to use is not actually served.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Sequence


class ChatError(RuntimeError):
    pass


# A model can emit HALF of a UTF-16 surrogate pair (one token of a split emoji or symbol).
# The server sends it as a "\\udXXX" escape, json.loads turns it into a lone surrogate, and
# that string can then never be written as UTF-8 -- the first file write or cache line
# that touches it crashes the job. Valid pairs are already merged into one character by
# json.loads, so only the broken halves match here; they become U+FFFD.
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def clean_text(s: str) -> str:
    return _LONE_SURROGATE.sub("\ufffd", s) if s else s


def _is_local(url: str) -> bool:
    u = url.lower()
    return any(h in u for h in ("://127.0.0.1", "://localhost", "://[::1]", "://0.0.0.0"))


class ChatClient:
    def __init__(self, base_url: str, model: str, *, api_key: str = "",
                 chat_kwargs: Optional[Dict] = None, timeout: float = 900.0,
                 retries: int = 4):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.chat_kwargs = dict(chat_kwargs or {})
        self.timeout = timeout
        self.retries = retries
        handlers = [urllib.request.ProxyHandler({})] if _is_local(self.base_url) else []
        self._opener = urllib.request.build_opener(*handlers)
        self.n_calls = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------
    def _post(self, path: str, body: Dict) -> Dict:
        data = json.dumps(body).encode("utf-8")
        hdr = {"Content-Type": "application/json"}
        if self.api_key:
            hdr["Authorization"] = f"Bearer {self.api_key}"
        last = None
        for k in range(self.retries + 1):
            req = urllib.request.Request(self.base_url + path, data=data, headers=hdr,
                                         method="POST")
            try:
                with self._opener.open(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")[:600]
                last = f"HTTP {e.code}: {msg}"
                # 400 = the request itself is wrong (context length, unknown model):
                # retrying cannot fix it.
                if e.code in (400, 401, 403, 404, 422):
                    break
            except Exception as e:                           # timeouts, resets
                last = f"{type(e).__name__}: {e}"
            time.sleep(min(60.0, 2.0 * (2 ** k)) + random.random())
        raise ChatError(last or "unknown error")

    def chat(self, messages: List[Dict], *, n: int = 1, temperature: float = 0.0,
             top_p: float = 1.0, max_tokens: int = 512, seed: Optional[int] = None,
             model: Optional[str] = None) -> List[str]:
        body: Dict = {"model": model or self.model, "messages": messages, "n": int(n),
                      "temperature": float(temperature), "top_p": float(top_p),
                      "max_tokens": int(max_tokens)}
        if seed is not None:
            body["seed"] = int(seed)
        if self.chat_kwargs:
            body["chat_template_kwargs"] = self.chat_kwargs
        out = self._post("/chat/completions", body)
        with self._lock:
            self.n_calls += 1
        ch = sorted(out.get("choices") or [], key=lambda c: c.get("index", 0))
        return [clean_text(str(((c.get("message") or {}).get("content")) or "")) for c in ch]

    def served_models(self) -> List[str]:
        req = urllib.request.Request(self.base_url + "/models", method="GET")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with self._opener.open(req, timeout=60) as r:
            return [m["id"] for m in json.load(r).get("data", [])]


def require_served(client: ChatClient, name: str) -> None:
    """Refuse to score a row with a model that is not served: vLLM falls back to nothing,
    it 404s, but a wrong name in a cached result would silently be the base model."""
    try:
        served = client.served_models()
    except Exception as e:
        raise SystemExit(f"[FATAL] cannot list served models at {client.base_url}: {e}")
    print(f"[serve] served: {served}")
    if name not in served:
        raise SystemExit(f"[FATAL] {name!r} is NOT served at {client.base_url}. The row "
                         f"would report a different model under this label.")


def pmap(fn: Callable, items: Sequence, *, workers: int = 16, label: str = "",
         every: int = 50) -> List:
    """Ordered parallel map with progress lines. Exceptions propagate."""
    out: List = [None] * len(items)
    if not items:
        return out
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
            done += 1
            if label and (done % every == 0 or done == len(items)):
                print(f"[{label}] {done}/{len(items)}  ({(time.time()-t0)/60:.1f}m)",
                      flush=True)
    return out


def client_from_env(model: str, *, chat_kwargs: Optional[Dict] = None,
                    url: str = "", api_key: str = "") -> ChatClient:
    url = url or os.environ.get("VLLM_URL", "")
    if not url:
        raise SystemExit("[FATAL] no server URL: pass --vllm-url or export VLLM_URL")
    return ChatClient(url, model, api_key=api_key, chat_kwargs=chat_kwargs)