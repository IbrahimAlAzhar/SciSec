#!/usr/bin/env python3
"""
vllm_backend.py -- a small, thread-safe client for a local vLLM OpenAI-compatible server.

    from vllm_backend import VLLMClient, run_parallel
    cli = VLLMClient("http://127.0.0.1:8000/v1", "gemma3-27b-it")
    out = cli.chat([{"role": "user", "content": "hi"}], max_tokens=16)

Same policies as the vllm_backend.py of your attack runs (lessons that cost time there):
    * VLLM_API_KEY: never needed. If it IS set, it is sent as the bearer token (the
      server must have the same value); a 401/403 raises AuthError at once instead of
      being retried for minutes. probe() catches it before a stage starts.
    * proxies: trust_env=False and a 15 s connect timeout. HTTP(S)_PROXY is set on the
      cluster and would swallow a request to 127.0.0.1, which then HANGS rather than fails.
    * qwen3: enable_thinking=False is sent automatically for qwen models (only for them:
      other chat templates can choke on the unknown kwarg), and any <think>...</think>
      block that still appears is stripped.
    * the seed is sent only when sampling (temperature > 0); D7 decodes greedily.

WHAT IT HANDLES
    * structured output: tries `response_format={"type":"json_schema",...}` (vLLM >= 0.6),
      then the older `guided_json` extra-body field, then plain text. It degrades once, on
      the first 400 that names the feature, and remembers the choice.
    * context overflow: a 400 about the context length raises ContextLengthError so the
      caller can shorten the document and retry instead of losing the row.
    * retries with backoff for timeouts / 5xx; counters for calls and tokens.
    * run_parallel(): a thread pool, so vLLM can batch many requests at once.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional

try:
    import httpx
except Exception:                                                # pragma: no cover
    httpx = None


class ContextLengthError(RuntimeError):
    pass


class TransientError(RuntimeError):
    """The server did not answer (timeouts, connection errors, 5xx) after all retries.
    Callers should NOT cache this as a result: re-running should retry it."""
    pass


class AuthError(TransientError):
    """HTTP 401/403: the server has VLLM_API_KEY set and this client does not have the same
    value. Waiting will not fix it. A TransientError, so it is never cached as a result."""
    pass


_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


_CTX_PAT = re.compile(r"maximum context length|maximum model length|longer than the "
                      r"maximum|reduce the length|context length|too many tokens|"
                      r"prompt is too long|exceeds", re.I)
_GUIDED_PAT = re.compile(r"response_format|json_schema|guided|structured|grammar|"
                         r"xgrammar|outlines", re.I)


class VLLMClient:
    def __init__(self, url: str, model: str = "", timeout: float = 1200.0,
                 retries: int = 4, api_key: Optional[str] = None,
                 chat_template_kwargs: Optional[Dict] = None, name: str = "llm",
                 max_connections: int = 64):
        if not url:
            raise ValueError("no vLLM url given")
        self.url = url.rstrip("/")
        self.retries = int(retries)
        self.name = name
        explicit_ctk = chat_template_kwargs is not None
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        if api_key is None:
            api_key = os.environ.get("VLLM_API_KEY") or None
            if api_key:
                print(f"[{name}] VLLM_API_KEY is set: sending it as the bearer token. The "
                      f"server must have been started with the SAME value.", flush=True)
        self.guided_mode = "response_format"          # -> guided_json -> none
        self._lock = threading.Lock()
        self.n_calls = self.n_errors = self.n_ctx_errors = 0
        self.prompt_tokens = self.completion_tokens = 0
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.headers = headers
        self.timeout = float(timeout)
        if httpx is not None:
            self._c = httpx.Client(timeout=httpx.Timeout(self.timeout, connect=15.0),
                                   trust_env=False, headers=headers,
                                   limits=httpx.Limits(max_connections=max_connections,
                                                       max_keepalive_connections=max_connections))
        else:
            self._c = None
        self.model = model or ""
        self.served = self.list_models(quiet=True)
        if self.served and (not self.model or self.model not in self.served):
            if self.model:
                print(f"[{name}] model {self.model!r} not served; using {self.served[0]!r} "
                      f"(served: {self.served})", flush=True)
            self.model = self.served[0]
        if not explicit_ctk and "qwen" in self.model.lower():
            self.chat_template_kwargs = {"enable_thinking": False}

    # ------------------------------------------------------------------ transport
    def _request(self, method: str, path: str, body: Optional[Dict] = None):
        full = f"{self.url}{path}"
        if self._c is not None:
            r = self._c.request(method, full, json=body)
            return r.status_code, r.text
        import urllib.request
        import urllib.error
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(full, data=data, method=method, headers=self.headers)
        try:
            with opener.open(req, timeout=self.timeout) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")

    def list_models(self, quiet: bool = False) -> List[str]:
        try:
            code, txt = self._request("GET", "/models")
            if code == 200:
                return [m["id"] for m in json.loads(txt).get("data", [])]
        except Exception as e:
            if not quiet:
                print(f"[{self.name}] /models failed: {type(e).__name__}: {e}", flush=True)
        return []

    def probe(self) -> str:
        """One tiny request before a stage starts. A dead endpoint, a wrong model name or
        an API-key mismatch found on document 1 of 4,000 wastes a queue slot; found here
        it costs a second. Raises on failure; returns the reply."""
        out = self.chat([{"role": "user", "content": "Reply with the single word: ready"}],
                        max_tokens=8)
        return out["text"]

    def wait_ready(self, timeout_s: float = 1800.0, every: float = 10.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.list_models(quiet=True):
                return True
            time.sleep(every)
        return False

    # ------------------------------------------------------------------ chat
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512,
             temperature: float = 0.0, json_schema: Optional[Dict] = None,
             schema_name: str = "output", logprobs: bool = False,
             top_logprobs: int = 0, seed: int = 0) -> Dict[str, Any]:
        """-> {text, logprobs, finish_reason, prompt_tokens, completion_tokens, guided}"""
        last: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            body: Dict[str, Any] = {"model": self.model, "messages": messages,
                                    "temperature": float(temperature),
                                    "max_tokens": int(max_tokens)}
            if temperature > 0:
                body["seed"] = int(seed)          # only meaningful when sampling
            if self.chat_template_kwargs:
                body["chat_template_kwargs"] = self.chat_template_kwargs
            if logprobs:
                body["logprobs"] = True
                if top_logprobs:
                    body["top_logprobs"] = int(top_logprobs)
            mode = self.guided_mode if json_schema else "none"
            if mode == "response_format":
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": schema_name,
                                                           "schema": json_schema}}
            elif mode == "guided_json":
                body["guided_json"] = json_schema
            try:
                code, txt = self._request("POST", "/chat/completions", body)
            except Exception as e:                       # timeout / connection reset
                last = e
                time.sleep(min(60.0, 3.0 * (attempt + 1)))
                continue
            if code == 200:
                j = json.loads(txt)
                ch = j["choices"][0]
                usage = j.get("usage") or {}
                lp = (ch.get("logprobs") or {}).get("content") if ch.get("logprobs") else None
                with self._lock:
                    self.n_calls += 1
                    self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                    self.completion_tokens += int(usage.get("completion_tokens") or 0)
                text = (ch.get("message") or {}).get("content") or ""
                if "</think>" in text:
                    text = _THINK.sub("", text).strip()
                return dict(text=text, logprobs=lp, finish_reason=ch.get("finish_reason"),
                            prompt_tokens=int(usage.get("prompt_tokens") or 0),
                            completion_tokens=int(usage.get("completion_tokens") or 0),
                            guided=mode)
            if code in (401, 403):
                raise AuthError(
                    f"[{self.name}] HTTP {code}: the server has VLLM_API_KEY set and this "
                    f"client does not have the same value. Waiting will not fix it: unset "
                    f"VLLM_API_KEY for BOTH the server and the job, or export the same value "
                    f"to both.")
            if code == 404:
                served = self.list_models(quiet=True)
                raise RuntimeError(f"404 for model {self.model!r}; served: {served}")
            if code == 400:
                if _CTX_PAT.search(txt) and not _GUIDED_PAT.search(txt[:200]):
                    with self._lock:
                        self.n_ctx_errors += 1
                    raise ContextLengthError(txt[:400])
                if json_schema and mode != "none" and _GUIDED_PAT.search(txt):
                    with self._lock:
                        nxt = {"response_format": "guided_json",
                               "guided_json": "none"}[mode]
                        if self.guided_mode == mode:
                            print(f"[{self.name}] structured output via {mode} rejected "
                                  f"({txt[:160]!r}); falling back to {nxt}", flush=True)
                            self.guided_mode = nxt
                    continue
                if _CTX_PAT.search(txt):
                    with self._lock:
                        self.n_ctx_errors += 1
                    raise ContextLengthError(txt[:400])
                raise RuntimeError(f"400 from server: {txt[:400]}")
            last = RuntimeError(f"HTTP {code}: {txt[:300]}")
            time.sleep(min(60.0, 3.0 * (attempt + 1)))
        with self._lock:
            self.n_errors += 1
        raise TransientError(f"[{self.name}] chat failed after {self.retries + 1} tries: "
                             f"{type(last).__name__}: {last}")

    def stats(self) -> str:
        return (f"calls={self.n_calls} errors={self.n_errors} ctx_errors={self.n_ctx_errors}"
                f" prompt_tok={self.prompt_tokens} completion_tok={self.completion_tokens}"
                f" guided={self.guided_mode}")


# =====================================================================================
def run_parallel(fn: Callable[[Any], Any], items: Iterable[Any], workers: int = 16,
                 log: Callable[[str], None] = print, every: int = 50,
                 desc: str = "") -> List[Any]:
    """Apply fn to every item on a thread pool; returns results in completion order.
    fn must catch its own exceptions and return something storable."""
    items = list(items)
    out, t0 = [], time.time()
    if not items:
        return out
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [ex.submit(fn, it) for it in items]
        for k, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            if k % every == 0 or k == len(items):
                el = time.time() - t0
                rate = k / el if el > 0 else 0.0
                eta = (len(items) - k) / rate if rate > 0 else float("nan")
                log(f"  [{desc}] {k}/{len(items)}  {rate:.2f}/s  eta {eta/60:.1f} min")
    return out


# =====================================================================================
def extract_json(t: Any) -> Optional[Dict]:
    """First balanced JSON object in a model reply. Tolerates <think> blocks and code
    fences; returns None when nothing parses."""
    if t is None:
        return None
    t = str(t).strip()
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[-1].strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if "```" in t:
            t = t[: t.rindex("```")]
    try:
        v = json.loads(t)
        if isinstance(v, dict):
            return v
    except Exception:
        pass
    i = t.find("{")
    while i >= 0:
        depth, ins, esc = 0, False, False
        for j in range(i, len(t)):
            ch = t[j]
            if ins:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    ins = False
                continue
            if ch == '"':
                ins = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(t[i:j + 1])
                        if isinstance(v, dict):
                            return v
                    except Exception:
                        break
                    break
        i = t.find("{", i + 1)
    return None


if __name__ == "__main__":                                       # quick smoke test
    url = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1")
    c = VLLMClient(url, os.environ.get("VLLM_MODEL", ""))
    print("served:", c.served)
    r = c.chat([{"role": "user", "content": "Reply with the JSON {\"ok\": true}."}],
               max_tokens=20, json_schema={"type": "object",
                                           "properties": {"ok": {"type": "boolean"}},
                                           "required": ["ok"]})
    print(r["text"], "|", c.stats())
    sys.exit(0)
