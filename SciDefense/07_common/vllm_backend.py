"""
vllm_backend.py — talk to a local vLLM OpenAI-compatible server.

WHY THIS FILE EXISTS
=======================================================================================
The defence needs the model for TWO things, and both must go through the same place:

    1. the audit agents          planner + 3 workers + leader, up to 3 rounds
    2. the stage-2 re-reviewer   the sanitised paper goes back to the SAME reviewer

If only the agents used vLLM and the re-reviewer stayed on the in-process HuggingFace
path, one 40GB card would hold a vLLM engine at gpu_memory_utilization=0.90 AND a second
4-bit copy of the same weights. That OOMs, and it OOMs late — after the server has spent
seven minutes coming up. So `VLLMReviewer` implements the same duck type the rest of the
code already expects from `bbmodel.LocalReviewer`:

    .name          str
    .is_closed     False
    .generate(prompt, max_new_tokens=...) -> str        (used by the agents)
    .review(paper_body) -> Review                       (used by stage 2)

and it reuses `harness.SYSTEM_PROMPT`, `harness.REVIEW_INSTRUCTION` and
`harness.parse_review` — the SAME prompt and the SAME parser the attack benchmark used.
That is not tidiness. `restoration` divides by (ra - r0), where ra and r0 came out of the
benchmark run; if the re-review used a different prompt or a different parser, the
numerator and denominator would be on different scales and the ratio would be noise.

WHY vLLM IS WORTH THE SETUP HERE
=======================================================================================
The agent tier is the expensive part: an escalated document costs 5 LLM calls per round,
up to 3 rounds, and every call re-reads a 6000-character region. Under HuggingFace
`generate()` those are sequential, uncached, one sequence at a time. vLLM gives paged KV
cache and continuous batching, so the repeated region prefix is not recomputed for every
worker. On the escalation-heavy runs this is the difference between a job that finishes
and one that walltimes out.

THE FIVE THINGS THAT COST YOU TIME LAST TIME, ALL HANDLED
=======================================================================================
  * NEVER set VLLM_API_KEY, not even to "EMPTY". vLLM turns on bearer auth for every
    /v1/* route the moment that variable EXISTS, and every request then 401s while the
    server looks perfectly healthy. This module refuses to send a key it was not given
    and says so loudly if it gets a 401.
  * transformers must be >=4.53,<5. vLLM 0.11 calls
    tokenizer.all_special_tokens_extended, removed in transformers 5.
  * no_proxy / NO_PROXY must include 127.0.0.1, AND the http client must be built with
    trust_env=False — otherwise a site-wide http_proxy swallows loopback traffic and the
    request hangs rather than failing.
  * pick the port at runtime; two jobs co-scheduled on one node must not collide.
  * Qwen3 emits <think>...</think> unless chat_template_kwargs disables it. Thinking text
    in front of a JSON object is tokens spent making the parse harder, and the agents are
    scored on their JSON.
"""

from __future__ import annotations

import os
import re
import socket
import sys
from typing import Optional

from harness import REVIEW_INSTRUCTION, SYSTEM_PROMPT, parse_review

RETRY_NOTE = ("\n\nIMPORTANT: your previous answer did not follow the format. Reply with "
              "the fields above, one per line, and include a line that reads exactly "
              "'RATING: <integer 1-10>' and a line that reads exactly 'DECISION: Accept' "
              "or 'DECISION: Reject'.")

_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def pick_free_port() -> int:
    """
    A port the OS says is free right now.

    Hard-coding 8000 is how two jobs co-scheduled on one node collide: the second vLLM
    dies with 'address already in use' and the job then waits out its full readiness
    timeout for a server that belongs to somebody else.
    """
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _client(base_url: str, api_key: Optional[str]):
    """
    An OpenAI client that will not be hijacked by a site proxy.

    Both urllib and httpx read http_proxy from the environment by default, and on this
    cluster that sends 127.0.0.1 traffic to the site proxy, where it hangs instead of
    failing. trust_env=False is the part that actually fixes it; exporting no_proxy alone
    is not enough because the client may be constructed before the export lands.
    """
    from openai import OpenAI
    try:
        import httpx
        http_client = httpx.Client(trust_env=False, timeout=httpx.Timeout(600.0,
                                                                          connect=15.0))
    except Exception:  # pragma: no cover - httpx ships with openai, but do not hard-fail
        http_client = None
        print("[vllm] httpx unavailable — cannot disable proxy env; if requests hang, "
              "unset http_proxy/https_proxy in the PBS file")

    # An API key is sent ONLY if the caller actually has one. The OpenAI SDK requires the
    # argument to be non-empty, so a placeholder goes in the constructor -- but if the
    # server was started without VLLM_API_KEY it ignores the header entirely, and if it
    # was started WITH one, the value below has to match it.
    kw = {"base_url": base_url, "api_key": api_key or "EMPTY"}
    if http_client is not None:
        kw["http_client"] = http_client
    return OpenAI(**kw)


class VLLMReviewer:
    """
    A reviewer served by vLLM, duck-type-compatible with bbmodel.LocalReviewer.

    `model` MUST equal the server's --served-model-name exactly. serve_vllm.sh passes the
    model DIRECTORY as the served name, so that is what belongs here.
    """

    is_closed = False

    def __init__(self, name: str, base_url: str, served_model: str,
                 api_key: Optional[str] = None, max_new_tokens: int = 600,
                 temperature: float = 0.0, max_chars: int = 60000,
                 enable_thinking: bool = False):
        self.name = name
        self.served_model = served_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_chars = max_chars
        self.client = _client(base_url, api_key)
        # Qwen3 emits <think>...</think> unless the chat template is told not to. Other
        # families ignore the kwarg, so sending it unconditionally for qwen is safe and
        # sending it for everyone is not worth the risk of a template that errors on
        # unknown kwargs.
        self.extra_body = ({"chat_template_kwargs": {"enable_thinking": False}}
                           if (not enable_thinking and "qwen" in served_model.lower())
                           else None)
        self._warned_401 = False

    # ---------------------------------------------------------------- raw completion
    def _chat(self, prompt: str, max_new_tokens: Optional[int] = None,
              system: Optional[str] = None) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt[: self.max_chars]})
        kw = {"model": self.served_model, "messages": msgs,
              "temperature": self.temperature,
              "max_tokens": max_new_tokens or self.max_new_tokens}
        if self.extra_body:
            kw["extra_body"] = self.extra_body
        r = self.client.chat.completions.create(**kw)
        out = (r.choices[0].message.content or "").strip()
        # strip any residual thinking block defensively: the kwarg is honoured by the
        # template, not enforced by the server, and a template that ignores it would
        # otherwise put prose in front of every JSON reply the agents emit
        return _THINK.sub("", out).strip()

    def generate(self, prompt: str, max_new_tokens: int = 600) -> str:
        """What make_llm() hands to the agents."""
        for attempt in range(4):
            try:
                return self._chat(prompt, max_new_tokens)
            except Exception as e:
                msg = str(e)
                if ("401" in msg or "Unauthorized" in msg) and not self._warned_401:
                    self._warned_401 = True
                    print("[vllm] 401 Unauthorized. The server was started with "
                          "VLLM_API_KEY set but this client does not have the same "
                          "value. Waiting will not fix it — unset VLLM_API_KEY for BOTH "
                          "the server and this job, or export the identical value to "
                          "both.", file=sys.stderr)
                    return ""
                if attempt == 3:
                    print(f"[vllm] generate failed: {type(e).__name__}: {e}",
                          file=sys.stderr)
                    return ""
                import time
                time.sleep(2 ** attempt)
        return ""

    # ---------------------------------------------------------------- stage-2 re-review
    def review(self, paper_body: str):
        """
        The SAME prompt and parser the attack benchmark used.

        Identical text in, identical parsing out — otherwise `restoration`, which divides
        a difference measured here by a difference measured in the benchmark run, would
        be comparing two different scales.
        """
        prompt = REVIEW_INSTRUCTION and (paper_body + "\n\n" + REVIEW_INSTRUCTION)
        raw = self._chat(prompt, self.max_new_tokens, system=SYSTEM_PROMPT)
        rv = parse_review(raw)
        if rv.rating is None:
            raw2 = self._chat(prompt + RETRY_NOTE, self.max_new_tokens,
                              system=SYSTEM_PROMPT)
            rv2 = parse_review(raw2)
            if rv2.rating is not None:
                return rv2
            rv.raw = raw
        return rv


def make_vllm_reviewer(name: str, url: str, served_model: str, **kw) -> VLLMReviewer:
    """
    Build the reviewer and prove the endpoint answers BEFORE the run starts.

    A dead or mis-keyed endpoint discovered on document 1 of 200 is a wasted job; one
    discovered here costs a second.
    """
    key = os.environ.get("VLLM_API_KEY") or None
    if key:
        print("[vllm] VLLM_API_KEY is set. The server must have been started with the "
              "SAME value or every request will 401. Unsetting it everywhere is simpler.")
    rv = VLLMReviewer(name, url.rstrip("/"), served_model, api_key=key, **kw)
    probe = rv.generate("Reply with the single word: ready", max_new_tokens=8)
    if not probe:
        raise SystemExit(
            f"[vllm] {url} did not answer a one-token probe. The server is not serving, "
            f"or the served model name is wrong. It must match --served-model-name "
            f"exactly; this job is asking for {served_model!r}.")
    print(f"[vllm] endpoint {url} model={served_model} — probe returned {probe[:40]!r}")
    return rv
