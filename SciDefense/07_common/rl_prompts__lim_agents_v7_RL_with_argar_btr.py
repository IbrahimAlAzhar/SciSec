#!/usr/bin/env python3
"""
rl_prompts.py — the agents' prompts, imported from llim_agents_v4_SFT_DPO, never rewritten.

THE INVARIANT
=======================================================================================
The prompt used to SFT an adapter, the prompt used to ROLL OUT the RL policy, and the
prompt used at INFERENCE must be byte-identical. If they drift, the RL stage becomes a
domain-transfer experiment: the policy is optimised on a distribution that does not occur
at test time, it underperforms the adapter it started from, and every log looks healthy
the whole way.

So this module does NOT contain a prompt. It imports ft_roles from the V4 SFT/DPO
directory and re-exports exactly the four things every stage needs. If ft_roles cannot be
imported, this raises rather than falling back to a local copy -- a local copy is how the
drift starts.

    build_prompt(role, ...) -> (system, user)
    messages(role, system, user, system_role=)  -> chat message list
    region(document, chars)                     -> the limitations region
    ROLES / SEVERITY / FAMILY_HINTS             -> the taxonomy the leader may name

WHAT THIS MODULE ADDS ON TOP
    `evidence_subset` — the LEADER prompt built from an arbitrary SUBSET of the workers'
    findings. Nothing in V4 needed that; the Shapley credit rule in rl_credit does, since
    a coalition value is "what would the leader decide given only these agents' evidence".
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import rl_config as C


def _bootstrap() -> None:
    cands = [C.V4_FT_DIR, C.V4_DIR, C.V5_DIR,
             os.path.dirname(os.path.abspath(__file__))]
    cands += [os.environ.get(k) for k in
              ("WB_BB_DIR", "WB_UCA_DIR", "WB_T35_DIR", "WB_COMP_DIR",
               "WB_FEAS_DIR", "WB_SRC_DIR", "WB_FLASH_DIR")]
    if os.path.isdir(C.ROOT):
        try:
            cands += [os.path.join(C.ROOT, d) for d in sorted(os.listdir(C.ROOT))
                      if d.startswith("lim_") or d.startswith("attack_")]
        except OSError:
            pass
    for c in cands:
        if c and os.path.isdir(c) and c not in sys.path:
            sys.path.append(c)


_bootstrap()

try:
    import ft_roles as _R
except Exception as _e:                                        # pragma: no cover
    raise SystemExit(
        f"[FATAL] cannot import ft_roles from {C.V4_FT_DIR}: "
        f"{type(_e).__name__}: {_e}\n"
        f"        This package deliberately has NO prompt of its own. The prompts must be\n"
        f"        the ones the SFT and DPO adapters were trained on, or the RL stage is\n"
        f"        optimising against a distribution that does not occur at inference.\n"
        f"        Set V4_FT_DIR to the llim_agents_v4_SFT_DPO directory and retry.")

# re-exports
ROLES = _R.ROLES
SEVERITY = _R.SEVERITY
FAMILY_HINTS = _R.FAMILY_HINTS
region = _R.region
messages = _R.messages
build_prompt = _R.build_prompt
schema_block = _R.schema_block
gold_target = _R.gold_target
render_target = _R.render_target
score_output = _R.score_output


def max_tokens_for(role: str, default: int = 600) -> int:
    r = ROLES.get(role)
    return int(getattr(r, "max_tokens", default)) if r is not None else default


def evidence_subset(document: str, findings: Dict[str, List[Dict]],
                    keep: Sequence[str], *, screen_score: Optional[float] = None,
                    region_chars: int = 6000) -> Tuple[str, str]:
    """
    The LEADER prompt built from only the workers in `keep`.

    Workers that are excluded are shown as having found nothing, NOT omitted from the
    prompt. That distinction is the whole point of a coalition value: a leader told
    "PRESSURE found nothing" is in a different epistemic state from a leader never told
    about PRESSURE at all, and a Shapley value computed over the second is measuring the
    prompt's shape rather than the agent's contribution.
    """
    sub = {w: (findings.get(w) or []) if w in keep else [] for w in
           ("INSTRUCTION", "PRESSURE", "VOICE")}
    return build_prompt("LEADER", document=document, screen_score=screen_score,
                        findings=sub, region_chars=region_chars)


def describe() -> str:
    return (f"[prompts] imported from {os.path.dirname(_R.__file__)}\n"
            f"[prompts] roles: {', '.join(ROLES)}\n"
            f"[prompts] this is the SAME module the SFT/DPO datasets were built with, "
            f"so the RL rollouts are on-distribution for those adapters.")


if __name__ == "__main__":
    print(describe())
    s, u = build_prompt("LEADER", document="A short manuscript. Limitations: small n.",
                        screen_score=0.12, findings={"INSTRUCTION": [], "PRESSURE": [],
                                                     "VOICE": []})
    print("\n--- LEADER system (first 400 chars) ---")
    print(s[:400])
    print("\n--- LEADER user ---")
    print(u[:900])
