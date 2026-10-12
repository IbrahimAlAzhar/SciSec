#!/usr/bin/env python3
"""
ft_trajectories.py — sample k candidate outputs per (document, role) from the policy
model, score every one against gold, and write them all out.

WHY REJECTION SAMPLING AND NOT PLAIN DISTILLATION
=======================================================================================
The obvious thing is to run the zero-shot agents once per document and train on whatever
they said. That trains the model to reproduce its own mistakes — and on your V4 tables the
mistakes are the whole problem: FPR 0.400 means two clean papers in five were flagged, so
two SFT targets in five would be teaching the model to flag clean papers.

So: sample k completions at temperature, keep the ones that are CORRECT against gold, and
train on those. Correctness is decided by the gold label and by exact string containment of
the quoted span — arithmetic, not judgement. The judge only ranks quality among samples
that are already correct.

This also produces the DPO pairs for free and in the right shape. A preference pair must
be two completions OF THE SAME PROMPT; the chosen and rejected then differ only in the
behaviour being trained. (Pairing a clean document's answer against an attacked document's
answer would be two different prompts, and DPO on that optimises nothing about detection —
it just teaches the model that one document is preferable to another.)

WHAT COMES OUT
    trajectories/<model>/samples.jsonl   one line per (document, role, sample)
    trajectories/<model>/stats.json      accept rates per role and per family
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

import ft_config as C
import ft_roles as R
import ft_vllm as V


# ======================================================================================
# chunking, for the CHUNK_VERIFIER role
# ======================================================================================
def chunks_for(document: str, insert: str, *, chunk_chars: int = 800,
               max_chunks: int = 3, rng: Optional[random.Random] = None
               ) -> List[Tuple[str, bool]]:
    """
    A few passages from the limitations region, each tagged with whether it contains the
    inserted text.

    Both kinds are needed. Only-positive chunks would teach the verifier to answer
    "instruction" to anything it is handed — the verifier is the tier that escalates, and a
    verifier that always escalates costs the LLM calls the tiering exists to avoid.
    """
    rng = rng or random.Random(0)
    reg = R.region(document, 6000)
    if not reg:
        return []
    parts = [reg[i:i + chunk_chars] for i in range(0, len(reg), chunk_chars)]
    parts = [p for p in parts if len(p.strip()) > 120]
    if not parts:
        return []
    flat_ins = " ".join((insert or "").split())
    tagged = []
    for p in parts:
        has = bool(flat_ins) and (flat_ins[:80] in " ".join(p.split()))
        tagged.append((p, has))
    pos = [t for t in tagged if t[1]]
    neg = [t for t in tagged if not t[1]]
    out = pos[:1] + neg[: max(0, max_chunks - len(pos[:1]))]
    if not out:
        out = tagged[:max_chunks]
    return out[:max_chunks]


def sample_id(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


# ======================================================================================
# generation
# ======================================================================================
def worker_findings_from_samples(per_role_best: Dict[str, Optional[Dict]]
                                 ) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for w in ("INSTRUCTION", "PRESSURE", "VOICE"):
        obj = per_role_best.get(w) or {}
        f = obj.get("findings") if isinstance(obj, dict) else None
        out[w] = [x for x in (f or []) if isinstance(x, dict)]
    return out


def run(df: pd.DataFrame, cl: V.VLLMClient, model_key: str, out_dir: str, *,
        roles: Sequence[str] = C.ROLES,
        n_samples: int = 4, temperature: float = 0.9, top_p: float = 0.95,
        max_docs: int = 10 ** 9, seed: int = 0, region_chars: int = 6000,
        save_every: int = 20) -> Dict:
    spec = C.MODELS[model_key]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "samples.jsonl")
    f = open(path, "w", buffering=1)
    rng = random.Random(seed)

    stats = {"n_docs": 0, "n_samples": 0, "by_role": {}, "by_family": {}}

    def bump(d: Dict, key: str, field: str, n: int = 1) -> None:
        d.setdefault(key, {"n": 0, "usable": 0, "parse_fail": 0, "fp": 0, "miss": 0})
        d[key][field] += n

    t0 = time.time()
    docs = df.head(max_docs) if max_docs < len(df) else df
    for k, row in enumerate(docs.itertuples(index=False)):
        label = int(getattr(row, "label"))
        doc = str(getattr(row, "document") or "")
        ins = str(getattr(row, "insert") or "")
        cond = str(getattr(row, "condition") or "")
        pid = str(getattr(row, "paper_id"))
        if not doc.strip():
            continue
        stats["n_docs"] += 1
        best: Dict[str, Optional[Dict]] = {}

        # ---- CHUNK_VERIFIER: several passages per document ------------------------
        if "CHUNK_VERIFIER" in roles:
            for ch, has in chunks_for(doc, ins, rng=rng):
                sysmsg, user = R.build_prompt("CHUNK_VERIFIER", chunk=ch)
                msgs = R.messages("CHUNK_VERIFIER", sysmsg, user,
                                  system_role=spec.system_role)
                try:
                    outs = cl.chat(msgs, temperature=temperature, top_p=top_p, n=n_samples,
                                   max_tokens=R.ROLES["CHUNK_VERIFIER"].max_tokens,
                                   seed=seed)
                except V.VLLMError as e:
                    print(f"  [gen] CHUNK_VERIFIER {pid}: {e}")
                    outs = []
                for j, raw in enumerate(outs):
                    obj, sc = R.score_output("CHUNK_VERIFIER", raw, label=label,
                                             chunk=ch, insert=ins,
                                             chunk_has_insert=has)
                    bump(stats["by_role"], "CHUNK_VERIFIER", "n")
                    if not sc.parsed:
                        bump(stats["by_role"], "CHUNK_VERIFIER", "parse_fail")
                    if sc.usable:
                        bump(stats["by_role"], "CHUNK_VERIFIER", "usable")
                        best.setdefault("CHUNK_VERIFIER", obj)
                    elif sc.label_correct is False:
                        bump(stats["by_role"], "CHUNK_VERIFIER",
                             "fp" if not (label == 1 and has) else "miss")
                    f.write(json.dumps({
                        "sample_id": sample_id(model_key, pid, cond, "CHUNK_VERIFIER",
                                               ch[:64], str(j)),
                        "model": model_key, "role": "CHUNK_VERIFIER", "paper_id": pid,
                        "condition": cond, "label": label, "family": cond,
                        "chunk_has_insert": int(has), "k": j,
                        "system": sysmsg, "user": user, "completion": raw,
                        "insert": ins, "chunk": ch,
                        "score": sc.as_dict()}, ensure_ascii=False) + "\n")
                    stats["n_samples"] += 1

        # ---- the three evidence workers -------------------------------------------
        for w in ("INSTRUCTION", "PRESSURE", "VOICE"):
            if w not in roles:
                continue
            sysmsg, user = R.build_prompt(w, document=doc, region_chars=region_chars)
            msgs = R.messages(w, sysmsg, user, system_role=spec.system_role)
            try:
                outs = cl.chat(msgs, temperature=temperature, top_p=top_p, n=n_samples,
                               max_tokens=R.ROLES[w].max_tokens, seed=seed)
            except V.VLLMError as e:
                print(f"  [gen] {w} {pid}: {e}")
                outs = []
            for j, raw in enumerate(outs):
                obj, sc = R.score_output(w, raw, label=label, document=doc, insert=ins)
                bump(stats["by_role"], w, "n")
                if not sc.parsed:
                    bump(stats["by_role"], w, "parse_fail")
                if sc.usable:
                    bump(stats["by_role"], w, "usable")
                    best.setdefault(w, obj)
                elif label == 0 and sc.n_findings:
                    bump(stats["by_role"], w, "fp")
                f.write(json.dumps({
                    "sample_id": sample_id(model_key, pid, cond, w, str(j)),
                    "model": model_key, "role": w, "paper_id": pid, "condition": cond,
                    "label": label, "family": cond, "k": j,
                    "system": sysmsg, "user": user, "completion": raw,
                    "insert": ins, "document": doc,
                    "score": sc.as_dict()}, ensure_ascii=False) + "\n")
                stats["n_samples"] += 1

        # ---- LEADER: conditioned on the workers' ACCEPTED findings ----------------
        # Conditioning on accepted findings, not on raw ones, is deliberate. The leader's
        # job at inference is to survive imperfect evidence, but its TRAINING prompt has to
        # be reproducible, and "whatever the workers happened to say this time" is not.
        if "LEADER" in roles:
            fnd = worker_findings_from_samples(best)
            screen = getattr(row, "screen_score", None)
            try:
                screen = float(screen) if screen is not None and screen == screen else None
            except Exception:
                screen = None
            sysmsg, user = R.build_prompt("LEADER", document=doc, screen_score=screen,
                                          findings=fnd, region_chars=region_chars)
            msgs = R.messages("LEADER", sysmsg, user, system_role=spec.system_role)
            try:
                outs = cl.chat(msgs, temperature=temperature, top_p=top_p, n=n_samples,
                               max_tokens=R.ROLES["LEADER"].max_tokens, seed=seed)
            except V.VLLMError as e:
                print(f"  [gen] LEADER {pid}: {e}")
                outs = []
            for j, raw in enumerate(outs):
                obj, sc = R.score_output("LEADER", raw, label=label, document=doc,
                                         insert=ins)
                bump(stats["by_role"], "LEADER", "n")
                bump(stats["by_family"], cond, "n")
                if not sc.parsed:
                    bump(stats["by_role"], "LEADER", "parse_fail")
                if sc.usable:
                    bump(stats["by_role"], "LEADER", "usable")
                    bump(stats["by_family"], cond, "usable")
                elif sc.label_correct is False:
                    bump(stats["by_role"], "LEADER", "fp" if label == 0 else "miss")
                    bump(stats["by_family"], cond, "fp" if label == 0 else "miss")
                f.write(json.dumps({
                    "sample_id": sample_id(model_key, pid, cond, "LEADER", str(j)),
                    "model": model_key, "role": "LEADER", "paper_id": pid,
                    "condition": cond, "label": label, "family": cond, "k": j,
                    "system": sysmsg, "user": user, "completion": raw,
                    "insert": ins, "document": doc, "worker_findings": fnd,
                    "score": sc.as_dict()}, ensure_ascii=False) + "\n")
                stats["n_samples"] += 1

        if (k + 1) % save_every == 0:
            el = (time.time() - t0) / 60
            print(f"[gen] {k+1}/{len(docs)} documents, {stats['n_samples']} samples, "
                  f"{cl.n_calls} calls, {el:.1f}m", flush=True)

    f.close()
    stats["n_llm_calls"] = cl.n_calls
    stats["minutes"] = (time.time() - t0) / 60
    with open(os.path.join(out_dir, "stats.json"), "w") as sf:
        json.dump(stats, sf, indent=2)

    print("\n" + "=" * 92)
    print(f"TRAJECTORIES — {model_key}")
    print("=" * 92)
    print(f"  {stats['n_docs']} documents -> {stats['n_samples']} samples "
          f"in {stats['minutes']:.1f} min ({cl.n_calls} LLM calls)")
    print(f"\n  {'role':<16} {'samples':>8} {'usable':>8} {'accept':>8} "
          f"{'parse_fail':>11} {'FP':>6} {'miss':>6}")
    for role, d in stats["by_role"].items():
        acc = d["usable"] / d["n"] if d["n"] else 0.0
        print(f"  {role:<16} {d['n']:>8} {d['usable']:>8} {acc:>7.1%} "
              f"{d['parse_fail']:>11} {d['fp']:>6} {d['miss']:>6}")
    print("\n  HOW TO READ THE ACCEPT RATE")
    print("    Too HIGH (>90%) and there is nothing to learn: the zero-shot model already")
    print("    agrees with gold, so SFT will move little and DPO will have no natural")
    print("    rejected samples. Too LOW (<10%) and the targets will mostly be the")
    print("    constructed gold ones, which teaches format more than judgement.")
    print("    20-70% is the band where this pipeline earns its keep.")
    print("    A high parse_fail is a FORMAT problem, not a reasoning one, and SFT fixes")
    print("    it almost immediately — expect it to dominate the first epoch's gain.")
    print(f"\n[saved] {path}")
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description="sample and score agent trajectories")
    p.add_argument("--model", required=True, choices=sorted(C.MODELS))
    p.add_argument("--documents", default=C.wpath(C.D_TRAINDATA, "train_documents.csv"))
    p.add_argument("--out-dir", dest="out_dir", default="")
    p.add_argument("--vllm-url", dest="vllm_url", default=os.environ.get("VLLM_URL", ""))
    p.add_argument("--vllm-model", dest="vllm_model",
                   default=os.environ.get("VLLM_MODEL", ""))
    p.add_argument("--roles", nargs="+", default=list(C.ROLES), choices=list(C.ROLES))
    p.add_argument("--n-samples", dest="n_samples", type=int, default=C.GEN.n_samples)
    p.add_argument("--temperature", type=float, default=C.GEN.temperature)
    p.add_argument("--top-p", dest="top_p", type=float, default=C.GEN.top_p)
    p.add_argument("--max-docs", dest="max_docs", type=int, default=10 ** 9)
    p.add_argument("--region-chars", dest="region_chars", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if not os.path.exists(a.documents):
        raise SystemExit(f"[FATAL] {a.documents} not found — run ft_attack.py first")
    if not a.vllm_url:
        raise SystemExit("[FATAL] --vllm-url is required")

    df = pd.read_csv(a.documents, low_memory=False)
    print(f"[data] {a.documents}: {len(df)} documents "
          f"({int((df['label']==1).sum())} attacked, {int((df['label']==0).sum())} clean)")

    out_dir = a.out_dir or C.wpath(C.D_TRAJ, a.model.replace("/", "_"))
    spec = C.MODELS[a.model]
    cl = V.VLLMClient(a.vllm_url, a.vllm_model or spec.path,
                      temperature=a.temperature, top_p=a.top_p,
                      chat_kwargs=spec.chat_kwargs)
    if spec.chat_kwargs:
        print(f"[serve] chat_template_kwargs={spec.chat_kwargs}")
    print(f"[serve] {a.vllm_url}   served: {cl.models()}")
    run(df, cl, a.model, out_dir, roles=a.roles, n_samples=a.n_samples,
        temperature=a.temperature, top_p=a.top_p, max_docs=a.max_docs, seed=a.seed,
        region_chars=a.region_chars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
