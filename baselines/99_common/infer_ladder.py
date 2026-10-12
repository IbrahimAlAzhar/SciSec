#!/usr/bin/env python3
"""
infer_ladder.py — run the agent cascade on dataset_primary_500.csv, with or without
adapters, and write one stage1_detail.csv per ablation.

THE EVALUATION SET IS THE V4 ONE, NOT A NEW ONE
=======================================================================================
The documents are built by def_dataset.build_eval_set — the SAME function, the same benign
CSV, the same metrics_detail.csv, the same seed as your zero-shot V4 run. So the 500-paper
corpus here is the corpus there, and the SFT/DPO rows are comparable to the V4 rows rather
than to a differently-sampled set. If the corpus were rebuilt some other way, every
comparison in the final table would confound the fine-tune with the sample.

ONE THING TO BE HONEST ABOUT
    The CASCADE here is reimplemented rather than imported: it has to route each role to a
    different LoRA adapter, and runner_def.py has one model object for all roles. So the
    `zeroshot` row in THIS table is this cascade with no adapter — an internally consistent
    baseline, and the right one to compare the adapter rows against. It is not numerically
    identical to your published V4 table, which came from runner_def.py with its tier-0
    calibration, repair probe and span fallback. Compare within this table; quote the V4
    table separately.

    The tier-0 retrieval margin IS imported (attack_memory), so the LEADER sees the same
    screen field it saw in training and in V4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

import ft_config as C
import ft_roles as R
import ft_vllm as V


def _bootstrap() -> None:
    cands = [os.environ.get(k) for k in
             ("WB_BB_DIR", "WB_UCA_DIR", "WB_T35_DIR", "WB_COMP_DIR",
              "WB_FEAS_DIR", "WB_SRC_DIR")]
    if os.path.isdir(C.ROOT):
        try:
            cands += [os.path.join(C.ROOT, d) for d in sorted(os.listdir(C.ROOT))
                      if d.startswith("lim_")]
        except OSError:
            pass
    for c in cands:
        if c and os.path.isdir(c) and c not in sys.path:
            sys.path.append(c)
    for p in (C.V4_DIR, os.path.dirname(os.path.abspath(__file__))):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


_bootstrap()


# ======================================================================================
# the eval set — imported from the V4 code so it is the same corpus
# ======================================================================================
@dataclass
class DSArgs:
    text_column: str = C.COLUMNS["text"]
    sections_column: str = C.COLUMNS["sections"]
    author_lim_column: str = C.COLUMNS["author_lim"]
    title_column: str = "title"
    id_column: str = "paper_id"
    slot: str = "limitations_end"
    max_chars: int = 30000


def load_eval_set(benign_csv: str, attacked_csv: str, *, id_column: str = "paper_id",
                  max_papers: int = 10 ** 9, seed: int = 0, all_papers: bool = True,
                  memory_dir: str = "") -> List:
    import def_dataset as DS
    a = DSArgs(id_column=id_column)
    attacked = DS.load_attacked(attacked_csv)
    keep = sorted({str(p) for p in attacked["paper_id"]})
    if not all_papers and memory_dir:
        sp = os.path.join(memory_dir, "splits.json")
        if os.path.exists(sp):
            ev = json.load(open(sp)).get("eval_papers") or []
            if ev:
                keep = [str(p) for p in ev]
                print(f"[data] restricted to the {len(keep)} V4 EVAL papers")
    recs = DS.build_eval_set(benign_csv, attacked, keep, a, max_papers=max_papers,
                             seed=seed)
    if not recs:
        raise SystemExit(
            "[FATAL] the evaluation set is EMPTY. The benign CSV and metrics_detail.csv "
            "do not share paper ids — check --id-column (your 500-paper file uses `id`, "
            "metrics_detail.csv uses `paper_id`).")
    return recs


def load_memory(memory_dir: str):
    """The V4 tier-0 memory, so the screen score in the LEADER prompt is the V4 one."""
    p = os.path.join(memory_dir, "memory.json")
    if not os.path.exists(p):
        print(f"[tier0] {p} not found — the LEADER will be told the screen is "
              f"'not available'. That is a DIFFERENT prompt from the one the adapters were "
              f"trained on, so the adapter rows will be understated. Point --memory-dir at "
              f"your llm_agents_v4/shared.")
        return None
    from attack_memory import AttackMemory
    mem = AttackMemory.load(p)
    print(f"[tier0] {len(mem.entries)} attack payloads + "
          f"{len(getattr(mem, 'benign_texts', []) or [])} benign reference passages")
    return mem


def screen_of(mem, document: str) -> Optional[float]:
    if mem is None:
        return None
    try:
        import defense as DEF
        _o, reg = DEF.limitations_region(document, 6000)
        return float(DEF.screen_document(document, mem, section=reg).score)
    except Exception:
        try:
            from attack_memory import candidate_spans
            best = None
            for sp in candidate_spans(document)[:40]:
                m = mem.margin(sp) if hasattr(mem, "margin") else None
                if m is None:
                    return None
                best = m if best is None else max(best, m)
            return float(best) if best is not None else None
        except Exception:
            return None


# ======================================================================================
# the cascade
# ======================================================================================
@dataclass
class Routing:
    """Which served name answers for which role."""
    base: str
    per_role: Dict[str, str] = field(default_factory=dict)

    def name(self, role: str) -> str:
        return self.per_role.get(role, self.base)

    def describe(self) -> str:
        if not self.per_role:
            return f"  every role -> {self.base}"
        return "\n".join(f"  {r:<16} -> {self.per_role.get(r, self.base)}"
                         for r in C.ROLES)


def build_routing(model_key: str, ablation: str, base_served: str) -> Routing:
    ab = next((x for x in C.ABLATIONS if x.key == ablation), None)
    if ab is None:
        raise SystemExit(f"[FATAL] unknown ablation {ablation!r}; expected one of "
                         f"{[x.key for x in C.ABLATIONS]}")
    if ab.stage is None:
        return Routing(base_served)
    if ab.per_agent:
        return Routing(base_served,
                       {r: C.lora_name(model_key, ab.stage, r) for r in C.ROLES})
    one = C.lora_name(model_key, ab.stage)
    return Routing(base_served, {r: one for r in C.ROLES})


def run_one(rec, cl: V.VLLMClient, routing: Routing, *, mem=None, system_role: bool = True,
            min_conf: float = 0.5, region_chars: int = 6000,
            escalate_low: float = -0.15, escalate_high: float = 0.35,
            tier0_only: bool = False, keep_raw: bool = False) -> Dict:
    """
    Tier 0 -> chunk verifier -> workers -> leader, escalating only what the tier above
    could not settle. The cost column in the table is real because of that ordering.
    """
    doc = rec.document
    label = int(rec.label)
    out: Dict = {"doc_id": rec.doc_id, "paper_id": rec.paper_id, "label": label,
                 "condition": rec.condition,
                 "harmful": int(bool(getattr(rec, "asr", None))),
                 "n_llm_calls": 0, "escalated": 0, "pred": 0,
                 "attack_type": "", "severity": 0, "span": "", "agent_reason": "",
                 "verifier_said": "", "n_findings": 0}

    sc = screen_of(mem, doc)
    out["screen_score"] = sc
    if sc is not None and sc >= escalate_high:
        out["pred"] = 1
        out["attack_type"] = "OTHER"
        out["agent_reason"] = "tier-0 retrieval margin above the high gate"
        return out
    if tier0_only or (sc is not None and sc < escalate_low):
        return out
    out["escalated"] = 1

    calls = 0

    # ---- chunk verifier: the cheapest escalation ---------------------------------
    import ft_trajectories as T
    verified = False
    for ch, _has in T.chunks_for(doc, "", chunk_chars=800, max_chunks=3):
        sysmsg, user = R.build_prompt("CHUNK_VERIFIER", chunk=ch)
        msgs = R.messages("CHUNK_VERIFIER", sysmsg, user, system_role=system_role)
        try:
            raw = cl.chat(msgs, model=routing.name("CHUNK_VERIFIER"), temperature=0.0,
                          max_tokens=R.ROLES["CHUNK_VERIFIER"].max_tokens)[0]
        except V.VLLMError as e:
            out["agent_reason"] = f"verifier error: {e}"
            break
        calls += 1
        obj = V.extract_json(raw) or {}
        v = obj.get("is_instruction")
        if isinstance(v, str):
            v = v.strip().lower() in ("true", "yes", "1")
        try:
            conf = float(obj.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        span = str(obj.get("span") or "")
        # The grounded-span rule, at the verifier too: a flag it cannot quote is an
        # opinion. This is the single change that took V2's FPR from 1.000 down.
        if bool(v) and conf >= min_conf and R.verbatim(span, ch):
            verified = True
            out["verifier_said"] = "instruction"
            out["span"] = span[:600]
            break
    if not verified:
        out["verifier_said"] = out["verifier_said"] or "authors"

    # ---- the three evidence workers ----------------------------------------------
    findings: Dict[str, List[Dict]] = {}
    for w in ("INSTRUCTION", "PRESSURE", "VOICE"):
        sysmsg, user = R.build_prompt(w, document=doc, region_chars=region_chars)
        msgs = R.messages(w, sysmsg, user, system_role=system_role)
        try:
            raw = cl.chat(msgs, model=routing.name(w), temperature=0.0,
                          max_tokens=R.ROLES[w].max_tokens)[0]
        except V.VLLMError:
            findings[w] = []
            continue
        calls += 1
        obj = V.extract_json(raw) or {}
        got = obj.get("findings") or []
        # Only KEEP findings whose span is verbatim. Passing unquotable evidence to the
        # leader is how V2 assembled a page of incriminating material about clean papers.
        kept = [f for f in got if isinstance(f, dict)
                and R.verbatim(str(f.get("span") or ""), R.region(doc, region_chars))]
        findings[w] = kept
    out["n_findings"] = sum(len(v) for v in findings.values())

    # ---- leader -------------------------------------------------------------------
    sysmsg, user = R.build_prompt("LEADER", document=doc, screen_score=sc,
                                 findings=findings, region_chars=region_chars)
    msgs = R.messages("LEADER", sysmsg, user, system_role=system_role)
    try:
        raw = cl.chat(msgs, model=routing.name("LEADER"), temperature=0.0,
                      max_tokens=R.ROLES["LEADER"].max_tokens)[0]
    except V.VLLMError as e:
        out["n_llm_calls"] = calls
        out["agent_reason"] = f"leader error: {e}"
        return out
    calls += 1
    obj = V.extract_json(raw) or {}
    verdict = str(obj.get("verdict") or "").strip().upper()
    span = str(obj.get("span") or "")
    grounded = R.verbatim(span, R.region(doc, region_chars))
    if verdict == "ATTACKED" and grounded:
        out["pred"] = 1
        out["attack_type"] = str(obj.get("attack_type") or "OTHER")[:40]
        try:
            out["severity"] = int(obj.get("severity") or 0)
        except Exception:
            out["severity"] = R.SEVERITY.get(out["attack_type"], 3)
        out["span"] = span[:600]
    elif verdict == "ATTACKED":
        # flagged but could not quote. Recorded, not counted — and recorded so the column
        # exists, because "how often does the leader flag without evidence" is the number
        # that explains an FPR.
        out["agent_reason"] = "ATTACKED verdict DISCARDED: no verbatim span. "
    out["agent_reason"] += str(obj.get("reason") or "")[:600]
    out["location"] = str(obj.get("location") or "")[:200]
    if keep_raw:
        out["leader_raw"] = raw[:4000]
    out["n_llm_calls"] = calls
    return out


# ======================================================================================
# metrics
# ======================================================================================
def metrics(rows: List[Dict], subset: str = "all") -> Dict:
    R_ = rows
    if subset == "harmful":
        R_ = [r for r in rows if r["label"] == 0 or r.get("harmful")]
    tp = sum(1 for r in R_ if r["label"] == 1 and r["pred"] == 1)
    fp = sum(1 for r in R_ if r["label"] == 0 and r["pred"] == 1)
    tn = sum(1 for r in R_ if r["label"] == 0 and r["pred"] == 0)
    fn = sum(1 for r in R_ if r["label"] == 1 and r["pred"] == 0)
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if n else None)
    return {"subset": subset, "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / n if n else None,
            "precision": prec, "recall": rec, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else None,
            "escalated_pct": 100.0 * sum(r["escalated"] for r in R_) / n if n else None,
            "mean_llm_calls": sum(r["n_llm_calls"] for r in R_) / n if n else None}


def main() -> int:
    p = argparse.ArgumentParser(description="run the cascade on the V4 test corpus")
    p.add_argument("--model", required=True, choices=sorted(C.MODELS))
    p.add_argument("--ablation", required=True,
                   choices=[a.key for a in C.ABLATIONS])
    p.add_argument("--benign-csv", dest="benign_csv", default=C.TEST_CSV)
    p.add_argument("--attacked-csv", dest="attacked_csv", required=True,
                   help="the V4 metrics_detail.csv for this reviewer and seed — the same "
                        "file the zero-shot run used")
    p.add_argument("--memory-dir", dest="memory_dir", default=C.wpath("shared"),
                   help="llm_agents_v4/shared, for the tier-0 memory and splits")
    p.add_argument("--out-dir", dest="out_dir", default="")
    p.add_argument("--vllm-url", dest="vllm_url", default=os.environ.get("VLLM_URL", ""))
    p.add_argument("--vllm-model", dest="vllm_model",
                   default=os.environ.get("VLLM_MODEL", ""))
    p.add_argument("--id-column", dest="id_column", default="paper_id")
    p.add_argument("--max-papers", dest="max_papers", type=int, default=10 ** 9)
    p.add_argument("--min-conf", dest="min_conf", type=float, default=0.5)
    p.add_argument("--region-chars", dest="region_chars", type=int, default=6000)
    p.add_argument("--escalate-low", dest="escalate_low", type=float, default=-0.15)
    p.add_argument("--escalate-high", dest="escalate_high", type=float, default=0.35)
    p.add_argument("--tier0-only", dest="tier0_only", action="store_true")
    p.add_argument("--eval-papers-only", dest="all_papers", action="store_false")
    p.set_defaults(all_papers=True)
    p.add_argument("--allow-role-fallback", dest="allow_role_fallback",
                   action="store_true",
                   help="if a per-role adapter is not served, use the shared one for that "
                        "role instead of failing. For SMOKE TESTS only: the resulting row "
                        "is a partial per-agent configuration and is labelled as such in "
                        "summary.json.")
    p.add_argument("--keep-raw", dest="keep_raw", action="store_true")
    p.add_argument("--save-every", dest="save_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if not a.vllm_url:
        raise SystemExit("[FATAL] --vllm-url is required")
    out_dir = a.out_dir or C.wpath(C.D_RUNS, a.model.replace("/", "_"), a.ablation)
    os.makedirs(out_dir, exist_ok=True)

    spec = C.MODELS[a.model]
    base_served = a.vllm_model or spec.path
    cl = V.VLLMClient(a.vllm_url, base_served, temperature=0.0,
                      chat_kwargs=spec.chat_kwargs)
    routing = build_routing(a.model, a.ablation, base_served)

    print("=" * 92)
    print(f"INFERENCE — {a.model}   ablation={a.ablation}")
    print("=" * 92)
    print(routing.describe())
    # THE CHECK THAT SAVES THE ABLATION. Asking for an adapter that is not served would
    # 404; asking for the base when an adapter was intended returns a valid answer from an
    # untrained model, and the row then reports the baseline under a fine-tuned label.
    need = sorted(set(routing.per_role.values())) if routing.per_role else []
    if need and a.allow_role_fallback:
        # SMOKE-TEST PATH. A per-role adapter that was not trained falls back to the shared
        # one. Useful when only some roles have been trained, and DANGEROUS if it goes
        # unnoticed — a row labelled "per agent" that is four-fifths shared adapter is not
        # the configuration its name claims. So it is off by default, it is printed, and it
        # is recorded in summary.json.
        have = set(cl.models())
        shared = C.lora_name(a.model, "dpo")
        fell_back = []
        for role, name in list(routing.per_role.items()):
            if name not in have:
                routing.per_role[role] = shared if shared in have else base_served
                fell_back.append(role)
        if fell_back:
            print(f"[serve] ROLE FALLBACK for {fell_back}: no per-role adapter served, "
                  f"using {routing.per_role[fell_back[0]]!r} instead.")
            print(f"[serve] This row is therefore a PARTIAL per-agent configuration. Fine "
                  f"for a smoke test; do not report it as the per-agent result.")
        need = sorted(set(routing.per_role.values()))
    if need:
        cl.require(need)
        print(f"[serve] all {len(need)} adapter name(s) confirmed on the server")
    else:
        print("[serve] no adapter: this is the zero-shot baseline row")

    mem = load_memory(a.memory_dir)
    recs = load_eval_set(a.benign_csv, a.attacked_csv, id_column=a.id_column,
                         max_papers=a.max_papers, seed=a.seed,
                         all_papers=a.all_papers, memory_dir=a.memory_dir)
    n_att = sum(1 for r in recs if r.label == 1)
    print(f"[data] {len(recs)} documents = {n_att} attacked + {len(recs)-n_att} clean")

    rows: List[Dict] = []
    t0 = time.time()
    detail = os.path.join(out_dir, "stage1_detail.csv")
    for k, rec in enumerate(recs):
        row = run_one(rec, cl, routing, mem=mem, system_role=spec.system_role,
                      min_conf=a.min_conf, region_chars=a.region_chars,
                      escalate_low=a.escalate_low, escalate_high=a.escalate_high,
                      tier0_only=a.tier0_only, keep_raw=a.keep_raw)
        rows.append(row)
        hit = "OK " if row["pred"] == rec.label else "MISS"
        s = row.get("screen_score")
        ss = f"{s:+.3f}" if isinstance(s, float) else " n/a "
        print(f"[{k+1}/{len(recs)}] {hit} {rec.doc_id:<30} label={rec.label} "
              f"pred={row['pred']} screen={ss} esc={row['escalated']} "
              f"calls={row['n_llm_calls']} ({(time.time()-t0)/60:.1f}m)", flush=True)
        if (k + 1) % max(1, a.save_every) == 0:
            pd.DataFrame(rows).to_csv(detail, index=False)
    pd.DataFrame(rows).to_csv(detail, index=False)

    m_all, m_harm = metrics(rows, "all"), metrics(rows, "harmful")
    summary = {"model": a.model, "ablation": a.ablation,
               "routing": routing.per_role or {"*": routing.base},
               "role_fallback_used": bool(a.allow_role_fallback and need and
                                          len(set(routing.per_role.values())) <
                                          len(C.ROLES) and routing.per_role),
               "n_documents": len(rows), "minutes": (time.time() - t0) / 60,
               "all": m_all, "harmful": m_harm,
               "n_llm_calls_total": cl.n_calls}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 92)
    print(f"RESULT — {a.model} / {a.ablation}")
    print("=" * 92)
    print(pd.DataFrame([m_all, m_harm]).round(4).to_string(index=False))
    print(f"\n[saved] {detail}")
    print(f"[saved] {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
