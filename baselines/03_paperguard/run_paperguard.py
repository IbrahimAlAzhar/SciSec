#!/usr/bin/env python3
"""
run_paperguard.py — PaperGuard's chunk-based embedding search on the SFT/DPO test corpus,
with gpt-4o-mini as the verifier LLM.

SAME INPUT, SAME OUTPUT FORMAT
=======================================================================================
INPUT   The documents come from infer_ladder.load_eval_set -> def_dataset.build_eval_set,
        with the same benign CSV, the same metrics_detail.csv, the same --id-column and the
        same --seed 0 as job_c_gemma_infer.pbs. So the 990 documents scored here are the
        990 documents the SFT/DPO rows were scored on, in the same order.

OUTPUT  (all under PG_WORK = defense_baselines/paperguard_defense)
        runs/<ablation>/stage1_detail.csv   one row per document, the columns
                                            infer_ladder.py writes (doc_id, paper_id,
                                            label, condition, harmful, n_llm_calls,
                                            escalated, pred, ...) plus pg_* columns
        runs/<ablation>/summary.json        the infer_ladder schema: "all" and "harmful"
        runs/<ablation>/pg_detail.jsonl     every verdict in full (resumable)
        runs/<ablation>/registry.json       the known-attack set actually used
        runs/eval_docs.jsonl.gz             the documents, for pg_rerate.py
        <ablation> = paperguard_orig__gpt-4o-mini | paperguard_train__gpt-4o-mini

So ft_table.py, category_defense_metrics.py and count_defense_labels.py read these files
unchanged, and pg_table.py can check row by row that the corpus is the SFT/DPO one.

WHAT THE COLUMNS MEAN FOR THIS METHOD
    pred         1 iff some known pattern was verified in its top-k chunks
    n_llm_calls  verifier calls for DETECTION (<= registry size; stops at first positive)
    escalated    1 for every document that reached the verifier. PaperGuard has no
                 cheaper tier in front of the LLM, so this is 1 everywhere (Esc = 100%).
    pg_score     max cosine similarity between any known pattern and any chunk — the
                 retrieval stage's continuous score, used for AUROC / R@FPR only
    pg_n_loc_calls  extra verifier calls spent LOCALIZING flagged documents (not in Calls)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import pandas as pd

import pg_config as P

for _p in (P.FT_CODE_DIR, P.V4_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import pg_core as G                                   # noqa: E402
from pg_llm import ChatLLM                            # noqa: E402


# ======================================================================================
# the corpus
# ======================================================================================
def load_documents(a) -> List[Dict]:
    """The SFT/DPO test documents, via THEIR loader. Nothing here rebuilds the corpus."""
    if a.eval_docs:
        return read_eval_docs(a.eval_docs)
    try:
        from infer_ladder import load_eval_set
    except Exception as e:
        raise SystemExit(
            f"[FATAL] cannot import infer_ladder.load_eval_set from FT_CODE_DIR="
            f"{P.FT_CODE_DIR} ({type(e).__name__}: {e}). The baseline must read the SAME "
            f"corpus as the SFT/DPO rows, so it does not fall back to anything else. Set "
            f"FT_CODE_DIR / V4_DIR and the WB_* paths the way job_c_gemma_infer.pbs does.")
    recs = load_eval_set(a.benign_csv, a.attacked_csv, id_column=a.id_column,
                         max_papers=a.max_papers, seed=a.seed,
                         all_papers=a.all_papers, memory_dir=a.memory_dir)
    docs = []
    for k, rec in enumerate(recs):
        d = {"k": k, "doc_id": str(rec.doc_id), "paper_id": str(rec.paper_id),
             "label": int(rec.label), "condition": str(rec.condition),
             # THE SAME RULE infer_ladder.run_one uses, so HarmRec has the same denominator
             "harmful": int(bool(getattr(rec, "asr", None))),
             "document": rec.document or ""}
        for extra in ("row_index", "insert", "strategy"):
            v = getattr(rec, extra, None)
            if v is not None and not (isinstance(v, float) and v != v):
                d[extra] = v if isinstance(v, (int, float, str)) else str(v)
        docs.append(d)
    return docs


def read_eval_docs(path: str) -> List[Dict]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_eval_docs(path: str, docs: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def signature(docs: List[Dict]) -> str:
    h = hashlib.sha1()
    for d in docs:
        h.update(f"{d['paper_id']}\x1f{d['condition']}\x1f{d['label']}\x1f".encode())
        h.update(hashlib.sha1(d["document"].encode("utf-8")).digest())
    return h.hexdigest()


# ======================================================================================
# metrics — the formulas of infer_ladder.metrics, unchanged
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


# ======================================================================================
# main
# ======================================================================================
class CachedVerifier:
    """
    Verifier answers cached by (model, temperature, max_tokens, prompt).

    Why: the ablation registry starts with the baseline's 10 patterns, which retrieve the
    same chunks and so build byte-identical verifier prompts. Reusing those answers makes
    the ablation differ from the baseline ONLY by the added patterns, not by API sampling
    noise, and it does not pay twice for them. Failed calls are never cached.
    """

    def __init__(self, llm: ChatLLM, path: str):
        self.llm, self.path = llm, path
        self.cache: Dict[str, str] = {}
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                try:
                    r = json.loads(line)
                    self.cache[r["h"]] = r["out"]
                except Exception:
                    pass
        self._f = open(path, "a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()
        self.hits = 0

    def __call__(self, prompt: str) -> str:
        h = hashlib.sha1(json.dumps([self.llm.model, self.llm.temperature,
                                     self.llm.max_tokens, prompt]).encode("utf-8")
                         ).hexdigest()
        with self._lock:
            if h in self.cache:
                self.hits += 1
                return self.cache[h]
        out = self.llm(prompt)
        with self._lock:
            self.cache[h] = out
            self._f.write(json.dumps({"h": h, "out": out}, ensure_ascii=False) + "\n")
        return out


def build_registry(a, docs: List[Dict], embedder: G.Embedder) -> List[G.KnownAttack]:
    if a.registry == "paperguard":
        reg = G.registry_paperguard(seed=a.pg_seed, a1_size=a.a1_size)
        print(f"[registry] PaperGuard A1 (seed={a.pg_seed}, a1_size={a.a1_size}): "
              f"{len(reg)} known prompts")
    elif a.registry == "combined":
        path = a.train_documents or P.find_train_documents()
        if not path or not os.path.exists(path):
            raise SystemExit(
                "[FATAL] --registry combined needs the attacked TRAIN corpus "
                "(train_documents.csv from ft_attack.py). Looked in:\n  " +
                "\n  ".join(P.train_documents_candidates()))
        reg = G.registry_combined(path, embedder, seed=a.pg_seed, a1_size=a.a1_size,
                                  per_family=a.per_family,
                                  exclude_paper_ids={d["paper_id"] for d in docs},
                                  select=a.registry_select)
        n_pg = sum(1 for r in reg if r.source == "paperguard_A1")
        print(f"[registry] ABLATION (not a baseline): PaperGuard A1 ({n_pg} prompts, "
              f"same order as the baseline) + {len(reg) - n_pg} TRAIN medoids")
    else:
        raise SystemExit(f"[FATAL] unknown registry {a.registry!r}")
    for i, r in enumerate(reg):
        t = " ".join(r.text.split())
        print(f"[registry]   {i:>2} {r.name:<36} {t[:90]!r}")
    return reg


def main() -> int:
    p = argparse.ArgumentParser(description="PaperGuard chunk-based EmbSearch baseline")
    p.add_argument("--registry", choices=sorted(P.REGISTRIES), default="paperguard")
    p.add_argument("--corpus-model", "--model", dest="model", default=P.CORPUS_MODEL,
                   help="which attacked corpus (the SFT/DPO rows' metrics_detail.csv); "
                        "NOT the detector")
    # --- corpus: the same flags as infer_ladder.py
    p.add_argument("--benign-csv", dest="benign_csv", default=P.TEST_CSV)
    p.add_argument("--attacked-csv", dest="attacked_csv", default="")
    p.add_argument("--memory-dir", dest="memory_dir", default="")
    p.add_argument("--id-column", dest="id_column", default="id")
    p.add_argument("--max-papers", dest="max_papers", type=int, default=10 ** 9)
    p.add_argument("--seed", type=int, default=0, help="eval-set seed (infer_ladder's)")
    p.add_argument("--eval-papers-only", dest="all_papers", action="store_false")
    p.set_defaults(all_papers=True)
    p.add_argument("--eval-docs", dest="eval_docs", default="",
                   help="read documents from an eval_docs.jsonl.gz written by an earlier "
                        "run instead of rebuilding them (identical corpus, no attack code "
                        "needed)")
    # --- PaperGuard settings
    p.add_argument("--top-k", dest="top_k", type=int, default=P.PG.top_k)
    p.add_argument("--max-len", dest="max_len", type=int, default=P.PG.max_len)
    p.add_argument("--a1-size", dest="a1_size", type=int, default=P.PG.a1_size)
    p.add_argument("--pg-seed", dest="pg_seed", type=int, default=P.PG.seed)
    p.add_argument("--train-documents", dest="train_documents", default="")
    p.add_argument("--per-family", dest="per_family", type=int, default=1)
    p.add_argument("--registry-select", dest="registry_select",
                   choices=["medoid", "random"], default="medoid")
    p.add_argument("--embed-backend", dest="embed_backend",
                   choices=["openai", "hf", "hash"],
                   default=os.environ.get("EMBED_BACKEND", "openai"))
    p.add_argument("--embed-model", dest="embed_model",
                   default=os.environ.get("EMBED_MODEL", P.PG.embed_model))
    p.add_argument("--embed-cache", dest="embed_cache", default="")
    p.add_argument("--verifier-max-tokens", dest="v_max_tokens", type=int,
                   default=P.PG.verifier_max_tokens)
    p.add_argument("--verifier-temperature", dest="v_temperature", type=float,
                   default=P.PG.verifier_temperature)
    p.add_argument("--no-localize", dest="localize", action="store_false")
    p.set_defaults(localize=True)
    # --- the verifier LLM
    p.add_argument("--llm", default=P.LLM, help="verifier chat model (default gpt-4o-mini)")
    p.add_argument("--llm-backend", dest="llm_backend", choices=["openai", "vllm"],
                   default="openai")
    p.add_argument("--llm-base-url", dest="llm_base_url", default="")
    # --- io
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--work", default=P.PG_WORK)
    p.add_argument("--out-dir", dest="out_dir", default="")
    p.add_argument("--fresh", action="store_true",
                   help="ignore pg_detail.jsonl from an earlier, interrupted run")
    a = p.parse_args()

    ablation = P.ablation_name(a.registry, a.llm)
    label = P.row_label(a.registry, a.llm)
    out_dir = a.out_dir or P.run_dir(ablation, a.work)
    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.dirname(out_dir.rstrip("/"))
    if not a.attacked_csv:
        a.attacked_csv = P.attacked_csv(a.model)
    if a.embed_backend == "hash":
        print("[WARN] --embed-backend hash is a SELF-TEST embedder with no semantics. "
              "Nothing produced with it may be reported.")
    print("=" * 92)
    print(f"PAPERGUARD EMBSEARCH — {label}")
    print("=" * 92)
    print(f"  corpus     {a.model} attack set: {a.attacked_csv}")
    print(f"  verifier   {a.llm} ({a.llm_backend})  temperature {a.v_temperature}, "
          f"max_tokens {a.v_max_tokens}")
    print(f"  embedder   {a.embed_backend}:{a.embed_model}")
    print(f"  top-k {a.top_k}   chunk max_len {a.max_len}   localize {a.localize}")
    print(f"  out        {out_dir}")

    # ---------------------------------------------------------------- corpus
    shared_docs = os.path.join(model_dir, "eval_docs.jsonl.gz")
    docs = load_documents(a)
    sig = signature(docs)
    if os.path.exists(shared_docs) and not a.eval_docs:
        old = read_eval_docs(shared_docs)
        if signature(old) != sig:
            if a.max_papers < 10 ** 9:
                print(f"[data] {shared_docs} holds a different corpus (capped run); "
                      f"writing this one next to the run instead")
                shared_docs = os.path.join(out_dir, "eval_docs.jsonl.gz")
                write_eval_docs(shared_docs, docs)
            else:
                raise SystemExit(
                    f"[FATAL] {shared_docs} was written from a DIFFERENT corpus than the "
                    f"one just built. Two registries scored on different documents are not "
                    f"comparable. Delete it only if you know why it changed.")
    elif not a.eval_docs:
        write_eval_docs(shared_docs, docs)
    n_att = sum(d["label"] for d in docs)
    print(f"[data] {len(docs)} documents = {n_att} attacked + {len(docs) - n_att} clean, "
          f"{len({d['paper_id'] for d in docs})} papers  (signature {sig[:12]})")

    # ---------------------------------------------------------------- defense
    cache = a.embed_cache or os.path.join(a.work, "emb_cache",
                                          f"{a.embed_backend}_{a.embed_model.replace('/', '_')}.sqlite")
    emb = G.Embedder(a.embed_backend, a.embed_model, cache_path=cache,
                     batch_size=P.PG.embed_batch)
    registry = build_registry(a, docs, emb)
    reg_hash = hashlib.sha1(json.dumps([r.text for r in registry]).encode()).hexdigest()
    with open(os.path.join(out_dir, "registry.json"), "w") as f:
        json.dump({"registry": a.registry, "hash": reg_hash,
                   "items": [r.as_dict() for r in registry]}, f, indent=2,
                  ensure_ascii=False)
    pg = G.PaperGuardEmbSearch(registry, emb, top_k=a.top_k, max_len=a.max_len)
    t_emb = time.time()
    seg_data = pg.embed_documents([d["document"] for d in docs])
    t_emb = (time.time() - t_emb) / 60

    ver = ChatLLM(a.llm, backend=a.llm_backend, base_url=a.llm_base_url,
                  temperature=a.v_temperature, max_tokens=a.v_max_tokens, seed=0)
    try:
        print(f"[llm] {a.llm} answered the probe: {ver.check()!r}")
    except Exception as e:
        raise SystemExit(f"[FATAL] the verifier LLM is not reachable: {e}")
    vcache_path = os.path.join(model_dir, f"verifier_cache__{P.llm_slug(a.llm)}.jsonl")
    vfn = CachedVerifier(ver, vcache_path)
    print(f"[llm] verifier cache {vcache_path}: {len(vfn.cache)} stored answers")

    # ---------------------------------------------------------------- resume
    detail_jsonl = os.path.join(out_dir, "pg_detail.jsonl")
    cfg_key = json.dumps({"reg": reg_hash, "sig": sig, "top_k": a.top_k,
                          "max_len": a.max_len, "emb": emb.cache_model_key,
                          "vt": a.v_temperature, "vm": a.v_max_tokens,
                          "llm": f"{a.llm_backend}:{a.llm}",
                          "loc": a.localize}, sort_keys=True)
    cfg_path = os.path.join(out_dir, "run_config.json")
    done: Dict[int, Dict] = {}
    if os.path.exists(detail_jsonl) and not a.fresh:
        old_cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        if old_cfg.get("key") == cfg_key:
            n_retry = 0
            for line in open(detail_jsonl):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if int(r.get("n_errors") or 0) > 0:
                    # a transport failure is not a verdict; score that document again
                    done.pop(int(r["k"]), None)
                    n_retry += 1
                    continue
                done[int(r["k"])] = r
            print(f"[resume] {len(done)} documents already scored with this exact "
                  f"configuration; continuing ({n_retry} with verifier errors will be "
                  f"re-scored)")
        else:
            print("[resume] earlier pg_detail.jsonl was produced with a different "
                  "configuration — starting over")
            os.replace(detail_jsonl, detail_jsonl + ".stale")
    with open(cfg_path, "w") as f:
        json.dump({"key": cfg_key, "args": vars(a)}, f, indent=2)

    lock = threading.Lock()
    fout = open(detail_jsonl, "a", buffering=1)
    t0 = time.time()
    todo = [d for d in docs if d["k"] not in done]

    def work(d: Dict) -> Dict:
        segs, E = seg_data[d["k"]]
        v = pg.detect(segs, E, vfn, localize=a.localize)
        r = {"k": d["k"], "doc_id": d["doc_id"], **G.to_jsonable(v),
             "located_text": [segs[i] for i in v.located],
             "flag_topk_text": [segs[i][:1200] for i in v.flag_topk]}
        with lock:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        return r

    n_done = len(done)
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(work, d): d for d in todo}
        for fu in as_completed(futs):
            d = futs[fu]
            r = fu.result()
            done[d["k"]] = r
            n_done += 1
            hit = "OK " if int(r["is_attack"]) == d["label"] else "MISS"
            print(f"[{n_done}/{len(docs)}] {hit} {d['doc_id']:<30} label={d['label']} "
                  f"pred={int(r['is_attack'])} calls={r['n_calls']} "
                  f"sim={(r['max_sim'] or 0):.3f} q={r['flag_query_name'] or '-'} "
                  f"({(time.time() - t0) / 60:.1f}m)", flush=True)
    fout.close()

    # ---------------------------------------------------------------- rows
    rows = []
    for d in docs:
        r = done[d["k"]]
        rows.append({
            "doc_id": d["doc_id"], "paper_id": d["paper_id"], "label": d["label"],
            "condition": d["condition"], "harmful": d["harmful"],
            **({"row_index": d["row_index"]} if "row_index" in d else {}),
            "n_llm_calls": int(r["n_calls"]),
            "escalated": int(r["n_calls"] > 0),
            "pred": int(bool(r["is_attack"])),
            "attack_type": "OTHER" if r["is_attack"] else "",
            "severity": 0,
            "span": (" ".join(r.get("located_text") or [])[:600]),
            "agent_reason": (r.get("explanation") or "")[:600],
            "verifier_said": "unsafe" if r["is_attack"] else "safe",
            "n_findings": len(r.get("located") or []),
            "pg_score": r.get("max_sim"),
            "pg_flag_query": r.get("flag_query_name") or "",
            "pg_n_segments": r.get("n_segments"),
            "pg_n_queries": r.get("n_queries"),
            "pg_n_loc_calls": int(r.get("n_loc_calls") or 0),
            "pg_located": json.dumps(r.get("located") or []),
            "pg_loc_method": r.get("loc_method") or "",
            "pg_parse_lenient": int((r.get("parse") or {}).get("lenient", 0)),
            "pg_parse_fail": int((r.get("parse") or {}).get("fail", 0)),
            "pg_errors": int(r.get("n_errors") or 0),
        })
    detail = os.path.join(out_dir, "stage1_detail.csv")
    pd.DataFrame(rows).to_csv(detail, index=False)

    m_all, m_harm = metrics(rows, "all"), metrics(rows, "harmful")
    tot = lambda k: int(sum(r[k] for r in rows))
    summary = {
        "model": a.model, "ablation": ablation, "label": label,
        "role": P.REGISTRIES[a.registry].role,
        "method": "PaperGuard chunk-based embedding search (retrieve-then-verify)",
        "corpus": a.model, "verifier": a.llm,
        "routing": {"*": a.llm},
        "n_documents": len(rows), "minutes": (time.time() - t0) / 60,
        "minutes_embedding": t_emb,
        "all": m_all, "harmful": m_harm,
        "n_llm_calls_total": tot("n_llm_calls"),
        "pg": {
            "registry": a.registry, "registry_size": len(registry),
            "registry_names": [r.name for r in registry], "registry_hash": reg_hash,
            "top_k": a.top_k, "max_len": a.max_len,
            "embedder": emb.cache_model_key,
            "embed_api_calls": emb.n_api_calls, "embedded_texts": emb.n_embedded,
            "embed_truncated_inputs": emb.n_truncated,
            "verifier_llm": a.llm, "verifier_backend": a.llm_backend,
            "verifier_usage": ver.stats(), "verifier_cache_hits": vfn.hits,
            "verifier_temperature": a.v_temperature,
            "verifier_max_tokens": a.v_max_tokens,
            "localize": a.localize,
            "n_loc_calls_total": tot("pg_n_loc_calls"),
            "mean_loc_calls_flagged": (float(np.mean([r["pg_n_loc_calls"] for r in rows
                                                      if r["pred"]]))
                                       if any(r["pred"] for r in rows) else None),
            "parse_lenient": tot("pg_parse_lenient"), "parse_fail": tot("pg_parse_fail"),
            "verifier_errors": tot("pg_errors"),
            "corpus_signature": sig,
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 92)
    role = P.REGISTRIES[a.registry].role.upper()
    print(f"RESULT [{role}] — {ablation}  ({label})")
    print("=" * 92)
    print(pd.DataFrame([m_all, m_harm]).round(4).to_string(index=False))
    pgs = summary["pg"]
    u = ver.stats()
    print(f"\n  verifier calls: detection {summary['n_llm_calls_total']}, localization "
          f"{pgs['n_loc_calls_total']}  ({u['prompt_tokens']} prompt + "
          f"{u['completion_tokens']} completion tokens this session, "
          f"{u['retries']} retries)")
    print(f"  parse: {pgs['parse_lenient']} lenient, {pgs['parse_fail']} unreadable; "
          f"{pgs['verifier_errors']} verifier transport errors")
    if pgs["verifier_errors"] > 0.01 * max(1, summary["n_llm_calls_total"]):
        print("  [WARN] more than 1% of verifier calls FAILED. Those patterns were scored "
              "safe, so recall is understated. Re-run (the run resumes) once the server "
              "is reachable.")
    print(f"\n[saved] {detail}")
    print(f"[saved] {os.path.join(out_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
