#!/usr/bin/env python3
"""
v8_eval.py — evaluate ONE ablation on the TEST pairs, for both tasks.

Per pair (benign B, attacked A of the same paper):

    detection     cascade on B and on A                        -> 2 documents, labels 0/1
    review        R0 = reviewer(B), RA = reviewer(A)            (base model, no adapter)
    repair        if a document is flagged: LEADER-localize -> excise -> normalize
                  RD  = reviewer(repaired A)                    restoration
                  RDB = reviewer(repaired B)                    collateral on clean papers

Columns written for EVERY stage (benign / attacked / def_attacked / def_benign):
    <stage>_rating  <stage>_decision  <stage>_weaknesses (JSON list)  <stage>_summary
    <stage>_review  (the full reviewer text, so you can read it)

    runs/<model>/<ablation>/pairs_detail.csv.gz   one row per pair
    runs/<model>/<ablation>/documents.jsonl.gz    repaired documents, for inspection
    runs/<model>/<ablation>/run_meta.json         routing, settings, counts

THE REVIEW CACHE
    R0 and RA do not depend on the ablation: the reviewer is always the untrained base
    model. They are cached in runs/<model>/_review_cache.jsonl (keyed by document hash),
    so the zero-shot run pays for them once and every other ablation reuses them. A
    repaired document that is byte-identical across ablations is reused too.

ROWS
    --max-rows N           N pairs per source (stable order, so every ablation scores the
                           SAME pairs); 0 = all
    --sources a b c        restrict to these input sources
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
from typing import Dict, List, Optional

import pandas as pd

import ft_config as C
import v8_cascade as K
import v8_tools as T
from v8_llm import Backend, Reviewer, Routing, build_routing, make_backend

STAGES = ("benign", "attacked", "def_attacked", "def_benign")


# ======================================================================================
# review cache
# ======================================================================================
class ReviewCache:
    def __init__(self, path: str, reviewer: Reviewer):
        self.path, self.rv = path, reviewer
        self.lock = threading.Lock()
        self.mem: Dict[str, Dict] = {}
        self.hits = self.miss = 0
        if path and os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                try:
                    d = json.loads(line)
                    self.mem[d["key"]] = d["review"]
                except Exception:
                    continue

    def key(self, doc: str) -> str:
        return hashlib.sha1((self.rv.served + "|" + self.rv.source + "|" + doc)
                            .encode("utf-8", "ignore")).hexdigest()

    def get(self, doc: str) -> Dict:
        k = self.key(doc)
        with self.lock:
            if k in self.mem:
                self.hits += 1
                return dict(self.mem[k], calls=0, cached=True)
        r = self.rv.review(doc)
        with self.lock:
            self.miss += 1
            if r.get("rating") is not None:
                self.mem[k] = {x: r[x] for x in ("rating", "decision", "weaknesses",
                                                  "summary", "raw")}
                if self.path:
                    with open(self.path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(T.scrub({"key": k, "review": self.mem[k]}),
                                           ensure_ascii=False) + "\n")
        return dict(r, cached=False)


# ======================================================================================
# one pair
# ======================================================================================
def _rev_cols(stage: str, r: Optional[Dict]) -> Dict:
    r = r or {}
    return {f"{stage}_rating": r.get("rating"), f"{stage}_decision": r.get("decision", ""),
            f"{stage}_weaknesses": json.dumps(r.get("weaknesses") or [], ensure_ascii=False),
            f"{stage}_summary": r.get("summary", ""), f"{stage}_review": r.get("raw", "")}


def _det_cols(p: str, d: Dict) -> Dict:
    return {f"{p}_pred": d.get("pred", 0), f"{p}_score": d.get("score", 0.0),
            f"{p}_verdict": d.get("verdict", ""), f"{p}_confidence": d.get("confidence"),
            f"{p}_span": d.get("span", ""), f"{p}_reason": d.get("reason", ""),
            f"{p}_attack_type": (d.get("leader") or {}).get("attack_type", ""),
            f"{p}_escalated": d.get("escalated", 0), f"{p}_calls": d.get("calls", 0),
            f"{p}_n_items": len(d.get("items") or []),
            f"{p}_n_fabricated": d.get("n_fabricated", 0),
            f"{p}_signals": json.dumps(d.get("signals") or {}),
            f"{p}_items": json.dumps(d.get("items") or [], ensure_ascii=False)[:20000],
            f"{p}_verifier": json.dumps(d.get("verifier") or {}, ensure_ascii=False)[:8000],
            f"{p}_evidence": json.dumps(d.get("evidence") or {}, ensure_ascii=False)[:8000],
            f"{p}_audit": json.dumps(d.get("audit")) if d.get("audit") else "",
            f"{p}_errors": d.get("errors", 0)}


def run_pair(r: pd.Series, llm: Backend, route: Routing, cfg: K.CascadeConfig,
             cache: Optional[ReviewCache], reviewer: Optional[Reviewer],
             do_review: bool) -> Dict:
    t0 = time.time()
    B = str(r["benign_document"] or "")
    A_ = str(r["attacked_document"] or "")
    gold = [tuple(x) for x in json.loads(r.get("gold_offsets") or "[]")]
    row = {k: r.get(k) for k in ("pair_uid", "input_source", "split", "dataset",
                                 "paper_id", "condition", "threat_family", "attack_kind",
                                 "variant_label", "gold_from", "harmful_hist",
                                 "historical_base_rating", "historical_attacked_rating",
                                 "data_origin", "attack_generator", "strategy",
                                 "seen_in_train")}
    docs_out = {"pair_uid": r["pair_uid"]}
    calls = 0

    dB = K.detect(B, llm, route, cfg, reviewer if cfg.audit else None)
    dA = K.detect(A_, llm, route, cfg, reviewer if cfg.audit else None)
    row.update(_det_cols("det_benign", dB))
    row.update(_det_cols("det_attacked", dA))
    calls += dB.get("calls", 0) + dA.get("calls", 0)

    # ---- repair: localize + excise + normalize, flagged documents only ----------------
    def fix(doc: str, det: Dict, tag: str):
        nonlocal calls
        if not det.get("pred"):
            return doc, {"flagged": 0}
        lz = K.localize(doc, det, llm, route, cfg)
        calls += lz["calls"]
        new, info = K.repair(doc, lz["spans"])
        info.update({"flagged": 1, "delete": lz["delete"], "spans": lz["spans"],
                     "n_fabricated": lz["n_fabricated"], "fallback": lz["fallback"]})
        docs_out[f"{tag}_document"] = new
        return new, info

    newA, iA = fix(A_, dA, "def_attacked")
    newB, iB = fix(B, dB, "def_benign")
    prf = T.span_prf(iA.get("spans") or [], gold) if iA.get("flagged") else \
        T.span_prf([], gold)
    row.update({
        "loc_attacked_flagged": iA.get("flagged", 0),
        "loc_attacked_delete": json.dumps(iA.get("delete") or [], ensure_ascii=False)[:20000],
        "loc_attacked_spans": json.dumps(iA.get("spans") or []),
        "loc_attacked_fabricated": iA.get("n_fabricated", 0),
        "loc_attacked_fallback": int(bool(iA.get("fallback"))),
        "loc_attacked_normalized": json.dumps(iA.get("normalized") or {}),
        "loc_benign_flagged": iB.get("flagged", 0),
        "loc_benign_chars_removed": iB.get("chars_removed", 0),
        "loc_benign_delete": json.dumps(iB.get("delete") or [], ensure_ascii=False)[:8000],
        "span_precision": prf["precision"], "span_recall": prf["recall"],
        "span_f1": prf["f1"], "span_iou": prf["iou"], "span_collateral": prf["collateral"],
        "span_residual": prf["residual"], "gold_chars": prf["gold_chars"],
        "removed_chars": prf["pred_chars"]})

    # ---- reviews at every stage --------------------------------------------------------
    if do_review and cache is not None:
        R0 = cache.get(B)
        RA = cache.get(A_)
        RD = cache.get(newA) if iA.get("flagged") else dict(RA, calls=0)
        RDB = cache.get(newB) if iB.get("flagged") else dict(R0, calls=0)
        calls += sum(x.get("calls", 0) for x in (R0, RA, RD, RDB))
        for st, rv in zip(STAGES, (R0, RA, RD, RDB)):
            row.update(_rev_cols(st, rv))
        ref = T.split_items(r.get("reference_weaknesses"))
        W = {st: json.loads(row[f"{st}_weaknesses"]) for st in STAGES}
        for st in STAGES:
            cv = T.coverage(ref, W[st])
            row[f"{st}_wcov_recall"] = cv["recall"]
            row[f"{st}_wcov_precision"] = cv["precision"]
        row["wsim_attacked_benign"] = T.set_similarity(W["attacked"], W["benign"])
        row["wsim_defended_benign"] = T.set_similarity(W["def_attacked"], W["benign"])
        row["n_reference_weaknesses"] = len(ref)
    row["calls_total"] = calls
    row["seconds"] = round(time.time() - t0, 1)
    return {"row": row, "docs": docs_out}


# ======================================================================================
def select(df: pd.DataFrame, max_rows: int, sources: List[str]) -> pd.DataFrame:
    d = df[df["split"] == "test"].copy()
    if sources:
        d = d[d["input_source"].isin(sources)]
    d["_k"] = [hashlib.sha1(str(u).encode()).hexdigest() for u in d["pair_uid"]]
    d = d.sort_values(["input_source", "_k"])
    if max_rows > 0:
        d = d.groupby("input_source", group_keys=False).head(max_rows)
    return d.drop(columns="_k").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="evaluate one v8 ablation (detect + repair)")
    ap.add_argument("--model", required=True, choices=sorted(C.MODELS))
    ap.add_argument("--ablation", required=True, choices=list(C.ABL))
    ap.add_argument("--inputs", default="")
    ap.add_argument("--out-dir", dest="out_dir", default="")
    ap.add_argument("--max-rows", dest="max_rows", type=int, default=0)
    ap.add_argument("--sources", nargs="*", default=[])
    ap.add_argument("--no-review", dest="review", action="store_false",
                    help="detection only: skip the reviewer (task b)")
    ap.add_argument("--audit", action="store_true", help="enable the behavioural AUDITOR")
    ap.add_argument("--min-conf", dest="min_conf", type=float, default=0.5)
    ap.add_argument("--max-windows", dest="max_windows", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--fresh", dest="resume", action="store_false")
    ap.add_argument("--run-tag", dest="run_tag", default="",
                    help="write to runs/<model>/<ablation>__<tag>/ (a pilot that never touches "
                         "the main run's folder or review cache)")
    a = ap.parse_args()

    spec = C.MODELS[a.model]
    mk = a.model.replace("/", "_")
    inp = a.inputs or C.wpath(C.D_INPUTS, mk, "pairs_all.csv.gz")
    if not os.path.exists(inp):
        raise SystemExit(f"[FATAL] {inp} not found -- run v8_inputs.py first")
    df = select(pd.read_csv(inp, low_memory=False), a.max_rows, a.sources)
    sfx = f"__{a.run_tag}" if a.run_tag else ""
    out_dir = a.out_dir or C.wpath(C.D_RUNS, mk, a.ablation + sfx)
    os.makedirs(out_dir, exist_ok=True)

    llm = make_backend(a.model)
    base = getattr(llm, "base", "") or a.model
    route = build_routing(a.model, a.ablation, base)
    print(f"[eval] {a.model}  ablation={a.ablation}  pairs={len(df)}  "
          f"sources={dict(df['input_source'].value_counts())}")
    print(route.describe())
    if route.adapters() and hasattr(llm, "require"):
        llm.require(route.adapters())          # an adapter not served = baseline mislabelled
        print(f"[serve] all {len(route.adapters())} adapter(s) confirmed on the server")

    cfg = K.CascadeConfig(system_role=spec.system_role, min_conf=a.min_conf,
                          max_windows=a.max_windows, audit=a.audit)
    reviewer = Reviewer(llm, base, system_role=spec.system_role) if (a.review or a.audit) else None
    cache = ReviewCache(C.wpath(C.D_RUNS, mk, f"_review_cache{sfx}.jsonl"), reviewer) \
        if reviewer else None
    if reviewer:
        print(f"[review] reviewer prompt source: {reviewer.source} "
              f"({'the attack benchmark harness' if reviewer.source == 'harness' else 'v8 fallback -- harness not importable'})")

    part = os.path.join(out_dir, "pairs_detail.partial.jsonl")
    dpath = os.path.join(out_dir, "documents.partial.jsonl")
    done = set()
    if a.resume and os.path.exists(part):
        for line in open(part, encoding="utf-8"):
            try:
                done.add(json.loads(line)["pair_uid"])
            except Exception:
                pass
        print(f"[resume] {len(done)} pairs already done")
    elif os.path.exists(part):
        os.remove(part)
    todo = [r for _, r in df.iterrows() if r["pair_uid"] not in done]
    lock = threading.Lock()
    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        futs = {ex.submit(run_pair, r, llm, route, cfg, cache, reviewer, a.review): r["pair_uid"]
                for r in todo}
        for fu in as_completed(futs):
            try:
                res = fu.result()
            except Exception as e:
                print(f"[error] {futs[fu]}: {type(e).__name__}: {e}")
                continue
            with lock:
                with open(part, "a", encoding="utf-8") as f:
                    f.write(json.dumps(T.scrub(res["row"]), ensure_ascii=False, default=str) + "\n")
                if len(res["docs"]) > 1:
                    with open(dpath, "a", encoding="utf-8") as f:
                        f.write(json.dumps(T.scrub(res["docs"]), ensure_ascii=False) + "\n")
                n_done += 1
                if n_done % 10 == 0 or n_done == len(todo):
                    el = (time.time() - t0) / 60
                    print(f"[eval] {n_done}/{len(todo)}  {el:.1f} min  "
                          f"llm calls {llm.n_calls}", flush=True)

    rows = [json.loads(l) for l in open(part, encoding="utf-8") if l.strip()] \
        if os.path.exists(part) else []
    keep = set(df["pair_uid"])
    rows = [r for r in rows if r["pair_uid"] in keep]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "pairs_detail.csv.gz"), index=False,
                              compression="gzip")
    if os.path.exists(dpath):
        with open(dpath, "rb") as fi, gzip.open(os.path.join(out_dir, "documents.jsonl.gz"),
                                                "wb") as fo:
            fo.write(fi.read())
    meta = {"model": a.model, "ablation": a.ablation, "n_pairs": len(rows),
            "routing": route.table or {"*": route.base}, "min_conf": a.min_conf,
            "audit": a.audit, "review": a.review, "max_rows": a.max_rows,
            "sources": a.sources, "run_tag": a.run_tag, "reviewer_source": reviewer.source if reviewer else "",
            "review_cache": {"hits": cache.hits, "miss": cache.miss} if cache else None,
            "llm_calls": llm.n_calls, "minutes": (time.time() - t0) / 60}
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n[saved] {out_dir}/pairs_detail.csv.gz  ({len(rows)} pairs)")
    if cache:
        print(f"[review] cache: {cache.hits} reused, {cache.miss} new reviews")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())