#!/usr/bin/env python3
"""
run_spotlight.py — the Spotlighting defence baseline, end to end.

    python run_spotlight.py inputs  --model gemma3-27b-it            # CPU: freeze the docs
    python run_spotlight.py inspect --model gemma3-27b-it --tokenizer $MODEL_DIR
    python run_spotlight.py review  --model gemma3-27b-it --vllm-url http://127.0.0.1:P/v1 \\
                                    --served $MODEL_DIR --variants plain instruct delimit datamark
    python run_spotlight.py report  [--models ...]                   # tables, CPU

STAGES
  inputs   (a) ours via infer_ladder.load_eval_set, (b) ARGAR/BTR via ab_common.load_pairs,
           frozen into inputs/<model>/docs[__TAG].jsonl.gz + manifest (sha256-checked).
  inspect  counts, an example of every prompt, and (with --tokenizer) the prompt length in
           the reviewer's own tokens vs --max-len — before any GPU time is spent.
  review   one pass per variant over every DISTINCT document text (resumable cache):
           rating (Table 2) + self-report flag (Table 1 adaptation).
           runs/<TAG|default>/<model>/<variant>/
               docs.csv            one row per document (eval order), every field
               meta.json           prompts' provenance, config, counts, timing
               argar_btr/scores.csv + meta.json        ab_common schema -> ab_aggregate.py
           runs/<TAG|default>/_ladder/<model>/spotlight_<variant>/
               stage1_detail.csv + summary.json        (a) only, infer_ladder schema ->
                                   your category_defense_metrics.py / ft_table.py
  report   spot_report.py: Table 1 rows, Table 2 rows, the end-to-end table, per family,
           per source, per reviewer.

VARIANTS  plain instruct delimit datamark datamark_rand encode  (spotlighting.VARIANTS)
          a marker override is written variant:marker, e.g.  datamark:caret
          `plain` is the undefended reviewer: its ratings ARE R_0 / R_A for every row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Dict, List

import pandas as pd

import spot_config as C
import spot_data as D
import spot_review as RV
import spotlighting as SP

DEFAULT_VARIANTS = ["plain", "instruct", "delimit", "datamark"]


# ======================================================================================
# helpers
# ======================================================================================
def stable_split(paper_id: str, val_frac: float = 0.20) -> str:
    """ab_common.stable_split / common.stable_split, same salt."""
    h = int(hashlib.sha1(f"defbase::{paper_id}".encode()).hexdigest()[:8], 16)
    return "val" if (h % 10_000) / 10_000.0 < val_frac else "test"


def ladder_metrics(rows: List[Dict], subset: str = "all") -> Dict:
    """infer_ladder.metrics, copied, so summary.json has the schema ft_table.py reads."""
    R_ = rows if subset != "harmful" else [r for r in rows if r["label"] == 0 or r.get("harmful")]
    tp = sum(1 for r in R_ if r["label"] == 1 and r["pred"] == 1)
    fp = sum(1 for r in R_ if r["label"] == 0 and r["pred"] == 1)
    tn = sum(1 for r in R_ if r["label"] == 0 and r["pred"] == 0)
    fn = sum(1 for r in R_ if r["label"] == 1 and r["pred"] == 0)
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if n else None)
    return {"subset": subset, "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / n if n else None, "precision": prec, "recall": rec,
            "f1": f1, "fpr": fp / (fp + tn) if fp + tn else None,
            "escalated_pct": 100.0 * sum(r["escalated"] for r in R_) / n if n else None,
            "mean_llm_calls": sum(r["n_llm_calls"] for r in R_) / n if n else None}


def common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, choices=sorted(C.MODELS))
    p.add_argument("--run-tag", dest="run_tag", default=os.environ.get("RUN_TAG", ""))


# ======================================================================================
# inputs
# ======================================================================================
def cmd_inputs(a) -> int:
    D.build_inputs(a.model, sources=a.sources, run_tag=a.run_tag, max_rows=a.max_rows,
                   force=a.force, benign_csv=a.benign_csv, attacked_csv=a.attacked_csv,
                   id_column=a.id_column, seed=a.seed, ours_docs=a.ours_docs,
                   ours_index=a.ours_index, argar_pairs_csv=a.argar_pairs_csv,
                   subset=a.subset)
    return 0


# ======================================================================================
# inspect
# ======================================================================================
def _load_tokenizer(path: str):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        print(f"[tok] {path}: {type(tok).__name__}")
        return tok
    except Exception as e:
        print(f"[tok] could not load a tokenizer from {path}: {type(e).__name__}: {e}")
        return None


def _n_tokens(tok, msgs: List[Dict]) -> int:
    try:
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        return len(ids["input_ids"] if isinstance(ids, dict) else ids)
    except Exception:
        return len(tok("\n".join(m["content"] for m in msgs), add_special_tokens=True)
                   ["input_ids"]) + 16


def cmd_inspect(a) -> int:
    docs = D.load_docs(a.model, a.run_tag)
    man = json.load(open(C.manifest_path(a.model, a.run_tag)))
    print(C.describe())
    print(f"\n[inspect] {a.model}: {len(docs)} documents, {docs['doc_sha'].nunique()} "
          f"distinct texts, {man.get('n_pairs')} pairs")
    att = docs[docs.label == 1]
    print(att.groupby(["source", "family"]).size().rename("attacked").to_string())
    for k, v in (man.get("parts") or {}).items():
        print(f"[inspect] part {k}: " + json.dumps({x: v[x] for x in v if x not in (
            "per_source",)}, default=str)[:600])
    L = docs["document"].str.len()
    print(f"[inspect] manuscript length: median {int(L.median())} chars, "
          f"{int((L > C.REVIEW.max_chars).mean() * 100)}% longer than the "
          f"{C.REVIEW.max_chars}-char view")
    base, src = C.import_reviewer_prompt(a.prompt_file, a.allow_builtin_prompt)
    print(f"[inspect] reviewer prompt: {src}  ({len(base)} chars)")
    spec = C.MODELS[a.model]
    tok = _load_tokenizer(a.tokenizer)
    cfg = SP.SpotConfig(marker_type=a.marker, marker_min_len=a.marker_min_len,
                        marker_max_len=a.marker_max_len, seed=a.seed)
    sample = att.sort_values("document", key=lambda s: s.str.len(), ascending=False)
    sample = pd.concat([sample.head(a.n_sample), att.sample(min(len(att), a.n_sample),
                                                            random_state=0)])
    budget = a.max_len - C.REVIEW.max_tokens - (C.REVIEW.detect_max_tokens + 64)
    bad = 0
    for vs in a.variants:
        v, m, vdir = RV.parse_variant(vs)
        rv = RV.SpotReviewer(None, model_key=a.model, variant_spec=vs, base_system=base,
                             base_src=src, extract_json=C.local_extract_json,
                             extract_src="-", system_role=spec.system_role, spot_cfg=cfg,
                             tokenizer=tok if v == "datamark_rand" else None, cache=False)
        lens, toks = [], []
        for _, r in sample.iterrows():
            msgs, s = rv.build(RV.view_of(r["document"], C.REVIEW.max_chars))
            lens.append(sum(len(x["content"]) for x in msgs))
            if tok is not None:
                toks.append(_n_tokens(tok, msgs))
        msg0, s0 = rv.build(RV.view_of(att.iloc[0]["document"], C.REVIEW.max_chars))
        print("\n" + "-" * 92)
        print(f"VARIANT {vdir}: {SP.VARIANT_LABEL[v]}   marker={s0.marker!r} "
              f"n_markers={s0.n_markers}")
        print(f"  prompt chars: max {max(lens)}  mean {sum(lens) / len(lens):.0f}")
        if toks:
            over = sum(1 for t in toks if t > budget)
            bad += over
            print(f"  prompt TOKENS ({spec.key} tokenizer): max {max(toks)}  mean "
                  f"{sum(toks) / len(toks):.0f}  -> {over}/{len(toks)} sampled prompts exceed "
                  f"the {budget}-token budget at MAX_LEN={a.max_len}")
        for x in msg0:
            c = x["content"]
            print(f"  [{x['role']}] {len(c)} chars")
            print("    " + c[:900].replace("\n", "\n    "))
            if len(c) > 1500:
                print("    [...]\n    " + c[-500:].replace("\n", "\n    "))
    n = docs["doc_sha"].nunique()
    print("\n" + "=" * 92)
    print(f"[cost] {n} distinct texts x {len(a.variants)} variants = {n * len(a.variants)} "
          f"reviews (+ the same number of short self-report turns)")
    if bad:
        print(f"[cost] WARNING: {bad} sampled prompts would overflow MAX_LEN={a.max_len}. "
              f"They will be retried with a shorter view (recorded as `shrinks`); raise "
              f"MAX_LEN or use MARKER=caret to avoid it.")
    return 0


# ======================================================================================
# review
# ======================================================================================
EXPORT_COLS = ["eval_index", "doc_id", "pair_uid", "role", "label", "source", "family",
               "condition", "method", "paper_id", "harmful", "harmful_src", "doc_sha"]
RESULT_COLS = ["rating", "parse_ok", "decision", "pred", "score", "det_parse_ok",
               "det_conf", "evidence", "weaknesses", "leaked_markers", "n_markers",
               "n_sanitized", "view_chars", "shrinks", "prompt_chars", "prompt_tokens",
               "completion_tokens", "calls", "cached", "error"]


def ladder_dir(model_key: str, vdir: str, run_tag: str = "") -> str:
    """runs/<TAG>/_ladder/<model>/spotlight_<variant>/ — infer_ladder's runs/<model>/
    <ablation>/ layout, so ft_table.py --runs-dir runs/<TAG>/_ladder and
    category_defense_metrics.py read these rows unchanged, named spotlight_<variant>."""
    return C.wpath(C.D_RUNS, run_tag or "default", "_ladder", C.mkey(model_key),
                   f"spotlight_{vdir}")


def write_variant(docs: pd.DataFrame, res: Dict[str, Dict], out_dir: str, meta: Dict,
                  model_key: str, vdir: str, run_tag: str = "") -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    d = docs[EXPORT_COLS + ["ref_weaknesses"]].copy()
    for c in RESULT_COLS:
        d[c] = [res.get(k, {}).get(c) for k in d["doc_sha"]]
    d["pred"] = d["pred"].fillna(0).astype(int)
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    d["rating"] = pd.to_numeric(d["rating"], errors="coerce")
    d.to_csv(os.path.join(out_dir, "docs.csv"), index=False)

    # ---- ours -> infer_ladder stage1_detail.csv (+ summary.json) -------------------
    o = d[d["source"] == "ours"]
    if len(o):
        od = ladder_dir(model_key, vdir, run_tag)
        os.makedirs(od, exist_ok=True)
        st = pd.DataFrame({
            "doc_id": o["doc_id"], "paper_id": o["paper_id"], "label": o["label"],
            "condition": o["condition"], "harmful": o["harmful"],
            "n_llm_calls": o["calls"].fillna(0).astype(int), "escalated": 1,
            "pred": o["pred"], "attack_type": "", "severity": 0,
            "span": o["evidence"].fillna(""), "agent_reason": "spotlighting self-report",
            "verifier_said": "", "n_findings": 0, "screen_score": o["score"],
            "rating": o["rating"], "parse_ok": o["parse_ok"]})
        st.to_csv(os.path.join(od, "stage1_detail.csv"), index=False)
        recs = st.to_dict("records")
        summ = {"model": model_key, "ablation": f"spotlight_{vdir}",
                "n_documents": len(recs), "all": ladder_metrics(recs, "all"),
                "harmful": ladder_metrics(recs, "harmful"),
                "note": "pred = Spotlighting self-report (Table-1 adaptation)"}
        with open(os.path.join(od, "summary.json"), "w") as f:
            json.dump(summ, f, indent=2)

    # ---- argar/btr -> ab_common scores.csv -------------------------------------------
    b = d[d["source"].isin(["argar", "btr"])].copy()
    if len(b):
        bd = os.path.join(out_dir, "argar_btr")
        os.makedirs(bd, exist_ok=True)
        fam = b[b["label"] == 1].set_index("pair_uid")["family"].to_dict()
        b["family"] = b["pair_uid"].map(fam).fillna(b["family"])
        b["split"] = b["paper_id"].map(stable_split)
        b["n_chars"] = docs.loc[b.index, "document"].str.len()
        cols = ["doc_id", "pair_uid", "paper_id", "source", "family", "method",
                "condition", "split", "label", "harmful", "score", "pred", "calls",
                "n_chars", "rating"]
        b[cols].to_csv(os.path.join(bd, "scores.csv"), index=False)
        with open(os.path.join(bd, "meta.json"), "w") as f:
            import spot_report as SR
            json.dump({"label": SR.paste_label(vdir) + (" (datamark)" if vdir == SR.MAIN_VARIANT
                                                        else "") + ", self-report",
                       "variant": vdir, "baseline": "Spotlighting",
                       "thresholds": "native self-report flag (no tuning)",
                       "threshold_info": {"mode": "native"}}, f, indent=2)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return d


def quick_summary(d: pd.DataFrame, vdir: str) -> None:
    P = d[d["pair_uid"] != ""]
    c = P[P.label == 0].set_index("pair_uid")["rating"]
    t = P[P.label == 1].set_index("pair_uid")["rating"]
    j = pd.concat([c.rename("c"), t.rename("a")], axis=1).dropna()
    m = d["pred"]
    print(f"[{vdir}] parse {100 * d['parse_ok'].fillna(0).mean():.1f}%  "
          f"mean rating clean {c.mean():.3f}  attacked {t.mean():.3f}  "
          f"paired dR (own clean ref) {(j['a'] - j['c']).mean():+.3f}  "
          f"self-report: flagged {100 * m[d.label == 1].mean():.1f}% attacked / "
          f"{100 * m[d.label == 0].mean():.1f}% clean  "
          f"shrunk {int((d['shrinks'].fillna(0) > 0).sum())}  errors "
          f"{int(d['error'].fillna('').astype(bool).sum())}", flush=True)


def cmd_review(a) -> int:
    docs = D.load_docs(a.model, a.run_tag)
    spec = C.MODELS[a.model]
    base, src = C.import_reviewer_prompt(a.prompt_file, a.allow_builtin_prompt)
    ex, exsrc = C.import_extract_json()
    if not a.vllm_url:
        raise SystemExit("[FATAL] --vllm-url (or VLLM_URL) is required")
    served = a.served or os.environ.get("VLLM_MODEL", "") or spec.path
    cl = RV.ChatClient(a.vllm_url, served, chat_kwargs=spec.chat_kwargs,
                       timeout=C.REVIEW.timeout)
    RV.require_served(cl, served)
    rcfg = C.ReviewConfig(max_chars=a.max_chars, max_tokens=a.max_tokens,
                          temperature=a.temperature, detect=a.detect,
                          workers=a.workers)
    cfg = SP.SpotConfig(marker_type=a.marker, marker_min_len=a.marker_min_len,
                        marker_max_len=a.marker_max_len, p=a.p, min_gap=a.min_gap,
                        sanitize=not a.no_sanitize, seed=a.seed)
    tok = _load_tokenizer(a.tokenizer) if any(v.startswith("datamark_rand")
                                              for v in a.variants) else None
    uniq = docs.drop_duplicates("doc_sha")[["doc_sha", "document"]]
    items = list(zip(uniq["doc_sha"], uniq["document"]))
    variants = sorted(a.variants, key=lambda v: (v != "plain"))      # plain first
    for vs in variants:
        t0 = time.time()
        rv = RV.SpotReviewer(cl, model_key=a.model, variant_spec=vs, base_system=base,
                             base_src=src, extract_json=ex, extract_src=exsrc,
                             system_role=spec.system_role, spot_cfg=cfg, rcfg=rcfg,
                             tokenizer=tok)
        print("\n" + "#" * 92)
        print(rv.describe())
        if rv.variant == "encode":
            print("[review] NOTE (paper Sec. 5.2/5.3): base64 encoding should only be used "
                  "with the highest-capacity models; expect parse failures and rating "
                  "collapse on 8B-27B reviewers. Read `parse` before reading Restore.")
        res = rv.review_many(items, workers=rcfg.workers, label=f"{a.model}:{rv.vdir}")
        meta = {"model": a.model, "served": served, "variant": rv.variant,
                "variant_dir": rv.vdir, "label": SP.VARIANT_LABEL[rv.variant],
                "base_prompt": src, "parser": exsrc, "system_role": spec.system_role,
                "chat_kwargs": spec.chat_kwargs, "review": rcfg.__dict__,
                "spot": rv.spot_cfg.__dict__ if rv.variant in SP.MARKED_VARIANTS else None,
                "cfg_hash": rv.cfg_hash, "cache": rv.cache_path,
                "detect_question": RV.DETECT_Q if rcfg.detect else None,
                "inputs": C.inputs_path(a.model, a.run_tag),
                "inputs_sha256": json.load(open(C.manifest_path(a.model, a.run_tag)))
                .get("sha256"), "n_documents": len(docs), "n_distinct": len(items),
                "minutes": (time.time() - t0) / 60, "llm_calls_this_run": cl.n_calls,
                "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        out_dir = C.run_dir(a.model, rv.vdir, a.run_tag)
        d = write_variant(docs, res, out_dir, meta, a.model, rv.vdir, a.run_tag)
        quick_summary(d, rv.vdir)
        if a.ab_export_dir:
            dst = os.path.join(a.ab_export_dir, f"Spotlighting_{a.model}", rv.vdir)
            os.makedirs(dst, exist_ok=True)
            for fn in ("scores.csv", "meta.json"):
                p = os.path.join(out_dir, "argar_btr", fn)
                if os.path.exists(p):
                    with open(p, "rb") as fi, open(os.path.join(dst, fn), "wb") as fo:
                        fo.write(fi.read())
            print(f"[export] ab_aggregate copy -> {dst}")
        print(f"[saved] {out_dir}/docs.csv  ({(time.time() - t0) / 60:.1f} min)")
    return 0


# ======================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inputs", help="freeze the documents (CPU)")
    common_args(p)
    p.add_argument("--sources", nargs="+", default=["ours", "argar", "btr"],
                   choices=["ours", "argar", "btr"])
    p.add_argument("--max-rows", dest="max_rows", type=int, default=0,
                   help="pairs for ours / pairs per ARGAR-BTR source (0 = all)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--benign-csv", dest="benign_csv", default=C.TEST_CSV)
    p.add_argument("--attacked-csv", dest="attacked_csv", default="")
    p.add_argument("--id-column", dest="id_column", default="id")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subset", default="test", choices=["test", "full"])
    p.add_argument("--ours-docs", dest="ours_docs", default="",
                   help="use an already-materialised copy of the defense1 corpus "
                        "(jsonl[.gz] of DocRecords) instead of calling infer_ladder")
    p.add_argument("--ours-index", dest="ours_index", default="",
                   help="with --ours-docs = v7ab eval_records.jsonl: its eval_index.csv; keeps "
                        "the insertion rows = the documents SafeReview/System Defense used")
    p.add_argument("--argar-pairs-csv", dest="argar_pairs_csv", default="")

    def spot_args(q):
        q.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
        q.add_argument("--marker", default=os.environ.get("MARKER", "unicode"),
                       choices=list(SP.MARKER_TYPES))
        q.add_argument("--marker-min-len", dest="marker_min_len", type=int, default=1)
        q.add_argument("--marker-max-len", dest="marker_max_len", type=int, default=1)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--prompt-file", dest="prompt_file",
                       default=os.environ.get("REVIEWER_PROMPT_FILE", ""))
        q.add_argument("--allow-builtin-prompt", dest="allow_builtin_prompt",
                       action="store_true")
        q.add_argument("--tokenizer", default="",
                       help="the reviewer's HF tokenizer dir (inspect: token counts; "
                            "datamark_rand: token boundaries)")

    p = sub.add_parser("inspect", help="counts, prompts and token budget (CPU)")
    common_args(p)
    spot_args(p)
    p.add_argument("--max-len", dest="max_len", type=int, default=32768)
    p.add_argument("--n-sample", dest="n_sample", type=int, default=12)

    p = sub.add_parser("review", help="review every document under every variant (GPU)")
    common_args(p)
    spot_args(p)
    p.add_argument("--vllm-url", dest="vllm_url", default=os.environ.get("VLLM_URL", ""))
    p.add_argument("--served", default=os.environ.get("VLLM_MODEL", ""))
    p.add_argument("--workers", type=int, default=C.REVIEW.workers)
    p.add_argument("--max-chars", dest="max_chars", type=int, default=C.REVIEW.max_chars)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=C.REVIEW.max_tokens)
    p.add_argument("--temperature", type=float, default=C.REVIEW.temperature)
    p.add_argument("--no-detect", dest="detect", action="store_false")
    p.add_argument("--p", type=float, default=0.5, help="randomlyMarkData p")
    p.add_argument("--min-gap", dest="min_gap", type=int, default=1)
    p.add_argument("--no-sanitize", dest="no_sanitize", action="store_true")
    p.add_argument("--ab-export-dir", dest="ab_export_dir", default="",
                   help="also copy argar_btr/scores.csv there (ab_aggregate layout)")

    p = sub.add_parser("report", help="tables (CPU)")
    p.add_argument("--models", nargs="+", default=list(C.MODELS))
    p.add_argument("--run-tag", dest="run_tag", default=os.environ.get("RUN_TAG", ""))
    p.add_argument("--out-dir", dest="out_dir", default="")

    a = ap.parse_args()
    if a.cmd == "inputs":
        return cmd_inputs(a)
    if a.cmd == "inspect":
        return cmd_inspect(a)
    if a.cmd == "review":
        return cmd_review(a)
    import spot_report
    return spot_report.run(a.models, a.run_tag, a.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
