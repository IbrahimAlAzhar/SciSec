#!/usr/bin/env python3
"""
ab_aggregate.py — the ARGAR/BTR baseline table, in the same layout as the insertion
Panel B (one block per threat family), plus the per-variant appendix table.

    python ab_aggregate.py                       # output/default -> final/default
    python ab_aggregate.py --run-tag pilot5
    python ab_aggregate.py --no-agents           # baselines only

ROWS
    every output/<run>/<baseline>/<variant>/scores.csv written by the ab_*.py scripts
    (and by run_existing.py, if those scripts saved per-document scores), and -- unless
    --no-agents -- the LLM-agent rows already computed on the SAME test pairs:
        v8 agents   $V8_DIR/runs/gemma3-27b-it/<ablation>/pairs_detail.csv.gz
        v7 GRPO     $V7AB_DIR/runs/gemma3-27b-it/<row>/stage1_detail.csv  (via v7ab_report)
    restricted to ARGAR and BTR documents. v8 rows are further restricted to the pair_uids
    the baselines scored, so a --max-rows pilot compares like with like.

METRICS
    identical to v8_report.py: per family, attacked documents and their own benign twins;
    HarmRec on attacks with historical ASR > 0; R@FPR from the score with a strict '>' at
    the k-th highest benign score; AUROC with average ranks for ties.

OUTPUT  final/<run>/
    table_argar_btr.tex      Panel B layout: Repackaging (ARGAR), Perturbation (BTR), ALL
    table_by_variant.tex     appendix: Recall / FPR / AUROC per BTR method and ARGAR
    table_argar_btr.md, metrics_long.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

import ab_common as A

ORDER = [("D1_trivial", "flag_nothing"), ("D1_trivial", "flag_everything"),
         ("D3_ppl_regex", "regex_only"), ("D3_ppl_regex", "ppl_only"),
         ("D3_ppl_regex", "combined"), ("D6_protectai_deberta", "frozen"),
         ("PIGuard", "frozen"), ("D7_llamafirewall_ac", "ac_only_replay"),
         ("PaperGuard", "released_registry_gemma")]
APPENDIX_ONLY = {"card_default_0.5"}

V8_LABEL = {"zeroshot": "ZS LLM Agents v8 (Loc./Evid./Verif./Leader)", "sft": "v8 SFT",
            "dpo_only": "v8 DPO only", "sft_dpo": "v8 SFT+DPO",
            "sft_dpo_merged": "v8 SFT+DPO merged", "grpo_prov": "v8 GRPO Prov Reward"}
V7_LABEL = {"A_zeroshot_v5": "ZS LLM Agents v5 (naive)", "B_sft": "SFT, single adapter",
            "C_sft_dpo": "SFT then DPO, single adapter",
            "G_grpo_prov": "GRPO Prov Reward"}


def load_baselines(run: str) -> List[Dict]:
    rows = []
    for p in sorted(glob.glob(os.path.join(A.OUT_ROOT, run, "*", "*", "scores.csv"))):
        vdir = os.path.dirname(p)
        variant, baseline = os.path.basename(vdir), os.path.basename(os.path.dirname(vdir))
        meta = {}
        if os.path.exists(os.path.join(vdir, "meta.json")):
            meta = json.load(open(os.path.join(vdir, "meta.json")))
        d = pd.read_csv(p, low_memory=False)
        mode = str((meta.get("threshold_info") or {}).get("mode", meta.get("thresholds", "")))
        mark = r"$^\dagger$" if "own" in mode else ""
        rows.append({"key": (baseline, variant), "label": meta.get("label", f"{baseline} {variant}")
                     + mark, "d": d, "kind": "baseline", "meta": meta})
    rank = {k: i for i, k in enumerate(ORDER)}
    rows.sort(key=lambda r: (rank.get(r["key"], 999), r["label"]))
    return rows


def load_v8(sources, pair_uids: set) -> List[Dict]:
    out = []
    root = os.path.join(A.V8_DIR, "runs", A.MODEL_KEY)
    for ab, label in V8_LABEL.items():
        p = os.path.join(root, ab, "pairs_detail.csv.gz")
        if not os.path.exists(p):
            continue
        det = pd.read_csv(p, low_memory=False)
        det = det[det["input_source"].isin(sources)]
        if pair_uids:
            det = det[det["pair_uid"].astype(str).isin(pair_uids)]
        if det.empty:
            continue
        rows = []
        for _, r in det.iterrows():
            hh = A._f(r.get("harmful_hist"))
            dr = (A._f(r.get("attacked_rating")) or np.nan) - (A._f(r.get("benign_rating")) or np.nan)
            harmful = int(hh) if hh is not None else (int(dr >= 1.0) if dr == dr else 0)
            for lab, pre in ((0, "det_benign"), (1, "det_attacked")):
                rows.append({"pair_uid": r["pair_uid"], "source": r["input_source"],
                             "family": r["threat_family"],
                             "method": str(r.get("variant_label") or r.get("condition")),
                             "label": lab, "pred": int(r.get(f"{pre}_pred") or 0),
                             "score": A._f(r.get(f"{pre}_score")),
                             "calls": A._f(r.get(f"{pre}_calls")) or 0.0,
                             "harmful": harmful if lab else 0})
        out.append({"key": ("v8", ab), "label": label, "d": pd.DataFrame(rows),
                    "kind": "agents", "meta": {"from": p}})
    return out


def load_v7(sources) -> List[Dict]:
    if not os.path.isdir(A.V7AB_DIR):
        return []
    sys.path.insert(0, A.V7AB_DIR)
    try:
        import v7ab_report as R
        runs = R.load([A.MODEL_KEY], "")
    except Exception as e:
        print(f"[agg] v7ab rows unavailable: {type(e).__name__}: {e}")
        return []
    out = []
    for r in runs:
        d = r["d"]
        d = d[d["input_source"].isin(sources)].copy()
        if d.empty:
            continue
        fam = d["threat_family"] if "threat_family" in d.columns else \
            d["input_source"].map(A.FAMILY_OF_SOURCE)
        dd = pd.DataFrame({"source": d["input_source"].values, "family": fam.values,
                           "method": (d["variant_label"] if "variant_label" in d.columns
                                      else d.get("condition", d["input_source"])).astype(str).values,
                           "label": d["label"].astype(int).values,
                           "pred": d["pred"].astype(int).values,
                           "score": d["score"].astype(float).values,
                           "calls": d["n_llm_calls"].fillna(0).values if "n_llm_calls" in d.columns else 0.0,
                           "harmful": d["harmful"].fillna(0).astype(int).values})
        base = r["row"].split("__")[0]
        label = V7_LABEL.get(base, r["label"]) + " (v7)"
        out.append({"key": ("v7", r["row"]), "label": label, "d": dd, "kind": "agents",
                    "meta": {"from": "v7ab_report"}})
    return out


def f3(x) -> str:
    return "---" if (x is None or x != x) else f"{x:.3f}"


def esc(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("$\\_dagger", "$^\\dagger")


COLS = ["Npos", "Rec", "Prec", "F1", "FPR", "HarmRec", "R@FPR5", "R@FPR10", "AUROC"]
HDR = (r"\textbf{Method / Baseline} & \textbf{$N_{\mathrm{pos}}$} & \textbf{Recall} & "
       r"\textbf{Prec} & \textbf{F1} & \textbf{FPR} & \textbf{HarmRec} & \textbf{R@FPR5} & "
       r"\textbf{R@FPR10} & \textbf{AUROC} \\")


def tex_row(label: str, m: Dict) -> str:
    cells = [str(int(m.get("Npos", 0)))] + [f3(m.get(c)) for c in COLS[1:]]
    lab = label.replace("&", r"\&").replace("_", r"\_").replace(r"$^\dagger$".replace("_", r"\_"),
                                                                  r"$^\dagger$")
    return f"{lab}\n& " + " & ".join(cells) + r" \\"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-tag", default=os.environ.get("RUN_TAG", ""))
    ap.add_argument("--sources", nargs="+", default=list(A.SOURCES_DEFAULT))
    ap.add_argument("--no-agents", dest="agents", action="store_false")
    a = ap.parse_args()
    run = a.run_tag or "default"

    rows = load_baselines(run)
    if not rows:
        print(f"[agg] nothing under {os.path.join(A.OUT_ROOT, run)}")
        return 1
    uids = set()
    for r in rows:
        if "pair_uid" in r["d"].columns:
            uids |= set(r["d"]["pair_uid"].astype(str))
    if a.agents:
        rows += load_v8(a.sources, uids) + load_v7(a.sources)

    fams = [f for f in ("repackaging", "perturbation") if any(
        (r["d"]["family"] == f).any() for r in rows)]
    long = []
    for r in rows:
        d = r["d"]
        d = d[d["source"].isin(a.sources)]
        blocks = {f: d[d["family"] == f] for f in fams}
        blocks["ALL"] = d
        for b, g in blocks.items():
            m = A.det_metrics(g)
            long.append({"row": r["label"], "kind": r["kind"], "key": "/".join(r["key"]),
                         "block": b, **m})
        for (src, meth), g in d.groupby(["source", "method"]):
            m = A.det_metrics(g)
            long.append({"row": r["label"], "kind": r["kind"], "key": "/".join(r["key"]),
                         "block": f"method:{src}:{meth}", **m})
    L = pd.DataFrame(long)
    od = os.path.join(os.environ.get("AB_FINAL", os.path.join(A.HERE, "final")), run)
    os.makedirs(od, exist_ok=True)
    L.to_csv(os.path.join(od, "metrics_long.csv"), index=False)

    # ---- main table: Panel B layout ---------------------------------------------------
    title = {"repackaging": "Multi-round repackaging (ARGAR)",
             "perturbation": "Character/word perturbations (BTR)",
             "ALL": "ARGAR + BTR combined"}
    lines = [r"\medskip", r"\textbf{Panel B: Detection baselines on ARGAR and BTR (gemma-3-27B "
             r"reviewer, v8 test half)}\\[3pt]", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{lrrrrrrrrr}", r"\toprule", HDR]
    md = []
    for b in fams + ["ALL"]:
        lines += [r"\midrule", rf"\multicolumn{{10}}{{l}}{{\textit{{{title[b]}}}}} \\", r"\midrule"]
        md += [f"\n### {title[b]}\n", "| Method | " + " | ".join(COLS) + " |",
               "|---" * (len(COLS) + 1) + "|"]
        prev_kind = None
        for r in rows:
            if r["key"][1] in APPENDIX_ONLY:
                continue
            sub = L[(L["row"] == r["label"]) & (L["block"] == b)]
            if sub.empty or not sub.iloc[0]["Npos"]:
                continue
            if prev_kind == "baseline" and r["kind"] == "agents":
                lines.append(r"\addlinespace")
            prev_kind = r["kind"]
            m = sub.iloc[0].to_dict()
            lines.append(tex_row(r["label"], m))
            md.append(f"| {r['label']} | " + " | ".join(
                [str(int(m['Npos']))] + [f3(m[c]) for c in COLS[1:]]) + " |")
    lines += [r"\bottomrule", r"\end{tabular}%", "}"]
    notes = sorted({r["label"] for r in rows if r"\dagger" in r["label"]})
    lines.append(r"\par\smallskip\footnotesize Thresholded rows use the operating point frozen "
                 r"from each detector's insertion run (FPR $\le$ 5\% on its validation papers); "
                 r"R@FPR and AUROC are threshold-free. D7 and PaperGuard use their native "
                 r"decision rules with Gemma-3-27B as the auditor/verifier."
                 + (r" $^\dagger$\,threshold re-tuned on 20\% of the ARGAR/BTR papers "
                    r"(insertion predictions not found); evaluated on the remaining 80\%."
                    if notes else ""))
    open(os.path.join(od, "table_argar_btr.tex"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(od, "table_argar_btr.md"), "w").write("\n".join(md) + "\n")

    # ---- appendix: per variant ----------------------------------------------------------
    meths = sorted({blk for blk in L["block"] if blk.startswith("method:")})
    names = [r["label"] for r in rows if r["key"][1] not in APPENDIX_ONLY]
    ap_lines = [r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "r" * (3 * len(meths)) + "}", r"\toprule",
                "Method & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{esc(m.split(':', 2)[2])}}}"
                                         for m in meths) + r" \\",
                " & " + " & ".join(["Rec & FPR & AUROC"] * len(meths)) + r" \\", r"\midrule"]
    for n in names:
        cells = []
        for m in meths:
            s = L[(L["row"] == n) & (L["block"] == m)]
            cells += [f3(s.iloc[0]["Rec"]), f3(s.iloc[0]["FPR"]), f3(s.iloc[0]["AUROC"])] \
                if len(s) else ["---"] * 3
        ap_lines.append(n.replace("&", r"\&").replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    ap_lines += [r"\bottomrule", r"\end{tabular}%", "}"]
    open(os.path.join(od, "table_by_variant.tex"), "w").write("\n".join(ap_lines) + "\n")

    print("\n".join(md))
    print(f"\n[agg] wrote {od}/table_argar_btr.tex, table_by_variant.tex, table_argar_btr.md, "
          f"metrics_long.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
