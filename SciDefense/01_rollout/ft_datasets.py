#!/usr/bin/env python3
"""
ft_datasets.py — emit the files TRL actually trains on.

FOUR DATASETS COME OUT OF ONE PASS
    datasets/sft/<model>/all.jsonl            every role      -> the single-adapter run
    datasets/sft/<model>/<role>.jsonl         one role        -> the per-agent run
    datasets/dpo/<model>/all.jsonl            every role
    datasets/dpo/<model>/<role>.jsonl         one role

FORMAT: `prompt` / `completion` for SFT and `prompt` / `chosen` / `rejected` for DPO, all
as CHAT MESSAGE LISTS. TRL applies the tokenizer's chat template itself when the fields are
conversational, which is what keeps gemma (no system role), qwen and mistral consistent
without three code paths. A pre-rendered string would bake one model's special tokens into
a file the other two then train on.

THE SPLIT
    A validation split by PAPER, never by row. The same manuscript appears as one benign
    and up to four attacked documents, and across five roles — so a row-level split puts
    the same paper on both sides and the eval loss stops measuring generalisation. That is
    the single most common way a fine-tune looks better than it is.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import ft_config as C
import ft_roles as R


def to_messages(rec: Dict, *, system_role: bool) -> List[Dict[str, str]]:
    return R.messages(rec["role"], rec["system"], rec["user"], system_role=system_role)


def split_by_paper(recs: List[Dict], val_frac: float, seed: int) -> Tuple[List, List]:
    papers = sorted({r["paper_id"] for r in recs})
    rng = random.Random(seed)
    rng.shuffle(papers)
    n_val = max(1, int(round(val_frac * len(papers)))) if papers else 0
    val = set(papers[:n_val])
    tr = [r for r in recs if r["paper_id"] not in val]
    va = [r for r in recs if r["paper_id"] in val]
    return tr, va


def write_jsonl(path: str, rows: Iterable[Dict]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def balance_report(recs: List[Dict], what: str) -> str:
    lab = Counter(int(r["label"]) for r in recs)
    fam = Counter(r.get("condition", "") for r in recs)
    L = [f"  {what}: {len(recs)} records   attacked={lab.get(1,0)} clean={lab.get(0,0)}"]
    if lab.get(1, 0) and lab.get(0, 0):
        share = lab[1] / (lab[1] + lab[0])
        L.append(f"    positive share {share:.1%}")
        if share > 0.8 or share < 0.2:
            L.append("    ^ IMBALANCED. A detector trained at this ratio learns the prior, "
                     "not the signal: predicting the majority class alone would score "
                     f"{max(share, 1-share):.0%} accuracy. Cap the attacked side with "
                     "--max-per-paper, or accept it and report FPR rather than accuracy.")
    top = ", ".join(f"{k}={v}" for k, v in fam.most_common(6))
    L.append(f"    families: {top}")
    return "\n".join(L)


def build(model_key: str, judged_dir: str, out_sft: str, out_dpo: str, *,
          val_frac: float = 0.1, seed: int = 0, max_per_paper: int = 10 ** 9,
          roles: Sequence[str] = C.ROLES) -> Dict:
    spec = C.MODELS[model_key]
    sft_in = os.path.join(judged_dir, "sft_records.jsonl")
    dpo_in = os.path.join(judged_dir, "pairs.jsonl")
    for p in (sft_in, dpo_in):
        if not os.path.exists(p):
            raise SystemExit(f"[FATAL] {p} not found — run ft_judge.py first")

    sft_recs = [json.loads(l) for l in open(sft_in) if l.strip()]
    dpo_recs = [json.loads(l) for l in open(dpo_in) if l.strip()]
    sft_recs = [r for r in sft_recs if r["role"] in roles]
    dpo_recs = [r for r in dpo_recs if r["role"] in roles]

    # optional per-paper cap, applied per role so one prolific paper cannot dominate
    if max_per_paper < 10 ** 9:
        def cap(recs):
            seen: Dict[Tuple[str, str], int] = defaultdict(int)
            out = []
            for r in recs:
                k = (r["role"], r["paper_id"])
                if seen[k] >= max_per_paper:
                    continue
                seen[k] += 1
                out.append(r)
            return out
        sft_recs, dpo_recs = cap(sft_recs), cap(dpo_recs)

    stats: Dict = {"model": model_key, "system_role": spec.system_role, "roles": {}}
    print(f"\n[build] {model_key}   system_role={spec.system_role}"
          f"{'' if spec.system_role else '  (system text merged into the user turn)'}")

    # ------------------------------------------------------------------ SFT
    def sft_rows(recs):
        for r in recs:
            yield {"prompt": to_messages(r, system_role=spec.system_role),
                   "completion": [{"role": "assistant", "content": r["completion"]}],
                   "role_name": r["role"], "paper_id": r["paper_id"],
                   "condition": r.get("condition", ""), "label": int(r["label"]),
                   "source": r.get("source", "")}

    tr, va = split_by_paper(sft_recs, val_frac, seed)
    n_tr = write_jsonl(os.path.join(out_sft, "all.train.jsonl"), sft_rows(tr))
    n_va = write_jsonl(os.path.join(out_sft, "all.val.jsonl"), sft_rows(va))
    print(f"\n[sft] all roles -> train {n_tr}, val {n_va}")
    print(balance_report(tr, "sft/all train"))
    stats["sft_all"] = {"train": n_tr, "val": n_va}

    # ------------------------------------------------------------------ DPO
    def dpo_rows(recs):
        for r in recs:
            yield {"prompt": to_messages(r, system_role=spec.system_role),
                   "chosen": [{"role": "assistant", "content": r["chosen"]}],
                   "rejected": [{"role": "assistant", "content": r["rejected"]}],
                   "role_name": r["role"], "paper_id": r["paper_id"],
                   "condition": r.get("condition", ""), "label": int(r["label"]),
                   "chosen_source": r.get("chosen_source", ""),
                   "rejected_source": r.get("rejected_source", "")}

    tr, va = split_by_paper(dpo_recs, val_frac, seed)
    n_tr = write_jsonl(os.path.join(out_dpo, "all.train.jsonl"), dpo_rows(tr))
    n_va = write_jsonl(os.path.join(out_dpo, "all.val.jsonl"), dpo_rows(va))
    print(f"\n[dpo] all roles -> train {n_tr}, val {n_va}")
    print(balance_report(tr, "dpo/all train"))
    stats["dpo_all"] = {"train": n_tr, "val": n_va}

    # ------------------------------------------------------------------ per role
    print(f"\n  {'role':<16} {'sft train':>10} {'sft val':>8} {'dpo train':>10} "
          f"{'dpo val':>8}")
    for role in roles:
        rs = [r for r in sft_recs if r["role"] == role]
        rd = [r for r in dpo_recs if r["role"] == role]
        a, b = split_by_paper(rs, val_frac, seed)
        c, d = split_by_paper(rd, val_frac, seed)
        n1 = write_jsonl(os.path.join(out_sft, f"{role}.train.jsonl"), sft_rows(a))
        n2 = write_jsonl(os.path.join(out_sft, f"{role}.val.jsonl"), sft_rows(b))
        n3 = write_jsonl(os.path.join(out_dpo, f"{role}.train.jsonl"), dpo_rows(c))
        n4 = write_jsonl(os.path.join(out_dpo, f"{role}.val.jsonl"), dpo_rows(d))
        stats["roles"][role] = {"sft_train": n1, "sft_val": n2,
                                "dpo_train": n3, "dpo_val": n4}
        print(f"  {role:<16} {n1:>10} {n2:>8} {n3:>10} {n4:>8}")
        if n1 < 100:
            print(f"    ^ {n1} examples is thin for a rank-{C.TRAIN.lora_r} adapter. "
                  f"Either raise --conditions-per-paper in ft_attack.py, or train this "
                  f"role from the shared 'all' set instead.")

    with open(os.path.join(out_sft, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[saved] {out_sft}/")
    print(f"[saved] {out_dpo}/")
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description="emit SFT and DPO datasets for TRL")
    p.add_argument("--model", required=True, choices=sorted(C.MODELS))
    p.add_argument("--judged-dir", dest="judged_dir", default="")
    p.add_argument("--val-frac", dest="val_frac", type=float, default=0.1)
    p.add_argument("--max-per-paper", dest="max_per_paper", type=int, default=10 ** 9)
    p.add_argument("--roles", nargs="+", default=list(C.ROLES), choices=list(C.ROLES))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    mk = a.model.replace("/", "_")
    build(a.model, a.judged_dir or C.wpath(C.D_JUDGED, mk),
          C.wpath(C.D_SFT, mk), C.wpath(C.D_DPO, mk),
          val_frac=a.val_frac, seed=a.seed, max_per_paper=a.max_per_paper,
          roles=a.roles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
