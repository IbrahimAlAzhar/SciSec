#!/usr/bin/env python3
"""
v8_plan.py — what a job runs, chosen with ONE switch.

    python v8_plan.py --mode zeroshot     # a) the v8 LLM agents, no training. TEST data only.
    python v8_plan.py --mode sft_dpo      # b) rollouts -> SFT -> DPO on top of SFT
    python v8_plan.py --mode grpo         # c) rollouts -> SFT -> GRPO on top of SFT
    python v8_plan.py --mode all          #    every row: zero-shot, SFT, DPO-only, SFT+DPO,
                                          #    SFT-merged+DPO, GRPO (+ per-agent with --per-agent)

Add --with-zeroshot to b) or c) to also (re)score the zero-shot row in the same job. The
table always shows every row that exists on disk, so running a), then b), then c) as
separate jobs gives the same table as --mode all.

The PBS calls this with --shell and evals the output; printing it yourself shows the plan.
"""

from __future__ import annotations

import argparse

DATA = ["selftest", "data_test", "inputs"]
TRAIN_DATA = ["data_train"]
TAIL = ["judge", "table"]

PLANS = {
    "zeroshot": dict(stages=DATA + ["zeroshot"] + TAIL, evals=[], dpo_only=0,
                     what="the v8 LLM agents with no adapter; uses ONLY the test data"),
    "sft_dpo": dict(stages=["selftest", "data_train", "data_test", "inputs", "rollout",
                            "datasets", "sft", "dpo", "eval"] + TAIL,
                    evals=["sft", "sft_dpo"], dpo_only=0,
                    what="SFT on the rollouts, then DPO continuing the SFT adapter"),
    "grpo": dict(stages=["selftest", "data_train", "data_test", "inputs", "rollout",
                         "datasets", "sft", "grpo", "eval"] + TAIL,
                 evals=["sft", "grpo_prov"], dpo_only=0,
                 what="SFT on the rollouts, then GRPO (provenance reward) from the SFT adapter"),
    "all": dict(stages=["selftest", "data_train", "data_test", "inputs", "zeroshot",
                        "rollout", "datasets", "sft", "dpo", "merged", "grpo", "eval"] + TAIL,
                evals=["sft", "dpo_only", "sft_dpo", "sft_dpo_merged", "grpo_prov"],
                dpo_only=1, what="every row"),
}


def plan(mode: str, with_zeroshot: bool, per_agent: bool) -> dict:
    p = dict(PLANS[mode])
    st = list(p["stages"])
    if with_zeroshot and "zeroshot" not in st:
        st.insert(st.index("inputs") + 1, "zeroshot")
    ev = list(p["evals"])
    if per_agent and mode in ("sft_dpo", "all"):
        ev.append("sft_dpo_per_agent")
    return {"STAGES": ",".join(st), "EVAL_ABLATIONS": " ".join(ev),
            "DPO_ONLY": str(p["dpo_only"]), "MODE_WHAT": p["what"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=list(PLANS))
    ap.add_argument("--with-zeroshot", dest="with_zeroshot", action="store_true")
    ap.add_argument("--per-agent", dest="per_agent", action="store_true")
    ap.add_argument("--api", action="store_true",
                    help="an API reviewer (gpt-4o-mini): zero-shot only")
    ap.add_argument("--shell", action="store_true", help="print KEY='value' lines for bash")
    a = ap.parse_args()
    if a.api and a.mode != "zeroshot":
        raise SystemExit(f"[FATAL] MODE={a.mode}: gpt-4o-mini has no weights to train; "
                         f"use MODE=zeroshot")
    p = plan(a.mode, a.with_zeroshot, a.per_agent)
    if a.api:
        p["STAGES"] = p["STAGES"].replace("data_train,", "")
    if a.shell:
        for k, v in p.items():
            print(f"{k}='{v}'")
    else:
        print(f"MODE={a.mode}: {p['MODE_WHAT']}")
        print(f"  stages          {p['STAGES']}")
        print(f"  evaluated rows  {p['EVAL_ABLATIONS'] or '(zero-shot only)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
