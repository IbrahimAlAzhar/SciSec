#!/usr/bin/env python3
"""
ft_judge.py — turn scored samples into (chosen, rejected) pairs and SFT targets.

WHAT THE JUDGE IS ALLOWED TO DECIDE, AND WHAT IT IS NOT
=======================================================================================
NOT ALLOWED: whether an answer is correct. That is known. The label came from the attack
we applied, and whether a quote is verbatim is string containment. Routing ground truth
through gpt-4o-mini would add the judge's error rate to a number we already have exactly,
and every downstream metric would inherit it.

ALLOWED: which of two ALREADY-CORRECT answers is better. Both say ATTACKED, both quote real
text, both name the family — but one quotes the actual instruction and explains what it
does to a reader, and the other quotes a fragment and says "this looks suspicious". That is
a judgement about writing quality, the judge is good at it, and it is exactly the signal
SFT needs in order to teach reasoning rather than format.

So the judge is called ONLY when a prompt has two or more correct samples. On a prompt with
one correct sample there is nothing to rank; on a prompt with none, the target is the
constructed gold answer. That bounds the cost to roughly (#prompts with >=2 correct) calls.

HOW A PAIR IS FORMED — AND THE MISTAKE THIS AVOIDS
    chosen and rejected are two completions OF THE SAME PROMPT. Pairing a clean document's
    answer against an attacked document's answer is the intuitive reading of "good sample
    vs bad sample", but the prompts would differ, and DPO on differing prompts does not
    teach detection — the gradient can satisfy the objective by preferring one document
    over another, which is not a behaviour that transfers to a new paper.

    rejected is preferably a REAL mistake the model made. Where the model made none,
    a controlled corruption is synthesised, capped at a fraction of the set: a preference
    set built mostly from straw men teaches the model to avoid errors it was never going to
    make, while leaving the ones it does make untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import ft_config as C
import ft_roles as R

try:
    import httpx
except Exception:                                    # pragma: no cover
    httpx = None


# ======================================================================================
# the judge
# ======================================================================================
JUDGE_SYSTEM = (
    "You compare two answers produced by an automated reviewer-integrity checker. BOTH "
    "answers have already been verified as FACTUALLY CORRECT: the verdict matches ground "
    "truth and every quoted span appears verbatim in the manuscript. You are NOT asked "
    "which is correct.\n\n"
    "You are asked which is the better answer to LEARN FROM, on four criteria, in this "
    "order of importance:\n"
    "  1. EVIDENCE PRECISION — does the quoted span isolate the manipulated text itself, "
    "or does it drag in surrounding legitimate prose?\n"
    "  2. REASONING — does the explanation say what the text DOES to a reader, or does it "
    "only assert that something is suspicious?\n"
    "  3. CALIBRATION — is the stated confidence proportionate to the evidence given?\n"
    "  4. RESTRAINT — is it free of claims that go beyond what is quoted?\n\n"
    "Answer with ONE JSON object and nothing else:\n"
    '{"winner": "A" or "B", "margin": "clear" or "slight", "why": "one sentence"}')


def judge_prompt(role: str, user_prompt: str, a: str, b: str,
                 max_ctx: int = 2500) -> str:
    ctx = user_prompt if len(user_prompt) <= max_ctx else \
        user_prompt[:900] + "\n[...]\n" + user_prompt[-1500:]
    return (f"ROLE UNDER TEST: {role}\n\n"
            f"WHAT THE CHECKER WAS SHOWN (abridged):\n\"\"\"\n{ctx}\n\"\"\"\n\n"
            f"ANSWER A:\n{a}\n\nANSWER B:\n{b}\n\n"
            f"Which is the better answer to learn from?")


class Judge:
    """gpt-4o-mini over HTTPS, with a disk cache and a hard call budget."""

    def __init__(self, model: str = "", cache_dir: str = "", max_calls: int = 100000,
                 timeout: float = 120.0):
        self.model = model or C.JUDGE_MODEL
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.max_calls = max_calls
        self.n_calls = self.n_cached = self.n_failed = self.n_budget_skipped = 0
        self.key = os.environ.get("OPENAI_API_KEY", "")
        self._c = httpx.Client(timeout=timeout) if httpx is not None else None

    # ------------------------------------------------------------------ cache
    def _cpath(self, h: str) -> str:
        return os.path.join(self.cache_dir, f"{h}.json") if self.cache_dir else ""

    def _get(self, h: str) -> Optional[Dict]:
        p = self._cpath(h)
        if p and os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _put(self, h: str, v: Dict) -> None:
        p = self._cpath(h)
        if not p:
            return
        try:
            with open(p, "w") as f:
                json.dump(v, f)
        except Exception:
            pass

    # ------------------------------------------------------------------ call
    def compare(self, role: str, user_prompt: str, a: str, b: str) -> Optional[Dict]:
        """
        Returns {"winner": "A"|"B", "margin": ..., "why": ...} or None.

        None means "no opinion", and the caller must then fall back to a deterministic
        tie-break rather than dropping the pair — a judge outage should cost quality, not
        data.
        """
        payload = json.dumps([role, user_prompt[-3000:], a, b], sort_keys=True)
        h = hashlib.sha1((self.model + "|v1|" + payload).encode()).hexdigest()
        hit = self._get(h)
        if hit is not None:
            self.n_cached += 1
            return hit or None
        if self.n_calls >= self.max_calls:
            self.n_budget_skipped += 1
            return None
        if not self.key or self._c is None:
            self.n_failed += 1
            return None
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user",
                          "content": judge_prompt(role, user_prompt, a, b)}],
            "temperature": 0.0, "max_tokens": 200,
        }
        try:
            r = self._c.post("https://api.openai.com/v1/chat/completions",
                             headers={"Authorization": f"Bearer {self.key}"}, json=body)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            self.n_calls += 1
        except Exception as e:
            self.n_failed += 1
            if self.n_failed <= 3:
                print(f"  [judge] {type(e).__name__}: {e}")
            return None
        from ft_vllm import extract_json
        obj = extract_json(txt) or {}
        w = str(obj.get("winner", "")).strip().upper()
        if w not in ("A", "B"):
            self._put(h, {})
            return None
        out = {"winner": w, "margin": str(obj.get("margin", "")).strip().lower(),
               "why": str(obj.get("why", ""))[:300]}
        self._put(h, out)
        return out

    def report(self) -> str:
        return (f"[judge] {self.model}: {self.n_calls} calls, {self.n_cached} cache hits, "
                f"{self.n_failed} failed, {self.n_budget_skipped} skipped for budget")


# ======================================================================================
# deterministic quality tie-break, used when the judge has no opinion
# ======================================================================================
def heuristic_quality(role: str, obj: Dict, insert: str) -> float:
    """
    A small, explicit score. Not a substitute for the judge — a fallback so a missing API
    key degrades the pairs' quality instead of emptying the dataset.
    """
    if not isinstance(obj, dict):
        return 0.0
    s = 0.0
    span = str(obj.get("span") or "")
    if role in ("INSTRUCTION", "PRESSURE", "VOICE"):
        f = obj.get("findings") or []
        span = str((f[0] or {}).get("span") or "") if f and isinstance(f[0], dict) else ""
    if span and insert:
        ns, ni = " ".join(span.split()), " ".join(insert.split())
        if ns and ni:
            # precision: how much of the quote is inside the real insert
            s += 2.0 if ns in ni else 0.0
            # length proportion: a quote that is 20% of the insert is worse evidence than
            # one that is 80%, and a quote LONGER than the insert has pulled in the
            # authors' own prose
            ratio = len(ns) / max(1, len(ni))
            s += 1.5 * (ratio if ratio <= 1.0 else max(0.0, 2.0 - ratio))
    reason = str(obj.get("reason") or "")
    words = len(reason.split())
    s += 1.0 if 12 <= words <= 60 else (0.3 if words else 0.0)
    # a reason that merely asserts suspicion teaches nothing
    if re.search(r"\b(suspicious|looks like|seems|might be)\b", reason, re.I):
        s -= 0.5
    try:
        conf = float(obj.get("confidence", 0.0))
        s += 0.5 if 0.5 <= conf <= 0.95 else 0.0
    except Exception:
        pass
    if role == "LEADER" and obj.get("attack_type"):
        s += 0.5
    return s


# ======================================================================================
# synthesising a rejected sample, when the model made no mistake
# ======================================================================================
def synth_rejected(role: str, gold: Dict, *, label: int, document: str, insert: str,
                   rng: random.Random) -> Optional[Tuple[Dict, str]]:
    """
    A controlled corruption of the correct answer, targeting the failure modes your V4
    tables actually show. Returns (object, kind) or None.

        label 0  ->  flag it anyway            the FPR 0.400 failure
        label 1  ->  drop the quote            the ungrounded-verdict failure
                 ->  quote the authors' prose  the right-verdict-wrong-evidence failure
                 ->  say CLEAN                 the recall failure
    """
    g = dict(gold)
    if role == "LEADER":
        if label == 0:
            kinds = ["false_positive"]
        else:
            kinds = ["no_span", "wrong_span", "miss"]
        kind = rng.choice(kinds)
        if kind == "false_positive":
            # quote real text from the document, so the ONLY defect is the verdict — a
            # negative that also fails the quote check would let the model satisfy the
            # preference by learning the quote rule instead of the restraint.
            reg = R.region(document)
            sents = [s for s in re.split(r"(?<=[.!?])\s+", reg) if len(s) > 60]
            q = rng.choice(sents) if sents else reg[:160]
            return ({"verdict": "ATTACKED", "confidence": 0.78,
                     "attack_type": "A2_reframing", "severity": 2, "span": q.strip()[:300],
                     "location": "the limitations section",
                     "reason": "The limitations are phrased in a way that seems designed "
                               "to soften criticism, which looks like manipulation."},
                    kind)
        if kind == "no_span":
            g.update({"span": "", "confidence": 0.9,
                      "reason": "The limitations section has clearly been manipulated to "
                                "influence the reviewer."})
            return g, kind
        if kind == "wrong_span":
            reg = R.region(document)
            body = reg.replace(insert, " ") if insert else reg
            sents = [s for s in re.split(r"(?<=[.!?])\s+", body) if len(s) > 60]
            if not sents:
                return None
            g.update({"span": rng.choice(sents).strip()[:300]})
            return g, kind
        return ({"verdict": "CLEAN", "confidence": 0.7, "attack_type": "", "severity": 0,
                 "span": "", "location": "",
                 "reason": "The limitations read like an ordinary discussion of the "
                           "study's scope."}, kind)

    if role == "CHUNK_VERIFIER":
        v = bool(gold.get("is_instruction"))
        return ({"is_instruction": (not v), "confidence": 0.75, "span": "",
                 "reason": "The passage reads as addressed to the reader." if not v
                           else "The passage describes the authors' own work."},
                "flipped")

    # the three evidence workers
    if label == 0:
        reg = R.region(document)
        sents = [s for s in re.split(r"(?<=[.!?])\s+", reg) if len(s) > 60]
        if not sents:
            return None
        return ({"findings": [{"span": rng.choice(sents).strip()[:300],
                               "why": "This phrasing could be intended to steer the "
                                      "reader."}],
                 "confidence": 0.7}, "false_positive")
    return ({"findings": [], "confidence": 0.4}, "missed")


# ======================================================================================
# the main pass
# ======================================================================================
@dataclass
class Pair:
    group_id: str
    model: str
    role: str
    paper_id: str
    condition: str
    label: int
    system: str
    user: str
    chosen: str
    rejected: str
    chosen_source: str          # "sampled" | "gold"
    rejected_source: str        # "sampled" | "synth:<kind>"
    judge: Optional[Dict] = None

    def as_dict(self) -> Dict:
        return self.__dict__.copy()


def _rank_unusable(s: Dict) -> int:
    """
    Which real mistake makes the most informative `rejected`.

    A wrong verdict is the most valuable negative — it is the error that costs accuracy.
    An unparseable completion is the least: it teaches format, which SFT already fixed, and
    it is so far out of distribution that the DPO gradient on it is mostly noise.
    """
    sc = s.get("score") or {}
    if sc.get("label_correct") is False:
        return 0
    if sc.get("span_on_target") is False:
        return 1
    if sc.get("span_grounded") is False:
        return 2
    if not sc.get("schema_ok"):
        return 3
    if not sc.get("parsed"):
        return 4
    return 5


def build_pairs(samples: List[Dict], judge: Optional[Judge], *, seed: int = 0,
                max_synth_frac: float = C.GEN.max_synthetic_rejected_frac
                ) -> Tuple[List[Pair], List[Dict], Dict]:
    rng = random.Random(seed)
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        gid = hashlib.sha1(
            "|".join([s.get("model", ""), s.get("role", ""), s.get("paper_id", ""),
                      s.get("condition", ""), s.get("chunk", "")[:64],
                      s.get("user", "")[-200:]]).encode()).hexdigest()[:16]
        groups[gid].append(s)

    pairs: List[Pair] = []
    sft: List[Dict] = []
    stats = {"n_groups": len(groups), "chosen_sampled": 0, "chosen_gold": 0,
             "rejected_sampled": 0, "rejected_synth": 0, "no_pair": 0,
             "judge_used": 0, "judge_no_opinion": 0, "synth_kinds": {},
             "by_role": {}}

    order = sorted(groups)
    rng.shuffle(order)
    n_synth_allowed_total = int(max_synth_frac * len(order)) + 1
    n_synth = 0

    for gid in order:
        g = groups[gid]
        s0 = g[0]
        role, label = s0["role"], int(s0["label"])
        insert = s0.get("insert", "")
        document = s0.get("document", "") or s0.get("chunk", "")
        st = stats["by_role"].setdefault(role, {"groups": 0, "pairs": 0, "sft": 0,
                                                "gold_targets": 0, "synth": 0})
        st["groups"] += 1

        usable = [s for s in g if (s.get("score") or {}).get("usable")]
        unusable = [s for s in g if not (s.get("score") or {}).get("usable")]

        # ---- chosen -------------------------------------------------------------
        if len(usable) >= 2 and judge is not None:
            a, b = usable[0], usable[1]
            v = judge.compare(role, s0["user"], a["completion"].strip(),
                              b["completion"].strip())
            if v is None:
                stats["judge_no_opinion"] += 1
                from ft_vllm import extract_json
                qa = heuristic_quality(role, extract_json(a["completion"]) or {}, insert)
                qb = heuristic_quality(role, extract_json(b["completion"]) or {}, insert)
                best = a if qa >= qb else b
            else:
                stats["judge_used"] += 1
                best = a if v["winner"] == "A" else b
            chosen_txt, chosen_src, jinfo = best["completion"].strip(), "sampled", v
        elif usable:
            chosen_txt, chosen_src, jinfo = usable[0]["completion"].strip(), "sampled", None
        else:
            gold = R.gold_target(role, label=label, document=document, insert=insert,
                                 condition=s0.get("condition", ""),
                                 chunk=s0.get("chunk", ""),
                                 chunk_has_insert=bool(s0.get("chunk_has_insert")))
            chosen_txt, chosen_src, jinfo = R.render_target(gold), "gold", None
            st["gold_targets"] += 1

        stats["chosen_sampled" if chosen_src == "sampled" else "chosen_gold"] += 1
        sft.append({"group_id": gid, "model": s0["model"], "role": role,
                    "paper_id": s0["paper_id"], "condition": s0.get("condition", ""),
                    "label": label, "system": s0["system"], "user": s0["user"],
                    "completion": chosen_txt, "source": chosen_src})
        st["sft"] += 1

        # ---- rejected -----------------------------------------------------------
        rej_txt = rej_src = ""
        if unusable:
            worst = sorted(unusable, key=_rank_unusable)[0]
            # An unparseable completion is a poor negative; prefer a well-formed wrong
            # answer when the group has one.
            if (worst.get("score") or {}).get("parsed") or len(unusable) == 1:
                rej_txt, rej_src = worst["completion"].strip(), "sampled"
        if not rej_txt and n_synth < n_synth_allowed_total:
            from ft_vllm import extract_json
            base = extract_json(chosen_txt) or R.gold_target(
                role, label=label, document=document, insert=insert,
                condition=s0.get("condition", ""), chunk=s0.get("chunk", ""),
                chunk_has_insert=bool(s0.get("chunk_has_insert")))
            syn = synth_rejected(role, base, label=label, document=document,
                                 insert=insert, rng=rng)
            if syn is not None:
                obj, kind = syn
                rej_txt, rej_src = R.render_target(obj), f"synth:{kind}"
                stats["synth_kinds"][kind] = stats["synth_kinds"].get(kind, 0) + 1
                n_synth += 1
                st["synth"] += 1

        if not rej_txt or rej_txt == chosen_txt:
            stats["no_pair"] += 1
            continue
        stats["rejected_sampled" if rej_src == "sampled" else "rejected_synth"] += 1
        pairs.append(Pair(gid, s0["model"], role, s0["paper_id"],
                          s0.get("condition", ""), label, s0["system"], s0["user"],
                          chosen_txt, rej_txt, chosen_src, rej_src, jinfo))
        st["pairs"] += 1

    stats["n_pairs"] = len(pairs)
    stats["n_sft"] = len(sft)
    stats["synth_frac"] = (stats["rejected_synth"] / len(pairs)) if pairs else 0.0
    return pairs, sft, stats


def main() -> int:
    p = argparse.ArgumentParser(description="judge samples and build SFT / DPO records")
    p.add_argument("--model", required=True, choices=sorted(C.MODELS))
    p.add_argument("--samples", default="")
    p.add_argument("--out-dir", dest="out_dir", default="")
    p.add_argument("--judge-model", dest="judge_model", default=C.JUDGE_MODEL)
    p.add_argument("--max-judge-calls", dest="max_judge_calls", type=int, default=20000)
    p.add_argument("--no-judge", dest="use_judge", action="store_false",
                   help="skip gpt-4o-mini entirely and rank with the deterministic "
                        "heuristic. Cheaper and fully offline; the pairs are then ranked "
                        "on evidence precision and reason length rather than on quality.")
    p.set_defaults(use_judge=True)
    p.add_argument("--max-synth-frac", dest="max_synth_frac", type=float,
                   default=C.GEN.max_synthetic_rejected_frac)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    mkey = a.model.replace("/", "_")
    samples_path = a.samples or C.wpath(C.D_TRAJ, mkey, "samples.jsonl")
    out_dir = a.out_dir or C.wpath(C.D_JUDGED, mkey)
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(samples_path):
        raise SystemExit(f"[FATAL] {samples_path} not found — run ft_trajectories.py first")

    samples = [json.loads(l) for l in open(samples_path) if l.strip()]
    print(f"[data] {len(samples)} samples from {samples_path}")

    judge = None
    if a.use_judge:
        if not os.environ.get("OPENAI_API_KEY"):
            print("[judge] OPENAI_API_KEY is not set. Falling back to the deterministic "
                  "ranking for every group — the pipeline still produces a full dataset, "
                  "it just ranks equally-correct answers by evidence precision and reason "
                  "length instead of by quality.")
        judge = Judge(a.judge_model, cache_dir=os.path.join(out_dir, "judge_cache"),
                      max_calls=a.max_judge_calls)

    pairs, sft, stats = build_pairs(samples, judge, seed=a.seed,
                                   max_synth_frac=a.max_synth_frac)

    with open(os.path.join(out_dir, "pairs.jsonl"), "w") as f:
        for pr in pairs:
            f.write(json.dumps(pr.as_dict(), ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "sft_records.jsonl"), "w") as f:
        for s in sft:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    if judge is not None:
        stats["judge"] = {"calls": judge.n_calls, "cached": judge.n_cached,
                          "failed": judge.n_failed,
                          "budget_skipped": judge.n_budget_skipped}
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 92)
    print(f"PAIRS AND TARGETS — {a.model}")
    print("=" * 92)
    print(f"  {stats['n_groups']} prompt groups -> {stats['n_sft']} SFT targets, "
          f"{stats['n_pairs']} DPO pairs")
    print(f"  SFT targets: {stats['chosen_sampled']} from the model's own correct "
          f"samples, {stats['chosen_gold']} constructed from gold")
    print(f"  DPO rejected: {stats['rejected_sampled']} real mistakes, "
          f"{stats['rejected_synth']} synthesised "
          f"({100*stats['synth_frac']:.0f}% synthetic, cap "
          f"{100*a.max_synth_frac:.0f}%)")
    if stats["synth_kinds"]:
        print("  synthesised negatives by kind: " +
              ", ".join(f"{k}={v}" for k, v in sorted(stats["synth_kinds"].items())))
    print(f"  {stats['no_pair']} group(s) yielded no pair (no usable contrast)")
    print(f"\n  {'role':<16} {'groups':>8} {'sft':>6} {'pairs':>7} {'gold tgt':>9} "
          f"{'synth rej':>10}")
    for role, d in stats["by_role"].items():
        print(f"  {role:<16} {d['groups']:>8} {d['sft']:>6} {d['pairs']:>7} "
              f"{d['gold_targets']:>9} {d['synth']:>10}")
    if judge is not None:
        print("\n  " + judge.report())
    if stats["chosen_gold"] > 0.7 * max(1, stats["n_sft"]):
        print("\n  [warning] most SFT targets are CONSTRUCTED, not sampled. The model "
              "almost never produced a correct answer, so this run will mostly teach "
              "output FORMAT. That is still worth having — a parse failure scores as a "
              "non-answer — but do not expect reasoning gains from it. Raise "
              "--n-samples or check the prompts first.")
    print(f"\n[saved] {out_dir}/pairs.jsonl")
    print(f"[saved] {out_dir}/sft_records.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
