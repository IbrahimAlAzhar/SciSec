"""
repair.py — V4: stage 2 for the PERTURBATION arm, and a behavioural detector built on it.

WHAT V3 SAID, AND WHY V4 CAN NOW SAY MORE
=======================================================================================
V3's perturbation arm was detection-only, and its reason was sound:

    "You cannot delete the attack (it is distributed over every paragraph) and you cannot
     neutralise framing (the framing was not touched). Reconstructing the authors'
     original wording would mean rewriting their whole manuscript from a corrupted copy."

That is still true of the general case. What changed is `perturb_forensics.RepairIndex`:
a REPAIRABLE token is one that is a single edit from a word this corpus actually uses, and
for those tokens the reconstruction is not a rewrite — it is a lookup with a named source
and a named target. "expreiments" -> "experiments" is checkable by eye. That is a much
smaller and much more defensible claim than "we restore the manuscript", and it is enough
to make stage 2 measurable on the two families that matter most.

WHAT IS AND IS NOT REPAIRED
=======================================================================================
    REPAIRED, confidently
      character corruption   a repairable token becomes its nearest in-corpus word
      intra-word punctuation "exper,iments" -> "experiments"
      homoglyphs             Cyrillic/Greek look-alikes folded back to Latin
      spacing anomalies      " ," -> ","

    QUARANTINED, and reported separately
      style-transferred sentences. There is no way to recover the authors' original
      sentence from a King James paraphrase of it. Dropping the sentence removes the
      attack and removes content with it, so it is done only when the sentence is a
      severe style outlier, it is counted, and the do-no-harm gate can veto it.

    NOT REPAIRED AT ALL, and said so
      synonym substitution. TextFooler and BertAttack leave real, correctly-spelled words
      in grammatical sentences. Nothing here can tell "significant" was once "notable"
      without the original, and guessing would corrupt clean papers. The report records
      `unrepaired_family="synonym_swap"` when that is the family the screen believes it
      saw, so a restoration number is never quietly credited to a family it did not fix.

THE DO-NO-HARM GATE
=======================================================================================
Same principle as renarrate.py, different failure to guard against. A repairer that is
too eager rewrites a clean paper's technical vocabulary into whatever the corpus happens
to contain — "ResNet" becomes "reset", "GeLU" becomes "glue" — and the collateral column
on the benign arm is what would expose it. The gate refuses to commit when:

    * every number in the manuscript did not survive verbatim (numeric_recall < 1.0)
    * the repair rate is implausibly high (a real corruption edits a few percent of words;
      20% means the vocabulary is wrong, not the paper)
    * quarantine would remove more than a small fraction of the document

A refusal is reported as a refusal. It is never returned as a repair with a restoration
number attached.

THE REPAIR PROBE — A DETECTOR THAT MEASURES INSTEAD OF ASKING
=======================================================================================
Both earlier versions failed in the same place: a model asked "is this document attacked?"
and handed a page of suspicious-looking evidence says yes. V2's insertion arm reached
FPR 1.000 that way; V3 fixed it by demanding a grounded span, at the cost of recall.

The probe asks the model NOTHING. It repairs the document and re-reviews it:

    probe_delta = rating(document as submitted) - rating(document after repair)

On a clean paper the repair is a near-no-op, so the two ratings agree and probe_delta ~ 0.
On a corrupted paper the repair undoes what the attack did, and the rating moves back by
roughly the amount the attack moved it. The evidence is a NUMBER THE MODEL PRODUCED
WITHOUT BEING ASKED THE QUESTION, so yes-bias has nothing to act on.

It costs one extra review call per probed document, which is why it runs only inside the
escalation band and only when `--repair-probe` is passed. The threshold is calibrated on
benign documents like every other gate in V4 (calibrate.py), so its false-positive rate is
bounded by construction rather than hoped for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from perturb_forensics import RepairIndex, style_profile, style_report, _SENT
from perturbation_defense import _WORD, _norm

_NUM = re.compile(r"\d+(?:[.,]\d+)*%?")
# punctuation wedged between two letters — PuncAttack's signature
_INTRA = re.compile(r"([A-Za-z]{2})([,;:!\"#$%&*+/=?@^_`|~.])([A-Za-z]{2})")
_SPACE_PUNCT = re.compile(r"\s+([,.;:!?])")
_DOUBLE_PUNCT = re.compile(r"([,.;:!?])\1{1,}")


def _match_case(src: str, repl: str) -> str:
    """Keep the original capitalisation: 'Expreiments' -> 'Experiments', not 'experiments'."""
    if src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


@dataclass
class RepairReport:
    repaired_text: str = ""
    n_char_repairs: int = 0
    n_punct_repairs: int = 0
    n_homoglyph_repairs: int = 0
    n_quarantined: int = 0
    quarantined_chars: int = 0
    repairs: List[Tuple[str, str]] = field(default_factory=list)
    quarantined: List[str] = field(default_factory=list)
    # gate
    committed: int = 0
    gate_reason: str = ""
    numeric_recall: Optional[float] = None
    length_ratio: Optional[float] = None
    repair_rate: float = 0.0
    unrepaired_family: str = ""

    @property
    def n_repairs(self) -> int:
        return (self.n_char_repairs + self.n_punct_repairs
                + self.n_homoglyph_repairs + self.n_quarantined)

    def as_row(self, prefix: str = "rep_") -> Dict:
        p = prefix
        return {
            f"{p}committed": self.committed,
            f"{p}n_repairs": self.n_repairs,
            f"{p}n_char": self.n_char_repairs,
            f"{p}n_punct": self.n_punct_repairs,
            f"{p}n_homoglyph": self.n_homoglyph_repairs,
            f"{p}n_quarantined": self.n_quarantined,
            f"{p}quarantined_chars": self.quarantined_chars,
            f"{p}repair_rate": round(self.repair_rate, 5),
            f"{p}numeric_recall": self.numeric_recall,
            f"{p}length_ratio": self.length_ratio,
            f"{p}gate_reason": self.gate_reason[:200],
            f"{p}unrepaired_family": self.unrepaired_family,
            f"{p}examples": " | ".join(f"{a}->{b}" for a, b in self.repairs[:10]),
        }


def repair_document(text: str, vocab: Set[str], index: Optional[RepairIndex] = None,
                    *, quarantine_style: bool = True,
                    style_z: float = 8.0,
                    max_repair_rate: float = 0.20,
                    max_quarantine_frac: float = 0.05,
                    numeric_recall_min: float = 1.0,
                    length_ratio_min: float = 0.85,
                    min_vocab: int = 3000,
                    family: str = "") -> RepairReport:
    """
    Undo what can be undone, refuse the rest, and say which is which.

    `family` is the screen's best guess at the attack family. It is used only to fill in
    `unrepaired_family` — the repairer's behaviour does not depend on it, because acting
    on a guessed family would let a wrong guess silently disable the repair that would
    have worked.
    """
    r = RepairReport(repaired_text=text or "")
    src = text or ""
    if not src.strip():
        r.gate_reason = "empty document"
        return r

    idx = index if index is not None else (RepairIndex(vocab) if vocab else None)
    out = src

    # ---- 1. homoglyphs, via the module the insertion arm already uses -----------------
    try:
        from normalize import normalize as _normalize
        nm = _normalize(out)
        if getattr(nm, "text", None) and nm.text != out:
            r.n_homoglyph_repairs = int(getattr(nm, "n_homoglyph", 0) or 0) or 1
            out = nm.text
    except Exception:
        pass

    # ---- 2. punctuation driven inside words ------------------------------------------
    def _unpunct(m):
        return m.group(1) + m.group(3)
    out, n_intra = _INTRA.subn(_unpunct, out)
    out, n_sp = _SPACE_PUNCT.subn(r"\1", out)
    out, n_dbl = _DOUBLE_PUNCT.subn(r"\1", out)
    r.n_punct_repairs = n_intra + n_sp + n_dbl

    # ---- 3. character corruption ------------------------------------------------------
    #
    # THE MINIMUM-VOCABULARY GUARD, AND THE BUG THAT PUT IT HERE.
    # A test with a 73-word reference vocabulary produced the repair "three" -> "there".
    # Both are ordinary English words; "three" was only "out-of-corpus" because the
    # corpus was too small to contain it. That is the exact shape of the failure this
    # module must never ship — a repairer that rewrites the authors' correct words into
    # different correct words, quietly, on a clean manuscript.
    #
    # A frequency or edit-distance threshold cannot fix it, because the source word's
    # corpus frequency is zero BY DEFINITION of being a candidate. What fixes it is
    # refusing to believe a thin reference: with a real corpus of a hundred papers,
    # "three" is in the vocabulary and never becomes a candidate at all. So below
    # `min_vocab` word types, character repair is DISABLED rather than attempted, and the
    # report says so instead of returning a confident no-op.
    if vocab and len(vocab) < min_vocab:
        r.gate_reason = (f"character repair DISABLED: the reference vocabulary has only "
                         f"{len(vocab)} word types (< {min_vocab}). A thin vocabulary "
                         f"makes ordinary words look out-of-corpus and invents repairs "
                         f"between them ('three' -> 'there'). Punctuation and homoglyph "
                         f"repair still apply; character repair does not.")
        idx = None
    if idx is not None and vocab:
        seen: Dict[str, Optional[str]] = {}

        def _fix(m):
            w = m.group(0)
            nw = _norm(w)
            if len(nw) < 4 or nw in vocab or w.isupper():
                return w
            if nw not in seen:
                seen[nw] = idx.nearest(nw)
            tgt = seen[nw]
            if not tgt:
                return w
            r.n_char_repairs += 1
            if len(r.repairs) < 25:
                r.repairs.append((w, tgt))
            return _match_case(w, tgt)

        out = _WORD.sub(_fix, out)

    # ---- 4. style quarantine ----------------------------------------------------------
    # Last, and separately: it DELETES content. Everything above is a substitution that
    # preserves length and meaning; this does not, so it is counted apart and the gate
    # bounds it independently.
    if quarantine_style:
        st = style_report(out, z_threshold=style_z)
        if st.n_outliers:
            drop = set(st.examples)
            kept = []
            for s in _SENT.split(out):
                head = s.strip()[:200]
                if head and head in drop:
                    r.n_quarantined += 1
                    r.quarantined_chars += len(s)
                    r.quarantined.append(head)
                    continue
                kept.append(s)
            out = " ".join(kept)

    # ---- 5. the do-no-harm gate -------------------------------------------------------
    n_words = max(1, len(_WORD.findall(src)))
    r.repair_rate = (r.n_char_repairs + r.n_punct_repairs) / n_words
    src_nums = _NUM.findall(src)
    out_nums = set(_NUM.findall(out))
    r.numeric_recall = (sum(1 for x in src_nums if x in out_nums) / len(src_nums)
                        if src_nums else 1.0)
    r.length_ratio = len(out) / max(1, len(src))
    r.unrepaired_family = family if family in ("synonym_swap",) else ""

    reasons = []
    if r.numeric_recall is not None and r.numeric_recall < numeric_recall_min:
        reasons.append(f"numeric recall {r.numeric_recall:.3f} < {numeric_recall_min} — "
                       f"the repair changed one of the paper's own numbers, which is "
                       f"damage, not repair")
    if r.repair_rate > max_repair_rate:
        reasons.append(f"repair rate {100*r.repair_rate:.1f}% > "
                       f"{100*max_repair_rate:.0f}% — a real corruption edits a few "
                       f"percent of words; this means the reference vocabulary is wrong "
                       f"for this paper, not that the paper is corrupted")
    if r.quarantined_chars > max_quarantine_frac * len(src):
        reasons.append(f"quarantine would remove {100*r.quarantined_chars/len(src):.1f}% "
                       f"of the document")
    if r.length_ratio is not None and r.length_ratio < length_ratio_min:
        reasons.append(f"length ratio {r.length_ratio:.2f} < {length_ratio_min}")

    if reasons:
        r.committed = 0
        r.gate_reason = "REFUSED: " + "; ".join(reasons)
        r.repaired_text = src           # hand back the original, untouched
        return r

    r.committed = 1
    r.repaired_text = out
    _thin = r.gate_reason if r.gate_reason.startswith("character repair DISABLED") else ""
    r.gate_reason = ((_thin + " " if _thin else "")
                     + f"committed: {r.n_char_repairs} character repair(s), "
                     f"{r.n_punct_repairs} punctuation, {r.n_homoglyph_repairs} homoglyph, "
                     f"{r.n_quarantined} sentence(s) quarantined; every number preserved")
    return r


# ======================================================================================
# the repair probe — behavioural detection
# ======================================================================================
@dataclass
class ProbeResult:
    ran: int = 0
    probe_delta: Optional[float] = None      # rating(as submitted) - rating(repaired)
    rating_submitted: Optional[float] = None
    rating_repaired: Optional[float] = None
    n_llm_calls: int = 0
    note: str = ""

    def as_row(self) -> Dict:
        return {
            "probe_ran": self.ran,
            "probe_delta": self.probe_delta,
            "probe_rating_submitted": self.rating_submitted,
            "probe_rating_repaired": self.rating_repaired,
            "probe_calls": self.n_llm_calls,
            "probe_note": self.note[:200],
        }


def _rating_of(rv) -> Optional[float]:
    if rv is None:
        return None
    v = rv.get("rating") if isinstance(rv, dict) else getattr(rv, "rating", None)
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def repair_probe(document: str, rep: RepairReport, reviewer,
                 submitted_rating: Optional[float] = None,
                 review_fn=None) -> ProbeResult:
    """
    Ask the model nothing; measure it instead.

    `submitted_rating` is the rating the benchmark already recorded for this document. When
    it is available the probe costs ONE call, not two — and it should be used, because
    re-reviewing the submitted document here would produce a second, slightly different
    rating for the same text and the delta would carry that noise for no reason.
    """
    p = ProbeResult()
    if not rep.committed or rep.n_repairs == 0:
        p.note = ("no repair was committed, so there is nothing to compare — the probe is "
                  "SKIPPED, which is not the same as a probe that returned zero")
        return p
    if reviewer is None:
        p.note = "no reviewer available"
        return p

    fn = review_fn or reviewer.review
    r_sub = submitted_rating
    if r_sub is None:
        r_sub = _rating_of(fn(document))
        p.n_llm_calls += 1
    r_rep = _rating_of(fn(rep.repaired_text))
    p.n_llm_calls += 1
    p.ran = 1
    p.rating_submitted, p.rating_repaired = r_sub, r_rep
    if r_sub is None or r_rep is None:
        p.note = "one of the two reviews was unparseable; probe_delta is EMPTY, not zero"
        return p
    p.probe_delta = r_sub - r_rep
    p.note = (f"the same reviewer scored the submitted text {r_sub:.1f} and the repaired "
              f"text {r_rep:.1f}; a clean paper is unchanged by repair, so a large "
              f"positive delta is the attack's gain being undone")
    return p
