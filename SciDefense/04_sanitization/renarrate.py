"""
renarrate.py — stage-2 remediation for REPACKAGING, where deletion is the wrong operation.

THE PROBLEM WITH DELETING
=======================================================================================
`defense.remove_spans` is correct for an inserted payload: foreign text goes in, foreign
text comes out, the manuscript is whole again. Against ARGAR it is actively harmful. The
attacked text IS the abstract, the introduction, the discussion. Cutting them out does not
restore the paper to its pre-attack state; it produces a mutilated document, and the
reviewer's rating would then fall for the wrong reason. A defence that scores well on
`restoration` by destroying the paper has proved nothing — that is the same failure mode
`collateral` was added to catch on the insertion side.

So the remediation here is NEUTRALISATION: rewrite the overreaching framing back to a
plain, factual register, and keep everything else — every number, every experiment, the
authors' limitations, the paper's structure. The rating should fall because the paper is
no longer over-claiming, not because it has been damaged.

THE INTEGRITY GATE, AND WHY IT POINTS BOTH WAYS
=======================================================================================
ARGAR polices its own edits: numeric_recall >= 0.98, limitation_recall >= 0.60. It has to,
or its "presentation-only" claim collapses. The defence needs the SAME gate for the exact
opposite reason: our neutralising editor is an LLM rewriting a real author's paper, and if
it drops their numbers or their limitations then the defence has done the damage the
attack was careful not to do. Same thresholds, opposite purpose. A rewrite that fails the
gate is REJECTED and the paper is left untouched — the defence's own do-no-harm rule.

THE DO-NO-HARM GATE (adapted from ARGAR's direction gate, Eq. 2)
=======================================================================================
ARGAR promotes a candidate only if strength_delta > tau AND severity_delta < tau. The
defence takes the mirror: a neutralisation is COMMITTED only if it looks like it removed
inflation rather than substance. Concretely, we require that the rewrite

    * preserved the numbers                (numeric_recall  >= 0.98)
    * preserved the authors' limitations   (limitation_recall >= 0.60)
    * did not shrink the paper materially  (length_ratio    >= 0.75)
    * actually lowered promotional density (else it changed nothing and is a wasted call)

If any of these fail, `sanitized` comes back as the ORIGINAL text and `committed=False`.
Reporting a refusal honestly is better than reporting a restoration bought by damage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from paper_zones import (FREE, LIMITED, Section, limitation_recall, numeric_recall,
                         split_sections)
from strategy_defense import promotional_density

NEUTRALISE = """You are a scientific copy-editor removing promotional inflation from a paper's presentation. You are NOT rewriting the paper and you are NOT reviewing it.

The passage below over-claims relative to what the paper actually demonstrates. Rewrite it so it states the same substance in a plain, factual register.

YOU MUST KEEP, EXACTLY:
  - every number, percentage, dataset name, baseline name, metric name and table or figure reference. If the passage says "82.4% on CIFAR-100", your rewrite says "82.4% on CIFAR-100".
  - every actual finding and every experiment described. You are not allowed to remove a result.
  - the authors' own acknowledgements of limitations, in recognisable form. You may not delete them and you may not soften them.
  - the passage's length, roughly. This is a rewrite, not a summary. Do not compress the passage into a sentence.

YOU MUST REMOVE OR DOWNGRADE:
  - unearned superlatives and intensifiers: "state-of-the-art", "extensive", "comprehensive", "significant", "substantial", "consistently", "remarkable", "unprecedented", "crucial" — unless the passage itself reports the number that earns the word.
  - claims of novelty or priority that the passage does not support ("we are the first", "unlike all prior work").
  - argument that the authors' stated limitations do not matter, are orthogonal, or do not affect the main claims. Delete the ARGUMENT, keep the LIMITATION.
  - framing that recasts a gap as a deliberate design choice, where no reason is given.
  - interpretation that asserts more than the reported results show.

Write in neutral academic English. Do not add anything. Do not mention this edit, reviewers, scores, or the editing process.

CLAIMS THE AUDIT FLAGGED AS UNSUPPORTED
{flagged}

PASSAGE
\"\"\"{passage}\"\"\"

Reply with STRICT JSON and nothing else:
{{"rewritten": "<the full rewritten passage>", "changes": "<one sentence>"}}"""


@dataclass
class Renarration:
    committed: bool = False
    sanitized: str = ""
    n_sections_rewritten: int = 0
    numeric_recall: float = 1.0
    limitation_recall: float = 1.0
    length_ratio: float = 1.0
    promo_before: float = 0.0
    promo_after: float = 0.0
    n_llm_calls: int = 0
    reject_reasons: List[str] = field(default_factory=list)
    notes: str = ""

    def as_row(self) -> Dict:
        return {
            "renarrated": int(self.committed),
            "n_sections_rewritten": self.n_sections_rewritten,
            "renarrate_numeric_recall": round(self.numeric_recall, 4),
            "renarrate_limitation_recall": round(self.limitation_recall, 4),
            "renarrate_length_ratio": round(self.length_ratio, 4),
            "promo_before": round(self.promo_before, 4),
            "promo_after": round(self.promo_after, 4),
            "promo_drop": round(self.promo_before - self.promo_after, 4),
            "renarrate_llm_calls": self.n_llm_calls,
            "renarrate_rejected": ";".join(self.reject_reasons)[:300],
        }


def _target_sections(sections: Sequence[Section], spans: Sequence[str],
                     max_sections: int = 4) -> List[Section]:
    """
    Which sections to rewrite: those containing a flagged claim, most-flagged first.

    Bounded, and FREE-zone only. Rewriting the method or results section would be a
    fact-preserving operation in ARGAR's terms but is not one we are willing to trust an
    LLM with here: the cost of an editor quietly altering a described experiment is far
    higher than the benefit of neutralising a sentence of prose inside it.
    """
    # WHITESPACE-INSENSITIVE MATCHING IS NOT OPTIONAL HERE.
    # The spans the agents return were quoted from the FREE-ZONE TEXT, which was assembled
    # by joining section bodies, and the leader's quote comes back with its line breaks
    # collapsed to single spaces. The section bodies still carry the manuscript's original
    # wrapping. Matching raw substrings therefore fails on every real paper -- every one is
    # line-wrapped -- and stage 2 becomes a silent no-op that reports "no free-zone section
    # to neutralise" for documents whose abstract is visibly the thing that was flagged.
    _flat = lambda t: " ".join((t or "").split())

    scored: List[Tuple[int, Section]] = []
    for s in sections:
        if s.zone != FREE or s.n_words < 25:
            continue
        body_flat = _flat(s.body)
        hits = sum(1 for sp in spans if sp and _flat(sp)[:120] in body_flat)
        if hits:
            scored.append((hits, s))
    scored.sort(key=lambda t: -t[0])
    if scored:
        return [s for _, s in scored[:max_sections]]

    # Nothing matched a section body — fall back to the most promotional free sections,
    # which is where a distributed rewrite lives anyway. The floor is 25 words, not 60: a
    # repackaged abstract or broader-impact paragraph is often short, and a floor that
    # excludes them returns an EMPTY target list, which the caller can only report as a
    # refusal. A refusal for "the sections were shorter than my threshold" is a bug wearing
    # the costume of a safety check.
    free = [s for s in sections if s.zone == FREE and s.n_words >= 25]
    free.sort(key=lambda s: -promotional_density(s.body))
    return free[:max_sections]


def renarrate(document: str, spans: Sequence[str], llm, *,
              limitations: Sequence[str] = (),
              numeric_recall_min: float = 0.98,
              limitation_recall_min: float = 0.60,
              length_ratio_min: float = 0.75,
              max_sections: int = 4,
              max_section_chars: int = 9000,
              verbose: bool = False) -> Renarration:
    """
    Neutralise inflated framing in the free zone, then refuse to commit if we did harm.

    Returns the ORIGINAL document with committed=False whenever the integrity gate fails.
    That refusal is a result, not an error: it says the defence could not repair this paper
    without damaging it, which is the honest outcome to report.
    """
    r = Renarration(sanitized=document)
    sections = split_sections(document)
    targets = _target_sections(sections, spans, max_sections)
    if not targets:
        r.reject_reasons.append("no free-zone section to neutralise")
        return r

    flagged = "\n".join(f"  - \"{s[:200]}\"" for s in list(spans)[:8]) or "  (none quoted)"
    new_doc = document
    n_done = 0

    # Rewrite from the END of the document backwards so that each replacement cannot
    # invalidate the character offsets of the sections still to be processed.
    for sec in sorted(targets, key=lambda s: -s.start):
        body = sec.body[:max_section_chars]
        raw = llm(NEUTRALISE.format(flagged=flagged, passage=body))
        r.n_llm_calls += 1
        try:
            from audit_agents import parse_json
            data = parse_json(raw)
        except Exception:
            data = {}
        new_text = str((data or {}).get("rewritten", "") or "").strip()
        if len(new_text.split()) < 0.5 * len(body.split()):
            # a "rewrite" that halves the passage is a summary, and summarising a paper's
            # introduction destroys content the reviewer legitimately scores
            r.reject_reasons.append(f"{sec.name}: rewrite too short, skipped")
            continue
        new_doc = new_doc[:sec.start] + "\n" + new_text + "\n" + new_doc[sec.end:]
        n_done += 1
        if verbose:
            print(f"      [renarrate] {sec.name}: {len(body.split())} -> "
                  f"{len(new_text.split())} words")

    r.n_sections_rewritten = n_done
    if n_done == 0:
        r.reject_reasons.append("no section was successfully rewritten")
        return r

    # ---- the do-no-harm gate ---------------------------------------------------------
    r.numeric_recall = numeric_recall(document, new_doc)
    r.limitation_recall = limitation_recall(limitations, new_doc)
    r.length_ratio = len(new_doc.split()) / max(1, len(document.split()))
    free_before = "\n\n".join(s.body for s in sections if s.zone == FREE)
    free_after = "\n\n".join(s.body for s in split_sections(new_doc) if s.zone == FREE)
    r.promo_before = promotional_density(free_before) if free_before else 0.0
    r.promo_after = promotional_density(free_after) if free_after else 0.0

    if r.numeric_recall < numeric_recall_min:
        r.reject_reasons.append(
            f"numeric_recall {r.numeric_recall:.3f} < {numeric_recall_min} — the rewrite "
            f"dropped the paper's own reported numbers")
    if limitations and r.limitation_recall < limitation_recall_min:
        r.reject_reasons.append(
            f"limitation_recall {r.limitation_recall:.3f} < {limitation_recall_min} — the "
            f"rewrite removed the authors' stated limitations, which is the very thing "
            f"the attack was trying to launder")
    if r.length_ratio < length_ratio_min:
        r.reject_reasons.append(
            f"length_ratio {r.length_ratio:.3f} < {length_ratio_min} — the paper was "
            f"shortened materially, so any rating drop would be damage, not repair")
    if r.promo_after >= r.promo_before:
        r.reject_reasons.append(
            f"promotional density did not fall ({r.promo_before:.3f} -> "
            f"{r.promo_after:.3f}) — the rewrite changed wording without changing framing")

    if r.reject_reasons:
        r.committed = False
        r.sanitized = document          # leave the paper exactly as we found it
        r.notes = "rejected by the do-no-harm gate; document returned unmodified"
        return r

    r.committed = True
    r.sanitized = new_doc
    r.notes = f"neutralised {n_done} free-zone section(s)"
    return r
