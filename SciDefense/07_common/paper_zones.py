"""
paper_zones.py — section splitting and the FREE / LIMITED / FIXED zone model.

WHY THIS FILE EXISTS: ARGAR IS A DIFFERENT THREAT MODEL
=======================================================================================
The defence you already have assumes an INSERTED SPAN. Foreign text is pasted into the
Limitations section, retrieval finds it because it matches known payloads, and stage 2
DELETES it. Every metric in defense.py rests on that: `span_f1` scores against a gold
insert, `remove_spans` cuts text out, `restoration` measures the rating falling back.

ARGAR breaks all three assumptions at once:

  * NOTHING IS INSERTED. It rewrites the abstract, introduction, related work, discussion
    and conclusion. There is no foreign payload, so there is no gold span, so span_f1 is
    undefined and retrieval has nothing to match against.
  * NOTHING CAN BE DELETED. The attacked text IS the paper's abstract. Deleting it does
    not restore the paper, it destroys it. `remove_spans` is the wrong operation.
  * THE ATTACK IS DISTRIBUTED AND ADAPTIVE. Six closed-loop rounds against the reviewer's
    own feedback, promoted only through a direction gate. It is not one payload in one
    slot; it is the whole free zone moved together.

So the honest starting point is: the existing defence scores near ZERO recall on ARGAR,
by construction, and that is worth reporting rather than hiding. What follows is a SECOND
detection family that runs alongside the first, not a replacement for it.

THE INVARIANT ARGAR HANDS US
=======================================================================================
ARGAR's own zone rules are its weakness. It is forbidden from fabricating data:

    "EVERY number that appears in the original paper must still appear in your edited
     version" ... "may not claim any experiment, dataset, baseline or comparison that
     was not actually performed"

That means the EVIDENCE is frozen while the CLAIMS inflate. A repackaged paper says more
and more about how important and novel and comprehensive it is, over a fixed and
unchanged set of numbers. That ratio — promotional language per unit of evidence — is
measurable, and it is measurable WITHOUT the original paper, which matters because at
deployment you never have the original to compare against.

The zone model is what makes that measurable per-region rather than per-document, and it
is what tells stage 2 which remediation is legal:

    FIXED   author limitations, numbers, tables, equations
            -> an attack HERE is an insertion. Delete it. (the existing defence)
    FREE    abstract, introduction, related work, discussion, conclusion
            -> an attack HERE is repackaging. Deleting would destroy the paper.
               Neutralise the framing instead. (renarrate.py)
    LIMITED method, experiments, results
            -> rephrasing allowed, data not. Claims here must trace to numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

FREE = "free"
LIMITED = "limited"
FIXED = "fixed"

# Heading -> zone. Ordered longest-first at match time so "related work" beats "work".
# Taken from ARGAR's ZONE_RULES, with the ONE change this project requires: the authors'
# stated limitations are FIXED here. ARGAR may recontextualise them; the defence treats
# them as the ground truth an attacker is trying to launder, so they are never rewritten
# by our own remediation either.
ZONE_MAP: List[Tuple[str, str]] = [
    ("broader impact", FREE), ("impact statement", FREE), ("ethics statement", FREE),
    ("related work", FREE), ("background", FREE), ("introduction", FREE),
    ("abstract", FREE), ("conclusion", FREE), ("concluding remarks", FREE),
    ("discussion", FREE), ("future work", FREE), ("motivation", FREE),
    ("overview", FREE), ("summary", FREE), ("title", FREE),

    ("experimental setup", LIMITED), ("experimental results", LIMITED),
    ("experiments", LIMITED), ("evaluation", LIMITED), ("results", LIMITED),
    ("ablation", LIMITED), ("analysis", LIMITED), ("implementation", LIMITED),
    ("method", LIMITED), ("methods", LIMITED), ("methodology", LIMITED),
    ("approach", LIMITED), ("model", LIMITED), ("architecture", LIMITED),
    ("preliminaries", LIMITED), ("problem formulation", LIMITED), ("dataset", LIMITED),

    ("limitations", FIXED), ("limitation", FIXED), ("threats to validity", FIXED),
    ("appendix", FIXED), ("references", FIXED), ("acknowledgments", FIXED),
    ("acknowledgements", FIXED), ("proof", FIXED), ("theorem", FIXED),
]

_HEAD_RE = re.compile(
    r"^\s{0,3}(?:#{1,4}\s*|\*{0,2})"
    r"(?:(?:\d+(?:\.\d+)*)[.)]?\s+)?"          # 3.  /  3.1)
    r"([A-Z][A-Za-z0-9 &/,'\-]{2,60})"
    r"\*{0,2}\s*:?\s*$",
    re.M,
)

# A "number that counts as evidence": 82.4, 82.4%, 1,024, 3x, p<0.05. Bare small integers
# ("Section 3", "Figure 2", "we use 3 seeds") are excluded by the length/decimal rule
# because they are structural, not results.
_EVIDENCE_NUM = re.compile(
    r"(?<![\w.])(?:"
    r"\d+\.\d+%?"                     # 82.4 or 82.4%
    r"|\d{1,3}(?:,\d{3})+"            # 1,024
    r"|\d+%"                          # 45%
    r"|\d+\s*(?:x|×)\b"               # 3x
    r"|\d{3,}"                        # 1024
    r")(?![\w.])"
)

_EVIDENCE_TERMS = re.compile(
    r"\b(?:accuracy|f1|auc|bleu|rouge|precision|recall|perplexity|error rate|"
    r"significan\w+|p\s*[<=]\s*0?\.\d+|std|standard deviation|confidence interval|"
    r"ablation|baseline|benchmark|dataset|test set|held[- ]out|cross[- ]validat\w+|"
    r"seeds?|trials?|runs?|Table\s*\d|Figure\s*\d)\b", re.I)


@dataclass
class Section:
    name: str
    heading: str
    body: str
    zone: str
    start: int          # char offset of the BODY in the full document
    end: int

    @property
    def n_words(self) -> int:
        return len(self.body.split())


def zone_of(heading: str) -> str:
    """
    Which zone a heading belongs to. Unknown headings are LIMITED, not FREE.

    An unknown heading in a plain-text paper is usually a method or results subsection
    that the splitter did not recognise. Defaulting it to FREE would let the neutralising
    editor rewrite it, and rewriting an unrecognised section is exactly how a defence
    damages a paper it was supposed to protect. LIMITED is the safe default: readable,
    scoreable, not rewritable.
    """
    h = (heading or "").strip().lower().strip("#*: ")
    for key, z in sorted(ZONE_MAP, key=lambda kv: -len(kv[0])):
        if key in h:
            return z
    return LIMITED


def split_sections(document: str) -> List[Section]:
    """
    Split a plain-text manuscript into sections with zones and character offsets.

    Offsets are kept because stage 2 needs to map a finding back to a position in the
    ORIGINAL document; a section list without offsets cannot support span removal.
    """
    doc = document or ""
    heads = [(m.start(), m.end(), m.group(1).strip()) for m in _HEAD_RE.finditer(doc)]
    # drop headings that are really sentences: a heading line is short
    heads = [h for h in heads if len(h[2].split()) <= 9]

    out: List[Section] = []
    if not heads:
        return [Section("BODY", "BODY", doc, LIMITED, 0, len(doc))]

    if heads[0][0] > 200:
        out.append(Section("PREAMBLE", "PREAMBLE", doc[: heads[0][0]], FREE, 0,
                           heads[0][0]))
    for i, (hs, he, name) in enumerate(heads):
        body_start = he
        body_end = heads[i + 1][0] if i + 1 < len(heads) else len(doc)
        body = doc[body_start:body_end]
        if len(body.split()) < 5:
            continue
        out.append(Section(name.lower(), name, body, zone_of(name), body_start, body_end))
    return out or [Section("BODY", "BODY", doc, LIMITED, 0, len(doc))]


def zone_text(sections: Sequence[Section], zone: str) -> str:
    return "\n\n".join(s.body for s in sections if s.zone == zone)


def evidence_count(text: str) -> Dict[str, int]:
    """
    How much hard evidence a passage actually carries.

    This is the denominator of the repackaging signal. ARGAR cannot change it — its own
    rules forbid inventing a number, an experiment or a comparison — so any inflation of
    claim language has to show up as a change in the RATIO, not in this count.
    """
    t = text or ""
    return {
        "n_numbers": len(_EVIDENCE_NUM.findall(t)),
        "n_evidence_terms": len(_EVIDENCE_TERMS.findall(t)),
        "n_words": len(t.split()),
    }


def numeric_multiset(text: str) -> Dict[str, int]:
    """Every evidence-bearing number and how often it appears."""
    out: Dict[str, int] = {}
    for m in _EVIDENCE_NUM.findall(text or ""):
        k = m.strip().rstrip("%").replace(",", "")
        out[k] = out.get(k, 0) + 1
    return out


def numeric_recall(original: str, candidate: str) -> float:
    """
    Fraction of the original's evidence numbers still present in the candidate.

    Used in two directions. ARGAR uses it to police its OWN edits. The defence uses it to
    police the neutralising editor in renarrate.py: if our remediation drops the paper's
    numbers, we have damaged the manuscript, and that is a worse failure than missing the
    attack. Same measurement, opposite purpose.
    """
    a, b = numeric_multiset(original), numeric_multiset(candidate)
    if not a:
        return 1.0
    kept = sum(min(v, b.get(k, 0)) for k, v in a.items())
    return kept / sum(a.values())


def limitation_recall(limitations: Sequence[str], candidate: str,
                      min_overlap: float = 0.5) -> float:
    """
    Are the authors' own stated limitations still recognisably present?

    Content-word overlap rather than exact match, because legitimate reformatting moves
    wording. A limitation counts as surviving if half its content words are still there.
    """
    lims = [l for l in (limitations or []) if l and l.strip()]
    if not lims:
        return 1.0
    cand = set(re.findall(r"[a-z]{4,}", (candidate or "").lower()))
    kept = 0
    for l in lims:
        words = set(re.findall(r"[a-z]{4,}", l.lower()))
        if words and len(words & cand) / len(words) >= min_overlap:
            kept += 1
    return kept / len(lims)
