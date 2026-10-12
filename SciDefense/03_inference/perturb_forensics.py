"""
perturb_forensics.py — V4: the signals that actually separate BTR from a normal paper.

WHAT WAS WRONG WITH V3's PERTURBATION SCREEN
=======================================================================================
V3 scored a document mostly on `nonword_rate`: the fraction of words absent from a
vocabulary built from benign index papers. That is the right instinct and the wrong
statistic, because it cannot tell these two apart:

    "expreiments"   DeepWordBug corrupted "experiments"          -> ATTACK
    "perplexity"    a real term this particular corpus lacks     -> ORDINARY PAPER

Both are out-of-corpus. So the threshold has to sit high enough to tolerate every paper
that uses vocabulary the index papers happened not to use, and by then it is too high to
catch a light corruption rate. That is the same shape of failure as the insertion arm's
constant classifier, one level down.

THE FIX: REPAIRABILITY, NOT RARITY
=======================================================================================
A corrupted word is one edit away from a word the corpus uses. A genuinely novel
technical term is not.

    expreiments -> experiments   edit distance 1, target IS in the corpus   REPAIRABLE
    perplexity  -> ???           nothing in the corpus within 1 edit        NOVEL

So V4 splits `nonword_rate` into two disjoint rates, and only the first one is evidence:

    repairable_rate   out-of-corpus words that ARE within one edit of a corpus word
    novel_rate        out-of-corpus words that are not near anything

`repairable_rate` is near zero on ordinary papers (English is not that dense: most real
words have no in-corpus neighbour at distance 1) and rises directly with the corruption
budget. `novel_rate` is what a normal paper full of new terminology looks like, and it
carries a much smaller weight.

This is also what makes stage 2 possible for the first time on this arm — see repair.py.
A repairable token comes with its repair, so the finding is falsifiable in exactly the way
V3's grounded-span rule demanded of the insertion arm: the detector must be able to say
"this word, replaced by that word", not merely "something feels off".

STYLE FORENSICS, FOR StyleAdv
=======================================================================================
StyleAdv rewrites whole sentences into bible / shakespeare / tweets / lyrics / poetry
register. Character and word statistics barely move — every word is a real word, spelled
correctly. What moves is FUNCTION-WORD DISTRIBUTION, and it moves in a way that is
measurable WITHIN the document, with no reference corpus at all:

    build each sentence's profile over a fixed function-word / morphology feature set,
    take the document's own median profile as the reference,
    report how many sentences sit far from their own document.

A paper written by one set of authors in one sitting is stylistically homogeneous. Three
sentences of King James English inside a methods section are not, and the divergence is
enormous — this is the highest-precision signal in the file. Because the reference is the
document's own median, it needs no vocabulary, survives LOCO trivially, and cannot be
defeated by a domain the index papers never covered.

THE HONEST RANKING OF THESE SIGNALS
=======================================================================================
    style_divergence     near-perfect precision, fires only on StyleAdv
    repairable_rate      strong and falsifiable, fires on DeepWordBug
    intraword_punct      strong, fires on PuncAttack
    novel_rate           weak; a proxy for TextFooler/BertAttack and for nothing else
    collocation oddity   weakest. Synonym substitution leaves real words in correct
                         grammar, and this project does not have a language model of the
                         domain's collocations. Reported, weighted low, and NOT claimed
                         as a solution to that family.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from perturbation_defense import (STOPWORDS, _INTRAWORD_PUNCT, _PUNCT_ANOMALY, _TRIPLE,
                                  _WORD, _norm, _script_mixed)

_SENT = re.compile(r"(?<=[.!?])\s+")

# ======================================================================================
# 1. repairability — a deletion-neighbourhood index over the reference vocabulary
# ======================================================================================
def _deletes(word: str) -> Set[str]:
    """Every string obtained by deleting one character. The SymSpell trick."""
    return {word[:i] + word[i + 1:] for i in range(len(word))}


class RepairIndex:
    """
    Answers "is this token one edit from a word the corpus actually uses?" in O(len).

    A naive scan would compare every out-of-corpus token against every vocabulary entry:
    50k words x 300 suspicious tokens x 600 documents is 9 billion comparisons and the
    screen would cost more than the agents it is supposed to avoid. The deletion
    neighbourhood makes it a dictionary lookup: two words are within one insert / delete /
    substitute / (adjacent) transpose exactly when their deletion sets intersect or one
    contains the other's key.

    MEMORY: one entry per (vocabulary word x its length). A 50k-word vocabulary of ~8
    characters is ~400k short strings, which is tens of megabytes — acceptable, and built
    once per run rather than per document.
    """

    def __init__(self, vocab: Iterable[str], min_len: int = 4, max_len: int = 24):
        self.vocab: Set[str] = {w for w in vocab if min_len <= len(w) <= max_len}
        self.min_len = min_len
        self.index: Dict[str, List[str]] = defaultdict(list)
        for w in self.vocab:
            self.index[w].append(w)
            for d in _deletes(w):
                self.index[d].append(w)
        self.n_keys = len(self.index)

    def nearest(self, token: str, max_candidates: int = 40) -> Optional[str]:
        """
        The most plausible in-corpus source of `token`, or None.

        "Most plausible" = the longest candidate, because a corruption usually preserves
        length and the longest match is the least likely to be a coincidental short word.
        Returns None for tokens with no candidate at all — those are NOVEL, and calling
        them repairable would be exactly the false positive this module exists to avoid.
        """
        t = token
        if len(t) < self.min_len:
            return None
        if t in self.vocab:
            return None                       # not out-of-corpus at all
        seen: Set[str] = set()
        for key in (t, *_deletes(t)):
            for cand in self.index.get(key, ())[:max_candidates]:
                if cand != t:
                    seen.add(cand)
        if not seen:
            return None
        # Guard against a coincidence: require the candidate to share the first character
        # OR to be a transposition. Without it, short tokens acquire spurious "repairs"
        # ("cat" -> "at" -> "hat") and the repair rate on clean text drifts upward.
        good = [c for c in seen if c[0] == t[0] or sorted(c) == sorted(t)]
        pool = good or list(seen)
        return max(pool, key=len)


# ======================================================================================
# 2. style forensics — within-document register divergence
# ======================================================================================
# A deliberately SMALL, closed feature set: REGISTER MARKERS ONLY.
#
# WHAT IS DELIBERATELY ABSENT, AND WHY IT MATTERS
# An earlier draft of this set also counted digits and citations. That was a bug, and a
# revealing one: a methods sentence reading "we use a temperature of 0.05 and a batch size
# of 512" has three digits where the document's median sentence has none, so it scored as
# an enormous style outlier — and on a clean paper. Digits and citations are CONTENT, not
# register. A sentence is not out of register because it reports numbers.
#
# What survives is what a style transfer actually changes and a topic cannot: archaic
# morphology, interjections, contractions, address to the reader, and verse punctuation.
_STYLE_FEATURES: Tuple[Tuple[str, re.Pattern], ...] = (
    ("archaic_pronoun", re.compile(r"\b(thou|thee|thy|thine|ye|hath|doth|dost|shalt|"
                                   r"art|wert|unto|whence|whither|hither|thither)\b", re.I)),
    ("archaic_verb", re.compile(r"\b\w+(?:eth|est)\b", re.I)),
    ("biblical_frame", re.compile(r"\b(behold|verily|lo|amen|spake|begat|smote|"
                                  r"came to pass|saith)\b", re.I)),
    ("exclamation", re.compile(r"[!]")),
    ("first_person_sing", re.compile(r"\b(i|me|my|mine)\b", re.I)),
    ("contraction", re.compile(r"\b\w+'(?:s|t|re|ve|ll|d|m)\b", re.I)),
    ("interjection", re.compile(r"\b(oh|ah|hey|wow|lol|omg|honestly|literally|"
                                r"basically|ngl|tbh|yeah|nah|ok|okay)\b", re.I)),
    ("second_person", re.compile(r"\b(you|your|yours)\b", re.I)),
    ("conjunction_open", re.compile(r"^\s*(and|but|or|so|then|now)\b", re.I)),
    ("poetic_slash", re.compile(r"\s/\s")),
    ("emotive_adj", re.compile(r"\b(wild|crazy|beautiful|lovely|sweet|dear|cruel|"
                               r"bitter|weary|golden|silent)\b", re.I)),
    ("imperative_address", re.compile(r"\b(hark|mark thou|tell me|take my|come|listen)\b",
                                      re.I)),
)
_STYLE_NAMES = tuple(n for n, _ in _STYLE_FEATURES)


def style_profile(sentence: str) -> List[float]:
    """Per-100-word rates over the fixed feature set. Length-normalised on purpose."""
    n = max(6, len(sentence.split()))
    return [100.0 * len(rx.findall(sentence)) / n for _, rx in _STYLE_FEATURES]


def _median(xs: Sequence[float]) -> float:
    v = sorted(xs)
    if not v:
        return 0.0
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


@dataclass
class StyleReport:
    n_sentences: int = 0
    n_outliers: int = 0
    outlier_rate: float = 0.0
    max_divergence: float = 0.0
    mean_divergence: float = 0.0
    examples: List[str] = field(default_factory=list)
    top_features: List[str] = field(default_factory=list)


_HEADING = re.compile(r"^\s*#{1,6}\s*[^\n]*\n", re.M)


def style_report(text: str, z_threshold: float = 5.0, min_words: int = 6,
                 mad_floor: float = 1.5) -> StyleReport:
    """
    How many sentences do not sound like the rest of their own document?

    The reference is the document's MEDIAN profile and the scale is the median absolute
    deviation, not the mean and the standard deviation. That is not fussiness: a handful
    of extreme style-transferred sentences would drag a mean and inflate a standard
    deviation enough to hide themselves — the outliers would define the norm they are
    supposed to be outliers from. Median/MAD is the standard fix and it is exactly the
    situation it exists for.
    """
    r = StyleReport()
    # Section headings are stripped first. "## Discussion" glued to the front of the next
    # sentence changes that sentence's word count and its function-word rates for a reason
    # that has nothing to do with register, and it was enough to make the first sentence
    # of every section look like an outlier.
    body = _HEADING.sub("", text or "")
    sents = [s.strip() for s in _SENT.split(body) if len(s.split()) >= min_words]
    r.n_sentences = len(sents)
    if len(sents) < 12:
        # Too few sentences for a document-internal reference to mean anything. Report
        # nothing rather than a number computed from four sentences.
        return r

    profs = [style_profile(s) for s in sents]
    cols = list(zip(*profs))
    med = [_median(c) for c in cols]
    # MAD per feature, floored so a feature that is exactly constant (usually zero) does
    # not divide by zero and does not turn a single occurrence into an infinite z.
    #
    # THE FLOOR IS THE SENSITIVITY DIAL AND IT IS SET DELIBERATELY.
    # On a scientific paper the median of every register feature is 0, so the MAD is 0 and
    # ANY occurrence would otherwise be infinitely anomalous. Since the profile is a rate
    # per 100 words, a floor of 1.5 means one marker in a 30-word sentence (3.3 per 100)
    # gives z = 2.2 — not an outlier. A King James sentence carries several markers at
    # once ("behold", "verily", "unto", "thou", "-eth") and clears 5.0 comfortably. That
    # asymmetry is what keeps the false-positive rate on clean prose at zero.
    mad = [max(mad_floor, 1.4826 * _median([abs(x - m) for x in c]))
           for c, m in zip(cols, med)]

    divs, feat_hits = [], Counter()
    for s, p in zip(sents, profs):
        z = [(abs(v - m) / d) for v, m, d in zip(p, med, mad)]
        # The document-level divergence of one sentence is its LARGEST feature z, not the
        # sum: a style attack shows up as one or two features going far out, and summing
        # would dilute that across the twelve features that did not move.
        dv = max(z) if z else 0.0
        divs.append(dv)
        if dv >= z_threshold:
            r.n_outliers += 1
            feat_hits[_STYLE_NAMES[z.index(max(z))]] += 1
            if len(r.examples) < 5:
                r.examples.append(s[:200])
    r.outlier_rate = r.n_outliers / len(sents)
    r.max_divergence = max(divs) if divs else 0.0
    r.mean_divergence = sum(divs) / len(divs) if divs else 0.0
    r.top_features = [f"{k}x{v}" for k, v in feat_hits.most_common(4)]
    return r


# ======================================================================================
# 3. the V4 perturbation report
# ======================================================================================
@dataclass
class PerturbV4Report:
    score: float = 0.0
    # the two halves of V3's nonword_rate
    repairable_rate: float = 0.0
    novel_rate: float = 0.0
    n_words: int = 0
    n_repairable: int = 0
    n_novel: int = 0
    # surface anomalies
    intraword_punct_per_1k: float = 0.0
    punct_anomaly_per_1k: float = 0.0
    triple_char_per_1k: float = 0.0
    mixed_script_per_1k: float = 0.0
    # style
    style_outlier_rate: float = 0.0
    style_max_divergence: float = 0.0
    style_n_outliers: int = 0
    style_top_features: str = ""
    # retrieval against the BTR signature index (btr_signatures.py)
    signature_margin: Optional[float] = None
    signature_condition: str = ""
    # evidence
    repairs: List[Tuple[str, str]] = field(default_factory=list)   # (token, suggestion)
    novel_examples: List[str] = field(default_factory=list)
    style_examples: List[str] = field(default_factory=list)
    vocab_size: int = 0
    # per-family sub-scores, so the report can say WHICH family it thinks it saw
    family_scores: Dict[str, float] = field(default_factory=dict)

    def best_family(self) -> str:
        if not self.family_scores:
            return ""
        return max(self.family_scores.items(), key=lambda kv: kv[1])[0]

    def as_row(self) -> Dict:
        return {
            "perturb_score": round(self.score, 4),
            "perturb_repairable_rate": round(self.repairable_rate, 5),
            "perturb_novel_rate": round(self.novel_rate, 5),
            "perturb_n_repairable": self.n_repairable,
            "perturb_n_novel": self.n_novel,
            "perturb_n_words": self.n_words,
            "perturb_intraword_punct_per_1k": round(self.intraword_punct_per_1k, 3),
            "perturb_punct_anomaly_per_1k": round(self.punct_anomaly_per_1k, 3),
            "perturb_triple_char_per_1k": round(self.triple_char_per_1k, 3),
            "perturb_mixed_script_per_1k": round(self.mixed_script_per_1k, 3),
            "perturb_style_outlier_rate": round(self.style_outlier_rate, 4),
            "perturb_style_max_divergence": round(self.style_max_divergence, 2),
            "perturb_style_n_outliers": self.style_n_outliers,
            "perturb_style_features": self.style_top_features,
            "perturb_signature_margin": (None if self.signature_margin is None
                                         else round(self.signature_margin, 4)),
            "perturb_signature_condition": self.signature_condition,
            "perturb_family": self.best_family(),
            "perturb_repairs": " | ".join(f"{a}->{b}" for a, b in self.repairs[:10]),
            "perturb_examples": " | ".join(self.novel_examples[:8]),
        }

    def evidence_text(self) -> str:
        """
        What the agents are shown. Deliberately concrete: token-level repairs and verbatim
        sentences, never a bare score. An agent handed "the perturbation score is 2.4" can
        only agree with it; an agent handed "expreiments -> experiments" can check.
        """
        if self.vocab_size == 0:
            return ("  (no reference vocabulary was built, so the perturbation screen could "
                    "not run — this is 'NOT MEASURED', not 'clean')")
        lines = [
            f"  perturbation score {self.score:.2f}  (best-matching family: "
            f"{self.best_family() or 'none'})",
            f"  REPAIRABLE words {100*self.repairable_rate:.2f}% "
            f"({self.n_repairable} of {self.n_words}) — out-of-corpus words that are ONE "
            f"EDIT from a word this corpus does use. That is the fingerprint of "
            f"character-level editing; genuinely novel terminology has no such neighbour.",
            f"  NOVEL words {100*self.novel_rate:.2f}% ({self.n_novel}) — out-of-corpus "
            f"and near nothing. Ordinary papers have these; they are weak evidence.",
        ]
        if self.repairs:
            lines.append("  repairs the screen can name: "
                         + ", ".join(f'"{a}" -> "{b}"' for a, b in self.repairs[:8]))
        if self.intraword_punct_per_1k or self.punct_anomaly_per_1k:
            lines.append(f"  punctuation inside words {self.intraword_punct_per_1k:.2f}/1k, "
                         f"punctuation anomalies {self.punct_anomaly_per_1k:.2f}/1k")
        if self.mixed_script_per_1k:
            lines.append(f"  mixed-script tokens {self.mixed_script_per_1k:.2f}/1k "
                         f"(Latin letters swapped for Cyrillic/Greek look-alikes)")
        if self.style_n_outliers:
            lines.append(f"  STYLE: {self.style_n_outliers} of {int(self.style_outlier_rate and self.style_n_outliers/self.style_outlier_rate) or 0} "
                         f"sentences do not sound like the rest of THIS document "
                         f"(max divergence {self.style_max_divergence:.1f} MADs; "
                         f"{self.style_top_features or 'n/a'})")
            for s in self.style_examples[:3]:
                lines.append(f'      "{s}"')
        if self.signature_margin is not None:
            lines.append(f"  retrieval against the BTR signature index: margin "
                         f"{self.signature_margin:+.3f} toward "
                         f"{self.signature_condition or 'nothing'}")
        lines.append("  NOTE: a scientific paper legitimately contains novel terms and "
                     "unusual notation. What is measured is the RATE, and REPAIRABILITY "
                     "is what separates corruption from novelty.")
        return "\n".join(lines)


def perturbation_report_v4(text: str, vocab: Set[str],
                           repair_index: Optional[RepairIndex] = None,
                           memory=None,
                           max_repairs: int = 400) -> PerturbV4Report:
    """
    Score a document for BTR-style perturbation.

    `memory` is optional: when an AttackMemory carrying the btr_signatures entries is
    passed, its signed margin is added as one more piece of evidence. It is NOT the
    dominant term — the surface statistics are stronger and cheaper — but it is what makes
    the tier-0 RAG guardrail actually cover this family, and it is what LOCO removes.
    """
    r = PerturbV4Report(vocab_size=len(vocab))
    t = text or ""
    words = _WORD.findall(t)
    r.n_words = len(words)
    if r.n_words < 50:
        return r

    if vocab:
        idx = repair_index if repair_index is not None else RepairIndex(vocab)
        seen_rep, seen_nov = set(), set()
        checked = 0
        for w in words:
            nw = _norm(w)
            if len(nw) < 3 or nw in vocab or w.isupper():
                continue                      # acronyms are legitimately rare
            checked += 1
            if checked > max_repairs * 4:
                break
            src = idx.nearest(nw)
            if src is not None:
                r.n_repairable += 1
                if nw not in seen_rep and len(r.repairs) < 12:
                    seen_rep.add(nw)
                    r.repairs.append((w, src))
            else:
                r.n_novel += 1
                if nw not in seen_nov and len(r.novel_examples) < 12:
                    seen_nov.add(nw)
                    r.novel_examples.append(w)
        r.repairable_rate = r.n_repairable / r.n_words
        r.novel_rate = r.n_novel / r.n_words

    per1k = 1000.0 / r.n_words
    r.intraword_punct_per_1k = len(_INTRAWORD_PUNCT.findall(t)) * per1k
    r.punct_anomaly_per_1k = len(_PUNCT_ANOMALY.findall(t)) * per1k
    r.triple_char_per_1k = len(_TRIPLE.findall(t)) * per1k
    r.mixed_script_per_1k = sum(1 for tok in re.findall(r"\S+", t)
                                if _script_mixed(tok)) * per1k

    st = style_report(t)
    r.style_outlier_rate = st.outlier_rate
    r.style_max_divergence = st.max_divergence
    r.style_n_outliers = st.n_outliers
    r.style_top_features = ",".join(st.top_features)
    r.style_examples = st.examples

    if memory is not None:
        try:
            m, cond, _a, _b = memory.margin(t[:4000])
            r.signature_margin, r.signature_condition = m, cond
        except Exception:
            pass

    # ---- per-family sub-scores -------------------------------------------------------
    # Kept separate rather than folded into one number, because the report has to be able
    # to say WHICH family it believes it saw. A single scalar cannot, and "we detect BTR"
    # is a much weaker claim than "we detect DeepWordBug at 0.9 recall and TextFooler at
    # 0.3", which is the honest sentence.
    r.family_scores = {
        # 60x: a 2% repairable rate is already far outside anything clean text produces
        "char_corruption": 60.0 * r.repairable_rate + 0.30 * r.triple_char_per_1k,
        "punct_injection": 0.70 * r.intraword_punct_per_1k + 0.40 * r.punct_anomaly_per_1k,
        # COUNT, NOT RATE, and this correction matters more than it looks.
        # An earlier version used 14 x outlier_rate. On the 13-sentence fixture one
        # style-transferred sentence scored 1.08 — sensible. On a real 200-sentence
        # manuscript the SAME attack (StyleAdv rewrites a handful of sentences, not the
        # paper) would score 0.21 and vanish under the gate. The rate falls as the paper
        # gets longer, but the attack does not get weaker; three King James sentences are
        # three King James sentences whether the paper has 13 or 300.
        # Capped at 5 so a document that is entirely in another register does not produce
        # an unbounded score that swamps every other family's evidence.
        "style_transfer": (2.0 * min(r.style_n_outliers, 5)
                           + 0.10 * min(r.style_max_divergence, 30.0)),
        "homoglyph": 1.50 * r.mixed_script_per_1k,
        # the weak one, and weighted like it
        "synonym_swap": 8.0 * r.novel_rate,
    }
    r.score = max(r.family_scores.values()) if r.family_scores else 0.0
    # A signature retrieval hit adds a bounded amount. Bounded on purpose: if it could
    # dominate, a paper that happens to quote an attack (a paper about adversarial NLP,
    # say) would be flagged on retrieval alone, and that is the exact failure the
    # insertion arm already had to fix once.
    if r.signature_margin is not None and r.signature_margin > 0:
        r.score += min(0.60, 0.60 * r.signature_margin)
    return r


def build_repair_index(vocab: Iterable[str]) -> RepairIndex:
    idx = RepairIndex(vocab)
    print(f"[perturb-v4] repair index: {len(idx.vocab)} vocabulary words -> "
          f"{idx.n_keys} deletion keys")
    return idx
