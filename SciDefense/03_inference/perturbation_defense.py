"""
perturbation_defense.py — the THIRD threat model: adversarial perturbation of the paper.

WHY A THIRD ARM
=======================================================================================
"Breaking the Reviewer" (BertAttack, DeepWordBug, PuncAttack, StyleAdv, TextFooler) is
neither of the attacks the defence already handles:

    INSERTION    a payload is pasted in            -> retrieval matches it, delete it
    REPACKAGING  the framing is rewritten          -> claims inflate over fixed evidence
    PERTURBATION the ORIGINAL WORDS ARE EDITED     -> neither of the above fires

Nothing is inserted, so retrieval has no payload to match. Nothing is promotional, so the
repackaging screen sits at zero. The attack is thousands of small character- and word-level
edits spread across the whole manuscript, chosen to move a classifier while keeping meaning
(the baseline's own `semantic_similarity` column records how well it succeeded).

So both existing screens report CLEAN on every one of these documents, by construction.
Same honesty rule as the ARGAR arm: measure it, report it, and add the arm that can see it.

WHAT THESE ATTACKS LEAVE BEHIND
=======================================================================================
Each family has a different tell, and they are cheap to measure without a reference copy:

  DeepWordBug   character insert/delete/swap/substitute -> NON-WORDS ("expreiments",
                "evaluatoin"). The single strongest signal in this family.
  PuncAttack    punctuation inserted inside and around words -> "exper,iments", " ,",
                doubled marks, space-before-punctuation.
  BertAttack /  synonym substitution by a masked LM -> real words, but words that do not
  TextFooler    occur anywhere else in a corpus of the same kind of papers, and unusual
                collocations. Harder, and honestly the weakest of the three signals.
  StyleAdv      style transfer -> register shift, caught partly by the same rare-word rate
                and partly by the VOICE agent.

THE VOCABULARY IS BUILT FROM YOUR OWN CORPUS, NOT A DICTIONARY
=======================================================================================
A dictionary is the wrong reference for scientific text: "ResNet", "CIFAR", "perplexity"
and "hyperparameter" are all out-of-dictionary and all perfectly normal here. Flagging
them would make every paper look attacked.

So the vocabulary is built from the BENIGN text of the INDEX papers — the same disjoint
split the attack memory uses. A token is suspicious when it appears in the document under
audit but nowhere in a hundred papers of the same kind. That makes the reference
domain-correct and keeps the leakage guarantee: eval papers never contribute to it.

WHAT THIS ARM CANNOT DO
=======================================================================================
Stage 2 has no honest repair here. You cannot delete the attack (it is distributed over
every paragraph) and you cannot neutralise framing (the framing was not touched).
Reconstructing the authors' original wording would mean rewriting their whole manuscript
from a corrupted copy, which is a much larger claim than this project can support. So the
perturbation arm reports DETECTION ONLY, and `sanitize_and_rescore` records
mode="detect_only" rather than manufacturing a restoration number.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

_WORD = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")


def _norm(w: str) -> str:
    """
    Normalise a token for vocabulary lookup.

    The word pattern keeps internal hyphens and apostrophes (they are part of real terms
    like "state-of-the-art" and "authors'"), but it also captures the TRAILING one in
    "ResNet-50" and "CIFAR-100", where the digits are dropped and a bare hyphen is left
    behind. Without stripping it, "resnet-" and "resnet" are different vocabulary types,
    domain terms fail to match their own corpus, and the out-of-corpus rate is inflated on
    every real paper -- turning the perturbation screen into a false-positive machine.
    """
    return w.strip("-'").lower()
_TOKEN = re.compile(r"\S+")

# punctuation sitting INSIDE an alphabetic word: "exper,iments", "eval.uation"
_INTRAWORD_PUNCT = re.compile(r"[A-Za-z]{2}[,;:!\"#$%&*+/=?@^_`|~][A-Za-z]{2}")
# a space before a closing mark, or two marks in a row: " ," / ",," / " ."
_PUNCT_ANOMALY = re.compile(r"\s[,.;:!?]|[,;:!?]{2,}|[,.;:!?]\s*[,.;:!?]")
# the same letter three or more times: "commmittee"
_TRIPLE = re.compile(r"([A-Za-z])\1{2,}")

STOPWORDS = set(
    "the a an and or but if of to in on for with without by from as at is are was were be "
    "been being this that these those it its we our they their he she his her you your not "
    "no more most other such than then there here when where which who whom while can could "
    "may might must shall should will would do does did done have has had".split())


def build_vocabulary(texts: Iterable[str], min_count: int = 2) -> Set[str]:
    """
    The reference vocabulary, from the benign INDEX papers.

    `min_count >= 2` on purpose: a word appearing exactly once across the whole corpus is
    as likely to be an OCR artifact as a real term, and admitting those into the reference
    would let genuine noise mask an attack. Stopwords are added unconditionally so a short
    corpus cannot accidentally make "whom" look adversarial.
    """
    c: Counter = Counter()
    for t in texts:
        c.update(_norm(w) for w in _WORD.findall(t or "") if len(_norm(w)) >= 3)
    vocab = {w for w, n in c.items() if n >= min_count} | STOPWORDS
    print(f"[perturb] vocabulary: {len(vocab)} word types from {c.total() if hasattr(c,'total') else sum(c.values())} "
          f"tokens of benign index text (min_count={min_count})")
    return vocab


def save_vocabulary(vocab: Set[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(vocab)))


def load_vocabulary(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def _script_mixed(tok: str) -> bool:
    """A token mixing Latin with Cyrillic/Greek look-alikes — classic homoglyph swapping."""
    scripts = set()
    for ch in tok:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        scripts.add(name.split()[0])
        if len(scripts) > 1:
            return True
    return False


@dataclass
class PerturbationReport:
    score: float = 0.0
    nonword_rate: float = 0.0
    intraword_punct_per_1k: float = 0.0
    punct_anomaly_per_1k: float = 0.0
    triple_char_per_1k: float = 0.0
    mixed_script_per_1k: float = 0.0
    n_words: int = 0
    n_nonwords: int = 0
    examples: List[str] = field(default_factory=list)
    vocab_size: int = 0

    def as_row(self) -> Dict:
        return {
            "perturb_score": round(self.score, 4),
            "perturb_nonword_rate": round(self.nonword_rate, 5),
            "perturb_intraword_punct_per_1k": round(self.intraword_punct_per_1k, 3),
            "perturb_punct_anomaly_per_1k": round(self.punct_anomaly_per_1k, 3),
            "perturb_triple_char_per_1k": round(self.triple_char_per_1k, 3),
            "perturb_mixed_script_per_1k": round(self.mixed_script_per_1k, 3),
            "perturb_n_words": self.n_words,
            "perturb_n_nonwords": self.n_nonwords,
            "perturb_examples": " | ".join(self.examples[:8]),
        }

    def evidence_text(self) -> str:
        if self.vocab_size == 0:
            return ("  (no reference vocabulary was built, so the perturbation screen could "
                    "not run — this is 'not measured', NOT 'clean')")
        lines = [
            f"  perturbation score {self.score:.2f}",
            f"  out-of-corpus word rate {100*self.nonword_rate:.2f}% "
            f"({self.n_nonwords} of {self.n_words} words are absent from a "
            f"{self.vocab_size}-type vocabulary built from benign papers of the same kind)",
            f"  punctuation inside words: {self.intraword_punct_per_1k:.2f} per 1k words; "
            f"punctuation anomalies: {self.punct_anomaly_per_1k:.2f} per 1k",
            f"  tripled characters: {self.triple_char_per_1k:.2f} per 1k; "
            f"mixed-script tokens: {self.mixed_script_per_1k:.2f} per 1k",
        ]
        if self.examples:
            lines.append("  suspicious tokens (verbatim): "
                         + ", ".join(f'"{e}"' for e in self.examples[:10]))
        lines.append("  NOTE: a scientific paper legitimately contains novel terms. What is "
                     "measured here is the RATE, against a corpus of the same kind of paper.")
        return "\n".join(lines)


def perturbation_report(text: str, vocab: Set[str]) -> PerturbationReport:
    """
    Score a document for character- and word-level adversarial perturbation.

    Returns a report whose `score` is non-negative and on its own scale — it is not
    comparable to the retrieval margin or the repackaging score, and DefenceConfig keeps
    separate thresholds for it.
    """
    r = PerturbationReport(vocab_size=len(vocab))
    t = text or ""
    words = _WORD.findall(t)
    r.n_words = len(words)
    if r.n_words < 50:
        return r

    if vocab:
        # w.isupper() exempts acronyms (LSTM, GAN, NLP): they are legitimately rare in a
        # word-frequency sense and flagging them would penalise technical writing.
        bad = [w for w in words
               if len(_norm(w)) >= 3 and _norm(w) not in vocab and not w.isupper()]
        r.n_nonwords = len(bad)
        r.nonword_rate = len(bad) / r.n_words
        seen: Set[str] = set()
        for w in bad:
            if _norm(w) not in seen:
                seen.add(_norm(w))
                r.examples.append(w)
            if len(r.examples) >= 12:
                break

    per1k = 1000.0 / r.n_words
    r.intraword_punct_per_1k = len(_INTRAWORD_PUNCT.findall(t)) * per1k
    r.punct_anomaly_per_1k = len(_PUNCT_ANOMALY.findall(t)) * per1k
    r.triple_char_per_1k = len(_TRIPLE.findall(t)) * per1k
    r.mixed_script_per_1k = sum(1 for tok in _TOKEN.findall(t) if _script_mixed(tok)) * per1k

    # Weights, and why they are ordered this way:
    #   nonword rate dominates -- it is the direct fingerprint of character-level editing
    #     and of synonym substitution pulling in words the domain never uses. Scaled by 40
    #     so a 5% rate contributes 2.0, which is the "clearly attacked" region.
    #   punctuation terms are next: PuncAttack is otherwise invisible to the word-level
    #     signal, and clean text has almost none of these.
    #   mixed script is rare but nearly decisive when it appears, so it carries a high
    #     per-unit weight over a term that is normally exactly zero.
    r.score = (40.0 * r.nonword_rate
               + 0.60 * r.intraword_punct_per_1k
               + 0.35 * r.punct_anomaly_per_1k
               + 0.40 * r.triple_char_per_1k
               + 1.50 * r.mixed_script_per_1k)
    return r
