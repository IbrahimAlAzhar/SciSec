"""
btr_signatures.py — V4: "Breaking the Reviewer" attack templates, put INTO the tier-0
                        retrieval index so the RAG guardrail can see this family at all.

WHY THE EXISTING INDEX CANNOT SEE THIS ATTACK
=======================================================================================
`attack_memory` retrieves over char n-grams of KNOWN PAYLOADS. That works for insertion
because a payload is foreign text with literal substrings to match. BTR inserts nothing:
DeepWordBug and PuncAttack edit characters inside the authors' own words, TextFooler and
BertAttack swap single words for synonyms, StyleAdv re-writes sentences into a different
register. There is no payload, so retrieval has nothing to retrieve and the margin on a
BTR document sits wherever ordinary prose sits. Measured, not assumed — V3's perturbation
arm exists precisely because the insertion screen reported clean on every one of these.

WHAT GOES INTO THE INDEX INSTEAD
=======================================================================================
Not payloads — SIGNATURES. Each family leaves a characteristic surface, and a short
exemplar of that surface is a legitimate index entry: retrieval then asks "does any span
of this manuscript look like corrupted text / like a register shift?" rather than "does
any span match a known instruction?".

    btr_char_corruption   words with a character inserted, deleted, swapped or substituted
                          ("expreiments", "evaluatoin", "signficant") — DeepWordBug
    btr_punct_injection   punctuation driven inside and around words ("exper,iments",
                          "the results , show") — PuncAttack
    btr_synonym_swap      real words in collocations the domain never uses ("the
                          methodology exhibits considerable potency") — TextFooler,
                          BertAttack
    btr_style_bible       KJV register inside a methods section — StyleAdv(bible)
    btr_style_shakespeare Early Modern English — StyleAdv(shakespeare)
    btr_style_tweets      social-media register — StyleAdv(tweets)
    btr_style_lyrics      song-lyric register — StyleAdv(lyrics)
    btr_style_poetry      verse register — StyleAdv(poetry)

THE EXEMPLARS ARE SYNTHETIC AND THAT IS DELIBERATE
=======================================================================================
They are written here, not copied from the baseline's outputs. Copying BTR's own attacked
documents into the index would be training on the test set: retrieval would then match a
document to itself and the recall number would be a lookup, not a detection. Synthetic
exemplars of the SURFACE mean the index carries the family's shape and none of its
instances, so a BTR document the defence has never seen still retrieves against them.

LOCO
=======================================================================================
Every entry carries a `condition` naming its family, so `AttackMemory.without()` — the
existing LOCO machinery — removes one BTR family from the index and the generalisation
claim holds for this arm exactly as it does for the others. `--loco btr_style_bible`
tests whether the defence catches a style attack whose register it was never shown.
"""

from __future__ import annotations

import random
import re
from typing import Dict, Iterable, List, Sequence, Tuple

GROUP = "btr_perturbation"

# ---------------------------------------------------------------------------------
# The scientific sentences the signatures are derived from. Ordinary, unremarkable
# prose — the point is that the CORRUPTED versions differ from these only in surface.
# ---------------------------------------------------------------------------------
_BASE_SENTENCES = [
    "We evaluate the proposed method on three benchmark datasets and report the mean "
    "accuracy over five random seeds.",
    "The experiments demonstrate a significant improvement over the strongest published "
    "baseline under identical training conditions.",
    "Our analysis shows that the performance gain is consistent across model sizes and "
    "does not depend on the choice of optimizer.",
    "The proposed architecture reduces inference latency while maintaining comparable "
    "classification accuracy on the held-out test split.",
    "We conduct an ablation study to isolate the contribution of each component of the "
    "training objective.",
    "The results indicate that the model generalizes to the out-of-distribution "
    "evaluation set without additional fine-tuning.",
    "We describe the dataset construction procedure and the annotation protocol used by "
    "the three independent annotators.",
    "Statistical significance is assessed with a paired bootstrap test over one thousand "
    "resamples of the evaluation set.",
]

# ---------------------------------------------------------------------------------
# 1. CHARACTER CORRUPTION  (DeepWordBug: insert / delete / swap / substitute)
# ---------------------------------------------------------------------------------
_CHAR_EXEMPLARS = [
    "We evalaute the propsoed methdo on three benchmakr datasets and report the mena "
    "accuracy over five randmo seeds.",
    "The expreiments demonstarte a signficant improvemnet over the strongest publised "
    "baesline under identcial training condtions.",
    "Our analsyis shows that the performacne gian is consistetn across modle sizes and "
    "does not depnd on the choice of optmizer.",
    "The propsed architectrue reduces infernece latnecy while maintaning comparabel "
    "classificaton accuarcy on the held-out tset split.",
    "We conudct an ablatoin stdy to isolate the contribtuion of each compnoent of the "
    "trainng objectiev.",
    "The reuslts indciate that the modle generalizse to the out-of-distribtuion "
    "evaluaton set wihtout additonal fine-tunign.",
]

# ---------------------------------------------------------------------------------
# 2. PUNCTUATION INJECTION  (PuncAttack)
# ---------------------------------------------------------------------------------
_PUNCT_EXEMPLARS = [
    "We eval,uate the pro.posed method on three ben;chmark datasets and report the mean "
    "accuracy , over five random seeds .",
    "The exper,iments demon.strate a signi;ficant improvement over the strongest "
    "published baseline , under identical training conditions ..",
    "Our anal.ysis shows that the perfor,mance gain is con;sistent across model sizes , "
    "and does not depend on the choice of opti.mizer .",
    "The proposed archi,tecture reduces infer.ence latency ; while maintaining "
    "comparable classi;fication accuracy on the held-out test split ,.",
    "We con.duct an abl,ation study to isolate the contri;bution of each com.ponent of "
    "the training objective ,,",
]

# ---------------------------------------------------------------------------------
# 3. SYNONYM SUBSTITUTION  (TextFooler, BertAttack)
#
# Real words, correct grammar, wrong register. The tell is COLLOCATION: "considerable
# potency", "erects upon", "gauge the efficacy" are all fine English and none of them is
# how this literature writes. This is the weakest of the three signals and it is labelled
# as such everywhere it is reported.
# ---------------------------------------------------------------------------------
_SYNONYM_EXEMPLARS = [
    "We gauge the proffered methodology upon trio benchmark datasets and divulge the "
    "median exactness athwart five haphazard seeds.",
    "The trials manifest a weighty amelioration athwart the mightiest promulgated "
    "baseline beneath identical instruction circumstances.",
    "Our dissection exhibits that the execution windfall is unwavering athwart mannequin "
    "magnitudes and reposes not upon the election of optimizer.",
    "The proffered edifice abridges deduction tardiness whilst upholding commensurate "
    "categorization exactness upon the withheld trial partition.",
    "We prosecute an amputation inquiry to segregate the contribution of every ingredient "
    "of the pedagogy objective.",
]

# ---------------------------------------------------------------------------------
# 4. STYLE TRANSFER  (StyleAdv: bible / shakespeare / tweets / lyrics / poetry)
#
# These are the highest-precision signatures in the file. A methods section does not
# contain a King James verse, and nothing in ordinary scientific writing is mistaken for
# one — which is why the style arm carries a high weight over a term that is normally
# exactly zero.
# ---------------------------------------------------------------------------------
_STYLE_EXEMPLARS: Dict[str, List[str]] = {
    "btr_style_bible": [
        "And it came to pass that we did evaluate the method upon three datasets, and "
        "behold, the accuracy was made manifest unto us over five seeds.",
        "Verily I say unto thee, the experiments did shew forth a great improvement over "
        "the baseline, and it was good in our sight.",
        "And the Lord of the ablation spake, saying: isolate thou each component, that "
        "the contribution thereof be known unto all the people.",
    ],
    "btr_style_shakespeare": [
        "What method is this, that doth upon three datasets prove itself, and o'er five "
        "seeds its mean accuracy declare?",
        "Hark! the experiments do shew improvement most significant, whereto the baseline "
        "yieldeth as the night to morn.",
        "Nay, gentle reader, mark thou well: the model doth generalize, nor needeth it "
        "further tuning for to serve.",
    ],
    "btr_style_tweets": [
        "ok so we tested this on 3 datasets and honestly the numbers are pretty wild lol "
        "5 seeds and it just works",
        "not gonna lie the baseline got absolutely destroyed here, improvement is huge, "
        "same training setup and everything",
        "ablation study time!! turns out every single component actually matters, who "
        "would have guessed",
    ],
    "btr_style_lyrics": [
        "Oh we ran it on three datasets, and the numbers came back true, five little "
        "seeds and an average, that's all we had to do.",
        "The baseline it was standing tall, until our model came along, and now it's "
        "falling, falling, falling, and that's the end of that old song.",
        "Take my ablation, take my study, take each component one by one, and tell me "
        "what is left behind when all the cutting's done.",
    ],
    "btr_style_poetry": [
        "Three datasets, and five seeds sown; / the mean accuracy, patiently grown, / "
        "reported here as it was known.",
        "Improvement, significant and bright, / above the baseline's fading light, / "
        "under conditions kept aright.",
        "Each component, severed and weighed, / the contribution thus displayed, / a "
        "careful ablation, carefully made.",
    ],
}


def signature_texts() -> List[Tuple[str, str]]:
    """
    Every signature as (condition, text). The condition is the LOCO key.

    Kept as a plain function rather than a constant so a caller can extend it (a new
    StyleAdv register, say) without editing the index builder.
    """
    out: List[Tuple[str, str]] = []
    out += [("btr_char_corruption", t) for t in _CHAR_EXEMPLARS]
    out += [("btr_punct_injection", t) for t in _PUNCT_EXEMPLARS]
    out += [("btr_synonym_swap", t) for t in _SYNONYM_EXEMPLARS]
    for cond, texts in _STYLE_EXEMPLARS.items():
        out += [(cond, t) for t in texts]
    return out


CONDITIONS = sorted({c for c, _ in signature_texts()})

# Which BTR attack name maps to which signature conditions. Used by --loco so holding out
# "the DeepWordBug family" removes the right index entries rather than a guessed name.
ATTACK_TO_CONDITIONS: Dict[str, Tuple[str, ...]] = {
    "DeepWordBug": ("btr_char_corruption",),
    "PuncAttack": ("btr_punct_injection",),
    "TextFooler": ("btr_synonym_swap",),
    "BertAttack": ("btr_synonym_swap",),
    "StyleAdv": tuple(sorted(_STYLE_EXEMPLARS)),
}


def loco_conditions(attacks: Iterable[str]) -> List[str]:
    """
    Translate BTR attack names into the index conditions LOCO must remove.

    Passing `--loco StyleAdv` has to drop all FIVE style registers, not a condition named
    "StyleAdv" that does not exist in the index. Getting this wrong would leave the
    signatures in place and report a generalisation number that is really a lookup.
    """
    out: List[str] = []
    for a in attacks or ():
        a = str(a).strip()
        if a in ATTACK_TO_CONDITIONS:
            out.extend(ATTACK_TO_CONDITIONS[a])
        elif a in CONDITIONS:
            out.append(a)
        else:
            out.append(a)          # not ours — pass through to the other arms' LOCO
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def memory_entries(exclude: Sequence[str] = ()):
    """
    The signatures as `attack_memory.MemoryEntry` objects, ready to index.

    Imported lazily so this module stays importable (and testable) without numpy.
    """
    from attack_memory import MemoryEntry
    drop = set(exclude or ())
    return [MemoryEntry(text=t, condition=c, group=GROUP,
                        source_paper="synthetic", source_model="btr_signature")
            for c, t in signature_texts() if c not in drop]


def describe() -> str:
    from collections import Counter
    c = Counter(cond for cond, _ in signature_texts())
    lines = [f"[btr-rag] {sum(c.values())} signature exemplars across {len(c)} families "
             f"added to the tier-0 retrieval index:"]
    for cond in sorted(c):
        lines.append(f"[btr-rag]   {cond:<24} {c[cond]} exemplar(s)")
    lines.append("[btr-rag] These are SYNTHETIC surfaces, not copies of the baseline's "
                 "outputs — indexing its own attacked documents would make recall a "
                 "lookup rather than a detection.")
    return "\n".join(lines)


# ======================================================================================
# a corruption generator, for tests and for calibration only
# ======================================================================================
def corrupt(text: str, kind: str = "char", rate: float = 0.06, seed: int = 0) -> str:
    """
    Apply a BTR-style corruption to text. FOR TESTS AND CALIBRATION ONLY.

    This exists so `test_def.py` can assert the detectors fire on the surface they claim
    to detect, and so a calibration run can be sanity-checked without the baseline's
    outputs being present. It is NOT the attack: the real DeepWordBug chooses its edits by
    word-importance ranking against the victim model, and this chooses them at random. A
    detector tuned on this and evaluated on this would be measuring nothing, which is why
    every number in the report comes from the baseline's real outputs.
    """
    rng = random.Random(seed)
    words = re.split(r"(\s+)", text or "")
    idxs = [i for i, w in enumerate(words) if re.fullmatch(r"[A-Za-z]{5,}", w)]
    if not idxs:
        return text or ""
    n = max(1, int(len(idxs) * rate))
    for i in rng.sample(idxs, min(n, len(idxs))):
        w = words[i]
        j = rng.randrange(1, len(w) - 1)
        if kind == "punct":
            words[i] = w[:j] + rng.choice(",.;:") + w[j:]
        elif kind == "swap":
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        elif kind == "delete":
            words[i] = w[:j] + w[j + 1:]
        else:                                     # substitute
            words[i] = w[:j] + rng.choice("abcdefghijklmnopqrstuvwxyz") + w[j + 1:]
    return "".join(words)
