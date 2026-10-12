"""
baseline_adapters.py — read every baseline's wide CSV into the defence's DocRecord.

THE SHAPE PROBLEM
=======================================================================================
The defence was built against `metrics_detail.csv`: LONG format, one row per
(paper, condition), with `insert`, `base_rating`, `rating`, `asr` as plain columns.

Every baseline here is WIDE instead: one row per PAPER, with each attack condition folded
into a family of suffixed columns. Four different naming schemes, and — more importantly —
three different THREAT MODELS and three different levels of what they actually give you.
That last point decides what the defence can even measure, so it is worth stating per
baseline before any code runs:

  ARGAR          argar_attacked_text__<model>            FULL attacked text.  REPACKAGING.
                 No payload and no gold span, because nothing was inserted. span_f1 is
                 undefined here and is deliberately NOT reported.

  FLASH          flash_attack_<cond>_payload             THE PAYLOAD ITSELF.  INSERTION.
                 This is the best-instrumented of the four: the payload is the gold span,
                 so the attacked document is rebuilt the same way the defence's own
                 benchmark builds one, and span precision/recall/F1 are all well defined.

  BTR            btr_<Method>_attacked_text              FULL attacked text.  PERTURBATION.
                 The original words were edited in place. Neither existing screen can see
                 it (see perturbation_defense.py), and there is no honest stage-2 repair,
                 so this arm is detection-only.

  PaperGuard     attack_reviews.csv :: payload_text        THE PAYLOAD.  INSERTION.
                 NOT in the wide attack_papers_<model>.csv (which carries only the produced
                 REVIEW), but in the LONG side-files the same run writes:
                     attack/attack_reviews.csv   one row per (paper, condition), with
                                                 payload_text, category, payload_name,
                                                 injection_position, rating, is_attack
                     attack/attack_pairs.csv     one row per attacked pair, with
                                                 clean_rating, attacked_rating,
                                                 delta_rating, asr, flip
                 So PaperGuard is the SECOND baseline with a gold span, alongside FLASH,
                 and it is richer than FLASH in one respect: `injection_position` records
                 WHERE the payload went, so the document is rebuilt at the real slot rather
                 than at an assumed one. 13 conditions, per-condition recall.

WHY REBUILDING THE FLASH DOCUMENT IS BETTER THAN TAKING A TEXT COLUMN
=======================================================================================
For FLASH the payload is inserted into the benign paper by `align`/`runner.build_anchor`,
the SAME code path the insertion benchmark used. That guarantees the benign and attacked
documents differ ONLY by the payload — not by tokenisation, truncation or formatting — so
a classifier cannot separate them on an artifact of how each was assembled. It also gives
stage 2 an exact gold span for free.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from def_dataset import CLEAN_CONDITION, DocRecord, read_anchor, read_record

PAPERGUARD_NOTE = """
PaperGuard needs its LONG side-files, not the wide attack_papers_<model>.csv.

The wide file carries pg_<model>_<cond>_review -- the REVIEW the attacked paper produced --
which is not something the defence can screen; it reads manuscripts. The payload lives in
the long files the same run writes, next to it:

    <.../outputs/<model>/attack/attack_reviews.csv   payload_text  <-- THE PAYLOAD
    <.../outputs/<model>/attack/attack_pairs.csv     clean/attacked ratings, asr, flip

Point --baseline-dir at that attack/ folder (or pass --pg-reviews-csv / --pg-pairs-csv
explicitly). The benign manuscripts still come from --baseline-csv or --benign-csv.
"""


# ======================================================================================
# helpers
# ======================================================================================
def _num(v) -> Optional[float]:
    """A float, or None. NaN, '', 'nan' and 'None' all become None, never 0.0."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _int01(v) -> Optional[int]:
    """ASR / success flags arrive as 1/0, True/False, 'True', 1.0 — normalise all of them."""
    if isinstance(v, bool):
        return int(v)
    s = str(v).strip().lower()
    if s in ("true", "yes"):
        return 1
    if s in ("false", "no"):
        return 0
    f = _num(v)
    return None if f is None else int(f >= 0.5)


def _text(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = str(v)
    return "" if s.strip().lower() in ("nan", "none", "null") else s


def _strategies(v) -> str:
    """`accepted_strategies` may be a JSON list, a Python list literal, or a plain string."""
    s = _text(v).strip()
    if not s:
        return ""
    for parse in (json.loads, ast.literal_eval):
        try:
            out = parse(s)
            if isinstance(out, (list, tuple)):
                return "+".join(str(x) for x in out[:3])
        except Exception:
            pass
    return s[:80]


def _paper_id(row, i: int, id_col: str) -> str:
    v = _text(row.get(id_col)) or _text(row.get("id")) or _text(row.get("pg_paper_id"))
    return v or f"row{i:04d}"


# ======================================================================================
# per-baseline schemas
# ======================================================================================
@dataclass
class BaselineSpec:
    name: str
    threat: str                       # insertion | repackaging | perturbation
    conditions: Tuple[str, ...]
    # column templates; {model} and {cond} are filled in
    attacked_text: Optional[str] = None
    payload: Optional[str] = None
    clean_rating: Optional[str] = None
    attacked_rating: Optional[str] = None
    delta_rating: Optional[str] = None
    asr: Optional[str] = None
    flip: Optional[str] = None
    condition_label: Optional[str] = None      # ARGAR: the strategies actually accepted
    status: Optional[str] = None
    note: str = ""


SPECS: Dict[str, BaselineSpec] = {
    # ---------------------------------------------------------------- ARGAR
    "argar": BaselineSpec(
        name="argar", threat="repackaging",
        conditions=("",),                       # one attacked version per paper, not per condition
        attacked_text="argar_attacked_text__{model}",
        clean_rating="argar_baseline_avg_score__{model}",
        attacked_rating="argar_attacked_avg_score__{model}",
        delta_rating="argar_delta_score__{model}",
        asr="argar_attack_success__{model}",
        condition_label="argar_accepted_strategies__{model}",
        status="argar_best_round__{model}",
        note="ratings are means over N independent reviews, not a single rating — still "
             "this reviewer's own baseline and attacked scores, so d_rating and "
             "restoration keep their meaning and are simply less noisy",
    ),
    # ---------------------------------------------------------------- FLASH
    "flash": BaselineSpec(
        name="flash", threat="insertion",
        conditions=("rating_pressure", "decision_pressure", "limitation_suppression",
                    "decoy_weaknesses"),
        payload="flash_attack_{cond}_payload",
        attacked_rating="flash_attack_{cond}_rating",
        delta_rating="flash_attack_{cond}_delta_rating",
        asr="flash_attack_{cond}_success",
        flip="flash_attack_{cond}_flip",
        note="no clean-rating column: the baseline reports rating and delta_rating, so "
             "clean = rating - delta_rating, computed per row",
    ),
    # ---------------------------------------------------------------- Breaking the Reviewer
    "btr": BaselineSpec(
        name="btr", threat="perturbation",
        conditions=("BertAttack", "DeepWordBug", "PuncAttack", "StyleAdv", "TextFooler"),
        attacked_text="btr_{cond}_attacked_text",
        clean_rating="btr_{cond}_rating_clean",
        attacked_rating="btr_{cond}_rating_attacked",
        delta_rating="btr_{cond}_delta_rating",
        asr="btr_{cond}_asr",
        flip="btr_{cond}_flipped",
        note="detection only: the edits are spread across the whole manuscript, so there "
             "is nothing to delete and no framing to neutralise",
    ),
    # ---------------------------------------------------------------- PaperGuard
    # PaperGuard is loaded by load_paperguard() from its LONG side-files, not by the wide
    # column walker -- the payload is in attack_reviews.csv, not in attack_papers_*.csv.
    # This spec is kept for the threat label and for the condition list.
    "paperguard": BaselineSpec(
        name="paperguard", threat="insertion",
        conditions=("direct", "fewshot", "alignment", "role_play", "tone", "oblique",
                    "contextual", "authority", "stepwise_multi", "meta_then_command",
                    "code_switch", "camouflaged", "obfuscated"),
        attacked_text=None,                    # <-- NOT PRESENT. see PAPERGUARD_NOTE
        payload=None,
        clean_rating="pg_{model}_clean_rating",
        attacked_rating="pg_{model}_{cond}_attacked_rating",
        delta_rating="pg_{model}_{cond}_delta_rating",
        asr="pg_{model}_{cond}_asr",
        flip="pg_{model}_{cond}_flip",
        note="loaded from attack_reviews.csv (payload_text) + attack_pairs.csv; the "
             "payload is a GOLD SPAN and injection_position gives the real slot",
    ),
}

# the model key as it appears INSIDE the column names of each baseline
MODEL_KEYS: Dict[str, Dict[str, str]] = {
    "argar":      {"gemma3-27b-it": "gemma3_27b", "gpt-4o-mini": "gpt_4o_mini",
                   "mistral-small-3.1-24b": "mistral_small_24b", "qwen3-8b": "qwen3_8b"},
    "paperguard": {"gemma3-27b-it": "gemma3_27b", "gpt-4o-mini": "gpt_4o_mini",
                   "mistral-small-3.1-24b": "mistral_small_3_1_24b", "qwen3-8b": "qwen3_8b"},
    "flash":      {"gemma3-27b-it": "gemma3_27b", "gpt-4o-mini": "gpt_4o_mini",
                   "mistral-small-3.1-24b": "mistral_small_24b", "qwen3-8b": "qwen3_8b"},
    "btr":        {"gemma3-27b-it": "gemma3_27b", "gpt-4o-mini": "gpt_4o_mini",
                   "mistral-small-3.1-24b": "mistral_small_3_1", "qwen3-8b": "qwen3_8b"},
}


def model_key(baseline: str, model: str) -> str:
    return MODEL_KEYS.get(baseline, {}).get(model, model.replace("-", "_").replace(".", "_"))


def _col(tmpl: Optional[str], model: str, cond: str) -> Optional[str]:
    return None if not tmpl else tmpl.format(model=model, cond=cond)


def _get(row, col: Optional[str]):
    return None if not col else row.get(col)


# ======================================================================================
# the loader
# ======================================================================================
def load_baseline(csv_path: str, baseline: str, model: str, args,
                  *, conditions: Optional[Sequence[str]] = None,
                  max_papers: int = 10 ** 9,
                  attacked_text_col: Optional[str] = None,
                  payload_col: Optional[str] = None,
                  benign_per_paper: bool = True) -> List[DocRecord]:
    """
    Read one baseline CSV into paired benign / attacked DocRecords.

    PAIRED, exactly as the insertion benchmark is: each paper contributes ONE benign
    document (its own unmodified text) and one attacked document per condition. The classes
    stay balanced at the paper level and the content is held constant across the pair, so
    accuracy cannot be inflated by topic differences between the classes.
    """
    spec = SPECS.get(baseline)
    if spec is None:
        raise SystemExit(f"unknown baseline {baseline!r}; known: {sorted(SPECS)}")
    if not os.path.exists(csv_path):
        raise SystemExit(f"{baseline}: input CSV not found: {csv_path}")

    mkey = model_key(baseline, model)
    df = pd.read_csv(csv_path, low_memory=False).reset_index(drop=True)
    conds = list(conditions) if conditions else list(spec.conditions)

    # ---- resolve the text source, and refuse clearly if there is not one --------------
    text_tmpl = attacked_text_col or spec.attacked_text
    pay_tmpl = payload_col or spec.payload
    if not text_tmpl and not pay_tmpl:
        msg = (f"\n{'='*86}\n{baseline}: no attacked-text or payload column is available.\n"
               f"{'='*86}\n")
        if baseline == "paperguard":
            msg += PAPERGUARD_NOTE
        msg += (f"\nColumns in {os.path.basename(csv_path)} that look related:\n  "
                + "\n  ".join([c for c in df.columns
                               if any(k in c.lower() for k in
                                      ("text", "payload", "insert", "attacked"))][:25])
                + "\n\nPass --attacked-text-col or --payload-col to name the right one.\n")
        raise SystemExit(msg)

    # ---- verify the columns exist BEFORE spending an hour of GPU ---------------------
    missing_conds = []
    for c in conds:
        probe = _col(text_tmpl or pay_tmpl, mkey, c)
        if probe and probe not in df.columns:
            missing_conds.append((c, probe))
    if len(missing_conds) == len(conds):
        raise SystemExit(
            f"{baseline}/{model}: none of the expected columns exist. Tried e.g. "
            f"{missing_conds[0][1]!r}. Check the model key — this baseline spells "
            f"{model!r} as {mkey!r} inside its column names. Available columns starting "
            f"with the right prefix:\n  "
            + "\n  ".join([c for c in df.columns if c.startswith(baseline[:3])][:20]))
    for c, probe in missing_conds:
        print(f"[{baseline}] condition {c!r}: no column {probe!r} — skipped")
    conds = [c for c in conds if (c, _col(text_tmpl or pay_tmpl, mkey, c)) not in missing_conds]

    out: List[DocRecord] = []
    n_papers = 0
    n_skipped_empty = 0

    for i in range(len(df)):
        if n_papers >= max_papers:
            break
        rec = read_record(df, i, args)
        row = df.iloc[i]
        pid = _paper_id(row, i, args.id_column)
        benign_text = _text(rec.get(args.text_column))
        if len(benign_text.split()) < 100:
            n_skipped_empty += 1
            continue

        anchor = read_anchor(rec, args) if pay_tmpl else None
        made_any = False

        for cond in conds:
            atk_text, insert = "", ""

            if pay_tmpl:
                # FLASH: rebuild the document the SAME way the benchmark does, so benign
                # and attacked differ only by the payload -- and the payload is the gold
                # span, which makes span P/R/F1 well defined for this baseline.
                payload = _text(_get(row, _col(pay_tmpl, mkey, cond))).strip()
                if not payload or anchor is None:
                    continue
                try:
                    atk_text = anchor.render(payload, args.slot)[: args.max_chars]
                    insert = payload
                except Exception as e:
                    print(f"  [skip] {pid}/{cond}: could not render payload: "
                          f"{type(e).__name__}: {e}")
                    continue
            else:
                atk_text = _text(_get(row, _col(text_tmpl, mkey, cond))).strip()
                if len(atk_text.split()) < 100:
                    continue

            r0 = _num(_get(row, _col(spec.clean_rating, mkey, cond)))
            ra = _num(_get(row, _col(spec.attacked_rating, mkey, cond)))
            dr = _num(_get(row, _col(spec.delta_rating, mkey, cond)))
            # FLASH reports rating and delta but no clean column: recover it rather than
            # leaving base_rating None, which would make restoration undefined for the
            # entire baseline.
            if r0 is None and ra is not None and dr is not None:
                r0 = ra - dr
            if dr is None and ra is not None and r0 is not None:
                dr = ra - r0

            asr = _int01(_get(row, _col(spec.asr, mkey, cond)))
            if asr is None and dr is not None:
                asr = int(dr >= 1.0)

            label_cond = cond or baseline
            if spec.condition_label:
                s = _strategies(_get(row, _col(spec.condition_label, mkey, cond)))
                if s:
                    label_cond = s
            # ARGAR: a campaign whose best round is the baseline promoted NOTHING, so its
            # "attacked" text equals the original. Counting it as a positive would score
            # the defence against a paper nobody modified.
            if spec.status:
                best = _text(_get(row, _col(spec.status, mkey, cond))).strip().lower()
                if best in ("baseline", "", "none"):
                    print(f"  [{pid}] no round promoted — benign arm only")
                    continue

            out.append(DocRecord(
                doc_id=f"{pid}|{baseline}:{label_cond}", paper_id=pid, row_index=i,
                label=1, condition=f"{baseline}:{label_cond}", group=baseline,
                document=atk_text, insert=insert,
                base_rating=r0, attacked_rating=ra, d_rating=dr, asr=asr,
                slot=args.slot, max_chars=args.max_chars))
            made_any = True

        if benign_per_paper:
            # One benign document per paper regardless of how many conditions fired, so a
            # paper with 13 conditions cannot swamp the benign class and make accuracy a
            # function of how many attacks the baseline happened to run.
            r0b = _num(_get(row, _col(spec.clean_rating, mkey, conds[0] if conds else "")))
            out.append(DocRecord(
                doc_id=f"{pid}|{baseline}:clean", paper_id=pid, row_index=i, label=0,
                condition=CLEAN_CONDITION, group="control",
                document=benign_text[: args.max_chars], insert="",
                base_rating=r0b, attacked_rating=r0b, d_rating=0.0, asr=0,
                slot=args.slot, max_chars=args.max_chars))
        if made_any or benign_per_paper:
            n_papers += 1

    n_att = sum(1 for r in out if r.label == 1)
    print(f"[{baseline}/{model}] {n_papers} papers -> {len(out)} documents "
          f"({n_att} attacked over {len(conds)} condition(s), {len(out)-n_att} benign); "
          f"threat={spec.threat}")
    if spec.note:
        print(f"[{baseline}] note: {spec.note}")
    if n_skipped_empty:
        print(f"[{baseline}] {n_skipped_empty} rows skipped: benign text under 100 words")
    if not out:
        raise SystemExit(f"{baseline}/{model}: no usable documents were built from "
                         f"{csv_path}")
    return out



# ======================================================================================
# PaperGuard — LONG side-files, and the richest instrumentation of the four
# ======================================================================================
# Where the injection actually went. PaperGuard records `injection_position` per row, so
# the document can be rebuilt at the REAL slot instead of an assumed one. That matters for
# the defence: the audit region is chosen by slot, and screening the tail of a manuscript
# whose payload was placed in the abstract would measure nothing.
#
# Anything not in this map falls back to args.slot WITH A WARNING rather than silently, so
# an unrecognised position shows up in the log instead of quietly shifting every document.
PG_SLOT_MAP: Dict[str, str] = {
    "limitations_end": "limitations_end", "limitations": "limitations_end",
    "end_of_limitations": "limitations_end", "after_limitations": "limitations_end",
    "limitations_start": "limitations_start", "start_of_limitations": "limitations_start",
    "abstract": "abstract_end", "abstract_end": "abstract_end",
    "intro": "intro_end", "introduction": "intro_end", "intro_end": "intro_end",
    "conclusion": "conclusion_end", "conclusion_end": "conclusion_end",
    "end": "doc_end", "document_end": "doc_end", "doc_end": "doc_end",
    "start": "doc_start", "document_start": "doc_start", "doc_start": "doc_start",
}


def _pg_slot(v, default: str, warned: set) -> str:
    raw = _text(v).strip().lower().replace(" ", "_").replace("-", "_")
    if not raw:
        return default
    if raw in PG_SLOT_MAP:
        return PG_SLOT_MAP[raw]
    if raw not in warned:
        warned.add(raw)
        print(f"[paperguard] injection_position {raw!r} is not in PG_SLOT_MAP — using "
              f"{default!r}. Add it to the map if the defence should audit a different "
              f"region for this condition.")
    return default


def load_paperguard(reviews_csv: str, pairs_csv: Optional[str], benign_csv: str, args,
                    *, conditions: Optional[Sequence[str]] = None,
                    max_papers: int = 10 ** 9,
                    benign_per_paper: bool = True) -> List[DocRecord]:
    """
    Build PaperGuard's eval set from attack_reviews.csv (+ attack_pairs.csv).

    THE PAYLOAD IS THE POINT. `payload_text` is the exact string PaperGuard injected, so
    the attacked document is REBUILT by inserting it into the benign paper through the same
    anchor code the insertion benchmark uses. Two consequences, both of which the wide file
    could not have given us:

      * benign and attacked differ by EXACTLY the payload — no tokenisation, truncation or
        formatting difference for a classifier to separate them on
      * the payload is a GOLD SPAN, so stage-2 character precision/recall/F1/IoU are well
        defined. Only FLASH and PaperGuard support that among the four baselines.

    JOINING THE TWO FILES
    `attack_reviews.csv` has the payload and the per-row rating; `attack_pairs.csv` has the
    matched clean/attacked ratings and the ASR the baseline itself computed. We prefer the
    pairs file's numbers, because they are the baseline's own definition of its own success
    and re-deriving them here would risk reporting a different d_rating than the attack
    paper does. The reviews file's `is_attack == False` rows are the fallback source of the
    clean rating when pairs is absent.

    `row_index` is the join key into the benign dataframe, and it is the right one: it is
    literally the position of the paper in the source dataframe, so it survives paper_id
    formatting differences between files. paper_id is the fallback.
    """
    if not os.path.exists(reviews_csv):
        raise SystemExit(f"paperguard: {reviews_csv} not found.\n{PAPERGUARD_NOTE}")
    rv = pd.read_csv(reviews_csv, low_memory=False)
    need = {"payload_text", "row_index"}
    if not need.issubset(rv.columns):
        raise SystemExit(f"paperguard: {os.path.basename(reviews_csv)} lacks {need - set(rv.columns)}. "
                         f"Columns present: {list(rv.columns)[:30]}")

    # attack rows only: the is_attack==False rows are the clean controls and carry no payload
    if "is_attack" in rv.columns:
        clean_rows = rv[rv["is_attack"].map(lambda v: _int01(v) == 0)]
        rv = rv[rv["is_attack"].map(lambda v: _int01(v) == 1)]
    else:
        clean_rows = rv.iloc[0:0]
        rv = rv[rv["payload_text"].notna()]
    rv = rv[rv["payload_text"].astype(str).str.strip().ne("")]

    # clean rating per row_index, from the is_attack==False rows
    clean_by_row: Dict[int, float] = {}
    for _, r in clean_rows.iterrows():
        ri = _num(r.get("row_index"))
        rt = _num(r.get("rating"))
        if ri is not None and rt is not None:
            clean_by_row[int(ri)] = rt

    # attack_pairs.csv: the baseline's OWN matched numbers, keyed by (row_index, payload)
    pairs: Dict[Tuple[int, str], Dict] = {}
    if pairs_csv and os.path.exists(pairs_csv):
        pr = pd.read_csv(pairs_csv, low_memory=False)
        for _, r in pr.iterrows():
            ri = _num(r.get("row_index"))
            if ri is None:
                continue
            key = (int(ri), _text(r.get("payload_name")) or _text(r.get("category")))
            pairs[key] = {
                "clean": _num(r.get("clean_rating")),
                "attacked": _num(r.get("attacked_rating")),
                "delta": _num(r.get("delta_rating")),
                "asr": _int01(r.get("asr")),
                "flip": _int01(r.get("flip")),
            }
            clean_by_row.setdefault(int(ri), pairs[key]["clean"])
        print(f"[paperguard] {len(pairs)} matched pairs from "
              f"{os.path.basename(pairs_csv)} — using the baseline's own clean/attacked "
              f"ratings and ASR rather than re-deriving them")
    else:
        print(f"[paperguard] no attack_pairs.csv — clean ratings come from the "
              f"is_attack==False rows of the reviews file ({len(clean_by_row)} found)")

    bdf = pd.read_csv(benign_csv, low_memory=False).reset_index(drop=True)
    if conditions:
        keep = {str(c) for c in conditions}
        col = "payload_name" if "payload_name" in rv.columns else "category"
        rv = rv[rv[col].astype(str).isin(keep)]

    out: List[DocRecord] = []
    warned: set = set()
    seen_papers: set = set()
    n_no_anchor = 0

    for _, r in rv.iterrows():
        ri = _num(r.get("row_index"))
        if ri is None or int(ri) >= len(bdf):
            continue
        ri = int(ri)
        if len(seen_papers) >= max_papers and ri not in seen_papers:
            continue

        rec = read_record(bdf, ri, args)
        benign_text = _text(rec.get(args.text_column))
        if len(benign_text.split()) < 100:
            continue
        anchor = read_anchor(rec, args)
        if anchor is None:
            n_no_anchor += 1
            continue

        payload = _text(r.get("payload_text")).strip()
        cond = (_text(r.get("payload_name")) or _text(r.get("category")) or "payload")
        slot = _pg_slot(r.get("injection_position"), args.slot, warned)
        try:
            atk = anchor.render(payload, slot)[: args.max_chars]
        except Exception as e:
            print(f"  [skip] row {ri}/{cond}: render at slot {slot!r} failed: "
                  f"{type(e).__name__}: {e}")
            continue

        pid = _text(r.get("paper_id")) or _paper_id(bdf.iloc[ri], ri, args.id_column)
        p = pairs.get((ri, _text(r.get("payload_name")))) or \
            pairs.get((ri, _text(r.get("category")))) or {}
        r0 = p.get("clean", clean_by_row.get(ri))
        ra = p.get("attacked", _num(r.get("rating")))
        dr = p.get("delta")
        if dr is None and ra is not None and r0 is not None:
            dr = ra - r0
        asr = p.get("asr")
        if asr is None and dr is not None:
            asr = int(dr >= 1.0)

        out.append(DocRecord(
            doc_id=f"{pid}|paperguard:{cond}", paper_id=str(pid), row_index=ri, label=1,
            condition=f"paperguard:{cond}", group=_text(r.get("category")) or "paperguard",
            document=atk, insert=payload,          # GOLD SPAN
            base_rating=r0, attacked_rating=ra, d_rating=dr, asr=asr,
            slot=slot, max_chars=args.max_chars))
        seen_papers.add(ri)

    if benign_per_paper:
        for ri in sorted(seen_papers):
            rec = read_record(bdf, ri, args)
            pid = _paper_id(bdf.iloc[ri], ri, args.id_column)
            r0 = clean_by_row.get(ri)
            out.append(DocRecord(
                doc_id=f"{pid}|paperguard:clean", paper_id=str(pid), row_index=ri, label=0,
                condition=CLEAN_CONDITION, group="control",
                document=_text(rec.get(args.text_column))[: args.max_chars], insert="",
                base_rating=r0, attacked_rating=r0, d_rating=0.0, asr=0,
                slot=args.slot, max_chars=args.max_chars))

    n_att = sum(1 for x in out if x.label == 1)
    n_cond = len({x.condition for x in out if x.label == 1})
    print(f"[paperguard] {len(seen_papers)} papers -> {len(out)} documents "
          f"({n_att} attacked over {n_cond} condition(s), {len(out)-n_att} benign); "
          f"threat=insertion, GOLD SPANS available so span P/R/F1 are well defined")
    if n_no_anchor:
        print(f"[paperguard] {n_no_anchor} rows skipped: no usable anchor in the benign paper")
    if not out:
        raise SystemExit(f"paperguard: no usable documents built from {reviews_csv}")
    return out


def benign_texts_from(csv_path: str, args, max_rows: int = 10 ** 9,
                      only_papers: Optional[Sequence[str]] = None) -> List[str]:
    """
    The benign manuscripts, for building the perturbation arm's reference vocabulary.

    `only_papers` IS THE LEAKAGE GUARD, and it is not optional in a real run.
    The vocabulary decides which words count as out-of-corpus. If it is built from the
    same papers being scored, every clean document scores ~0 by construction — its own
    vocabulary is in the reference — and the separation you measure is an artifact of the
    setup rather than a property of the detector. Passing the INDEX half here, exactly as
    the attack memory does, keeps the reference disjoint from what is being classified.

    Read straight from the wide CSV — the attacked columns are ignored — so the vocabulary
    is domain-correct: "ResNet", "CIFAR" and "perplexity" are out-of-dictionary but
    perfectly ordinary here, and a dictionary would flag every paper in the corpus.
    """
    df = pd.read_csv(csv_path, low_memory=False).reset_index(drop=True)
    keep = {str(p) for p in only_papers} if only_papers else None
    out, n_skipped = [], 0
    for i in range(min(len(df), max_rows)):
        rec = read_record(df, i, args)
        if keep is not None and _paper_id(df.iloc[i], i, args.id_column) not in keep:
            n_skipped += 1
            continue
        t = _text(rec.get(args.text_column))
        if len(t.split()) >= 100:
            out.append(t)
    if keep is not None:
        print(f"[vocab] built from {len(out)} INDEX papers; {n_skipped} eval papers "
              f"excluded so the reference is disjoint from what is scored")
    return out


# ======================================================================================
# Breaking the Reviewer — the LONG side-file loader (V4)
# ======================================================================================
BTR_NOTE = """
BREAKING THE REVIEWER: TWO WAYS IN, AND THE SECOND ONE IS BETTER
--------------------------------------------------------------------------------------
  WIDE   attack/<model>/attack_augmented_input.csv
         one row per paper, conditions folded into btr_<Attack>_* columns. Handled by the
         generic `load_baseline` walker, because the BTR spec's column templates already
         match what run_attack.py::write_outputs emits.

  LONG   attack/<model>/attack_results.jsonl          <-- USE THIS ONE
         one record per (paper, attack), and it carries BOTH `clean_content` and
         `attacked_content`.

WHY THE LONG FILE MATTERS SO MUCH HERE
  The perturbation arm has never had a GOLD SPAN. On the insertion arm the inserted
  payload is known exactly, so span precision/recall/F1 and IoU are free. On BTR nothing
  is inserted, so V3 could report detection accuracy and nothing about localisation —
  which is half of what this project claims to do ("find the place and remove it").

  With both copies present the gold span is computable exactly: the sentences that DIFFER
  between clean_content and attacked_content are, by construction, the sentences the
  attack touched. That turns "we detect BTR" into "we detect BTR and we can say which
  sentences it edited, and here is the F1".

  THE CLEAN COPY IS GROUND TRUTH ONLY. It is attached to the record as `insert` (the same
  field the insertion arm uses for its gold payload) and, exactly as there, it is never
  shown to the screens, the agents or the repairer. Handing the defence the original
  document would make the whole problem trivial and the number meaningless.
"""

_BTR_SENT = re.compile(r"(?<=[.!?])\s+")


def btr_gold_spans(clean_text: str, attacked_text: str, max_spans: int = 40) -> List[str]:
    """
    The sentences BTR actually edited: aligned position by position, kept where they differ.

    Positional alignment rather than a diff algorithm, deliberately. Every BTR
    transformation is SUBSTITUTIVE — characters, words or whole sentences are replaced in
    place, never inserted or deleted — so sentence i of the attacked document corresponds
    to sentence i of the clean one. A generic diff would try to explain a heavily
    corrupted sentence as a delete plus an insert and would report two spans where one
    edit happened.
    """
    cs = [s.strip() for s in _BTR_SENT.split(clean_text or "") if s.strip()]
    as_ = [s.strip() for s in _BTR_SENT.split(attacked_text or "") if s.strip()]
    out = []
    for i in range(min(len(cs), len(as_))):
        if cs[i] != as_[i]:
            out.append(as_[i])
        if len(out) >= max_spans:
            break
    return out


def load_btr_long(results_path: str, args, conditions: Optional[Sequence[str]] = None,
                  max_papers: int = 10 ** 9, include_benign: bool = True
                  ) -> List["DocRecord"]:
    """
    Build the defence evaluation set from BTR's attack_results.jsonl.

    ONE BENIGN ROW PER PAPER, NOT PER ATTACK. Every attack on a paper shares the same
    clean_content, so emitting a benign row per (paper, attack) would enter the identical
    document five times and inflate the benign half fivefold. The class balance — and
    therefore accuracy and FPR — would then be a property of how many attacks the baseline
    happened to run, which is not a property of the defence.
    """
    if not os.path.exists(results_path):
        raise SystemExit(f"{results_path} not found.\n{BTR_NOTE}")
    keep_conds = {str(c) for c in conditions} if conditions else None

    rows = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        raise SystemExit(f"{results_path} has no readable records.\n{BTR_NOTE}")

    out: List[DocRecord] = []
    seen_clean: Dict[str, str] = {}
    n_papers = 0
    for r in rows:
        attack = str(r.get("attack") or "")
        if keep_conds and attack not in keep_conds:
            continue
        pid = str(r.get("paper_id") or r.get("row_index") or "")
        clean = _text(r.get("clean_content"))
        adv = _text(r.get("attacked_content"))
        if not adv or not clean:
            continue
        if pid not in seen_clean:
            if n_papers >= max_papers:
                continue
            n_papers += 1
            seen_clean[pid] = clean
            if include_benign:
                out.append(DocRecord(
                    doc_id=f"{pid}__A0_clean", paper_id=pid,
                    row_index=int(r.get("row_index") or 0), label=0,
                    condition="A0_clean", group="control", document=clean,
                    insert="", base_rating=_num(r.get("rating_clean")),
                    attacked_rating=None, d_rating=None, asr=0,
                    slot=args.slot, max_chars=args.max_chars))
        elif pid not in seen_clean:
            continue

        gold = btr_gold_spans(clean, adv)
        out.append(DocRecord(
            doc_id=f"{pid}__{attack}", paper_id=pid,
            row_index=int(r.get("row_index") or 0), label=1,
            condition=attack, group="btr_perturbation", document=adv,
            # GOLD ONLY. The joined edited sentences play the role the inserted payload
            # plays on the insertion arm: they are what span precision/recall are scored
            # against, and they are never shown to any component of the defence.
            insert="\n".join(gold),
            base_rating=_num(r.get("rating_clean")),
            attacked_rating=_num(r.get("rating_attacked")),
            d_rating=_num(r.get("delta_rating")),
            asr=_int01(r.get("asr")),
            slot=args.slot, max_chars=args.max_chars))

    n_att = sum(1 for d in out if d.label == 1)
    n_ben = sum(1 for d in out if d.label == 0)
    conds = sorted({d.condition for d in out if d.label == 1})
    print(f"[btr] {len(out)} documents from {os.path.basename(results_path)}: "
          f"{n_att} attacked across {len(conds)} families {conds}, {n_ben} benign "
          f"(one clean copy per paper, not one per attack)")
    n_gold = sum(1 for d in out if d.label == 1 and d.insert)
    print(f"[btr] gold spans computed for {n_gold}/{n_att} attacked documents by aligning "
          f"clean_content against attacked_content — this is what gives the perturbation "
          f"arm span precision/recall for the first time. The clean copy is GROUND TRUTH "
          f"ONLY and is never shown to the defence.")
    if not out:
        raise SystemExit(f"btr: no usable documents built from {results_path}")
    return out
