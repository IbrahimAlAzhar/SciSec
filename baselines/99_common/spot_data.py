#!/usr/bin/env python3
"""
spot_data.py — the documents the Spotlighting rows are scored on, built by YOUR loaders.

    (a) ours       infer_ladder.load_eval_set(benign_csv, attacked_csv, id_column="id",
                   seed=0)  — the call in job_c_<model>_infer.pbs (defense1 code). It
                   returns consecutive (clean, attacked) document pairs; the clean row of
                   each pair is the A0_clean twin category_defense_metrics.py pairs it with.
    (b) argar/btr  ab_common.load_pairs(["argar","btr"], subset="test") — the v8 test
                   pairs (baselines_for_argar_btr). One pair = benign_document +
                   attacked_document of the same paper.
    (c) combined   the union, scored separately and together.

Everything is frozen into ONE file per reviewer, inputs/<model>/docs[__TAG].jsonl.gz, with
a manifest (counts, sha256, where every part came from). The review stage only reads that
file, so every variant of the defence is scored on byte-identical documents, and a
re-submitted job cannot silently pick up a rebuilt corpus.

Row schema (one row per DOCUMENT, in evaluation order):
    eval_index, doc_id, pair_uid, role (clean|attacked), label (0|1), source (ours|argar|
    btr), family (instruction|rhetorical|other|repackaging|perturbation), condition,
    method, paper_id, harmful (benchmark ASR flag), harmful_src, ref_weaknesses, origin,
    doc_sha, document
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

import spot_config as C


def _s(x) -> str:
    return "" if x is None or (isinstance(x, float) and x != x) else str(x)


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", "ignore")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ======================================================================================
# (a) OUR ATTACKS — defense1's loader, imported
# ======================================================================================
def _rec_get(rec, name, default=None):
    if isinstance(rec, dict):
        return rec.get(name, default)
    return getattr(rec, name, default)


def _rec_extras(rec) -> Dict:
    """Small scalar attributes of a DocRecord (insert, d_rating, ...) kept for audit."""
    try:
        d = dict(rec) if isinstance(rec, dict) else dict(vars(rec))
    except TypeError:
        return {}
    out = {}
    for k, v in d.items():
        if k in ("document", "doc_id", "paper_id", "label", "condition") or k.startswith("_"):
            continue
        if isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, str) and len(v) <= 4000:
            out[k] = v
    return out


def call_defense1_loader(benign_csv: str, attacked_csv: str, *, id_column: str = "id",
                         max_papers: int = 10 ** 9, seed: int = 0) -> List:
    """infer_ladder.load_eval_set, exactly as job_c_<model>_infer.pbs calls it, with
    infer_ladder taken from FT_CODE_DIR and def_dataset from V4_DIR (job_c's
    PYTHONPATH=$CODE_DIR:$V4_DIR order), whatever other copies exist."""
    for d, f in ((C.FT_CODE_DIR, "infer_ladder.py"), (C.V4_DIR, "def_dataset.py")):
        if not os.path.exists(os.path.join(d, f)):
            raise SystemExit(f"[FATAL] {os.path.join(d, f)} not found. Set "
                             f"{'FT_CODE_DIR' if f.startswith('infer') else 'V4_DIR'}.")
    added = []
    for d in (C.V4_DIR, C.FT_CODE_DIR):               # -> sys.path = [FT, V4, ...]
        sys.path.insert(0, d)
        added.append(d)
    try:
        if "def_dataset" in sys.modules and os.path.dirname(os.path.abspath(
                sys.modules["def_dataset"].__file__)) != os.path.abspath(C.V4_DIR):
            del sys.modules["def_dataset"]
        IL = C.import_from(C.FT_CODE_DIR, "infer_ladder")
        print(f"[ours] infer_ladder: {IL.__file__}")
        recs = IL.load_eval_set(benign_csv, attacked_csv, id_column=id_column,
                                max_papers=max_papers, seed=seed, all_papers=True)
        dd = sys.modules.get("def_dataset")
        if dd is not None:
            print(f"[ours] def_dataset : {dd.__file__}")
        for m in ("align", "runner", "strategies"):
            if m in sys.modules:
                print(f"[ours] {m:<11} : {getattr(sys.modules[m], '__file__', '?')}")
        return recs
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(
            f"[FATAL] infer_ladder.load_eval_set failed: {type(e).__name__}: {e}\n"
            f"        (a) uses YOUR defense1 loader so the 990 documents are the ones the\n"
            f"        SFT/DPO rows were scored on. Check FT_CODE_DIR / V4_DIR and that the\n"
            f"        attack modules (strategies, align, runner) are on PYTHONPATH — the PBS\n"
            f"        file adds them exactly as job_c_<model>_infer.pbs does.")
    finally:
        for d in added:
            try:
                sys.path.remove(d)
            except ValueError:
                pass


def compare_with_defense1_runs(model_key: str, recs: Sequence) -> Dict:
    """Is (a) the corpus your defense1 rows were scored on? If FT_CODE_DIR/runs/<model>/
    <ablation>/stage1_detail.csv exists, compare the ordered (paper_id, condition, label)
    sequence with ours. A warning, not a failure: the file may be from a pilot."""
    import glob
    seq = [(_s(_rec_get(r, "paper_id")).strip(), _s(_rec_get(r, "condition")),
            int(_rec_get(r, "label", 0) or 0)) for r in recs]
    out: Dict = {"checked": [], "match": None}
    for p in sorted(glob.glob(os.path.join(C.FT_CODE_DIR, "runs", model_key, "*",
                                           "stage1_detail.csv")))[:3]:
        try:
            d = pd.read_csv(p, dtype={"paper_id": str}, low_memory=False,
                            usecols=lambda c: c in ("paper_id", "condition", "label"))
        except Exception as e:
            out["checked"].append({"file": p, "error": str(e)[:200]})
            continue
        other = [(str(a).strip(), _s(b), int(c)) for a, b, c in
                 zip(d["paper_id"], d["condition"], d["label"])]
        same = other == seq
        first = next((i for i, (x, y) in enumerate(zip(seq, other)) if x != y),
                     None if len(seq) == len(other) else min(len(seq), len(other)))
        out["checked"].append({"file": p, "n_theirs": len(other), "n_ours": len(seq),
                               "identical": same, "first_difference": first})
        out["match"] = bool(same) if out["match"] is None else (out["match"] or same)
        print(f"[ours] vs {p}: " + ("IDENTICAL document sequence" if same else
              f"DIFFERENT (theirs {len(other)} rows, ours {len(seq)}, first difference "
              f"at row {first})"))
    if not out["checked"]:
        print(f"[ours] no defense1 stage1_detail.csv under {C.FT_CODE_DIR}/runs/{model_key}/ "
              f"to compare with (fine)")
    return out


def ref_weaknesses_map(benign_csv: str, id_column: str = "id") -> Dict[str, str]:
    if not benign_csv or not os.path.exists(benign_csv):
        return {}
    try:
        d = pd.read_csv(benign_csv, low_memory=False,
                        usecols=lambda c: c in (id_column, "paper_id", "weaknesses"))
    except Exception as e:
        print(f"[ours] could not read weaknesses from {benign_csv}: {e}")
        return {}
    key = id_column if id_column in d.columns else ("paper_id" if "paper_id" in d.columns else "")
    if not key or "weaknesses" not in d.columns:
        return {}
    return {str(k).strip(): _s(w) for k, w in zip(d[key], d["weaknesses"])}


def ours_rows(recs: Sequence, weak: Dict[str, str], origin: str,
              max_pairs: int = 0) -> Tuple[List[Dict], Dict]:
    """Records -> document rows with a pair_uid linking every attacked document to its
    clean twin: the IMMEDIATELY PRECEDING clean row when it is the same paper (the
    build_eval_set layout category_defense_metrics.py relies on), else the most recent
    clean row of that paper."""
    rows: List[Dict] = []
    last_clean: Dict[str, int] = {}
    stats = {"n_records": len(recs), "n_pairs": 0, "n_unpaired_attacked": 0,
             "n_nonadjacent_pairs": 0, "n_clean_unused": 0}
    used_clean = set()
    n_pairs = 0
    for i, rec in enumerate(recs):
        lab = int(_rec_get(rec, "label", 0) or 0)
        pid = _s(_rec_get(rec, "paper_id")).strip()
        cond = _s(_rec_get(rec, "condition")) or ("A0_clean" if lab == 0 else "")
        doc = _s(_rec_get(rec, "document"))
        asr = _rec_get(rec, "asr", None)
        r = {"doc_id": _s(_rec_get(rec, "doc_id")) or f"ours:{i}", "paper_id": pid,
             "label": lab, "role": "attacked" if lab else "clean", "source": "ours",
             "condition": cond, "method": cond,
             "family": C.family_of_condition(cond) if lab else "clean",
             "harmful": int(bool(asr)) if lab else 0, "harmful_src": "benchmark asr",
             "ref_weaknesses": weak.get(pid, ""), "origin": origin, "document": doc,
             "pair_uid": "", "extras": json.dumps(_rec_extras(rec), default=str)[:6000]}
        if lab == 0:
            last_clean[pid] = len(rows)
            rows.append(r)
            continue
        if max_pairs and n_pairs >= max_pairs:
            break                      # trailing clean rows without a pair are dropped below
        prev = len(rows) - 1
        twin = None
        if prev >= 0 and rows[prev]["role"] == "clean" and rows[prev]["paper_id"] == pid:
            twin = prev
        elif pid in last_clean:
            twin = last_clean[pid]
            stats["n_nonadjacent_pairs"] += 1
        if twin is None:
            stats["n_unpaired_attacked"] += 1
            rows.append(r)
            continue
        uid = f"ours:{n_pairs:05d}:{pid}:{cond}"
        if rows[twin]["pair_uid"] == "":
            rows[twin]["pair_uid"] = uid
        else:
            # the same clean row twins two attacks (not the build_eval_set layout, but
            # handled): duplicate it so every pair owns its clean twin row
            dup = dict(rows[twin], pair_uid=uid, doc_id=rows[twin]["doc_id"] + f"#{n_pairs}")
            rows.append(dup)
        used_clean.add(twin)
        r["pair_uid"] = uid
        rows.append(r)
        n_pairs += 1
    stats["n_pairs"] = n_pairs
    stats["n_clean_unused"] = sum(1 for i, r in enumerate(rows)
                                  if r["role"] == "clean" and r["pair_uid"] == "")
    # unused clean rows (a paper whose attacks were all dropped) carry no pair: drop them
    rows = [r for r in rows if r["pair_uid"]]
    return rows, stats


def build_ours(model_key: str, *, benign_csv: str = "", attacked_csv: str = "",
               id_column: str = "id", seed: int = 0, max_pairs: int = 0,
               ours_docs: str = "", ours_index: str = "") -> Tuple[List[Dict], Dict]:
    benign_csv = benign_csv or C.TEST_CSV
    attacked_csv = attacked_csv or C.attacked_csv(model_key)
    if ours_docs:
        # an already-materialised copy of the corpus, one JSON DocRecord per document:
        #   * PaperGuard's runs/<model>/eval_docs.jsonl.gz (the defense1 corpus), or
        #   * v7ab's inputs/<model>/eval_records.jsonl WITH --ours-index eval_index.csv:
        #     the exact documents SafeReview / System Defense were scored on (only the
        #     input_source == "insertion" rows are kept; ARGAR/BTR come from (b)).
        op = gzip.open if ours_docs.endswith(".gz") else open
        with op(ours_docs, "rt", encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        origin = f"file:{ours_docs}"
        if ours_index:
            ix = pd.read_csv(ours_index, low_memory=False)
            if len(ix) != len(recs):
                raise SystemExit(f"[FATAL] {ours_docs} has {len(recs)} records but "
                                 f"{ours_index} has {len(ix)} rows")
            src = ix["input_source"].fillna("insertion").astype(str).tolist() \
                if "input_source" in ix.columns else ["insertion"] * len(recs)
            if "doc_id" in ix.columns and not all(
                    str(r.get("doc_id")) == str(d) for r, d in zip(recs, ix["doc_id"])):
                raise SystemExit(f"[FATAL] {ours_docs} and {ours_index} disagree on doc_id order")
            recs = [r for r, s_ in zip(recs, src) if s_ == "insertion"]
            origin += f" + {ours_index} (input_source == insertion)"
    else:
        if not os.path.exists(attacked_csv):
            raise SystemExit(f"[FATAL] {attacked_csv} not found (the metrics_detail.csv "
                             f"job_c_<model>_infer.pbs uses). Pass --attacked-csv.")
        recs = call_defense1_loader(benign_csv, attacked_csv, id_column=id_column,
                                    seed=seed)
        origin = f"infer_ladder.load_eval_set({os.path.basename(benign_csv)}, {attacked_csv})"
        defense1_check = compare_with_defense1_runs(model_key, recs)
    weak = ref_weaknesses_map(benign_csv, id_column)
    if not recs:
        raise SystemExit("[FATAL] (a) produced no records")
    rows, st = ours_rows(recs, weak, origin, max_pairs=max_pairs)
    if not ours_docs:
        st["defense1_corpus_check"] = defense1_check
    st.update({"benign_csv": benign_csv, "attacked_csv": attacked_csv, "origin": origin,
               "n_with_ref_weaknesses": sum(1 for r in rows if r["ref_weaknesses"])})
    print(f"[ours] {st['n_records']} records -> {st['n_pairs']} pairs "
          f"({st['n_unpaired_attacked']} attacked rows without a clean twin dropped, "
          f"{st['n_nonadjacent_pairs']} non-adjacent twins)")
    return rows, st


# ======================================================================================
# (b) ARGAR / BTR — baselines_for_argar_btr/ab_common.py, imported
# ======================================================================================
def _import_ab_common():
    try:
        A = C.import_from(C.ARGAR_DIR, "ab_common")
        print(f"[argar/btr] ab_common: {A.__file__}")
        return A
    except Exception as e:
        print(f"[argar/btr] ab_common not importable from {C.ARGAR_DIR} "
              f"({type(e).__name__}: {e}); using the minimal pairs reader")
        return None


def choose_tag(model_key: str, A=None) -> Tuple[str, bool]:
    """The ARGAR/BTR file tag for this reviewer: its own attacks when they exist,
    otherwise the gemma3-27b pairs (transfer — R_0/R_A are re-reviewed by THIS reviewer,
    so the rating metrics stay valid; only the benchmark `harmful` flag is gemma's)."""
    forced = os.environ.get("ARGAR_TAG", "")
    if forced:
        return forced, forced == C.MODELS[model_key].tag
    spec = C.MODELS[model_key]
    v8 = getattr(A, "V8_DATA", C.V8_DATA_DIR) if A else C.V8_DATA_DIR
    dvb = getattr(A, "DVB_DIR", C.DVB_DIR) if A else C.DVB_DIR
    for t in (spec.tag,) + tuple(spec.alt_tags):
        if os.path.exists(os.path.join(v8, f"test_pairs__{t}.csv.gz")) or (
                os.path.exists(os.path.join(dvb, f"argar__{t}.csv.gz")) and
                os.path.exists(os.path.join(dvb, f"btr__{t}.csv.gz"))):
            return t, True
    return C.ARGAR_REFERENCE_TAG, model_key == "gemma3-27b-it"


def _minimal_pairs(sources, pairs_csv: str, tag: str, max_rows: int) -> pd.DataFrame:
    p = pairs_csv or os.path.join(C.V8_DATA_DIR, f"test_pairs__{tag}.csv.gz")
    if not os.path.exists(p):
        raise SystemExit(f"[FATAL] no ARGAR/BTR pairs: {p} (and ab_common not importable)")
    df = pd.read_csv(p, low_memory=False)
    if "split" in df.columns:
        df = df[df["split"].fillna("test") == "test"]
    df = df[df["input_source"].isin(list(sources))].copy()
    df["_k"] = [hashlib.sha1(str(u).encode()).hexdigest() for u in df["pair_uid"]]
    df = df.sort_values(["input_source", "_k"])
    if max_rows > 0:
        df = df.groupby("input_source", group_keys=False).head(max_rows)
    df = df.drop(columns="_k").reset_index(drop=True)
    df.attrs["source_note"] = p
    return df


def build_argar_btr(model_key: str, *, sources: Sequence[str] = ("argar", "btr"),
                    subset: str = "test", max_rows: int = 0, pairs_csv: str = ""
                    ) -> Tuple[List[Dict], Dict]:
    A = _import_ab_common()
    tag, native = choose_tag(model_key, A)
    if A is not None:
        old = A.MODEL_TAG
        A.MODEL_TAG = tag
        try:
            pairs = A.load_pairs(list(sources), subset, max_rows, pairs_csv)
        finally:
            A.MODEL_TAG = old
        method_of = A.method_of
        fam_of_src = A.FAMILY_OF_SOURCE
    else:
        pairs = _minimal_pairs(sources, pairs_csv, tag, max_rows)

        def method_of(r):
            for c in ("variant_label", "attack_kind", "condition"):
                v = r.get(c)
                if v is not None and str(v).strip() and str(v).lower() != "nan":
                    return str(v).strip()
            return str(r.get("input_source"))
        fam_of_src = {"argar": "repackaging", "btr": "perturbation"}
    src_note = pairs.attrs.get("source_note", "")
    rows: List[Dict] = []
    n_same = 0
    for _, r in pairs.iterrows():
        uid = _s(r.get("pair_uid"))
        src = _s(r.get("input_source"))
        fam = _s(r.get("threat_family")) or fam_of_src.get(src, src)
        hh = _f(r.get("harmful_hist"))
        if hh is None:
            a = _f(r.get("historical_asr"))
            hh = None if a is None else float(a > 0)
        meth = method_of(r)
        ben, att = _s(r.get("benign_document")), _s(r.get("attacked_document"))
        n_same += int(ben == att)
        common = {"pair_uid": uid, "paper_id": _s(r.get("paper_id")), "source": src,
                  "method": meth, "condition": _s(r.get("condition")) or meth,
                  "ref_weaknesses": _s(r.get("reference_weaknesses")),
                  "harmful_src": ("historical asr" + ("" if native else
                                                      f" ({tag} attacks, transferred)")),
                  "origin": f"ab_common.load_pairs({src_note})" if A is not None
                  else f"minimal reader ({src_note})",
                  "extras": json.dumps({"variant_label": _s(r.get("variant_label")),
                                        "attack_kind": _s(r.get("attack_kind")),
                                        "historical_base_rating": _f(r.get("historical_base_rating")),
                                        "historical_attacked_rating":
                                            _f(r.get("historical_attacked_rating"))})}
        rows.append(dict(common, doc_id=f"{uid}|B", label=0, role="clean", family="clean",
                         harmful=0, document=ben))
        rows.append(dict(common, doc_id=f"{uid}|A", label=1, role="attacked", family=fam,
                         harmful=int(hh) if hh is not None else 0, document=att))
    st = {"pairs_from": src_note, "tag": tag, "native_attacks": native,
          "n_pairs": len(pairs), "per_source": pairs["input_source"].value_counts().to_dict(),
          "n_attacked_identical_to_benign": n_same, "loader": "ab_common" if A else "minimal"}
    print(f"[argar/btr] {len(pairs)} pairs from {src_note}  tag={tag} "
          f"({'this reviewer' if native else 'TRANSFER: attacks made against ' + tag})")
    if n_same:
        print(f"[argar/btr] WARNING {n_same} attacked documents equal their benign twin")
    return rows, st


# ======================================================================================
# freeze / load
# ======================================================================================
def build_inputs(model_key: str, *, sources: Sequence[str] = ("ours", "argar", "btr"),
                 run_tag: str = "", max_rows: int = 0, force: bool = False,
                 benign_csv: str = "", attacked_csv: str = "", id_column: str = "id",
                 seed: int = 0, ours_docs: str = "", ours_index: str = "",
                 argar_pairs_csv: str = "", subset: str = "test") -> str:
    out = C.inputs_path(model_key, run_tag)
    man = C.manifest_path(model_key, run_tag)
    if os.path.exists(out) and os.path.exists(man) and not force:
        m = json.load(open(man))
        if sha256_file(out) != m.get("sha256"):
            raise SystemExit(f"[FATAL] {out} does not match its manifest sha256 — it was "
                             f"edited or rebuilt by another run. Delete it (and the "
                             f"manifest) or pass --force to rebuild.")
        print(f"[inputs] reusing {out}  ({m.get('n_documents')} documents, sha256 ok)")
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows: List[Dict] = []
    manifest: Dict = {"model": model_key, "run_tag": run_tag, "sources": list(sources),
                      "max_rows": max_rows, "parts": {}}
    if "ours" in sources:
        r, st = build_ours(model_key, benign_csv=benign_csv, attacked_csv=attacked_csv,
                           id_column=id_column, seed=seed, max_pairs=max_rows,
                           ours_docs=ours_docs, ours_index=ours_index)
        rows += r
        manifest["parts"]["ours"] = st
    ab = [s for s in sources if s in ("argar", "btr")]
    if ab:
        r, st = build_argar_btr(model_key, sources=ab, subset=subset, max_rows=max_rows,
                                pairs_csv=argar_pairs_csv)
        rows += r
        manifest["parts"]["argar_btr"] = st
    if not rows:
        raise SystemExit("[FATAL] no documents were built")
    seen = set()
    for i, r in enumerate(rows):
        r["eval_index"] = i
        r["doc_sha"] = sha1(r["document"])
        if r["doc_id"] in seen:
            r["doc_id"] = f"{r['doc_id']}#{i}"
        seen.add(r["doc_id"])
    tmp = out + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    d = pd.DataFrame(rows)
    att = d[d["label"] == 1]
    manifest.update({
        "file": out, "sha256": sha256_file(out), "n_documents": len(d),
        "n_unique_texts": int(d["doc_sha"].nunique()),
        "n_pairs": int(att["pair_uid"].nunique()),
        "per_source": att["source"].value_counts().to_dict(),
        "per_family": att["family"].value_counts().to_dict(),
        "harmful_per_source": att.groupby("source")["harmful"].sum().astype(int).to_dict(),
        "mean_chars": float(d["document"].str.len().mean()),
    })
    with open(man, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[inputs] {len(d)} documents ({manifest['n_unique_texts']} distinct texts), "
          f"{manifest['n_pairs']} pairs -> {out}")
    for k, v in sorted(manifest["per_family"].items()):
        print(f"[inputs]   {k:<14} {v:>5} attacked documents")
    return out


def load_docs(model_key: str, run_tag: str = "") -> pd.DataFrame:
    p = C.inputs_path(model_key, run_tag)
    if not os.path.exists(p):
        raise SystemExit(f"[FATAL] {p} not found — run `python run_spotlight.py inputs "
                         f"--model {model_key}{' --run-tag ' + run_tag if run_tag else ''}` "
                         f"first")
    with gzip.open(p, "rt", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    d = pd.DataFrame(rows)
    d["label"] = d["label"].astype(int)
    d["harmful"] = d["harmful"].fillna(0).astype(int)
    for c in ("ref_weaknesses", "pair_uid", "source", "family", "condition", "method"):
        d[c] = d[c].fillna("").astype(str)
    return d
