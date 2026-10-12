"""
pg_core.py — PaperGuard's chunk-based embedding search, with gpt-4o-mini as the verifier.

WHAT THE METHOD IS (PaperGuard §4.2, defense/emb_search.py)
=======================================================================================
    1. Segment the manuscript into text chunks  C = _split_text(paper, max_len=800)
    2. Embed chunks and every known attack pattern a in A_known into one space
       (text-embedding-3-large), and for each a retrieve the top-k chunks by cosine
       similarity (sklearn NearestNeighbors, metric="cosine").
    3. Ask a verifier LLM, once per known pattern, whether ANY of its top-k chunks
       "expresses the same intent or instructions" as the pattern. The paper is flagged
       as attacked iff at least one pattern is verified. Patterns are checked in registry
       order and the loop stops at the first positive, exactly as detect() does.

WHAT CHANGED, AND ONLY THIS
    * verifier:     Azure GPT-4o  ->  gpt-4o-mini on the OpenAI API (pg_llm.py). Same
                    prompt (executed from the original source, see pg_orig.py),
                    temperature 0.0, max_tokens 200 — PaperGuard's defaults.
    * modality:     figures dropped. Our corpus is text only; there is nothing to embed on
                    the visual side.
    * top-k:        clamped to the number of chunks. NearestNeighbors raises when a short
                    document has fewer than k chunks; the original would crash there.
    * parse errors: see pg_orig.parse_assessment_lenient. The original drops the paper.
    * API errors:   the original maps an Azure HttpResponseError to UNSAFE (the Azure
                    content filter firing is evidence). Here a call that still fails after
                    pg_llm's retries is not evidence of anything, so it is recorded,
                    counted, scored SAFE for that pattern, and the document is re-scored
                    when the job is resubmitted; pg_table.py prints the count.
    * caching:      embeddings are cached on disk (sqlite, float32, keyed by model+text).
                    The values are the API's own; the cache only avoids paying twice for
                    the chunks a clean/attacked pair shares.

LOCALIZATION (the "localize and mitigate" claim of the paper)
    For a flagged document the evidence is the top-k chunks of the pattern that fired.
    Each of them is re-verified ON ITS OWN with the same prompt — the paper's "(a, c)
    pairs" formulation — and the chunks that are verified individually are the located
    attack. If none is, the single most similar chunk is taken. These calls are counted in
    n_loc_calls and kept OUT of n_llm_calls, so the Calls column prices detection only.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import pg_orig as O


# ======================================================================================
# embeddings
# ======================================================================================
class EmbedCache:
    """sqlite key/value store: sha1(model \x1f text) -> float32 vector."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, d INTEGER, "
                        "v BLOB)")
        self.db.commit()

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha1((model + "\x1f" + text).encode("utf-8")).hexdigest()

    def get_many(self, keys: Sequence[str]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        ks = list(dict.fromkeys(keys))
        for i in range(0, len(ks), 500):
            chunk = ks[i:i + 500]
            q = "SELECT k, d, v FROM emb WHERE k IN (%s)" % ",".join("?" * len(chunk))
            for k, d, v in self.db.execute(q, chunk):
                out[k] = np.frombuffer(v, dtype=np.float32, count=d).copy()
        return out

    def put_many(self, items: Iterable[Tuple[str, np.ndarray]]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO emb (k, d, v) VALUES (?, ?, ?)",
            [(k, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes())
             for k, v in items])
        self.db.commit()

    def __len__(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM emb").fetchone()[0])


class Embedder:
    """
    backend "openai" — PaperGuard's own: text-embedding-3-large through the OpenAI API
                       (the `openai` SDK if installed, the same REST call through httpx
                       otherwise). Needs OPENAI_API_KEY.
    backend "hf"     — a local sentence-embedding model (path in `model`), for when the
                       compute node has no route to the API. NOT the released setting;
                       the table labels a run made this way.
    backend "hash"   — deterministic hashed bag of words. For self-tests ONLY; it has no
                       semantics and must never produce a reported number.
    """

    def __init__(self, backend: str = "openai", model: str = "text-embedding-3-large",
                 cache_path: str = "", batch_size: int = 256,
                 max_input_chars: int = 24000, max_batch_chars: int = 400000):
        self.backend = backend
        self.model = model
        self.batch_size = batch_size
        # text-embedding-3-large rejects inputs over 8191 tokens. A chunk that long only
        # arises from a "sentence" with no punctuation (a flattened table); its EMBEDDING
        # sees the first 24k characters, the verifier still sees the whole chunk.
        self.max_input_chars = max_input_chars
        self.max_batch_chars = max_batch_chars
        self.cache = EmbedCache(cache_path) if cache_path else None
        self.n_api_calls = 0
        self.n_embedded = 0
        self.n_cache_hits = 0
        self.n_truncated = 0
        self._hf = None
        self._oa = None

    @property
    def cache_model_key(self) -> str:
        return f"{self.backend}:{self.model}"

    # --------------------------------------------------------------- backends
    def _openai(self, texts: List[str]) -> List[List[float]]:
        if self._oa is None:
            try:
                from openai import OpenAI           # what PaperGuard itself calls
                self._oa = ("sdk", OpenAI())
            except ImportError:
                import httpx
                key = os.environ.get("OPENAI_API_KEY", "")
                if not key:
                    raise SystemExit("[FATAL] OPENAI_API_KEY is not set (source path/to/.keys)")
                self._oa = ("http", httpx.Client(timeout=120.0), key)
        last = None
        for attempt in range(6):
            try:
                if self._oa[0] == "sdk":
                    resp = self._oa[1].embeddings.create(model=self.model, input=texts)
                    ordered = [None] * len(resp.data)
                    for item in resp.data:
                        ordered[item.index] = item.embedding
                else:
                    base = (os.environ.get("OPENAI_BASE_URL")
                            or "https://api.openai.com/v1").rstrip("/")
                    r = self._oa[1].post(
                        f"{base}/embeddings",
                        headers={"Authorization": f"Bearer {self._oa[2]}"},
                        json={"model": self.model, "input": texts})
                    r.raise_for_status()
                    data = r.json()["data"]
                    ordered = [None] * len(data)
                    for item in data:
                        ordered[item["index"]] = item["embedding"]
                self.n_api_calls += 1
                return ordered
            except SystemExit:
                raise
            except Exception as e:                      # rate limit / transport
                last = e
                time.sleep(min(60.0, 2.0 ** attempt))
        raise RuntimeError(f"embedding API failed after 6 attempts: "
                           f"{type(last).__name__}: {last}")

    def _hf_embed(self, texts: List[str]) -> List[np.ndarray]:
        if self._hf is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._hf = ("st", SentenceTransformer(self.model, device=None))
            except Exception:
                import torch
                from transformers import AutoModel, AutoTokenizer
                tok = AutoTokenizer.from_pretrained(self.model)
                mdl = AutoModel.from_pretrained(self.model).eval()
                if torch.cuda.is_available():
                    mdl = mdl.cuda()
                self._hf = ("tf", tok, mdl)
        if self._hf[0] == "st":
            return list(self._hf[1].encode(texts, batch_size=32,
                                           normalize_embeddings=True,
                                           show_progress_bar=False))
        import torch
        _, tok, mdl = self._hf
        out = []
        for i in range(0, len(texts), 16):
            b = tok(texts[i:i + 16], padding=True, truncation=True, max_length=512,
                    return_tensors="pt").to(mdl.device)
            with torch.no_grad():
                h = mdl(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out += list(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return out

    @staticmethod
    def _hash_embed(texts: List[str], dim: int = 512) -> List[np.ndarray]:
        out = []
        for t in texts:
            v = np.zeros(dim, dtype=np.float32)
            for w in re.findall(r"\w+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
                v[h % dim] += 1.0 if (h >> 9) & 1 else -1.0
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return out

    # --------------------------------------------------------------- public
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = [t if t else " " for t in texts]
        inp = []
        for t in texts:
            if len(t) > self.max_input_chars:
                self.n_truncated += 1
                t = t[: self.max_input_chars]
            inp.append(t)
        mk = self.cache_model_key
        keys = [EmbedCache.key(mk, t) for t in inp]
        have = self.cache.get_many(keys) if self.cache is not None else {}
        self.n_cache_hits += sum(1 for k in keys if k in have)
        todo: Dict[str, str] = {}
        for k, t in zip(keys, inp):
            if k not in have and k not in todo:
                todo[k] = t
        if todo:
            tk = list(todo)
            new: Dict[str, np.ndarray] = {}
            i = 0
            while i < len(tk):
                # batches of <= batch_size items and <= max_batch_chars characters
                j, chars = i, 0
                while j < len(tk) and j - i < self.batch_size and \
                        (j == i or chars + len(todo[tk[j]]) <= self.max_batch_chars):
                    chars += len(todo[tk[j]])
                    j += 1
                batch = [todo[k] for k in tk[i:j]]
                if self.backend == "openai":
                    vecs = self._openai(batch)
                elif self.backend == "hf":
                    vecs = self._hf_embed(batch)
                elif self.backend == "hash":
                    vecs = self._hash_embed(batch)
                else:
                    raise SystemExit(f"[FATAL] unknown embed backend {self.backend!r}")
                for k, v in zip(tk[i:j], vecs):
                    new[k] = np.asarray(v, dtype=np.float32)
                if self.cache is not None:
                    self.cache.put_many([(k, new[k]) for k in tk[i:j]])
                self.n_embedded += len(batch)
                i = j
            have.update(new)
        return np.stack([have[k] for k in keys]).astype(np.float32)


# ======================================================================================
# retrieval — EmbSearchDefense._retrieve_topk, plus the distances it throws away
# ======================================================================================
def retrieve_topk(seg_emb: np.ndarray, qry_emb: np.ndarray, top_k: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    (indices [n_query, k], cosine similarity [n_query, k]).

    Same call as the original (NearestNeighbors, metric='cosine'), with
    return_distance=True so the similarity is available as a continuous score for AUROC
    and R@FPR. The neighbour indices are identical to what return_distance=False returns.
    """
    k = max(1, min(int(top_k), int(seg_emb.shape[0])))
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='auto')
        nn.fit(seg_emb.astype(np.float32))
        dist, idx = nn.kneighbors(qry_emb.astype(np.float32), return_distance=True)
        return idx, 1.0 - dist
    except ImportError:                                    # pragma: no cover
        a = seg_emb / np.maximum(np.linalg.norm(seg_emb, axis=1, keepdims=True), 1e-12)
        b = qry_emb / np.maximum(np.linalg.norm(qry_emb, axis=1, keepdims=True), 1e-12)
        sim = b @ a.T
        idx = np.argsort(-sim, axis=1, kind="stable")[:, :k]
        return idx, np.take_along_axis(sim, idx, axis=1)


# ======================================================================================
# registries (A_known)
# ======================================================================================
@dataclass
class KnownAttack:
    name: str
    text: str
    source: str = ""
    meta: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return {"name": self.name, "text": self.text, "source": self.source,
                "meta": self.meta}


def registry_paperguard(seed: int = 42, a1_size: int = 10) -> List[KnownAttack]:
    """A1 of main_defense.py, from PaperGuard's own prompts.py."""
    a1, a2, a3 = O.known_prompts(seed=seed, a1_size=a1_size)
    return [KnownAttack(name=f"{c}#{i}", text=t, source="paperguard_A1",
                        meta={"class": c, "alters_format": bool(f), "seed": seed,
                              "a1_size": a1_size, "a2_size": len(a2),
                              "a3_size": len(a3)})
            for i, (c, t, f) in enumerate(a1)]


def registry_from_train(train_csv: str, embedder: "Embedder", *, per_family: int = 1,
                        exclude_paper_ids: Iterable[str] = (), select: str = "medoid",
                        seed: int = 42, max_chars: int = 4000) -> List[KnownAttack]:
    """
    Known attacks drawn from OUR attacked TRAIN documents (ft_attack.py output).

    One representative payload per attack family by default: the payload closest to the
    family centroid in the same embedding space the search uses ("medoid"). PaperGuard's
    registry holds 10 hand-written prompts; nine families x one medoid is the matched size.

    The TEST papers are excluded and then ASSERTED absent: a registry that contained a
    test paper's own insert would make that paper's detection a lookup.
    """
    import pandas as pd
    df = pd.read_csv(train_csv, low_memory=False)
    need = {"label", "condition", "insert", "paper_id"}
    if need - set(df.columns):
        raise SystemExit(f"[FATAL] {train_csv} lacks columns {sorted(need - set(df.columns))}")
    ex = {str(p).strip() for p in exclude_paper_ids}
    att = df[(df["label"] == 1) & df["insert"].notna()].copy()
    att["paper_id"] = att["paper_id"].astype(str).str.strip()
    n0 = len(att)
    att = att[~att["paper_id"].isin(ex)]
    att["insert"] = att["insert"].astype(str).str.strip()
    att = att[att["insert"].str.len() >= 20].drop_duplicates("insert")
    print(f"[registry:train] {train_csv}: {n0} attacked rows, {n0 - len(att)} removed "
          f"(test-paper overlap / empty / duplicate), {len(att)} candidate payloads")
    overlap = set(att["paper_id"]) & ex
    if overlap:
        raise SystemExit(f"[FATAL] registry candidates include TEST papers: "
                         f"{sorted(overlap)[:10]}")
    rng = np.random.default_rng(seed)
    out: List[KnownAttack] = []
    for fam in sorted(att["condition"].unique(), key=_family_sort_key):
        g = att[att["condition"] == fam].reset_index(drop=True)
        texts = [t[:max_chars] for t in g["insert"].tolist()]
        if select == "random":
            pick = list(rng.choice(len(g), size=min(per_family, len(g)), replace=False))
        else:
            E = embedder.embed(texts)
            E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
            c = E.mean(0)
            c = c / max(float(np.linalg.norm(c)), 1e-12)
            pick = list(np.argsort(-(E @ c), kind="stable")[:per_family])
        for j in pick:
            r = g.iloc[int(j)]
            out.append(KnownAttack(
                name=f"{fam}:{r.get('strategy', '')}", text=texts[int(j)],
                source=os.path.basename(train_csv),
                meta={"family": fam, "strategy": str(r.get("strategy", "")),
                      "paper_id": str(r["paper_id"]), "n_family": int(len(g)),
                      "select": select}))
        print(f"[registry:train]   {fam:<20} {len(g):>5} payloads -> "
              f"{', '.join(str(g.iloc[int(j)].get('strategy', '')) for j in pick)}")
    if not out:
        raise SystemExit("[FATAL] the train registry is empty")
    return out


def registry_combined(train_csv: str, embedder: "Embedder", *, seed: int = 42,
                      a1_size: int = 10, **train_kw) -> List[KnownAttack]:
    """
    ABLATION registry: PaperGuard's A1 first (same prompts, same order as the baseline),
    then our TRAIN medoids. Because the verdict is "any pattern verified", this ablation
    can only ADD detections (and false positives) relative to the baseline; the shared
    first 10 patterns see exactly the same top-k chunks and the same verifier prompt.
    """
    base = registry_paperguard(seed=seed, a1_size=a1_size)
    extra = registry_from_train(train_csv, embedder, seed=seed, **train_kw)
    seen = {" ".join(k.text.split()) for k in base}
    extra = [k for k in extra if " ".join(k.text.split()) not in seen]
    return base + extra


def _family_sort_key(c: str):
    m = re.match(r"A(\d+)", str(c))
    return (int(m.group(1)) if m else 10 ** 6, str(c))


def family_number(c: str) -> Optional[int]:
    m = re.match(r"A(\d+)", str(c))
    return int(m.group(1)) if m else None


# ======================================================================================
# the defense
# ======================================================================================
@dataclass
class Verdict:
    is_attack: bool = False
    n_calls: int = 0
    n_loc_calls: int = 0
    n_segments: int = 0
    n_queries: int = 0
    flag_query: int = -1
    flag_query_name: str = ""
    flag_topk: List[int] = field(default_factory=list)
    flag_topk_sims: List[float] = field(default_factory=list)
    explanation: str = ""
    raw: str = ""
    max_sim: float = float("nan")          # continuous retrieval score (all queries)
    max_sim_query: int = -1
    parse: Dict[str, int] = field(default_factory=lambda: {"ok": 0, "lenient": 0,
                                                            "fail": 0})
    n_errors: int = 0
    located: List[int] = field(default_factory=list)
    loc_method: str = ""
    n_seg_truncated_in_prompt: int = 0


class PaperGuardEmbSearch:
    def __init__(self, registry: Sequence[KnownAttack], embedder: Embedder, *,
                 top_k: int = 5, max_len: int = 800, max_seg_chars_in_prompt: int = 6000):
        self.registry = list(registry)
        self.queries = [a.text for a in self.registry]
        self.embedder = embedder
        self.top_k = top_k
        self.max_len = max_len
        # A pathological chunk (a flattened table with no sentence punctuation) can run to
        # tens of thousands of characters, and five of them overflow a 16k-token context
        # and 400 the request. Only such chunks are shortened in the PROMPT, and the number
        # of times it happened is reported. Ordinary chunks are <= 800 characters.
        self.max_seg_chars_in_prompt = max_seg_chars_in_prompt
        self._qry_emb: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ embedding
    def segments(self, document: str) -> List[str]:
        return O.split_text(document, max_len=self.max_len)

    def query_embeddings(self) -> np.ndarray:
        if self._qry_emb is None:
            self._qry_emb = self.embedder.embed(self.queries)
        return self._qry_emb

    def embed_documents(self, documents: Sequence[str], log_every: int = 50
                        ) -> List[Tuple[List[str], np.ndarray]]:
        """Segment + embed every document up front, in large batches (main thread)."""
        self.query_embeddings()
        out: List[Tuple[List[str], np.ndarray]] = []
        segs_all = [self.segments(d) for d in documents]
        flat = [s for segs in segs_all for s in segs]
        print(f"[embed] {len(documents)} documents -> {len(flat)} chunks "
              f"({len(set(flat))} unique) with {self.embedder.cache_model_key}")
        # embed in slices so progress is visible and a crash loses little (cache)
        uniq = list(dict.fromkeys(flat))
        step = 4096
        for i in range(0, len(uniq), step):
            self.embedder.embed(uniq[i:i + step])
            print(f"[embed]   {min(i + step, len(uniq))}/{len(uniq)} unique chunks "
                  f"(api calls {self.embedder.n_api_calls}, cache hits "
                  f"{self.embedder.n_cache_hits})", flush=True)
        for segs in segs_all:
            if not segs:
                out.append((segs, np.zeros((0, 1), dtype=np.float32)))
            else:
                out.append((segs, self.embedder.embed(segs)))
        return out

    # ------------------------------------------------------------------ verification
    def _verify(self, verifier: Callable[[str], str], qry: str, seg_texts: List[str],
                v: Verdict) -> int:
        shown = []
        for t in seg_texts:
            if len(t) > self.max_seg_chars_in_prompt:
                v.n_seg_truncated_in_prompt += 1
                t = t[: self.max_seg_chars_in_prompt]
            shown.append(t)
        prompt = O.verifier_prompt(qry, shown)
        try:
            raw = verifier(prompt)
        except Exception as e:
            v.n_errors += 1
            v.explanation = f"verifier error: {type(e).__name__}: {str(e)[:300]}"
            return 0
        unsafe, expl, status = O.parse_assessment_lenient(raw)
        v.parse[status] = v.parse.get(status, 0) + 1
        if unsafe:
            v.raw = (raw or "")[:2000]
            v.explanation = expl[:1000]
        return unsafe

    def detect(self, segs: List[str], seg_emb: np.ndarray,
               verifier: Callable[[str], str], *, localize: bool = True) -> Verdict:
        v = Verdict(n_segments=len(segs), n_queries=len(self.queries))
        if not segs:
            return v
        idx, sim = retrieve_topk(seg_emb, self.query_embeddings(), self.top_k)
        top1 = sim[:, 0]
        v.max_sim = float(np.max(top1))
        v.max_sim_query = int(np.argmax(top1))
        for qi, qry in enumerate(self.queries):
            seg_texts = [segs[si] for si in idx[qi].tolist()]
            unsafe = self._verify(verifier, qry, seg_texts, v)
            v.n_calls += 1
            if unsafe == 1:
                v.is_attack = True
                v.flag_query = qi
                v.flag_query_name = self.registry[qi].name
                v.flag_topk = [int(x) for x in idx[qi].tolist()]
                v.flag_topk_sims = [round(float(x), 6) for x in sim[qi].tolist()]
                break
        if v.is_attack and localize:
            self.localize(segs, verifier, v)
        return v

    def localize(self, segs: List[str], verifier: Callable[[str], str], v: Verdict) -> None:
        qry = self.queries[v.flag_query]
        hits = []
        for si in v.flag_topk:
            probe = Verdict()
            u = self._verify(verifier, qry, [segs[si]], probe)
            v.n_loc_calls += 1
            v.n_errors += probe.n_errors
            for k, n in probe.parse.items():
                v.parse[k] = v.parse.get(k, 0) + n
            if u == 1:
                hits.append(si)
        if hits:
            v.located, v.loc_method = sorted(hits), "verified"
        else:
            v.located, v.loc_method = [v.flag_topk[0]], "top1_fallback"


# ======================================================================================
# gold regions, located regions, sanitising — word level, whitespace-insensitive
# ======================================================================================
def _words_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    ws, sp = [], []
    for m in re.finditer(r"\S+", text or ""):
        ws.append(m.group(0))
        sp.append((m.start(), m.end()))
    return ws, sp


def gold_word_mask(clean_doc: str, attacked_doc: str) -> np.ndarray:
    """
    Boolean mask over the ATTACKED document's words: True where the word is not in the
    clean document at that position (difflib 'insert'/'replace' opcodes). Because the two
    documents are rendered from the same anchor, the diff is the attack.
    """
    import difflib
    a, _ = _words_with_spans(clean_doc)
    b, _ = _words_with_spans(attacked_doc)
    mask = np.zeros(len(b), dtype=bool)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            mask[j1:j2] = True
    return mask


def segment_word_ranges(document: str, segments: Sequence[str]
                        ) -> List[Optional[Tuple[int, int]]]:
    """
    Word index range [start, end) of each chunk inside `document`.

    _split_text joins a paragraph's sentences with single spaces and strips, so a chunk is
    a contiguous run of the document's words. Searched in order, from where the previous
    chunk ended, so a repeated sentence maps to the right occurrence.
    """
    words, _ = _words_with_spans(document)
    out: List[Optional[Tuple[int, int]]] = []
    pos = 0
    n = len(words)
    for s in segments:
        sw = s.split()
        L = len(sw)
        found = None
        if L:
            for start in list(range(pos, n - L + 1)) + list(range(0, min(pos, n - L + 1))):
                if words[start] == sw[0] and words[start:start + L] == sw:
                    found = (start, start + L)
                    break
        out.append(found)
        if found:
            pos = found[1]
    return out


def located_word_mask(document: str, segments: Sequence[str], located: Sequence[int]
                      ) -> np.ndarray:
    words, _ = _words_with_spans(document)
    mask = np.zeros(len(words), dtype=bool)
    rng = segment_word_ranges(document, segments)
    for si in located:
        if 0 <= si < len(rng) and rng[si]:
            a, b = rng[si]
            mask[a:b] = True
    return mask


def sanitize(document: str, segments: Sequence[str], located: Sequence[int]
             ) -> Tuple[str, int]:
    """
    The document with the located chunks cut out of the ORIGINAL text (line breaks and
    everything else outside the cut are untouched). Returns (sanitised, n_chars_removed).
    """
    words, spans = _words_with_spans(document)
    rng = segment_word_ranges(document, segments)
    cuts = []
    for si in sorted(set(located)):
        if 0 <= si < len(rng) and rng[si]:
            a, b = rng[si]
            cuts.append((spans[a][0], spans[b - 1][1]))
    if not cuts:
        return document, 0
    cuts.sort()
    merged = [list(cuts[0])]
    for s, e in cuts[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out, last, removed = [], 0, 0
    for s, e in merged:
        out.append(document[last:s])
        removed += e - s
        last = e
    out.append(document[last:])
    return "".join(out), removed


def localization_scores(gold: np.ndarray, loc: np.ndarray) -> Dict[str, float]:
    n = min(len(gold), len(loc))
    g, l = gold[:n], loc[:n]
    inter = int((g & l).sum())
    ng, nl = int(g.sum()), int(l.sum())
    return {"gold_words": ng, "located_words": nl, "overlap_words": inter,
            "loc_precision": inter / nl if nl else float("nan"),
            "loc_recall": inter / ng if ng else float("nan"),
            "loc_hit": float(inter > 0) if ng else float("nan")}


def to_jsonable(v: Verdict) -> Dict:
    d = dict(v.__dict__)
    for k in ("max_sim",):
        if isinstance(d[k], float) and d[k] != d[k]:
            d[k] = None
    return d
