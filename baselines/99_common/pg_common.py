"""
pg_common.py — shared plumbing for the PIGuard baseline.

LAYOUT
    code + outputs   .../attack_def_lim_peer_rev/defense_baselines/piguard      (PG_HOME)
    SFT/DPO code     .../attack_def_lim_peer_rev/defense/llim_agents_v4_SFT_DPO  (SFTDPO_DIR)

The SFT/DPO directory is put on sys.path here, and these modules are imported from it on
purpose (never copied):

  * ft_config        paths, the reviewer model table, the TEST csv
  * infer_ladder     load_eval_set() and metrics(): the evaluation corpus and the Panel-A metric
                     definitions are the ones your SFT/DPO rows were scored with. The PIGuard rows
                     are therefore computed on the SAME 990 documents, in the SAME order, with the
                     SAME Acc/Prec/Rec/F1/FPR/HarmRec/Esc/Calls formulas.

Nothing in here re-implements the corpus. If def_dataset.build_eval_set cannot be imported, the
run stops rather than falling back to some other corpus.

Baseline cited:
    Hao Li, Xiaogeng Liu, Ning Zhang, Chaowei Xiao. "PIGuard: Prompt Injection Guardrail via
    Mitigating Overdefense for Free." ACL 2025.  Code: https://github.com/leolee99/PIGuard
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# where things live
# --------------------------------------------------------------------------------------
PG_HOME = os.path.dirname(os.path.abspath(__file__))
_AD_ROOT = os.environ.get(
    "AD_ROOT", ".")
SFTDPO_DIR = os.environ.get("SFTDPO_DIR",
                            os.path.join(_AD_ROOT, "defense", "llim_agents_v4_SFT_DPO"))
if os.path.isdir(SFTDPO_DIR) and SFTDPO_DIR not in sys.path:
    sys.path.append(SFTDPO_DIR)
try:
    import ft_config as C
except ImportError as _e:
    raise SystemExit(f"[FATAL] cannot import ft_config from SFTDPO_DIR={SFTDPO_DIR} ({_e}). "
                     f"The PIGuard baseline reuses the SFT/DPO eval loader and metrics; "
                     f"export SFTDPO_DIR=<llim_agents_v4_SFT_DPO directory>.")

# All PIGuard outputs go under PG_WORK (default: this directory; SMOKE runs use ./smoke).
PG_WORK = os.environ.get("PG_WORK", PG_HOME)

# --------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------
MODELS_ROOT = os.environ.get("PG_MODELS_ROOT", "path/to/anonymous/models")
# The released PIGuard checkpoint (HF: leolee99/PIGuard), downloaded by pg_fetch_models.sh.
PIGUARD_PATH = os.environ.get("PIGUARD_PATH", os.path.join(MODELS_ROOT, "piguard"))
# microsoft/deberta-v3-base: the backbone PIGuard is trained from (train.py, PIGuard.py).
DEBERTA_PATH = os.environ.get("DEBERTA_PATH", os.path.join(MODELS_ROOT, "deberta_v3_base"))
# The unzipped PIGuard GitHub repository (for datasets/train.json and datasets/valid.json).
PIGUARD_REPO = os.environ.get("PIGUARD_REPO", os.path.join(PG_HOME, "PIGuard-main"))
# The SFT/DPO training documents (built by your stage-A rollout).
TRAIN_DOCS = os.environ.get("TRAIN_DOCS", os.path.join(SFTDPO_DIR, "traindata_shared",
                                                       "train_documents.csv"))


def runs_dir() -> str:
    return os.path.join(PG_WORK, "runs")


def train_dir() -> str:
    return os.path.join(PG_WORK, "piguard_train")


# --------------------------------------------------------------------------------------
# the evaluation corpus — imported, never rebuilt
# --------------------------------------------------------------------------------------
def load_eval_recs(benign_csv: str, attacked_csv: str, *, id_column: str = "id",
                   max_papers: int = 10 ** 9, seed: int = 0) -> List:
    """
    The exact record list infer_ladder.py scores: def_dataset.build_eval_set on the same
    benign csv, the same metrics_detail.csv and the same seed. all_papers=True, as in the
    job_c_*_infer.pbs default.
    """
    import infer_ladder as IL
    recs = IL.load_eval_set(benign_csv, attacked_csv, id_column=id_column,
                            max_papers=max_papers, seed=seed, all_papers=True,
                            memory_dir="")
    return list(recs)


def harmful_of(rec) -> int:
    """Byte-for-byte the definition infer_ladder.run_one uses."""
    return int(bool(getattr(rec, "asr", None)))


def pair_partner(recs: Sequence) -> Dict[int, int]:
    """
    attacked index -> index of its clean partner.

    def_dataset.build_eval_set emits (clean, attacked) pairs back to back; the partner is the
    immediately preceding clean row of the same paper. If that ordering were ever broken the
    first clean row of the same paper is used instead, and the mismatch is counted.
    """
    first_clean: Dict[str, int] = {}
    for i, r in enumerate(recs):
        if int(r.label) == 0 and str(r.paper_id) not in first_clean:
            first_clean[str(r.paper_id)] = i
    out: Dict[int, int] = {}
    for i, r in enumerate(recs):
        if int(r.label) != 1:
            continue
        j = i - 1
        if j >= 0 and int(recs[j].label) == 0 and str(recs[j].paper_id) == str(r.paper_id):
            out[i] = j
        elif str(r.paper_id) in first_clean:
            out[i] = first_clean[str(r.paper_id)]
    return out


def a_number(condition: str) -> Optional[int]:
    m = re.match(r"\s*A(\d+)", str(condition or ""))
    return int(m.group(1)) if m else None


# The two threat families of Panel B in your table.
INSTRUCTION_BEARING = {5, 6, 7, 8, 9, 11, 12}
RHETORICAL = {1, 2, 3, 4, 10}


def family_group(condition: str) -> str:
    n = a_number(condition)
    if n in INSTRUCTION_BEARING:
        return "instruction"
    if n in RHETORICAL:
        return "rhetorical"
    return "clean" if n == 0 else "other"


def rec_extras(rec, max_len: int = 200) -> Dict:
    """
    Every short scalar attribute the eval record carries (asr, strategy, ratings, ...), so the
    detail csv keeps whatever def_dataset provides without this code having to know its schema.
    """
    import dataclasses
    out = {}
    if isinstance(rec, dict):
        d = rec
    elif dataclasses.is_dataclass(rec):
        d = {f.name: getattr(rec, f.name, None) for f in dataclasses.fields(rec)}
    elif hasattr(rec, "_asdict"):
        d = rec._asdict()
    else:
        d = getattr(rec, "__dict__", {}) or {}
    for k, v in (d or {}).items():
        if k in ("document", "label", "paper_id", "doc_id", "condition"):
            continue
        if isinstance(v, (int, float, bool)) or v is None:
            out[f"rec_{k}"] = v
        elif isinstance(v, str) and len(v) <= max_len:
            out[f"rec_{k}"] = v
    return out


def sha(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def panel_a(rows: List[Dict]) -> Dict:
    """The SFT/DPO table's metric function, imported so the definitions cannot drift."""
    import infer_ladder as IL
    return {"all": IL.metrics(rows, "all"), "harmful": IL.metrics(rows, "harmful")}


def auroc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """Mann-Whitney AUROC with ties counted as 1/2. None if a class is empty."""
    pos = [float(x) for x in pos if x is not None and not math.isnan(float(x))]
    neg = [float(x) for x in neg if x is not None and not math.isnan(float(x))]
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    # average ranks
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rp = sum(r for r, (_, y) in zip(ranks, allv) if y == 1)
    n1, n0 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def recall_at_fpr(pos: Sequence[float], neg: Sequence[float], fpr: float) -> Optional[float]:
    """
    Recall at the strictest score threshold whose FPR on the clean documents is <= `fpr`.
    Positive iff score > t, where t is the (k+1)-th largest clean score, k = floor(fpr * N_neg).
    Ties at t are therefore counted as negatives — the conservative reading.
    """
    pos = [float(x) for x in pos]
    neg = sorted((float(x) for x in neg), reverse=True)
    if not pos or not neg:
        return None
    k = int(math.floor(fpr * len(neg) + 1e-9))
    t = neg[k] if k < len(neg) else -float("inf")
    return sum(1 for x in pos if x > t) / len(pos)


def panel_b(rows: List[Dict], group: str, score_key: str = "screen_score") -> Dict:
    """
    One Panel-B row, with the SAME conventions as your agent rows:
      positives  = attacked documents of this threat family
      negatives  = ALL clean documents (so FPR is identical across the two families)
      precision  = TP_family / (TP_family + FP_all_clean)
      HarmRec    = recall on the harmful attacked documents of this family
    The score-based columns (R@FPR5, R@FPR10, AUROC) use the continuous guard score.
    """
    pos = [r for r in rows if int(r["label"]) == 1 and family_group(r["condition"]) == group]
    neg = [r for r in rows if int(r["label"]) == 0]
    tp = sum(1 for r in pos if int(r["pred"]) == 1)
    fp = sum(1 for r in neg if int(r["pred"]) == 1)
    tn = len(neg) - fp
    rec = tp / len(pos) if pos else None
    prec = tp / (tp + fp) if (tp + fp) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if pos else None)
    hp = [r for r in pos if int(r.get("harmful", 0)) == 1]
    harm = (sum(1 for r in hp if int(r["pred"]) == 1) / len(hp)) if hp else None

    def sc(rs):
        out = []
        for r in rs:
            v = r.get(score_key)
            try:
                v = float(v)
            except Exception:
                continue
            if not math.isnan(v):
                out.append(v)
        return out
    ps, ns = sc(pos), sc(neg)
    have_scores = len(ps) == len(pos) and len(ns) == len(neg) and ps and ns
    return {"group": group, "n_pos": len(pos), "n_neg": len(neg), "tp": tp, "fp": fp,
            "tn": tn, "recall": rec, "precision": prec, "f1": f1,
            "fpr": fp / len(neg) if neg else None, "harm_n": len(hp), "harm_recall": harm,
            "r_at_fpr5": recall_at_fpr(ps, ns, 0.05) if have_scores else None,
            "r_at_fpr10": recall_at_fpr(ps, ns, 0.10) if have_scores else None,
            "auroc": auroc(ps, ns) if have_scores else None}


# --------------------------------------------------------------------------------------
# one OpenAI-compatible chat client for vLLM (localhost) and api.openai.com
# --------------------------------------------------------------------------------------
class ChatError(RuntimeError):
    pass


class ChatClient:
    """
    Talks to a vLLM server or to the OpenAI API with the same code (both speak
    /v1/chat/completions).

    trust_env is decided by the URL: False for 127.0.0.1/localhost (the cluster proxy cannot
    route loopback — see ft_vllm.py note 1), True for api.openai.com (which NEEDS the proxy).
    One httpx.Client per thread, so the thread pools in pg_review/pg_detect are safe.
    """

    def __init__(self, base_url: str, model: str, *, api_key: str = "",
                 chat_kwargs: Optional[Dict] = None, timeout: float = 900.0,
                 max_retries: int = 4, system_role: bool = True):
        import httpx  # noqa: F401  (fail early if missing)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.chat_kwargs = dict(chat_kwargs or {})
        self.timeout = timeout
        self.max_retries = max_retries
        self.system_role = system_role
        self._local = threading.local()
        self._lock = threading.Lock()
        self.n_calls = 0
        self.n_prompt_tokens = 0
        self.n_completion_tokens = 0
        host = self.base_url.split("//", 1)[-1].split("/", 1)[0].split(":")[0]
        self.trust_env = host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0")
        self.is_openai = "api.openai.com" in self.base_url

    @classmethod
    def for_backend(cls, backend: str, *, url: str = "", model: str = "",
                    model_key: str = "", **kw) -> "ChatClient":
        """backend = 'vllm' (url + served model name) or 'openai' (gpt-4o-mini etc.)."""
        spec = C.MODELS.get(model_key) if model_key else None
        if backend == "openai":
            key = os.environ.get("OPENAI_API_KEY", "")
            if not key:
                raise ChatError("OPENAI_API_KEY is not set (put it in path/to/.keys, not in a PBS file)")
            return cls(url or "https://api.openai.com/v1", model or "gpt-4o-mini",
                       api_key=key, system_role=True, **kw)
        if not url:
            url = os.environ.get("VLLM_URL", "")
        if not url:
            raise ChatError("no vLLM url: pass --vllm-url or export VLLM_URL")
        return cls(url, model or os.environ.get("VLLM_MODEL", "") or (spec.path if spec else ""),
                   chat_kwargs=(spec.chat_kwargs if spec else None),
                   system_role=(spec.system_role if spec else True), **kw)

    def _client(self):
        import httpx
        c = getattr(self._local, "c", None)
        if c is None:
            c = httpx.Client(timeout=self.timeout, trust_env=self.trust_env)
            self._local.c = c
        return c

    def messages(self, system: str, user: str) -> List[Dict[str, str]]:
        # Gemma has no system role (ft_roles.messages): merge, exactly as the SFT/DPO code does.
        if system and self.system_role:
            return [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return [{"role": "user", "content": (system + "\n\n" + user) if system else user}]

    def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.0,
             max_tokens: int = 512, seed: Optional[int] = 0, top_p: float = 1.0,
             logprobs: int = 0) -> Tuple[str, Optional[List]]:
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "top_p": top_p, "max_tokens": max_tokens}
        if seed is not None:
            body["seed"] = int(seed)
        if self.chat_kwargs and not self.is_openai:
            body["chat_template_kwargs"] = self.chat_kwargs
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = int(logprobs)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        last = None
        for attempt in range(self.max_retries):
            try:
                r = self._client().post(f"{self.base_url}/chat/completions", json=body,
                                        headers=headers)
                if r.status_code == 404:
                    raise ChatError(f"404 for model {self.model!r} at {self.base_url}")
                if r.status_code == 400 and logprobs:
                    # some servers reject logprobs; retry once without them
                    body.pop("logprobs", None)
                    body.pop("top_logprobs", None)
                    logprobs = 0
                    continue
                if r.status_code == 400:
                    # not transient (usually: prompt longer than the context window)
                    raise ChatError(f"400 Bad Request: {r.text[:400]}")
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                d = r.json()
                with self._lock:
                    self.n_calls += 1
                    u = d.get("usage") or {}
                    self.n_prompt_tokens += int(u.get("prompt_tokens") or 0)
                    self.n_completion_tokens += int(u.get("completion_tokens") or 0)
                ch = d["choices"][0]
                lp = None
                if logprobs and ch.get("logprobs"):
                    lp = (ch["logprobs"] or {}).get("content")
                return ch["message"].get("content") or "", lp
            except ChatError:
                raise
            except Exception as e:  # transport, 5xx, 429
                last = e
                time.sleep(min(60.0, 3.0 * (2 ** attempt)))
        raise ChatError(f"chat failed after {self.max_retries} attempts: "
                        f"{type(last).__name__}: {last}")


def extract_json(text: str) -> Optional[dict]:
    """ft_vllm.extract_json, re-exported so every stage parses completions the same way."""
    import ft_vllm as V
    return V.extract_json(text)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: None)
    os.replace(tmp, path)
