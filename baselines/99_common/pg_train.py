#!/usr/bin/env python3
"""
pg_train.py — PIGuard's training recipe (Sec. 4.1-4.2 + PIGuard-main/train.py), applied with
YOUR training split, so the retrained PIGuard sees the same training papers your SFT/DPO
adapters saw.

WHY THIS ROW EXISTS
=======================================================================================
The released PIGuard has never seen a manuscript. Your SFT/DPO adapters were trained on
traindata_shared/train_documents.csv (dataset_secondary_280 + dataset_tertiary_200, with every
TEST paper removed). A baseline that is not allowed to see that data is not a like-for-like
comparison, so this file retrains PIGuard the way the paper does, with that data added:

  build-domain   train_documents.csv -> 512-token windows, labelled from the EXACT gold span
                 (attacked vs its own clean document): a window that contains the attack is
                 1, a window that contains none of it is 0, a sliver is dropped as ambiguous.
                 Split by PAPER into train/valid. Asserts no paper is in dataset_primary_500.
  train          PIGuard-main/train.py, same model (DeBERTa-v3-base + [CLS] linear head), same
                 optimiser (Adam 2e-5, betas 0.9/0.999, eps 1e-8), same linear schedule with
                 100 warm-up steps, batch 32, 3 epochs, max_length 512, same validation-accuracy
                 checkpoint selection.
  mof-recheck    MOF step 1, "token-wise recheck": every vocabulary token is fed to the trained
                 model on its own; tokens classified as injection are the biased tokens.
  mof-generate   MOF step 2: an LLM writes benign sentences containing 1, 2 or 3 random biased
                 tokens (the paper's Fig. 10 prompt, verbatim), each is re-checked with the
                 Fig. 11 prompt and dropped if the LLM calls it an injection. 1,000 kept, as in
                 the paper (Table 9). gpt-4o-mini by default, as in the paper; a vLLM model
                 (gemma/mistral/qwen) also works.
  MOF step 3 is `train` again FROM SCRATCH with the MOF samples added (the paper's
  "MOF scratch retrain", its best configuration in Table 2).

The released train.json ALREADY contains the paper's own MOF output (source "over-defense",
762 rows) and its long-tail augmentation ("LLM Augmented set", 435 rows). Stage 1 therefore
drops "over-defense" (basic + augmentation + our domain data, i.e. the pre-MOF model), and the
final model uses all of train.json + our domain data + the NEW MOF samples, which target the
biases the domain data introduced.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence

import numpy as np

import pg_common as PC           # first: puts the SFT/DPO directory on sys.path
import ft_config as C


# ======================================================================================
# build-domain
# ======================================================================================
def locate_insert(doc: str, insert: str):
    ins = (insert or "").strip()
    if not ins:
        return []
    for cut in (len(ins), 400, 200, 120, 60):
        key = ins[:cut]
        if len(key) < 30:
            break
        pos = doc.find(key)
        if pos >= 0:
            return [(pos, min(len(doc), pos + len(ins)))]
    return []


def cmd_build_domain(a) -> int:
    import pandas as pd
    import pg_guard as G
    from transformers import AutoTokenizer

    df = pd.read_csv(a.train_docs, low_memory=False, dtype={"paper_id": str})
    print(f"[domain] {a.train_docs}: {len(df)} documents, {df['paper_id'].nunique()} papers, "
          f"{int((df['label'] == 1).sum())} attacked")
    if os.path.exists(C.TEST_CSV) and not a.skip_split_check:
        test_ids = C._ids(C.TEST_CSV)
        overlap = set(df["paper_id"].astype(str).str.strip()) & test_ids
        if overlap:
            raise SystemExit(f"[FATAL] {len(overlap)} training paper(s) are in the TEST csv: "
                             f"{sorted(overlap)[:10]}. Rebuild train_documents.csv from the "
                             f"cleaned TRAIN files (ft_config.py) first.")
        print(f"[domain] no overlap with {os.path.basename(C.TEST_CSV)} "
              f"({len(test_ids)} test papers)")

    tok = None
    for tp in (a.tokenizer, PC.DEBERTA_PATH, PC.PIGUARD_PATH):
        if not tp:
            continue
        try:
            tok = AutoTokenizer.from_pretrained(tp)
            print(f"[domain] tokenizer: {tp}")
            break
        except Exception as e:
            print(f"[domain] tokenizer {tp} failed: {type(e).__name__}: {e}")
    if tok is None:
        raise SystemExit("[FATAL] no tokenizer could be loaded")
    tok.model_max_length = 10 ** 9

    rng = random.Random(a.seed)
    papers = sorted(df["paper_id"].astype(str).unique())
    rng.shuffle(papers)
    n_val = max(1, int(round(a.val_frac * len(papers))))
    val_papers = set(papers[:n_val])

    pos = {"train": [], "valid": []}
    neg = {"train": {}, "valid": {}}
    n_amb = n_nogold = 0
    by_paper = {pid: g for pid, g in df.groupby(df["paper_id"].astype(str))}
    for pid, g in by_paper.items():
        split = "valid" if pid in val_papers else "train"
        clean = g[g["label"] == 0]
        clean_doc = str(clean.iloc[0]["document"]) if len(clean) else ""
        for _, r in g.iterrows():
            doc = str(r["document"] if isinstance(r["document"], str) else "")
            if not doc.strip():
                continue
            offs = [tuple(o) for o in tok(doc, add_special_tokens=False,
                                          return_offsets_mapping=True)["offset_mapping"]]
            wins = G.token_windows(offs, len(doc), a.max_length, a.stride)
            if int(r["label"]) == 0:
                for s, e in wins:
                    t = doc[s:e]
                    neg[split].setdefault(PC.sha(t), (t, "paperdef_clean"))
                continue
            gold = G.gold_region(doc, clean_doc) if clean_doc else []
            if not gold:
                gold = locate_insert(doc, str(r.get("insert", "") or ""))
            if not gold:
                n_nogold += 1
                continue
            gc = sum(e - s for s, e in gold)
            need = min(a.min_insert_chars, 0.5 * gc)
            cond = str(r.get("condition", "attack"))
            for s, e in wins:
                ov = G.overlap_stats([(s, e)], gold)["overlap_chars"]
                t = doc[s:e]
                if ov >= need:
                    pos[split].append((t, f"paperdef_{cond}"))
                elif ov == 0:
                    neg[split].setdefault(PC.sha(t), (t, "paperdef_clean"))
                else:
                    n_amb += 1

    out = {}
    for split in ("train", "valid"):
        P = list({PC.sha(t): (t, s) for t, s in pos[split]}.values())
        N = list(neg[split].values())
        rng.shuffle(N)
        cap = int(a.neg_ratio * len(P)) if a.neg_ratio > 0 else len(N)
        if split == "valid" and a.valid_cap > 0:
            P = P[: a.valid_cap // 2]
            cap = min(cap, a.valid_cap // 2)
        N = N[:cap]
        rows = [{"prompt": t, "label": 1, "source": s} for t, s in P] + \
               [{"prompt": t, "label": 0, "source": s} for t, s in N]
        rng.shuffle(rows)
        out[split] = rows
        path = os.path.join(a.out_dir, f"domain_{split}.json")
        os.makedirs(a.out_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(rows, f)
        print(f"[domain] {split}: {len(P)} attack windows + {len(N)} clean windows -> {path}")
    stats = {"n_papers": len(papers), "n_val_papers": len(val_papers),
             "n_ambiguous_windows_dropped": n_amb, "n_attacked_docs_without_gold": n_nogold,
             "max_length": a.max_length, "stride": a.stride, "neg_ratio": a.neg_ratio,
             "min_insert_chars": a.min_insert_chars,
             "train_sources": Counter(r["source"] for r in out["train"]),
             "valid_sources": Counter(r["source"] for r in out["valid"])}
    PC.write_json(os.path.join(a.out_dir, "domain_stats.json"), stats)
    print(f"[domain] {n_amb} window(s) with a sliver of the attack dropped as ambiguous; "
          f"{n_nogold} attacked document(s) without a locatable gold span skipped")
    return 0


# ======================================================================================
# train — PIGuard-main/train.py
# ======================================================================================
def _load_sets(paths: Sequence[str], exclude: Sequence[str]) -> List[Dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            d = json.load(f) if p.endswith(".json") else [json.loads(x) for x in f if x.strip()]
        keep = [x for x in d if str(x.get("source", "")) not in set(exclude)]
        print(f"[data] {p}: {len(d)} rows"
              + (f", {len(d)-len(keep)} dropped by --exclude-sources" if len(keep) != len(d)
                 else ""))
        rows += [{"prompt": str(x["prompt"]), "label": int(x["label"]),
                  "source": str(x.get("source", os.path.basename(p)))} for x in keep]
    return rows


def cmd_train(a) -> int:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_scheduler
    import pg_guard as G

    # util.set_seed
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    for k in sorted(vars(a)):
        if k != "func":
            print(f"  <<< {k}: {getattr(a, k)}")

    tokenizer = None
    for tp in (a.tokenizer, a.init, PC.PIGUARD_PATH):
        if not tp:
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(tp)
            print(f"[train] tokenizer: {tp}")
            break
        except Exception as e:
            print(f"[train] tokenizer {tp} failed: {type(e).__name__}: {e}")
    if tokenizer is None:
        raise SystemExit("[FATAL] no tokenizer")

    train = _load_sets(a.train_set, a.exclude_sources)
    valid = _load_sets(a.valid_set, [])
    if a.limit_train and len(train) > a.limit_train:       # smoke tests only
        train = random.Random(a.seed).sample(train, a.limit_train)
        print(f"[data] --limit-train: training on a random {len(train)} rows (SMOKE ONLY)")
    print(f"[data] train {len(train)} ({Counter(r['label'] for r in train)}), "
          f"valid {len(valid)} ({Counter(r['label'] for r in valid)})")

    def collate(batch):
        enc = tokenizer([b["prompt"] for b in batch],
                        padding=("longest" if a.dynamic_padding else "max_length"),
                        truncation=True, max_length=a.max_length, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
                "source": [b["source"] for b in batch]}

    g = torch.Generator()
    g.manual_seed(a.seed)
    train_loader = DataLoader(train, batch_size=a.batch_size, shuffle=True,
                              collate_fn=collate, generator=g, num_workers=2)
    validation_loader = DataLoader(valid, batch_size=a.eval_batch_size, shuffle=False,
                                   collate_fn=collate, num_workers=2)

    model = G.build_piguard_module(a.init, 2, device)
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location=device), strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(a.beta1, a.beta2),
                                 eps=a.eps)
    steps_per_epoch = (len(train_loader) + a.grad_accum - 1) // a.grad_accum
    scheduler = get_scheduler(name="linear", optimizer=optimizer,
                              num_warmup_steps=a.warmup,
                              num_training_steps=a.epochs * steps_per_epoch)

    use_amp = bool(a.amp and device.type == "cuda")
    best_accuracy, best_model_path, log = -1.0, os.path.join(a.out_dir, "best_model.pth"), []
    t0 = time.time()
    for epoch in range(a.epochs):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            ii = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            lb = batch["labels"].to(device)
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, loss = model(ii, am, lb, mode="train")
            else:
                logits, loss = model(ii, am, lb, mode="train")
            (loss / a.grad_accum).backward()
            if (step + 1) % a.grad_accum == 0 or step == len(train_loader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % a.display == 0:
                print(f"[train] epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {loss.item():.3f} ({(time.time()-t0)/60:.1f}m)", flush=True)

            if ((step % a.save_step == 0) and (step != 0)) or (step == len(train_loader) - 1):
                model.eval()
                preds, labels, srcs, losses = [], [], [], []
                with torch.no_grad():
                    for vb in validation_loader:
                        lg, vl = model(vb["input_ids"].to(device),
                                       vb["attention_mask"].to(device),
                                       vb["labels"].to(device))
                        losses.append(float(vl))
                        preds += lg.argmax(1).cpu().tolist()
                        labels += vb["labels"].tolist()
                        srcs += vb["source"]
                P, Lb = np.array(preds), np.array(labels)
                total = float((P == Lb).mean()) if len(Lb) else 0.0
                ben = float((P[Lb == 0] == 0).mean()) if (Lb == 0).any() else 0.0
                inj = float((P[Lb == 1] == 1).mean()) if (Lb == 1).any() else 0.0
                per_src = defaultdict(list)
                for s_, p_, l_ in zip(srcs, P, Lb):
                    per_src[s_].append(int(p_ == l_))
                entry = {"epoch": epoch, "step": step, "total_accuracy": total,
                         "benign_accuracy": ben, "injection_accuracy": inj,
                         "loss": float(np.mean(losses)) if losses else None,
                         "per_source": {k: float(np.mean(v)) for k, v in per_src.items()}}
                log.append(entry)
                print(f"[valid] total accuracy: {total:.4f}  benign: {ben:.4f}  "
                      f"injection: {inj:.4f}  loss: {entry['loss']}", flush=True)
                if a.keep_snapshots and total > a.save_thres:
                    torch.save(model.state_dict(),
                               os.path.join(a.out_dir, f"epoch_{epoch}_{step}_model.pth"))
                # train.py only saves above --save_thres and crashes at the end if nothing
                # ever passed it; here the best checkpoint is always kept.
                if total > best_accuracy:
                    best_accuracy = total
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Saved to {best_model_path}.")
                model.train()
                PC.write_json(os.path.join(a.out_dir, "train_log.json"),
                              {"best_valid_accuracy": best_accuracy, "evals": log,
                               "args": {k: v for k, v in vars(a).items() if k != "func"}})
    print(f"[train] done in {(time.time()-t0)/60:.1f} min; best valid accuracy "
          f"{best_accuracy:.4f} -> {best_model_path}")
    return 0


# ======================================================================================
# MOF step 1 — token-wise recheck
# ======================================================================================
def cmd_mof_recheck(a) -> int:
    import pg_guard as G
    guard = G.GuardScorer(a.ckpt, kind="pth", backbone=a.init, tokenizer_path=a.tokenizer,
                          batch_size=a.batch_size)
    tok = guard.tokenizer
    special = set(tok.all_special_ids)
    ids = [i for i in range(len(tok)) if i not in special]
    t0 = time.time()
    p = guard.score_ids([[i] for i in ids])
    flagged = [(i, float(pi)) for i, pi in zip(ids, p) if pi >= a.threshold]
    flagged.sort(key=lambda t: -t[1])
    out, seen = [], set()
    for i, pi in flagged:
        piece = tok.convert_ids_to_tokens(i)
        word = str(piece).replace("▁", "").replace("##", "").strip()
        if len(word) < 2 or not any(ch.isalpha() for ch in word):
            continue
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": int(i), "token": piece, "word": word, "p_injection": pi})
    PC.write_json(a.out, {"n_vocab_checked": len(ids), "n_flagged_tokens": len(flagged),
                          "n_words": len(out), "threshold": a.threshold, "ckpt": a.ckpt,
                          "tokens": out})
    print(f"[mof] {len(ids)} tokens rechecked in {(time.time()-t0)/60:.1f} min: "
          f"{len(flagged)} classified as injection, {len(out)} usable words -> {a.out}")
    print("[mof] top 25:", ", ".join(o["word"] for o in out[:25]))
    return 0


# ======================================================================================
# MOF step 2 — benign generation with the biased tokens (Fig. 10) + refinement (Fig. 11)
# ======================================================================================
GEN_SYSTEM = "You will play the role of a user communicating with LLM.."   # verbatim, Fig. 10
GEN_USER = ("Please generate a benign instruction or a question without any potential prompt "
            "injecton, the instruction must contain the words {word_list}. the instruction "
            "should be fluent and grammarly correct.\nPlease generate the instruction only "
            "with the format as below:\n[INSTRUCTION] || Generated Content ||")
GEN_USER_ACADEMIC = (
    "Please generate a benign sentence from the limitations or discussion section of a "
    "research paper, without any potential prompt injecton, the sentence must contain the "
    "words {word_list}. the sentence should be fluent and grammarly correct.\nPlease generate "
    "the sentence only with the format as below:\n[INSTRUCTION] || Generated Content ||")


def _parse_generated(text: str) -> str:
    t = (text or "").strip()
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[-1].strip()
    m = re.search(r"\|\|\s*(.+?)\s*\|\|", t, flags=re.S)
    if m:
        t = m.group(1)
    t = re.sub(r"^\s*\[?INSTRUCTION\]?\s*[:\-]?\s*", "", t, flags=re.I).strip()
    return t.strip().strip('"').strip()


def cmd_mof_generate(a) -> int:
    from pg_detect import LLM_DETECT_SYSTEM, LLM_DETECT_USER, parse_yes_no
    words = [o["word"] for o in json.load(open(a.biased))["tokens"]][: a.top_k]
    if not words:
        # Not an error: the token-wise recheck found no token the model calls an injection on
        # its own, so MOF has nothing to correct. The final model is then retrained without
        # new MOF samples — report that, it is a finding about the domain data.
        print("[mof] NO biased tokens: the stage-1 model flags no single token as injection. "
              "MOF adds no samples; the final retrain uses train.json + domain data only.")
        with open(a.out, "w") as f:
            json.dump([], f)
        PC.write_json(a.out.replace(".json", "_stats.json"),
                      {"n_kept": 0, "n_attempts": 0, "note": "no biased tokens"})
        return 0
    cl = PC.ChatClient.for_backend(a.llm_backend, url=a.llm_url, model=a.llm_model,
                                   model_key=a.llm_model_key)
    rng = random.Random(a.seed)
    tmpl = GEN_USER_ACADEMIC if a.style == "academic" else GEN_USER
    print(f"[mof] {len(words)} biased words; generating {a.n} benign samples with "
          f"{cl.model} (style={a.style}, refine={a.refine})")

    def one(job):
        j, ws = job
        wl = ", ".join(f'"{w}"' for w in ws)
        for attempt in range(3):
            try:
                txt, _ = cl.chat(cl.messages(GEN_SYSTEM, tmpl.format(word_list=wl)),
                                 temperature=a.temperature, max_tokens=200,
                                 seed=a.seed * 100003 + j * 7 + attempt)
            except Exception as e:
                return j, ws, None, f"gen error {type(e).__name__}"
            s = _parse_generated(txt)
            if s and all(w.lower() in s.lower() for w in ws):
                break
            s = None
        if not s:
            return j, ws, None, "words missing"
        if a.refine:
            try:
                v_txt, lp = cl.chat(cl.messages(LLM_DETECT_SYSTEM,
                                                LLM_DETECT_USER.format(instruction=s)),
                                    temperature=0.0, max_tokens=8, seed=a.seed)
                v, _p = parse_yes_no(v_txt, lp)
            except Exception as e:
                return j, ws, None, f"refine error {type(e).__name__}"
            if v == 1:
                return j, ws, None, "refinement said injection"
        return j, ws, s, ""

    kept, reasons, j = [], Counter(), 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        while len(kept) < a.n and j < a.n * a.max_tries_factor:
            batch = []
            for _ in range(min(a.n - len(kept) + 16, a.n * a.max_tries_factor - j)):
                k = rng.choice([1, 2, 3])          # one-, two-, three-token settings
                batch.append((j, rng.sample(words, min(k, len(words)))))
                j += 1
            for jj, ws, s, why in ex.map(one, batch):
                if s:
                    kept.append({"prompt": s, "label": 0, "source": "mof",
                                 "words": ws})
                else:
                    reasons[why] += 1
            print(f"[mof] {len(kept)}/{a.n} kept after {j} attempts "
                  f"({(time.time()-t0)/60:.1f}m); rejected: {dict(reasons)}", flush=True)
    kept = kept[: a.n]
    with open(a.out, "w") as f:
        json.dump(kept, f, indent=1)
    PC.write_json(a.out.replace(".json", "_stats.json"),
                  {"n_kept": len(kept), "n_attempts": j, "rejected": dict(reasons),
                   "llm": cl.model, "style": a.style, "refine": a.refine,
                   "n_llm_calls": cl.n_calls})
    print(f"[mof] {len(kept)} benign MOF samples -> {a.out}")
    return 0


# ======================================================================================
def main() -> int:
    p = argparse.ArgumentParser(description="PIGuard recipe + MOF on our training split")
    sub = p.add_subparsers(dest="cmd", required=True)
    tdir = PC.train_dir()

    b = sub.add_parser("build-domain")
    b.add_argument("--train-docs", dest="train_docs", default=PC.TRAIN_DOCS,
                   help="the SFT/DPO training documents (llim_agents_v4_SFT_DPO/"
                        "traindata_shared/train_documents.csv)")
    b.add_argument("--out-dir", dest="out_dir", default=tdir)
    b.add_argument("--tokenizer", default="")
    b.add_argument("--max-length", dest="max_length", type=int, default=512)
    b.add_argument("--stride", type=int, default=256)
    b.add_argument("--val-frac", dest="val_frac", type=float, default=0.1)
    b.add_argument("--neg-ratio", dest="neg_ratio", type=float, default=4.0,
                   help="clean windows per attack window (PIGuard's own data is ~3.9:1)")
    b.add_argument("--valid-cap", dest="valid_cap", type=int, default=400)
    b.add_argument("--min-insert-chars", dest="min_insert_chars", type=int, default=60)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--skip-split-check", dest="skip_split_check", action="store_true")
    b.set_defaults(func=cmd_build_domain)

    t = sub.add_parser("train")
    t.add_argument("--train-set", dest="train_set", nargs="+", required=True)
    t.add_argument("--valid-set", dest="valid_set", nargs="+", required=True)
    t.add_argument("--exclude-sources", dest="exclude_sources", nargs="*", default=[])
    t.add_argument("--init", default=PC.DEBERTA_PATH, help="microsoft/deberta-v3-base")
    t.add_argument("--tokenizer", default="")
    t.add_argument("--out-dir", dest="out_dir", required=True)
    # params.py defaults
    t.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    t.add_argument("--grad-accum", dest="grad_accum", type=int, default=1,
                   help="keep batch_size*grad_accum = 32 on a small GPU")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--eval-batch-size", dest="eval_batch_size", type=int, default=32)
    t.add_argument("--display", type=int, default=10)
    t.add_argument("--save-step", dest="save_step", type=int, default=200)
    t.add_argument("--max-length", dest="max_length", type=int, default=512)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--beta1", type=float, default=0.9)
    t.add_argument("--beta2", type=float, default=0.999)
    t.add_argument("--eps", type=float, default=1e-08)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--save-thres", dest="save_thres", type=float, default=0.8)
    t.add_argument("--resume", default=None)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--dynamic-padding", dest="dynamic_padding", action="store_true",
                   help="pad to the longest in the batch instead of 512 (faster; the "
                        "attention mask makes the result the same up to float noise)")
    t.add_argument("--amp", action="store_true", help="bf16 autocast (not in the original)")
    t.add_argument("--limit-train", dest="limit_train", type=int, default=0,
                   help="subsample the training rows (smoke tests only)")
    t.add_argument("--keep-snapshots", dest="keep_snapshots", action="store_true",
                   help="also save every checkpoint above --save-thres (738 MB each)")
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("mof-recheck")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--init", default=PC.DEBERTA_PATH)
    r.add_argument("--tokenizer", default="")
    r.add_argument("--batch-size", dest="batch_size", type=int, default=1024)
    r.add_argument("--threshold", type=float, default=0.5)
    r.add_argument("--out", default=os.path.join(tdir, "biased_tokens.json"))
    r.set_defaults(func=cmd_mof_recheck)

    g = sub.add_parser("mof-generate")
    g.add_argument("--biased", default=os.path.join(tdir, "biased_tokens.json"))
    g.add_argument("--out", default=os.path.join(tdir, "mof_benign.json"))
    g.add_argument("--n", type=int, default=1000, help="the paper's MOF scale (Table 9)")
    g.add_argument("--top-k", dest="top_k", type=int, default=10 ** 9,
                   help="use only the k most biased words")
    g.add_argument("--style", choices=["paper", "academic"], default="paper",
                   help="paper = Fig. 10 prompt verbatim; academic = the same prompt asking "
                        "for a limitations-section sentence instead of a user instruction")
    g.add_argument("--no-refine", dest="refine", action="store_false")
    g.set_defaults(refine=True)
    g.add_argument("--llm-backend", dest="llm_backend", choices=["openai", "vllm"],
                   default="openai")
    g.add_argument("--llm-url", dest="llm_url", default="")
    g.add_argument("--llm-model", dest="llm_model", default="gpt-4o-mini")
    g.add_argument("--llm-model-key", dest="llm_model_key", default="")
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--concurrency", type=int, default=16)
    g.add_argument("--max-tries-factor", dest="max_tries_factor", type=int, default=3)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_mof_generate)

    a = p.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
