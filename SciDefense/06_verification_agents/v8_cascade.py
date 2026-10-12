"""
v8_cascade.py — the inference pipeline, as step functions the rollouts reuse verbatim.

    signals -> LOCATOR (per window) -> items
            -> gate: no items -> CLEAN, stop (1-3 calls; most clean papers end here)
            -> EVIDENCE (one call, all items) -> VERIFIER (one call, all items, blind)
            -> LEADER detect -> [optional AUDITOR on uncertain verdicts]
    flagged -> LEADER localize -> excise + normalize -> repaired document

Every step is a function of (document, upstream outputs), so rollouts can sample k
completions at one step while holding the upstream steps at their greedy outputs -- the
same conditioning the agent sees at inference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import v8_agents as A
import v8_tools as T
from v8_llm import Backend, Reviewer, Routing


@dataclass
class CascadeConfig:
    system_role: bool = True
    window_chars: int = 3200
    overlap: int = 600
    tail_chars: int = 6000
    max_windows: int = 3
    max_items: int = 6
    evidence_k: int = 3
    min_conf: float = 0.5
    region_chars: int = 3500
    audit: bool = False              # the optional behavioural AUDITOR
    audit_low: float = 0.35
    audit_high: float = 0.65
    audit_delta: float = 1.0         # rating points the candidate must move
    temperature: float = 0.0


_KIND_W = {"directive": 3, "persona": 3, "obfuscation": 3, "pressure": 2, "spin": 1}


def _call(llm: Backend, route: Routing, task: str, agent: str, system: str, user: str,
          cfg: CascadeConfig, max_tokens: int, counter: Dict[str, int]) -> str:
    msgs = A.messages(system, user, system_role=cfg.system_role)
    counter["calls"] = counter.get("calls", 0) + 1
    try:
        return llm.chat(msgs, model=route.name(task, agent), temperature=cfg.temperature,
                        max_tokens=max_tokens)[0]
    except Exception as e:                       # one failed call must not kill a row
        counter["errors"] = counter.get("errors", 0) + 1
        return f"__ERROR__ {type(e).__name__}: {str(e)[:200]}"


# ======================================================================================
# step 1 — LOCATOR over windows, items assembled deterministically
# ======================================================================================
def prepare(doc: str, cfg: CascadeConfig):
    sigs = T.signals(doc)
    wins = T.windows(doc, sigs, window_chars=cfg.window_chars, overlap=cfg.overlap,
                     tail_chars=cfg.tail_chars, max_windows=cfg.max_windows)
    return sigs, wins


def window_signals(sigs: Sequence[T.Signal], w: T.Window) -> List[T.Signal]:
    return [s for s in sigs if w.start <= s.start < w.end]


def locator_prompt(doc: str, w: T.Window, sigs: Sequence[T.Signal]) -> Tuple[str, str]:
    return A.build_locator(w.text(doc), w.section, window_signals(sigs, w))


def assemble_items(doc: str, wins: Sequence[T.Window], raws: Sequence[str],
                   sigs: Sequence[T.Signal], max_items: int) -> Tuple[List[A.Item], Dict]:
    """
    Locator outputs -> located, de-duplicated, prioritised items.

    Only VERBATIM quotes survive (a program checks this, not an agent). Strong
    obfuscation signals that no candidate covers become items of their own, so a
    perturbation the locator overlooked still reaches the verifier.
    """
    loc = T.Located(doc)
    cands: List[Tuple[int, A.Item]] = []
    stats = {"n_raw_candidates": 0, "n_fabricated": 0, "n_claims": 0}
    for w, raw in zip(wins, raws):
        cs, cl, _conf = A.parse_locator(A.extract_json(raw))
        wt = w.text(doc)
        for c in cs:
            stats["n_raw_candidates"] += 1
            sp = str(c.get("span") or "")
            if T.verbatim(sp, wt) is not True:
                stats["n_fabricated"] += 1
                continue
            hit = loc.find(sp)
            if not hit:
                continue
            kind = str(c.get("kind") or "").lower()
            cands.append((_KIND_W.get(kind, 2), A.Item(
                0, doc[hit[0]:hit[1]], hit[0], hit[1], "candidate", kind,
                str(c.get("requested_behavior") or "")[:200])))
        for c in cl:
            sp = str(c.get("span") or "")
            if T.verbatim(sp, wt) is not True:
                continue
            hit = loc.find(sp)
            if not hit:
                continue
            stats["n_claims"] += 1
            cands.append((1, A.Item(0, doc[hit[0]:hit[1]], hit[0], hit[1], "claim",
                                    "spin", claim=str(c.get("claim") or "")[:200])))
    for s in sigs:
        if s.kind in ("zero_width", "bidi_control", "homoglyph", "reviewer_directive"):
            if not any(a.start <= s.start < a.end for _w, a in cands):
                # widen a single-character signal to its sentence so there is text to judge
                a, b = s.start, s.end
                a = max(0, doc.rfind(".", 0, a) + 1)
                e = doc.find(".", b)
                b = min(len(doc), (e + 1) if e >= 0 else b + 200)
                cands.append((3 if s.kind != "reviewer_directive" else 2, A.Item(
                    0, doc[a:b], a, b, "signal", "obfuscation"
                    if s.kind != "reviewer_directive" else "directive",
                    f"(program signal: {s.kind})")))
    # de-duplicate overlapping items, keeping the higher-priority / longer one
    cands.sort(key=lambda z: (-z[0], -(z[1].end - z[1].start)))
    kept: List[A.Item] = []
    for _w, it in cands:
        if any(T.intersect([(it.start, it.end)], [(k.start, k.end)]) >
               0.5 * min(it.end - it.start, k.end - k.start) for k in kept):
            continue
        kept.append(it)
        if len(kept) >= max_items:
            break
    kept.sort(key=lambda it: it.start)
    for i, it in enumerate(kept, 1):
        it.id = i
    return kept, stats


def retrieve_for(doc: str, items: Sequence[A.Item], k: int) -> Dict[int, List[Dict]]:
    return {it.id: T.retrieve(doc, (it.claim or "") + " " + it.span,
                              exclude=[(it.start, it.end)], k=k) for it in items}


# ======================================================================================
# the full detect pass
# ======================================================================================
def detect(doc: str, llm: Backend, route: Routing, cfg: CascadeConfig,
           reviewer: Optional[Reviewer] = None) -> Dict:
    ctr: Dict[str, int] = {}
    sigs, wins = prepare(doc, cfg)
    loc_raws = []
    for w in wins:
        s, u = locator_prompt(doc, w, sigs)
        loc_raws.append(_call(llm, route, "detect", "LOCATOR", s, u, cfg,
                              A.ROLES["LOCATOR"].max_tokens, ctr))
    items, istats = assemble_items(doc, wins, loc_raws, sigs, cfg.max_items)
    out = {"n_windows": len(wins), "signals": T.signal_summary(sigs),
           "items": [it.as_dict() for it in items], **istats,
           "evidence": {}, "verifier": {}, "leader": {}, "escalated": 0,
           "pred": 0, "score": 0.0, "audit": None}
    if not items:
        out.update({"verdict": "CLEAN", "confidence": 0.9,
                    "reason": "gate: no verbatim candidate and no obfuscation signal",
                    "calls": ctr.get("calls", 0), "errors": ctr.get("errors", 0),
                    "score": 0.02 * min(5, sum(s.weight for s in sigs))})
        return out
    out["escalated"] = 1
    retrieved = retrieve_for(doc, items, cfg.evidence_k)
    s, u = A.build_evidence(items, retrieved)
    ev = A.parse_evidence(A.extract_json(_call(llm, route, "detect", "EVIDENCE", s, u, cfg,
                                               A.ROLES["EVIDENCE"].max_tokens, ctr)))
    s, u = A.build_verifier(items, doc)
    vf = A.parse_verifier(A.extract_json(_call(llm, route, "detect", "VERIFIER", s, u, cfg,
                                               A.ROLES["VERIFIER"].max_tokens, ctr)))
    s, u = A.build_leader(doc, sigs, items, ev, vf, region_chars=cfg.region_chars)
    ld = A.parse_leader(A.extract_json(_call(llm, route, "detect", "LEADER", s, u, cfg,
                                             A.ROLES["LEADER"].max_tokens, ctr)))
    grounded = T.verbatim(ld["span"], doc) is True
    pred = int(ld["verdict"] == "ATTACKED" and ld["confidence"] >= cfg.min_conf
               and grounded)

    # ---- optional behavioural audit, uncertain verdicts only -------------------------
    audit = None
    if (cfg.audit and reviewer is not None and ld["parsed"]
            and cfg.audit_low <= ld["confidence"] <= cfg.audit_high):
        live = [it for it in items if (vf.get(it.id) or {}).get("decision") != "DISMISSED"]
        if live:
            r_full = reviewer.review(doc)
            r_cut = reviewer.review(T.excise(doc, [(it.start, it.end) for it in live]))
            ctr["calls"] = ctr.get("calls", 0) + r_full["calls"] + r_cut["calls"]
            d = None
            if r_full["rating"] is not None and r_cut["rating"] is not None:
                d = r_full["rating"] - r_cut["rating"]
            audit = {"rating_with": r_full["rating"], "rating_without": r_cut["rating"],
                     "delta": d, "flipped": False}
            # A changed score is SUPPORTING evidence, never ground truth: it can only tip
            # an uncertain CLEAN whose candidate the verifier did not dismiss.
            if (not pred and d is not None and d >= cfg.audit_delta):
                top = max(live, key=lambda it: it.end - it.start)
                ld = dict(ld, verdict="ATTACKED", span=top.span,
                          reason=(ld.get("reason", "") + " [audit: removing this text "
                                  f"lowered the rating by {d:.1f}]").strip())
                pred, grounded = 1, True
                audit["flipped"] = True
    conf = ld["confidence"] if ld["parsed"] else 0.5
    # The AUROC score follows the decision rule: an ATTACKED verdict the leader could not
    # quote is half-weight (it cannot be acted on), and an audit flip is at least min_conf.
    if ld["verdict"] == "ATTACKED":
        score = conf if grounded else 0.5 * conf
    else:
        score = 1.0 - conf
    if audit and audit.get("flipped"):
        score = max(score, cfg.min_conf)
    out.update({"evidence": ev, "verifier": vf, "leader": ld, "verdict": ld["verdict"],
                "confidence": conf, "reason": ld.get("reason", ""),
                "span": ld.get("span", ""), "span_grounded": grounded, "pred": pred,
                "score": float(score), "audit": audit,
                "calls": ctr.get("calls", 0), "errors": ctr.get("errors", 0)})
    return out


# ======================================================================================
# localize + repair
# ======================================================================================
def _sentence(doc: str, span: T.Span) -> T.Span:
    a = max(0, max(doc.rfind(". ", 0, span[0]), doc.rfind("\n", 0, span[0])) + 1)
    e = re.search(r"[.!?](\s|$)", doc[span[1]:])
    b = span[1] + (e.end() if e else 0)
    return (a, min(len(doc), b))


def localize(doc: str, det: Dict, llm: Backend, route: Routing,
             cfg: CascadeConfig) -> Dict:
    """Ask the repair lead what to delete. Only called on flagged documents."""
    ctr: Dict[str, int] = {}
    items = [A.Item(**d) for d in det.get("items") or []]
    vf = det.get("verifier") or {}
    vf = {int(k): v for k, v in vf.items()}
    s, u = A.build_localize(doc, items, vf)
    raw = _call(llm, route, "localize", "LEADER", s, u, cfg,
                A.ROLES["LEADER_LOCALIZE"].max_tokens, ctr)
    p = A.parse_localize(A.extract_json(raw))
    spans, n_fab = T.spans_from_quotes(doc, p["delete"])
    fallback = False
    if not spans and det.get("span") and det.get("span_grounded"):
        # the leader quoted the manipulation but the repair lead returned nothing usable:
        # delete the quoted sentence rather than leave a flagged attack in place
        hit = T.Located(doc).find(det["span"])
        if hit:
            spans, fallback = [_sentence(doc, hit)], True
    return {"delete": p["delete"], "spans": spans, "n_fabricated": n_fab,
            "fallback": fallback, "note": p.get("note", ""),
            "calls": ctr.get("calls", 0), "errors": ctr.get("errors", 0)}


def repair(doc: str, spans: Sequence[T.Span]) -> Tuple[str, Dict]:
    """Delete, then undo character-level tricks. Applied only to FLAGGED documents."""
    cut = T.excise(doc, spans)
    fixed, counts = T.normalize_text(cut)
    return fixed, {"chars_removed": T.total(spans), "normalized": counts,
                   "len_before": len(doc), "len_after": len(fixed)}
