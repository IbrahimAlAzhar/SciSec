"""
pg_orig.py — load the pieces of PaperGuard's released defense FROM ITS OWN SOURCE FILES.

WHY THIS MODULE EXISTS
=======================================================================================
The baseline has to be citable as "PaperGuard's chunk-based embedding search", so the
parts that define the method are not re-typed here. They are executed from the vendored,
unmodified copies in paperguard_orig/ (md5 sums in README.md):

    defense/emb_search.py   EmbSearchDefense._split_text            -> split_text()
                            EmbSearchDefense._judge_query_with_topk  -> verifier_prompt()
                               (only the two statements that build `seg_block` and
                                `prompt`; the Azure client around them is replaced by
                                the gpt-4o-mini call in pg_llm.py)
    defense/llm_judge.py    _parse_json_response, _parse_assessment_response
                                                                    -> parse_assessment()
    defense/prompts.py      split_prompts_into_groups(seed, a1_size) -> known_prompts()
                               (the A1 "known attack" registry of main_defense.py)
    prompts/prompt_template.py  PromptTemplate                      -> PromptTemplate
                               (the review-generation prompt, used for re-rating)

Importing defense/__init__.py is avoided on purpose: it imports openai, azure.ai.inference
and anthropic at module level, none of which the baseline needs. Functions are therefore
pulled out with `ast` and executed in a namespace that holds only what they reference.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import random
import re
import sys
import types
from typing import Any, Callable, Dict, List, Tuple

ORIG = os.environ.get("PG_ORIG",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "paperguard_orig"))

EMB_SEARCH = os.path.join(ORIG, "defense", "emb_search.py")
LLM_JUDGE = os.path.join(ORIG, "defense", "llm_judge.py")
PROMPTS = os.path.join(ORIG, "defense", "prompts.py")
OBFUSCATE = os.path.join(ORIG, "tools", "obfuscate_generation.py")
PROMPT_TEMPLATE = os.path.join(ORIG, "prompts", "prompt_template.py")


def _read(path: str) -> str:
    if not os.path.exists(path):
        raise SystemExit(f"[FATAL] PaperGuard source file missing: {path}\n"
                         f"        paperguard_orig/ must sit next to pg_orig.py (or set "
                         f"PG_ORIG).")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _find_def(tree: ast.AST, name: str, cls: str = "") -> ast.FunctionDef:
    scope = tree
    if cls:
        scope = next((n for n in ast.walk(tree)
                      if isinstance(n, ast.ClassDef) and n.name == cls), None)
        if scope is None:
            raise SystemExit(f"[FATAL] class {cls} not found in PaperGuard source")
    for n in ast.walk(scope):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise SystemExit(f"[FATAL] {cls + '.' if cls else ''}{name} not found in PaperGuard "
                     f"source — the vendored file is not the one this port was written "
                     f"against.")


def _exec_def(node: ast.FunctionDef, ns: Dict[str, Any], filename: str) -> Callable:
    mod = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, filename, "exec"), ns)
    return ns[node.name]


# ======================================================================================
# 1. segmentation — EmbSearchDefense._split_text, executed verbatim
# ======================================================================================
def _load_split_text() -> Callable[..., List[str]]:
    tree = ast.parse(_read(EMB_SEARCH))
    node = _find_def(tree, "_split_text", cls="EmbSearchDefense")
    fn = _exec_def(node, {"re": re, "List": List}, EMB_SEARCH)

    def split_text(text: str, max_len: int = 800) -> List[str]:
        return fn(None, text, max_len=max_len)          # `self` is unused in the original
    return split_text


# ======================================================================================
# 2. verifier prompt — the `seg_block` and `prompt` statements of
#    EmbSearchDefense._judge_query_with_topk, executed verbatim
# ======================================================================================
def _load_verifier_prompt() -> Callable[[str, List[str]], str]:
    tree = ast.parse(_read(EMB_SEARCH))
    node = _find_def(tree, "_judge_query_with_topk", cls="EmbSearchDefense")
    keep = []
    for st in node.body:
        if isinstance(st, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ("seg_block", "prompt")
                for t in st.targets):
            keep.append(st)
    names = sorted({t.id for st in keep for t in st.targets if isinstance(t, ast.Name)})
    if names != ["prompt", "seg_block"]:
        raise SystemExit(f"[FATAL] could not locate the verifier prompt statements in "
                         f"{EMB_SEARCH} (found {names})")
    mod = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(mod)
    code = compile(mod, EMB_SEARCH, "exec")

    def verifier_prompt(qry: str, seg_texts: List[str]) -> str:
        ns: Dict[str, Any] = {"qry": qry, "seg_texts": list(seg_texts)}
        exec(code, ns)
        return ns["prompt"]
    return verifier_prompt


# ======================================================================================
# 3. verifier output parser — llm_judge._parse_json_response/_parse_assessment_response
# ======================================================================================
def _load_parser() -> Callable[[str], Dict[str, Any]]:
    tree = ast.parse(_read(LLM_JUDGE))
    ns: Dict[str, Any] = {"json": json, "Dict": Dict, "Any": Any}
    _exec_def(_find_def(tree, "_parse_json_response"), ns, LLM_JUDGE)
    return _exec_def(_find_def(tree, "_parse_assessment_response"), ns, LLM_JUDGE)


# ======================================================================================
# 4. the known-attack registry of main_defense.py (A1 = split_prompts_into_groups(...)[0])
# ======================================================================================
def _load_module(name: str, path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def known_prompts(seed: int = 42, a1_size: int = 10
                  ) -> Tuple[List[Tuple[str, str, bool]], List, List]:
    """
    (A1, A2, A3) exactly as main_defense.py builds them.

    prompts.py does `from tools.obfuscate_generation import ...`. A package called `tools`
    may already exist on this cluster's PYTHONPATH (the attack code puts a dozen sibling
    directories there), so the vendored module is registered under that name for the
    duration of the import and whatever was there before is put back.
    """
    saved = {k: sys.modules.get(k) for k in ("tools", "tools.obfuscate_generation")}
    saved_path = list(sys.path)       # prompts.py appends paperguard_orig/ to sys.path
    try:
        pkg = types.ModuleType("tools")
        pkg.__path__ = [os.path.dirname(OBFUSCATE)]
        sys.modules["tools"] = pkg
        sys.modules["tools.obfuscate_generation"] = _load_module(
            "tools.obfuscate_generation", OBFUSCATE)
        pr = _load_module("paperguard_defense_prompts", PROMPTS)
        # main_defense.py seeds the global RNG before splitting; split_prompts_into_groups
        # reseeds it itself, so the result depends on `seed` alone. The caller's global
        # random state is restored afterwards so nothing else in this process shifts.
        state = random.getstate()
        try:
            random.seed(seed)
            a1, a2, a3 = pr.split_prompts_into_groups(seed=seed, a1_size=a1_size)
        finally:
            random.setstate(state)
        return a1, a2, a3
    finally:
        sys.path[:] = saved_path
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def prompt_template_class():
    """PaperGuard's PromptTemplate (review generation + parse_review_output)."""
    return _load_module("paperguard_prompt_template", PROMPT_TEMPLATE).PromptTemplate


split_text = _load_split_text()
verifier_prompt = _load_verifier_prompt()
parse_assessment = _load_parser()


def parse_assessment_lenient(raw: str) -> Tuple[int, str, str]:
    """
    (is_unsafe, explanation, status).

    status "ok"      — PaperGuard's own parser read it.
    status "lenient" — its json.loads failed (almost always an explanation cut off by the
                       200-token cap), but the "is_unsafe" field itself is intact and is
                       read with a regex. The label is not guessed: it is the model's.
    status "fail"    — no is_unsafe value anywhere. Scored SAFE and counted. (The original
                       main_defense.py would raise here and drop the paper from the
                       evaluation, which would change N and break the clean/attack pairing
                       every other row of the table uses.)
    """
    try:
        d = parse_assessment(raw or "")
        return int(d["is_unsafe"]), str(d.get("explanation") or ""), "ok"
    except Exception:
        pass
    m = re.search(r'"?is_unsafe"?\s*[:=]\s*"?\s*([01]|true|false)', raw or "", re.I)
    if m:
        v = m.group(1).lower()
        ex = re.search(r'"explanation"\s*:\s*"(.*)', raw or "", re.S)
        return (1 if v in ("1", "true") else 0,
                (ex.group(1)[:500] if ex else ""), "lenient")
    return 0, "", "fail"
