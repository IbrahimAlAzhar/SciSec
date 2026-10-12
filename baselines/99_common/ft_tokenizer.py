"""
ft_tokenizer.py — load a tokenizer that can actually apply a chat template.

WHY THIS FILE EXISTS
=======================================================================================
Your Mistral-Small-3.1-24B checkpoint has NO `chat_template` on the tokenizer:

    ValueError: Cannot use chat template functions because tokenizer.chat_template is
    not set and no template argument was passed!

vLLM served the same directory happily, which is why the failure only appeared at the
training step, four stages in. vLLM carries its own fallbacks; `transformers` does not.

Two fixes, in order:
  1. Newer checkpoints ship the template as a SEPARATE FILE, `chat_template.jinja`, which
     some transformers versions do not fold into the tokenizer automatically. If that file
     is there, it is loaded — the model's own template, exactly.
  2. Otherwise a known-correct template for that model family is installed.

Falling back matters more than it looks: the template decides where the prompt ends and
the completion begins, and TRL masks the prompt using exactly that boundary. A wrong
template does not crash — it trains on the wrong tokens.
"""

from __future__ import annotations

import json
import os
from typing import Optional

# Mistral v3 / tekken instruct format. The assistant turn ends with EOS, which is what
# makes the completion boundary well defined for TRL's prompt masking.
MISTRAL_TEMPLATE = """{%- if messages[0]['role'] == 'system' -%}
    {%- set system_message = messages[0]['content'] -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{{- bos_token -}}
{%- for message in loop_messages -%}
    {%- if message['role'] == 'user' -%}
        {%- if loop.first and system_message is defined -%}
            {{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' -}}
        {%- else -%}
            {{- ' [INST] ' + message['content'] + ' [/INST]' -}}
        {%- endif -%}
    {%- elif message['role'] == 'assistant' -%}
        {{- ' ' + message['content'] + eos_token -}}
    {%- else -%}
        {{- raise_exception('Only user and assistant roles are supported after the system message.') -}}
    {%- endif -%}
{%- endfor -%}"""

# ChatML, for any other model that turns up without a template.
CHATML_TEMPLATE = """{%- for message in messages -%}
{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"""


def _family(model_path: str, model_key: str = "") -> str:
    s = (model_key + " " + model_path).lower()
    for fam in ("mistral", "gemma", "qwen", "llama"):
        if fam in s:
            return fam
    return ""


def fallback_template(model_path: str, model_key: str = "") -> str:
    return MISTRAL_TEMPLATE if _family(model_path, model_key) == "mistral" \
        else CHATML_TEMPLATE


def load_tokenizer(model_path: str, model_key: str = "", *, verbose: bool = True):
    """
    AutoTokenizer, with a chat template guaranteed present and a pad token set.
    """
    from transformers import AutoTokenizer

    kw = dict(trust_remote_code=True)
    # Mistral's tekken tokenizer needs this flag in recent transformers, and without it
    # the library warns that tokenisation will be INCORRECT — which would silently shift
    # every label by a token or two.
    if _family(model_path, model_key) == "mistral":
        try:
            tok = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True, **kw)
            if verbose:
                print("[tok] loaded with fix_mistral_regex=True")
        except TypeError:
            tok = AutoTokenizer.from_pretrained(model_path, **kw)
            if verbose:
                print("[tok] this transformers version has no fix_mistral_regex flag; "
                      "loaded without it")
    else:
        tok = AutoTokenizer.from_pretrained(model_path, **kw)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        if verbose:
            print(f"[tok] pad_token was unset -> using eos {tok.eos_token!r}")

    if getattr(tok, "chat_template", None):
        if verbose:
            print("[tok] chat template: from the tokenizer")
        return tok

    # 1. the separate file some checkpoints ship
    for name in ("chat_template.jinja", "chat_template.json"):
        p = os.path.join(model_path, name)
        if not os.path.exists(p):
            continue
        try:
            raw = open(p).read()
            tmpl = (json.loads(raw).get("chat_template") if name.endswith(".json")
                    else raw)
            if tmpl:
                tok.chat_template = tmpl
                if verbose:
                    print(f"[tok] chat template: loaded from {name} — the model's own "
                          f"template, which this transformers version did not pick up "
                          f"automatically")
                return tok
        except Exception as e:
            if verbose:
                print(f"[tok] could not read {name}: {type(e).__name__}: {e}")

    # 2. a known-correct template for the family
    tok.chat_template = fallback_template(model_path, model_key)
    if verbose:
        fam = _family(model_path, model_key) or "generic"
        print(f"[tok] chat template: NONE on the tokenizer and no chat_template.jinja in "
              f"the model directory — installing the built-in {fam} template.")
        print(f"[tok] This is a real fallback, not a formality: the template decides where "
              f"the prompt ends and the completion begins, and TRL masks the prompt on "
              f"exactly that boundary. If generations later look mis-segmented, this line "
              f"is why.")
    return tok


def selftest(model_path: str = "", model_key: str = "") -> None:
    """Render the two templates without transformers, so this can be checked offline."""
    try:
        from jinja2 import Environment
    except Exception:
        print("[tok] jinja2 not available; skipping the offline render check")
        return
    msgs_sys = [{"role": "system", "content": "SYS"},
                {"role": "user", "content": "USER"},
                {"role": "assistant", "content": "ASSISTANT"}]
    msgs_nosys = msgs_sys[1:]
    env = Environment()
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    for name, tmpl in (("mistral", MISTRAL_TEMPLATE), ("chatml", CHATML_TEMPLATE)):
        for label, msgs in (("with system", msgs_sys), ("no system", msgs_nosys)):
            out = env.from_string(tmpl).render(messages=msgs, bos_token="<s>",
                                               eos_token="</s>",
                                               add_generation_prompt=False)
            print(f"[{name}/{label}] {out!r}")


if __name__ == "__main__":
    selftest()
