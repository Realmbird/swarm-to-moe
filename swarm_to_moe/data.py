"""Datasets, answer extraction and Harmony output parsing."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------- answers

_NUM = r"-?\$?\s?-?\d+(?:,\d{3})*(?:\.\d+)?|-?\$?\s?-?\.\d+"
_NUM_RE = re.compile(_NUM)
_MARKERS = re.compile(r"(answer\s+is|answer\s*:|final\s+answer)", re.IGNORECASE)


def _clean(num: str) -> str:
    s = num.replace(",", "").replace("$", "").replace(" ", "")
    if s.startswith("--"):
        s = s[1:]
    return s


def extract_gsm8k_answer(text: str) -> str | None:
    """Plain-text answer extraction.

    Priority: the first number after the last '####'; else the first number after the last
    'answer is' / 'Answer:' / 'final answer'; else the last number in the text. Commas and '$' stripped,
    sign and decimals kept. None if no number."""
    if text is None:
        return None
    if "####" in text:
        m = _NUM_RE.search(text.rsplit("####", 1)[1])
        if m:
            return _clean(m.group())
    marks = list(_MARKERS.finditer(text))
    if marks:
        m = _NUM_RE.search(text[marks[-1].end():])
        if m:
            return _clean(m.group())
    nums = _NUM_RE.findall(text)
    if nums:
        return _clean(nums[-1])
    return None


def _to_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def gsm8k_correct(pred_text: str, gold: str) -> bool:
    p = _to_float(extract_gsm8k_answer(pred_text))
    g = _to_float(gold)
    if p is None or g is None:
        return False
    return abs(p - g) <= 1e-6 * max(1.0, abs(g))


def gsm8k_gold(answer_field: str) -> str:
    return answer_field.rsplit("####", 1)[1].strip().replace(",", "")


# ---------------------------------------------------------------------------- harmony

FINAL_MARK = "<|channel|>final<|message|>"
_STOPS = ("<|return|>", "<|end|>", "<|call|>")
_SPECIAL = re.compile(r"<\|[a-z_]+\|>")


def _cut_at_stop(s: str) -> str:
    cut = len(s)
    for st in _STOPS:
        i = s.find(st)
        if i >= 0:
            cut = min(cut, i)
    return s[:cut]


def split_harmony(text: str) -> tuple[str, str | None]:
    """Split gpt-oss Harmony output into (analysis, final).

    final = text after the LAST '<|channel|>final<|message|>' up to the next <|return|>/<|end|>/<|call|>
    (or end of text); None if no final channel appeared (e.g. truncated during analysis).
    analysis = the analysis-channel message bodies (joined by newlines); if there are no channel markers
    at all, the whole text before any final channel with special tokens removed."""
    if FINAL_MARK in text:
        head, tail = text.rsplit(FINAL_MARK, 1)
        final = _cut_at_stop(tail).strip()
    else:
        head, final = text, None
    parts = head.split("<|channel|>analysis<|message|>")
    if len(parts) > 1:
        analysis = "\n".join(_cut_at_stop(p).strip() for p in parts[1:])
    else:
        analysis = _SPECIAL.sub(" ", head).strip()
    return analysis, final


def score_gsm8k_harmony(text: str, gold: str) -> dict:
    """Score a Harmony completion: extract from the final channel if present, else from the whole text
    (special tokens removed) and flag ``truncated``."""
    analysis, final = split_harmony(text)
    truncated = final is None
    src = final if final is not None else _SPECIAL.sub(" ", text)
    pred = extract_gsm8k_answer(src)
    return {"pred": pred, "correct": gsm8k_correct(src, gold), "truncated": truncated}


# ---------------------------------------------------------------------------- subsets / loading


def seeded_subset(items: Sequence, n: int, seed: int) -> list:
    items = list(items)
    if n > len(items):
        raise ValueError(f"n={n} > {len(items)} items")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=n, replace=False)
    return [items[i] for i in idx]


GSM8K_SUFFIX = "\n\nSolve the problem. End your reply with the final numeric answer in the form '#### <number>'."


def load_gsm8k(split: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [{"question": r["question"], "answer": gsm8k_gold(r["answer"]), "answer_full": r["answer"]} for r in ds]


def format_mmlu(item: dict) -> str:
    letters = "ABCD"
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(item["choices"]))
    return f"{item['question']}\n\n{opts}\n\nAnswer with the letter of the correct option."


def load_mmlu_val(n: int | None = None, seed: int = 0) -> list[dict]:
    """cais/mmlu 'all' validation split; seeded subset of n items (all if n is None)."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="validation")
    items = [{"question": r["question"], "choices": list(r["choices"]), "answer": int(r["answer"]),
              "subject": r["subject"], "text": None} for r in ds]
    for it in items:
        it["text"] = format_mmlu(it)
    if n is None:
        return items
    return seeded_subset(items, min(n, len(items)), seed)


def format_choices(question: str, labels: Sequence[str], texts: Sequence[str]) -> str:
    opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    return f"{question}\n\n{opts}\n\nAnswer with the letter of the correct option."


def load_arc(split: str = "train", n: int | None = None, seed: int = 0) -> list[dict]:
    """allenai/ai2_arc ARC-Challenge; 'text' = question + lettered choices."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    items = [{"question": r["question"], "labels": list(r["choices"]["label"]), "choices": list(r["choices"]["text"]),
              "answer": r["answerKey"],
              "text": format_choices(r["question"], r["choices"]["label"], r["choices"]["text"])} for r in ds]
    return items if n is None else seeded_subset(items, min(n, len(items)), seed)


def load_ultrachat(n: int, seed: int = 0, split: str = "train_sft") -> list[str]:
    """First user turn of seeded HuggingFaceH4/ultrachat_200k conversations."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    out = []
    for i in idx.tolist():
        msgs = ds[int(i)]["messages"]
        first = next((m["content"] for m in msgs if m["role"] == "user"), ds[int(i)].get("prompt"))
        out.append(first)
    return out


PROFILE_SEED = 0


def profiling_texts(n_per_source: int = 500, seed: int = PROFILE_SEED) -> dict[str, list[str]]:
    """User texts used to profile experts for the partition (paper draft): n each from GSM8K train,
    ARC-Challenge train, MMLU validation and UltraChat train_sft, all seeded. E0-b excludes these."""
    return {
        "gsm8k_train": [x["question"] for x in seeded_subset(load_gsm8k("train"), n_per_source, seed)],
        "arc_challenge_train": [x["text"] for x in load_arc("train", n_per_source, seed)],
        "mmlu_val": [x["text"] for x in load_mmlu_val(n_per_source, seed)],
        "ultrachat_train_sft": load_ultrachat(n_per_source, seed),
    }
