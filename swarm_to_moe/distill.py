"""Self-distillation data pipeline and LoRA training step (E2 step 1).

Student input = the user text with NO role system prompt (chat template, generation prompt, reasoning low).
Student target = the teacher's (role-prompted base model) Harmony final-channel answer:
    "<|channel|>final<|message|>" + target_text + "<|return|>"
Only target tokens are supervised (prompt labels = -100).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

IGNORE = -100
FINAL_PREFIX = "<|channel|>final<|message|>"
FINAL_STOP = "<|return|>"


@dataclass
class DistillExample:
    role: str
    input_text: str
    target_text: str


def build_examples(inputs: Sequence[str], teacher_outputs: Mapping[str, Sequence[str | None]]) -> list[DistillExample]:
    """One example per (role, input) with a non-None teacher target, role-major order."""
    out = []
    for role, targets in teacher_outputs.items():
        if len(targets) != len(inputs):
            raise ValueError(f"role {role!r}: {len(targets)} targets for {len(inputs)} inputs")
        for x, t in zip(inputs, targets):
            if t is None:
                continue
            out.append(DistillExample(role=role, input_text=x, target_text=t))
    return out


def default_prompt(tokenizer, input_text: str) -> str:
    """Chat-formatted prompt with no system/role prompt, ending in the assistant generation header."""
    from .prompts import build_messages

    msgs = build_messages(input_text, None)
    try:
        from .model import apply_chat
    except ImportError:  # pragma: no cover - model.py always defines it today
        apply_chat = None
    if apply_chat is not None:
        return apply_chat(tokenizer, msgs)
    try:
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, reasoning_effort="low")
    except Exception:
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def _single_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"{text!r} does not tokenize to a single special token: {ids}")
    return ids[0]


def encode_example(tokenizer, ex: DistillExample, max_len: int,
                   prompt_builder: Callable[[object, str], str] | None = None) -> dict:
    """Returns {"input_ids": LongTensor [T], "labels": LongTensor [T], "agent": role}.

    The prompt is never truncated; the target is truncated (keeping its final ``<|return|>`` token) so that
    T <= max_len. ValueError if the prompt leaves no room for at least one target content token."""
    build = prompt_builder or default_prompt
    prompt = build(tokenizer, ex.input_text)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(p_ids) >= max_len - 2:
        raise ValueError(f"prompt has {len(p_ids)} tokens; max_len={max_len} leaves no room for a target")
    stop_id = _single_id(tokenizer, FINAL_STOP)
    t_ids = tokenizer(FINAL_PREFIX + ex.target_text, add_special_tokens=False)["input_ids"]
    room = max_len - len(p_ids) - 1  # minus the stop token
    t_ids = list(t_ids[:room]) + [stop_id]
    ids = torch.tensor(list(p_ids) + t_ids, dtype=torch.long)
    labels = ids.clone()
    labels[: len(p_ids)] = IGNORE
    return {"input_ids": ids, "labels": labels, "agent": ex.role}


def collate(encoded: Sequence[dict], pad_id: int) -> dict:
    """Right-pad to the longest example; attention_mask 0 and labels -100 on padding."""
    if not encoded:
        raise ValueError("empty batch")
    T = max(len(e["input_ids"]) for e in encoded)
    B = len(encoded)
    ids = torch.full((B, T), int(pad_id), dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    lab = torch.full((B, T), IGNORE, dtype=torch.long)
    for i, e in enumerate(encoded):
        n = len(e["input_ids"])
        ids[i, :n] = e["input_ids"]
        att[i, :n] = 1
        lab[i, :n] = e["labels"]
    return {"input_ids": ids, "attention_mask": att, "labels": lab, "agents": [e["agent"] for e in encoded]}


def shifted_ce(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Mean next-token cross entropy over valid (!= -100) shifted labels, in float32. Returns (loss, n_valid)."""
    lb = labels[:, 1:].to(logits.device)
    valid = lb != IGNORE
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("batch has no supervised (label != -100) tokens")
    # select the supervised positions before upcasting: full-vocab float32 logits are large on the real model
    lg = logits[:, :-1, :][valid].float()
    loss = F.cross_entropy(lg, lb[valid], reduction="sum") / n_valid
    return loss, n_valid


def lora_forward_loss(model, hook, batch: Mapping) -> tuple[torch.Tensor, int]:
    """Forward with the routing hook told the row agents; returns (loss tensor with grad, n_valid tokens).
    Checks for supervised tokens before running the model."""
    device = next(model.parameters()).device
    labels = batch["labels"]
    if int((labels[:, 1:] != IGNORE).sum()) == 0:
        raise ValueError("batch has no supervised (label != -100) tokens")
    hook.set_row_agents(list(batch["agents"]))
    ids = batch["input_ids"].to(device)
    att = batch["attention_mask"].to(device)
    with torch.enable_grad():
        out = model(input_ids=ids, attention_mask=att, use_cache=False)
        return shifted_ce(out.logits, labels.to(device))


def lora_train_step(model, hook, handle, batch: Mapping, optimizer, grad_clip: float | None = 1.0) -> float:
    """One optimizer step on ``batch`` (no gradient accumulation). Returns the float loss."""
    with torch.enable_grad():
        loss, _ = lora_forward_loss(model, hook, batch)
        loss.backward()
    params = list(handle.parameters())
    if grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())
