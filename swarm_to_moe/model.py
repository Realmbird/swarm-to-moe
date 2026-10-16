"""Loading gpt-oss-20b (dequantised bf16, Python routing path) and batched greedy generation with routing."""
from __future__ import annotations

import hashlib
import time
from typing import Sequence

import torch

DEFAULT_MODEL = "openai/gpt-oss-20b"


def load_gpt_oss(model_id: str = DEFAULT_MODEL, revision: str | None = None, device: str = "cuda",
                 dtype=torch.bfloat16):
    """Returns (model, tokenizer, provenance).

    MXFP4 weights are dequantised (``Mxfp4Config(dequantize=True)``) so the experts are plain bf16
    ``GptOssExperts`` and routing goes through the Python ``GptOssMLP.forward`` the hook replaces
    (CLAUDE.md section 1). Attention is eager: GptOss does not support sdpa in transformers 5.20."""
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        device_map=device,
        quantization_config=Mxfp4Config(dequantize=True),
        attn_implementation="eager",
    )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    commit = None
    try:
        from huggingface_hub import model_info

        commit = model_info(model_id, revision=revision).sha
    except Exception:
        commit = getattr(model.config, "_commit_hash", None)
    template = tok.chat_template if isinstance(tok.chat_template, str) else str(tok.chat_template)
    provenance = {
        "model_id": model_id,
        "revision_requested": revision,
        "commit": commit,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "chat_template_sha256": hashlib.sha256((template or "").encode()).hexdigest(),
        "dtype": str(dtype),
        "attn_implementation": "eager",
        "experts_implementation": getattr(model.config, "_experts_implementation", None),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return model, tok, provenance


def apply_chat(tokenizer, messages: list[dict], reasoning_effort: str | None = "low") -> str:
    if reasoning_effort is not None:
        try:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                 reasoning_effort=reasoning_effort)
        except Exception:
            pass
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def _stop_ids(model, tokenizer) -> set[int]:
    ids = set()
    gc_eos = getattr(model.generation_config, "eos_token_id", None)
    for x in (gc_eos, tokenizer.eos_token_id):
        if x is None:
            continue
        ids.update(x if isinstance(x, (list, tuple)) else [x])
    return ids


def router_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy (nats) of softmax over all experts; logits [..., E] -> [...] float32."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    return -(lp.exp() * lp).sum(-1)


@torch.no_grad()
def generate_batched(model, tokenizer, prompts: Sequence[str], max_new_tokens: int, batch_size: int,
                     hook=None, row_agents: Sequence | None = None, left_pad: bool = True,
                     entropy: bool = False, progress: bool = True):
    """Greedy batched generation.

    Returns (texts, routing) where texts are the decoded generated tokens with special tokens KEPT
    (Harmony format; use ``data.split_harmony``), and routing is None without a hook, else a list (one per
    batch) of dicts with CPU tensors:
      indices     int16 [L, B, T, k]   recorded routing, T = prompt_len + n_generated - 1
      prompt_mask bool  [B, T]         real (non-pad) prompt tokens
      gen_mask    bool  [B, T]         positions whose input is a generated token before the row stopped
                                       (the stop token itself is excluded)
      entropy     float32 [L, B, T]    router entropy per token (only if ``entropy=True``; needs the hook
                                       to have record_logits=True; full logits are dropped after this)
      rows        list[int]            indices into ``prompts``
    Also returns per-run stats in ``generate_batched.last_stats`` (tokens, seconds, tok/s)."""
    from tqdm.auto import tqdm

    if left_pad:
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    stops = _stop_ids(model, tokenizer)
    device = next(model.parameters()).device
    texts: list[str] = [None] * len(prompts)
    routing = [] if hook is not None else None
    n_tok, t0 = 0, time.time()
    starts = range(0, len(prompts), batch_size)
    bar = tqdm(starts, desc="generate", disable=not progress)
    for s in bar:
        rows = list(range(s, min(s + batch_size, len(prompts))))
        enc = tokenizer([prompts[i] for i in rows], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        P = enc["input_ids"].shape[1]
        if hook is not None:
            if row_agents is not None:
                hook.set_row_agents([row_agents[i] for i in rows])
            hook.reset_records()
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id,
                             temperature=None, top_p=None, top_k=None)
        gen = out[:, P:].cpu()
        B, N = gen.shape
        # valid generated tokens: before (excluding) the first stop token
        is_stop = torch.zeros_like(gen, dtype=torch.bool)
        for sid in stops:
            is_stop |= gen == sid
        before_stop = torch.cumsum(is_stop.int(), dim=1) == 0
        n_tok += int((before_stop | (is_stop & (torch.cumsum(is_stop.int(), 1) == 1))).sum())
        for j, i in enumerate(rows):
            texts[i] = _decode_until_stop(tokenizer, gen[j], stops)  # keeps the stop token, drops padding
        if hook is not None:
            idx = hook.records.stack()  # [L,B,P+N-1,k]
            T = idx.shape[2]
            attn = enc["attention_mask"].cpu().bool()
            prompt_mask = torch.zeros(B, T, dtype=torch.bool)
            prompt_mask[:, :P] = attn
            gen_mask = torch.zeros(B, T, dtype=torch.bool)
            gen_mask[:, P:] = before_stop[:, : T - P]
            rec = {"indices": idx.to(torch.int16), "prompt_mask": prompt_mask, "gen_mask": gen_mask, "rows": rows}
            if entropy:
                lg = hook.records.stack_logits()
                rec["entropy"] = router_entropy(lg)
                del lg
            hook.reset_records()
            routing.append(rec)
        dt = time.time() - t0
        bar.set_postfix(tok_s=f"{n_tok / max(dt, 1e-9):.0f}")
    dt = time.time() - t0
    generate_batched.last_stats = {"generated_tokens": n_tok, "seconds": dt, "tok_per_s": n_tok / max(dt, 1e-9)}
    if progress:
        print(f"[generate_batched] {n_tok} tokens in {dt:.1f}s = {n_tok / max(dt, 1e-9):.1f} tok/s")
    return texts, routing


def _decode_until_stop(tokenizer, ids: torch.Tensor, stops: set[int]) -> str:
    keep = []
    for t in ids.tolist():
        keep.append(t)
        if t in stops:
            break
    return tokenizer.decode(keep, skip_special_tokens=False)


@torch.no_grad()
def prefill_routing(model, tokenizer, prompts: Sequence[str], hook, row_agents=None):
    """One prefill forward on a left-padded batch; returns (indices int16 [L,B,P,k], attention_mask [B,P])."""
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True, add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    if row_agents is not None:
        hook.set_row_agents(row_agents)
    hook.reset_records()
    model(**enc)
    idx = hook.records.stack().to(torch.int16)
    hook.reset_records()
    return idx, enc["attention_mask"].cpu().bool()
