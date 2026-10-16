"""Expert profiling for the partition recipe: usage counts, peak individual output norms, router sigma."""
from __future__ import annotations

import types
from typing import Sequence

import numpy as np
import torch

from .hook import FreeRouting, MoERoutingHook


class PeakTracker:
    """Wraps every ``layer.mlp.experts.forward``. The wrapped forward returns the ORIGINAL implementation's
    output (so the model output is unchanged bit-for-bit), and separately recomputes each hit expert's
    individual output (before the routing weight) with the HF eager formula to track the max L2 norm over
    valid tokens."""

    def __init__(self, model):
        inner = getattr(model, "model", model)
        self.experts = [layer.mlp.experts for layer in inner.layers]
        E = self.experts[0].num_experts
        self.peak = torch.zeros(len(self.experts), E, dtype=torch.float32)
        self.valid: torch.Tensor | None = None  # flat [B*T] bool of non-pad tokens, or None = all

    def install(self):
        for li, ex in enumerate(self.experts):
            if "forward" in ex.__dict__:
                raise RuntimeError("experts.forward already patched")
            orig = type(ex).forward
            ex.forward = types.MethodType(self._make(li, orig), ex)
        return self

    def uninstall(self):
        for ex in self.experts:
            ex.__dict__.pop("forward", None)

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()
        return False

    def _make(self, li, orig):
        tracker = self

        def forward(ex, hidden_states, router_indices=None, routing_weights=None):
            out = orig(ex, hidden_states, router_indices, routing_weights)
            with torch.no_grad():
                valid = tracker.valid
                if valid is not None:
                    valid = valid.to(hidden_states.device)
                    if valid.numel() != hidden_states.shape[0]:
                        valid = None
                for e in torch.unique(router_indices).tolist():
                    if e < 0 or e >= ex.num_experts:
                        continue
                    tok = (router_indices == e).any(-1)
                    if valid is not None:
                        tok = tok & valid
                    if not bool(tok.any()):
                        continue
                    x = hidden_states[tok]
                    gate_up = x @ ex.gate_up_proj[e] + ex.gate_up_proj_bias[e]
                    y = ex._apply_gate(gate_up) @ ex.down_proj[e] + ex.down_proj_bias[e]
                    m = y.float().norm(dim=-1).max().item()
                    if m > tracker.peak[li, e]:
                        tracker.peak[li, e] = m
            return out

        return forward


@torch.no_grad()
def profile_experts(model, tokenizer, texts: Sequence[str], batch_size: int, return_sigma: bool = False,
                    progress: bool = True):
    """Prefill-only profiling over ``texts`` (already chat-formatted strings), left-padded batches.

    Returns (usage [L,E] float64 = top-k selection counts over non-pad tokens,
             peak  [L,E] float64 = max L2 norm of each expert's individual output over non-pad tokens)
    and, with ``return_sigma``, router_sigma [L] = std (float64 accumulation of float32 logits) of all router
    logits of that layer over all non-pad profiled tokens and all experts."""
    from tqdm.auto import tqdm

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    inner = getattr(model, "model", model)
    L = len(inner.layers)
    E = inner.layers[0].mlp.experts.num_experts
    usage = np.zeros((L, E), dtype=np.float64)
    s1 = np.zeros(L)
    s2 = np.zeros(L)
    cnt = np.zeros(L)
    tracker = PeakTracker(model)
    with MoERoutingHook(model, rule=FreeRouting(), record_logits=return_sigma) as hook, tracker:
        for s in tqdm(range(0, len(texts), batch_size), desc="profile", disable=not progress):
            enc = tokenizer(list(texts[s:s + batch_size]), return_tensors="pt", padding=True,
                            add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            mask = enc["attention_mask"].bool()
            tracker.valid = mask.reshape(-1)
            hook.reset_records()
            model(**enc)
            idx = hook.records.stack()  # [L,B,T,k]
            m = mask.cpu()
            for l in range(L):
                usage[l] += np.bincount(idx[l][m].reshape(-1).numpy(), minlength=E)[:E]
            if return_sigma:
                lg = hook.records.stack_logits()  # [L,B,T,E]
                sel = lg[:, m].double()  # [L,N,E]
                s1 += sel.sum((1, 2)).numpy()
                s2 += (sel ** 2).sum((1, 2)).numpy()
                cnt += sel.shape[1] * sel.shape[2]
                del lg, sel
            hook.reset_records()
    tracker.valid = None
    peak = tracker.peak.double().numpy()
    if return_sigma:
        mean = s1 / cnt
        sigma = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 0.0))
        return usage, peak, sigma
    return usage, peak
