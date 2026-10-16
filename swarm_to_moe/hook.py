"""Per-row routing rules for gpt-oss MoE layers (HF transformers), plus routing records.

The hook replaces each ``layer.mlp.forward`` with an instance-level bound closure. At the MLP level the
hidden states are still 3-D (B, T, H), so batch rows (= agents) are known. Rows whose agent is ``None``
(and every row under ``FreeRouting``) use the router's own outputs unchanged, so the free path is
bit-identical to the unhooked model.

Rules
-----
FreeRouting()                       plain top-k.
HardSplit(partition, 3, 1)          top-n_shared_k among the shared pool + top-n_private_k among own private.
SoftNudge(partition, delta)         biased = logits + delta * sigma * (own_private - other_roles_private),
                                    sigma = per-token population std (correction=0) of that token's router
                                    logits over the E experts, in float32, detached. Selection by top-k of
                                    ``biased``; mixing weights = softmax of UNBIASED logits at the selection.
                                    delta == 0 takes exactly the FreeRouting code path.
ForceExpert(layer, expert)          at ``layer`` only, the lowest-scoring selected slot is replaced by
                                    ``expert`` if it is not already selected; other layers free.
All custom weights are ``softmax(gather(logits, idx), dim=-1, dtype=logits.dtype)``, i.e. the router's
own dtype rule.
"""
from __future__ import annotations

import types
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .partition import Partition

# ---------------------------------------------------------------------------- rules


class FreeRouting:
    kind = "free"
    needs_agents = False

    def to_dict(self):
        return {"kind": self.kind}


class HardSplit:
    kind = "hard"
    needs_agents = True

    def __init__(self, partition: Partition, n_shared_k: int = 3, n_private_k: int = 1):
        self.partition = partition
        self.n_shared_k = int(n_shared_k)
        self.n_private_k = int(n_private_k)

    def to_dict(self):
        return {"kind": self.kind, "n_shared_k": self.n_shared_k, "n_private_k": self.n_private_k,
                "partition_hash": self.partition.hash()}


class SoftNudge:
    kind = "soft"
    needs_agents = True

    def __init__(self, partition: Partition, delta: float, sigma="per_token"):
        """sigma: "per_token" (std of each token's logits over experts, population std, float32, detached)
        or a tensor/array of shape [L] with a fixed per-layer sigma (paper draft: std of router scores in
        that layer, measured on the profiling set)."""
        self.partition = partition
        self.delta = float(delta)
        if isinstance(sigma, str):
            if sigma != "per_token":
                raise ValueError("sigma must be 'per_token' or a per-layer tensor of shape [L]")
            self.sigma = sigma
        else:
            t = torch.as_tensor(np.asarray(sigma, dtype=np.float32)).float().reshape(-1)
            if t.numel() != partition.num_layers:
                raise ValueError(f"per-layer sigma must have {partition.num_layers} entries")
            self.sigma = t

    def sigma_for(self, layer: int, logits_f32: torch.Tensor) -> torch.Tensor:
        if isinstance(self.sigma, str):
            return logits_f32.std(dim=-1, keepdim=True, correction=0).detach()
        return self.sigma[layer].to(logits_f32.device)

    def to_dict(self):
        sig = self.sigma if isinstance(self.sigma, str) else self.sigma.tolist()
        return {"kind": self.kind, "delta": self.delta, "sigma": sig,
                "partition_hash": self.partition.hash()}


class ForceExpert:
    kind = "force"
    needs_agents = False

    def __init__(self, layer: int, expert: int):
        self.layer = int(layer)
        self.expert = int(expert)

    def to_dict(self):
        return {"kind": self.kind, "layer": self.layer, "expert": self.expert}


# ---------------------------------------------------------------------------- records


class RoutingRecords:
    """Per-layer lists of per-call tensors [B, T_call, ...]; stacked along T on CPU."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.reset()

    def reset(self):
        self.indices = [[] for _ in range(self.num_layers)]
        self.logits = [[] for _ in range(self.num_layers)]
        self.weights = [[] for _ in range(self.num_layers)]
        self.n_calls = 0

    @staticmethod
    def _stack(per_layer, dtype):
        if not per_layer or not per_layer[0]:
            raise RuntimeError("nothing recorded")
        return torch.stack([torch.cat(calls, dim=1) for calls in per_layer], dim=0).to("cpu", dtype)

    def stack(self) -> torch.Tensor:
        return self._stack(self.indices, torch.long)

    def stack_logits(self) -> torch.Tensor:
        return self._stack(self.logits, torch.float32)

    def stack_weights(self) -> torch.Tensor:
        return self._stack(self.weights, torch.float32)


# ---------------------------------------------------------------------------- hook


def _moe_layers(model):
    inner = getattr(model, "model", model)
    return [layer.mlp for layer in inner.layers]


class MoERoutingHook:
    def __init__(self, model, rule=None, record: bool = True, record_logits: bool = False,
                 record_weights: bool = False):
        self.model = model
        self.rule = rule if rule is not None else FreeRouting()
        self.record = record
        self.record_logits = record_logits
        self.record_weights = record_weights
        self.mlps = _moe_layers(model)
        self.records = RoutingRecords(len(self.mlps))
        self.row_agents: list | None = None
        self.installed = False
        self._mask_cache: dict = {}

    # ------------------------------------------------------------------ lifecycle
    def install(self):
        if self.installed:
            return self
        for i, mlp in enumerate(self.mlps):
            if "forward" in mlp.__dict__:
                raise RuntimeError(f"layer {i} mlp already has an instance-level forward (another hook?)")
            mlp.forward = types.MethodType(self._make_forward(i), mlp)
        self.installed = True
        return self

    def uninstall(self):
        for mlp in self.mlps:
            mlp.__dict__.pop("forward", None)
        self.installed = False

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()
        return False

    # ------------------------------------------------------------------ config
    def set_rule(self, rule):
        self.rule = rule
        self._mask_cache = {}
        if self.row_agents is not None:
            self.set_row_agents(self.row_agents)

    def set_row_agents(self, agents: Sequence[str | None] | None):
        if agents is None:
            self.row_agents = None
            return
        agents = list(agents)
        part = getattr(self.rule, "partition", None)
        if part is not None:
            bad = [a for a in agents if a is not None and a not in part.roles]
            if bad:
                raise ValueError(f"unknown agents {bad}; partition roles are {part.roles}")
        self.row_agents = agents

    def reset_records(self):
        self.records.reset()

    # ------------------------------------------------------------------ helpers
    def _role_masks(self, device):
        """(shared [L,E] bool, own [R,L,E] bool) on device, cached."""
        key = (str(device), id(self.rule))
        if key not in self._mask_cache:
            p = self.rule.partition
            shared = p.shared_mask().to(device)
            own = torch.stack([p.private_mask(r) for r in p.roles]).to(device)
            self._mask_cache[key] = (shared, own)
        return self._mask_cache[key]

    def _agent_rows(self, B: int):
        """Returns (role_idx LongTensor [B] with -1 for None, any_agent bool)."""
        if self.row_agents is None:
            raise ValueError(f"rule {self.rule.kind!r} needs set_row_agents(...) before forward")
        if len(self.row_agents) != B:
            raise ValueError(f"set_row_agents got {len(self.row_agents)} agents but batch size is {B}")
        roles = self.rule.partition.roles
        idx = [roles.index(a) if a is not None else -1 for a in self.row_agents]
        return torch.tensor(idx, dtype=torch.long), any(i >= 0 for i in idx)

    def _make_forward(self, layer_idx: int):
        hook = self

        def forward(mlp, hidden_states):
            B, T, H = hidden_states.shape
            flat = hidden_states.reshape(-1, H)
            router = mlp.router
            k = router.top_k
            logits, scores, indices = router(flat)  # exact router path
            rule = hook.rule
            kind = rule.kind
            custom = None  # (idx [B,T,k], agented rows bool [B]) or None

            if kind == "free" or (kind == "soft" and rule.delta == 0.0):
                if rule.needs_agents and hook.row_agents is not None and len(hook.row_agents) != B:
                    raise ValueError(f"set_row_agents got {len(hook.row_agents)} agents but batch size is {B}")
            elif kind == "force":
                if layer_idx == rule.layer:
                    sel = indices.clone()
                    present = (sel == rule.expert).any(-1)
                    sel[~present, k - 1] = rule.expert  # topk is sorted desc: last slot is the lowest
                    custom = (sel.reshape(B, T, k), torch.ones(B, dtype=torch.bool, device=flat.device))
            elif kind in ("hard", "soft"):
                role_idx, any_agent = hook._agent_rows(B)
                # non-privatised layer (e.g. P-late below layer 12): plain base top-k for every row
                if any_agent and rule.partition.is_privatised(layer_idx):
                    lg = logits.reshape(B, T, -1)
                    shared, own = hook._role_masks(lg.device)  # [L,E], [R,L,E]
                    role_idx = role_idx.to(lg.device)
                    agented = role_idx >= 0
                    own_rows = own[role_idx.clamp(min=0), layer_idx]  # [B,E]
                    own_rows = own_rows & agented[:, None]
                    if kind == "hard":
                        if rule.n_shared_k + rule.n_private_k != k:
                            raise ValueError("n_shared_k + n_private_k must equal the router's top_k")
                        sh = lg.masked_fill(~shared[layer_idx][None, None, :], float("-inf"))
                        _, i_sh = torch.topk(sh, rule.n_shared_k, dim=-1)
                        pv = lg.masked_fill(~own_rows[:, None, :], float("-inf"))
                        _, i_pv = torch.topk(pv, rule.n_private_k, dim=-1)
                        sel = torch.cat([i_sh, i_pv], dim=-1)
                    else:
                        any_priv = own[:, layer_idx].any(0)  # [E] any role's private
                        other_rows = any_priv[None, :] & ~own_rows
                        sign = own_rows.float() - other_rows.float()  # [B,E]
                        lf = lg.float()
                        sigma = rule.sigma_for(layer_idx, lf)
                        biased = lf + rule.delta * sigma * sign[:, None, :]
                        _, sel = torch.topk(biased, k, dim=-1)
                    custom = (sel, agented)
            else:
                raise ValueError(f"unknown rule kind {kind!r}")

            if custom is not None:
                sel, rows = custom
                sel_flat = sel.reshape(-1, k)
                w_custom = F.softmax(torch.gather(logits, 1, sel_flat), dim=-1, dtype=logits.dtype)
                row_tok = rows[:, None].expand(B, T).reshape(-1, 1)
                if bool(rows.all()):
                    indices, scores = sel_flat, w_custom
                    out = mlp.experts(flat, indices, scores)
                else:
                    # Mixed batch: free rows must be bit-identical to the unhooked model, and expert matmuls
                    # are not guaranteed row-independent across different token subsets, so run the experts
                    # once exactly as the base model does and once with the custom routing, then select rows.
                    out_free = mlp.experts(flat, indices, scores)
                    indices = torch.where(row_tok, sel_flat, indices)
                    scores = torch.where(row_tok, w_custom, scores)
                    out_custom = mlp.experts(flat, indices, scores)
                    out = torch.where(row_tok, out_custom, out_free)
            else:
                out = mlp.experts(flat, indices, scores)
            out = out.reshape(B, T, H)

            if hook.record:
                if layer_idx == 0:
                    hook.records.n_calls += 1
                hook.records.indices[layer_idx].append(indices.detach().reshape(B, T, k).to("cpu", copy=True))
                if hook.record_logits:
                    hook.records.logits[layer_idx].append(
                        logits.detach().reshape(B, T, -1).to("cpu", torch.float32, copy=True))
                if hook.record_weights:
                    hook.records.weights[layer_idx].append(
                        scores.detach().reshape(B, T, k).to("cpu", torch.float32, copy=True))
            return out, scores

        return forward


# ---------------------------------------------------------------------------- checks


def check_rule_holds(indices: torch.Tensor, rule, agents: Sequence[str | None] | None = None,
                     logits: torch.Tensor | None = None) -> bool:
    """Check recorded routing [L,B,T,k] against the rule.

    HardSplit: every token of every agented row has exactly n_shared_k shared + n_private_k own-private.
    ForceExpert: the forced expert is present in every token's selection at that layer.
    FreeRouting / SoftNudge: no hard constraint; only the shape is checked, unless ``logits`` [L,B,T,E] are
    given, in which case the selection must be a (tie-robust) top-k of the (biased) logits: the minimum
    selected score >= the maximum unselected score, for every token.
    """
    if isinstance(indices, np.ndarray):
        indices = torch.from_numpy(indices)
    indices = indices.long().cpu()
    if indices.ndim != 4:
        return False
    L, B, T, k = indices.shape
    kind = rule.kind
    if kind == "hard":
        if agents is None or len(agents) != B:
            return False
        p = rule.partition
        owner = torch.from_numpy(p.owner_matrix())  # [L,E]
        for b, a in enumerate(agents):
            if a is None:
                continue
            code = p.roles.index(a) + 1
            pl = list(p.privatised_layers())
            if not pl:
                continue
            codes = torch.stack([owner[l][indices[l, b]] for l in pl])  # [L',T,k]
            if not ((codes == 0).sum(-1) == rule.n_shared_k).all():
                return False
            if not ((codes == code).sum(-1) == rule.n_private_k).all():
                return False
        return True
    if kind == "force":
        if not 0 <= rule.layer < L:
            return False
        return bool((indices[rule.layer] == rule.expert).any(-1).all())
    if kind in ("free", "soft"):
        if logits is None:
            return True
        lf = logits.float().cpu()
        if kind == "soft" and rule.delta != 0.0:
            if agents is None or len(agents) != B:
                return False
            p = rule.partition
            sign = torch.zeros(L, B, lf.shape[-1])
            any_priv = torch.stack([p.private_mask(r) for r in p.roles]).any(0)  # [L,E]
            for b, a in enumerate(agents):
                if a is None:
                    continue
                own = p.private_mask(a)
                sign[:, b] = own.float() - (any_priv & ~own).float()
            if isinstance(rule.sigma, str):
                sigma = lf.std(dim=-1, keepdim=True, correction=0)
            else:
                sigma = rule.sigma.reshape(L, 1, 1, 1)
            lf = lf + rule.delta * sigma * sign[:, :, None, :]
        sel_scores = torch.gather(lf, -1, indices)
        unsel = lf.scatter(-1, indices, float("-inf"))
        lo, hi = sel_scores.min(-1).values, unsel.max(-1).values
        # small tolerance: the hook computed sigma on the model device, this recomputation is on CPU
        return bool((lo >= hi - 1e-4 * (1.0 + hi.abs())).all())
    raise ValueError(f"unknown rule kind {kind!r}")
