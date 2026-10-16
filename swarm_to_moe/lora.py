"""LoRA adapters on the private experts of a gpt-oss MoE (E2 step 1).

For every layer ``l``, every role ``r`` in ``roles`` and every expert ``e`` in ``partition.private[r][l]`` we add
low-rank deltas to that expert's ``gate_up_proj[e]`` ([H, 2I]) and ``down_proj[e]`` ([I, H]):

    dW_gu[e] = (A_gu[e] @ B_gu[e]) * alpha / rank      A_gu [H, rank], B_gu [rank, 2I]
    dW_dn[e] = (A_dn[e] @ B_dn[e]) * alpha / rank      A_dn [I, rank], B_dn [rank, H]

A ~ N(0, 1/fan_in) (fan_in = H or I), B = 0, so a freshly attached adapter is an exact no-op.
LoRA parameters are float32 regardless of the base dtype (AdamW on bf16 LoRA weights is unstable).

How the forward stays exact
---------------------------
The stock ``GptOssExperts.forward`` dispatches on ``config._experts_implementation`` (``grouped_mm`` on both the
real model and the tiny test model), which is not bit-identical to an eager per-expert loop. So instead of
re-implementing the experts, the patched forward returns

    out = stock_forward(h, idx, w) + sum_{lora'd e hit} w[t, slot] * (f_e^lora(h_t) - f_e^base(h_t))

where the difference is computed WITHOUT forming either full output:

    gu_base = h @ W_gu[e] + b_gu[e]                        (base dtype, saves only a view of W)
    gu_lora = gu_base.f32 + ((h.f32 @ A_gu) @ B_gu) * s
    dg      = gate(gu_lora) - gate(gu_base.f32)            (float32)
    diff    = (dg.to(W dtype) @ W_dn[e]).f32 + ((gate(gu_lora) @ A_dn) @ B_dn) * s

(the down projection is linear, so its bias cancels). With B = 0, ``gu_lora`` is ``gu_base.f32 + 0`` bit for bit,
``dg`` is exactly 0, ``diff`` is exactly 0 and ``out`` is the stock output plus exact zeros: the zero-init model is
bit-identical to the base model for any expert kernel. Gradients reach the LoRA parameters only through ``diff``;
the gradient w.r.t. the hidden states is that of f^lora (stock + diff - base), so training is exact.
"""
from __future__ import annotations

import json
import math
import types
import warnings
from dataclasses import asdict, dataclass
from typing import Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .partition import Partition


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0

    @property
    def scale(self) -> float:
        return float(self.alpha) / float(self.rank)


_PARAM_NAMES = ("A_gu", "B_gu", "A_dn", "B_dn")


class ExpertLoRA(nn.Module):
    """LoRA parameters for the lora'd experts of one ``GptOssExperts`` module (attached as ``experts.lora``)."""

    def __init__(self, hidden_size: int, intermediate_size: int, experts_roles: Mapping[int, str], cfg: LoRAConfig,
                 device=None):
        super().__init__()
        self.rank = int(cfg.rank)
        self.scale = cfg.scale
        self.dropout = float(cfg.dropout)
        self.role_of = {int(e): str(r) for e, r in experts_roles.items()}
        self.expert_ids = sorted(self.role_of)
        H, I, k = hidden_size, intermediate_size, self.rank
        kw = {"device": device, "dtype": torch.float32}
        self.A_gu = nn.ParameterDict()
        self.B_gu = nn.ParameterDict()
        self.A_dn = nn.ParameterDict()
        self.B_dn = nn.ParameterDict()
        for e in self.expert_ids:
            key = str(e)
            self.A_gu[key] = nn.Parameter(torch.randn(H, k, **kw) / math.sqrt(H))
            self.B_gu[key] = nn.Parameter(torch.zeros(k, 2 * I, **kw))
            self.A_dn[key] = nn.Parameter(torch.randn(I, k, **kw) / math.sqrt(I))
            self.B_dn[key] = nn.Parameter(torch.zeros(k, H, **kw))
        # [E+1] bool lookup "is expert e lora'd", built lazily on the routing device
        self.register_buffer("_is_lora", torch.zeros(0, dtype=torch.bool), persistent=False)

    def params_of(self, e: int):
        key = str(int(e))
        return self.A_gu[key], self.B_gu[key], self.A_dn[key], self.B_dn[key]

    def delta_gu(self, e: int) -> torch.Tensor:
        a, b, _, _ = self.params_of(e)
        return (a @ b) * self.scale

    def delta_dn(self, e: int) -> torch.Tensor:
        _, _, a, b = self.params_of(e)
        return (a @ b) * self.scale


def _lora_experts_forward(ex, hidden_states, router_indices=None, routing_weights=None):
    """Instance-level replacement for ``GptOssExperts.forward``; see the module docstring."""
    out = type(ex).forward(ex, hidden_states, router_indices, routing_weights)
    lora: ExpertLoRA = ex.lora
    if not lora.expert_ids:
        return out
    E = ex.num_experts
    if lora._is_lora.numel() != E + 1 or lora._is_lora.device != router_indices.device:
        m = torch.zeros(E + 1, dtype=torch.bool, device=router_indices.device)
        m[lora.expert_ids] = True
        lora._is_lora = m
    with torch.no_grad():
        # one host sync per layer: all (token, slot) pairs routed to a lora'd expert, grouped by expert
        tok, slot = torch.nonzero(lora._is_lora[router_indices.clamp(0, E)], as_tuple=True)
        exp = router_indices[tok, slot]
        order = torch.argsort(exp, stable=True)
        tok, slot = tok[order], slot[order]
        counts = torch.bincount(exp, minlength=E).tolist()
    groups = [(e, c) for e, c in enumerate(counts) if c > 0]
    if not groups:
        return out
    tok_g = torch.split(tok, [c for _, c in groups])
    slot_g = torch.split(slot, [c for _, c in groups])
    corr = torch.zeros(hidden_states.shape[0], hidden_states.shape[1], dtype=torch.float32,
                       device=hidden_states.device)
    s = lora.scale
    drop = lora.dropout if (lora.training and lora.dropout > 0) else 0.0
    with torch.autocast(hidden_states.device.type, enabled=False):  # keep the delta path in float32
        for (e, _), token_idx, sl in zip(groups, tok_g, slot_g):
            corr = _expert_correction(ex, lora, e, hidden_states, token_idx, sl, routing_weights, corr, s, drop)
    return out + corr.to(out.dtype)


def _expert_correction(ex, lora, e, hidden_states, token_idx, slot, routing_weights, corr, s, drop):
    a_gu, b_gu, a_dn, b_dn = lora.params_of(e)
    x = hidden_states[token_idx]
    gu_base = x @ ex.gate_up_proj[e] + ex.gate_up_proj_bias[e]
    xf = x.float()
    if drop:
        xf = F.dropout(xf, drop, training=True)
    gu_base_f = gu_base.float()
    gu_lora = gu_base_f + ((xf @ a_gu) @ b_gu) * s
    g_lora = ex._apply_gate(gu_lora)
    dg = g_lora - ex._apply_gate(gu_base_f)
    g_in = F.dropout(g_lora, drop, training=True) if drop else g_lora
    diff = (dg.to(ex.down_proj.dtype) @ ex.down_proj[e]).float() + ((g_in @ a_dn) @ b_dn) * s
    w = routing_weights[token_idx, slot, None].float()
    return corr.index_add(0, token_idx, diff * w)


def _layers(model):
    inner = getattr(model, "model", model)
    return list(inner.layers)


def _is_our_forward(ex) -> bool:
    f = ex.__dict__.get("forward")
    return f is not None and getattr(f, "__func__", None) is _lora_experts_forward


class ExpertLoRAHandle:
    def __init__(self, model, partition: Partition, roles: Sequence[str], cfg: LoRAConfig,
                 modules: dict[int, ExpertLoRA], saved_requires_grad: dict):
        self.model = model
        self.partition = partition
        self.roles = tuple(roles)
        self.cfg = cfg
        self.modules = modules  # layer -> ExpertLoRA
        self._saved_requires_grad = saved_requires_grad
        self.attached = True

    # ------------------------------------------------------------------ parameters
    def _iter(self):
        for l in sorted(self.modules):
            m = self.modules[l]
            for e in m.expert_ids:
                yield l, e, m

    def parameters(self) -> Iterator[nn.Parameter]:
        for _, e, m in self._iter():
            yield from m.params_of(e)

    def parameters_for_role(self, role: str) -> Iterator[nn.Parameter]:
        if role not in self.roles:
            raise ValueError(f"role {role!r} not in {self.roles}")
        for _, e, m in self._iter():
            if m.role_of[e] == role:
                yield from m.params_of(e)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def expert_roles(self) -> dict[tuple[int, int], str]:
        return {(l, e): m.role_of[e] for l, e, m in self._iter()}

    # ------------------------------------------------------------------ deltas
    @torch.no_grad()
    def delta_norms(self) -> dict[tuple[int, int], float]:
        out = {}
        for l, e, m in self._iter():
            g = m.delta_gu(e).float()
            d = m.delta_dn(e).float()
            out[(l, e)] = float(torch.sqrt(g.pow(2).sum() + d.pow(2).sum()))
        return out

    @torch.no_grad()
    def random_init_(self, seed: int, match: Mapping[tuple[int, int], float]):
        """Placebo: A, B ~ N(0, 1) from ``torch.Generator().manual_seed(seed)`` (drawn on CPU in a fixed order, so
        the noise is device-independent), then both B's of each expert are rescaled so that the expert's total
        delta Frobenius norm equals ``match[(layer, expert)]``. Keys missing from ``match`` raise KeyError."""
        g = torch.Generator().manual_seed(int(seed))
        for l, e, m in self._iter():
            for p in m.params_of(e):
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32))
        cur = self.delta_norms()
        for l, e, m in self._iter():
            target = float(match[(int(l), int(e))])
            c = cur[(l, e)]
            f = 0.0 if c == 0 else target / c
            _, b_gu, _, b_dn = m.params_of(e)
            b_gu.mul_(f)
            b_dn.mul_(f)
        return self

    # ------------------------------------------------------------------ lifecycle
    def remove(self):
        if not self.attached:
            return
        layers = _layers(self.model)
        for l, m in self.modules.items():
            ex = layers[l].mlp.experts
            if getattr(ex, "lora", None) is m:
                if _is_our_forward(ex):
                    del ex.__dict__["forward"]
                del ex.lora
        for p, flag in self._saved_requires_grad.items():
            p.requires_grad_(flag)
        self.attached = False

    @torch.no_grad()
    def merge_(self):
        """Write every delta into the expert weights (float32 add, cast to the weight dtype), then ``remove()``.

        Returns {(layer, expert): relative merge error} = ||applied change - delta|| / ||delta||. In bf16 a small
        delta is partly rounded away (bf16 has 8 mantissa bits: a delta of rms ~1e-4 on weights of std ~0.02 loses
        ~20% of its norm, ~1e-5 loses ~85%), so the merged bf16 model is NOT the attached model; keep the adapter
        attached (exact float32 delta path) for measurements. Warns if any error exceeds 5%."""
        if not self.attached:
            raise RuntimeError("handle is not attached")
        layers = _layers(self.model)
        errs = {}
        for l, e, m in self._iter():
            ex = layers[l].mlp.experts
            num = den = 0.0
            for P, d in ((ex.gate_up_proj, m.delta_gu(e)), (ex.down_proj, m.delta_dn(e))):
                d = d.to(P.device)
                old = P[e].float()
                P[e] = (old + d).to(P.dtype)
                num += float(((P[e].float() - old) - d).pow(2).sum())
                den += float(d.pow(2).sum())
            errs[(l, e)] = math.sqrt(num / den) if den > 0 else 0.0
        self.remove()
        worst = max(errs.values(), default=0.0)
        if worst > 0.05:
            warnings.warn(f"merge_: up to {worst:.1%} of a LoRA delta was lost to rounding in the weight dtype")
        return errs

    # ------------------------------------------------------------------ io
    def state_dict(self) -> dict[str, torch.Tensor]:
        sd = {}
        for l, m in self.modules.items():
            for k, v in m.state_dict().items():
                sd[f"layers.{l}.{k}"] = v.detach().clone()
        return sd

    @torch.no_grad()
    def load_state_dict(self, sd: Mapping[str, torch.Tensor]):
        mine = {f"layers.{l}.{k}": v for l, m in self.modules.items() for k, v in m.state_dict(keep_vars=True).items()}
        missing = set(mine) - set(sd)
        extra = set(sd) - set(mine)
        if missing or extra:
            raise ValueError(f"LoRA state mismatch: missing {sorted(missing)[:5]}, unexpected {sorted(extra)[:5]}")
        for k, p in mine.items():
            if p.shape != sd[k].shape:
                raise ValueError(f"{k}: shape {tuple(sd[k].shape)} != {tuple(p.shape)}")
            p.copy_(sd[k].to(p.device, p.dtype))
        return self

    def save(self, path, extra: Mapping | None = None):
        obj = {
            "cfg": asdict(self.cfg),
            "roles": list(self.roles),
            "partition_hash": self.partition.hash(),
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
        }
        if extra is not None:
            # JSON round trip: plain types only, so the checkpoint loads with torch.load(weights_only=True)
            obj["extra"] = json.loads(json.dumps(dict(extra), default=str))
        torch.save(obj, str(path))


def attach_expert_lora(model, partition: Partition, roles: Sequence[str], cfg: LoRAConfig | None = None
                       ) -> ExpertLoRAHandle:
    """Freeze the base model and attach zero-init LoRA to the private experts of ``roles``."""
    cfg = cfg or LoRAConfig()
    roles = tuple(roles)
    bad = [r for r in roles if r not in partition.roles]
    if bad:
        raise ValueError(f"roles {bad} not in partition roles {partition.roles}")
    layers = _layers(model)
    if len(layers) != partition.num_layers:
        raise ValueError(f"model has {len(layers)} layers, partition {partition.num_layers}")
    saved = {}
    for p in model.parameters():
        saved[p] = p.requires_grad
        p.requires_grad_(False)
    modules = {}
    for l, layer in enumerate(layers):
        ex = layer.mlp.experts
        if "forward" in ex.__dict__:
            if _is_our_forward(ex):
                # e.g. a deepcopy of a model that already carried an adapter: replace it
                warnings.warn(f"layer {l}: replacing an existing expert LoRA")
                del ex.__dict__["forward"]
                if hasattr(ex, "lora"):
                    del ex.lora
            else:
                raise RuntimeError(f"layer {l}: experts.forward is already patched by something else")
        er = {}
        for r in roles:
            for e in partition.private[r][l]:
                er[int(e)] = r
        if not er:
            continue
        m = ExpertLoRA(ex.hidden_size, ex.intermediate_size, er, cfg, device=ex.gate_up_proj.device)
        ex.lora = m  # registered submodule -> parameter names contain "lora"
        ex.forward = types.MethodType(_lora_experts_forward, ex)
        modules[l] = m
    for m in modules.values():
        for p in m.parameters():
            p.requires_grad_(True)
    return ExpertLoRAHandle(model, partition, roles, cfg, modules, saved)


def load_expert_lora_checkpoint(path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=True)


def load_expert_lora(path, model, partition: Partition, roles: Sequence[str] | None = None) -> ExpertLoRAHandle:
    ck = load_expert_lora_checkpoint(path)
    if ck["partition_hash"] != partition.hash():
        raise ValueError(f"checkpoint partition hash {ck['partition_hash']} != {partition.hash()}")
    saved_roles = tuple(ck["roles"])
    if roles is None:
        roles = saved_roles
    elif tuple(roles) != saved_roles:
        raise ValueError(f"roles {tuple(roles)} != checkpoint roles {saved_roles}")
    handle = attach_expert_lora(model, partition, roles, LoRAConfig(**ck["cfg"]))
    handle.load_state_dict(ck["state_dict"])
    handle.extra = ck.get("extra")
    return handle
