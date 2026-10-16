"""Expert partitions: per layer, a shared pool plus one private stripe per agent role.

A ``Partition`` assigns every (layer, expert) pair to exactly one owner: the shared pool or one role.
``build_partition`` implements Martin's recipe (CLAUDE.md section 1); ``random_partition`` is the P-rand control.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


def _canon_layers(layers) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(sorted(int(e) for e in layer)) for layer in layers)


@dataclass(eq=True)
class Partition:
    num_layers: int
    num_experts: int
    roles: tuple[str, ...]
    shared: tuple[tuple[int, ...], ...]
    private: Mapping[str, tuple[tuple[int, ...], ...]] = field(default_factory=dict)

    def __post_init__(self):
        self.num_layers = int(self.num_layers)
        self.num_experts = int(self.num_experts)
        self.roles = tuple(str(r) for r in self.roles)
        self.shared = _canon_layers(self.shared)
        # dict equality is order independent, so equality is by content
        self.private = {str(r): _canon_layers(v) for r, v in dict(self.private).items()}

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        if set(self.private) != set(self.roles):
            raise ValueError(f"private roles {sorted(self.private)} != roles {sorted(self.roles)}")
        if len(self.roles) != len(set(self.roles)):
            raise ValueError("duplicate role names")
        if "shared" in self.roles:
            raise ValueError("'shared' is reserved and cannot be a role name")
        if len(self.shared) != self.num_layers:
            raise ValueError("shared must have one entry per layer")
        for r in self.roles:
            if len(self.private[r]) != self.num_layers:
                raise ValueError(f"private[{r!r}] must have one entry per layer")
        for layer in range(self.num_layers):
            count = np.zeros(self.num_experts, dtype=np.int64)
            for groups in [self.shared[layer]] + [self.private[r][layer] for r in self.roles]:
                for e in groups:
                    if not 0 <= e < self.num_experts:
                        raise ValueError(f"expert {e} out of range at layer {layer}")
                    count[e] += 1
            if (count > 1).any():
                raise ValueError(f"layer {layer}: experts owned more than once: {np.nonzero(count > 1)[0].tolist()}")
            if (count == 0).any():
                raise ValueError(f"layer {layer}: unassigned experts: {np.nonzero(count == 0)[0].tolist()}")

    # ------------------------------------------------------------------ lookups
    def privatised_layers(self) -> tuple[int, ...]:
        """Layers where at least one role has a non-empty private set. A non-privatised layer has
        shared = all experts and is unconstrained for every rule (e.g. P-late leaves layers 0-11 free)."""
        return tuple(l for l in range(self.num_layers) if any(self.private[r][l] for r in self.roles))

    def is_privatised(self, layer: int) -> bool:
        return any(self.private[r][layer] for r in self.roles)

    def owner(self, layer: int, expert: int) -> str:
        if expert in self.shared[layer]:
            return "shared"
        for r in self.roles:
            if expert in self.private[r][layer]:
                return r
        raise ValueError(f"(layer={layer}, expert={expert}) has no owner")

    def owner_matrix(self) -> np.ndarray:
        """[L, E] int: 0 = shared, i+1 = roles[i]; -1 = unassigned (never for a valid partition)."""
        om = np.full((self.num_layers, self.num_experts), -1, dtype=np.int64)
        for layer in range(self.num_layers):
            om[layer, list(self.shared[layer])] = 0
            for i, r in enumerate(self.roles):
                om[layer, list(self.private[r][layer])] = i + 1
        return om

    def shared_mask(self) -> torch.Tensor:
        return torch.from_numpy(self.owner_matrix() == 0)

    def private_mask(self, role: str) -> torch.Tensor:
        if role not in self.roles:
            raise ValueError(f"unknown role {role!r}")
        return torch.from_numpy(self.owner_matrix() == self.roles.index(role) + 1)

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        return {
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "roles": list(self.roles),
            "shared": [list(x) for x in self.shared],
            "private": {r: [list(x) for x in self.private[r]] for r in sorted(self.private)},
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, d: Mapping) -> "Partition":
        return cls(
            num_layers=d["num_layers"],
            num_experts=d["num_experts"],
            roles=tuple(d["roles"]),
            shared=d["shared"],
            private=d["private"],
        )

    @classmethod
    def from_json(cls, s: str) -> "Partition":
        return cls.from_dict(json.loads(s))

    def save(self, path) -> None:
        with open(path, "w") as f:
            f.write(self.to_json(indent=2))

    @classmethod
    def load(cls, path) -> "Partition":
        with open(path) as f:
            return cls.from_json(f.read())

    def hash(self) -> str:
        """12 hex chars of sha256 over the canonical (sorted keys, sorted expert lists, compact) JSON."""
        canon = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------- builders


def build_partition(
    usage: np.ndarray,
    peak: np.ndarray,
    roles: Sequence[str],
    n_shared: int,
    super_frac: float = 0.01,
    layers: Iterable[int] | None = None,
    balance_ratio: float = 1.2,
    max_swaps: int = 100,
) -> Partition:
    """Martin's recipe.

    1. Super experts: global top ``round(super_frac * L * E)`` (>= 1) (layer, expert) pairs by ``peak``.
    2. Per layer, shared = that layer's super experts, filled with the most-used experts (by ``usage``)
       up to ``n_shared`` (if supers alone exceed ``n_shared`` all are kept).
    3. Remaining experts, sorted by usage descending, are snake-dealt to roles. To balance usage totals
       across layers as well as within a layer, the deal order of each layer is the roles sorted by their
       cumulative private usage so far (lowest total picks first), ties by role order.
    4. Refinement (paper draft section 3.2): while max/min role total private usage > ``balance_ratio``,
       swap one expert between the highest- and lowest-total roles within one layer (the swap that brings
       the two totals closest together), at most ``max_swaps`` times. Set sizes are unchanged.

    ``layers``: privatise only these layers (None = all); other layers have every expert shared.
    Super experts are always chosen globally over all L*E pairs.
    """
    usage = np.asarray(usage, dtype=np.float64)
    peak = np.asarray(peak, dtype=np.float64)
    if usage.shape != peak.shape or usage.ndim != 2:
        raise ValueError("usage and peak must both be [L, E]")
    L, E = usage.shape
    roles = tuple(roles)
    R = len(roles)
    n_super = max(1, int(round(super_frac * L * E)))
    order = np.argsort(-peak.ravel(), kind="stable")[:n_super]
    supers: list[set[int]] = [set() for _ in range(L)]
    for f in order:
        layer, e = divmod(int(f), E)
        supers[layer].add(e)

    shared: list[list[int]] = []
    private: dict[str, list[list[int]]] = {r: [] for r in roles}
    totals = np.zeros(R)
    priv_layers = set(range(L)) if layers is None else {int(l) for l in layers}
    for layer in range(L):
        if layer not in priv_layers:
            shared.append(list(range(E)))
            for r in roles:
                private[r].append([])
            continue
        sh = set(supers[layer])
        by_use = [int(e) for e in np.argsort(-usage[layer], kind="stable")]
        for e in by_use:
            if len(sh) >= n_shared:
                break
            sh.add(e)
        shared.append(sorted(sh))
        rest = [e for e in by_use if e not in sh]
        deal = sorted(range(R), key=lambda i: (totals[i], i))
        snake = deal + deal[::-1]
        buckets: list[list[int]] = [[] for _ in range(R)]
        for j, e in enumerate(rest):
            i = snake[j % (2 * R)]
            buckets[i].append(e)
            totals[i] += usage[layer, e]
        for i, r in enumerate(roles):
            private[r].append(sorted(buckets[i]))

    _balance_swaps(private, usage, roles, balance_ratio, max_swaps)
    p = Partition(num_layers=L, num_experts=E, roles=roles, shared=shared, private=private)
    p.validate()
    return p


def _balance_swaps(private: dict, usage: np.ndarray, roles, ratio: float, max_swaps: int) -> int:
    """Greedy pairwise swaps between the highest/lowest-usage roles until max/min <= ratio. In place."""
    def totals():
        return np.array([sum(usage[l, e] for l in range(usage.shape[0]) for e in private[r][l]) for r in roles])

    n = 0
    for n in range(max_swaps):
        t = totals()
        if t.min() <= 0 or t.max() / t.min() <= ratio:
            return n
        hi, lo = roles[int(t.argmax())], roles[int(t.argmin())]
        gap = t.max() - t.min()
        best = None  # (resulting |gap|, layer, e_hi, e_lo)
        for l in range(usage.shape[0]):
            for eh in private[hi][l]:
                for el in private[lo][l]:
                    d = usage[l, eh] - usage[l, el]
                    if d <= 0:
                        continue
                    new_gap = abs(gap - 2 * d)
                    if new_gap < gap and (best is None or new_gap < best[0]):
                        best = (new_gap, l, eh, el)
        if best is None:
            return n
        _, l, eh, el = best
        private[hi][l] = sorted([e for e in private[hi][l] if e != eh] + [el])
        private[lo][l] = sorted([e for e in private[lo][l] if e != el] + [eh])
    return n


def random_partition(num_layers: int, num_experts: int, roles: Sequence[str], n_shared: int, seed: int,
                     layers: Iterable[int] | None = None) -> Partition:
    """Seeded uniform random deal: per layer a random permutation; first n_shared shared, rest in equal chunks.
    ``layers``: privatise only these layers (None = all); others are fully shared. The permutation is drawn
    for every layer regardless, so the privatised layers match the all-layer deal with the same seed."""
    roles = tuple(roles)
    R = len(roles)
    if (num_experts - n_shared) % R:
        raise ValueError("num_experts - n_shared must be divisible by the number of roles")
    per = (num_experts - n_shared) // R
    rng = np.random.default_rng(seed)
    shared, private = [], {r: [] for r in roles}
    priv_layers = set(range(num_layers)) if layers is None else {int(l) for l in layers}
    for layer in range(num_layers):
        perm = rng.permutation(num_experts).tolist()
        if layer not in priv_layers:
            shared.append(perm)
            for r in roles:
                private[r].append([])
            continue
        shared.append(perm[:n_shared])
        for i, r in enumerate(roles):
            private[r].append(perm[n_shared + i * per : n_shared + (i + 1) * per])
    p = Partition(num_layers=num_layers, num_experts=num_experts, roles=roles, shared=shared, private=private)
    p.validate()
    return p
