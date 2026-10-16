"""Partially privatised partitions (P-late style): only some layers have private stripes."""
import numpy as np
import torch

from swarm_to_moe.hook import FreeRouting, HardSplit, MoERoutingHook, SoftNudge, check_rule_holds
from swarm_to_moe.partition import Partition, build_partition, random_partition

from .conftest import N_SHARED, NUM_EXPERTS, NUM_LAYERS, ROLES

LATE = (2,)


def _late_partition():
    shared, private = [], {r: [] for r in ROLES}
    for l in range(NUM_LAYERS):
        if l in LATE:
            shared.append(tuple(range(N_SHARED)))
            for i, r in enumerate(ROLES):
                private[r].append(tuple(range(N_SHARED + 2 * i, N_SHARED + 2 * i + 2)))
        else:
            shared.append(tuple(range(NUM_EXPERTS)))
            for r in ROLES:
                private[r].append(())
    return Partition(num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS, roles=ROLES, shared=shared, private=private)


def test_late_partition_validates_and_hash_differs(partition):
    p = _late_partition()
    p.validate()
    assert p.privatised_layers() == LATE
    assert p.hash() != partition.hash()
    assert Partition.from_json(p.to_json()) == p


def test_hard_split_late_only_constrains_late_layers(tiny_model, input_ids):
    p = _late_partition()
    agents = ["A", "B", "C", "D"]
    with MoERoutingHook(tiny_model, rule=FreeRouting()) as hook:
        with torch.no_grad():
            tiny_model(input_ids)
        base = hook.records.stack()
    rule = HardSplit(p)
    with MoERoutingHook(tiny_model, rule=rule) as hook:
        hook.set_row_agents(agents)
        with torch.no_grad():
            tiny_model(input_ids)
        idx = hook.records.stack()
    assert torch.equal(idx[0], base[0]) and torch.equal(idx[1], base[1])
    assert check_rule_holds(idx, rule, agents)
    owner = p.owner_matrix()
    for b, role in enumerate(agents):
        codes = owner[2][idx[2, b].numpy()]
        assert ((codes == 0).sum(-1) == 3).all()
        assert ((codes == ROLES.index(role) + 1).sum(-1) == 1).all()


def test_soft_nudge_late_no_bias_early_and_per_layer_sigma(tiny_model, input_ids):
    p = _late_partition()
    with MoERoutingHook(tiny_model, rule=FreeRouting()) as hook:
        with torch.no_grad():
            tiny_model(input_ids)
        base = hook.records.stack()
    rule = SoftNudge(p, delta=50.0, sigma=torch.ones(NUM_LAYERS))
    with MoERoutingHook(tiny_model, rule=rule, record_logits=True) as hook:
        hook.set_row_agents(["A", "B", "C", "D"])
        with torch.no_grad():
            tiny_model(input_ids)
        idx = hook.records.stack()
        lg = hook.records.stack_logits()
    assert torch.equal(idx[0], base[0]) and torch.equal(idx[1], base[1])
    assert check_rule_holds(idx, rule, ["A", "B", "C", "D"], logits=lg)


def test_builders_accept_layers_kwarg():
    rng = np.random.default_rng(0)
    L, E = 24, 32
    usage, peak = rng.exponential(size=(L, E)), rng.random((L, E))
    late = build_partition(usage, peak, ROLES, n_shared=16, layers=range(12, 24))
    late.validate()
    assert late.privatised_layers() == tuple(range(12, 24))
    assert all(len(late.shared[l]) == E for l in range(12))
    full = build_partition(usage, peak, ROLES, n_shared=16)
    assert full.hash() != late.hash()
    r = random_partition(L, E, ROLES, 16, seed=0, layers=range(12, 24))
    r.validate()
    assert r.privatised_layers() == tuple(range(12, 24))
    # swap refinement: every role's total private usage within 1.2x of every other's
    totals = np.array([sum(usage[l, e] for l in range(L) for e in full.private[x][l]) for x in ROLES])
    assert totals.max() / totals.min() <= 1.2
