import json

import numpy as np
import pytest
import torch

from swarm_to_moe.partition import Partition, build_partition, random_partition

from .conftest import N_SHARED, NUM_EXPERTS, NUM_LAYERS, ROLES


def test_validate_disjoint_and_covering(partition):
    partition.validate()  # must not raise


def test_validate_rejects_overlap():
    shared = ((0, 1),)
    private = {"A": ((1, 2),), "B": ((3,),)}
    with pytest.raises(ValueError):
        Partition(num_layers=1, num_experts=4, roles=("A", "B"), shared=shared, private=private).validate()


def test_validate_rejects_gap():
    shared = ((0,),)
    private = {"A": ((1,),), "B": ((2,),)}  # expert 3 unassigned
    with pytest.raises(ValueError):
        Partition(num_layers=1, num_experts=4, roles=("A", "B"), shared=shared, private=private).validate()


def test_owner_lookup(partition):
    assert partition.owner(0, 0) == "shared"
    assert partition.owner(0, N_SHARED) == "A"
    assert partition.owner(2, N_SHARED + 2) == "B"
    assert partition.owner(1, NUM_EXPERTS - 1) == "D"


def test_masks(partition):
    shared = partition.shared_mask()
    assert shared.shape == (NUM_LAYERS, NUM_EXPERTS) and shared.dtype == torch.bool
    assert shared.sum().item() == NUM_LAYERS * N_SHARED
    total = shared.clone()
    for role in ROLES:
        m = partition.private_mask(role)
        assert m.sum().item() == NUM_LAYERS * 2
        assert not (total & m).any()
        total |= m
    assert total.all()


def test_owner_matrix_codes(partition):
    om = partition.owner_matrix()  # 0 = shared, 1..R = roles in partition.roles order
    assert om.shape == (NUM_LAYERS, NUM_EXPERTS)
    assert om[0, 0] == 0
    assert om[0, N_SHARED] == 1
    assert om[0, NUM_EXPERTS - 1] == len(ROLES)


def test_json_roundtrip_and_hash_stable(partition):
    s = partition.to_json()
    back = Partition.from_json(s)
    assert back == partition
    assert back.hash() == partition.hash()
    assert len(partition.hash()) == 12
    # hash must depend on content, not on dict ordering
    d = json.loads(s)
    d["private"] = dict(reversed(list(d["private"].items())))
    assert Partition.from_json(json.dumps(d)).hash() == partition.hash()


def test_hash_changes_when_content_changes(partition):
    other = random_partition(NUM_LAYERS, NUM_EXPERTS, ROLES, N_SHARED, seed=1)
    assert other.hash() != partition.hash()


def test_random_partition_valid_and_seeded():
    p1 = random_partition(NUM_LAYERS, NUM_EXPERTS, ROLES, N_SHARED, seed=7)
    p2 = random_partition(NUM_LAYERS, NUM_EXPERTS, ROLES, N_SHARED, seed=7)
    p3 = random_partition(NUM_LAYERS, NUM_EXPERTS, ROLES, N_SHARED, seed=8)
    p1.validate()
    assert p1 == p2
    assert p1 != p3
    for layer in range(NUM_LAYERS):
        assert len(p1.shared[layer]) == N_SHARED
        for role in ROLES:
            assert len(p1.private[role][layer]) == (NUM_EXPERTS - N_SHARED) // len(ROLES)


def test_build_partition_recipe(rng):
    """Martin's recipe: super experts (top 1% by peak) + most-used -> shared; rest snake-dealt by usage."""
    L, E = 24, 32
    usage = rng.random((L, E))
    peak = rng.random((L, E))
    # plant one obvious super expert per layer that is otherwise rarely used
    super_idx = rng.integers(0, E, size=L)
    for l in range(L):
        usage[l, super_idx[l]] = 0.0
        peak[l, super_idx[l]] = 100.0 + l
    p = build_partition(usage, peak, ROLES, n_shared=16, super_frac=0.01)
    p.validate()
    for l in range(L):
        shared = set(p.shared[l])
        assert len(shared) == 16
        # the most-used expert in the layer is shared
        assert int(np.argmax(usage[l])) in shared
        # every private set has the same size
        sizes = {len(p.private[r][l]) for r in ROLES}
        assert sizes == {4}
    # the top-1% by peak (L*E*0.01 ~ 7-8 experts globally) must all be shared
    flat = np.argsort(peak.ravel())[::-1][: max(1, int(round(0.01 * L * E)))]
    for f in flat:
        l, e = divmod(int(f), E)
        assert e in set(p.shared[l])


def test_build_partition_balances_usage(rng):
    """Snake-dealing should give roles roughly equal total usage, far more equal than a random deal."""
    L, E = 24, 32
    usage = rng.exponential(size=(L, E))
    peak = rng.random((L, E))
    p = build_partition(usage, peak, ROLES, n_shared=16, super_frac=0.01)
    totals = np.array([sum(usage[l, e] for l in range(L) for e in p.private[r][l]) for r in ROLES])
    spread = (totals.max() - totals.min()) / totals.mean()
    assert spread < 0.10, f"usage spread across roles too large: {spread:.3f}"
