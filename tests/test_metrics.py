import numpy as np
import pytest
import torch

from swarm_to_moe.metrics import (
    mcnemar_holm,
    owner_fractions,
    paired_bootstrap_ci,
    role_id_cv,
    role_preference,
    routing_jaccard,
    routing_jsd,
    usage_counts,
)

from .conftest import N_SHARED, NUM_EXPERTS, NUM_LAYERS, ROLES, TOP_K


def _synthetic_indices(partition, agents, T=20, match_p=1.0, seed=0):
    """Build [L, B, T, k] indices: 3 shared + 1 private per token; the private slot goes to the row's
    own role with prob match_p, else to a uniformly chosen other role."""
    rng = np.random.default_rng(seed)
    L, B = partition.num_layers, len(agents)
    out = np.zeros((L, B, T, TOP_K), dtype=np.int64)
    for l in range(L):
        for b, role in enumerate(agents):
            for t in range(T):
                shared = rng.choice(partition.shared[l], size=3, replace=False)
                if rng.random() < match_p:
                    r = role
                else:
                    r = rng.choice([x for x in ROLES if x != role])
                priv = rng.choice(partition.private[r][l])
                out[l, b, t] = np.concatenate([shared, [priv]])
    return torch.from_numpy(out)


# ----------------------------------------------------------------------------- usage


def test_usage_counts_shape_and_total(partition):
    agents = ["A", "B", "C", "D"]
    idx = _synthetic_indices(partition, agents, T=10)
    counts = usage_counts(idx, NUM_EXPERTS)  # [B, L, E]
    assert counts.shape == (4, NUM_LAYERS, NUM_EXPERTS)
    assert (counts.sum(-1) == 10 * TOP_K).all()


def test_usage_counts_respects_mask(partition):
    agents = ["A", "B"]
    idx = _synthetic_indices(partition, agents, T=10)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[0, :5] = False  # ignore first 5 tokens of row 0 (e.g. prompt tokens / padding)
    counts = usage_counts(idx, NUM_EXPERTS, mask=mask)
    assert counts[0].sum(-1).tolist() == [5 * TOP_K] * NUM_LAYERS
    assert counts[1].sum(-1).tolist() == [10 * TOP_K] * NUM_LAYERS


# ----------------------------------------------------------------------------- owner fractions


def test_owner_fractions_perfect_match(partition):
    agents = ["A", "B", "C", "D"]
    idx = _synthetic_indices(partition, agents, match_p=1.0)
    f = owner_fractions(idx, partition, agents)  # dict of [B, L] arrays
    assert set(f) >= {"match", "other", "shared"}
    assert np.allclose(f["shared"], 0.75)
    assert np.allclose(f["match"], 0.25)
    assert np.allclose(f["other"], 0.0)
    assert np.allclose(f["match"] + f["other"] + f["shared"], 1.0)


def test_owner_fractions_no_match(partition):
    agents = ["A", "B", "C", "D"]
    idx = _synthetic_indices(partition, agents, match_p=0.0)
    f = owner_fractions(idx, partition, agents)
    assert np.allclose(f["match"], 0.0)
    assert np.allclose(f["other"], 0.25)


def test_role_preference_sign_and_control(partition):
    """preference = match - other/(R-1): >0 when the router prefers own experts, ~0 under uniform choice,
    and shuffled role labels must drive it to ~0 even when the true routing is perfectly matched."""
    agents = ["A", "B", "C", "D"] * 8
    idx = _synthetic_indices(partition, agents, match_p=1.0, T=30)
    pref = role_preference(idx, partition, agents)  # [B, L]
    assert pref.shape == (32, NUM_LAYERS)
    assert (pref > 0.2).all()

    uniform = _synthetic_indices(partition, agents, match_p=0.25, T=30, seed=1)
    pref_u = role_preference(uniform, partition, agents)
    assert abs(pref_u.mean()) < 0.03

    rng = np.random.default_rng(0)
    shuffled = list(rng.permutation(agents))
    pref_s = role_preference(idx, partition, shuffled)
    assert pref_s.mean() < pref.mean() - 0.15
    assert abs(pref_s.mean()) < 0.08


# ----------------------------------------------------------------------------- distances


def test_routing_jsd_properties():
    L, E = 3, 16
    p = np.full((L, E), 1.0 / E)
    q = np.zeros((L, E))
    q[:, 0] = 1.0
    assert np.allclose(routing_jsd(p, p), 0.0)
    d = routing_jsd(p, q)
    assert d.shape == (L,)
    assert (d > 0).all() and (d <= np.log(2) + 1e-9).all()
    assert np.allclose(routing_jsd(p, q), routing_jsd(q, p))
    # accepts raw counts too (normalised internally)
    assert np.allclose(routing_jsd(p * 100, q * 7), d)


def test_routing_jaccard():
    a = torch.tensor([[[[0, 1, 2, 3]]]])
    b = torch.tensor([[[[0, 1, 2, 3]]]])
    c = torch.tensor([[[[0, 1, 9, 8]]]])
    assert routing_jaccard(a, b) == pytest.approx(1.0)
    assert routing_jaccard(a, c) == pytest.approx(2 / 6)


# ----------------------------------------------------------------------------- statistics


def test_paired_bootstrap_ci_covers_true_difference(rng):
    n = 400
    a = rng.normal(0.0, 1.0, n)
    b = a + 0.5 + rng.normal(0.0, 0.1, n)
    mean, lo, hi = paired_bootstrap_ci(b, a, n_boot=1000, seed=0)
    assert lo < 0.5 < hi
    assert mean == pytest.approx(0.5, abs=0.05)
    assert lo > 0.0  # clearly significant


def test_paired_bootstrap_ci_null(rng):
    a = rng.normal(size=300)
    b = a + rng.normal(scale=0.05, size=300)
    mean, lo, hi = paired_bootstrap_ci(b, a, n_boot=1000, seed=0)
    assert lo < 0.0 < hi


def test_paired_bootstrap_is_seeded(rng):
    a = rng.normal(size=100)
    b = a + 0.3
    assert paired_bootstrap_ci(b, a, n_boot=200, seed=3) == paired_bootstrap_ci(b, a, n_boot=200, seed=3)


def test_mcnemar_holm(rng):
    n = 500
    ref = rng.random(n) < 0.8
    same = ref.copy()
    # symmetric noise: flip 6 correct->wrong and 6 wrong->correct -> b == c, not significant
    correct_idx = np.flatnonzero(ref)
    wrong_idx = np.flatnonzero(~ref)
    same[rng.choice(correct_idx, 6, replace=False)] = False
    same[rng.choice(wrong_idx, 6, replace=False)] = True
    worse = ref.copy()
    hurt = ref & (rng.random(n) < 0.25)
    worse[hurt] = False  # clearly worse -> significant
    res = mcnemar_holm({"base": ref, "same": same, "worse": worse}, reference="base")
    assert set(res) == {"same", "worse"}
    assert res["worse"]["p_holm"] < 0.01 and res["worse"]["significant"]
    assert not res["same"]["significant"]
    assert res["worse"]["p_holm"] >= res["worse"]["p_raw"]
    assert res["worse"]["n_discordant"] == int((ref != worse).sum())


def test_role_id_cv_separable_and_chance(rng):
    n_per = 40
    R = 4
    D = 24 * 32
    centers = rng.normal(size=(R, D)) * 3
    X = np.concatenate([centers[r] + rng.normal(size=(n_per, D)) for r in range(R)])
    y = np.repeat(np.arange(R), n_per)
    acc, chance = role_id_cv(X, y, folds=5, seed=0)
    assert chance == pytest.approx(0.25)
    assert acc > 0.95

    X_noise = rng.normal(size=X.shape)
    acc_n, _ = role_id_cv(X_noise, y, folds=5, seed=0)
    assert acc_n < 0.45


def test_role_id_cv_accepts_string_labels(rng):
    X = rng.normal(size=(40, 10))
    y = np.array(["A", "B"] * 20)
    acc, chance = role_id_cv(X, y, folds=5, seed=0)
    assert chance == pytest.approx(0.5)
    assert 0.0 <= acc <= 1.0
