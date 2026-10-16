"""Tests for the MoE routing hook. These mirror Martin's three sanity tests (CLAUDE.md §1):
  1. hook-off (FreeRouting) reproduces base tokens exactly;
  2. the rule holds on every token;
  3. forcing a named expert changes output (proves the hook is live).
"""
import torch

from swarm_to_moe.hook import (
    ForceExpert,
    FreeRouting,
    HardSplit,
    MoERoutingHook,
    SoftNudge,
    check_rule_holds,
)

from .conftest import N_SHARED, NUM_EXPERTS, NUM_LAYERS, ROLES, TOP_K


def _logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


# ----------------------------------------------------------------------------- identity


def test_free_routing_reproduces_base_exactly(tiny_model, input_ids):
    base = _logits(tiny_model, input_ids)
    with MoERoutingHook(tiny_model, rule=FreeRouting()) as hook:
        hooked = _logits(tiny_model, input_ids)
    after = _logits(tiny_model, input_ids)
    assert torch.equal(base, hooked)
    assert torch.equal(base, after)  # uninstall restores the original forward


def test_soft_nudge_delta_zero_equals_base(tiny_model, input_ids, partition):
    base = _logits(tiny_model, input_ids)
    with MoERoutingHook(tiny_model, rule=SoftNudge(partition, delta=0.0)) as hook:
        hook.set_row_agents(["A", "B", "C", "D"])
        out = _logits(tiny_model, input_ids)
    assert torch.equal(base, out)


def test_rows_with_no_agent_route_freely(tiny_model, input_ids, partition):
    base = _logits(tiny_model, input_ids)
    with MoERoutingHook(tiny_model, rule=HardSplit(partition)) as hook:
        hook.set_row_agents([None, "B", None, "D"])
        out = _logits(tiny_model, input_ids)
    assert torch.equal(base[0], out[0])
    assert torch.equal(base[2], out[2])
    assert not torch.equal(base[1], out[1])


# ----------------------------------------------------------------------------- records


def test_records_shape_single_forward(tiny_model, input_ids):
    with MoERoutingHook(tiny_model, rule=FreeRouting()) as hook:
        _logits(tiny_model, input_ids)
        idx = hook.records.stack()
    B, T = input_ids.shape
    assert idx.shape == (NUM_LAYERS, B, T, TOP_K)
    assert idx.dtype == torch.long
    assert idx.min() >= 0 and idx.max() < NUM_EXPERTS
    # each token's top-k are distinct experts
    sorted_idx, _ = idx.sort(dim=-1)
    assert (sorted_idx[..., 1:] != sorted_idx[..., :-1]).all()


def test_records_match_router_topk(tiny_model, input_ids):
    """Recorded indices must be the router's own top-k (as a set per token) under FreeRouting."""
    layer0 = tiny_model.model.layers[0]
    captured = {}

    def grab(mod, args, out):
        captured["h"] = out.detach().clone()  # the MLP receives the layernorm OUTPUT

    h = layer0.post_attention_layernorm.register_forward_hook(grab)
    try:
        with MoERoutingHook(tiny_model, rule=FreeRouting()) as hook:
            _logits(tiny_model, input_ids)
            rec = hook.records.stack()[0]  # [B, T, k]
    finally:
        h.remove()
    router = layer0.mlp.router
    with torch.no_grad():
        _, _, ref = router(captured["h"].reshape(-1, captured["h"].shape[-1]))
    ref = ref.reshape(rec.shape)
    assert torch.equal(rec.sort(-1).values, ref.sort(-1).values)


def test_records_accumulate_across_generate(tiny_model, input_ids):
    new = 3
    with MoERoutingHook(tiny_model, rule=FreeRouting()) as hook:
        with torch.no_grad():
            tiny_model.generate(input_ids, max_new_tokens=new, do_sample=False, pad_token_id=0)
        idx = hook.records.stack()
    B, T = input_ids.shape
    # prefill T tokens, then one forward per generated token except the last
    assert idx.shape == (NUM_LAYERS, B, T + new - 1, TOP_K)
    assert hook.records.n_calls == new


def test_reset_records(tiny_model, input_ids):
    with MoERoutingHook(tiny_model, rule=FreeRouting()) as hook:
        _logits(tiny_model, input_ids)
        hook.reset_records()
        assert hook.records.n_calls == 0
        _logits(tiny_model, input_ids[:2, :5])
        assert hook.records.stack().shape == (NUM_LAYERS, 2, 5, TOP_K)


def test_record_logits_optional(tiny_model, input_ids):
    with MoERoutingHook(tiny_model, rule=FreeRouting(), record_logits=True) as hook:
        _logits(tiny_model, input_ids)
        lg = hook.records.stack_logits()
    B, T = input_ids.shape
    assert lg.shape == (NUM_LAYERS, B, T, NUM_EXPERTS)
    assert lg.dtype == torch.float32


# ----------------------------------------------------------------------------- hard split


def test_hard_split_rule_holds_every_token(tiny_model, input_ids, partition):
    agents = ["A", "B", "C", "D"]
    rule = HardSplit(partition)
    with MoERoutingHook(tiny_model, rule=rule) as hook:
        hook.set_row_agents(agents)
        _logits(tiny_model, input_ids)
        idx = hook.records.stack()
    assert check_rule_holds(idx, rule, agents)
    owner = partition.owner_matrix()  # [L, E]
    for l in range(NUM_LAYERS):
        for b, role in enumerate(agents):
            codes = owner[l][idx[l, b].numpy()]  # [T, k]
            assert ((codes == 0).sum(-1) == 3).all(), "exactly 3 shared per token"
            assert ((codes == ROLES.index(role) + 1).sum(-1) == 1).all(), "exactly 1 own-private per token"


def test_hard_split_private_pick_is_argmax_within_private(tiny_model, input_ids, partition):
    """The private slot must be the highest-logit expert among the agent's private set (not arbitrary)."""
    agents = ["A", "A", "C", "C"]
    rule = HardSplit(partition)
    with MoERoutingHook(tiny_model, rule=rule, record_logits=True) as hook:
        hook.set_row_agents(agents)
        _logits(tiny_model, input_ids)
        idx = hook.records.stack()
        lg = hook.records.stack_logits()
    for l in range(NUM_LAYERS):
        for b, role in enumerate(agents):
            priv = list(partition.private[role][l])
            chosen_priv = [e for e in idx[l, b, 0].tolist() if e in priv]
            assert len(chosen_priv) == 1
            best = priv[int(lg[l, b, 0, priv].argmax())]
            assert chosen_priv[0] == best


def test_check_rule_holds_detects_violation(partition):
    rule = HardSplit(partition)
    agents = ["A"]
    # all-shared selection violates "1 private"
    bad = torch.tensor([[[[0, 1, 2, 3]]]])  # [L=1, B=1, T=1, k=4]
    bad = bad.expand(NUM_LAYERS, 1, 1, 4)
    assert not check_rule_holds(bad, rule, agents)


def test_hard_split_changes_output_per_row(tiny_model, input_ids, partition):
    base = _logits(tiny_model, input_ids)
    with MoERoutingHook(tiny_model, rule=HardSplit(partition)) as hook:
        hook.set_row_agents(["A", "B", "C", "D"])
        out = _logits(tiny_model, input_ids)
    assert not torch.equal(base, out)
    # Different agents on identical input must differ from each other.
    same = input_ids[:1].repeat(4, 1)
    with MoERoutingHook(tiny_model, rule=HardSplit(partition)) as hook:
        hook.set_row_agents(["A", "B", "C", "D"])
        out2 = _logits(tiny_model, same)
    assert not torch.equal(out2[0], out2[1])


# ----------------------------------------------------------------------------- soft nudge


def test_soft_nudge_large_delta_prefers_own_private(tiny_model, input_ids, partition):
    agents = ["A", "B", "C", "D"]
    rule = SoftNudge(partition, delta=50.0)
    with MoERoutingHook(tiny_model, rule=rule) as hook:
        hook.set_row_agents(agents)
        _logits(tiny_model, input_ids)
        idx = hook.records.stack()
    owner = partition.owner_matrix()
    for l in range(NUM_LAYERS):
        for b, role in enumerate(agents):
            codes = owner[l][idx[l, b].numpy()]
            own = ROLES.index(role) + 1
            # both own-private experts selected, no other agent's private experts selected
            assert ((codes == own).sum(-1) == 2).all()
            others = [ROLES.index(r) + 1 for r in ROLES if r != role]
            assert not any((codes == o).any() for o in others)


def test_soft_nudge_weights_come_from_unbiased_scores(tiny_model, input_ids, partition):
    """With a huge delta the *selection* changes but mixing weights must still be softmax of raw logits
    over the selected set. We check this indirectly: the hook exposes the weights it used."""
    agents = ["A", "A", "A", "A"]
    rule = SoftNudge(partition, delta=50.0)
    with MoERoutingHook(tiny_model, rule=rule, record_logits=True, record_weights=True) as hook:
        hook.set_row_agents(agents)
        _logits(tiny_model, input_ids)
        idx = hook.records.stack()
        lg = hook.records.stack_logits()
        w = hook.records.stack_weights()  # [L, B, T, k] float32
    sel = torch.gather(lg, -1, idx)
    expected = torch.softmax(sel, dim=-1)
    assert torch.allclose(w, expected, atol=1e-5)


def test_soft_nudge_moderate_delta_partial_overlap(tiny_model, input_ids, partition):
    """delta=0.5 sigma should move routing but not completely (Jaccard with base strictly between 0 and 1)."""
    with MoERoutingHook(tiny_model, rule=FreeRouting()) as hook:
        _logits(tiny_model, input_ids)
        base_idx = hook.records.stack()
    with MoERoutingHook(tiny_model, rule=SoftNudge(partition, delta=0.5)) as hook:
        hook.set_row_agents(["A", "B", "C", "D"])
        _logits(tiny_model, input_ids)
        soft_idx = hook.records.stack()
    inter = 0
    union = 0
    for a, b in zip(base_idx.reshape(-1, TOP_K), soft_idx.reshape(-1, TOP_K)):
        sa, sb = set(a.tolist()), set(b.tolist())
        inter += len(sa & sb)
        union += len(sa | sb)
    j = inter / union
    assert 0.3 < j < 1.0, f"Jaccard {j:.3f}"


# ----------------------------------------------------------------------------- forced expert


def test_force_expert_changes_output_and_is_recorded(tiny_model, input_ids):
    base = _logits(tiny_model, input_ids)
    rule = ForceExpert(layer=1, expert=5)
    with MoERoutingHook(tiny_model, rule=rule) as hook:
        out = _logits(tiny_model, input_ids)
        idx = hook.records.stack()
    assert not torch.equal(base, out)
    assert (idx[1] == 5).any(-1).all(), "expert 5 present in every token's top-k at layer 1"
    # other layers untouched
    with MoERoutingHook(tiny_model, rule=FreeRouting()) as hook:
        _logits(tiny_model, input_ids)
        free = hook.records.stack()
    assert torch.equal(idx[0].sort(-1).values, free[0].sort(-1).values)


# ----------------------------------------------------------------------------- misc


def test_set_row_agents_length_mismatch_raises(tiny_model, input_ids, partition):
    import pytest

    with MoERoutingHook(tiny_model, rule=HardSplit(partition)) as hook:
        hook.set_row_agents(["A", "B"])  # batch is 4
        with pytest.raises(ValueError):
            _logits(tiny_model, input_ids)


def test_unknown_agent_raises(tiny_model, input_ids, partition):
    import pytest

    with MoERoutingHook(tiny_model, rule=HardSplit(partition)) as hook:
        with pytest.raises(ValueError):
            hook.set_row_agents(["A", "B", "C", "Z"])


def test_hook_only_patches_moe_layers(tiny_model, input_ids):
    originals = [type(l.mlp).forward for l in tiny_model.model.layers]
    with MoERoutingHook(tiny_model, rule=FreeRouting()):
        patched = [l.mlp.forward for l in tiny_model.model.layers]
        assert all(p.__func__ is not o if hasattr(p, "__func__") else True for p, o in zip(patched, originals))
    restored = [l.mlp.__dict__.get("forward") for l in tiny_model.model.layers]
    assert all(r is None for r in restored), "instance-level forward override must be removed on uninstall"
