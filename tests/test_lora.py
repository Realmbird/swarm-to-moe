"""LoRA on private experts, trained with the routing rule active (E2 step 1).

API under test (swarm_to_moe/lora.py):
  LoRAConfig(rank, alpha, dropout)
  attach_expert_lora(model, partition, roles, cfg) -> ExpertLoRAHandle
  handle.parameters(), handle.parameters_for_role(role), handle.num_parameters()
  handle.remove(); handle.merge_()  (writes deltas into expert weights, then removes)
  handle.state_dict() / handle.load_state_dict(sd); save(path) / load(path, model, partition, roles)
  handle.delta_norms() -> {(layer, expert): float}
  handle.random_init_(seed, match=other_handle)  (placebo: noise with matched per-expert delta norm)
"""
import copy

import pytest
import torch

from swarm_to_moe.hook import FreeRouting, HardSplit, MoERoutingHook
from swarm_to_moe.lora import LoRAConfig, attach_expert_lora, load_expert_lora

from .conftest import NUM_LAYERS, ROLES


def _logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


def _expert_weight_snapshot(model):
    return [
        (l.mlp.experts.gate_up_proj.detach().clone(), l.mlp.experts.down_proj.detach().clone())
        for l in model.model.layers
    ]


@pytest.fixture
def model(tiny_model):
    # deep copy so LoRA/merge never leaks into the session-scoped fixture
    return copy.deepcopy(tiny_model)


def test_zero_init_is_identity(model, input_ids, partition):
    base = _logits(model, input_ids)
    handle = attach_expert_lora(model, partition, ROLES, LoRAConfig(rank=4))
    assert torch.equal(_logits(model, input_ids), base)
    with MoERoutingHook(model, rule=HardSplit(partition)) as hook:
        hook.set_row_agents(["A", "B", "C", "D"])
        hard_with = _logits(model, input_ids)
    handle.remove()
    with MoERoutingHook(model, rule=HardSplit(partition)) as hook:
        hook.set_row_agents(["A", "B", "C", "D"])
        hard_without = _logits(model, input_ids)
    assert torch.equal(hard_with, hard_without)
    assert all(l.mlp.experts.__dict__.get("forward") is None for l in model.model.layers)


def test_parameter_count_and_role_split(model, partition):
    cfg = LoRAConfig(rank=4)
    handle = attach_expert_lora(model, partition, ROLES, cfg)
    ex = model.model.layers[0].mlp.experts
    H = ex.hidden_size
    I2 = 2 * ex.intermediate_size
    I = ex.intermediate_size
    per_expert = cfg.rank * (H + I2) + cfg.rank * (I + H)
    n_private = sum(len(partition.private[r][l]) for r in ROLES for l in range(NUM_LAYERS))
    assert handle.num_parameters() == per_expert * n_private
    assert sum(p.numel() for p in handle.parameters()) == handle.num_parameters()
    per_role = {r: sum(p.numel() for p in handle.parameters_for_role(r)) for r in ROLES}
    assert sum(per_role.values()) == handle.num_parameters()
    assert len(set(per_role.values())) == 1  # equal private set sizes -> equal param counts
    # only LoRA params require grad
    for n, p in model.named_parameters():
        if "lora" not in n.lower():
            assert not p.requires_grad, n


def test_gradients_isolated_to_active_role(model, input_ids, partition):
    handle = attach_expert_lora(model, partition, ROLES, LoRAConfig(rank=4))
    with MoERoutingHook(model, rule=HardSplit(partition)) as hook:
        hook.set_row_agents(["A", "A", "A", "A"])
        out = model(input_ids).logits
        loss = out.float().logsumexp(-1).mean()
        loss.backward()
    grads = {r: [p.grad for p in handle.parameters_for_role(r)] for r in ROLES}
    assert any(g is not None and g.abs().sum() > 0 for g in grads["A"])
    for r in ("B", "C", "D"):
        assert all(g is None or g.abs().sum() == 0 for g in grads[r]), f"role {r} got gradient"


def test_training_step_changes_output_and_only_lora(model, input_ids, partition):
    handle = attach_expert_lora(model, partition, ROLES, LoRAConfig(rank=4))
    before = _expert_weight_snapshot(model)
    base = _logits(model, input_ids)
    opt = torch.optim.SGD(handle.parameters(), lr=1.0)
    with MoERoutingHook(model, rule=HardSplit(partition)) as hook:
        hook.set_row_agents(["A", "B", "C", "D"])
        for _ in range(3):
            opt.zero_grad()
            loss = model(input_ids).logits.float().logsumexp(-1).mean()
            loss.backward()
            opt.step()
        after_train = _logits(model, input_ids)
    assert not torch.equal(after_train, base)
    for (g0, d0), l in zip(before, model.model.layers):
        assert torch.equal(g0, l.mlp.experts.gate_up_proj) and torch.equal(d0, l.mlp.experts.down_proj)


def test_merge_matches_lora_forward(model, input_ids, partition):
    handle = attach_expert_lora(model, partition, ROLES, LoRAConfig(rank=4))
    torch.manual_seed(1)
    for p in handle.parameters():
        with torch.no_grad():
            p.normal_(std=0.1)
    with MoERoutingHook(model, rule=HardSplit(partition)) as hook:
        hook.set_row_agents(["A", "B", "C", "D"])
        with_lora = _logits(model, input_ids)
    norms_before = handle.delta_norms()
    handle.merge_()
    assert all(l.mlp.experts.__dict__.get("forward") is None for l in model.model.layers)
    with MoERoutingHook(model, rule=HardSplit(partition)) as hook:
        hook.set_row_agents(["A", "B", "C", "D"])
        merged = _logits(model, input_ids)
    assert torch.allclose(with_lora, merged, atol=1e-4, rtol=1e-4)
    # untouched experts (shared) must be bit-identical to before; private ones changed
    assert len(norms_before) == sum(len(partition.private[r][l]) for r in ROLES for l in range(NUM_LAYERS))
    assert all(v > 0 for v in norms_before.values())


def test_merge_leaves_shared_experts_untouched(model, input_ids, partition):
    before = _expert_weight_snapshot(model)
    handle = attach_expert_lora(model, partition, ROLES, LoRAConfig(rank=4))
    for p in handle.parameters():
        with torch.no_grad():
            p.fill_(0.05)
    handle.merge_()
    for l, ((g0, d0), layer) in enumerate(zip(before, model.model.layers)):
        for e in partition.shared[l]:
            assert torch.equal(g0[e], layer.mlp.experts.gate_up_proj[e])
            assert torch.equal(d0[e], layer.mlp.experts.down_proj[e])
        for r in ROLES:
            for e in partition.private[r][l]:
                assert not torch.equal(g0[e], layer.mlp.experts.gate_up_proj[e])


def test_save_load_roundtrip(model, input_ids, partition, tmp_path):
    handle = attach_expert_lora(model, partition, ROLES, LoRAConfig(rank=4))
    torch.manual_seed(2)
    for p in handle.parameters():
        with torch.no_grad():
            p.normal_(std=0.1)
    with MoERoutingHook(model, rule=HardSplit(partition)) as hook:
        hook.set_row_agents(["A", "B", "C", "D"])
        ref = _logits(model, input_ids)
    path = tmp_path / "lora.pt"
    handle.save(path)
    handle.remove()
    handle2 = load_expert_lora(path, model, partition, ROLES)
    assert handle2.cfg.rank == 4
    with MoERoutingHook(model, rule=HardSplit(partition)) as hook:
        hook.set_row_agents(["A", "B", "C", "D"])
        back = _logits(model, input_ids)
    assert torch.equal(ref, back)


def test_placebo_random_init_matches_norms(model, input_ids, partition):
    trained = attach_expert_lora(model, partition, ROLES, LoRAConfig(rank=4))
    torch.manual_seed(3)
    for p in trained.parameters():
        with torch.no_grad():
            p.normal_(std=0.1)
    ref_norms = trained.delta_norms()
    trained.remove()
    base = _logits(model, input_ids)
    placebo = attach_expert_lora(model, partition, ROLES, LoRAConfig(rank=4))
    placebo.random_init_(seed=0, match=ref_norms)
    got = placebo.delta_norms()
    for k, v in ref_norms.items():
        assert got[k] == pytest.approx(v, rel=1e-3)
    assert not torch.equal(_logits(model, input_ids), base)
    # different seed -> different weights, same norms
    placebo2 = attach_expert_lora(copy.deepcopy(model), partition, ROLES, LoRAConfig(rank=4))
    placebo2.random_init_(seed=1, match=ref_norms)
    a = next(iter(placebo.parameters()))
    b = next(iter(placebo2.parameters()))
    assert not torch.equal(a, b)


def test_subset_of_roles(model, partition):
    handle = attach_expert_lora(model, partition, ("A", "B"), LoRAConfig(rank=2))
    assert set(handle.roles) == {"A", "B"}
    assert len(handle.delta_norms()) == sum(len(partition.private[r][l]) for r in ("A", "B") for l in range(NUM_LAYERS))
