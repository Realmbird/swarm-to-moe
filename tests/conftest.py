"""Shared fixtures: a tiny random gpt-oss model on CPU so the real HF MoE code path is exercised.

Geometry chosen so Martin's hard split (top-3 shared + top-1 private) fits in top-4:
  16 experts, top-4, 4 roles -> 8 shared + 4 roles x 2 private per layer.
"""
import numpy as np
import pytest
import torch
from transformers import GptOssConfig, GptOssForCausalLM

from swarm_to_moe.partition import Partition

ROLES = ("A", "B", "C", "D")
NUM_LAYERS = 3
NUM_EXPERTS = 16
TOP_K = 4
N_SHARED = 8


@pytest.fixture(scope="session")
def tiny_config():
    return GptOssConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=1,
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_config):
    torch.manual_seed(0)
    model = GptOssForCausalLM(tiny_config).eval()
    # Spread router logits out so top-k choices are not degenerate ties.
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.router.weight.mul_(8.0)
    return model


@pytest.fixture
def input_ids():
    g = torch.Generator().manual_seed(123)
    return torch.randint(2, 256, (4, 12), generator=g)


@pytest.fixture(scope="session")
def partition():
    """Deterministic hand-built partition: experts 0-7 shared, then 2 per role in order."""
    shared = tuple(tuple(range(N_SHARED)) for _ in range(NUM_LAYERS))
    private = {
        role: tuple(tuple(range(N_SHARED + 2 * i, N_SHARED + 2 * i + 2)) for _ in range(NUM_LAYERS))
        for i, role in enumerate(ROLES)
    }
    return Partition(num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS, roles=ROLES, shared=shared, private=private)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
