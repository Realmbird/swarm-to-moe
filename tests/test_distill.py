"""Self-distillation data pipeline and training step (E2 step 1).

API under test (swarm_to_moe/distill.py):
  DistillExample(role, input_text, target_text)
  build_examples(inputs, teacher_outputs: dict[role -> list[str|None]]) -> list[DistillExample]  (drops None targets)
  encode_example(tokenizer, ex, max_len, prompt_builder=None) -> dict(input_ids: LongTensor, labels: LongTensor, agent: str)
  collate(encoded: list[dict], pad_id) -> dict(input_ids [B,T], attention_mask [B,T], labels [B,T], agents: list[str])
  lora_train_step(model, hook, handle, batch, optimizer, grad_clip=1.0) -> float loss
"""
import copy
import os

import pytest
import torch

from swarm_to_moe.distill import DistillExample, build_examples, collate, encode_example, lora_train_step
from swarm_to_moe.hook import HardSplit, MoERoutingHook
from swarm_to_moe.lora import LoRAConfig, attach_expert_lora

from .conftest import ROLES

GPT_OSS_SNAPSHOT_GLOB = "/workspace/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*"


def _real_tokenizer():
    import glob

    paths = glob.glob(GPT_OSS_SNAPSHOT_GLOB)
    if not paths:
        pytest.skip("gpt-oss-20b tokenizer not cached")
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return AutoTokenizer.from_pretrained(paths[0])


# ----------------------------------------------------------------------------- examples


def test_build_examples_counts_and_drops_missing():
    inputs = ["q0", "q1", "q2"]
    teacher = {"planner": ["p0", "p1", "p2"], "critic": ["c0", None, "c2"]}
    ex = build_examples(inputs, teacher)
    assert len(ex) == 5
    assert all(isinstance(e, DistillExample) for e in ex)
    assert {(e.role, e.input_text) for e in ex if e.role == "critic"} == {("critic", "q0"), ("critic", "q2")}
    assert [e.target_text for e in ex if e.role == "planner"] == ["p0", "p1", "p2"]


def test_build_examples_rejects_length_mismatch():
    with pytest.raises(ValueError):
        build_examples(["a", "b"], {"r": ["x"]})


# ----------------------------------------------------------------------------- encoding (real tokenizer, offline)


def test_encode_example_masks_prompt_and_terminates():
    tok = _real_tokenizer()
    ex = DistillExample(role="planner", input_text="What is 2+2?", target_text="Plan: add the numbers. Answer: 4")
    enc = encode_example(tok, ex, max_len=512)
    ids, labels = enc["input_ids"], enc["labels"]
    assert ids.dtype == torch.long and labels.shape == ids.shape
    assert enc["agent"] == "planner"
    n_prompt = int((labels == -100).sum())
    assert n_prompt > 0 and n_prompt < len(ids)
    # masked positions are exactly a prefix
    assert (labels[:n_prompt] == -100).all() and (labels[n_prompt:] != -100).all()
    # the prompt does NOT contain the role system prompt (student sees no role)
    prompt_text = tok.decode(ids[:n_prompt], skip_special_tokens=False)
    assert "What is 2+2?" in prompt_text
    assert "planner" not in prompt_text.lower()
    # target ends with a Harmony terminator token
    tail = tok.decode(ids[-1:], skip_special_tokens=False)
    assert tail in ("<|return|>", "<|end|>")
    # target text is reproduced
    assert "Answer: 4" in tok.decode(ids[n_prompt:], skip_special_tokens=True)


def test_encode_example_truncates_target_not_prompt():
    tok = _real_tokenizer()
    long_target = "word " * 2000
    ex = DistillExample(role="critic", input_text="Short question?", target_text=long_target)
    enc = encode_example(tok, ex, max_len=128)
    assert len(enc["input_ids"]) <= 128
    n_prompt = int((enc["labels"] == -100).sum())
    assert n_prompt < 128 and "Short question?" in tok.decode(enc["input_ids"][:n_prompt], skip_special_tokens=False)


# ----------------------------------------------------------------------------- collate


def test_collate_right_pads_and_masks():
    a = {"input_ids": torch.tensor([5, 6, 7]), "labels": torch.tensor([-100, 6, 7]), "agent": "A"}
    b = {"input_ids": torch.tensor([8, 9]), "labels": torch.tensor([-100, 9]), "agent": "B"}
    batch = collate([a, b], pad_id=0)
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 9, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[-100, 6, 7], [-100, 9, -100]]
    assert batch["agents"] == ["A", "B"]


# ----------------------------------------------------------------------------- training step on the tiny model


def test_lora_train_step_overfits_fixed_batch(tiny_model, partition):
    model = copy.deepcopy(tiny_model)
    handle = attach_expert_lora(model, partition, ROLES, LoRAConfig(rank=4, alpha=8))
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(2, 256, (4, 16), generator=g)
    labels = ids.clone()
    labels[:, :6] = -100
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": labels,
        "agents": ["A", "B", "C", "D"],
    }
    opt = torch.optim.AdamW(handle.parameters(), lr=5e-3)
    with MoERoutingHook(model, rule=HardSplit(partition), record=False) as hook:
        losses = [lora_train_step(model, hook, handle, batch, opt) for _ in range(25)]
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert losses[-1] < losses[0] * 0.9, losses
    # hook must have been told the agents (private slot active for every row)
    assert hook.row_agents == ["A", "B", "C", "D"]


def test_lora_train_step_ignores_masked_labels(tiny_model, partition):
    """All labels masked -> loss is zero/NaN-free handling: expect a ValueError rather than silent NaN."""
    model = copy.deepcopy(tiny_model)
    handle = attach_expert_lora(model, partition, ROLES, LoRAConfig(rank=2))
    ids = torch.randint(2, 256, (2, 8))
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": torch.full_like(ids, -100),
        "agents": ["A", "B"],
    }
    opt = torch.optim.SGD(handle.parameters(), lr=0.1)
    with MoERoutingHook(model, rule=HardSplit(partition), record=False) as hook:
        with pytest.raises(ValueError):
            lora_train_step(model, hook, handle, batch, opt)
