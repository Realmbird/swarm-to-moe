import numpy as np
import torch

from swarm_to_moe.profile import PeakTracker, profile_experts

from .conftest import NUM_EXPERTS, NUM_LAYERS, TOP_K


class CharTokenizer:
    """Minimal left-padding tokenizer stub: one id per character (ids 2..255), pad id 0."""

    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    padding_side = "left"

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False):
        seqs = [[2 + (ord(c) % 250) for c in t] for t in texts]
        T = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), T, dtype=torch.long)
        att = torch.zeros(len(seqs), T, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, T - len(s):] = torch.tensor(s)
            att[i, T - len(s):] = 1
        return {"input_ids": ids, "attention_mask": att}


TEXTS = ["hello world", "a much longer piece of text here", "x", "routing probes", "abc def ghi"]


def test_peak_tracker_does_not_change_output(tiny_model, input_ids):
    with torch.no_grad():
        base = tiny_model(input_ids).logits
        with PeakTracker(tiny_model) as tr:
            out = tiny_model(input_ids).logits
        after = tiny_model(input_ids).logits
    assert torch.allclose(base, out, atol=1e-5)
    assert torch.equal(base, after)
    assert all(ex.__dict__.get("forward") is None for ex in tr.experts)
    assert (tr.peak > 0).any()


def test_profile_usage_and_peak(tiny_model):
    tok = CharTokenizer()
    usage, peak, sigma = profile_experts(tiny_model, tok, TEXTS, batch_size=2, return_sigma=True, progress=False)
    n_tokens = sum(len(t) for t in TEXTS)
    assert usage.shape == (NUM_LAYERS, NUM_EXPERTS) and peak.shape == (NUM_LAYERS, NUM_EXPERTS)
    assert np.allclose(usage.sum(-1), n_tokens * TOP_K)  # padding excluded
    assert (peak[usage > 0] > 0).all()
    assert (peak[usage == 0] == 0).all()
    assert sigma.shape == (NUM_LAYERS,) and (sigma > 0).all()
    # model unpatched afterwards
    assert all(l.mlp.__dict__.get("forward") is None for l in tiny_model.model.layers)
    assert all(l.mlp.experts.__dict__.get("forward") is None for l in tiny_model.model.layers)
