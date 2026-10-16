"""Pure helpers of scripts/run_e0c.py (teacher-forced fixed-text routing)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "run_e0c", Path(__file__).resolve().parents[1] / "scripts" / "run_e0c.py")
e0c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e0c)


def test_prepare_continuation_drops_stop_and_truncates():
    assert e0c.prepare_continuation([5, 6, 7, 99, 8], {99}, 10) == ([5, 6, 7], False)
    assert e0c.prepare_continuation([5, 6, 7, 8], {99}, 3) == ([5, 6, 7], True)
    assert e0c.prepare_continuation([99], {99}, 3) == ([], False)


@pytest.mark.parametrize("padding", ["left", "right"])
def test_build_batch_masks_and_identical_continuations(padding):
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8], [9]]
    cont = [10, 11, 12]
    ids, attn, cm, pos = e0c.build_batch(prompts, [cont] * 3, pad_id=0, padding=padding)
    assert ids.shape == (3, 8)
    for b, p in enumerate(prompts):
        assert ids[b][cm[b]].tolist() == cont
        assert int(attn[b].sum()) == len(p) + len(cont)
        assert ids[b][attn[b].bool()].tolist() == p + cont
        assert pos[b][attn[b].bool()].tolist() == list(range(len(p) + len(cont)))
        assert not (cm[b] & ~attn[b].bool()).any()
    if padding == "right":
        assert (ids[:, 0] == torch.tensor([1, 4, 9])).all()
    else:
        assert (ids[:, -1] == 12).all()


def test_build_batch_routes_like_unbatched(tiny_model):
    """Right padding: continuation routing of a row in a batch equals that row run alone."""
    from swarm_to_moe.hook import FreeRouting, MoERoutingHook

    prompts = [[3, 4, 5, 6, 7, 8], [9, 10]]
    cont = [20, 21, 22, 23]
    ids, attn, cm, pos = e0c.build_batch(prompts, [cont] * 2, pad_id=0, padding="right")
    with MoERoutingHook(tiny_model, FreeRouting()) as hook, torch.no_grad():
        tiny_model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
        batched = hook.records.stack()
        for b, p in enumerate(prompts):
            hook.reset_records()
            tiny_model(input_ids=torch.tensor([p + cont]), use_cache=False)
            alone = hook.records.stack()
            assert torch.equal(batched[:, b][:, cm[b]], alone[:, 0, len(p):])


def test_cross_predict_grouped_and_table():
    rng = np.random.default_rng(0)
    roles = ["a", "b", "c", "d"]
    n = 40
    centers = rng.normal(size=(4, 6)) * 4
    # training: label = role, features = role center + noise
    X_tr = np.concatenate([centers[i] + rng.normal(size=(n, 6)) for i in range(4)])
    y_tr = np.repeat(roles, n)
    g_tr = np.tile(np.arange(n), 4)
    # test: prompt role r, text role s; features follow the PROMPT role
    X_te, p_lab, t_lab, g_te = [], [], [], []
    for i, r in enumerate(roles):
        for s in roles:
            X_te.append(centers[i] + rng.normal(size=(n, 6)))
            p_lab += [r] * n
            t_lab += [s] * n
            g_te.append(np.arange(n))
    pred = e0c.cross_predict_grouped(X_tr, y_tr, g_tr, np.concatenate(X_te), np.concatenate(g_te), 5, 0)
    tab = e0c.prompt_vs_text_table(pred, p_lab, t_lab, roles)
    assert tab["offdiag_mean_p_prompt"] > 0.9
    assert tab["offdiag_mean_p_text"] < 0.1
    assert abs(tab["offdiag_mean_p_prompt"] + tab["offdiag_mean_p_text"] + tab["offdiag_mean_p_other"] - 1) < 1e-9
    with pytest.raises(ValueError):
        e0c.cross_predict_grouped(X_tr, y_tr, g_tr, X_tr[:1], np.array([n + 5]), 5, 0)
