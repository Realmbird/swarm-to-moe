#!/usr/bin/env python3
"""E0-c: is the role readable from FREE routing when the continuation text is held FIXED? (teacher forcing)

E0-b found role-ID from routing on generated tokens = 0.84 but text TF-IDF on the same generations = 0.83:
each role writes different text and routing tracks tokens, so the routing signal may be just the text.
E0-c removes that confound. For every input, the continuation is the base model's own E0-b generation
(raw Harmony text as stored in e0b.json, analysis channel included); only the system prompt varies.
Prefill-only forward (no sampling), router free, routing collected over continuation positions only.

Conditions (prompt arm | continuation text arm), 200 inputs each:
  fixed text : (r | no_role) for r in 4 roles, plus (no_role | no_role)          -> metrics (a), (c), (d), (e)
  cross      : (r | s) for all 16 (r, s) role pairs; the diagonal is "own text"  -> metric (b)

Features per (condition, input), as in run_e0b.py: per-layer L1-normalised expert-usage histogram over
continuation tokens (24x32) and per-layer mean router entropy (24). Folds grouped by input.
Writes results/E0/e0c.json and results/E0/e0c_hist.npz.
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from swarm_to_moe.hook import FreeRouting, MoERoutingHook  # noqa: E402
from swarm_to_moe.metrics import (  # noqa: E402
    cv_splits,
    l1_per_layer,
    role_id_cv,
    role_preference,
    routing_jsd,
    usage_counts,
)
from swarm_to_moe.model import DEFAULT_MODEL, _stop_ids, apply_chat, load_gpt_oss, router_entropy  # noqa: E402
from swarm_to_moe.partition import Partition  # noqa: E402
from swarm_to_moe.prompts import DEFAULT_ROLES, build_messages  # noqa: E402
from swarm_to_moe.runutil import read_json, run_meta, write_json  # noqa: E402

# same layer groups as run_e0b.py
LAYER_GROUPS = {"0-7": range(0, 8), "8-15": range(8, 16), "16-23": range(16, 24), "10-14": range(10, 15)}
ROLES = [r.name for r in DEFAULT_ROLES]
NO_ROLE = "no_role"


# ---------------------------------------------------------------------------- pure helpers (tested)


def prepare_continuation(ids, stop_ids, max_len: int) -> tuple[list[int], bool]:
    """Continuation token ids for teacher forcing: cut at (and drop) the first stop token, then truncate to
    ``max_len``. E0-b never fed the stop token back as input (its gen_mask excludes it), so dropping it keeps
    the routed positions comparable. Returns (ids, truncated)."""
    ids = list(ids)
    for j, t in enumerate(ids):
        if t in stop_ids:
            ids = ids[:j]
            break
    return ids[:max_len], len(ids) > max_len


def build_batch(prompt_ids, cont_ids, pad_id: int, padding: str = "right"):
    """Concatenate prompt + continuation per row and pad. Returns (input_ids [B,T], attention_mask [B,T],
    cont_mask [B,T], position_ids [B,T]) as torch tensors on CPU.

    cont_mask marks exactly the continuation tokens (never padding). position_ids count real tokens from 0
    (pad positions get 0), which is what ``generate`` uses on left-padded batches; with right padding they
    equal the default arange on the real tokens. Asserts the continuation slice equals ``cont_ids``."""
    if padding not in ("left", "right"):
        raise ValueError("padding must be 'left' or 'right'")
    B = len(prompt_ids)
    lens = [len(p) + len(c) for p, c in zip(prompt_ids, cont_ids)]
    T = max(lens)
    ids = torch.full((B, T), int(pad_id), dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.bool)
    cmask = torch.zeros((B, T), dtype=torch.bool)
    pos = torch.zeros((B, T), dtype=torch.long)
    for b, (p, c) in enumerate(zip(prompt_ids, cont_ids)):
        n = len(p) + len(c)
        off = T - n if padding == "left" else 0
        ids[b, off:off + n] = torch.tensor(list(p) + list(c), dtype=torch.long)
        attn[b, off:off + n] = True
        cmask[b, off + len(p):off + n] = True
        pos[b, off:off + n] = torch.arange(n)
        assert ids[b][cmask[b]].tolist() == list(c), "continuation ids changed while batching"
    return ids, attn.long(), cmask, pos


def cross_predict_grouped(X_train, y_train, g_train, X_test, g_test, folds: int = 5, seed: int = 0):
    """Train on (X_train, y_train) with the same grouped folds / pipeline as ``metrics.role_id_cv`` and predict
    each test row with the fold model whose training groups did NOT contain that row's group (input). Every
    test group must appear in g_train. Returns predictions for X_test (array of labels)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_train = np.asarray(X_train, np.float64).reshape(len(X_train), -1)
    X_test = np.asarray(X_test, np.float64).reshape(len(X_test), -1)
    y_train, g_train, g_test = np.asarray(y_train), np.asarray(g_train), np.asarray(g_test)
    if not set(np.unique(g_test)) <= set(np.unique(g_train)):
        raise ValueError("every test group must occur in the training groups")
    pred = np.empty(len(X_test), dtype=y_train.dtype)
    done = np.zeros(len(X_test), dtype=bool)
    for tr, te in cv_splits(y_train, folds, seed, g_train):
        held = np.isin(g_test, np.unique(g_train[te]))
        if not held.any():
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_train[tr], y_train[tr])
        pred[held] = clf.predict(X_test[held])
        done |= held
    assert done.all()
    return pred


def prompt_vs_text_table(pred, prompt_lab, text_lab, roles) -> dict:
    """Per (prompt role r, text role s): fraction of rows predicted as r (prompt wins), as s (text wins) and as
    anything else. On the diagonal r == s prompt and text coincide."""
    pred, prompt_lab, text_lab = map(np.asarray, (pred, prompt_lab, text_lab))
    p_prompt = np.full((len(roles), len(roles)), np.nan)
    p_text = np.full_like(p_prompt, np.nan)
    p_other = np.full_like(p_prompt, np.nan)
    for i, r in enumerate(roles):
        for j, s in enumerate(roles):
            m = (prompt_lab == r) & (text_lab == s)
            if not m.any():
                continue
            p_prompt[i, j] = float((pred[m] == r).mean())
            p_text[i, j] = float((pred[m] == s).mean())
            p_other[i, j] = float(((pred[m] != r) & (pred[m] != s)).mean())
    off = ~np.eye(len(roles), dtype=bool)
    return {"rows_prompt_role_cols_text_role": list(roles),
            "p_predict_prompt_role": p_prompt.tolist(), "p_predict_text_role": p_text.tolist(),
            "p_predict_other": p_other.tolist(),
            "offdiag_mean_p_prompt": float(np.nanmean(p_prompt[off])),
            "offdiag_mean_p_text": float(np.nanmean(p_text[off])),
            "offdiag_mean_p_other": float(np.nanmean(p_other[off])),
            "diag_mean_p_prompt": float(np.nanmean(np.diag(p_prompt)))}


# ---------------------------------------------------------------------------- inputs (same code path as E0-b)


def rebuild_e0b_inputs(n_items: int, input_seed: int) -> list[str]:
    """Verbatim copy of the input construction in run_e0b.py main() (not factored out there)."""
    from swarm_to_moe.data import PROFILE_SEED, load_gsm8k, load_mmlu_val, profiling_texts, seeded_subset

    prof = profiling_texts(seed=PROFILE_SEED)
    excluded = set(itertools.chain.from_iterable(prof.values()))
    n_half = n_items // 2
    gsm = [x["question"] for x in load_gsm8k("train") if x["question"] not in excluded]
    mmlu = [x["text"] for x in load_mmlu_val() if x["text"] not in excluded]
    inputs = seeded_subset(gsm, n_half, input_seed) + seeded_subset(mmlu, n_items - n_half, input_seed)
    assert not (set(inputs) & excluded)
    return inputs


# ---------------------------------------------------------------------------- prefill + routing


@torch.no_grad()
def collect(model, tok, rows, batch_size: int, padding: str, half_rng, part=None):
    """rows: list of dicts {prompt_ids, cont_ids, pref_role (partition role name or None)}.
    Returns per-row counts [N,L,E] int32 over continuation tokens, mean entropy [N,L], role preference [N,L]
    (NaN when pref_role is None), and per-row token-random-halves counts [N,2,L,E] (int32), plus stats."""
    from tqdm.auto import tqdm

    L, E = model.config.num_hidden_layers, model.config.num_local_experts
    N = len(rows)
    counts = np.zeros((N, L, E), np.int32)
    ent = np.zeros((N, L), np.float32)
    pref = np.full((N, L), np.nan, np.float64)
    halves = np.zeros((N, 2, L, E), np.int32)
    device = next(model.parameters()).device
    n_tok = n_cont = 0
    t0 = time.time()
    bar = tqdm(range(0, N, batch_size), desc="prefill")
    with MoERoutingHook(model, FreeRouting(), record=True, record_logits=True) as hook:
        for s in bar:
            rr = list(range(s, min(s + batch_size, N)))
            ids, attn, cmask, pos = build_batch([rows[i]["prompt_ids"] for i in rr],
                                                [rows[i]["cont_ids"] for i in rr], tok.pad_token_id, padding)
            hook.reset_records()
            model(input_ids=ids.to(device), attention_mask=attn.to(device), position_ids=pos.to(device),
                  use_cache=False)
            idx = hook.records.stack()  # [L,B,T,k]
            H = router_entropy(hook.records.stack_logits()).numpy()  # [L,B,T]; logits dropped right away
            hook.reset_records()
            cm = cmask.numpy()
            counts[rr] = usage_counts(idx, E, cmask)
            sel = idx.numpy()
            for b, i in enumerate(rr):
                if cm[b].any():
                    ent[i] = H[:, b, cm[b]].mean(-1)
                toks = np.nonzero(cm[b])[0]
                side = half_rng.integers(0, 2, size=len(toks))
                for h in (0, 1):
                    t = toks[side == h]
                    for l in range(L):
                        halves[i, h, l] = np.bincount(sel[l, b, t].ravel(), minlength=E)[:E]
            if part is not None:
                pr = [rows[i]["pref_role"] for i in rr]
                if any(p is not None for p in pr):
                    pv = role_preference(idx, part, pr, mask=cmask)  # NaN rows for None
                    pref[rr] = pv
            n_tok += int(attn.sum())
            n_cont += int(cmask.sum())
            dt = time.time() - t0
            bar.set_postfix(tok_s=f"{n_tok / max(dt, 1e-9):.0f}")
            del idx, H, sel
    dt = time.time() - t0
    stats = {"tokens": n_tok, "continuation_tokens": n_cont, "seconds": dt, "tok_per_s": n_tok / max(dt, 1e-9)}
    print(f"[prefill] {n_tok} tokens ({n_cont} continuation) in {dt:.1f}s = {stats['tok_per_s']:.0f} tok/s",
          flush=True)
    return counts, ent, pref, halves, stats


# ---------------------------------------------------------------------------- main


def mean_jsd(p, q):
    return float(routing_jsd(p, q).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--e0b", default="results/E0/e0b.json")
    ap.add_argument("--partition", default="results/E0/partition_all.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0, help="CV / shuffle / halves seed")
    ap.add_argument("--max-cont-tokens", type=int, default=128)
    ap.add_argument("--padding", choices=["right", "left"], default="right",
                    help="right (default): every row sits at cache positions 0.. exactly as if run alone, so "
                         "gpt-oss's sliding-window (128) attention never has pad slots inside its window. "
                         "left: as in generate()/E0-b, but the effective window of early continuation tokens "
                         "then shrinks by the row's pad count, which differs between conditions.")
    ap.add_argument("--skip-input-rebuild", action="store_true",
                    help="trust e0b.json['inputs'] without re-deriving them from the datasets")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N inputs only")
    ap.add_argument("--out", default="results/E0/e0c.json")
    args = ap.parse_args()
    t_start = time.time()
    out_path = Path(args.out)
    npz_path = out_path.with_name(out_path.stem + "_hist.npz")

    # ---------------------------------------------------------------- inputs + fixed texts from E0-b
    e0b = read_json(args.e0b)
    inputs = list(e0b["inputs"])
    if args.skip_input_rebuild:
        inputs_check = "skipped"
    else:
        rb = rebuild_e0b_inputs(e0b["run"]["args"]["n_items"], e0b["run"]["args"]["input_seed"])
        assert rb == inputs, "rebuilt E0-b inputs differ from e0b.json['inputs']"
        inputs_check = "rebuilt with run_e0b.py's code path and identical to e0b.json['inputs']"
    texts = e0b["texts"]
    assert set(ROLES + [NO_ROLE]) <= set(texts), f"e0b texts missing arms: {set(ROLES + [NO_ROLE]) - set(texts)}"
    n = len(inputs) if args.limit is None else min(args.limit, len(inputs))
    inputs = inputs[:n]
    assert all(len(texts[a]) >= n for a in ROLES + [NO_ROLE])
    sources = e0b.get("sources")

    part = Partition.load(args.partition)
    model, tok, prov = load_gpt_oss(args.model, revision=args.revision)
    L, E = model.config.num_hidden_layers, model.config.num_local_experts
    stops = _stop_ids(model, tok)
    role_obj = {r.name: r for r in DEFAULT_ROLES}
    role_to_part = {r: part.roles[i] for i, r in enumerate(ROLES)}

    # continuation ids: tokenized ONCE per (text arm, input) and reused verbatim in every condition
    cont = {}
    cont_info = {}
    for a in ROLES + [NO_ROLE]:
        cont[a], trunc, had_stop, raw_len = [], 0, 0, []
        for i in range(n):
            raw = tok(texts[a][i], add_special_tokens=False)["input_ids"]
            ids, t = prepare_continuation(raw, stops, args.max_cont_tokens)
            had_stop += int(any(x in stops for x in raw))
            trunc += int(t)
            raw_len.append(len(raw))
            cont[a].append(ids)
        lens = [len(c) for c in cont[a]]
        cont_info[a] = {"mean_tokens": float(np.mean(lens)), "min_tokens": int(min(lens)),
                        "max_tokens": int(max(lens)), "n_empty": int(sum(x == 0 for x in lens)),
                        "n_with_stop_token_dropped": had_stop, "n_truncated": trunc,
                        "mean_raw_tokens": float(np.mean(raw_len))}
    prompt_ids = {a: [tok(apply_chat(tok, build_messages(x, role_obj.get(a))), add_special_tokens=False)["input_ids"]
                      for x in inputs] for a in ROLES + [NO_ROLE]}

    conds = [(r, NO_ROLE) for r in ROLES] + [(NO_ROLE, NO_ROLE)] + [(r, s) for r in ROLES for s in ROLES]
    rows = []
    for (pa, ta) in conds:
        for i in range(n):
            rows.append({"prompt_ids": prompt_ids[pa][i], "cont_ids": cont[ta][i],
                         "pref_role": role_to_part.get(pa)})
    # identical continuation ids across conditions (same text arm, same input)
    for ta in ROLES + [NO_ROLE]:
        for i in range(n):
            ref = tuple(cont[ta][i])
            for ci, (pa, tb) in enumerate(conds):
                if tb == ta:
                    assert tuple(rows[ci * n + i]["cont_ids"]) == ref

    half_rng = np.random.default_rng(args.seed + 1000)
    counts, ent, pref, halves, stats = collect(model, tok, rows, args.batch_size, args.padding, half_rng, part)
    C = len(conds)
    counts = counts.reshape(C, n, L, E)
    ent = ent.reshape(C, n, L)
    pref = pref.reshape(C, n, L)
    halves_c = halves.reshape(C, n, 2, L, E).sum(1, dtype=np.int64)  # [C,2,L,E]
    ci = {c: k for k, c in enumerate(conds)}

    def hist(pa, ta):
        return l1_per_layer(counts[ci[(pa, ta)]])

    # ---------------------------------------------------------------- classifiers
    y = np.repeat(np.array(ROLES), n)
    groups = np.tile(np.arange(n), len(ROLES))

    def cv(X, yy=y, gg=groups):
        acc, chance = role_id_cv(X.reshape(len(X), -1), yy, folds=5, seed=args.seed, groups=gg)
        return {"accuracy": acc, "chance": chance}

    def shuffled(yy, gg):
        rng = np.random.default_rng(args.seed)
        ys = yy.copy()
        for g in np.unique(gg):  # within-input shuffle, as in run_e0b.py
            m = gg == g
            ys[m] = rng.permutation(yy[m])
        return ys

    def suite(pairs):
        Hh = np.concatenate([hist(pa, ta) for pa, ta in pairs])  # [4n,L,E]
        He = np.concatenate([ent[ci[(pa, ta)]] for pa, ta in pairs])  # [4n,L]
        d = {"all_layers": cv(Hh)}
        for g, ls in LAYER_GROUPS.items():
            d[f"layers_{g}"] = cv(Hh[:, list(ls)])
        d["entropy_only"] = cv(He)
        d["hist_plus_entropy"] = cv(np.concatenate([Hh.reshape(len(y), -1), He], axis=1))
        d["shuffled_labels_control"] = cv(Hh, yy=shuffled(y, groups))
        return d, Hh

    fixed_pairs = [(r, NO_ROLE) for r in ROLES]
    own_pairs = [(r, r) for r in ROLES]
    rid_fixed, X_fixed = suite(fixed_pairs)
    rid_own, X_own = suite(own_pairs)

    # cross: classifier trained on fixed text (and, secondary, on own text), applied to all 16 (r, s)
    cross_pairs = [(r, s) for r in ROLES for s in ROLES]
    X_cross = np.concatenate([hist(r, s) for r, s in cross_pairs])
    p_lab = np.repeat([r for r, _ in cross_pairs], n)
    t_lab = np.repeat([s for _, s in cross_pairs], n)
    g_cross = np.tile(np.arange(n), len(cross_pairs))
    pred_fx = cross_predict_grouped(X_fixed, y, groups, X_cross, g_cross, 5, args.seed)
    pred_own = cross_predict_grouped(X_own, y, groups, X_cross, g_cross, 5, args.seed)
    cross = {
        "classifier_trained_on_fixed_text": prompt_vs_text_table(pred_fx, p_lab, t_lab, ROLES),
        "classifier_trained_on_own_text": prompt_vs_text_table(pred_own, p_lab, t_lab, ROLES),
        "note": "grouped: each row predicted by the fold model that never saw its input. Rows = prompt role, "
                "cols = text role. Off-diagonal p_prompt high -> routing follows the system prompt; p_text high "
                "-> routing follows the content (deference proxy, PREREG).",
    }

    # no_role vs role (binary), fixed text
    nr_vs = {}
    nr = hist(NO_ROLE, NO_ROLE)
    for r in ROLES:
        X = np.concatenate([nr, hist(r, NO_ROLE)])
        yy = np.array([NO_ROLE] * n + [r] * n)
        nr_vs[r] = cv(X, yy=yy, gg=np.tile(np.arange(n), 2))

    # ---------------------------------------------------------------- JSD
    perm = np.random.default_rng(args.seed + 1).permutation(n)
    h1, h2 = perm[: n // 2], perm[n // 2:]

    def jsd_block(pairs, extra_nr=None):
        agg = {pa: counts[ci[(pa, ta)]].sum(0).astype(np.float64) for pa, ta in pairs}
        names = [pa for pa, _ in pairs]
        between = {f"{a}|{b}": mean_jsd(agg[a], agg[b]) for a, b in itertools.combinations(names, 2)}
        bm = float(np.mean(list(between.values())))
        tok_h = {pa: mean_jsd(halves_c[ci[(pa, ta)], 0], halves_c[ci[(pa, ta)], 1]) for pa, ta in pairs}
        inp_h = {pa: mean_jsd(counts[ci[(pa, ta)]][h1].sum(0), counts[ci[(pa, ta)]][h2].sum(0)) for pa, ta in pairs}
        f_tok = float(np.mean([tok_h[r] for r in names]))
        f_inp = float(np.mean([inp_h[r] for r in names]))
        d = {"between_role_pairs": between, "between_role_mean": bm,
             "floor_token_random_halves_per_arm": tok_h, "floor_token_random_halves_mean": f_tok,
             "floor_input_halves_per_arm": inp_h, "floor_input_halves_mean": f_inp,
             "floor_used": f_inp,
             "floor_definition": "mean over roles of JSD between disjoint random input halves (same condition)",
             "ratio_between_over_floor": bm / f_inp if f_inp > 0 else None,
             "ratio_between_over_token_halves_floor": bm / f_tok if f_tok > 0 else None,
             "per_layer_between_mean": np.mean([routing_jsd(agg[a], agg[b]) for a, b in
                                                itertools.combinations(names, 2)], axis=0).tolist()}
        if extra_nr is not None:
            a_nr = counts[ci[extra_nr]].sum(0).astype(np.float64)
            d["no_role_vs_role"] = {r: mean_jsd(a_nr, agg[r]) for r in names}
        return d

    jsd = {"fixed_text": jsd_block(fixed_pairs, extra_nr=(NO_ROLE, NO_ROLE)),
           "own_text": jsd_block(own_pairs)}

    # ---------------------------------------------------------------- role preference vs partition (record only)
    pref_out = {}
    for name, pairs in (("fixed_text", fixed_pairs), ("own_text", own_pairs)):
        pref_out[name] = {}
        for pa, ta in pairs:
            v = pref[ci[(pa, ta)]]
            pref_out[name][pa] = {"partition_role": role_to_part[pa], "mean": float(np.nanmean(v)),
                                  "per_layer": np.nanmean(v, axis=0).tolist()}

    acc_fx = rid_fixed["all_layers"]["accuracy"]
    acc_own = rid_own["all_layers"]["accuracy"]
    res = {
        "provenance": prov, "run": run_meta(args), "partition_hash": part.hash(),
        "e0b_source": {"path": args.e0b, "partition_hash": e0b.get("partition_hash"),
                       "provenance_commit": e0b.get("provenance", {}).get("commit"),
                       "role_id_generated_all_layers": e0b.get("role_id", {}).get("generated", {}).get("all_layers"),
                       "text_tfidf_baseline": e0b.get("text_tfidf_baseline")},
        "n_inputs": n, "sources": sources, "inputs_check": inputs_check,
        "roles": ROLES, "conditions": [f"{pa}|{ta}" for pa, ta in conds],
        "condition_format": "prompt_arm|continuation_text_arm",
        "continuation": {
            "source": "e0b.json texts: raw decoded generations with special tokens kept (Harmony: analysis "
                      "channel, <|end|><|start|>assistant, final channel); tokenized once per (text arm, input) "
                      "with add_special_tokens=False, cut before the first stop token, truncated to "
                      "--max-cont-tokens, and appended to apply_chat(build_messages(input, role)) token ids",
            "per_text_arm": cont_info, "padding": args.padding,
        },
        "folds": "StratifiedGroupKFold(5), groups = input index",
        "prefill_stats": stats,
        "role_id": {"fixed_text": rid_fixed, "own_text": rid_own},
        "cross_prompt_vs_text": cross,
        "no_role_vs_role_fixed_text": nr_vs,
        "jsd": jsd,
        "role_preference_vs_partition": pref_out,
        "summary": {"role_id_fixed_text_all_layers": acc_fx, "role_id_own_text_all_layers": acc_own,
                    "chance": 0.25,
                    "jsd_fixed_ratio_input_halves_floor": jsd["fixed_text"]["ratio_between_over_floor"]},
    }
    res["seconds"] = time.time() - t_start

    np.savez_compressed(
        npz_path,
        conditions=np.array([f"{pa}|{ta}" for pa, ta in conds]),
        counts=counts,  # [C,n,L,E] continuation-token usage counts
        entropy=ent,  # [C,n,L]
        role_preference=pref,  # [C,n,L]
        token_halves=halves_c,  # [C,2,L,E]
        cont_len=np.array([[len(cont[ta][i]) for i in range(n)] for _, ta in conds]),
    )
    print(f"wrote {npz_path}")
    write_json(out_path, res)

    # ---------------------------------------------------------------- compact summary
    tab = np.array(cross["classifier_trained_on_fixed_text"]["p_predict_prompt_role"])
    print(f"\nE0-c  n={n}  chance 0.25   (E0-b generated: {res['e0b_source']['role_id_generated_all_layers']}, "
          f"text TF-IDF: {res['e0b_source']['text_tfidf_baseline']})")
    print(f"role-ID fixed no_role text, all layers : {acc_fx:.3f}   "
          f"(entropy-only {rid_fixed['entropy_only']['accuracy']:.3f}, shuffled "
          f"{rid_fixed['shuffled_labels_control']['accuracy']:.3f})")
    print("   by layer group: " + ", ".join(f"{g} {rid_fixed[f'layers_{g}']['accuracy']:.3f}" for g in LAYER_GROUPS))
    print(f"role-ID own text (prefill), all layers : {acc_own:.3f}")
    print("P(fixed-text classifier predicts PROMPT role); rows = prompt, cols = text")
    print(" " * 12 + "".join(f"{s:>12}" for s in ROLES))
    for i, r in enumerate(ROLES):
        print(f"{r:>12}" + "".join(f"{tab[i, j]:12.3f}" for j in range(len(ROLES))))
    c = cross["classifier_trained_on_fixed_text"]
    print(f"off-diagonal mean: P(prompt) {c['offdiag_mean_p_prompt']:.3f}  P(text) {c['offdiag_mean_p_text']:.3f}  "
          f"P(other) {c['offdiag_mean_p_other']:.3f}")
    jf = jsd["fixed_text"]
    print(f"JSD fixed text: between {jf['between_role_mean']:.5f} / input-halves floor "
          f"{jf['floor_input_halves_mean']:.5f} = {jf['ratio_between_over_floor']} "
          f"(token-halves floor {jf['floor_token_random_halves_mean']:.6f}, "
          f"ratio {jf['ratio_between_over_token_halves_floor']})")


if __name__ == "__main__":
    main()
