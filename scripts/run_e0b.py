#!/usr/bin/env python3
"""E0-b: is the role prompt recoverable from FREE routing on the base model? (see PREREG.md)

Inputs: seeded 100 GSM8K-train + 100 MMLU-val questions, disjoint from the profiling texts. Arms: 4 DEFAULT_ROLES
+ no-role. Router free, greedy, 128 new tokens. Features per (arm, input): per-layer L1-normalised expert-usage
histograms (24x32) over generated tokens and, separately, prompt tokens; per-layer mean router entropy (24).
Classifier folds are grouped by input (the same question is never in train and test).
Writes results/E0/e0b.json and results/E0/e0b_hist.npz.
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from swarm_to_moe.data import (  # noqa: E402
    PROFILE_SEED,
    _SPECIAL,
    load_gsm8k,
    load_mmlu_val,
    profiling_texts,
    seeded_subset,
)
from swarm_to_moe.hook import FreeRouting, MoERoutingHook  # noqa: E402
from swarm_to_moe.metrics import (  # noqa: E402
    l1_per_layer,
    role_id_cv,
    role_preference,
    routing_jsd,
    text_id_cv,
    usage_counts,
)
from swarm_to_moe.model import DEFAULT_MODEL, apply_chat, generate_batched, load_gpt_oss  # noqa: E402
from swarm_to_moe.partition import Partition  # noqa: E402
from swarm_to_moe.prompts import DEFAULT_ROLES, build_messages  # noqa: E402
from swarm_to_moe.runutil import run_meta, write_json  # noqa: E402

LAYER_GROUPS = {"0-7": range(0, 8), "8-15": range(8, 16), "16-23": range(16, 24), "10-14": range(10, 15)}


def run_arm(model, tok, prompts, args, batch_size, half_rng=None):
    """Generate for one arm; returns per-row gen/prompt counts [N,L,E], entropy means [N,L], texts, and
    (optionally) token-level random-half counts [2,L,E] for the within-agent noise floor."""
    L, E = model.config.num_hidden_layers, model.config.num_local_experts
    N = len(prompts)
    gen_c = np.zeros((N, L, E), np.int32)
    prm_c = np.zeros((N, L, E), np.int32)
    gen_H = np.zeros((N, L), np.float32)
    prm_H = np.zeros((N, L), np.float32)
    halves = np.zeros((2, L, E), np.int64)
    pref_rows = []
    with MoERoutingHook(model, FreeRouting(), record_logits=True) as hook:
        texts, routing = generate_batched(model, tok, prompts, args.max_new_tokens, batch_size, hook=hook,
                                          entropy=True)
    for rec in routing:
        rows = rec["rows"]
        idx = rec["indices"].long()
        gm, pm = rec["gen_mask"], rec["prompt_mask"]
        gen_c[rows] = usage_counts(idx, E, gm)
        prm_c[rows] = usage_counts(idx, E, pm)
        H = rec["entropy"].numpy()  # [L,B,T]
        gmn, pmn = gm.numpy(), pm.numpy()
        for b, r in enumerate(rows):
            if gmn[b].any():
                gen_H[r] = H[:, b, gmn[b]].mean(-1)
            if pmn[b].any():
                prm_H[r] = H[:, b, pmn[b]].mean(-1)
        if half_rng is not None:
            sel = idx.numpy()  # [L,B,T,k]
            for b in range(len(rows)):
                toks = np.nonzero(gmn[b])[0]
                side = half_rng.integers(0, 2, size=len(toks))
                for h in (0, 1):
                    t = toks[side == h]
                    for l in range(L):
                        halves[h, l] += np.bincount(sel[l, b, t].ravel(), minlength=E)[:E]
        pref_rows.append((rows, rec["indices"], gm))  # int16 to save RAM
    return {"gen": gen_c, "prompt": prm_c, "gen_H": gen_H, "prompt_H": prm_H, "texts": texts,
            "halves": halves, "_raw": pref_rows, "stats": dict(generate_batched.last_stats)}


def mean_jsd(p, q):
    return float(routing_jsd(p, q).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--n-items", type=int, default=200, help="half GSM8K train, half MMLU val")
    ap.add_argument("--seed", type=int, default=0, help="CV / shuffle seed")
    ap.add_argument("--input-seed", type=int, default=1, help="input subset seed (profiling uses PROFILE_SEED)")
    ap.add_argument("--partition", default="results/E0/partition_all.json")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--floor-batch-size", type=int, default=32)
    ap.add_argument("--out-dir", default="results/E0")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---------------------------------------------------------------- inputs, disjoint from profiling
    prof = profiling_texts(seed=PROFILE_SEED)
    excluded = set(itertools.chain.from_iterable(prof.values()))
    n_half = args.n_items // 2
    gsm = [x["question"] for x in load_gsm8k("train") if x["question"] not in excluded]
    mmlu = [x["text"] for x in load_mmlu_val() if x["text"] not in excluded]
    inputs = seeded_subset(gsm, n_half, args.input_seed) + seeded_subset(mmlu, args.n_items - n_half, args.input_seed)
    sources = ["gsm8k_train"] * n_half + ["mmlu_val"] * (args.n_items - n_half)
    assert not (set(inputs) & excluded)

    part = Partition.load(args.partition)
    model, tok, prov = load_gpt_oss(args.model, revision=args.revision)
    arms = [r.name for r in DEFAULT_ROLES] + ["no_role"]
    role_obj = {r.name: r for r in DEFAULT_ROLES}
    prompts = {a: [apply_chat(tok, build_messages(x, role_obj.get(a))) for x in inputs] for a in arms}

    out = {}
    half_rng = np.random.default_rng(args.seed + 1000)
    for a in arms:
        print(f"=== arm {a} ===", flush=True)
        out[a] = run_arm(model, tok, prompts[a], args, args.batch_size, half_rng=half_rng)
    floor_role = arms[0]
    print(f"=== batch-shape floor: {floor_role} at batch {args.floor_batch_size} ===", flush=True)
    floor_run = run_arm(model, tok, prompts[floor_role], args, args.floor_batch_size)

    # ---------------------------------------------------------------- features
    roles = [r.name for r in DEFAULT_ROLES]
    n = len(inputs)
    L, E = model.config.num_hidden_layers, model.config.num_local_experts
    y = np.repeat(np.array(roles), n)
    groups = np.tile(np.arange(n), len(roles))
    gen_hist = np.concatenate([l1_per_layer(out[a]["gen"]) for a in roles])  # [4n,L,E]
    prm_hist = np.concatenate([l1_per_layer(out[a]["prompt"]) for a in roles])
    gen_H = np.concatenate([out[a]["gen_H"] for a in roles])  # [4n,L]

    def cv(X, yy=y, gg=groups):
        acc, chance = role_id_cv(X.reshape(len(X), -1), yy, folds=5, seed=args.seed, groups=gg)
        return {"accuracy": acc, "chance": chance}

    res = {"provenance": prov, "run": run_meta(args), "partition_hash": part.hash(),
           "n_inputs": n, "sources": {s: sources.count(s) for s in set(sources)},
           "arms": arms, "folds": "StratifiedGroupKFold(5), groups = input index",
           "gen_stats": {a: out[a]["stats"] for a in arms}}
    rid = {}
    for name, H in (("generated", gen_hist), ("prompt", prm_hist)):
        rid[name] = {"all_layers": cv(H)}
        for g, ls in LAYER_GROUPS.items():
            rid[name][f"layers_{g}"] = cv(H[:, list(ls)])
    rid["generated"]["entropy_only"] = cv(gen_H)
    rid["generated"]["hist_plus_entropy"] = cv(np.concatenate([gen_hist.reshape(len(y), -1), gen_H], axis=1))
    rng = np.random.default_rng(args.seed)
    y_shuf = y.copy()
    for g in range(n):  # shuffle labels within each input (keeps grouping and class balance)
        m = groups == g
        y_shuf[m] = rng.permutation(y[m])
    rid["generated"]["shuffled_labels_control"] = cv(gen_hist, yy=y_shuf)
    res["role_id"] = rid

    # no-role vs each role (binary)
    nr = l1_per_layer(out["no_role"]["gen"])
    res["no_role_vs_role"] = {}
    for a in roles:
        X = np.concatenate([nr, l1_per_layer(out[a]["gen"])])
        yy = np.array(["no_role"] * n + [a] * n)
        res["no_role_vs_role"][a] = cv(X, yy=yy, gg=np.tile(np.arange(n), 2))

    # text baseline (same grouped folds), on generated text with special tokens removed
    texts = [_SPECIAL.sub(" ", t) for a in roles for t in out[a]["texts"]]
    acc_t, ch_t = text_id_cv(texts, y, folds=5, seed=args.seed, groups=groups)
    res["text_tfidf_baseline"] = {"accuracy": acc_t, "chance": ch_t}

    # ---------------------------------------------------------------- JSD: between-role vs noise floors
    agg = {a: out[a]["gen"].sum(0).astype(np.float64) for a in arms}
    between = {f"{a}|{b}": mean_jsd(agg[a], agg[b]) for a, b in itertools.combinations(roles, 2)}
    between_mean = float(np.mean(list(between.values())))
    token_halves = {a: mean_jsd(out[a]["halves"][0], out[a]["halves"][1]) for a in arms}
    perm = np.random.default_rng(args.seed + 1).permutation(n)
    h1, h2 = perm[: n // 2], perm[n // 2:]
    input_halves = {a: mean_jsd(out[a]["gen"][h1].sum(0), out[a]["gen"][h2].sum(0)) for a in arms}
    batch_shape = mean_jsd(agg[floor_role], floor_run["gen"].sum(0))
    floor_token = float(np.mean([token_halves[a] for a in roles]))
    floor = max(floor_token, batch_shape)
    res["jsd"] = {
        "between_role_pairs": between,
        "between_role_mean": between_mean,
        "no_role_vs_role": {a: mean_jsd(agg["no_role"], agg[a]) for a in roles},
        "floor_token_random_halves_per_arm": token_halves,
        "floor_token_random_halves_mean_roles": floor_token,
        "floor_input_halves_per_arm": input_halves,
        "floor_batch_shape": {"role": floor_role, "batch_sizes": [args.batch_size, args.floor_batch_size],
                              "jsd": batch_shape},
        "floor_used": floor,
        "floor_definition": "max(mean token-random-halves JSD over roles, batch-shape JSD)",
        "ratio_between_over_floor": between_mean / floor if floor > 0 else None,
        "per_layer_between_mean": np.mean([routing_jsd(agg[a], agg[b]) for a, b in
                                           itertools.combinations(roles, 2)], axis=0).tolist(),
    }

    # ---------------------------------------------------------------- role preference vs the partition (record only)
    role_to_part = {r: part.roles[i] for i, r in enumerate(roles)}
    pref = {}
    for a in roles:
        vals = []
        for rows, idx, gm in out[a]["_raw"]:
            vals.append(role_preference(idx, part, [role_to_part[a]] * len(rows), mask=gm))
        pref[a] = {"partition_role": role_to_part[a], "mean": float(np.nanmean(np.concatenate(vals))),
                   "per_layer": np.nanmean(np.concatenate(vals), axis=0).tolist()}
    res["role_preference_vs_partition"] = pref

    acc_gen = rid["generated"]["all_layers"]["accuracy"]
    res["prereg"] = {
        "role_id_ge_40": acc_gen >= 0.40,
        "between_jsd_ge_3x_floor": between_mean >= 3 * floor,
        "pass": bool(acc_gen >= 0.40 and between_mean >= 3 * floor),
        "shuffled_control_near_chance": abs(rid["generated"]["shuffled_labels_control"]["accuracy"] - 0.25) < 0.1,
    }
    res["seconds"] = time.time() - t_start
    res["texts"] = {a: out[a]["texts"] for a in arms}
    res["inputs"] = inputs

    np.savez_compressed(
        out_dir / "e0b_hist.npz",
        arms=np.array(arms),
        gen_counts=np.stack([out[a]["gen"] for a in arms]),  # [5,n,L,E]
        prompt_counts=np.stack([out[a]["prompt"] for a in arms]),
        gen_entropy=np.stack([out[a]["gen_H"] for a in arms]),  # [5,n,L]
        prompt_entropy=np.stack([out[a]["prompt_H"] for a in arms]),
        floor_gen_counts=floor_run["gen"],
        token_halves=np.stack([out[a]["halves"] for a in arms]),
    )
    write_json(out_dir / "e0b.json", res)
    print(f"role-ID (generated, all layers): {acc_gen:.3f} (chance 0.25); text TF-IDF: {acc_t:.3f}; "
          f"between/floor JSD: {res['jsd']['ratio_between_over_floor']}")


if __name__ == "__main__":
    main()
