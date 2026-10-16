#!/usr/bin/env python3
"""E2 step 2: does the trained role diversity survive when the router is freed? (PREREG condition table)

Conditions (held-out inputs only; each run = one input-order seed x one batch size):
  B-free-role      base model, FreeRouting, matching role system prompt
  M-free-role      trained LoRA (attached, = merged up to float rounding), FreeRouting, role prompt
  M-free-shuffled  M-free-role routing re-scored against a derangement of the row labels
                   (prompts.shuffled_assignment). NB: if M-free-role has preference x, relabelling to another
                   role gives about -x/(R-1), not 0; compare it with B-free-role re-scored the same way.
  M-free-none      trained LoRA, FreeRouting, no system prompt (one row per input)
  M-rule           trained LoRA, HardSplit with the row's agent, no system prompt (the swarm as trained)
  P-free-role      placebo LoRA (random noise with the trained per-expert delta Frobenius norm), FreeRouting,
                   role prompt
Sequencing (one model load): base conditions first, then attach the trained LoRA (not merged: mathematically
identical, saves a reload) for M-*, then remove it and attach the placebo for P-*.

Metrics (swarm_to_moe.metrics): role_preference per layer group with paired bootstrap CIs over inputs,
role_id_cv on per-row L1-normalised generated-token routing histograms (grouped folds, groups = input), TF-IDF
text-ID on the same folds, GSM8K-test retention vs base for M-rule (each agent) and M-free-none.
The "deference proxy" (PREREG metric 4, exploratory) is not implemented here.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from swarm_to_moe.data import (  # noqa: E402
    _SPECIAL,
    GSM8K_SUFFIX,
    load_gsm8k,
    score_gsm8k_harmony,
    seeded_subset,
)
from swarm_to_moe.hook import FreeRouting, HardSplit, MoERoutingHook  # noqa: E402
from swarm_to_moe.lora import load_expert_lora, load_expert_lora_checkpoint, attach_expert_lora  # noqa: E402
from swarm_to_moe.metrics import (  # noqa: E402
    l1_per_layer,
    mcnemar_holm,
    paired_bootstrap_ci,
    role_id_cv,
    role_preference,
    text_id_cv,
    usage_counts,
)
from swarm_to_moe.model import DEFAULT_MODEL, apply_chat, generate_batched, load_gpt_oss  # noqa: E402
from swarm_to_moe.partition import Partition  # noqa: E402
from swarm_to_moe.prompts import DEFAULT_ROLES, build_messages, shuffled_assignment  # noqa: E402
from swarm_to_moe.runutil import read_json, run_meta, write_json  # noqa: E402

LAYER_GROUPS = {"0-11": range(0, 12), "12-23": range(12, 24), "10-14": range(10, 15)}
ROLE_PROMPTED = ("B-free-role", "M-free-role", "M-free-shuffled", "P-free-role")


# ----------------------------------------------------------------------------- generation


def run_condition(model, tok, part, prompts, rule, row_agents, pref_agents, shuf_agents, max_new, bs):
    """Generate with the routing ``rule``; returns texts, per-row generated-token counts [N,L,E], per-row role
    preference [N,L] against ``pref_agents`` (and ``shuf_agents``), and generation stats."""
    L, E = part.num_layers, part.num_experts
    N = len(prompts)
    counts = np.zeros((N, L, E), np.int32)
    pref = np.full((N, L), np.nan) if pref_agents is not None else None
    pref_s = np.full((N, L), np.nan) if shuf_agents is not None else None
    with MoERoutingHook(model, rule=rule) as hook:
        texts, routing = generate_batched(model, tok, prompts, max_new, bs, hook=hook,
                                          row_agents=row_agents if rule.needs_agents else None)
    for rec in routing:
        rows = rec["rows"]
        idx = rec["indices"].long()
        gm = rec["gen_mask"]
        counts[rows] = usage_counts(idx, E, gm)
        if pref_agents is not None:
            pref[rows] = role_preference(idx, part, [pref_agents[i] for i in rows], mask=gm)
        if shuf_agents is not None:
            pref_s[rows] = role_preference(idx, part, [shuf_agents[i] for i in rows], mask=gm)
    return {"texts": texts, "counts": counts, "pref": pref, "pref_shuf": pref_s,
            "stats": dict(generate_batched.last_stats)}


def group_pref_per_input(pref, input_idx, n_inputs):
    """[N,L] per-row preference -> {group: [n_inputs]} (nanmean over the group's layers, then over the rows of
    each input)."""
    out = {}
    for g, ls in LAYER_GROUPS.items():
        per_row = np.nanmean(pref[:, list(ls)], axis=1)
        v = np.full(n_inputs, np.nan)
        for i in range(n_inputs):
            m = (input_idx == i) & np.isfinite(per_row)
            if m.any():
                v[i] = per_row[m].mean()
        out[g] = v
    return out


def ci_vs(a, b, seed):
    m = np.isfinite(a) & np.isfinite(b)
    mean, lo, hi = paired_bootstrap_ci(a[m], b[m], seed=seed)
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": int(m.sum()), "excludes_0": bool(lo > 0 or hi < 0)}


def owner_split(counts, part):
    """Mean fraction of generated-token top-k slots on each owner (shared / each role's private)."""
    om = part.owner_matrix()  # [L,E]
    tot = counts.sum(-1).astype(np.float64)  # [N,L]
    res = {}
    for code, name in [(0, "shared")] + [(i + 1, r) for i, r in enumerate(part.roles)]:
        frac = (counts * (om == code)[None]).sum(-1) / np.maximum(tot, 1)
        res[name] = float(frac[tot > 0].mean()) if (tot > 0).any() else float("nan")
    return res


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--lora", default="results/E2/lora_rank16.pt")
    ap.add_argument("--partition", default="results/E0/partition_all.json")
    ap.add_argument("--inputs", default="results/E2/inputs.json")
    ap.add_argument("--n-items", type=int, default=300)
    ap.add_argument("--seeds", default="0,1,2", help="input-order seeds (with n-items = pool size they only "
                                                      "change order, i.e. batch composition)")
    ap.add_argument("--batch-sizes", default="32,64")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--placebo-seed", type=int, default=0)
    ap.add_argument("--retention-items", type=int, default=300)
    ap.add_argument("--retention-seed", type=int, default=0)
    ap.add_argument("--retention-max-new-tokens", type=int, default=384)
    ap.add_argument("--skip-retention", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/E2/measure.json")
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    bss = [int(b) for b in args.batch_sizes.split(",")]
    t_start = time.time()

    part = Partition.load(args.partition)
    part.validate()
    ck = load_expert_lora_checkpoint(args.lora)
    if ck["partition_hash"] != part.hash():
        raise ValueError(f"LoRA partition hash {ck['partition_hash']} != {part.hash()}")
    extra = ck.get("extra") or {}
    role_names = [r.name for r in DEFAULT_ROLES]
    role_map = extra.get("role_map") or {n: part.roles[i] for i, n in enumerate(role_names)}
    role_obj = {r.name: r for r in DEFAULT_ROLES}
    del ck

    pool = [it for it in read_json(args.inputs)["items"] if it["heldout"]]
    pool.sort(key=lambda it: it["input_id"])
    if args.n_items > len(pool):
        raise ValueError(f"--n-items {args.n_items} > {len(pool)} held-out inputs")
    ckpt_heldout = set(extra.get("heldout_input_ids") or [])
    if ckpt_heldout and not {it["input_id"] for it in pool} <= ckpt_heldout:
        raise ValueError("held-out inputs in --inputs do not match the LoRA checkpoint's held-out ids")

    model, tok, prov = load_gpt_oss(args.model, revision=args.revision)
    common = {"partition_hash": part.hash(), "lora_path": str(args.lora), "role_map": role_map,
              "max_new_tokens": args.max_new_tokens}

    def make_run(seed):
        order = np.random.default_rng(seed).permutation(len(pool))[: args.n_items]
        items = [pool[i] for i in order]
        n = len(items)
        # role-prompted rows: input-major, all four roles of an input adjacent -> every batch mixes roles
        rp_prompts, rp_input, rp_role = [], [], []
        for i, it in enumerate(items):
            for name in role_names:
                rp_prompts.append(apply_chat(tok, build_messages(it["input_text"], role_obj[name])))
                rp_input.append(i)
                rp_role.append(name)
        none_prompts = [apply_chat(tok, build_messages(it["input_text"], None)) for it in items]
        rule_prompts = [none_prompts[i] for i in rp_input]
        agents = [role_map[r] for r in rp_role]
        shuf = shuffled_assignment(agents, seed)
        return {"items": items, "n": n, "rp_prompts": rp_prompts, "rp_input": np.array(rp_input),
                "rp_role": rp_role, "agents": agents, "shuf": shuf, "none_prompts": none_prompts,
                "rule_prompts": rule_prompts}

    runs = {s: make_run(s) for s in seeds}
    raw = {}  # (cond, seed, bs) -> result

    def gen(cond, seed, bs, prompts, rule, row_agents, pref_agents, shuf_agents=None):
        print(f"=== {cond} seed={seed} bs={bs} ({len(prompts)} rows) ===", flush=True)
        t0 = time.time()
        r = run_condition(model, tok, part, prompts, rule, row_agents, pref_agents, shuf_agents,
                          args.max_new_tokens, bs)
        r["seconds"] = time.time() - t0
        raw[(cond, seed, bs)] = r
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ retention items
    ret_items = seeded_subset(load_gsm8k("test"), args.retention_items, args.retention_seed)
    ret_prompts = [apply_chat(tok, build_messages(x["question"] + GSM8K_SUFFIX, None)) for x in ret_items]
    ret_gold = [x["answer"] for x in ret_items]
    retention_raw = {}


    # ------------------------------------------------------------------ phase B: base model
    for s in seeds:
        R = runs[s]
        for bs in bss:
            gen("B-free-role", s, bs, R["rp_prompts"], FreeRouting(), None, R["agents"], R["shuf"])
    if not args.skip_retention:
        texts, _ = generate_batched(model, tok, ret_prompts, args.retention_max_new_tokens, bss[-1])
        retention_raw["base"] = texts

    # ------------------------------------------------------------------ phase M: trained LoRA attached
    trained = load_expert_lora(args.lora, model, part)
    trained_norms = trained.delta_norms()
    for s in seeds:
        R = runs[s]
        for bs in bss:
            gen("M-free-role", s, bs, R["rp_prompts"], FreeRouting(), None, R["agents"], R["shuf"])
            gen("M-free-none", s, bs, R["none_prompts"], FreeRouting(), None, None)
            gen("M-rule", s, bs, R["rule_prompts"], HardSplit(part), R["agents"], R["agents"])
    if not args.skip_retention:
        texts, _ = generate_batched(model, tok, ret_prompts, args.retention_max_new_tokens, bss[-1])
        retention_raw["M-free-none"] = texts
        rule_prompts = [p for p in ret_prompts for _ in part.roles]
        rule_agents = [a for _ in ret_prompts for a in part.roles]
        # generate_batched stacks the hook's records per batch, so recording must stay on
        with MoERoutingHook(model, rule=HardSplit(part)) as hook:
            texts, _ = generate_batched(model, tok, rule_prompts, args.retention_max_new_tokens, bss[-1],
                                        hook=hook, row_agents=rule_agents)
        for a in part.roles:
            retention_raw[f"M-rule:{a}"] = [t for t, ag in zip(texts, rule_agents) if ag == a]
    trained.remove()

    # ------------------------------------------------------------------ phase P: placebo
    placebo = attach_expert_lora(model, part, trained.roles, trained.cfg)
    placebo.random_init_(args.placebo_seed, match=trained_norms)
    pn = placebo.delta_norms()
    max_rel_err = max(abs(pn[k] - v) / max(v, 1e-12) for k, v in trained_norms.items())
    for s in seeds:
        R = runs[s]
        for bs in bss:
            gen("P-free-role", s, bs, R["rp_prompts"], FreeRouting(), None, R["agents"], R["shuf"])
    placebo.remove()

    # ------------------------------------------------------------------ metrics
    records = []
    texts_dump = {}
    for s in seeds:
        R = runs[s]
        n = R["n"]
        y = np.array(R["rp_role"])
        groups = R["rp_input"]
        for bs in bss:
            base_meta = {**common, "seed": s, "batch_size": bs, "n_inputs": n}
            per_input = {}
            for cond in ("B-free-role", "M-free-role", "P-free-role", "M-rule"):
                r = raw[(cond, s, bs)]
                per_input[cond] = group_pref_per_input(r["pref"], groups, n)
            per_input["M-free-shuffled"] = group_pref_per_input(raw[("M-free-role", s, bs)]["pref_shuf"], groups, n)
            per_input["B-free-shuffled"] = group_pref_per_input(raw[("B-free-role", s, bs)]["pref_shuf"], groups, n)
            zeros = np.zeros(n)
            for cond in ("B-free-role", "M-free-role", "M-free-shuffled", "B-free-shuffled", "P-free-role", "M-rule"):
                rec = {**base_meta, "condition": cond, "metric": "role_preference"}
                for g in LAYER_GROUPS:
                    rec[g] = ci_vs(per_input[cond][g], zeros, seed=s)
                records.append(rec)
            for a, b in (("M-free-role", "B-free-role"), ("P-free-role", "B-free-role"),
                         ("M-free-role", "P-free-role"), ("M-free-shuffled", "B-free-shuffled")):
                rec = {**base_meta, "condition": f"{a} minus {b}", "metric": "role_preference_diff"}
                for g in LAYER_GROUPS:
                    rec[g] = ci_vs(per_input[a][g], per_input[b][g], seed=s)
                records.append(rec)

            # role-ID from routing and from text, same grouped folds
            for cond in ("B-free-role", "M-free-role", "P-free-role", "M-rule"):
                r = raw[(cond, s, bs)]
                X = l1_per_layer(r["counts"]).reshape(len(y), -1)
                acc, chance = role_id_cv(X, y, folds=5, seed=s, groups=groups)
                txt = [_SPECIAL.sub(" ", t) for t in r["texts"]]
                tacc, tchance = text_id_cv(txt, y, folds=5, seed=s, groups=groups)
                records.append({**base_meta, "condition": cond, "metric": "role_id",
                                "routing_acc": acc, "text_acc": tacc, "chance": chance,
                                "owner_fractions": owner_split(r["counts"], part),
                                "gen_tok_per_s": r["stats"]["tok_per_s"], "seconds": r["seconds"]})
                texts_dump[f"{cond}|{s}|{bs}"] = r["texts"]

            # M-free-none: which role does the unprompted, unruled model become?
            r = raw[("M-free-none", s, bs)]
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                LogisticRegression(max_iter=2000))
            clf.fit([_SPECIAL.sub(" ", t) for t in raw[("M-rule", s, bs)]["texts"]], y)
            pred = clf.predict([_SPECIAL.sub(" ", t) for t in r["texts"]])
            records.append({**base_meta, "condition": "M-free-none", "metric": "who_it_becomes",
                            "text_pred_role_dist": {name: float((pred == name).mean()) for name in role_names},
                            "owner_fractions": owner_split(r["counts"], part),
                            "role_id": "n/a: one unlabelled output per input",
                            "gen_tok_per_s": r["stats"]["tok_per_s"], "seconds": r["seconds"]})
            texts_dump[f"M-free-none|{s}|{bs}"] = r["texts"]

    # retention
    retention = None
    if not args.skip_retention:
        correct = {k: np.array([score_gsm8k_harmony(t, g)["correct"] for t, g in zip(v, ret_gold)])
                   for k, v in retention_raw.items()}
        trunc = {k: float(np.mean([score_gsm8k_harmony(t, g)["truncated"] for t, g in zip(v, ret_gold)]))
                 for k, v in retention_raw.items()}
        base_acc = correct["base"].mean()
        rng = np.random.default_rng(0)
        bidx = rng.integers(0, len(ret_gold), size=(args.n_boot, len(ret_gold)))
        retention = {"n_items": len(ret_gold), "seed": args.retention_seed,
                     "max_new_tokens": args.retention_max_new_tokens, "base_acc": float(base_acc),
                     "mcnemar_holm_vs_base": mcnemar_holm(correct, "base"), "arms": {}}
        for k, c in correct.items():
            if k == "base":
                continue
            ratios = c[bidx].mean(1) / np.maximum(correct["base"][bidx].mean(1), 1e-12)
            retention["arms"][k] = {"acc": float(c.mean()),
                                    "retention": float(c.mean() / base_acc) if base_acc > 0 else None,
                                    "retention_ci": [float(np.quantile(ratios, 0.025)),
                                                     float(np.quantile(ratios, 0.975))],
                                    "truncated_rate": trunc[k]}
        retention["truncated_rate_base"] = trunc["base"]
        retention["gate_ge_80_every_agent"] = all(
            v["retention"] is not None and v["retention"] >= 0.80
            for k, v in retention["arms"].items() if k.startswith("M-rule:"))

    # pre-registered checks, per run
    def get(cond, metric, s, bs):
        return next(r for r in records if r["condition"] == cond and r["metric"] == metric
                    and r["seed"] == s and r["batch_size"] == bs)

    checks = []
    for s in seeds:
        for bs in bss:
            diff = get("M-free-role minus B-free-role", "role_preference_diff", s, bs)["12-23"]
            pdiff = get("P-free-role minus B-free-role", "role_preference_diff", s, bs)["12-23"]
            sdiff = get("M-free-shuffled minus B-free-shuffled", "role_preference_diff", s, bs)["12-23"]
            m_text = get("M-free-role", "role_id", s, bs)["text_acc"]
            gate_text = get("M-rule", "role_id", s, bs)["text_acc"]
            checks.append({
                "seed": s, "batch_size": bs,
                "gate_text_id_rule_on_ge_60": gate_text >= 0.60, "gate_text_id": gate_text,
                "pref_diff_12_23_ci_above_0": diff["ci_lo"] > 0, "pref_diff_12_23": diff,
                "text_id_M_free_role_ge_40": m_text >= 0.40, "text_id_M_free_role": m_text,
                "placebo_diff_12_23_ci_includes_0": not pdiff["excludes_0"], "placebo_diff_12_23": pdiff,
                "shuffled_diff_12_23_ci_includes_0": not sdiff["excludes_0"], "shuffled_diff_12_23": sdiff,
            })
    h1_all = all(c["pref_diff_12_23_ci_above_0"] and c["text_id_M_free_role_ge_40"]
                 and c["placebo_diff_12_23_ci_includes_0"] for c in checks)

    res = {
        "provenance": prov, "run": run_meta(args), **common,
        "placebo": {"seed": args.placebo_seed, "max_rel_norm_err": max_rel_err},
        "lora_cfg": {"rank": trained.cfg.rank, "alpha": trained.cfg.alpha},
        "conditions_note": __doc__,
        "records": records, "retention": retention, "prereg_checks": checks,
        "H1_all_runs": h1_all,
        "seconds": time.time() - t_start,
    }
    write_json(out_path, res)
    write_json(out_path.with_name(out_path.stem + "_texts.json"), {
        "inputs": {s: [it["input_id"] for it in runs[s]["items"]] for s in seeds},
        "rp_role": {s: runs[s]["rp_role"] for s in seeds}, "texts": texts_dump,
        "retention_texts": retention_raw})

    # ------------------------------------------------------------------ summary table
    print("\nrole preference, layers 12-23 (mean [95% CI]) and role-ID accuracy (chance 0.25)")
    print(f"{'condition':38s} {'seed':>4s} {'bs':>3s} {'pref 12-23':>26s} {'route-ID':>8s} {'text-ID':>8s}")
    for s in seeds:
        for bs in bss:
            for cond in ("B-free-role", "M-free-role", "P-free-role", "M-rule", "M-free-shuffled",
                         "M-free-role minus B-free-role", "P-free-role minus B-free-role"):
                metric = "role_preference_diff" if "minus" in cond else "role_preference"
                p = get(cond, metric, s, bs)["12-23"]
                rid = next((r for r in records if r["condition"] == cond and r["metric"] == "role_id"
                            and r["seed"] == s and r["batch_size"] == bs), None)
                ra = f"{rid['routing_acc']:.3f}" if rid else "-"
                ta = f"{rid['text_acc']:.3f}" if rid else "-"
                print(f"{cond:38s} {s:4d} {bs:3d} {p['mean']:+.4f} [{p['ci_lo']:+.4f},{p['ci_hi']:+.4f}] {ra:>8s} {ta:>8s}")
    if retention:
        print(f"\nGSM8K retention (base acc {retention['base_acc']:.3f}):")
        for k, v in retention["arms"].items():
            print(f"  {k:14s} acc {v['acc']:.3f} retention {v['retention']:.3f} "
                  f"[{v['retention_ci'][0]:.3f},{v['retention_ci'][1]:.3f}]")
    print(f"\nH1 (all runs): {h1_all}; done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
