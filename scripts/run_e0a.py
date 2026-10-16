#!/usr/bin/env python3
"""Deprioritised 2026-10-16 (E0-a dropped by the user); kept as-is, untested on GPU.

E0-a:reproducibility check of Martin's construction on GSM8K test (see experiments/E0-.../PREREG.md).

Arms: base (free), copies (4 free rows per item), soft (SoftNudge delta on P-all), hard_all / hard_late / hard_rand
(HardSplit on P-all / P-late / P-rand). Agent arms run every item once per agent (A, B, C, D), same prompt text.
Writes results/E0/e0a_<arm>.json (+ e0a_<arm>_hist.npz) and results/E0/e0a_summary.json.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from swarm_to_moe.data import GSM8K_SUFFIX, load_gsm8k, score_gsm8k_harmony, seeded_subset  # noqa: E402
from swarm_to_moe.hook import FreeRouting, HardSplit, MoERoutingHook, SoftNudge, check_rule_holds  # noqa: E402
from swarm_to_moe.metrics import (  # noqa: E402
    l1_per_layer,
    mcnemar_holm,
    mean_pairwise_kappa,
    paired_bootstrap_ci,
    role_id_cv,
    routing_jaccard_counts,
    usage_counts,
)
from swarm_to_moe.model import DEFAULT_MODEL, apply_chat, generate_batched, load_gpt_oss, prefill_routing  # noqa: E402
from swarm_to_moe.partition import Partition  # noqa: E402
from swarm_to_moe.prompts import build_messages  # noqa: E402
from swarm_to_moe.runutil import read_json, run_meta, write_json  # noqa: E402

ALL_ARMS = ("base", "copies", "soft", "hard_all", "hard_late", "hard_rand")


def retention_ci(agent, base, n_boot=2000, seed=0):
    agent, base = np.asarray(agent, float), np.asarray(base, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(base), size=(n_boot, len(base)))
    ra, rb = agent[idx].mean(1), base[idx].mean(1)
    r = np.divide(ra, rb, out=np.full_like(ra, np.nan), where=rb > 0)
    lo, hi = np.nanquantile(r, [0.025, 0.975])
    return float(agent.mean() / base.mean()) if base.mean() > 0 else float("nan"), float(lo), float(hi)


def build_rule(arm, parts, delta, sigma):
    if arm in ("base", "copies"):
        return FreeRouting()
    if arm == "soft":
        return SoftNudge(parts["all"], delta=delta, sigma=sigma)
    return HardSplit(parts[{"hard_all": "all", "hard_late": "late", "hard_rand": "rand"}[arm]])


def run_arm(arm, args, model, tok, prompts, golds, parts, sigma, prov, out_dir):
    roles = parts["all"].roles
    n = len(prompts)
    if arm == "base":
        rows = [(i, None) for i in range(n)]
    else:
        rows = [(i, a) for i in range(n) for a in roles]  # item-major: each batch holds whole items
    row_prompts = [prompts[i] for i, _ in rows]
    row_agents = [a for _, a in rows] if arm not in ("base", "copies") else None
    rule = build_rule(arm, parts, args.delta, sigma)
    L = model.config.num_hidden_layers
    E = model.config.num_local_experts

    gen_counts = np.zeros((len(rows), L, E), dtype=np.int32)
    jacc = {a: [0, 0] for a in roles}
    rule_ok = True
    texts = [None] * len(rows)
    t0 = time.time()
    with MoERoutingHook(model, rule) as hook:
        for s in range(0, len(rows), args.batch_size):
            sl = list(range(s, min(s + args.batch_size, len(rows))))
            bp = [row_prompts[j] for j in sl]
            ba = [row_agents[j] for j in sl] if row_agents is not None else None
            if arm != "base":
                # free prefill routing of the identical batch (same shape/padding) for the Jaccard comparison
                hook.set_rule(FreeRouting())
                free_idx, attn = prefill_routing(model, tok, bp, hook)
                hook.set_rule(rule)
            t, routing = generate_batched(model, tok, bp, args.max_new_tokens, len(bp), hook=hook,
                                          row_agents=ba, progress=False)
            rec = routing[0]
            for j, txt in zip(sl, t):
                texts[j] = txt
            idx = rec["indices"].long()
            gen_counts[sl] = usage_counts(idx, E, mask=rec["gen_mask"])
            if rule.kind == "hard":
                rule_ok = rule_ok and check_rule_holds(idx, rule, ba)
            if arm != "base":
                P = free_idx.shape[2]
                for a in roles:
                    rr = [k for k, j in enumerate(sl) if rows[j][1] == a]
                    if not rr:
                        continue
                    i_, u_ = routing_jaccard_counts(idx[:, rr, :P], free_idx[:, rr].long(), mask=attn[rr])
                    jacc[a][0] += i_
                    jacc[a][1] += u_
            done = sl[-1] + 1
            el = time.time() - t0
            st = generate_batched.last_stats
            print(f"[{arm}] {done}/{len(rows)} rows  {el:.0f}s  batch {st['tok_per_s']:.0f} tok/s", flush=True)

    scores = [score_gsm8k_harmony(texts[j], golds[rows[j][0]]) for j in range(len(rows))]
    res = {"arm": arm, "rule": rule.to_dict(), "provenance": prov, "run": run_meta(args),
           "partition_hashes": {k: p.hash() for k, p in parts.items()}, "seconds": time.time() - t0,
           "n_items": n, "rule_holds_every_token": bool(rule_ok) if rule.kind == "hard" else None}
    if arm == "base":
        correct = np.array([s["correct"] for s in scores])
        res["agents"] = {"base": {"accuracy": float(correct.mean()), "correct": correct.astype(int).tolist(),
                                  "truncated": [int(s["truncated"]) for s in scores],
                                  "truncation_rate": float(np.mean([s["truncated"] for s in scores])),
                                  "preds": [s["pred"] for s in scores]}}
    else:
        res["agents"] = {}
        for a in roles:
            js = [j for j, (_, ag) in enumerate(rows) if ag == a]
            correct = np.array([scores[j]["correct"] for j in js])
            res["agents"][a] = {
                "accuracy": float(correct.mean()), "correct": correct.astype(int).tolist(),
                "truncated": [int(scores[j]["truncated"]) for j in js],
                "truncation_rate": float(np.mean([scores[j]["truncated"] for j in js])),
                "preds": [scores[j]["pred"] for j in js],
                "jaccard_vs_free_prompt_tokens": jacc[a][0] / jacc[a][1] if jacc[a][1] else None,
            }
        # agent-ID from routing: per-layer L1-normalised generated-token histograms, folds grouped by item
        X = l1_per_layer(gen_counts).reshape(len(rows), -1)
        y = np.array([a for _, a in rows])
        groups = np.array([i for i, _ in rows])
        acc, chance = role_id_cv(X, y, folds=5, seed=args.seed, groups=groups)
        res["agent_id_from_routing"] = {"accuracy": acc, "chance": chance, "features": "gen-token usage, L1/layer"}
        res["error_kappa_mean_pairwise"] = mean_pairwise_kappa(
            {a: np.array(res["agents"][a]["correct"]) for a in roles})
    np.savez_compressed(out_dir / f"e0a_{arm}_hist.npz", gen_counts=gen_counts,
                        rows_item=np.array([i for i, _ in rows]),
                        rows_agent=np.array([str(a) for _, a in rows]))
    res["texts"] = texts
    write_json(out_dir / f"e0a_{arm}.json", res)
    return res


def summarize(out_dir, args):
    base_path = out_dir / "e0a_base.json"
    if not base_path.exists():
        print("no base arm yet; summary skipped")
        return
    base = read_json(base_path)
    bc = np.array(base["agents"]["base"]["correct"], bool)
    summary = {"run": run_meta(args), "provenance": base["provenance"], "partition_hashes": base["partition_hashes"],
               "base_accuracy": float(bc.mean()), "base_truncation_rate": base["agents"]["base"]["truncation_rate"],
               "arms": {}}
    all_correct = {"base": bc}
    for arm in ALL_ARMS[1:]:
        p = out_dir / f"e0a_{arm}.json"
        if not p.exists():
            continue
        r = read_json(p)
        arm_s = {"rule": r["rule"], "rule_holds_every_token": r["rule_holds_every_token"],
                 "agent_id_from_routing": r.get("agent_id_from_routing"),
                 "error_kappa_mean_pairwise": r.get("error_kappa_mean_pairwise"), "agents": {}}
        arm_corr = {"base": bc}
        for a, d in r["agents"].items():
            c = np.array(d["correct"], bool)
            arm_corr[a] = c
            all_correct[f"{arm}:{a}"] = c
            diff, lo, hi = paired_bootstrap_ci(c.astype(float), bc.astype(float), n_boot=2000, seed=args.seed)
            ret, rlo, rhi = retention_ci(c, bc, seed=args.seed)
            arm_s["agents"][a] = {"accuracy": d["accuracy"], "retention": ret, "retention_ci95": [rlo, rhi],
                                  "acc_diff_vs_base": diff, "acc_diff_ci95": [lo, hi],
                                  "truncation_rate": d["truncation_rate"],
                                  "jaccard_vs_free_prompt_tokens": d.get("jaccard_vs_free_prompt_tokens")}
        arm_s["mcnemar_holm_within_arm"] = mcnemar_holm(arm_corr, reference="base")
        rets = [v["retention"] for v in arm_s["agents"].values()]
        arm_s["mean_retention"] = float(np.mean(rets))
        arm_s["min_retention"] = float(np.min(rets))
        arm_s["truncation_rate"] = float(np.mean([v["truncation_rate"] for v in arm_s["agents"].values()]))
        summary["arms"][arm] = arm_s
    summary["mcnemar_holm_all_arms"] = mcnemar_holm(all_correct, reference="base")
    gates = {}
    if "hard_all" in summary["arms"]:
        h = summary["arms"]["hard_all"]
        gates["hard_all_mean_retention_in_70_88"] = 0.70 <= h["mean_retention"] <= 0.88
        gates["hard_all_some_agent_below_90"] = h["min_retention"] < 0.90
    if "soft" in summary["arms"]:
        gates["soft_every_agent_ge_95"] = summary["arms"]["soft"]["min_retention"] >= 0.95
    summary["prereg_checks"] = gates
    write_json(out_dir / "e0a_summary.json", summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--n-items", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--partition", dest="partition_all", default="results/E0/partition_all.json",
                    help="alias of --partition-all")
    ap.add_argument("--partition-all", dest="partition_all")
    ap.add_argument("--partition-late", default="results/E0/partition_late.json")
    ap.add_argument("--partition-rand", default="results/E0/partition_random.json")
    ap.add_argument("--router-sigma", default="results/E0/router_sigma.npy")
    ap.add_argument("--sigma", choices=["per_layer", "per_token"], default="per_layer")
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", default="results/E0")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        summarize(out_dir, args)
        return
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ALL_ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}; choose from {ALL_ARMS}")

    parts = {"all": Partition.load(args.partition_all)}
    if "hard_late" in arms:
        parts["late"] = Partition.load(args.partition_late)
    if "hard_rand" in arms:
        parts["rand"] = Partition.load(args.partition_rand)
    for p in parts.values():
        p.validate()
    sigma = "per_token"
    if args.sigma == "per_layer":
        sigma = torch.from_numpy(np.load(args.router_sigma)).float()

    model, tok, prov = load_gpt_oss(args.model, revision=args.revision)
    items = seeded_subset(load_gsm8k("test"), args.n_items, args.seed)
    prompts = [apply_chat(tok, build_messages(x["question"] + GSM8K_SUFFIX, None)) for x in items]
    golds = [x["answer"] for x in items]
    for arm in arms:
        print(f"=== arm {arm} ===", flush=True)
        run_arm(arm, args, model, tok, prompts, golds, parts, sigma, prov, out_dir)
        torch.cuda.empty_cache()
    summarize(out_dir, args)


if __name__ == "__main__":
    main()
