#!/usr/bin/env python3
"""Profile gpt-oss-20b (prefill only) and build P-all, P-late (layers 12-23) and P-rand partitions.

Profiling set (paper draft section 3.2): n/4 prompts each from GSM8K train, ARC-Challenge train, MMLU validation,
UltraChat train_sft (first user turn), seeded; formatted as chat prompts with the same template as the runs.
Writes to --out-dir: partition_all.json, partition_late.json, partition_random.json, usage.npy, peak.npy,
router_sigma.npy, profile_meta.json.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from swarm_to_moe.data import PROFILE_SEED, profiling_texts  # noqa: E402
from swarm_to_moe.model import DEFAULT_MODEL, apply_chat, load_gpt_oss  # noqa: E402
from swarm_to_moe.partition import build_partition, random_partition  # noqa: E402
from swarm_to_moe.profile import profile_experts  # noqa: E402
from swarm_to_moe.prompts import build_messages  # noqa: E402
from swarm_to_moe.runutil import run_meta, write_json  # noqa: E402

ROLES = ("A", "B", "C", "D")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--n-prompts", type=int, default=2000, help="total; split equally over the 4 sources")
    ap.add_argument("--seed", type=int, default=PROFILE_SEED)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-shared", type=int, default=16)
    ap.add_argument("--super-frac", type=float, default=0.01)
    ap.add_argument("--late-start", type=int, default=12)
    ap.add_argument("--rand-seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/E0")
    ap.add_argument("--out", default=None, help="(compat) path for partition_all.json; sets --out-dir to its folder")
    args = ap.parse_args()
    out = Path(args.out).parent if args.out else Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sources = profiling_texts(n_per_source=args.n_prompts // 4, seed=args.seed)
    model, tok, prov = load_gpt_oss(args.model, revision=args.revision)
    texts = [apply_chat(tok, build_messages(t, None)) for src in sources.values() for t in src]
    print(f"profiling {len(texts)} prompts: " + ", ".join(f"{k}={len(v)}" for k, v in sources.items()))

    usage, peak, sigma = profile_experts(model, tok, texts, batch_size=args.batch_size, return_sigma=True)
    np.save(out / "usage.npy", usage)
    np.save(out / "peak.npy", peak)
    np.save(out / "router_sigma.npy", sigma.astype(np.float32))

    L, E = usage.shape
    p_all = build_partition(usage, peak, ROLES, n_shared=args.n_shared, super_frac=args.super_frac)
    p_late = build_partition(usage, peak, ROLES, n_shared=args.n_shared, super_frac=args.super_frac,
                             layers=range(args.late_start, L))
    p_rand = random_partition(L, E, ROLES, args.n_shared, seed=args.rand_seed)
    hashes = {}
    for name, p in [("all", p_all), ("late", p_late), ("random", p_rand)]:
        p.validate()
        p.save(out / f"partition_{name}.json")
        hashes[name] = p.hash()
        print(f"partition_{name}.json  hash={p.hash()}")

    def role_totals(p):
        return {r: float(sum(usage[l, e] for l in range(L) for e in p.private[r][l])) for r in p.roles}

    write_json(out / "profile_meta.json", {
        "provenance": prov,
        "run": run_meta(args),
        "n_prompts_per_source": {k: len(v) for k, v in sources.items()},
        "profile_seed": args.seed,
        "total_tokens_profiled": float(usage.sum() / model.config.num_experts_per_tok / L),
        "hashes": hashes,
        "n_super": max(1, int(round(args.super_frac * L * E))),
        "role_private_usage": {"all": role_totals(p_all), "late": role_totals(p_late),
                               "random": role_totals(p_rand)},
        "router_sigma": sigma.tolist(),
    })


if __name__ == "__main__":
    main()
