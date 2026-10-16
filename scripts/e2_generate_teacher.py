#!/usr/bin/env python3
"""E2 step 1a: teacher generation for self-distillation (see experiments/E2-swarm-to-moe-self-distilled/PREREG.md).

Inputs: seeded --n-per-source items each from GSM8K train, ARC-Challenge train, MMLU validation and UltraChat
train_sft (first user turn), drawn with --seed (the partition profiling uses data.PROFILE_SEED) and with every
profiling text excluded, so the pool is disjoint from the partition's profiling set. The combined list is
shuffled with --seed; the last --n-heldout inputs (in that order) are held out and NEVER used for training.

Teacher = base gpt-oss-20b, greedy, --max-new-tokens, for each of the four DEFAULT_ROLES system prompts and for
"none" (no system prompt). One JSON line per (input, role):
  input_id, source, input_text, role, analysis, final, truncated, has_final, n_new_tokens
Writes --out (teacher.jsonl) and inputs.json next to it.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarm_to_moe.data import (  # noqa: E402
    PROFILE_SEED,
    load_arc,
    load_gsm8k,
    load_mmlu_val,
    load_ultrachat,
    profiling_texts,
    seeded_subset,
    split_harmony,
)
from swarm_to_moe.model import DEFAULT_MODEL, apply_chat, generate_batched, load_gpt_oss  # noqa: E402
from swarm_to_moe.prompts import DEFAULT_ROLES, build_messages  # noqa: E402
from swarm_to_moe.runutil import run_meta, write_json  # noqa: E402

import numpy as np  # noqa: E402

STOP_STRS = ("<|return|>", "<|end|>", "<|call|>")


def build_inputs(n_per_source: int, seed: int, n_heldout: int):
    if seed == PROFILE_SEED:
        raise ValueError(f"--seed must differ from the profiling seed {PROFILE_SEED}")
    prof = profiling_texts(n_per_source=500, seed=PROFILE_SEED)
    excluded = set(itertools.chain.from_iterable(prof.values()))
    pools = {
        "gsm8k_train": [x["question"] for x in load_gsm8k("train")],
        "arc_challenge_train": [x["text"] for x in load_arc("train")],
        "mmlu_val": [x["text"] for x in load_mmlu_val()],
        # ultrachat is large: draw 3x then filter, keeping the seeded order
        "ultrachat_train_sft": load_ultrachat(3 * n_per_source + len(excluded), seed=seed),
    }
    items = []
    for src, pool in pools.items():
        seen = set()
        pool = [t for t in pool if t and t not in excluded and not (t in seen or seen.add(t))]
        if src == "ultrachat_train_sft":
            chosen = pool[:n_per_source]
            if len(chosen) < n_per_source:
                raise ValueError("not enough UltraChat items after exclusion")
        else:
            chosen = seeded_subset(pool, n_per_source, seed)
        items += [{"source": src, "input_text": t} for t in chosen]
    order = np.random.default_rng(seed).permutation(len(items))
    items = [items[i] for i in order]
    for i, it in enumerate(items):
        it["input_id"] = i
        it["heldout"] = i >= len(items) - n_heldout
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--n-per-source", type=int, default=500)
    ap.add_argument("--n-heldout", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1, help=f"input seed (profiling uses {PROFILE_SEED})")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--roles", default="all", help="comma list of role names and/or 'none'; default all + none")
    ap.add_argument("--out", default="results/E2/teacher.jsonl")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    items = build_inputs(args.n_per_source, args.seed, args.n_heldout)
    model, tok, prov = load_gpt_oss(args.model, revision=args.revision)
    write_json(out.parent / "inputs.json", {
        "provenance": prov, "run": run_meta(args), "seed": args.seed, "profile_seed_excluded": PROFILE_SEED,
        "n_heldout": args.n_heldout, "items": items,
    })

    role_obj = {r.name: r for r in DEFAULT_ROLES}
    arms = [r.name for r in DEFAULT_ROLES] + ["none"] if args.roles == "all" else args.roles.split(",")
    for a in arms:
        if a != "none" and a not in role_obj:
            raise ValueError(f"unknown role {a!r}")

    summary = {}
    with open(out, "w") as f:
        for a in arms:
            print(f"=== teacher arm {a} ===", flush=True)
            prompts = [apply_chat(tok, build_messages(it["input_text"], role_obj.get(a))) for it in items]
            # length-sorted batches reduce padding; results are mapped back to input order
            lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
            order = sorted(range(len(prompts)), key=lambda i: lens[i])
            t0 = time.time()
            texts_sorted, _ = generate_batched(model, tok, [prompts[i] for i in order], args.max_new_tokens,
                                               args.batch_size)
            dt = time.time() - t0
            stats = dict(generate_batched.last_stats)
            texts = [None] * len(prompts)
            for j, i in enumerate(order):
                texts[i] = texts_sorted[j]
            n_trunc = n_nofinal = 0
            for it, text in zip(items, texts):
                analysis, final = split_harmony(text)
                n_new = len(tok(text, add_special_tokens=False)["input_ids"])
                stopped = text.rstrip().endswith(STOP_STRS)
                truncated = (not stopped) or final is None
                n_trunc += truncated
                n_nofinal += final is None
                f.write(json.dumps({
                    "input_id": it["input_id"], "source": it["source"], "input_text": it["input_text"],
                    "heldout": it["heldout"], "role": a, "analysis": analysis, "final": final,
                    "truncated": bool(truncated), "has_final": final is not None, "n_new_tokens": n_new,
                }) + "\n")
            f.flush()
            summary[a] = {"truncation_rate": n_trunc / len(items), "no_final_rate": n_nofinal / len(items),
                          "tok_per_s": stats["tok_per_s"], "generated_tokens": stats["generated_tokens"],
                          "seconds": dt}
            print(f"[{a}] truncated {n_trunc}/{len(items)} ({n_trunc / len(items):.1%}), no final "
                  f"{n_nofinal}, {stats['tok_per_s']:.0f} tok/s", flush=True)

    write_json(out.parent / "teacher_meta.json", {
        "provenance": prov, "run": run_meta(args),
        "generation": {"greedy": True, "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
                       "reasoning_effort": "low", "length_sorted_batches": True},
        "role_prompts": {r.name: r.system_prompt for r in DEFAULT_ROLES},
        "summary": summary,
    })
    print("\nrole          trunc   no_final  tok/s")
    for a, s in summary.items():
        print(f"{a:12s}  {s['truncation_rate']:6.1%}  {s['no_final_rate']:7.1%}  {s['tok_per_s']:6.0f}")


if __name__ == "__main__":
    main()
