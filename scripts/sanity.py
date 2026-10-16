#!/usr/bin/env python3
"""Martin's three sanity tests on the real model (CLAUDE.md section 1). Exit code != 0 on any FAIL.

  1. hook-off: FreeRouting and SoftNudge(delta=0) reproduce the unhooked greedy tokens exactly (32 prompts, 32 tokens);
  2. rule holds on every recorded token: HardSplit (P-all, and P-late if given) via check_rule_holds; SoftNudge
     delta=0.5 via the tie-robust top-k-of-biased-logits check;
  3. ForceExpert at a mid layer (least-used expert there) changes the output.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from swarm_to_moe.data import GSM8K_SUFFIX, load_gsm8k, seeded_subset  # noqa: E402
from swarm_to_moe.hook import ForceExpert, FreeRouting, HardSplit, MoERoutingHook, SoftNudge, check_rule_holds  # noqa: E402
from swarm_to_moe.model import DEFAULT_MODEL, apply_chat, load_gpt_oss  # noqa: E402
from swarm_to_moe.partition import Partition  # noqa: E402
from swarm_to_moe.prompts import build_messages  # noqa: E402
from swarm_to_moe.runutil import run_meta, write_json  # noqa: E402


@torch.no_grad()
def gen(model, tok, prompts, new, hook=None, agents=None):
    """Greedy generate one batch; returns (token ids [B,N] cpu, records or None)."""
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    if hook is not None:
        hook.set_row_agents(agents)
        hook.reset_records()
    out = model.generate(**enc, max_new_tokens=new, min_new_tokens=new, do_sample=False,
                         pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
    gen_ids = out[:, enc["input_ids"].shape[1]:].cpu()
    if hook is None:
        return gen_ids, None
    rec = {"indices": hook.records.stack()}
    if hook.record_logits:
        rec["logits"] = hook.records.stack_logits()
    hook.reset_records()
    return gen_ids, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--partition", default="results/E0/partition_all.json")
    ap.add_argument("--partition-late", default=None)
    ap.add_argument("--router-sigma", default="results/E0/router_sigma.npy")
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force-layer", type=int, default=12)
    ap.add_argument("--out", default="results/E0/sanity.json")
    args = ap.parse_args()

    part = Partition.load(args.partition)
    part.validate()
    sigma = "per_token"
    if args.router_sigma and Path(args.router_sigma).exists():
        sigma = torch.from_numpy(np.load(args.router_sigma)).float()
    model, tok, prov = load_gpt_oss(args.model, revision=args.revision)
    items = seeded_subset(load_gsm8k("test"), args.n_prompts, args.seed)
    prompts = [apply_chat(tok, build_messages(x["question"] + GSM8K_SUFFIX, None)) for x in items]
    agents = [part.roles[i % len(part.roles)] for i in range(len(prompts))]
    results = {}
    t0 = time.time()

    base, _ = gen(model, tok, prompts, args.new_tokens)

    # 1. hook-off reproduces base
    with MoERoutingHook(model, FreeRouting()) as hook:
        free_ids, free_rec = gen(model, tok, prompts, args.new_tokens, hook)
    with MoERoutingHook(model, SoftNudge(part, delta=0.0)) as hook:
        soft0_ids, _ = gen(model, tok, prompts, args.new_tokens, hook, agents)
    after, _ = gen(model, tok, prompts, args.new_tokens)
    results["hook_off_free"] = bool(torch.equal(base, free_ids))
    results["hook_off_soft_delta0"] = bool(torch.equal(base, soft0_ids))
    results["uninstall_restores"] = bool(torch.equal(base, after))

    # 2. rule holds on every recorded token
    rule = HardSplit(part)
    with MoERoutingHook(model, rule) as hook:
        hard_ids, rec = gen(model, tok, prompts, args.new_tokens, hook, agents)
    results["hard_all_rule_holds"] = bool(check_rule_holds(rec["indices"], rule, agents))
    results["hard_all_changes_output"] = bool(not torch.equal(base, hard_ids))
    if args.partition_late:
        pl = Partition.load(args.partition_late)
        rule_l = HardSplit(pl)
        with MoERoutingHook(model, rule_l) as hook:
            _, rec_l = gen(model, tok, prompts, args.new_tokens, hook, agents)
        results["hard_late_rule_holds"] = bool(check_rule_holds(rec_l["indices"], rule_l, agents))
        # non-privatised layers see unchanged prompt hidden states, so prompt-token routing there is base routing
        P = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].shape[1]
        early = [l for l in range(pl.num_layers) if not pl.is_privatised(l)]
        results["hard_late_early_layers_match_free_on_prompt"] = bool(
            torch.equal(rec_l["indices"][early, :, :P], free_rec["indices"][early, :, :P])
        ) if early else True
    soft = SoftNudge(part, delta=0.5, sigma=sigma)
    with MoERoutingHook(model, soft, record_logits=True) as hook:
        soft_ids, rec_s = gen(model, tok, prompts, args.new_tokens, hook, agents)
    results["soft_0.5_rule_holds"] = bool(check_rule_holds(rec_s["indices"], soft, agents, logits=rec_s["logits"]))
    del rec_s

    # 3. forcing a named expert changes output
    use = np.bincount(free_rec["indices"][args.force_layer].reshape(-1).numpy(), minlength=part.num_experts)
    expert = int(np.argmin(use))
    with MoERoutingHook(model, ForceExpert(args.force_layer, expert)) as hook:
        force_ids, rec_f = gen(model, tok, prompts, args.new_tokens, hook)
    results["force_expert_recorded"] = bool(check_rule_holds(rec_f["indices"], ForceExpert(args.force_layer, expert)))
    results["force_expert_changes_output"] = bool(not torch.equal(base, force_ids))

    ok = all(results.values())
    for k, v in results.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print("ALL PASS" if ok else "SOME FAILED")
    write_json(args.out, {
        "provenance": prov, "run": run_meta(args), "partition_hash": part.hash(),
        "sigma": "per_token" if isinstance(sigma, str) else sigma.tolist(),
        "force": {"layer": args.force_layer, "expert": expert},
        "results": results, "all_pass": ok, "seconds": time.time() - t0,
        "n_rows_differing": {
            "hard_all": int((base != hard_ids).any(-1).sum()),
            "soft_0.5": int((base != soft_ids).any(-1).sum()),
            "force": int((base != force_ids).any(-1).sum()),
        },
    })
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
