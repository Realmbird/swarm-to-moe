#!/usr/bin/env python3
"""E2 step 1b: train LoRA on the private experts under HardSplit (see experiments/E2-.../PREREG.md).

Examples: teacher rows (e2_generate_teacher.py) with role != "none", input not held out, not truncated and with a
non-empty final answer. Student input = no role prompt; target = the role's final answer. Each role is mapped to
a partition role in order (planner->A, critic->B, implementer->C, verifier->D); every row runs under HardSplit
with its partition role as agent, so each token passes one of that role's private experts in every layer. All
roles are trained jointly in shuffled mixed batches. Only LoRA parameters (rank r on gate_up_proj and down_proj of
the private experts) are trained.
"""
import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from swarm_to_moe.distill import DistillExample, collate, encode_example, lora_forward_loss  # noqa: E402
from swarm_to_moe.hook import HardSplit, MoERoutingHook  # noqa: E402
from swarm_to_moe.lora import LoRAConfig, attach_expert_lora  # noqa: E402
from swarm_to_moe.model import DEFAULT_MODEL, load_gpt_oss  # noqa: E402
from swarm_to_moe.partition import Partition  # noqa: E402
from swarm_to_moe.prompts import DEFAULT_ROLES  # noqa: E402
from swarm_to_moe.runutil import read_json, run_meta, write_json  # noqa: E402


def load_examples(teacher_path, inputs_path, role_map):
    heldout = {it["input_id"] for it in read_json(inputs_path)["items"] if it["heldout"]}
    examples, skipped = [], Counter()
    with open(teacher_path) as f:
        for line in f:
            r = json.loads(line)
            if r["role"] not in role_map:
                skipped["no_role_arm"] += 1
                continue
            if r["input_id"] in heldout or r.get("heldout"):
                skipped["heldout"] += 1
                continue
            if r["truncated"]:
                skipped["truncated"] += 1
                continue
            if not (r["final"] or "").strip():
                skipped["empty_final"] += 1
                continue
            examples.append(DistillExample(role=role_map[r["role"]], input_text=r["input_text"],
                                           target_text=r["final"]))
    return examples, dict(skipped), heldout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--teacher", default="results/E2/teacher.jsonl")
    ap.add_argument("--inputs", default=None, help="default: inputs.json next to --teacher")
    ap.add_argument("--partition", default="results/E0/partition_all.json")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=None, help="cap optimizer steps (smoke tests)")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--out", default="results/E2/lora_rank16.pt")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    inputs_path = Path(args.inputs) if args.inputs else Path(args.teacher).parent / "inputs.json"
    log_path = out.parent / "train_log.jsonl"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    part = Partition.load(args.partition)
    part.validate()
    role_names = [r.name for r in DEFAULT_ROLES]
    if len(part.roles) != len(role_names):
        raise ValueError(f"partition has {len(part.roles)} roles, DEFAULT_ROLES has {len(role_names)}")
    role_map = {name: part.roles[i] for i, name in enumerate(role_names)}
    write_json(out.parent / "role_map.json", {"role_map": role_map, "partition_hash": part.hash(),
                                              "partition": args.partition})

    examples, skipped, heldout = load_examples(args.teacher, inputs_path, role_map)
    print(f"{len(examples)} examples; skipped {skipped}; per role {Counter(e.role for e in examples)}")

    model, tok, prov = load_gpt_oss(args.model, revision=args.revision)
    encoded, n_too_long = [], 0
    for ex in examples:
        try:
            encoded.append(encode_example(tok, ex, args.max_len))
        except ValueError:
            n_too_long += 1
    print(f"encoded {len(encoded)} (prompt too long: {n_too_long}); "
          f"mean len {sum(len(e['input_ids']) for e in encoded) / max(1, len(encoded)):.0f}")

    cfg = LoRAConfig(rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    handle = attach_expert_lora(model, part, part.roles, cfg)
    print(f"LoRA params: {handle.num_parameters():,} "
          f"({', '.join(f'{r}={sum(p.numel() for p in handle.parameters_for_role(r)):,}' for r in part.roles)})")

    # Gradient checkpointing only runs in train mode in HF (``self.gradient_checkpointing and self.training``).
    # gpt-oss has no dropout (attention_dropout=0) so train mode does not change the math. Non-reentrant
    # checkpointing re-runs each decoder layer, including the routing hook's mlp.forward (record=False, static
    # row agents) and the LoRA experts.forward, in backward; both are deterministic functions of their inputs.
    # If enabling it or the first step fails we fall back to no checkpointing (see the retry below).
    use_ckpt = not args.no_grad_ckpt
    if use_ckpt:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
        except Exception as e:  # noqa: BLE001
            print(f"gradient checkpointing unavailable ({e!r}); continuing without")
            use_ckpt = False
    # No autocast: the base model is already bf16 end to end, and the LoRA delta path deliberately runs in
    # float32 (it disables autocast internally), so autocast would only add casts.

    params = list(handle.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    micro_per_epoch = math.ceil(len(encoded) / args.batch_size)
    total_steps = math.ceil(micro_per_epoch * args.epochs / args.grad_accum)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def save(step, final=False):
        handle.save(out if final else out.with_name(f"{out.stem}_step{step}{out.suffix}"), extra={
            "role_map": role_map, "partition_path": args.partition, "args": vars(args), "provenance": prov,
            "step": step, "total_steps": total_steps, "n_examples": len(encoded), "skipped": skipped,
            "heldout_input_ids": sorted(heldout), "final": final,
        })

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log_f = open(log_path, "a")
    step, micro, t0 = 0, 0, time.time()
    run_loss, run_n, n_tok = 0.0, 0, 0
    retried = False
    done = False
    with MoERoutingHook(model, rule=HardSplit(part), record=False) as hook:
        for epoch in range(args.epochs):
            order = list(range(len(encoded)))
            random.Random(args.seed + epoch).shuffle(order)
            batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
            for bi, rows in enumerate(batches):
                batch = collate([encoded[i] for i in rows], pad_id)
                try:
                    loss, n_valid = lora_forward_loss(model, hook, batch)
                    (loss / args.grad_accum).backward()
                except (RuntimeError, ValueError) as e:
                    if use_ckpt and not retried and micro == 0:
                        print(f"first step failed with gradient checkpointing ({e!r}); disabling it and retrying")
                        model.gradient_checkpointing_disable()
                        model.eval()
                        use_ckpt, retried = False, True
                        opt.zero_grad(set_to_none=True)
                        loss, n_valid = lora_forward_loss(model, hook, batch)
                        (loss / args.grad_accum).backward()
                    else:
                        raise
                run_loss += float(loss.detach())
                run_n += 1
                n_tok += n_valid
                micro += 1
                if micro % args.grad_accum and bi != len(batches) - 1:
                    continue
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0 or step == 1:
                    rec = {"step": step, "epoch": epoch, "loss": run_loss / max(1, run_n), "lr": sched.get_last_lr()[0],
                           "target_tokens": n_tok, "elapsed_s": time.time() - t0,
                           "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
                           "grad_ckpt": use_ckpt}
                    print(json.dumps(rec), flush=True)
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
                    run_loss, run_n = 0.0, 0
                if step % args.save_every == 0:
                    save(step)
                if step >= total_steps:
                    done = True
                    break
            if done:
                break
    log_f.close()
    model.eval()
    save(step, final=True)
    norms = handle.delta_norms()
    by_role = {r: [] for r in part.roles}
    for (l, e), v in norms.items():
        by_role[part.owner(l, e)].append(v)
    write_json(out.with_suffix(".meta.json"), {
        "provenance": prov, "run": run_meta(args), "role_map": role_map, "partition_hash": part.hash(),
        "lora": {"rank": cfg.rank, "alpha": cfg.alpha, "dropout": cfg.dropout, "dtype": "float32",
                 "n_params": handle.num_parameters()},
        "steps": step, "examples": len(encoded), "skipped": skipped, "prompt_too_long": n_too_long,
        "grad_ckpt": use_ckpt, "seconds": time.time() - t0,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        "delta_norm_mean_by_role": {r: sum(v) / max(1, len(v)) for r, v in by_role.items()},
    })
    if torch.cuda.is_available():
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print(f"saved {out} after {step} steps in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
