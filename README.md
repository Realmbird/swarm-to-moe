# swarm-to-moe

Does a swarm of role agents survive compression into a mixture of experts? Code and pre-registered
experiments for the swarm → MoE direction of the MoE ⇄ agent-swarm project (see `CLAUDE.md` for the
full context, measured results and standing conclusions).

Model organism: `openai/gpt-oss-20b` (24 layers × 32 experts, top-4), served through HF transformers
with the MXFP4 weights dequantised to bf16 so the Python router is in the loop.

## Layout

```
swarm_to_moe/
  partition.py   shared / per-role private expert sets per layer (Martin's recipe, random control, hash)
  hook.py        replaces GptOssMLP.forward: per-batch-row routing rules + routing records
                 rules: FreeRouting, HardSplit (3 shared + 1 own private), SoftNudge (±δ·σ), ForceExpert
  metrics.py     usage histograms, owner fractions, role preference, JSD/Jaccard,
                 paired bootstrap CI, McNemar + Holm, role-ID classifier (chance reported)
  prompts.py     role system prompts (placeholders until Martin's four roles are known), shuffled control
  data.py        GSM8K / MMLU loading, seeded subsets, Harmony final-channel parsing, answer scoring
  model.py       loading with provenance (revision, template hash, versions), batched greedy generation
  profile.py     expert usage + peak-output profiling used to build the partition
  lora.py        LoRA on private experts only (differentiable through the routing hook), merge, placebo init
  distill.py     self-distillation examples (no-role prompt -> role output), collate, LoRA train step
scripts/
  build_partition.py   profile 2k train/val prompts -> results/E0/partition_{all,late,random}.json + router sigma
  sanity.py            Martin's three sanity tests on the real model; must pass before any run
  run_e0b.py           E0-b: role identifiability from free routing on the base model (control arm)
  run_e0a.py           (deprioritised 2026-10-16) reproducibility check of Martin's split
  e2_generate_teacher.py   role-prompted teacher outputs on 2k inputs, 300 held out
  e2_train_lora.py         train each role into its private expert stripe, HardSplit on, router frozen
  e2_measure.py            rule off: role preference, role-ID from routing and text, placebo, shuffled
experiments/
  E0-base-routing-by-role/PREREG.md
  E1-free-router-role-preference/PREREG.md   (protocol for Martin's LoRA'd agents; blocked on checkpoints)
  E2-swarm-to-moe-self-distilled/PREREG.md   (main line: agents we own)
tests/                 CPU tests on a tiny random GptOss model (real HF code path)
results/               JSON outputs, one file per arm, provenance in every file
```

## Running

```bash
pip install -e . && python3 -m pytest -q                       # CPU, ~1 min
python3 scripts/build_partition.py --out results/E0/partition.json
python3 scripts/sanity.py --partition results/E0/partition.json  # must print PASS x3
python3 scripts/run_e0b.py --partition results/E0/partition_all.json --n-items 200 --seed 0
python3 scripts/e2_generate_teacher.py --out results/E2/teacher.jsonl
python3 scripts/e2_train_lora.py --teacher results/E2/teacher.jsonl --partition results/E0/partition_all.json
python3 scripts/e2_measure.py --lora results/E2/lora_rank16.pt --partition results/E0/partition_all.json
```

Notes that bit us:
- `attn_implementation="eager"`: gpt-oss has no SDPA path in transformers 5.x.
- Output is Harmony format (`analysis` channel, then `final`). Score the `final` channel; track truncation.
- Routing is not bit-reproducible across batch shapes. Route-based measures need repeated runs.

## Conventions

Pre-registration committed before the run it governs. Retention relative to base on the same items with
paired bootstrap CIs; McNemar + Holm for paired comparisons; chance baseline on every classifier.
Update `CLAUDE.md` §2–4 in the same PR as the result.
