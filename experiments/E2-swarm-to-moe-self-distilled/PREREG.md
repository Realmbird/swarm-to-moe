# E2 — Swarm → MoE with agents we own: does outside diversity survive compression into one MoE?

**Status:** pre-registered 2026-10-16 (Christopher Kinoshita, Realmbird), before any training run. Replaces E0-a; supersedes E1's dependence on Martin's checkpoints (E1 stays as the protocol to rerun on his agents when available).

## The question (CLAUDE.md §4, reading 2 + reading 3)

Martin's agents were different inside, same outside. We build the mirror case: four agents that are **plainly different outside**, compress them into **one** gpt-oss-20b checkpoint as private expert stripes, remove every per-row routing rule, and ask whether the diversity survives into router + private experts or gets eaten by the shared trunk. Either answer is a GA-paper result: behaviour lives in the agents' weights (router finds it) vs. behaviour lived in the enforced assignment / communication layer (router is blind to it).

## Construction

**Step 1, agents that differ outside (self-distillation).** Roles: planner, critic, implementer, verifier (`swarm_to_moe/prompts.py`); chosen so the *style and objective* differ, not the topic. Inputs: ~2,000 seeded prompts from GSM8K train, ARC-Challenge train, MMLU validation and UltraChat (same pool as the partition profiling, disjoint seed). Teacher = base model with the role's system prompt, greedy, Harmony `final` channel kept, 256 new tokens. Student target: **no role prompt** → that role's output. Each role's rows are trained under `HardSplit` with its own agent id on partition P-all, so every token of role r passes through one of r's private experts in every layer. Trainable parameters: LoRA (rank 16) on the private experts' `gate_up_proj` and `down_proj` only; router, attention, shared experts frozen. All four roles trained jointly in mixed batches. Budget: ≤ 3 epochs, ≤ 2 H200-hours. Checkpoint = the four LoRA sets, which merge into one weights file by construction (disjoint experts).

**"Different outside" gate (must pass before step 2 counts):** with the rule ON and **no role prompt**, 5-fold (grouped by input) TF-IDF text-ID accuracy across the four agents on 300 held-out inputs ≥ 60% (chance 25%; Martin's prompted roles reached 71%, expert agents before LoRA 27–32%). Also: retention vs base on 300 GSM8K test items ≥ 80% for every agent (we accept some loss; we are not optimising capability).

**Step 2, compress and free the router.** Merge LoRA into the expert weights. Set routing to `FreeRouting`. Nothing else changes.

## Measurements (all on 300 held-out inputs per condition, 2 batch shapes, 3 subset seeds)

For the merged model with the router free:

| condition | prompt | what it asks |
|---|---|---|
| M-free-role | matching role system prompt | does the free router send role-r prompts to r's experts? |
| M-free-none | no system prompt | what does the model become when nobody tells it who it is |
| M-free-shuffled | derangement of role labels | control: must be ≈ 0 preference |
| M-rule | HardSplit, correct agent | upper reference (the swarm as trained) |
| B-free-role | base model, role prompt | E0-b arm: the partition's built-in bias and prompt-only routing signal |
| P-free-role | placebo: random LoRA noise of matched Frobenius norm on the same private experts | rules out "any perturbation attracts the router" |

Primary metrics:

1. **Role preference** (`metrics.role_preference`): fraction of top-4 slots in the matching role's private experts minus mean fraction in other roles' private experts, per layer group (0–11, 12–23, and 10–14), paired bootstrap CI over inputs. Reported for M-free-role minus B-free-role (subtracts partition bias).
2. **Role-ID from routing** (`metrics.role_id_cv`, grouped folds, chance 25%) under M-free-role and M-free-none.
3. **Role-ID from text** (TF-IDF, same folds) under M-free-role and M-free-none, against the step-1 gate value (rule on). The drop from rule-on to rule-off is "how much of the outside difference the router keeps".
4. **Deference proxy:** per token, which role's private experts win when the prompt says one role but the task text pulls another (cross-role prompts: role r prompt on inputs where role s's teacher output was longest). Fraction of tokens routed to the prompt's role vs the content's role. This is the first routing-dynamics measurement (§4 reading 3) and is exploratory.

## Pass / fail (committed)

- **Survives (H1):** M-free-role preference minus B-free-role > 0 with CI excluding 0 in layers 12–23 **and** text-ID under M-free-role ≥ 40% **and** P-free-role, M-free-shuffled ≈ 0.
- **Eaten (H0):** preference CI includes 0 or is indistinguishable from placebo, **and** text-ID under M-free-role collapses toward the pre-LoRA 27–32%.
- **Partial:** router prefers the right experts (preference > 0) but text-ID < 40%: the router finds the stripe but the stripe no longer carries the role once it is one of four voices. Report as partial with both numbers.

## Controls and conventions

Plain copies (base, no prompt) and prompted roles on base (Martin's C1) as the outside-difference reference; mixed-model team is not applicable (single checkpoint by design) and is noted as a limitation; noise placebo as above; repeated seeds and two batch shapes for all route-based numbers; retention paired on identical items; chance on every classifier. Safety-linked and super experts stay shared (they are never trained). Everything logged: model revision, chat template hash, partition hash, LoRA config, data seeds, teacher generation config.

## What it buys the GA paper

If H1: roles written into weights are readable from routing with no control rule, so routing is a role/goal probe for trained swarm agents and E3 (goal adoption as routing convergence before text) is well-posed. If H0: the role in an expert-partitioned swarm is a property of the enforced assignment, which is a direct statement that behaviour lives in the communication/control layer, and the GA paper's probes should target transcripts and activations rather than routing.

## Cost

Teacher generation ≈ 8k samples × 256 tokens ≈ 2M tokens (≈ 15 min batched). Training ≤ 2 H200-hours. Measurement ≈ 30 min. Pod already running; no new spend requested.
