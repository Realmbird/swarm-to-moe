# E0 — Does the *untrained* gpt-oss-20b router already separate role prompts?

**Status:** pre-registered 2026-10-16, before any run. Christopher Kinoshita (Realmbird).
**Why this first:** CLAUDE.md §4 says the reproducibility check on the 20b results comes before anything else, and E1 (free router over merged LoRA'd role experts) needs the base-model arm as its control. E0 delivers both with no LoRA checkpoints and ~1–2 H200-hours.

## Question

Reading 3 of §4 (swarm dynamics as routing dynamics) starts with the weakest claim: on the plain base model, with the router free over all 32 experts, is the *role prompt* an agent is running under recoverable from its routing alone? If yes before any training, routing is already a role probe and E1 asks whether LoRA sharpens it. If no, E1 asks whether LoRA creates it.

## Parts

### E0-a Reproducibility check of Martin's construction (§1–2)

Rebuild the partition with Martin's recipe as written in the 5 Oct draft (§3.2): profile the base model on 2,000 prompts, 500 each from GSM8K **train**, ARC-Challenge **train**, MMLU **validation** and UltraChat; super experts = top 1% by peak output magnitude, forced shared; fill shared to 16 per layer by usage; snake-deal the remaining 16 to 4 agents by usage (A B C D D C B A …), balanced so no agent's private usage exceeds 1.2× another's. Build P-all (all 24 layers), P-late (layers 12–23 only; earlier layers fully shared) and P-rand (random deal, control). Soft rule uses σ_l = per-layer std of router scores from the same profiling pass (draft §3.1).

Arms, each on the same seeded 500-item GSM8K **test** subset, greedy, same chat template and max tokens: base; plain copies; soft δ=0.5 (P-all); hard P-all; **hard P-late**; hard P-rand.

- **Metric:** accuracy relative to base on the same items; paired bootstrap 95% CI; McNemar + Holm vs base.
- **Expectation (from §2 and the draft):** plain copies 100%; soft ≥ 98%; hard P-all agents spread 66–91% with mean ≈ 79% (the one measured number in the draft, 0.948 → 0.748). The draft's **P-late 93% and P-rand 76% are placeholders**; this run is the first measurement of them. This is a recipe replication, not a bit-exact reproduction (Martin's partition hash is not available to us). **Pass:** hard P-all mean retention within 70–88% and at least one agent < 90% (G1 fails as reported); soft retention ≥ 95% for every agent. For P-late we pre-commit the draft's claim as the hypothesis: mean retention ≥ 90% and worst agent ≥ 90% (G1 passes). **Fail** either way is informative: if hard P-all is *not* costly with our partition, Martin's result is partition-sensitive and §2 needs a caveat; if P-late does not pass G1, the draft's abstract claim must be withdrawn.
- **Sanity tests gating the run (must all pass or the run does not count):** hook-off reproduces base tokens exactly on 32 prompts; rule holds on every token (asserted by `check_rule_holds` over all recorded routing); forcing a named expert changes output.
- **Routing stats recorded for free:** per-agent top-4 Jaccard with base on identical text (expect ≈0.70 soft / ≈0.57 hard), agent-ID-from-routing (expect 100%).

### E0-b Role identifiability from free routing on the base model

Four role system prompts (planner, critic, implementer, verifier; see `swarm_to_moe/prompts.py` — **placeholder roles**, to be replaced by the four roles Martin LoRA'd once known) × 200 task inputs (seeded 100 GSM8K-train + 100 MMLU-val questions, disjoint from the profiling items) + a no-role arm. Router free. Greedy, 128 new tokens. Record top-4 indices per layer for prompt tokens and generated tokens separately.

- **Feature per (role, input):** per-layer expert-usage histogram over generated tokens, 24×32 = 768 dims, L1-normalised per layer; plus per-layer mean router entropy (24 dims), reported separately and concatenated.
- **Primary metric:** 5-fold cross-validated role-ID accuracy from routing features (logistic regression), chance 25%. Report per-layer-group accuracy for layers 0–7, 8–15, 16–23 and for the 10–14 band that both multi-agent probe papers used (Martin's research notes, item 1).
- **Prior stated up front:** RASET and RARE find routing is largely topic-driven. Because every role sees the *same* inputs here, topic is held fixed and any role signal is prompt/intent-shaped. The prior therefore predicts a null or weak result; a positive result is the surprising one. This is Martin's suggested experiment 1 (routing fingerprints with topic held fixed) run on prompted roles before any training.
- **Noise floor:** routing JSD between two runs of the *same* role on the same inputs across two batch shapes (routing is not bit-reproducible across batch shapes, §5), and between the same role on disjoint input halves. Between-role JSD must exceed 3× the within-role floor to count as "different" (Martin's G2 routing criterion).
- **Pass:** role-ID accuracy ≥ 40% (the G2 text threshold applied to routing, chance 25%) **and** between-role JSD ≥ 3× floor. **Null:** accuracy within the CI of chance, or JSD within the floor. Either outcome is a result: pass means role is already in the routing before training (routing is a probe for *prompted* role); null means E1's signal, if any, is created by LoRA.
- **Controls:** (i) shuffled role labels (must fall to chance); (ii) no-role arm vs each role (does "has a role at all" separate before "which role"); (iii) text-only baseline: the same classifier on bag-of-words TF-IDF of the generated text, so we can say whether routing carries role information beyond what the words already show. Mixed-model and placebo controls are not applicable here (single model, no intervention) and are deferred to E1.
- **Confound to report, not fix:** the role system prompt sits in the KV cache for every generated token, so some routing shift is mechanically expected. The interesting number is the *generated-token* classifier relative to the *prompt-token* classifier (prompt tokens trivially identify the role because the text differs).

## What E0 buys the GA paper

A free per-agent signal: if role is readable from routing on the base model, GA probes on the gpt-oss coding swarm can use routing histograms without any training (compare Risa 0.940). If it is not, that is a boundary on Risa's result for role-as-prompt rather than role-as-behaviour.

## Logging per run

model id + revision, chat template hash, transformers/torch versions, partition JSON + hash, seeds (subset, generation), batch size, max_new_tokens, GPU, wall clock, all three sanity-test results. Results to `results/E0/*.json`.

## Budget

Profiling 2k prompts (prefill only): ~5 min. E0-a: 9 arms × 500 items × ≤400 tokens ≈ 1.8M tokens ≈ 15–30 min at batch 128. E0-b: 5 arms × 200 × 128 ≈ 0.2M tokens, plus a second batch-shape pass. Total well under 2 H200-hours. Pod is already running; no new spend requested.
