# CLAUDE.md — MoE ⇄ agent-swarm project

Context file for Claude sessions working on this project. Read fully before touching code or proposing experiments. Keep it current: when a result lands or a decision is made, update the relevant section here rather than leaving it in chat.

## What this project is

Collaboration between Christopher Kinoshita (owner of this repo) and Martin Dinov (AI Alignment Foundation fellowship, p3-role project with Andi Bhongade and John Lund). Two linked questions:

1. **MoE → swarm** (Martin's side quest, Oct 2–7 2026): can one pretrained MoE run a swarm of agents by giving each agent its own experts? Built and measured on gpt-oss-20b. Verdict so far: technically works, no compute benefit, remaining value is behavioural/interp (see §3).
2. **Swarm → MoE** (Christopher's direction): if the mapping runs one way it should run the other. Treat a swarm of role agents as a mixture of experts and ask whether swarm dynamics (goal adoption, role confusion, deference) survive compression into routing dynamics. See §4.

Both feed Martin's main project: measuring goal adoption (GA) in multi-agent swarms across two environments (DisclosureWorld, ImpossibleBench-style coding swarm), with activation/text probes and a paper due ~early Nov 2026. Anything we do should be framed in terms of what it buys that paper.

## 1. The construction (Martin's design, gpt-oss-20b)

- gpt-oss-20b: 24 layers × 32 experts, top-4 per token, independent router per layer, 3.6B active.
- Agent = own KV cache (its own batch row) + a private stripe of experts in every layer + shared attention / embeddings / shared expert pool. A lone expert is NOT an agent (stateless FFN, no attention, no cross-layer identity). Do not re-derive this; it is settled.
- Partition per layer: 16 shared + 4 agents × 4 private. Chosen by profiling the base model on 2k prompts: super experts (top 1% by peak output) and most-used experts → shared; remainder dealt to agents snake-order, balanced by usage. Three partitions defined: P-all, P-late (layers 12–23 only), P-rand (control).
- Hard split: top-3 from shared pool + top-1 from own private set. Compute per token unchanged.
- Soft nudge: +δ·σ_l on own private experts' router scores, −δ·σ_l on other agents'; mixing weights from unbiased scores. δ=0.5 is the run configuration.
- Implementation: hook replacing the MoE forward in HF transformers (one level above the router, where batch rows are known). Quantised kernels bypass Python routing → weights loaded dequantised in bf16. One H200. ~3k tok/s at batch 128; hook overhead 7.6–15.7%.
- Sanity tests before every phase: hook-off reproduces base tokens exactly; rule holds on every token; forcing a named expert changes output (proves hook is live). Keep these.
- vLLM/SGLang (as of Oct 2026) do not support per-request expert masks or router biases; only route logging. Scaling beyond HF means patching vLLM or following ExpertWeave.

## 2. Measured results (Oct 5–6, 2026) — the real numbers

The paper draft (moe_swarm_paper.docx, Oct 5) is placeholders except base accuracies and the all-layer hard-split GSM8K number. Trust the figures posted in Slack #03-roles on Oct 6, not the draft's tables.

| | plain copies | soft nudge δ=0.5 | hard split (all layers) |
|---|---|---|---|
| GSM8K retention (4 agents) | 100% | 99–101% | 91 / 77 / 66 / 82% |
| MMLU retention | 100% | 98–100% | 87–90% |
| ARC retention | 100% | 98–99% | 97–98% |
| Top-4 expert overlap, same text (Jaccard) | 1.00 | 0.70 | 0.57 |
| Error agreement on GSM8K (kappa) | 0.82 | 0.61 | 0.41 |
| Identify agent from routing | — | 100% | 100% |
| Identify agent from text (chance 25%) | 25% | 27% | 32% |

- **G1 gate** (≥90% retention every agent, both tasks): soft passes trivially; hard fails (agent C, maths).
- **G2 gate** (3× routing-JSD noise floor AND ≥40% text-ID): routing half is built in; text half fails for both rules before LoRA. "Different inside, same outside."
- **Team tasks** (soft nudge only, 5 teams of 4, one seed, 20 tests, McNemar+Holm): null. Debate 90 vs 89%; assigned debate 95 vs 94%; split-evidence QA 42 vs 45%; IPD cooperation 18 vs 14%. Only effect surviving correction: soft-nudge+roles reaches fewer negotiation deals (39% vs 56%). Mixed team (3 agents + Qwen3-8B) did not stand out.
- **All-layer hard split GSM8K**: 0.948 → 0.748 (79%). P-late (layers 12–23) result is unconfirmed (draft claims 93%, marked placeholder).

**LoRA run** (reported in Oct 7 meeting notes by Andi; details not yet seen)

- LoRA on the 4 sub-agents' private experts completed. Without LoRA agents switch roles under prompt injection; with LoRA they hold roles, behave as 4 distinct agents, "performance improves".
- H200 capacity: 32 plain 20Bs per box vs 128 with 4× MoE parallelisation (120B ≈ 6–7 per box). Open tension: this is batch/KV capacity, not FLOPs; Martin's own verdict is "no compute benefit". Get the actual numbers and reconcile before citing either.
- Not yet known: post-LoRA text-ID accuracy, post-LoRA G1 retention, what "performance" means.

## 3. Standing conclusions (do not relitigate without new data)

- Expert partitioning gives no compute or scaling benefit over serving N copies from one weights file with N KV caches (what vLLM/SGLang already do). The only savings are one-file memory, which a plain swarm already gets.
- Untrained expert agents are modes of one model: they differ in routing and in which mistakes they make, not in what they write. The shared trunk dominates.
- Remaining value of MoE→swarm is control and measurement: (a) roles baked into experts via LoRA may resist prompt override better than prompted roles; (b) routing is a free per-agent signal (Risa: role-from-routing at 0.940 on gpt-oss coding agents); (c) a MoE-as-swarm may be a cheaper, more predictable model organism for swarm behaviour than N live agents. (c) is the hypothesis worth testing, not asserting.
- Same-model swarms are a monoculture; always keep a mixed-model baseline and a noise placebo.
- Expert masks remove safety far more easily than they add it. Keep safety-linked experts in the shared pool unless deliberately testing their removal.

## 4. Christopher's direction: swarm → MoE

Three readings; be explicit which one is meant:

1. **Agent-level sparse routing** (orchestrator picks k of N agents). Exists (MoA, Symbolic-MoE, Kimi). Not our contribution.
2. **Distillation**: N same-base role agents merged as experts under a learned router (Branch-Train-MiX recipe; Latent Agents distils debate into one dense model). The inverse of §1.
3. **Swarm dynamics as routing dynamics**: can GA / role confusion / deference be read off a router instead of transcripts? This is the framing that connects to the paper.

Mirror question to Martin's result: his agents were different inside, same outside. Take agents that are plainly different outside, compress them into one model, and ask whether the diversity survives into router + private experts or gets eaten by the shared trunk. Either answer is a result for the GA paper (behaviour lives in the agents vs. behaviour lives in the communication).

Known constraints: an MoE forward pass is one token stream, a swarm is many turns of talk — either collapse communication into the residual stream (committee mode; Martin rates it ablation-only) or keep per-agent KV caches with cross-attention (Hogwild-style), which is §1 with trained experts. Only same-base agents can be merged; mixed-model swarms (DeepSeek + gpt-oss) have no shared expert dimensionality.

**First experiment (near-free, uses existing artifacts)**

1. Take Martin's four LoRA'd role agents. Private expert sets are disjoint → merge into one checkpoint (his notes say this is trivial).
2. Remove the per-row routing rule. Let the base router choose freely over all 32 experts.
3. Feed role-prompted inputs. Measure: fraction of tokens routed to the matching role's private experts vs. other roles' vs. shared, per layer. Compare to the unmerged base model and to a shuffled-role control.
4. If the free router prefers the matching role's experts → the router recognises roles → routing is a role probe. Ties directly to Martin's Oct 12 spike (attention heads tracking roles on gpt-20b and qwen3-32b) and to Risa.

Cost: ~1 H200-hour. Prereqs: access to the private repo + LoRA checkpoints. Pre-registered as **E1** (`experiments/E1-free-router-role-preference/PREREG.md`); stays blocked on Martin's checkpoints.

**Decision 2026-10-16 (Christopher):** do not wait on Martin's artifacts and do not spend time reproducing his construction. The reproducibility check (E0-a) is dropped; E0-b (role identifiability from free routing on the base model, `experiments/E0-base-routing-by-role/PREREG.md`) is kept because it is the control arm. The main line is **E2** (`experiments/E2-swarm-to-moe-self-distilled/PREREG.md`): build four agents that are plainly different outside by self-distilling role-prompted outputs into LoRA on each role's private expert stripe (HardSplit on during training, router frozen), then switch the rule off and ask whether the free router finds the roles and whether role identity survives in text. Pass/fail and controls (placebo LoRA of matched norm, shuffled labels, base-free, prompted-roles-on-base) are in the pre-registration. Infra: `swarm_to_moe/` (hook, partition, metrics, lora, distill) and `scripts/e2_*.py` in this repo; runs on the project's Runpod H200 (`/workspace` HF cache). E3 (goal adoption as routing convergence before text) follows if E2 shows the router finds the roles.

## 5. Infra, budget, logistics

- **Compute**: Runpod (H200). Martin also has a DGX Spark at home. Project Runpod credits were nearly exhausted as of Oct 8; Martin requested a top-up from John. Check balance before any run; ask before anything over ~$50. Side quest to date: ~$70 + ~$100 LoRA.
- **Repo**: agencyenterprise/aiaf-fellowship-p3-role (private). Experiment folder AIAFC-118-moe-swarm-gpt-oss-20b. Christopher invited under GitHub account Realmbird on Oct 5. Scripts referenced in draft: scripts/gates.py, scripts/summarize.py; results in results/G1.json, results/G2.json, results/phase_b/*.metrics.json, results/phase_c_*.json.
- **Project conventions** (from John's review): pre-registrations committed before the runs they govern; one question-shaped issue per experiment matching its folder and notebook; finished work must reach main (a branch 117 commits ahead was flagged). Follow these.
- **Reproducibility**: pin model revision + chat template; seed eval subsets; profile on train/val splits only; record config, partition hash and seeds per run. Routing is not bit-reproducible across batch shapes → route-based measures need repeated runs; avoid batch-aware routing optimisations.
- **Coordination**: Slack DM with Martin (D0C02V2PJVB) and #03-roles (C0C179797NU). Martin is in Europe/London; syncs at 7–8am Pacific. Next sync: Fri Oct 16, 7am PT — Christopher owes the calendar invite.
- Christopher's Runpod MCP is not connected in Cowork sessions; use runpodctl with an API key or connect the MCP first.

## 6. Key references (full list in Martin's MoE swarm design notes.md, Oct 2)

Reverse mapping: Multi-Agent Systems are Mixtures of Experts (2605.25929); Bause et al. 2026 deliberation-as-MoE; Latent Agents (2604.24881). Merging/experts-from-models: Branch-Train-MiX (2403.07816); FlexOlmo (2507.07024); ESFT (2407.01906); MixLoRA (2404.15159). Closest forward-direction prior work: Mixture of Debaters (2606.29425); MoRSE (2608.09251); E^3 (2110.03360). Shared cache / latent comms: Hogwild! (2504.06261); LatentMAS (2511.20639); When Latent Agents Lie (2606.28958). Routing as signal: Risa (2608.22191); SteerMoE (2509.09660). Restriction costs: super experts (2507.23279); EMO (2605.06663). Monoculture: Bertalanic & Fortuna (2606.02646); Self-MoA (2502.00674).

## 7. How to work in this repo

- Check §2–3 before proposing an experiment; if it re-tests a standing conclusion, say why.
- Every experiment gets a one-paragraph pre-registration (hypothesis, metric, pass/fail, controls: plain copies, mixed-model team, noise placebo, repeated seeds) committed before the run.
- Report retention relative to the base model on the same items, with paired bootstrap CIs; use McNemar + Holm for paired comparisons. Chance baselines on every classifier plot.
- Measure behaviour, not perplexity (near-uniform routing on gpt-oss-20b makes perplexity uninformative for restricted agents).
- When a result lands, update §2 (numbers), §3 (conclusions) or §4 (plan) here in the same PR.
- Model routing for Claude sessions: use Fable for planning and tests; use Opus for everything else.
