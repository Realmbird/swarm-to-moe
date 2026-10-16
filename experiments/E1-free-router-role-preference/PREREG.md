# E1 — Does a free router recognise LoRA'd role experts? (swarm → MoE, first experiment)

**Status:** pre-registered 2026-10-16. **Blocked on:** Martin's four LoRA'd role-agent checkpoints and the partition they were trained under (private repo `agencyenterprise/aiaf-fellowship-p3-role`, folder `AIAFC-118-moe-swarm-gpt-oss-20b`). Do not run before E0 passes its sanity gates.

## Hypothesis

Martin's agents were "different inside, same outside" before LoRA; after LoRA they hold roles under prompt injection (§2). If role behaviour was written into the private experts, a router that is *free* over all 32 experts should send role-r-prompted tokens to role r's private experts more than to other roles' private experts, without any per-row rule telling it to.

- **H1:** role preference (defined below) is > 0 with 95% paired bootstrap CI excluding 0, in the merged LoRA model, for at least the late layer group (12–23).
- **H0 (eaten by the trunk):** preference ≈ 0: LoRA changed what the experts *do* but the router, which was not trained, has no reason to pick them; role lives in the forced routing, not in a recognisable expert signature.

Either answer is a result for the GA paper. H1 ⇒ routing is a role probe that survives removing the control rule (behaviour lives in the agents). H0 ⇒ the role is a property of the enforced *assignment*, i.e. of the communication/control layer, not of the weights.

## Construction

1. Merge: the four agents share all non-expert weights and have disjoint private expert sets, so the merged checkpoint = base weights with each role's LoRA applied to its own private experts. One checkpoint, one forward pass.
2. Routing: `FreeRouting` (no per-row rule). Record top-4 per layer per token.
3. Inputs: the same role prompts × task inputs as E0-b (roles replaced by Martin's actual four), greedy, 128 new tokens.

## Metric

For every (role r, input) sequence, over generated tokens, per layer:
`match` = fraction of top-4 slots in r's private set, `other` = fraction in the other three roles' private sets, `shared` = remainder.
**Role preference** = `match − other/3` (private sets are equal-sized so this is 0 under role-blind routing). Aggregate per layer group (0–11, 12–23) and overall, with paired bootstrap CI over inputs.

Secondary: 5-fold role-ID accuracy from routing histograms (chance 25%); between-role routing JSD vs within-role noise floor; GSM8K/MMLU retention of the merged free-routed model vs base (does removing the rule break the agents?).

## Arms and controls

| arm | model | routing | what it tells us |
|---|---|---|---|
| merged-free | merged LoRA checkpoint | free | **the test** |
| base-free | plain gpt-oss-20b, same partition labels | free | E0-b; preference here is the partition's built-in bias, subtract it |
| merged-shuffled | merged LoRA | free, role labels permuted (derangement) | must be ≈ 0 |
| merged-rule | merged LoRA | HardSplit / SoftNudge with correct agent per row | Martin's configuration; upper reference for retention and for routing separation |
| merged-norole | merged LoRA | free, no system prompt | is preference driven by the prompt or by the task text |
| placebo | base + random LoRA noise of matched norm on the same private experts | free | rules out "any perturbation of private experts attracts the router" |

Repeated seeds: 3 seeds for subset selection; two batch shapes per arm for the routing noise floor.

## Pass / fail

**Pass (H1):** merged-free preference in layers 12–23 > base-free preference, CI excluding 0, and merged-shuffled ≈ 0, and placebo ≈ 0. **Fail (H0):** merged-free CI includes 0 or is indistinguishable from placebo. **Ambiguous:** preference > 0 but role-ID from routing < 40%: the router leans toward the right experts but not enough to identify the agent; report as partial.

## Ties

Martin's Oct 12 spike (attention heads tracking roles on gpt-oss-20b and qwen3-32b); Risa (role-from-routing 0.940 on gpt-oss coding agents); Branch-Train-MiX (learned router over branched experts; here we ask what an *unlearned* router does first, since a learned router would trivially pass).

## Cost

~1 H200-hour once checkpoints are in hand. Safety note (§3): keep safety-linked experts shared; the merge touches only private experts.
