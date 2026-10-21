# Evaluation-suite analysis → what to build for an "ideal" suite

Audit of scenario coverage, evaluation methodology, and domain coverage,
with the gaps ranked by how much they weaken the lab's claims.

## 1. Scenario coverage gaps

Current scenarios (6) and the drift types each actually exercises:

| scenario | K | T | drift types covered | domains |
|---|---|---|---|---|
| topic | 3 | 60 | topic | sql only |
| workflow | 2 | 40 | workflow | toolcall only |
| domain | 3 | 60 | domain, topic, workflow | science, toolcall, sql |
| compound | 4 | 80 | topic, workflow | sql, toolcall |
| healthcare | 1 | 20 | domain (single family) | healthcare |
| trading | 1 | 20 | domain (single family) | trading |

**Gaps**

| # | Gap | Why it matters |
|---|---|---|
| S1 | **Three of five environments have only one family** (science, healthcare, trading). A single family cannot produce drift, so `healthcare` and `trading` measure *learning*, not *continual learning*. | The domain playbooks for healthcare and trading claim to be validated against drift, but their streams are single-family. The healthcare playbook's text says "measured I=0.00, F=0.00 on the triage stream" — trivially true when there is no second family to interfere with. |
| S2 | **The paper's four scenarios are not the lab's four scenarios.** The paper's are topic / workflow / domain / compound; the lab's `compound` is topic+workflow with no domain shift, and the lab's `domain` is the only true domain shift. | A reader comparing to the paper cannot map the lab's streams onto the paper's. |
| S3 | **No scenario mixes a domain shift with a workflow shift.** The hardest case — the environment changes *and* the required workflow flips — is never tested. | Compound drift is where the paper's failure modes compound (Fig. 4b: BFCL collision inside a drifting stream). |
| S4 | **No "revisit" scenario.** Every family appears exactly once. The paper's "before a group's block begins, 100% of retrievals come from earlier blocks" needs a family that is *seen, then not seen, then seen again* to measure return-to-competence. | Cannot measure whether an agent recovers a previously learned family. |
| S5 | **No curriculum/ordering control.** Stream order is fixed. | A lab that claims "this evolver is better" should show the claim survives reordering — otherwise the result is an artefact of one ordering. |
| S6 | **No single-family control stream for every domain** — only healthcare and trading have one. Science, SQL, toolcall do not. | Every domain should have a "learn one family" baseline so per-domain plasticity is comparable. |
| S7 | **No adversarial scenario**: two families with the *same* wording but opposite required behaviour beyond BFCL. | BFCL is the only retrieval-collision case in the lab. |

## 2. Evaluation methodology gaps

| # | Gap | Why it matters |
|---|---|---|
| E1 | **No forward-transfer metric.** The paper's central positive result is that an agent can carry competence from earlier blocks into later ones. We measure P, I, F, S, Term., Δ̄ — none of which isolate "learned here, used there". | Without forward transfer the lab cannot support its own headline claim (scoped learning helps) with the mechanism the paper attributes it to. |
| E2 | **No backward-transfer metric.** Forgetting is measured as a drop, but "positive backward transfer" (getting *better* on an old family after learning a new one) is a distinct and important effect. | The lab's `scoped` evolver should show exactly this (retained competence) and nothing measures it. |
| E3 | **No area-under-curve / learning-curve summary.** Δ̄ averages all (checkpoint, family) cells equally, so a run that spikes early and collapses late scores like a flat one. | The headline number hides the shape of the run. |
| E4 | **No significance testing.** A single run's +2.0 is indistinguishable from noise. Bands exist, but there is no test. | "Is +2.0 real?" is the first question any user asks. |
| E5 | **No confidence interval on the bands.** Bands publish mean ± std but not n, not the interval, and not a comparison test between two cells. | Cannot say "scoped is better than reasoningbank" with a number attached. |
| E6 | **No per-domain verdict.** The metrics are averaged over families; a method can look good overall while being worse in healthcare. | The user cares about *their* domain. |
| E7 | **Guards are documented but not enforced at the action level.** The trading cap ("the risk engine disposes") is checked by the grader, but the policy can still *learn* and replay an over-sized position; the healthcare "never diagnose" rule is only checked on the final text. | A guardrail that only the grader sees is a measurement, not a safety property. |
| E8 | **The benchmark docstring promises a `--refresh-benchmarks` flag that does not exist.** | Documented behaviour that does not work. |
| E9 | **Single metric scale, no ranking.** The user gets six numbers and no answer to "which evolver should I use?" | The suite should be able to recommend. |

## 3. What to build

### 3.1 New task families (enables the new scenarios)

| family | env | drift | purpose |
|---|---|---|---|
| `health-triage-pediatric` | healthcare | domain | second triage family with different acuity vocabulary (paediatric dosing, dehydration) — makes healthcare a real drift stream |
| `health-triage-mental` | healthcare | domain | third family: behavioural/psychiatric presentation with a different escalation rule |
| `trade-signal-macro` | trading | domain | second trading family: macro/correlation regime rather than breakout/breakdown |
| `trade-signal-meanrev` | trading | domain | third family: mean-reversion regime — near-duplicate wording with the opposite action (a second retrieval-collision case outside BFCL) |
| `science-assay-kinetics` | science | domain | second science family: kinetics vs. property assay, different instruments |

### 3.2 New scenarios

| scenario | shape | measures |
|---|---|---|
| `healthcare-drift` | health-triage → pediatric → mental | domain drift inside a regulated domain; escalation rules must not be overwritten by routine cases |
| `trading-drift` | trade-signal → macro → meanrev | regime drift; breakout rules must not fire on mean-reversion |
| `science-drift` | science-assay → science-assay-kinetics | instrument/protocol drift |
| `domain-workflow` | science → toolcall-clarify → toolcall-act | **domain shift AND workflow flip** — the case the lab never tested (S3) |
| `revisit` | sql-f1 → toolcall-act → sql-f1 | **return to a seen family** — measures recovery (S4) |
| `permutation` | topic with 3 orderings | **ordering control** (S5) |
| `single-<domain>` | one family per env | per-domain learning baseline (S6) |

### 3.3 New metrics

- **Forward transfer** FT: mean over families k of (A[τ_k,k] − A[σ_k,k]) where the gain comes from rules learned in *earlier* blocks — i.e. how much earlier learning helped a family it had never seen. Computed as the gain in the first checkpoint of k's block that cannot come from k's own block.
- **Backward transfer** BT: mean over families k of (A[T,k] − A[τ_k,k]) — positive means the agent got *better* on an old family after learning new ones.
- **Area under the advantage curve** AUC: the shape of Δ̄ over the horizon, normalised — distinguishes "spiky" from "sustained" learning.
- **Learning-curve summary**: per-family accuracy at every checkpoint as a compact series.
- **Verdict + significance**: a per-cell verdict against the bands, a two-sided test between two cells, and a recommended evolver per (stream, domain).
- **Guardrail audit**: whether any *learned* action violated a declared guardrail (trading cap, triage escalation, no-diagnosis) — the difference between "the grader caught it" and "the agent never proposed it".

### 3.4 Coverage report

A machine-readable matrix: scenario × drift type × domain × evolver, with
which cells are measured and which are missing, surfaced in the API
(`/api/coverage`) and the **Coverage** tab of the dashboard.

## What was built

| Gap | Fix | Where |
|---|---|---|
| S1 | 5 new task families: paediatric triage, mental-health triage, macro trading, mean-reversion trading, kinetics science | `app/envs.py` |
| S2 | Scenarios relabelled to the paper's taxonomy, with the drift type each actually exercises | `app/streams.py` |
| S3 | `domain-workflow` — environment change *and* workflow flip in one stream | `app/streams.py` |
| S4 | `revisit` — a family seen, interrupted, then seen again; `Stream.first_tau` and `Stream.revisits` | `app/streams.py` |
| S5 | `topic-order-a/b/c` — the same three databases in three orders | `app/streams.py` |
| S6 | `sql`, `toolcall`, `science` single-family controls (healthcare and trading already had them) | `app/streams.py` |
| S7 | The mean-reversion trading family is worded like momentum and needs the opposite action — a second retrieval-collision case outside BFCL | `app/envs.py` |
| E1 | `forward_transfer` — gain before a family's own block | `app/metrics.py` |
| E2 | `backward_transfer` — change after a family's own block | `app/metrics.py` |
| E3 | `area_under_curve` + a per-checkpoint `learning_curve` | `app/metrics.py` |
| E4 | Welch t-test on every cell comparison, with a p-value | `app/stats.py` |
| E5 | 95% confidence intervals on the difference, and `n` on every cell | `app/stats.py` |
| E6 | Per-stream, per-metric ranking and a recommendation that states its caveats | `app/stats.py` |
| E7 | `envs.guardrail_violations` — audits the *proposed action*, not just the graded answer; reported per run | `app/envs.py`, `app/runner.py` |
| E8 | `tests/sweep.py --refresh-benchmarks` now exists and writes the bands in the format the app loads | `tests/sweep.py` |
| E9 | `GET /api/stats/recommend` — "which evolver should I use" with evidence and caveats | `app/stats.py`, `app/main.py` |

## The evaluation bug this work exposed

The generic "wrong answer" the simulated policy falls back to was, for some
families, accidentally *correct*:

| family | declared competence | measured seed rate (before) | measured (after) |
|---|---|---|---|
| trade-signal-macro | 0.60 | **1.00** | 0.59 |
| health-triage | 0.54 | **0.66** | 0.54 |
| trade-signal | 0.68 | **0.85** | 0.70 |
| trade-signal-meanrev | 0.64 | **0.82** | 0.62 |

A "flat" proposal is the right answer for a flat market, and "acuity 3" is
the right answer for a standard case. The seed rate inflated, the headroom
to detect learning vanished, and `trading-drift` measured P ≈ +1 for every
method — the evaluation could not see learning even when it happened.

The fallback now avoids the gold: it picks a direction that is *not* the
gold's, and an acuity that is *not* the gold's. Every family's measured
seed rate now matches its declared competence to within 6.5pp, and two
tests hold that invariant (`test_base_competence_matches_the_declared_rate`,
`test_generic_fallback_is_actually_wrong`).

This is the kind of bug an evaluation suite exists to catch, and it is why
the single-family streams are worth having: they are the control that
exposes it.
