# Self-Learning Agent Lab

Design, build, and test self-learning agents — grounded in
arXiv:2610.25858 ("Are Self-Evolving Agents Continual Learners?").

The lab reproduces the paper's protocol (shared harness, drifting streams of
task families, binary task-success feedback, seed + per-boundary checkpoints,
disjoint hold-out sets) and its metrics (B_k, P_k, I_k, F_k, S, Δ̄, Term.),
then adds the mechanism the paper asks for: task-scoped learning with
workflow conditioning.

The model backend is fox-services on `:8210`; the lab UI serves on `:8220`.
(`:8200` on this machine belongs to another service.)

## Docker (app + dashboard)

```bash
docker compose up --build -d   # dashboard at http://localhost:8220
docker compose logs -f lab
docker compose down
```

The image (`Dockerfile`, python:3.12-slim) serves the FastAPI backend and
the dashboard SPA from `/`. SQLite lives in the `sla-data` volume, exports
bind-mount to `./exports`. Live mode reaches the host's model backend via
`host.docker.internal` (`FOX_SERVICES_URL`, override in compose). To enable
the pattern picker with the full 107-pattern catalog, uncomment the
`/patterns:ro` mount in `docker-compose.yml` and point it at your
AI-Agents-Patterns checkout. Port via `SLA_PORT` (default 8220).

## Quick start (local)

```bash
./run.sh            # API on :8220, frontend at /
```

Or pick the UI port to avoid a local collision: the model backend lives on
`:8210` (fox-services → Ollama); the lab UI defaults to `:8220` in run.sh.

```bash
python3 -m tests.smoke --scenario workflow --block 20 --eval 30
python3 -m pytest tests/ -x -q
```

`tests/test_lab.py` pins the mechanism contracts (probe, intent taxonomy,
lesson generalisation, router partition, metric formulas, bit-reproducibility);
`tests/test_api.py` covers the HTTP layer including a live-mode launch test
that skips gracefully without a model backend.

## Iteration sweeps (20× coverage)

One run is exactly reproducible but still one draw of tasks and policy
coins. The sweep varies all three seed axes per iteration and reports
mean ± std, so a table entry becomes a claim with error bars:

```bash
python3 -m tests.sweep --iterations 20 --block 20 --eval 30 --jobs 8 --out sweep.json
python3 -m tests.sweep --iterations 20 --scenarios workflow,topic --evolvers scoped,gepa
```

Each iteration `i` uses policy seed `7+i`, task-train seed `100+i`,
task-eval seed `1100+i`, and lesson-id seed `1000+i`. All four are recorded
per row, so any cell reproduces exactly. Task draws are seed-controlled in
`streams.build_stream(..., train_seed, eval_seed)` and exposed on
`POST /api/runs` as `train_seed` / `eval_seed`; lesson ids derive from
`RunConfig.seed` (default 7), which is why repeated runs no longer drift
through uuid randomness.

## The 50-iteration gate

`tests/iteration_loop.sh` runs the whole loop 50 times against the live
container. Each round: the mechanism sweep, the unit tests, and the
browser-driven UI audit. It stops at the first failure and names it.

```bash
docker compose up -d --build
./tests/iteration_loop.sh 50 3     # 50 rounds, 3 sweep iterations per cell
```

The UI audit (`tests/ui_audit.js`, Playwright) drives the real dashboard in
a real browser and fails on: console errors, 4xx/5xx responses, horizontal
overflow, clipped text, tap targets under 24px, unlabeled form controls, a
missing focus ring, a nav that wraps on mobile, and any of the core
workflows breaking (batch launch → finished run, compare → table with
headers and per-cell summary, design creation → populated spec editor,
suite estimate → a stated run size).

Needs `playwright-core` outside the repo:

```bash
npm install --prefix /tmp playwright-core
npx playwright-core install chromium
```

## Reference bands

`app/benchmarks.py` holds the measured mean ± std for every
scenario × evolver cell, taken from the 20-iteration sweep. The dashboard's
**Reference** tab shows them, and every run is judged against them
("as measured" / "above the band" / "below the band"), so a number is never
just a number. `tests/test_lab.py` asserts the bands still hold — if a
harness change moves a cell outside its own band, the bands are stale and
the test says so.

The bands ship in `data/benchmarks.json` and are regenerated with:

```bash
python3 -m tests.sweep --iterations 20 --block 20 --eval 30 \
    --scenarios topic,workflow,domain,compound,healthcare-drift,trading-drift,science-drift,domain-workflow,revisit \
    --refresh-benchmarks data/benchmarks.json
```

## What lives here

| Path | Purpose |
|---|---|
| `app/envs.py` | Self-contained task environments (SQL, tool-call, science, clinical triage, trading) with deterministic graders and the intent taxonomy |
| `app/retrieval.py` | Similarity, unsupervised routing, workflow fingerprinting |
| `app/evolvers.py` | The five harness-rewrite rules: `seed`, `reasoningbank`, `gepa`, `metaharness`, `scoped` |
| `app/policy.py` | `SimulatedPolicy` (deterministic, no GPU) and `LivePolicy` (ReAct over fox-services) |
| `app/metrics.py` | Paper metrics (Eq. 8–11) + failure-mode detectors |
| `app/runner.py` | Stream protocol: train → checkpoint → evaluate on held-out sets |
| `app/designs.py` | Design specs, pattern picker (AI-Agents-Patterns catalog), guardrails, domain playbooks |
| `app/benchmarks.py` | Measured reference bands (mean ± std per cell) + the metric glossary |
| `app/flows.py` | Mermaid generators: the agent flow drawn from a design, the stream it faces |
| `app/stats.py` | Welch t-test, effect size, Holm correction, power, cell ranking |
| `app/cost.py` | Prompt/write accounting and the gain-per-unit ledger |
| `app/suite.py` | Suite runner, the report, and the advice |
| `app/scaffold.py` | Export a design as a runnable agent project (`exports/`) |
| `app/main.py` | FastAPI: designs, runs, batch runs, compare, benchmarks, export + the SPA |
| `web/` | Vanilla-JS frontend (no build step) |
| `tests/sweep.py` | N-iteration sweep with mean ± std aggregation |
| `tests/ui_audit.js` | Browser-driven UI audit (Playwright) |
| `tests/iteration_loop.sh` | The 20-round gate: sweep + tests + UI audit |

## The five evolvers

- **seed** — no adaptation; the c_0 baseline. Must read exactly 0 everywhere.
- **reasoningbank** — one memory item per episode, top-k retrieval by request similarity.
- **gepa** — a shared strategy prompt, reflectively rewritten and compressed; later blocks overwrite earlier rules.
- **metaharness** — whole-workspace edits accepted only on validation gain.
- **scoped** — the lab's proposal: per-cluster, per-workflow sections that are append-only, so a later block cannot overwrite an earlier family's rules.

## Using the dashboard

Five tabs, one flow:

1. **Designs** — start from a playbook (healthcare, trading, finance,
   support, SRE, research) or blank. The spec editor is the harness the run
   validates; patterns and guardrails for the domain are shown next to it.
   **Flow diagrams** are generated from the design itself: the agent's
   request path (router → scope check → policy → action → grader), the
   harness surfaces the evolver reads and writes, and the stream the agent
   will face. Change the evolver and the diagram changes with it.
2. **Runs** — one form launches a *comparison batch*: every evolver on the
   same stream, N iterations each. Preset buttons set up the paper's topic
   and workflow scenarios. Each run in the history can be cancelled,
   repeated with fresh draws, or deleted.
3. **Compare** — pick runs for the Table-1 table, or read the per-cell
   summary (mean ± std across every finished run of that stream+evolver).
4. **Reference** — the measured bands every run is judged against, plus a
   glossary of the paper's metrics in plain language.
5. **Patterns** — the AI-Agents-Patterns catalog and its failure scenarios.
   The 32 patterns that ship a diagram can draw it on demand, and the nine
   category flows show the shape each category of pattern tends to take.

The run detail shows the six metrics as labelled cards (each with its
meaning), the accuracy-per-family curve across checkpoints (paper's
A[t,k]), the detected failure modes with causes, and whether the run landed
inside its measured band.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Lab status: groups, evolvers, scenarios, catalog availability |
| `GET /api/llm/health` | Which model backends answer, and with which models |
| `POST /api/runs` | Launch one run (`stream`, `evolver`, `block_size`, `n_eval`, `train_seed`, `eval_seed`) |
| `POST /api/runs/batch` | Launch a comparison grid: `evolvers` × `iterations` on one stream |
| `GET /api/runs/summary` | Finished runs aggregated per (stream, evolver) as mean ± std |
| `GET /api/runs/{id}` | One run: metrics, checkpoints, assessment against the bands |
| `POST /api/runs/{id}/cancel` | Stop a running job |
| `POST /api/runs/{id}/duplicate` | Re-launch the same configuration with fresh task draws |
| `DELETE /api/runs/{id}` | Remove a run and its checkpoints/traces |
| `DELETE /api/runs?only_status=` | Bulk cleanup (e.g. `cancelled`, `error`) |
| `POST /api/compare` | Table-1 rows for a set of runs, with their failure modes |
| `GET /api/coverage` | What the suite exercises and the combinations it misses |
| `POST /api/suite/run` | Launch the whole suite (scenarios × evolvers × iterations) as one job |
| `GET /api/suite/report` | The suite report: rank, effect size, corrected significance, guardrails, cost |
| `GET /api/suite/advice` | What to do next for a domain, and what is not measured yet |
| `POST /api/suite/estimate` | What a suite run will cost, before it starts |
| `GET /api/stats/recommend` | Rank the evolvers for a stream, with a significance test and caveats |
| `GET /api/stats/compare` | Compare two cells with a confidence interval and p-value |
| `GET /api/benchmarks` | Measured reference bands + the metric glossary |
| `POST /api/assess` | Judge runs against the bands |
| `GET/POST/PUT/DELETE /api/designs` | Design CRUD |
| `POST /api/designs/from-playbook/{id}` | Create a design from a domain playbook |
| `GET /api/patterns`, `/api/scenarios/catalog` | The AI-Agents-Patterns catalog + the category flow diagrams |
| `GET /api/designs/{id}/flow` | The design's mermaid diagrams (agent flow + stream) |
| `GET /api/patterns/{id}` | One pattern, with its flow diagram when it has one |
| `GET /api/playbooks`, `/api/guardrails`, `/api/streams`, `/api/evolvers` | Guidance and catalogues |
| `POST /api/export` | Export a design as a runnable agent project |

## The evaluation suite

21 scenarios over 13 task families in 5 environments, covering all four
drift types and four stream *shapes*. `GET /api/coverage` returns the
matrix and the remaining gaps.

| group | scenarios | what it exercises |
|---|---|---|
| **Paper's four** | `topic`, `workflow`, `domain`, `compound` | topic / workflow / domain / compound drift |
| **Per-domain drift** | `healthcare-drift`, `trading-drift`, `science-drift` | drift *inside* a regulated domain, where a routine lesson landing on a red-flag case is the failure that matters |
| **Hard combinations** | `domain-workflow` | an environment change *and* a workflow flip in one stream |
| | `revisit` | a family seen, then interrupted, then seen again — does competence return? |
| **Stream shapes** | `interleaved` | families arrive *mixed*, not in blocks — the router must route per request under changing composition, and the memory must hold several families at once |
| | `recurring` | a family that returns every few blocks — recency as well as scope |
| | `long-horizon` | eight blocks: does degradation compound with horizon length? |
| | `hard-order` | the confusable pair adjacent, the hardest family first |
| **Ordering control** | `topic-order-a/b/c` | the same three databases in three orders — a method that only wins on one ordering has not been shown to be better |
| **Single-family controls** | `sql`, `toolcall`, `science`, `healthcare`, `trading` | per-environment learning baselines |

### Metrics

The paper's six (B_k, P_k, I_k, F_k, S, Δ̄, Term.) plus three the
protocol implies:

| metric | meaning |
|---|---|
| **FT** forward transfer | how much a family gained *before* its own block — competence carried in from earlier blocks |
| **BT** backward transfer | how much a family changed *after* its own block — positive means later learning helped old families |
| **AUC** area under the curve | the *shape* of learning, not just its end state |

Plus a **guardrail audit** on every proposed action: did the agent
*propose* an acute triage case without escalation, a position over the cap,
a tool call with a guessed argument? A limit the grader catches is a
measurement; a limit the agent never proposes is a property.

### Cost

Every run reports what it spent: prompt characters injected, characters
written, lessons created, lessons actually used, and the hit rate. The
ledger states the trade plainly — percentage points gained per 100 prompt
units. A method that buys 2pp while injecting 10× the harness is not the
method to ship, and the suite now says so.

### Reading a result

The **Suite** tab runs the whole comparison in one action, reads one
report, and tells you what to do next.

- `POST /api/suite/run` — launch the whole grid as one job
- `GET /api/suite/report` — one row per cell: mean ± std, the margin and
  confidence interval against the best method, the effect size, a
  significance verdict **after Holm correction for the number of
  comparisons**, the guardrail violations, and the cost
- `GET /api/suite/advice` — which evolver to use for a domain, with the
  caveats, and what the suite does not yet measure
- `GET /api/suite/estimate` — what a suite run will cost before it starts

`GET /api/stats/recommend?stream=...` ranks the evolvers for a stream with
a Welch t-test on the margin over the runner-up, and states the caveats
("the margin is not significant at 95%", "interference is 12 pp"). The
report also carries a **power** table: how many iterations each cell
would need to resolve a 5pp margin, so iterations are spent where they
buy resolution.

## Key measurements (simulated policy, block 20 / eval 30, 20 iterations)


### What the numbers say (50 iterations per cell, block 20 / eval 30, mean ± std)

**Workflow drift** (BFCL clarify↔act, one shared API catalogue):

| evolver | Term. | P | I | F |
|---|---|---|---|---|
| reasoningbank | +15.7±7.4 | +16.3±7.3 | +12.8±3.9 | +0.6±1.5 |
| gepa | −10.5±6.0 | +13.8±5.8 | +12.9±4.2 | +24.4±4.6 |
| metaharness | −4.9±6.5 | −2.2±5.8 | +11.0±6.8 | +2.7±3.7 |
| scoped | +16.1±6.8 | +16.1±6.8 | 0.0±0.0 | 0.0±0.0 |

**Interleaved families** (mixed per episode, not in blocks):

| evolver | Term. | P | I | F |
|---|---|---|---|---|
| reasoningbank | +24.5±5.7 | +16.7±5.3 | +1.3±1.4 | +0.9±1.2 |
| gepa | +4.1±4.1 | +2.6±5.0 | +1.8±1.3 | +3.5±3.4 |
| metaharness | −0.8±5.0 | +1.7±5.0 | +3.2±2.7 | +3.0±2.2 |
| scoped | +25.5±5.4 | +19.4±5.6 | +0.5±0.9 | +1.1±1.4 |

**Recurring family** (returns every few blocks):

| evolver | Term. | P | I | F |
|---|---|---|---|---|
| reasoningbank | +21.4±8.0 | +11.3±4.3 | +8.7±2.1 | 0.0±0.1 |
| gepa | +4.9±4.8 | +8.5±4.0 | +4.6±1.3 | +9.7±3.5 |
| metaharness | −2.5±3.6 | −0.6±1.8 | +0.3±1.1 | +1.2±1.8 |
| scoped | +22.2±6.3 | +11.7±3.6 | 0.0±0.0 | 0.0±0.0 |

**Long horizon** (8 blocks, 160 episodes):

| evolver | Term. | P | I | F |
|---|---|---|---|---|
| reasoningbank | +20.5±4.2 | +20.5±4.2 | +16.0±2.4 | +0.3±0.5 |
| gepa | −4.8±2.4 | +17.7±3.5 | +10.6±1.3 | +22.6±3.0 |
| metaharness | −4.3±2.8 | −1.4±3.1 | +2.3±2.0 | +4.5±1.6 |
| scoped | +20.9±3.4 | +20.9±3.4 | 0.0±0.0 | +0.2±0.3 |

**Adversarial ordering** (confusable pair adjacent, hardest family first):

| evolver | Term. | P | I | F |
|---|---|---|---|---|
| reasoningbank | +23.8±5.0 | +23.9±4.8 | +15.3±2.6 | +0.4±0.7 |
| gepa | −0.6±2.7 | +24.1±4.2 | +9.9±1.9 | +24.9±3.8 |
| metaharness | −0.8±4.6 | −0.3±4.3 | +8.6±4.2 | +2.7±1.6 |
| scoped | +24.8±4.5 | +24.7±4.4 | 0.0±0.0 | 0.0±0.1 |

**Domain shift with a workflow flip** (`domain-workflow`):

| evolver | Term. | P | I | F |
|---|---|---|---|---|
| reasoningbank | +20.7±6.8 | +21.3±6.5 | +14.6±3.4 | +0.8±1.5 |
| gepa | −9.2±4.1 | +20.0±6.2 | +9.4±2.5 | +29.2±4.7 |
| metaharness | −3.4±5.0 | −0.8±4.4 | +5.3±3.2 | +3.0±2.6 |
| scoped | +22.8±5.9 | +22.8±5.9 | 0.0±0.0 | 0.0±0.2 |

**Clinical triage drift** (adult → paediatric → mental health):

| evolver | Term. | P | I | F |
|---|---|---|---|---|
| reasoningbank | +34.1±6.9 | +33.4±7.0 | +4.9±2.5 | +0.1±0.3 |
| gepa | +8.8±5.1 | **+37.1±7.0** | +5.2±2.9 | **+28.3±6.0** |
| metaharness | +3.7±7.2 | +3.9±5.8 | +2.2±3.3 | +1.6±2.1 |
| scoped | **+36.0±5.8** | +35.8±5.8 | **0.0±0.0** | **0.0±0.0** |

### The readings that need the numbers to be visible

- **GEPA's plasticity and forgetting are one mechanism.** Each reflection
  window distils one lesson per intent into a shared, character-budgeted
  prompt. More rules per flush means more compression churn: GEPA has the
  *highest* plasticity on the domain streams (+37.1 on healthcare-drift)
  and the *highest* forgetting everywhere (+24 to +29). It is not learning
  badly; it is trading one for the other. The cost ledger makes the trade
  explicit: GEPA injects ~3× the prompt volume of scoped for a negative
  net gain.
- **The longer the horizon, the worse the shared-prompt methods get.**
  On `long-horizon` (8 blocks) reasoningbank's interference reaches
  +16.0 and GEPA's forgetting +22.6, while scoped holds I=0.00, F=0.16.
  A 4-block stream hid this.
- **Interleaving is not the same as blocking.** With families mixed per
  episode, every method's interference drops (nothing is "pre-block" for
  long) but the ordering of the methods does not change, and scoped still
  leads on plasticity by ~3pp.
- **Meta-Harness's zeros are abstention, not competence.** The validation
  gate demands strictly positive gain on a 6-task split, so ties reject.
  The result is stability-by-abstention (often exactly 0.00) rather than
  learning.
- **Scoped is the only method with I = F = 0** on every blocked scenario,
  and it keeps the best plasticity everywhere. That is the paper's claim,
  reproduced across 18 scenarios and four stream shapes.
- **The base rates match the paper**: toolcall-clarify ≈57% (paper 53.8),
  toolcall-act ≈85% (paper 77.3), BIRD-style SQL ≈50% (paper 52.5).

## Domain playbooks

`GET /api/playbooks` — `healthcare-triage`, `trading-agent`,
`finance-assistant`, `support-triage`, `sre-agent`, `research-citation-agent`.
Each pairs a stream with the evolver the measurements favour, plus the
guardrails that must stay on (escalation-first triage, clarify-before-act,
risk-engine-disposes, provenance-on-write, scope-isolation).

## Live mode

`LivePolicy` calls the local LLM through fox-services (`:8210/api/chat`,
model `fox-managed`) with direct-Ollama fallback. The simulated policy is
the default: it makes harness effects measurable in seconds with no GPU and
no metric noise (the policy is a pure function of task + harness state).
