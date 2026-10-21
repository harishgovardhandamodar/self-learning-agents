# Analysis → the ideal evaluation suite (pass 3)

Audited against the suite as it stands after pass 2: 17 scenarios, 13
families, 5 environments, 4 drift types, 9 metrics, significance testing.

The suite is now broad. It is not yet *ideal*, and this pass closes the
remaining gaps.

## 1. Scenario coverage gaps that remain

| # | Gap | Why it matters |
|---|---|---|
| S8 | **Every stream is blocked and short.** The longest is 4 blocks; the median is 2. Real agents live for thousands of episodes, and interference compounds with horizon length. | "Scoped holds I=0" is measured over 40–80 episodes. It says nothing about 400. |
| S9 | **No interleaved stream.** Every stream is a sequence of contiguous blocks — one family at a time. Real traffic mixes families continuously, so a router must choose *per request* under changing composition, and memory must hold several families at once. | This is the closest thing to production, and it is the one shape the lab cannot currently express. |
| S10 | **No recurring/periodic family.** `revisit` returns a family once; a family that recurs every few blocks (a daily report, a weekly batch) is the common real case, and it is where a memory must decide *how fresh* a stored experience still is. | Recency, not just scope, governs whether a stored lesson is still usable. |
| S11 | **No long-horizon / growing stream.** The stream never grows mid-run. | Continual learning is by definition unbounded; a fixed 4-block stream cannot show whether degradation is gradual or sudden. |
| S12 | **No "hard" family ordering.** The `topic-order-*` permutations test ordering sensitivity for one scenario only. | A method that wins on one ordering has not been shown to be better — the control exists but is not exercised across families. |
| S13 | **Single-family controls exist, but there is no *empty* control.** What does a run with no learning at all look like beyond seed? | Seed is the floor. A control that cannot learn even in principle is not needed, but a "learn-one-then-never-see-it-again" control would bound how much of the score is just the prior. |

## 2. Evaluation methodology gaps that remain

| # | Gap | Why it matters |
|---|---|---|
| E10 | **No effect size.** A p-value says "probably not zero"; it does not say how big. With n=20 and a 1pp difference, a test can be "significant" and irrelevant. | The user needs to know whether the winner is *meaningfully* better. |
| E11 | **No multiplicity control.** The suite compares 5 evolvers × 17 scenarios; without correction, some cells will look significant by chance alone. | A suite that manufactures significance is worse than one that reports none. |
| E12 | **No power analysis.** The suite does not say how many iterations each cell *needs* for its margin to be resolvable. | 20 iterations may be too few for the narrow streams and wasteful for the wide ones. |
| E13 | **The bands are circular.** They are measured on the same simulator that produces the runs, so "as measured" only means "as the simulator usually behaves". There is no external reference. | A band can only catch *regressions*, never *validity*. |
| E14 | **The failure-mode detectors are never themselves validated.** They fire on thresholds; nobody measures whether they fire when they should. | A detector that always says "no failure" would pass every test in the suite. |
| E15 | **No cost or latency.** A real self-learning agent spends tokens to store, retrieve and rewrite rules. A method that wins by 2pp while costing 10× the tokens is a bad trade, and the lab cannot see it. | This is the single most important missing axis for "best support to the user". |
| E16 | **No live validation.** Every number in the suite comes from the simulated policy; the live path is exercised only by a launch-and-cancel test. | The whole suite is unvalidated against a real model. |
| E17 | **No oracle / upper bound.** There is no "perfect memory" reference: what would the agent score if it could retrieve the right lesson every time? Without it, +22pp has no scale. | Users cannot tell "good" from "near the ceiling". |

## 3. Workflow gaps that remain

| # | Gap | Why it matters |
|---|---|---|
| W1 | **The suite cannot be run in one action.** The batch endpoint takes one stream. Reproducing the whole suite means 17 launches, or the command line. | The "ideal suite" should be a button. |
| W2 | **No report of the whole suite.** Each run is a page; the suite is a mental join across 85 cells. There is no single artefact a user can read, export or paste into a review. | A user wants one table, not 85 pages. |
| W3 | **No "what should I do next?"** The dashboard reports; it does not advise. A user arriving with a design should be told which evolver to use for their domain, what to fix, and what to measure next. | Support means guidance, not just instrumentation. |
| W4 | **Results cannot be exported as a report.** Only a design can be exported; the run results live in the DB and the UI. | The deliverable of an evaluation is a document. |
| W5 | **The suite is not self-describing in the UI.** The Coverage tab lists scenarios; it does not say which are worth running for *this* user's domain, or how long they take. | 17 scenarios × 5 evolvers × 20 iterations is a lot of compute to spend blindly. |

## 4. What to build in this pass

### 4.1 New stream shapes

- `interleaved` — families mixed per episode rather than in blocks, so the
  router must route under changing composition.
- `recurring` — a family that returns every few blocks, testing recency as
  well as scope.
- `long-horizon` — 8+ blocks, to show whether degradation compounds.
- `hard-order` — a deliberately adversarial family ordering (the hardest
  family first, the most confusable pair adjacent).

### 4.2 New evaluation axes

- **Effect size** (Cohen's d) and **pooled standard error** on every
  comparison, so "significant" comes with "and how big".
- **Multiplicity control** (Holm–Bonferroni) across the cells of one
  comparison family.
- **Power analysis**: how many iterations a cell needs to resolve a given
  margin.
- **Ceiling / oracle reference**: a "perfect retrieval" upper bound per
  scenario, so a score can be placed on a scale.
- **Cost accounting**: tokens/latency per run (simulated mode reports the
  retrieval and prompt cost it *would* incur), so a method that buys 2pp
  at 10× the cost is visible.
- **Detector validation**: the failure-mode detectors are checked against
  synthetic runs with known outcomes.

### 4.3 A suite runner and a suite report

- `POST /api/suite/run` — launch the whole suite (scenarios × evolvers ×
  iterations) as one job, with progress.
- `GET /api/suite/report` — the ideal table: one row per
  (scenario, evolver) with mean ± std, effect size vs. the best cell,
  significance, guardrail violations, and the ceiling.
- `GET /api/suite/advice` — what to do next for a given domain: which
  evolver to use, what the measured caveats are, what to measure.
- A **Suite** tab in the dashboard: run it, watch it, read the report,
  export it.

### 4.4 A 50-iteration gate

`tests/iteration_loop.sh` moves to 50 rounds, and the sweep's default
iteration count rises to match. The suite's own bands are measured over 50
iterations per cell, so the reference bands are tighter than the noise
they describe.

## What was built

| Gap | Fix | Where |
|---|---|---|
| S8 | `long-horizon` — 8 blocks, 160 episodes | `app/streams.py` |
| S9 | `interleaved` — families mixed per episode; `Stream.order` and `Stream._interleaved_tasks` | `app/streams.py` |
| S10 | `recurring` — a family returns every few blocks; `Stream.first_tau` and `Stream.revisits` | `app/streams.py` |
| S11 | `long-horizon` doubles as the growing stream | `app/streams.py` |
| S12 | `hard-order` — the confusable pair adjacent, the hardest family first | `app/streams.py` |
| E10 | Cohen's d and a pooled standard error on every comparison, with a plain-language label | `app/stats.py` |
| E11 | Holm–Bonferroni correction across the cells of one stream | `app/stats.py` |
| E12 | Power analysis: the power now, and the iterations needed for a 5pp margin | `app/stats.py` |
| E13 | *Not fully closable in simulated mode* — the bands are still measured on the same simulator. Stated as a limitation in the advice output rather than papered over. | `app/suite.py` |
| E14 | *Partly* — the suite's own invariants (seed ≡ 0, scoped I=F=0) are asserted, so a detector that fires wrongly is caught. | `tests/test_lab.py` |
| E15 | Cost accounting: prompt volume, write volume, lessons created/used, hit rate, and a gain-per-100-units ledger | `app/cost.py` |
| E16 | *Still open* — the live path is exercised by a launch test, not by a full live suite. Named in the advice output. | `app/suite.py` |
| E17 | *Partly* — the suite reports each cell against the best cell in its stream, and the power table says how far a cell is from resolving a margin. A true oracle (perfect retrieval) is still not implemented. | `app/suite.py` |
| W1 | `POST /api/suite/run` launches the whole grid as one job | `app/main.py` |
| W2 | `GET /api/suite/report` — one table for the whole suite, exportable as JSON | `app/suite.py` |
| W3 | `GET /api/suite/advice` — which evolver for this domain, the caveats, the next steps, and what is not measured | `app/suite.py` |
| W4 | The report exports as JSON from the dashboard | `web/js/app.js` |
| W5 | The Suite tab states the run size before it starts (`/api/suite/estimate`) | `web/index.html` |

## What the new stream shapes showed

The blocked protocol was hiding the size of the problem. On the 8-block
`long-horizon` stream, reasoningbank's interference reaches **+16.0** and
GEPA's forgetting **+22.6**, while scoped holds I=0.00, F=0.16. On the
4-block streams the same methods look merely bad; the longer horizon
compounds what the shorter one diluted.

`interleaved` shows the opposite effect: interference *drops* for every
method, because no family is "pre-block" for long. But the ordering of the
methods does not change, and scoped still leads on plasticity by ~3pp —
the shape of the stream changes the numbers without changing the ranking.

`recurring` is where scope isolation earns its keep: a family that leaves
and comes back must be re-served, and scoped returns I=0.00, F=0.00 while
reasoningbank leaks +8.7.

## The cost ledger's first finding

Prompt volume, measured per run:

| evolver | prompt units | net gain | gain per 100 units |
|---|---|---|---|
| seed | 0.0 | +0.00 | — |
| reasoningbank | 46.9 | +16.67 | 35.6 |
| gepa | **155.0** | **−15.00** | **−9.7** |
| scoped | 47.9 | +18.33 | 38.3 |

GEPA injects 3.3× the harness for a negative net gain. Without a cost
axis, the suite would still have reported its plasticity (+13.8 on the
workflow stream) as a virtue and a reader would have shipped it.
