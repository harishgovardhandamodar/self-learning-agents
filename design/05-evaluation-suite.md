# Evaluation Suite — Modules, Pipelines & Verdicts

The suite is the product: an agent that cannot be measured cannot be
shipped. This document explains how the measurement is put together, what
each part contributes, and how a verdict is reached.

## Module map

```mermaid
flowchart TB
  subgraph SPEC["Specification"]
    ENVS["envs.py<br/>13 families · intents · graders"]
    STREAMS["streams.py<br/>18 scenarios + 3 orderings<br/>4 shapes"]
  end

  subgraph EXEC["Execution"]
    RUNNER["runner.py<br/>the protocol"]
    EVOLVERS["evolvers.py<br/>6 harness-rewrite rules"]
    POLICIES["policy.py<br/>simulated · live"]
    HARNESS["harness.py<br/>the writable surfaces"]
    RETRIEVAL["retrieval.py<br/>router · index · workflow probe"]
  end

  subgraph MEASURE["Measurement"]
    METRICS["metrics.py<br/>paper metrics · failure modes"]
    STATS["stats.py<br/>effect size · Holm · power"]
    COST["cost.py<br/>prompt volume · ledger"]
  end

  subgraph JUDGE["Judgement"]
    BENCH["benchmarks.py<br/>measured bands"]
    SUITE["suite.py<br/>grid · report · advice"]
    FLOWS["flows.py<br/>mermaid from a design"]
    DESIGNDOCS["design_docs.py<br/>the architecture documents"]
  end

  subgraph SUPPORT["Support"]
    DESIGNS["designs.py<br/>playbooks + guardrails"]
    DOCS["design/*.md<br/>the architecture documents"]
  end

  subgraph SURFACE["Surface"]
    API["main.py<br/>35 endpoints"]
    WEB["web/<br/>8 tabs"]
  end

  subgraph INFRA["Infrastructure"]
    LLM["llm.py<br/>fox-services → Ollama"]
    DB[("db.py<br/>SQLite records")]
  end

  ENVS --> STREAMS
  STREAMS --> RUNNER
  RUNNER --> EVOLVERS
  RUNNER --> POLICIES
  RUNNER --> HARNESS
  RUNNER --> RETRIEVAL
  RUNNER --> METRICS
  RUNNER --> COST
  METRICS --> STATS
  STATS --> SUITE
  BENCH --> SUITE
  SUITE --> API
  FLOWS --> API
  API --> WEB
  RUNNER --> DB
  POLICIES --> LLM
  DOCS --> DESIGNDOCS
  DESIGNDOCS --> API
  DESIGNS --> API
```

## Module map

The layers below are how the code is organised. The dependency graph that
follows is **generated from the imports**, not drawn by hand, so it cannot
drift from the code.

<!-- GENERATED: module-dependencies -->

## The measurement pipeline

```mermaid
flowchart TD
  START(["One run of one cell"]) --> A["A[t,k]: accuracy at every checkpoint<br/>for every family"]
  A --> B["B_k: the seed baseline per family"]
  B --> C["P_k = A[τ_k,k] − B_k<br/>plasticity: gained during its own block"]
  B --> D["I_k: mean drop below B_k<br/>before its own block"]
  B --> E["F_k = A[τ_k,k] − A[T,k]<br/>forgetting: lost after its own block"]
  B --> F["S = −(Ī + F̄)"]
  B --> G["FT: gained before its own block"]
  B --> H["BT: changed after its own block"]
  A --> I["AUC: the shape over the horizon"]
  C --> J["Failure modes:<br/>interference · forgetting<br/>foreign memory · collision<br/>overwriting · trade-off"]
  D --> J
  E --> J
  J --> K["Verdict against the measured bands"]
  K --> L["Effect size vs the best cell"]
  L --> M["Holm-corrected significance"]
  M --> N["Cost ledger: gain per 100 prompt units"]
  N --> O["Ceiling: how much of the oracle was captured"]
  O --> P(["The report row"])
```

Every quantity above is a function of the same `A[t,k]` table, so the
report's numbers cannot disagree with each other: the metrics, the failure
modes, the effect sizes and the cost all come from one run's checkpoints.

## The comparison, as a transaction

```mermaid
sequenceDiagram
  autonumber
  participant U as User
  participant S as Suite
  participant R as Runner
  participant B as Bands
  participant St as Statistics
  participant DB as SQLite

  U->>S: POST /api/suite/estimate (scope, evolvers, iterations)
  S-->>U: runs, episodes, and what it will cost
  U->>S: POST /api/suite/run
  S->>R: one run per (scenario × evolver × iteration)
  R->>DB: checkpoints, traces, result envelopes
  U->>S: GET /api/suite/report
  S->>DB: finished runs
  S->>S: group into cells by (stream, evolver, geometry)
  S->>St: Welch test, effect size, Holm correction
  S->>B: the measured band for each cell
  S-->>U: rank · margin · CI · effect · significance · cost · ceiling
  U->>S: GET /api/suite/advice?domain=...
  S-->>U: which evolver, the caveats, what is not measured
```

## What each statistical layer contributes

| layer | question it answers | what it cannot tell you |
|---|---|---|
| mean ± std | how much the cell varies | whether two cells differ |
| Welch t-test | do two cells differ | by how much, or against how many comparisons |
| effect size (Cohen's d) | how big the difference is in pooled SDs | whether it is distinguishable from zero |
| Holm–Bonferroni | does the difference survive comparing many cells | — |
| power | how many iterations this cell would need | whether the current n was enough |
| cost ledger | what the gain cost in prompt volume | — |
| oracle ceiling | how much of the achievable gain was captured | — |

A cell with a large mean and a huge standard deviation is not a result.
Reporting all seven together is what makes a cell's number mean something.

## The verdict path

```mermaid
flowchart TD
  RUN(["A finished run"]) --> CELL["Its cell:<br/>(stream, evolver, geometry)"]
  CELL --> BAND{"Is the cell measured?"}
  BAND -->|"no"| NOBAND["No band: reported as unknown,<br/>not as a pass"]
  BAND -->|"yes"| INBAND{"Is the value inside<br/>mean ± std?"}
  INBAND -->|"yes"| EXPECTED["as measured"]
  INBAND -->|"above"| BETTER["above the band"]
  INBAND -->|"below"| WORSE["below the band"]
  WORSE --> NOTE["with a reason:<br/>which metric, which band"]
  RUN --> GUARD{"Any guardrail violation<br/>in the proposed actions?"}
  GUARD -->|"yes"| UNSAFE["flagged: a method that wins on accuracy<br/>while proposing unsafe actions<br/>is not the method to ship"]
  RUN --> COSTQ{"Is the gain per<br/>100 prompt units negative?"}
  COSTQ -->|"yes"| EXPENSIVE["flagged: bought at a loss"]
```

The three flags are independent. A method can be exactly as measured,
have zero guardrail violations, and still be a bad choice if it spends
three times the harness to get there.

## The scenario shapes

```mermaid
flowchart LR
  subgraph BLOCKED["blocked (the paper's protocol)"]
    B1["one family at a time"]
  end
  subgraph INTERLEAVED["interleaved"]
    I1["families mixed per episode"]
  end
  subgraph RECURRING["recurring"]
    R1["a family returns every few blocks"]
  end
  subgraph LONG["long-horizon"]
    L1["eight blocks, 160 episodes"]
  end
  subgraph HARD["adversarial ordering"]
    H1["confusable pair adjacent"]
  end
  BLOCKED -->|"tests"| T1["does scope isolation hold?"]
  INTERLEAVED -->|"tests"| T2["can the router route under<br/>changing composition?"]
  RECURRING -->|"tests"| T3["does competence return<br/>after an interruption?"]
  LONG -->|"tests"| T4["does degradation compound?"]
  HARD -->|"tests"| T5["does the result survive<br/>an inconvenient order?"]
```

The shapes exist because the blocked protocol hid things. On the
eight-block stream, reasoningbank's interference reaches +16.0 and GEPA's
forgetting +22.6 — numbers the four-block streams made look mild. And on
the interleaved stream, interference *drops* for every method, because no
family is "pre-block" for long: the shape changes the numbers without
changing the ranking.

## What the suite does not measure

Stated plainly, because a suite that overstates its coverage is worse than
one that admits its edges.

```mermaid
flowchart TB
  GAPS["Known limits"] --> L1["Live-mode validation:<br/>every number comes from the<br/>simulated policy"]
  GAPS --> L2["Cost ceiling:<br/>prompt volume is measured,<br/>a budget is not"]
  GAPS --> L3["Single-family controls only:<br/>per-environment baselines,<br/>not per-intent ones"]
  GAPS --> L4["One simulator:<br/>the bands are measured on the same<br/>simulator that produces the runs"]
  GAPS --> L5["Simulated policy only:<br/>the simulated prior is a competence<br/>model, not a language model"]
  L1 --> FIX["The live comparison measures the<br/>gap and the advice states it"]
  L4 --> FIX2["The bands catch regressions,<br/>not validity — a validity claim<br/>needs a different simulator"]
```

The live comparison is the honest answer to the first limit: the local
model scores 100% where the simulator scores 50% on the trading family, so
the suite's **rankings** transfer and its **magnitudes** do not. The
suite's plasticity numbers are headroom, not expected production gains,
and the advice panel says so where the numbers are read.
