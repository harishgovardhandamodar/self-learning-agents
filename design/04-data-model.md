# Data Model — Storage, Snapshots & Reference Data

Three things persist: the lab's own records (SQLite), the harness state a
run carries (snapshots), and the measured reference data the suite judges
runs against. The last one is *reference*, not state — and confusing the
two is how the container once served stale bands for a day.

## ER diagram — the lab's records

```mermaid
erDiagram
  DESIGNS ||--o{ RUNS : "a design is validated by"
  RUNS ||--o{ CHECKPOINTS : "has"
  RUNS ||--o{ TRACES : "has"
  RUNS }o--|| STREAMS : "runs on"

  DESIGNS {
    TEXT id PK
    TEXT name
    TEXT domain
    TEXT description
    TEXT spec "JSON: harness + evolver + policy + stream config"
    TEXT tags "JSON array"
    REAL created_at
    REAL updated_at
  }

  RUNS {
    TEXT id PK
    TEXT design_id FK
    TEXT name
    TEXT stream_id
    TEXT evolver
    TEXT policy
    TEXT model
    TEXT status "pending|running|done|error|cancelled"
    REAL progress
    TEXT message
    TEXT config "JSON: evolver_cfg, policy_cfg, seeds, geometry"
    TEXT result "JSON envelope: metrics, failure_modes, retrieval, cost"
    TEXT error
    REAL created_at
    REAL updated_at
  }

  CHECKPOINTS {
    INTEGER id PK
    TEXT run_id FK
    INTEGER t "stream position"
    TEXT label "family at this boundary"
    TEXT kind "seed|eval"
    TEXT metrics "JSON: accuracy per family"
    TEXT diagnostics "JSON: mechanism state at this point"
    TEXT snapshot "JSON: the whole harness"
    REAL created_at
  }

  TRACES {
    INTEGER id PK
    TEXT run_id FK
    INTEGER t
    TEXT group_id
    TEXT kind "episode|update|eval"
    TEXT payload "JSON: one episode"
    REAL created_at
  }

  STREAMS {
    TEXT id PK
    TEXT name
    TEXT scenario
    TEXT config "JSON: groups, block size, hold-out size, order"
    REAL created_at
  }
```

The `result` column is an **envelope**, not a bare metrics blob: metrics,
the detected failure modes, the retrieval diagnostics, the prompt
attribution and the cost ledger travel together, so a report can be
rebuilt from the database without re-running anything.

## The harness snapshot

What a checkpoint stores is the whole writable state, so two checkpoints
can be diffed to see exactly what the method changed.

```mermaid
flowchart LR
  SNAP["Checkpoint snapshot"] --> SYS["system_prompt"]
  SNAP --> STRAT["strategy_prompt<br/>+ strategy_rules (traceable)"]
  SNAP --> MEM["memory<br/>MemoryItem[]"]
  SNAP --> SKILL["skills"]
  SNAP --> CTRL["controller"]
  SNAP --> SCOPED["scoped_sections<br/>one per scope"]
  SNAP --> CONFL["conflict_ledger"]
  SNAP --> IDX["lesson_index<br/>every lesson ever written"]
```

`strategy_rules` is the one that matters. The strategy prompt is a
*rendering* of a list of rules, and each rule carries the lesson id and
the cluster that wrote it. That is what makes the paper's Figure 3
measurable: a family's share of the prompt can be attributed, and a fall
in that share is a later block overwriting an earlier one rather than an
opaque edit.

`conflict_ledger` is the scoped evolver's honesty about itself: when one
discovered cluster carries two incompatible workflow fingerprints, that
collision is recorded rather than silently resolved.

## The reference data model

Two shipped files, outside the runtime volume, that the suite judges
against.

```mermaid
flowchart TB
  SWEEP["tests/sweep.py<br/>N iterations × scenarios × evolvers"] --> AGG["aggregate<br/>mean · std per cell"]
  AGG --> BANDS["reference/benchmarks.json<br/>bands[scenario][evolver][metric] = [mean, std]"]
  AGG --> REPORT["the report<br/>rank · effect size · Holm-corrected significance"]

  LIVE["tests/live_validation.py<br/>the same stream, live and simulated"] --> LIVEFILE["reference/live_validation.json<br/>simulated vs live baseline"]

  BANDS --> VERDICT["a run is judged<br/>as measured / above / below"]
  BANDS --> POWER["power: iterations needed<br/>for a 5pp margin"]
  BANDS --> CEILING["the oracle ceiling<br/>and the capture"]
  LIVEFILE --> ADVICE["the advice says<br/>what the numbers do and do not transfer"]
```

`bands[scenario][evolver][metric]` is a two-element array — mean and
standard deviation — because that is all a band needs to be useful, and
anything more would be precision the measurements do not support.

## The cell identity rule

Runs are only comparable when they used the same configuration, so a cell
is keyed by `(stream, evolver, block size, hold-out size)`.

```mermaid
flowchart LR
  R1["run A<br/>workflow/scoped/b20e30"] --> C1["cell<br/>workflow/scoped/b20e30"]
  R2["run B<br/>workflow/scoped/b20e30"] --> C1
  R3["run C<br/>workflow/scoped/b4e4"] --> C2["cell<br/>workflow/scoped/b4e4<br/>(separate)"]
  C1 --> AGG1["mean ± std"]
  C2 --> AGG2["mean ± std"]
  AGG1 --> CMP["comparable"]
  AGG2 --> CMP
```

Mixing geometries in one cell produces a variance that is mostly the
difference between the geometries, and a comparison against that cell says
nothing: a 22pp margin can come out "not significant" purely because a
4-episode probe run was in the bucket. The report shows each cell's
geometry so the reader can see it.

## Storage format & versioning

| what | where | format | versioned |
|---|---|---|---|
| designs, runs, checkpoints, traces | SQLite at `/app/data/sla.db` | SQL + JSON columns | schema created idempotently on start |
| harness state | inside a checkpoint's `snapshot` column | JSON | snapshot carries the harness `version` |
| reference bands | `reference/benchmarks.json` | JSON | ships with the image; regenerated by `--refresh-benchmarks` |
| live comparison | `reference/live_validation.json` | JSON | ships with the image; regenerated by `tests/live_validation.py` |
| exports | `./exports/<slug>/` | a runnable project | includes `design.json`, `run_agent.py`, `GUARDRAILS.md` |

The database is created with `CREATE TABLE IF NOT EXISTS`, so starting the
service against an existing file is safe and starting it against nothing
produces a working schema. Nothing in the schema depends on the order in
which runs were created.
