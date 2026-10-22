# Control Flow — Transactions, Activity & State

How a run actually moves. Four views: the run as a transaction between the
client and the service, the episode as a transaction between the harness
and the model, the activity of one training episode, and the state machine
a run passes through.

## The run transaction

The client asks for a comparison; the service does the work in the
background and the client polls. A run is never a request that blocks.

```mermaid
sequenceDiagram
  autonumber
  participant C as Client
  participant A as API
  participant J as Job registry
  participant R as Runner
  participant E as Evolver
  participant P as Policy
  participant G as Grader
  participant D as SQLite

  C->>A: POST /api/runs/batch (scenarios × evolvers × iterations)
  A->>A: validate evolvers, iterations, live backend
  A->>D: create run rows (status=pending)
  A->>J: spawn one asyncio task per run
  A-->>C: 200 {run_ids, count}

  loop each run, concurrently
    J->>R: execute(cfg)
    R->>D: read/write harness state
    R->>E: inject(harness, task, router)
    E-->>R: Injection (text + lessons)
    R->>P: act(task, injection, harness, router)
    P-->>R: PolicyResult (output, action, trace)
    R->>G: grade_detail(task, output)
    G-->>R: {success, reason}
    R->>E: observe(harness, task, action, success)
    E->>D: lesson written into the harness
    R->>D: checkpoint + traces
  end

  C->>A: GET /api/runs/{id}
  A-->>C: status, progress, metrics, checkpoints
  C->>A: GET /api/suite/report
  A-->>C: the whole suite as one table
```

Two properties are worth stating because they are easy to get wrong.
First, the grader is called **after** the policy, and only a boolean comes
back: no label, no gold answer, no similarity score crosses the boundary
into the harness. Second, `observe` is called on *every* episode but only
writes a lesson on a success — a failure teaches the harness nothing, which
is what makes the failure modes interesting.

## The episode transaction

The agent's own loop, with the harness in the middle. This is the loop the
paper's scenarios are about.

```mermaid
sequenceDiagram
  autonumber
  participant S as Stream
  participant R as Router
  participant E as Evolver
  participant H as Harness
  participant P as Policy
  participant L as LLM (live mode only)
  participant G as Grader

  S->>R: task (query + preamble, no label)
  R->>R: assign → cluster (scope)
  R-->>E: cluster id
  E->>E: scope key = cluster | workflow fingerprint
  E->>H: read the surfaces this evolver uses
  H-->>E: lessons, prompt, memory
  E-->>P: Injection (text + candidate lessons)

  P->>P: classify intent from the request text
  alt a lesson for this intent exists
    P->>P: apply the lesson (generalised, refilled with this task's values)
  else no lesson for this intent
    P->>P: answer from the prior (noisy, imperfect)
  end
  P->>P: perturb with probability (1 − skill)
  P-->>G: output
  G-->>P: {success, reason}
  P-->>E: trace (which lesson governed, was it foreign)
  E->>H: write a lesson (only on success)
```

The `alt` branch is the whole subject of the lab. When the candidate
lessons contain none for the current intent, the policy falls back to the
most similar one anyway — the agent has no way of knowing that the
knowledge it was handed was learned somewhere else. That fallback is the
interference mechanism, and it is why a *scope* check upstream of the
policy matters more than any amount of prompt polish.

## Activity — one training episode

```mermaid
flowchart TD
  START(["Task arrives"]) --> ROUTE["Router: which family?"]
  ROUTE --> SCOPE{"Scope key known?"}
  SCOPE -->|"no, unseen family"| PRIOR["Answer from the prior<br/>(noisy: wrong workflow, partial knowledge)"]
  SCOPE -->|"yes"| RETRIEVE["Retrieve candidates<br/>within the scope"]
  RETRIEVE --> INTENT["Classify the intent<br/>from the request text"]
  INTENT --> MATCH{"A lesson for this intent?"}
  MATCH -->|"yes"| APPLY["Apply the lesson<br/>(refilled with this task's values)"]
  MATCH -->|"no"| FALLBACK["Apply the most similar candidate<br/>← this is the interference path"]
  FALLBACK --> PERTURB["Perturb with probability (1 − skill)"]
  PRIOR --> PERTURB
  APPLY --> PERTURB
  PERTURB --> OUTPUT["Emit the answer"]
  OUTPUT --> GRADE{"Grader: task success?"}
  GRADE -->|"yes"| LEARN["Write a lesson into the harness<br/>+ provenance"]
  GRADE -->|"no"| DROP["Write nothing"]
  LEARN --> END(["Next task"])
  DROP --> END
```

## The run state machine

```mermaid
stateDiagram-v2
  [*] --> Pending: POST /api/runs
  Pending --> Running: task spawned
  Running --> Running: training episode
  Running --> Evaluating: block boundary
  Evaluating --> Running: checkpoint stored
  Running --> Done: stream finished
  Running --> Cancelled: POST /api/runs/{id}/cancel
  Running --> Error: exception
  Pending --> Cancelled: cancel before start
  Error --> [*]
  Cancelled --> [*]
  Done --> [*]
```

`Evaluating` is not a separate lifecycle state in the database — a run is
`running` from the first episode to the last — but it is a distinct
*phase* in the runner, and it is where the cost of a run concentrates: a
checkpoint evaluates **every** family in the stream, not just the one that
was just trained on. That is what makes forgetting visible at all.

## Data flow: one run, end to end

```mermaid
flowchart LR
  SOURCES["Seeds<br/>train · eval · policy · lesson-id"] --> TASKS["Task draws<br/>(disjoint offsets)"]
  TASKS --> TRAIN["Training tasks<br/>one block per family"]
  TASKS --> HOLD["Hold-out sets<br/>disjoint, per family"]
  TRAIN --> ACT["Policy actions"]
  HOLD --> EVAL["Checkpoint evaluations"]
  ACT --> HARNESS["Harness state<br/>memory · prompt · skills"]
  HARNESS --> ACT
  EVAL --> ACCT["A[t,k] table<br/>accuracy per checkpoint per family"]
  ACCT --> METRICS["Paper metrics<br/>P · I · F · S · Term. · Δ̄<br/>+ FT · BT · AUC"]
  METRICS --> DIAG["Failure modes<br/>+ retrieval + attribution"]
  ACT --> COST["Cost ledger<br/>prompt volume · lessons · hit rate"]
  METRICS --> BANDS["Compared against<br/>the measured bands"]
  BANDS --> REPORT["The report<br/>rank · effect size · significance<br/>ceiling capture · cost"]
  REPORT --> ADVICE["Advice for a domain"]
```

Every step is seeded. `A[t,k]` is the join of the two draws: the training
tasks drive the harness, the hold-out tasks measure it, and the offsets
keep them disjoint so a family is never graded on an instance the agent
was trained on.
