# UML — Components & Classes

Static structure of the lab. Four views: the component diagram, the core
domain classes, the evolver hierarchy (the part that actually matters), and
the interface contracts the runner relies on.

## Component diagram

```mermaid
flowchart TB
  subgraph PRES["Presentation layer (web/)"]
    TABS["Tab strip"]
    DSP["DesignsPane"]
    RNP["RunsPane"]
    CMP["ComparePane"]
    REFP["ReferencePane"]
    CVP["CoveragePane"]
    SUP["SuitePane"]
    PATP["PatternPane"]
    DSPN["DesignPane<br/>rail · outline · doc"]
    MD["MarkdownRenderer"]
    MMR["MermaidEngine<br/>render queue"]
  end

  subgraph API["API layer (app/main.py)"]
    ROUTES["35 endpoints"]
    JOBS["Job registry<br/>asyncio tasks + cancel flags"]
    ASSE["Assemblers<br/>_cell_summary · _run_row"]
  end

  subgraph DOMAIN["Domain layer"]
    RUNNER["Runner<br/>the stream protocol"]
    EVOLVERS["Evolvers (6)"]
    POLICIES["Policies (2)"]
    ENVIRON["Environments (5 envs, 13 families)"]
    RETRIEVAL["Router + SimilarityIndex<br/>+ workflow probe"]
    HARNESS["Harness<br/>the writable surfaces"]
    METRICS["Metrics<br/>paper metrics + failure modes"]
    STATS["Statistics<br/>effect size · Holm · power"]
    COST["Cost ledger"]
    SUITE["Suite<br/>grid · report · advice"]
    BENCH["Benchmarks<br/>measured bands"]
  end

  subgraph SUPPORT["Support"]
    DESIGNS["Designs + playbooks<br/>+ guardrails"]
    FLOWS["Flows<br/>mermaid from a design"]
    DOCS["Design documents<br/>design/*.md"]
  end

  subgraph INFRA["Infrastructure"]
    LLM["LLMClient<br/>fox-services → Ollama"]
    DB[("SQLite<br/>db.py")]
  end

  TABS --> DSP & RNP & CMP & REFP & CVP & SUP & PATP & DSPN
  DSPN --> MD
  MD --> MMR
  PATP --> MMR

  ROUTES --> JOBS
  JOBS --> RUNNER
  ROUTES --> SUITE
  ROUTES --> ASSE
  ASSE --> STATS
  ASSE --> BENCH
  ROUTES --> FLOWS
  ROUTES --> DOCS

  SUITE --> RUNNER
  SUITE --> STATS
  SUITE --> BENCH
  RUNNER --> EVOLVERS
  RUNNER --> POLICIES
  RUNNER --> ENVIRON
  RUNNER --> RETRIEVAL
  RUNNER --> HARNESS
  RUNNER --> METRICS
  RUNNER --> COST
  RUNNER --> DB
  POLICIES --> LLM
  POLICIES --> HARNESS
  EVOLVERS --> HARNESS
  EVOLVERS --> RETRIEVAL
  METRICS --> STATS
```

## Core domain classes

```mermaid
classDiagram
  class Task {
    +str id
    +str group
    +str env
    +str query
    +dict context
    +Any gold
    +str grader
    +dict meta
  }

  class TaskGroup {
    +str id
    +str name
    +str env
    +str drift
    +str context_md
    +list templates
    +str signature
    +intent_lexicon() dict
    +builtin_actions() dict
    +sample(n, offset, seed) list~Task~
  }

  class EnvView {
    +str signature
    +list groups
    +intent_lexicon() dict
    +builtin_actions() dict
    +schema_sql() str
  }

  class Stream {
    +str id
    +str scenario
    +list blocks
    +list groups
    +str order
    +int T
    +sigma dict
    +tau dict
    +first_tau dict
    +training_tasks() list~Task~
    +eval_tasks() dict
  }

  class Block {
    +str group
    +int start
    +int end
    +int n_train
  }

  class Harness {
    +str system_prompt
    +str strategy_prompt
    +list memory
    +list skills
    +str controller
    +dict scoped_sections
    +list conflict_ledger
    +list strategy_rules
    +list lesson_index
    +clone() Harness
    +snapshot() dict
  }

  class Lesson {
    +str id
    +str owner
    +str scope
    +str env
    +str query
    +dict action
    +str text
    +dict meta
  }

  class Injection {
    +str text
    +list lessons
    +str source
    +str retrieved_from
    +float similarity
    +str note
  }

  Task --> TaskGroup : drawn from
  TaskGroup --> EnvView : one preamble may back many
  Stream --> Block : sequence of
  Stream --> Task : training and hold-out draws
  Lesson --> Harness : written into
  Injection --> Lesson : carries
```

`EnvView` is the piece that keeps the lab honest. A task family is
identified by its **preamble**, not by its label: the two BFCL families
share one API catalogue, so a view of the environment holds *both* of them
and the agent must work out from the request which one it is in. A policy
handed `task.group` would be reading the answer key.

## The evolver hierarchy

The harness is the only thing that changes. Each evolver is a different
rule for rewriting it.

```mermaid
classDiagram
  class Evolver {
    +str id
    +str name
    +str paradigm
    +list surfaces
    +observe(harness, task, action, success, trace, router) Lesson
    +on_block_end(harness, block, ctx) dict
    +inject(harness, task, router) Injection
    +diagnostics(harness) dict
  }

  class SeedEvolver {
    <<no adaptation>>
  }

  class ReasoningBankEvolver {
    +int top_k
    +float threshold
    +str scope_mode
  }

  class GEPAEvolver {
    +int max_chars
    +int update_every
  }

  class MetaHarnessEvolver {
    +float min_gain
    +last_proposal
  }

  class ScopedEvolver {
    +int top_k
    +int max_per_scope
    +str _scope_key(task, router) str
  }

  class OracleEvolver {
    +str _scope_key(task, router) str
  }

  Evolver <|-- SeedEvolver
  Evolver <|-- ReasoningBankEvolver
  Evolver <|-- GEPAEvolver
  Evolver <|-- MetaHarnessEvolver
  Evolver <|-- ScopedEvolver
  ScopedEvolver <|-- OracleEvolver
```

`OracleEvolver` inherits everything from `ScopedEvolver` and changes one
method: `_scope_key`. That is deliberate — it makes the oracle a strict
superset of the scoped evolver's information (it knows the family instead
of guessing the cluster), so it is a true ceiling rather than just another
method. The suite reports how much of that ceiling the best real method
captured.

## Interfaces the runner relies on

```mermaid
classDiagram
  class Policy {
    <<interface>>
    +name
    +act(task, injection, harness, router) PolicyResult
  }

  class SimulatedPolicy {
    +float skill
    +int seed
    +dict probe_accuracy
    +float foreign_follow
    +_hit(task, salt, prob) bool
    +_intent_of(task, view) str
    +_governing(task, injection, router, intent) Lesson
    +_prior(task, view, intent) dict
    +_maybe_perturb(task, action, lesson, view) dict
  }

  class LivePolicy {
    +str model
    +int max_steps
    +_contract(task) str
    +_parse(task, raw) dict
    +_observe(task, action) str
  }

  class PolicyResult {
    +str output
    +dict action
    +dict trace
  }

  class Router {
    +float threshold
    +SimilarityIndex index
    +SimilarityIndex qindex
    +dict clusters
    +assign(task) tuple
    +scope_key(task, cluster_id) str
    +query_similarity(a, b) float
    +clone() Router
  }

  class Grader {
    <<module functions>>
    +grade(task, pred) bool
    +grade_detail(task, pred) dict
    +guardrail_violations(task, action, output) list
  }

  Policy <|.. SimulatedPolicy
  Policy <|.. LivePolicy
  Policy --> PolicyResult : returns
  SimulatedPolicy --> Router : similarity only
  Grader --> Task : judges against the gold
```

Two indices, not one. `Router.index` clusters on the **feature text**
(preamble + request) and decides which family a task belongs to.
`Router.qindex` matches on the **raw request** and is what a memory store
actually retrieves on. Conflating them is the bug the paper describes: the
agent retrieves the nearest *topic* and applies it to a task that needs the
opposite *workflow*.
