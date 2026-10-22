# System Context

**What this system is:** a lab for designing, building and *testing* self-learning agents — harnesses that rewrite themselves from experience, over a stream of task families that drift underneath them. It reproduces the protocol and the metrics of arXiv:2610.25858 ("Are Self-Evolving Agents Continual Learners?"), and adds the mechanism the paper asks for: task-scoped learning with workflow conditioning.

It is deliberately the same shape as the AI-Agents-Patterns dashboard: a single-page vanilla-JS app with a tab strip, a design tab that renders markdown + Mermaid documents live, and no build step. Mermaid is vendored under `web/vendor/` (the CDN is only a fallback), so the whole app works offline on a tailnet.

## Purpose & principles

- **Nothing is a screenshot.** Every diagram is Mermaid source stored in `design/*.md`, rendered at view time — a diagram that has changed cannot be mistaken for a correct one. The same rule applies to the *agent* diagrams: the flow drawn in the Design tab is generated from the running code.
- **The harness is the subject.** The model is frozen. Adaptation happens by rewriting the harness — prompt, memory, skills, controller — never the weights. Every method here is a different rule for rewriting it.
- **Guardrails outside the model.** Escalation-first triage, the position cap, "ask before acting" — these are enforced by the grader and by the audit, not by instructions in a prompt.
- **A number without a reference is not a result.** Every run is judged against measured bands, and every comparison carries an effect size, a corrected significance verdict and a cost.
- **Everything is inspectable.** Designs, runs, checkpoints, harness snapshots and retrieval traces are plain rows in SQLite; the whole collection exports as JSON.

## Context diagram

```mermaid
flowchart TB
  subgraph USER["People"]
    ENG["Engineer / researcher<br/>designing a self-learning agent"]
    DOM["Domain owner<br/>healthcare · finance · trading"]
    SEC["Reviewer<br/>what may it do?"]
  end

  subgraph SYS["Self-Learning Agent Lab (this app)"]
    DASH["Dashboard SPA<br/>designs · runs · compare · reference<br/>coverage · suite · patterns · design"]
    API["FastAPI service<br/>40 endpoints"]
  end

  subgraph EXT["External knowledge (static, versioned)"]
    PAT["AI-Agents-Patterns catalog<br/>107 patterns · 54 scenarios<br/>(optional mount)"]
    REF["reference/benchmarks.json<br/>measured bands, 50 iterations per cell"]
    LIVE["reference/live_validation.json<br/>live vs simulated baseline"]
  end

  subgraph LLM["Tailnet model backends (optional)"]
    FOX["fox-services :8210<br/>routing + concurrency"]
    OLL["Ollama :11434<br/>direct fallback"]
  end

  ENG -->|"design a harness"| DASH
  DOM -->|"what should it learn, safely?"| DASH
  SEC -->|"audit the guardrails"| DASH

  DASH --> API
  API --> PAT
  API --> REF
  API --> LIVE
  API -->|"live mode rollouts"| FOX
  API -->|"fallback"| OLL
```

## Container overview

```mermaid
flowchart LR
  subgraph BROWSER["Browser (single page, 8 tabs)"]
    TABS["Tab strip<br/>Designs · Runs · Compare · Reference<br/>Coverage · Suite · Patterns · Design"]
    DSP["DesignsPane<br/>spec editor · guardrails · flow"]
    RNP["RunsPane<br/>batch launch · history · report"]
    CMP["ComparePane<br/>table + per-cell summary"]
    REFP["ReferencePane<br/>bands + metric glossary"]
    CVP["CoveragePane<br/>scenario matrix + gaps"]
    SUP["SuitePane<br/>run · report · advice"]
    PATP["PatternPane<br/>catalog + flows"]
    DSPN["DesignPane<br/>rail · outline · doc"]
    MMR["MermaidEngine<br/>render queue: Promise"]
    MD["MarkdownRenderer<br/>blocks → HTML"]
    STORE["localStorage<br/>(last tab, preferences)"]
  end

  subgraph SERVICE["FastAPI service (:8220)"]
    ROUTES["Routes (40)<br/>designs · runs · suite · stats"]
    EXEC["Runner<br/>train → checkpoint → evaluate"]
    EVOLVERS["Evolvers<br/>seed · reasoningbank · gepa<br/>metaharness · scoped · oracle"]
    POLICIES["Policies<br/>simulated (deterministic)<br/>live (ReAct)"]
    ENVIRON["Environments<br/>sql · toolcall · science<br/>healthcare · trading"]
    METRICS["Metrics + statistics<br/>paper metrics · effect size · power"]
    SUITE["Suite<br/>grid · report · advice"]
    FLOWS["Flows<br/>mermaid from the design"]
    DB[("SQLite<br/>designs · runs · checkpoints<br/>traces · streams")]
  end

  TABS --> DSP & RNP & CMP & REFP & CVP & SUP & PATP & DSPN
  DSP --> EXEC
  RNP --> EXEC
  SUP --> SUITE
  SUITE --> EXEC
  EXEC --> EVOLVERS
  EXEC --> POLICIES
  EXEC --> ENVIRON
  EXEC --> METRICS
  EXEC --> DB
  ROUTES --> EXEC & SUITE & METRICS & FLOWS
  DSPN --> MD
  MD --> MMR
  PATP --> MMR
  EXEC --> MMR
  STORE --> TABS
```

## Trust boundaries

```mermaid
flowchart TB
  subgraph T1["Untrusted: the request stream"]
    TASKS["Task requests<br/>text only, no labels"]
  end

  subgraph T2["Trusted: the harness (the thing being rewritten)"]
    H["system prompt · strategy prompt<br/>memory · skills · controller<br/>scoped sections"]
  end

  subgraph T3["Trusted: the environment (ground truth)"]
    GRADERS["Deterministic graders<br/>SQL execution · tool-call match<br/>triage acuity · risk cap"]
    INTENTS["Intent taxonomy<br/>names the kind, not the answer"]
  end

  subgraph T4["Semi-trusted: the model backend"]
    LLM["fox-services / Ollama"]
  end

  subgraph T5["Untrusted: what the agent itself writes"]
    LESSON["Lessons and rules<br/>written from its own successes"]
  end

  TASKS -->|"never labelled"| H
  T5 -->|"candidate rules"| H
  H -->|"prompt + retrieved lessons"| LLM
  LLM -->|"candidate action"| GRADERS
  GRADERS -->|"binary success only"| T5
  INTENTS -->|"kind, not answer"| H
  GRADERS -->|"never exposed to the agent"| INTENTS
```

The important boundary is between **T3 and T5**: the environment's ground truth grades the agent, but only a *binary* success signal crosses back. The agent never sees a label, a gold answer, or a similarity to one. Everything the harness accumulates is derived from its own trajectory, which is why a lesson can be wrong and why the scoped evolver's job is to make sure the wrong ones do not reach the wrong tasks.

## Deployment

```mermaid
flowchart LR
  subgraph HOST["Host"]
    DOCKER["docker compose<br/>lab service"]
    VOL[("volume: sla-data<br/>/app/data/sla.db")]
    IMG["image: self-learning-agents<br/>app + web + reference/"]
  end

  subgraph TAIL["Tailnet / LAN"]
    USER["Browser<br/>http://host:8220"]
  end

  subgraph BACKEND["Model backend (host)"]
    FOX["fox-services :8210"]
  end

  USER -->|"host.docker.internal"| DOCKER
  DOCKER --> VOL
  DOCKER --> IMG
  DOCKER -->|"live mode"| FOX
```

The reference data (measured bands, the live comparison) ships **inside the image** at `/app/reference`, deliberately outside the runtime volume at `/app/data`. A volume mounted over the shipped path would shadow it, and the app would silently fall back to stale bands — which is exactly what happened once, and is now covered by a test.
