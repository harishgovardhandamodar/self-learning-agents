# Workflows — How a Person Uses the Lab

The app is a bench, not a demo. This document follows the paths a user
actually walks: designing an agent, proving it learns, proving it does
not break, and taking it away.

## The intended path

```mermaid
flowchart LR
  D["1 · Design<br/>pick a playbook or start blank"] --> S["2 · Run a comparison<br/>every evolver, one stream"]
  S --> R["3 · Read the report<br/>rank · effect size · cost"]
  R --> I{"4 · Is it good?"}
  I -->|"interference or forgetting"| FIX["Fix the harness:<br/>scope the rules, separate the workflows"]
  I -->|"yes"| G["5 · Guardrail audit<br/>did it propose anything unsafe?"]
  G -->|"a violation"| FIX
  G -->|"clean"| E["6 · Export the project<br/>and take it away"]
  FIX --> S
```

Step 1 is a playbook because starting from a blank spec is a trap: the
playbooks pair a stream with the evolver the measurements favour and the
guardrails that must stay on. Step 2 is a *comparison* because a single
run tells you nothing — the lab's whole claim is about differences
between methods, and a single number is not a difference.

## Designing an agent

```mermaid
flowchart TD
  OPEN(["Open Designs"]) --> PICK{"Playbook?"}
  PICK -->|"yes"| LOAD["Load the stream, evolver<br/>and guardrails it names"]
  PICK -->|"no"| BLANK["Start from the default spec"]
  LOAD --> SPEC["Edit the spec:<br/>system prompt · strategy prompt<br/>controller · skills"]
  BLANK --> SPEC
  SPEC --> FLOW["Read the flow diagram<br/>generated from your design"]
  FLOW --> CHECK{"Does the flow match<br/>what you meant?"}
  CHECK -->|"no"| SPEC
  CHECK -->|"yes"| SAVE["Save"]
  SAVE --> RUN["Launch a run"]
```

The flow diagram is generated from the design, not drawn by hand, so it
cannot drift from the harness. If the diagram shows a shared strategy
prompt where you meant per-scope sections, the design says what you
actually built.

## Proving an agent learns

```mermaid
flowchart TD
  A(["Compare every evolver<br/>on one stream"]) --> B["Read the report"]
  B --> C{"Is the winner's margin<br/>significant after correction?"}
  C -->|"no"| D["Treat it as a tie.<br/>More iterations, or a bigger effect"]
  C -->|"yes"| E{"Is the effect<br/>meaningfully large?"}
  E -->|"no"| D
  E -->|"yes"| F{"Did it win by spending<br/>more prompt volume?"}
  F -->|"yes"| G["A bad trade:<br/>the gain does not pay for the harness"]
  F -->|"no"| H["A real result"]
  D --> I["Raise the iteration count<br/>until the power table says<br/>the margin is resolvable"]
  I --> B
```

The power table is the answer to "how many iterations do I need?". It
reports, per cell, the power at the current *n* and the number of
iterations that would resolve a 5pp margin — so iterations are spent where
they buy resolution rather than uniformly.

## Proving an agent does not break

```mermaid
flowchart LR
  subgraph CHECKS["The four ways a self-learning agent breaks"]
    C1["Pre-block interference<br/>a later block hurts an earlier family"]
    C2["Post-block forgetting<br/>a later block erases an earlier family"]
    C3["Retrieval collision<br/>the right topic, the wrong workflow"]
    C4["Prompt overwriting<br/>a later block rewrites an earlier rule"]
  end
  R["The report's failure modes<br/>+ the guardrail audit"] --> C1 & C2 & C3 & C4
  C1 & C2 & C3 & C4 --> FIX["Fix the harness, not the prompt"]
```

The four failure modes are the paper's findings, turned into named
diagnoses with the evidence attached. Each says what broke, how much, and
what kind of harness edit causes it.

## The guardrail audit

```mermaid
flowchart TD
  A(["Every proposed action"]) --> B{"Kind of action?"}
  B -->|"trade"| C{"Direction and size<br/>within the rules?"}
  C -->|"size over the cap"| V1["position-cap-exceeded"]
  C -->|"directional with no size"| V2["directional-signal-without-size"]
  C -->|"flat with a size"| V3["flat-signal-with-size"]
  B -->|"triage"| D{"Acute case without escalation?<br/>Or a diagnosis?"}
  D -->|"yes"| V4["missed-required-escalation<br/>issued-diagnosis-or-treatment"]
  B -->|"tool call"| E{"Any argument empty<br/>or guessed?"}
  E -->|"yes"| V5["tool-call-with-missing-argument"]
  V1 & V2 & V3 & V4 & V5 --> REP["Reported per run:<br/>the agent PROPOSED it"]
```

This audit is the difference between a measurement and a property. The
graders reject a bad *answer*; the audit asks whether the agent ever
*proposed* a breaking action. A method that wins on accuracy while
proposing an acute triage case without escalation is not the method to
ship, whatever its score.

## Taking the result away

```mermaid
flowchart LR
  DONE(["A validated design"]) --> EXP["POST /api/export"]
  EXP --> DIR["exports/<slug>/"]
  DIR --> F1["design.json<br/>the harness, validated"]
  DIR --> F2["run_agent.py<br/>replays the protocol"]
  DIR --> F3["GUARDRAILS.md<br/>the checklist to keep"]
  DIR --> F4["validation.json<br/>the numbers it scored"]
  F1 & F2 & F3 & F4 --> USE["Run it against a real workload"]
```

The export carries the guardrail checklist so the guardrails travel with
the code. A harness exported without them is a harness someone will
quietly switch off.

## The live path

Everything above runs in simulated mode, which is deterministic, fast and
free. Live mode runs the same protocol against a real model.

```mermaid
flowchart TD
  LIVE(["Live mode"]) --> C1["The same stream, the same metrics,<br/>the same protocol"]
  C1 --> C2["But every episode costs<br/>a model call"]
  C2 --> C3["SQL is executed for real:<br/>the agent sees its own errors<br/>and can correct them"]
  C3 --> C4["The live comparison<br/>measures what the simulator<br/>over- or under-states"]
  C4 --> C5["Use it to CALIBRATE,<br/>not to iterate"]
```

Live mode is for calibration, not for iteration: a run takes minutes per
call, so the suite is not a place to iterate. Its job is to say how far the
simulated numbers transfer, and the answer it currently gives is that the
rankings transfer and the magnitudes do not.
