"""
Agent flow diagrams.

The design tab shows how an agent actually works, not just its config. Two
kinds of diagram:

  * AGENT FLOW - generated from the design itself: the harness surfaces, the
    evolver's read/write surfaces, the retrieval path, and the stream the
    agent will be tested on. This is the diagram of *this* design, so it
    changes when the design changes.

  * PATTERN DIAGRAMS - the mermaid diagrams that ship with the
    AI-Agents-Patterns catalog (32 of the 107 patterns carry one, plus one
    per category), so a user can see the shape of a pattern before adopting
    it.

The flow diagram is generated rather than hand-written on purpose: a
diagram that drifts from the code is worse than no diagram.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# How each evolver reads and writes the harness. Used to draw the flow.
EVOLVER_FLOW: Dict[str, Dict[str, Any]] = {
    "seed": {
        "reads": [],
        "writes": [],
        "note": "No adaptation. The frozen harness answers every task.",
    },
    "reasoningbank": {
        "reads": ["memory"],
        "writes": ["memory"],
        "note": "Every success becomes a memory item; retrieval is by request "
                "similarity with no scope check.",
    },
    "gepa": {
        "reads": ["strategy_prompt"],
        "writes": ["strategy_prompt"],
        "note": "One shared strategy prompt is rewritten and compressed, so "
                "later blocks overwrite earlier rules.",
    },
    "metaharness": {
        "reads": ["system_prompt", "strategy_prompt", "memory", "skills", "controller"],
        "writes": ["system_prompt", "strategy_prompt", "memory", "skills", "controller"],
        "note": "The whole workspace may be edited, but an edit is accepted "
                "only when it improves validation accuracy.",
    },
    "scoped": {
        "reads": ["memory", "scoped_sections"],
        "writes": ["memory", "scoped_sections", "conflict_ledger"],
        "note": "Lessons are filed per discovered cluster AND per workflow "
                "fingerprint, in append-only sections.",
    },
}

SURFACE_LABEL = {
    "system_prompt": "system prompt",
    "strategy_prompt": "shared strategy prompt",
    "memory": "memory store",
    "skills": "skill library",
    "controller": "controller",
    "scoped_sections": "scoped sections",
    "conflict_ledger": "conflict ledger",
}


def _escape(text: str) -> str:
    """Make a label safe for a mermaid node."""
    return (str(text or "")
            .replace('"', "'")
            .replace("\n", " ")
            .replace("|", "/")
            .replace("[", "(")
            .replace("]", ")")
            .replace("{", "(")
            .replace("}", ")")
            .replace("<", "(")
            .replace(">", ")"))


def agent_flow_diagram(design: Dict[str, Any], stream_groups: Optional[List[str]] = None,
                       group_names: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Draw the design as a flow: request -> router -> scope check -> policy
    -> harness surfaces -> action -> grader, with the evolver's own
    read/write path and the stream the agent will face.

    Every node referenced by an edge is also defined, and no edge uses
    mermaid's `&` chaining: both are common sources of parse errors, and a
    diagram that fails to render is worse than none.
    """
    spec = (design or {}).get("spec") or {}
    evolver = spec.get("evolver", "scoped")
    policy = spec.get("policy", "simulated")
    flow = EVOLVER_FLOW.get(evolver, EVOLVER_FLOW["scoped"])
    groups = stream_groups or []

    reads = [SURFACE_LABEL.get(s, s) for s in flow["reads"]]
    writes = [SURFACE_LABEL.get(s, s) for s in flow["writes"]]

    m: List[str] = ["flowchart TD"]
    # --- request path
    m.append('  REQ["Request"] --> ROUTER{"Router: which task family?"}')
    m.append('  ROUTER -->|"cluster + workflow fingerprint"| SCOPE{"In scope?"}')
    m.append('  SCOPE -->|"no: unseen family"| PRIOR["Prior: no matching rule"]')
    m.append('  PRIOR --> POLICY["Policy answers"]')
    m.append('  POLICY --> ACT["Action: SQL / tool call / triage / trade"]')
    m.append('  ACT --> GRADER{"Grader: task success?"}')
    m.append('  GRADER -->|"no"| NOLEARN["No lesson written"]')
    m.append('  NOLEARN --> NEXT["Next request"]')
    m.append('  NEXT --> REQ')

    # --- where the policy reads from
    if evolver == "scoped":
        m.append('  SCOPE -->|"yes: same scope"| MEM["Scoped memory<br/>cluster + workflow"]')
        m.append('  MEM --> SURF["Harness surfaces read"]')
        m.append('  MEM --> CONFLICT["Conflict ledger<br/>two workflows in one cluster"]')
    elif flow["reads"]:
        m.append('  SCOPE -->|"yes"| SURF["Harness surfaces read"]')
        m.append('  SURF --> POLICY')
    else:
        m.append('  SCOPE -->|"n/a: no adaptation"| SURF["Seed harness (frozen)"]')
        m.append('  SURF --> POLICY')

    # --- what the evolver writes, and how it comes back
    if writes:
        m.append('  GRADER -->|"yes"| LEARN["Evolver learns from the success"]')
        for i, w in enumerate(writes):
            node = f"W{i}"
            m.append(f'  LEARN --> {node}["{_escape(w)}"]')
            m.append(f'  {node} -.->|"read by the next request"| SURF')
    else:
        m.append('  GRADER -->|"yes"| LEARN["No adaptation: the success is discarded"]')
        m.append('  LEARN --> NEXT')
    m.append('  LEARN --> NEXT')

    m.append('  classDef learn fill:#1d3a2a,stroke:#4cc38a,color:#d7dee9;')
    m.append('  classDef guard fill:#3a2a1d,stroke:#ffb020,color:#d7dee9;')
    m.append('  classDef fail fill:#3a1d1d,stroke:#ff6b6b,color:#d7dee9;')
    m.append('  classDef store fill:#1d2836,stroke:#5aa9ff,color:#d7dee9;')
    m.append('  class LEARN learn;')
    m.append('  class SCOPE,ROUTER,GRADER guard;')
    m.append('  class NOLEARN fail;')
    m.append('  class SURF,MEM,CONFLICT store;')

    return {
        "id": "agent-flow",
        "title": "Agent flow for this design",
        "mermaid": "\n".join(m),
        "notes": [
            flow["note"],
            f"Policy: {policy}. Stream: {spec.get('stream', 'workflow')}"
            + (f" over {len(groups)} families" if groups else ""),
        ],
        "reads": reads,
        "writes": writes,
    }


def stream_diagram(groups: List[str], group_names: Optional[Dict[str, str]] = None,
                   block_size: int = 20) -> Dict[str, Any]:
    """
    Draw the stream the agent will face: blocks of drifting families, with
    the checkpoints where the harness is frozen and evaluated.
    """
    names = group_names or {}
    m = ["flowchart LR"]
    for i, g in enumerate(groups):
        label = _escape(names.get(g, g))
        m.append(f'  B{i}["Block {i+1}: {label}"]')
    for i in range(len(groups) - 1):
        m.append(f'  B{i} -->|"t={block_size * (i + 1)} checkpoint"| B{i+1}')
    m.append('  SEED["Seed harness c_0"] -.-> B0')
    m.append("  classDef seed fill:#2a3140,stroke:#8b94a5,color:#d7dee9;")
    m.append("  classDef block fill:#16283a,stroke:#5aa9ff,color:#d7dee9;")
    m.append("  class SEED seed;")
    if groups:
        m.append("  class " + ",".join(f"B{i}" for i in range(len(groups))) + " block;")
    return {
        "id": "stream",
        "title": "Stream the agent will face",
        "mermaid": "\n".join(m),
        "notes": [f"{len(groups)} families, {block_size} training tasks each. "
                  f"The agent is never told the family, the boundaries, or that it is drifting."],
    }


def pattern_diagram(pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The mermaid diagram that ships with a catalog pattern, if it has one."""
    src = (pattern or {}).get("diagram")
    if not src:
        return None
    return {
        "id": f"pattern-{pattern.get('id', 'unknown')}",
        "title": pattern.get("name", pattern.get("id", "Pattern")),
        "mermaid": src,
        "notes": [pattern.get("summary", "")] if pattern.get("summary") else [],
    }


def category_diagrams(catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The per-category diagrams from the catalog."""
    out = []
    for cat, src in (catalog.get("categoryDiagrams") or {}).items():
        if not src:
            continue
        out.append({
            "id": f"category-{cat}",
            "title": f"{cat.capitalize()} patterns",
            "mermaid": src,
            "notes": [],
        })
    return out