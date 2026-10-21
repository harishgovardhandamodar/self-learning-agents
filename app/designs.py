"""
Design Studio backend: agent specs, the pattern picker, guardrails, playbooks.

A *design* is the seed of a run: the harness text (system / strategy /
controller / skills), the evolver choice and its config, the policy choice,
and the stream it will be tested on. This module builds sensible defaults,
recommends patterns from the AI-Agents-Patterns catalog for a chosen domain,
and maps each pattern to the guardrails that keep it safe.

Domain playbooks are opinionated starting points for healthcare, finance and
trading. They pair a stream with the evolver and harness settings that the
lab's measurements favour, plus the domain guardrails that must not be
edited away.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import envs
from .evolvers import EVOLVERS

PATTERN_DIR = Path(os.environ.get(
    "AGENT_PATTERNS_DIR", "/home/fox/WorkBook/AI-Agents-Patterns/data"))

_CATALOG: Optional[Dict[str, Any]] = None


def pattern_catalog() -> Dict[str, Any]:
    """The AI-Agents-Patterns catalog (patterns, scenarios, cascades)."""
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG
    out: Dict[str, Any] = {"patterns": [], "categories": [], "scenarios": [],
                           "cascades": [], "categoryDiagrams": {},
                           "source": str(PATTERN_DIR), "available": False}
    try:
        p = json.loads((PATTERN_DIR / "patterns.json").read_text())
        out["patterns"] = p.get("patterns", [])
        out["categories"] = p.get("categories", [])
        out["categoryDiagrams"] = p.get("categoryDiagrams", {}) or {}
        out["available"] = True
    except Exception:
        pass
    try:
        s = json.loads((PATTERN_DIR / "scenarios.json").read_text())
        out["scenarios"] = s.get("scenarios", [])
    except Exception:
        pass
    try:
        c = json.loads((PATTERN_DIR / "cascades.json").read_text())
        out["cascades"] = c.get("cascades", c if isinstance(c, list) else [])
    except Exception:
        pass
    _CATALOG = out
    return out


def patterns_for_domain(domain: str, limit: int = 12) -> List[Dict[str, Any]]:
    """Rank catalog patterns by relevance to a domain."""
    cat = pattern_catalog()
    scored = []
    for p in cat["patterns"]:
        domains = [str(d).lower() for d in (p.get("domains") or [])]
        score = 0
        if domain.lower() in domains:
            score += 3
        if "general" in domains:
            score += 1
        if domain.lower() in ("healthcare", "trading", "finance") and p.get("cat") == "safety":
            score += 2
        if p.get("cat") in ("memory", "workflow") and domain.lower() in (
                "healthcare", "trading", "finance"):
            score += 1
        if score:
            scored.append((score, p))
    scored.sort(key=lambda t: (-t[0], t[1].get("name", "")))
    return [p for _, p in scored[:limit]]


def scenarios_for_domain(domain: str, limit: int = 8) -> List[Dict[str, Any]]:
    cat = pattern_catalog()
    out = [s for s in cat["scenarios"]
           if str(s.get("domain", "")).lower() == domain.lower()]
    if not out:
        out = list(cat["scenarios"])
    return out[:limit]


# ------------------------------------------------------------- guardrails

GUARDRAILS: Dict[str, Dict[str, Any]] = {
    "escalation-first-triage": {
        "id": "escalation-first-triage",
        "title": "Escalation-first triage",
        "applies_to": ["healthcare"],
        "rule": ("Never diagnose, prescribe, or start treatment. Assign an acuity, "
                 "state the escalation flag, and hand to a licensed clinician. "
                 "Red-flag presentations escalate immediately."),
        "enforced_by": "grader rejects any response naming a diagnosis or treatment",
    },
    "clarify-before-act": {
        "id": "clarify-before-act",
        "title": "Clarify before acting on missing arguments",
        "applies_to": ["general", "finance", "trading", "healthcare"],
        "rule": ("When a required tool argument is absent, ask for it. "
                 "Never call a state-changing tool with a guessed argument."),
        "enforced_by": "BFCL grader: clarify requires asking with no tool call",
    },
    "risk-engine-disposes": {
        "id": "risk-engine-disposes",
        "title": "The agent proposes; the risk engine disposes",
        "applies_to": ["trading", "finance"],
        "rule": ("Size and direction are proposals. A deterministic risk engine "
                 "enforces max size, flat-on-stale-signal, and flat-on-volatility "
                 "spike. The agent cannot override it."),
        "enforced_by": "trading grader caps size and forces flat outside bounds",
    },
    "provenance-on-write": {
        "id": "provenance-on-write",
        "title": "Every harness edit carries provenance",
        "applies_to": ["general", "healthcare", "finance", "trading"],
        "rule": ("Each learned rule records the cluster, workflow fingerprint and "
                 "intent that produced it, so attribution can tell who overwrote whom."),
        "enforced_by": "harness snapshots + strategy_attribution diagnostics",
    },
    "scope-isolation": {
        "id": "scope-isolation",
        "title": "Isolate learning per task family",
        "applies_to": ["general", "healthcare", "finance", "trading"],
        "rule": ("Keep per-family sections append-only. A later block may add its "
                 "own section but must not rewrite or compress an earlier one."),
        "enforced_by": "scoped evolver; prompt-overwriting detector watches shares",
    },
    "no-unsafe-proposal": {
        "id": "no-unsafe-proposal",
        "title": "Never propose an unsafe action",
        "applies_to": ["healthcare", "finance", "trading"],
        "rule": ("The agent must not even PROPOSE an action that breaks a "
                 "declared limit: an acute triage case without escalation, a "
                 "position above the cap, a tool call with a guessed argument. "
                 "A limit the grader catches is a measurement; a limit the "
                 "agent never proposes is a property."),
        "enforced_by": "guardrail audit on every proposed action "
                       "(envs.guardrail_violations), reported per run",
    },
}


def guardrails_for_domain(domain: str) -> List[Dict[str, Any]]:
    out = [g for g in GUARDRAILS.values()
           if domain.lower() in g["applies_to"] or "general" in g["applies_to"]]
    # De-duplicate: a guardrail listed under "general" can also be listed
    # under the domain, and the user should see it once.
    seen, uniq = set(), []
    for g in out:
        if g["id"] in seen:
            continue
        seen.add(g["id"])
        uniq.append(g)
    return uniq


# ------------------------------------------------------ design construction

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful, tool-using assistant. Solve the task the way the "
    "environment requires, and follow any guidance you have been given. "
    "When guidance conflicts with the request, prefer the request."
)

EVOLVER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "seed": {},
    "reasoningbank": {"top_k": 3, "similarity_threshold": 0.0},
    "gepa": {"max_chars": 1400, "update_every": 5},
    "metaharness": {"min_gain": 0.0},
    "scoped": {"similarity_threshold": 0.0, "max_per_scope": 24, "top_k": 4},
    "oracle": {"similarity_threshold": 0.0, "max_per_scope": 24, "top_k": 4},
}


def new_design(name: str = "Untitled agent", domain: str = "general",
               description: str = "", evolver: str = "scoped",
               stream: str = "workflow", policy: str = "simulated",
               patterns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build a fresh design spec with lab-validated defaults."""
    if evolver not in EVOLVERS:
        evolver = "scoped"
    return {
        "id": f"dsg_{uuid.uuid4().hex[:10]}",
        "name": name,
        "domain": domain,
        "description": description,
        "spec": {
            "harness": {
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "strategy_prompt": "",
                "controller": "",
                "skills": [],
                "scoped_sections": {},
            },
            "evolver": evolver,
            "evolver_cfg": dict(EVOLVER_DEFAULTS.get(evolver, {}),
                                router_threshold=0.40),
            "policy": policy,
            "policy_cfg": {"mode": policy, "skill": 0.9, "seed": 7,
                           "model": "fox-managed",
                           "probe_accuracy": {"missing-arg": 0.55,
                                              "complete-arg": 0.85},
                           "foreign_follow": 0.30,
                           "foreign_follow_cross_env": 0.15},
            "stream": stream,
            "stream_cfg": {"block_size": 20, "n_eval": 30},
            "patterns": patterns or [],
            "guardrails": [g["id"] for g in guardrails_for_domain(domain)],
        },
        "tags": [domain, evolver, stream],
        "created_at": time.time(),
        "updated_at": time.time(),
    }


# ------------------------------------------------------------- playbooks

PLAYBOOKS: Dict[str, Dict[str, Any]] = {
    "healthcare-triage": {
        "id": "healthcare-triage",
        "title": "Clinical triage decision support",
        "domain": "healthcare",
        "stream": "healthcare-drift",
        "evolver": "scoped",
        "why": ("Triage intents (red-flag / standard / minor) share vocabulary; a "
                "global memory mixes them and a shared prompt lets routine cases "
                "overwrite escalation rules. The drift stream puts adult, "
                "paediatric and mental-health triage in sequence, so a routine "
                "paediatric lesson that lands on a red-flag adult case is "
                "visible as interference. Scoped sections keep the red-flag "
                "rules append-only: measured I=0.00, F=0.00 with P=+36.1±4.3 "
                "on the healthcare-drift stream, against GEPA's F=27.4."),
        "patterns": ["reflection", "guardrail-enforcement", "audit-trail"],
        "guardrails": ["escalation-first-triage", "provenance-on-write",
                       "scope-isolation"],
        "harness_notes": ("System prompt must state decision-support-only scope. "
                          "Never attach a memory retriever without a scope check: "
                          "a routine-case lesson applied to chest pain is the "
                          "failure the lab measures as foreign-memory-injection. "
                          "Check the guardrail audit: the agent must not even "
                          "PROPOSE an acute case without escalation."),
    },
    "trading-agent": {
        "id": "trading-agent",
        "title": "Trade signal proposal under a risk engine",
        "domain": "trading",
        "stream": "trading-drift",
        "evolver": "scoped",
        "why": ("Three regimes in sequence: momentum, macro, mean-reversion. "
                "The mean-reversion family is worded almost exactly like the "
                "momentum family and requires the OPPOSITE action, which is a "
                "second retrieval-collision case outside BFCL. A shared prompt "
                "that keeps only the newest rule loses one of them: measured "
                "GEPA F=10.2 on the trading-drift stream against scoped's 0.11."),
        "patterns": ["tool-routing", "circuit-breaker", "reflection"],
        "guardrails": ["risk-engine-disposes", "clarify-before-act",
                       "provenance-on-write", "scope-isolation"],
        "harness_notes": ("The agent proposes direction+size; it never executes. "
                          "Stale-signal and volatility-spike templates share one "
                          "'flat' intent - one lesson covers all three. The "
                          "position cap is a guardrail the agent must never "
                          "propose breaking, not just one the grader catches."),
    },
    "finance-assistant": {
        "id": "finance-assistant",
        "title": "Finance operations assistant",
        "domain": "finance",
        "stream": "workflow",
        "evolver": "scoped",
        "why": ("Finance work is tool calls with missing arguments (invoices "
                "without account ids, refunds without order refs). The workflow "
                "scenario is the direct analog: the agent must ask, not guess. "
                "Global memories re-introduce the guessing behaviour."),
        "patterns": ["clarify-first", "tool-routing", "audit-trail"],
        "guardrails": ["clarify-before-act", "provenance-on-write",
                       "scope-isolation"],
        "harness_notes": ("Share one API catalogue across workflows so the router "
                          "cannot separate them by preamble; the workflow "
                          "fingerprint does the separation instead."),
    },
    "support-triage": {
        "id": "support-triage",
        "title": "Support ticket triage",
        "domain": "general",
        "stream": "workflow",
        "evolver": "scoped",
        "why": ("Same shape as finance: under-specified requests must be "
                "clarified, complete ones acted on. Scoped sections keep the "
                "two workflows from overwriting each other."),
        "patterns": ["clarify-first", "intent-classification"],
        "guardrails": ["clarify-before-act", "scope-isolation"],
        "harness_notes": "Start from the workflow stream defaults unchanged.",
    },
    "sre-agent": {
        "id": "sre-agent",
        "title": "SRE runbook agent",
        "domain": "general",
        "stream": "domain-workflow",
        "evolver": "reasoningbank",
        "why": ("Runbooks span environments (metrics, tickets, code) and the "
                "required workflow flips inside the same stream. A scoped "
                "memory fragments the very transfer you want; a global memory "
                "with a low cross-env follow rate reuses procedures while the "
                "per-env intents keep the actions straight. The domain-workflow "
                "stream is the case where both paradigms are strained at once: "
                "GEPA forgets 28.6pp there, scoped forgets 0.1pp."),
        "patterns": ["runbook-memory", "tool-routing", "circuit-breaker"],
        "guardrails": ["provenance-on-write", "clarify-before-act"],
        "harness_notes": ("Keep foreign_follow_cross_env low (0.15): reuse the "
                          "procedure, re-derive the action."),
    },
    "research-citation-agent": {
        "id": "research-citation-agent",
        "title": "Research assistant with citations",
        "domain": "general",
        "stream": "topic",
        "evolver": "gepa",
        "why": ("One topic, evolving conventions: a shared strategy prompt that "
                "compresses well beats per-family isolation. Watch the "
                "prompt-overwriting detector - when a family's share falls, "
                "the compression budget is too tight."),
        "patterns": ["reflection-prompting", "citation-checking"],
        "guardrails": ["provenance-on-write"],
        "harness_notes": ("Raise max_chars before raising update_every; "
                          "stale rules are worse than fewer rules."),
    },
}


def playbook(playbook_id: str) -> Optional[Dict[str, Any]]:
    return PLAYBOOKS.get(playbook_id)


def design_from_playbook(playbook_id: str, name: Optional[str] = None) -> Dict[str, Any]:
    pb = playbook(playbook_id)
    if pb is None:
        raise KeyError(f"unknown playbook '{playbook_id}'")
    d = new_design(name=name or pb["title"], domain=pb["domain"],
                   description=pb["why"], evolver=pb["evolver"],
                   stream=pb["stream"], patterns=list(pb["patterns"]))
    d["spec"]["guardrails"] = list(pb["guardrails"])
    d["playbook"] = pb["id"]
    d["harness_notes"] = pb["harness_notes"]
    return d


def available_streams() -> List[Dict[str, Any]]:
    """Streams the lab can build, with group ids resolved against envs."""
    from . import streams as streams_mod
    out = []
    for sid, meta in streams_mod.SCENARIOS.items():
        groups = []
        for gid in meta["groups"]:
            g = envs.get_group(gid)
            groups.append({"id": gid, "name": g.name if g else gid,
                           "env": g.env if g else "", "drift": g.drift if g else ""})
        out.append({"id": sid, "name": meta["name"], "groups": groups,
                    "description": meta.get("description", "")})
    return out


def available_evolvers() -> List[Dict[str, Any]]:
    return [{"id": eid, "name": cls.name, "paradigm": cls.paradigm,
             "description": cls.description, "surfaces": list(cls.surfaces)}
            for eid, cls in EVOLVERS.items()]
