"""
The harness: everything around a frozen model.

Following arXiv:2610.25858, a policy is a frozen LLM plus a harness. Adaptation
happens by rewriting the harness, never the weights:

    c_{t+1} = f(c_t, H_t)

The harness carries four writable surfaces:

  system_prompt   - the base instructions (stable, rarely edited)
  strategy_prompt - a shared/global block; GEPA rewrites this in place
  memory          - an external store of distilled experiences; ReasoningBank reads it
  skills          - a skill library; Meta-Harness edits this plus the controller
  controller      - the orchestration code; Meta-Harness rewrites it

`scoped_sections` is the extra surface the paper argues for: per-task-group
prompt sections that are append-only, so a later group cannot overwrite or
compress an earlier group's rules.

A harness is snapshot to JSON at every checkpoint so that runs can be diffed
and the mechanism diagnostics can attribute prompt characters to the group that
wrote them.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

_PROVENANCE_RE = re.compile(r"^\[[^\]]+\]\s*")


def strip_provenance(text: str) -> str:
    """Remove the `[owner]` provenance prefix the evolvers write on each rule."""
    if not text:
        return ""
    return "\n".join(_PROVENANCE_RE.sub("", ln) for ln in text.splitlines())


@dataclass
class MemoryItem:
    """A distilled experience. The unit of the experience-augmentation paradigm."""
    id: str
    group: str
    query: str
    reflection: str
    embedding: List[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    success: bool = True
    # Behavioural fingerprint of the workflow this item encodes. Used by the
    # scoped evolver to separate "same topic, opposite workflow" items.
    workflow: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("embedding", None)  # vectors live in the memory index, not the snapshot
        return d


@dataclass
class Harness:
    system_prompt: str = ""
    strategy_prompt: str = ""
    memory: List[MemoryItem] = field(default_factory=list)
    skills: List[Dict[str, Any]] = field(default_factory=list)
    controller: str = ""
    scoped_sections: Dict[str, str] = field(default_factory=dict)
    conflict_ledger: List[Dict[str, Any]] = field(default_factory=list)
    # The rules that make up the shared strategy prompt, each one traceable to
    # the lesson that wrote it. GEPA compresses this list, which is how an
    # earlier family's rule gets overwritten by a later one.
    strategy_rules: List[Dict[str, Any]] = field(default_factory=list)
    # Structured index of the lessons the harness currently carries. It is kept
    # in sync with the rendered surfaces (strategy_prompt / memory /
    # scoped_sections) so that a snapshot is a faithful, diffable picture of
    # what the agent has learned.
    lesson_index: List[Dict[str, Any]] = field(default_factory=list)
    version: int = 0
    created_at: float = field(default_factory=time.time)

    # ---------------------------------------------------------------- helpers

    def touch(self) -> None:
        self.version += 1

    def clone(self) -> "Harness":
        return Harness(
            system_prompt=self.system_prompt,
            strategy_prompt=self.strategy_prompt,
            memory=list(self.memory),
            skills=json.loads(json.dumps(self.skills)),
            controller=self.controller,
            scoped_sections=dict(self.scoped_sections),
            conflict_ledger=json.loads(json.dumps(self.conflict_ledger)),
            strategy_rules=json.loads(json.dumps(self.strategy_rules)),
            lesson_index=json.loads(json.dumps(self.lesson_index)),
            version=self.version,
            created_at=self.created_at,
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "strategy_prompt": self.strategy_prompt,
            "memory": [m.to_dict() for m in self.memory],
            "memory_count": len(self.memory),
            "skills": self.skills,
            "controller": self.controller,
            "scoped_sections": self.scoped_sections,
            "conflict_ledger": self.conflict_ledger,
            "strategy_rules": self.strategy_rules,
            "lesson_index": self.lesson_index,
            "version": self.version,
        }

    @classmethod
    def from_snapshot(cls, snap: Dict[str, Any]) -> "Harness":
        h = cls(
            system_prompt=snap.get("system_prompt", ""),
            strategy_prompt=snap.get("strategy_prompt", ""),
            skills=snap.get("skills", []),
            controller=snap.get("controller", ""),
            scoped_sections=snap.get("scoped_sections", {}),
            conflict_ledger=snap.get("conflict_ledger", []),
            strategy_rules=snap.get("strategy_rules", []),
            version=snap.get("version", 0),
        )
        h.memory = [MemoryItem(**{k: v for k, v in m.items() if k != "meta"} |
                                {"meta": m.get("meta", {})}) for m in snap.get("memory", [])]
        h.lesson_index = snap.get("lesson_index", [])
        return h

    # ------------------------------------------------- prompt assembly for a run

    def build_system_prompt(self, group_hint: Optional[str] = None) -> str:
        """
        Compose the prompt actually sent to the model.

        Note the asymmetry the paper's findings hinge on:
          * `strategy_prompt` is GLOBAL — it is injected for every task, every
            group, forever. That is precisely how a rule learned in one block
            leaks into an unseen block (pre-block interference).
          * `scoped_sections` are per-group. A scoped evolver only injects the
            section matching the group it believes the task belongs to, so an
            unseen group sees nothing.
        """
        parts: List[str] = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        if self.strategy_prompt:
            parts.append(
                "## Strategy (applies to every task)\n" + strip_provenance(self.strategy_prompt).strip()
            )
        if self.scoped_sections:
            if group_hint and group_hint in self.scoped_sections:
                parts.append(
                    f"## Task-family notes ({group_hint})\n"
                    + self.scoped_sections[group_hint].strip()
                )
            else:
                names = ", ".join(sorted(self.scoped_sections.keys()))
                parts.append(
                    "## Task-family notes\n"
                    "The following notes exist for other task families and do not "
                    f"apply here: {names}. Do not act on them."
                )
        if self.skills:
            skill_txt = "\n".join(
                f"- {s.get('name', 'skill')}: {s.get('description', '')}" for s in self.skills
            )
            parts.append("## Skills\n" + skill_txt)
        if self.controller:
            parts.append("## Controller notes\n" + self.controller.strip())
        return "\n\n".join(parts).strip()

    # ------------------------------------------- prompt attribution diagnostics

    def strategy_attribution(self) -> Dict[str, float]:
        """
        Share of strategy-prompt characters attributable to each group.

        The paper tracks this (Figure 3a,b) to show overwriting: a group's
        share falls when later blocks compress or replace its rules. The
        rendered prompt carries a `[owner]` provenance prefix on each rule so
        the share is attributable; `strip_provenance` removes it before the
        prompt reaches the model.
        """
        total = len(self.strategy_prompt or "")
        if total == 0:
            return {}
        shares: Dict[str, float] = {}
        for line in (self.strategy_prompt or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            owner = None
            if stripped.startswith("[") and "]" in stripped:
                owner = stripped[1:stripped.index("]")]
            if owner is None:
                lowered = stripped.lower()
                for key in self.scoped_sections:
                    if key.lower() in lowered:
                        owner = key
                        break
            if owner is None:
                owner = "_shared"
            shares[owner] = shares.get(owner, 0.0) + len(stripped)
        return {k: round(100.0 * v / total, 2) for k, v in shares.items()}

    def scoped_attribution(self) -> Dict[str, float]:
        """Character share of each scoped section relative to total prompt mass."""
        total = sum(len(v or "") for v in self.scoped_sections.values())
        if total == 0:
            return {}
        return {k: round(100.0 * len(v or "") / total, 2) for k, v in self.scoped_sections.items()}


def seed_harness(system_prompt: str, strategy_prompt: str = "",
                 controller: str = "", skills: Optional[List[Dict[str, Any]]] = None) -> Harness:
    """The seed harness c_0. Every method in a comparison starts from this."""
    return Harness(
        system_prompt=system_prompt,
        strategy_prompt=strategy_prompt,
        controller=controller,
        skills=skills or [],
    )


def new_id(prefix: str = "", rng=None) -> str:
    """
    A fresh id. Pass a seeded RNG for bit-reproducible runs: the policy's
    perturbation coins are salted with lesson ids, so uuid randomness would
    otherwise leak non-determinism into every metric even with fixed seeds.
    """
    if rng is not None:
        return f"{prefix}{rng.getrandbits(48):012x}"
    return f"{prefix}{uuid.uuid4().hex[:12]}"
