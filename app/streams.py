"""
Task streams: drifting sequences of task groups.

The lab reproduces the four drift scenarios of arXiv:2610.25858 §3.1/§4.1:

  topic      - the environment and workflow stay fixed, the subject changes
               (3 SQL databases in sequence)
  workflow   - the environment and tool catalogue stay fixed, the required
               workflow flips (ask-for-clarification vs act-directly)
  domain     - the environment itself changes (science lab -> tool use -> SQL)
  compound   - topic, domain and workflow shifts occur in sequence

Each stream is a sequence of contiguous blocks. The agent is never told the
group label, the block boundaries, or that it is on a drifting stream.

Position conventions follow the paper: the block of family k occupies
positions sigma_k < t <= tau_k.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import envs


@dataclass
class Block:
    group: str
    start: int          # first position of the block (1-based, inclusive)
    end: int            # last position of the block (inclusive)
    n_train: int


@dataclass
class Stream:
    id: str
    name: str
    scenario: str
    blocks: List[Block]
    groups: List[str]
    n_eval: int = 50
    description: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    # Task-sampling seeds. Training and hold-out draws use disjoint offsets,
    # and both are seedable so an N-iteration sweep can vary the task draws
    # (not just the policy coins) while every single iteration stays exactly
    # reproducible. Defaults preserve the historical draws.
    train_offset: int = 0
    train_seed: int = 1
    eval_offset: int = 100_000
    eval_seed: int = 2
    # How the stream is ordered. "blocked" is one family at a time (the
    # paper's protocol). "interleaved" mixes families per episode, which is
    # what production traffic looks like. "recurring" brings a family back
    # every few blocks.
    order: str = "blocked"

    # ------------------------------------------------------------- geometry

    @property
    def T(self) -> int:
        return sum(b.n_train for b in self.blocks)

    @property
    def sigma(self) -> Dict[str, int]:
        """
        Last stream position before the family's block starts.

        A family that appears more than once (the `revisit` scenario) gets
        the sigma of its FIRST block: the pre-block interference of a family
        is measured against the state before the agent first saw it.
        """
        out: Dict[str, int] = {}
        prev = 0
        for b in self.blocks:
            if b.group not in out:
                out[b.group] = prev
            prev = b.end
        return out

    @property
    def tau(self) -> Dict[str, int]:
        """
        Last stream position of the family's block.

        For a repeated family this is the END of its LAST block: plasticity
        is "what did the agent end up able to do", and forgetting is measured
        from there. The first appearance is tracked separately in
        `first_tau` so "did it learn it the first time" stays measurable.
        """
        out: Dict[str, int] = {}
        for b in self.blocks:
            out[b.group] = b.end
        return out

    @property
    def first_tau(self) -> Dict[str, int]:
        """End of each family's FIRST block (for repeated families)."""
        out: Dict[str, int] = {}
        for b in self.blocks:
            out.setdefault(b.group, b.end)
        return out

    @property
    def revisits(self) -> Dict[str, List[int]]:
        """Families that appear more than once, with their block positions."""
        counts: Dict[str, int] = {}
        for b in self.blocks:
            counts[b.group] = counts.get(b.group, 0) + 1
        out: Dict[str, List[int]] = {}
        for b in self.blocks:
            if counts.get(b.group, 0) > 1:
                out.setdefault(b.group, []).append(b.end)
        return out

    @property
    def checkpoints(self) -> List[int]:
        """Post-seed checkpoint positions: every block boundary."""
        return [b.end for b in self.blocks]

    def group_at(self, t: int) -> str:
        for b in self.blocks:
            if b.start <= t <= b.end:
                return b.group
        return self.blocks[-1].group if self.blocks else ""

    def block_of(self, group: str) -> Optional[Block]:
        for b in self.blocks:
            if b.group == group:
                return b
        return None

    # ------------------------------------------------------------- sampling

    def training_tasks(self) -> List[envs.Task]:
        """
        The stream, in arrival order.

        For a blocked stream this is blocks of training tasks. For an
        interleaved stream the families are mixed per episode, so the router
        must choose under a changing composition and the memory must hold
        several families at once - which is what production traffic looks
        like and what the paper's blocked protocol cannot show.
        """
        tasks: List[envs.Task] = []
        groups = envs.group_by_id()
        if self.order == "interleaved":
            return self._interleaved_tasks(groups)
        for b in self.blocks:
            g = groups.get(b.group)
            if g is None:
                continue
            tasks.extend(g.sample(b.n_train, offset=self.train_offset,
                                  seed=self.train_seed))
        return tasks

    def _interleaved_tasks(self, groups: Dict[str, Any]) -> List[envs.Task]:
        """
        Mix the families per episode instead of in blocks.

        Each family still contributes its block's quota of training tasks, but
        the arrival order is round-robin over the families, so no family is
        "the current one". The block boundaries are kept for the metrics, but
        they no longer describe the arrival order.
        """
        per_family: Dict[str, List[envs.Task]] = {}
        for b in self.blocks:
            g = groups.get(b.group)
            if g is None or b.group in per_family:
                continue
            per_family[b.group] = g.sample(b.n_train, offset=self.train_offset,
                                            seed=self.train_seed)
        out: List[envs.Task] = []
        idx = 0
        while any(per_family.values()):
            for gid in list(per_family):
                if per_family[gid]:
                    out.append(per_family[gid].pop(0))
            idx += 1
            if idx > 10000:  # pragma: no cover - runaway guard
                break
        return out

    def eval_tasks(self) -> Dict[str, List[envs.Task]]:
        """Disjoint hold-out set per family."""
        groups = envs.group_by_id()
        out: Dict[str, List[envs.Task]] = {}
        for b in self.blocks:
            g = groups.get(b.group)
            if g is None or b.group in out:
                continue
            out[b.group] = g.sample(self.n_eval, offset=self.eval_offset,
                                    seed=self.eval_seed)
        return out

    def validation_tasks(self, n: int = 6) -> Dict[str, List[envs.Task]]:
        """Small validation split per family for the Meta-Harness gate."""
        groups = envs.group_by_id()
        out: Dict[str, List[envs.Task]] = {}
        for b in self.blocks:
            g = groups.get(b.group)
            if g is None:
                continue
            out[b.group] = g.sample(n, offset=50_000, seed=self.eval_seed + 1)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "scenario": self.scenario,
            "description": self.description, "T": self.T, "groups": self.groups,
            "n_eval": self.n_eval, "blocks": [vars(b) for b in self.blocks],
            "sigma": self.sigma, "tau": self.tau, "checkpoints": self.checkpoints,
            "train_offset": self.train_offset, "train_seed": self.train_seed,
            "eval_offset": self.eval_offset, "eval_seed": self.eval_seed,
            "meta": self.meta,
        }


# --------------------------------------------------------------- scenarios

SCENARIOS: Dict[str, Dict[str, Any]] = {
    # ---- the paper's four drift scenarios ---------------------------------
    "topic": {
        "name": "Topic drift",
        "description": "Environment and workflow fixed; the subject matter changes across three databases.",
        "groups": ["sql-formula1", "sql-codebase", "sql-thrombosis"],
        "drift": "topic",
    },
    "workflow": {
        "name": "Workflow drift",
        "description": "Environment and tool catalogue fixed; the correct workflow flips between blocks.",
        "groups": ["toolcall-clarify", "toolcall-act"],
        "drift": "workflow",
    },
    "domain": {
        "name": "Domain drift",
        "description": "The environment changes: interactive science lab, then tool use, then SQL.",
        "groups": ["science-assay", "toolcall-act", "sql-formula1"],
        "drift": "domain",
    },
    "compound": {
        "name": "Compound drift",
        "description": "Topic, domain and workflow shifts occur in sequence over a longer horizon.",
        "groups": ["sql-codebase", "sql-formula1", "toolcall-clarify", "toolcall-act"],
        "drift": "compound",
    },
    # ---- per-domain drift streams ------------------------------------------
    "healthcare-drift": {
        "name": "Clinical triage drift (healthcare)",
        "description": "Adult, paediatric and mental-health triage in sequence. A routine lesson "
                       "must never downgrade a red-flag escalation.",
        "groups": ["health-triage", "health-triage-pediatric", "health-triage-mental"],
        "drift": "domain",
    },
    "trading-drift": {
        "name": "Trading regime drift (trading)",
        "description": "Momentum, macro and mean-reversion regimes in sequence. The "
                       "mean-reversion family is worded like the momentum family and "
                       "requires the opposite action.",
        "groups": ["trade-signal", "trade-signal-macro", "trade-signal-meanrev"],
        "drift": "domain",
    },
    "science-drift": {
        "name": "Assay protocol drift (science)",
        "description": "Point-property assay, then a kinetics protocol. Different instruments, "
                       "different reporting convention.",
        "groups": ["science-assay", "science-kinetics"],
        "drift": "domain",
    },
    # ---- hard combinations the original suite never tested -----------------
    "domain-workflow": {
        "name": "Domain shift with a workflow flip",
        "description": "The environment changes AND the required workflow flips inside the "
                       "same stream. The hardest case: a rule from the science lab and a "
                       "rule from the tool bench both live in one harness.",
        "groups": ["science-assay", "toolcall-clarify", "toolcall-act"],
        "drift": "compound",
    },
    "revisit": {
        "name": "Revisit a learned family",
        "description": "A family is seen, then a different one, then the first again. Measures "
                       "whether competence returns after an interruption.",
        "groups": ["sql-formula1", "toolcall-act", "sql-formula1"],
        "drift": "compound",
        "revisit": True,
    },
    # ---- per-domain single-family controls --------------------------------
    "healthcare": {
        "name": "Clinical triage (domain playbook)",
        "description": "Healthcare playbook stream: clinical triage with escalation-first rules.",
        "groups": ["health-triage"],
        "drift": "domain",
    },
    "trading": {
        "name": "Trading signals (domain playbook)",
        "description": "Trading playbook stream: signal proposal under a deterministic risk engine.",
        "groups": ["trade-signal"],
        "drift": "domain",
    },
    "sql": {
        "name": "Text-to-SQL (single family control)",
        "description": "One SQL family. The per-environment learning baseline.",
        "groups": ["sql-formula1"],
        "drift": "domain",
    },
    "toolcall": {
        "name": "Tool calls (single family control)",
        "description": "One tool-call family. The per-environment learning baseline.",
        "groups": ["toolcall-act"],
        "drift": "domain",
    },
    "science": {
        "name": "Science assay (single family control)",
        "description": "One science family. The per-environment learning baseline.",
        "groups": ["science-assay"],
        "drift": "domain",
    },
    # ---- stream SHAPES the blocked protocol cannot express -----------------
    "interleaved": {
        "name": "Interleaved families",
        "description": "The families arrive mixed, not in blocks. The router must "
                       "route per request under a changing composition and the memory "
                       "must hold several families at once - the closest thing here to "
                       "production traffic.",
        "groups": ["sql-formula1", "toolcall-clarify", "health-triage", "trade-signal"],
        "drift": "compound",
        "order": "interleaved",
    },
    "recurring": {
        "name": "Recurring family",
        "description": "A daily-report family recurs every few blocks while other "
                       "families come and go. Tests recency as well as scope: a stored "
                       "experience can be in scope and still be stale.",
        "groups": ["sql-formula1", "toolcall-act", "sql-formula1", "health-triage",
                   "sql-formula1"],
        "drift": "compound",
    },
    "long-horizon": {
        "name": "Long horizon (8 blocks)",
        "description": "Eight blocks of drifting families. Interference and forgetting "
                       "compound with horizon length, and a short stream cannot show "
                       "whether degradation is gradual or sudden.",
        "groups": ["sql-formula1", "sql-codebase", "sql-thrombosis", "science-assay",
                   "toolcall-clarify", "toolcall-act", "health-triage", "trade-signal"],
        "drift": "compound",
    },
    "hard-order": {
        "name": "Adversarial ordering",
        "description": "The confusable pair adjacent, the hardest family first. A method "
                       "that only wins on a convenient ordering has not been shown to be "
                       "better.",
        "groups": ["toolcall-clarify", "toolcall-act", "health-triage",
                   "health-triage-pediatric", "trade-signal", "trade-signal-meanrev"],
        "drift": "compound",
    },
}

# Permuted orderings of the topic families, for the ordering-control
# experiment: a method that only wins on one ordering has not been shown to
# be better, only to be lucky.
TOPIC_PERMUTATIONS: Dict[str, List[str]] = {
    "topic-order-a": ["sql-formula1", "sql-codebase", "sql-thrombosis"],
    "topic-order-b": ["sql-codebase", "sql-thrombosis", "sql-formula1"],
    "topic-order-c": ["sql-thrombosis", "sql-formula1", "sql-codebase"],
}


def build_stream(scenario: str, block_size: int = 50, n_eval: int = 50,
                 groups: Optional[List[str]] = None,
                 stream_id: Optional[str] = None,
                 name: Optional[str] = None,
                 train_seed: int = 1, eval_seed: int = 2,
                 train_offset: int = 0, eval_offset: int = 100_000) -> Stream:
    """Build a stream for a named scenario (or an explicit group list)."""
    if groups:
        gids = list(groups)
        sc = "custom"
        meta = SCENARIOS.get(scenario, {})
        title = name or f"Custom stream ({len(gids)} groups)"
    else:
        meta = SCENARIOS.get(scenario)
        if meta is None:
            raise KeyError(f"unknown scenario '{scenario}'. known: {sorted(SCENARIOS)}")
        gids = list(meta["groups"])
        sc = scenario
        title = name or meta["name"]

    blocks: List[Block] = []
    pos = 0
    for gid in gids:
        pos += 1
        blocks.append(Block(group=gid, start=pos, end=pos + block_size - 1, n_train=block_size))
        pos += block_size - 1

    return Stream(
        id=stream_id or f"{sc}-b{block_size}",
        name=title,
        scenario=sc,
        blocks=blocks,
        groups=gids,
        n_eval=n_eval,
        description=meta.get("description", ""),
        meta={"drift": meta.get("drift", "custom"), "block_size": block_size},
        train_offset=train_offset,
        train_seed=train_seed,
        eval_offset=eval_offset,
        eval_seed=eval_seed,
        order=meta.get("order", "blocked"),
    )


def list_scenarios() -> List[Dict[str, Any]]:
    out = []
    for key, meta in SCENARIOS.items():
        out.append({
            "id": key, "name": meta["name"], "description": meta["description"],
            "groups": meta["groups"], "drift": meta.get("drift", ""),
            "group_names": [
                (envs.get_group(g).name if envs.get_group(g) else g) for g in meta["groups"]
            ],
        })
    return out


def get_stream(stream_id: str, block_size: int = 50, n_eval: int = 50,
               train_seed: int = 1, eval_seed: int = 2) -> Optional[Stream]:
    """Resolve a stream id: 'topic', 'workflow', ... or 'topic-b20' for a size."""
    base = stream_id
    size = block_size
    if "-" in stream_id and stream_id.rsplit("-b", 1)[-1].isdigit():
        base, size_str = stream_id.rsplit("-b", 1)
        size = int(size_str)
    if base not in SCENARIOS:
        return None
    return build_stream(base, block_size=size, n_eval=n_eval,
                        train_seed=train_seed, eval_seed=eval_seed)


def stream_geometry(stream: Stream) -> Dict[str, Any]:
    """A compact description of the block layout, for the UI."""
    sigma, tau = stream.sigma, stream.tau
    return {
        "T": stream.T,
        "K": len(stream.groups),
        "blocks": [
            {
                "group": b.group,
                "name": (envs.get_group(b.group).name if envs.get_group(b.group) else b.group),
                "start": b.start, "end": b.end, "n_train": b.n_train,
                "sigma": sigma.get(b.group), "tau": tau.get(b.group),
            }
            for b in stream.blocks
        ],
        "checkpoints": stream.checkpoints,
    }
