"""
Run orchestration: drive a stream, evolve the harness, evaluate checkpoints.

The protocol follows the paper exactly:

  * one shared harness across the whole horizon
  * the agent is never told the group label, the block boundaries, or that it
    is on a drifting stream
  * binary task-success feedback after every training task
  * a checkpoint at the seed and at every block boundary
  * each frozen checkpoint is evaluated on the disjoint hold-out sets of
    *all* families in the stream, not just the active one
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import cost as cost_mod
from . import envs, metrics as metrics_mod
from .evolvers import Evolver, Lesson, get_evolver
from .harness import Harness, new_id, seed_harness
from .policy import PolicyResult, build_policy
from .retrieval import Router


@dataclass
class RunConfig:
    design: Dict[str, Any]
    stream: Any                       # streams.Stream
    evolver: str = "seed"
    evolver_cfg: Dict[str, Any] = field(default_factory=dict)
    policy_cfg: Dict[str, Any] = field(default_factory=dict)
    seed: int = 7
    eval_limit: Optional[int] = None   # cap hold-out size for quick runs
    max_episodes: Optional[int] = None  # cap training episodes (debug)

    @property
    def policy_mode(self) -> str:
        return (self.policy_cfg or {}).get("mode", "simulated")


ProgressCb = Optional[Callable[[Dict[str, Any]], None]]


def build_harness(design: Dict[str, Any]) -> Harness:
    """Materialise the seed harness c_0 from a design spec."""
    spec = design.get("spec") or design
    harness_cfg = spec.get("harness") or {}
    h = seed_harness(
        system_prompt=harness_cfg.get("system_prompt", ""),
        strategy_prompt=harness_cfg.get("strategy_prompt", ""),
        controller=harness_cfg.get("controller", ""),
        skills=harness_cfg.get("skills") or [],
    )
    for group_id, note in (harness_cfg.get("scoped_sections") or {}).items():
        h.scoped_sections[group_id] = note
    return h


def default_system_prompt() -> str:
    return (
        "You are a careful, tool-using assistant. Solve the task the way the "
        "environment requires, and follow any guidance you have been given. "
        "When guidance conflicts with the request, prefer the request."
    )


async def _eval_group(harness: Harness, group: envs.TaskGroup, tasks: List[envs.Task],
                       evolver: Evolver, policy, router: Router,
                       cost: Optional[Dict[str, Any]] = None,
                       ) -> Dict[str, Any]:
    """Evaluate a frozen harness on one family's hold-out set."""
    correct = 0
    events: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []
    for task in tasks:
        inj = evolver.inject(harness, task, router)
        result: PolicyResult = await policy.act(task, inj, harness, router)
        detail = envs.grade_detail(task, result.output)
        if detail["success"]:
            correct += 1
        owner = ""
        if inj.lessons:
            owner = inj.lessons[0].meta.get("group", "") or ""
        # Guardrail audit: did the agent PROPOSE a breaking action, not just
        # did the grader catch a bad answer?
        bad = envs.guardrail_violations(task, result.action or {}, result.output)
        if bad:
            violations.append({"group": group.id, "violations": bad,
                               "intent": (result.trace or {}).get("intent", "")})
        if cost is not None:
            used = bool((result.trace or {}).get("governed_by")) and \
                (result.trace or {}).get("governed_by") != "prior"
            cost_mod.account_episode(cost, inj, used)
        events.append({
            "t": None, "group": group.id, "source": inj.source, "owner": owner,
            "similarity": inj.similarity, "success": detail["success"],
            "note": inj.note, "lesson_scope": inj.lessons[0].scope if inj.lessons else "",
        })
    acc = 100.0 * correct / len(tasks) if tasks else 0.0
    return {"accuracy": acc, "correct": correct, "n": len(tasks),
            "events": events, "violations": violations}


async def execute(cfg: RunConfig, on_progress: ProgressCb = None,
                  llm=None, should_cancel: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """
    Run one stream. Returns the full result payload.

    The result contains the metrics block, the checkpoints (with harness
    snapshots and mechanism diagnostics), and the retrieved lessons at each
    checkpoint so the UI can show *why* a run behaved as it did.
    """
    stream = cfg.stream
    harness = build_harness(cfg.design)
    evolver_cfg = dict(cfg.evolver_cfg or {})
    # RunConfig.seed drives lesson-id generation unless the caller pinned an
    # evolver-level seed explicitly. Either way every run is reproducible.
    evolver_cfg.setdefault("seed", cfg.seed)
    evolver = get_evolver(cfg.evolver, evolver_cfg)
    policy = build_policy(cfg.policy_cfg, llm=llm)

    # The router threshold separates the SQL families (cross-group cosine
    # 0.035-0.097) while deliberately MERGING the two BFCL families
    # (cross-group cosine 0.655-0.706): they share one API catalogue, and the
    # paper's finding is that no surface feature can separate them. Any value
    # in roughly (0.10, 0.65) gives the same partition.
    router_threshold = float(cfg.evolver_cfg.get("router_threshold", 0.40))
    router = Router(threshold=router_threshold)

    eval_tasks = stream.eval_tasks()
    if cfg.eval_limit:
        eval_tasks = {g: ts[: cfg.eval_limit] for g, ts in eval_tasks.items()}
    validation_tasks = {
        g: ts for g, ts in stream.validation_tasks().items() if ts
    }

    checkpoints: List[Dict[str, Any]] = []
    all_events: List[Dict[str, Any]] = []
    all_violations: List[Dict[str, Any]] = []
    train_log: List[Dict[str, Any]] = []
    cost = cost_mod.empty_cost()
    started = time.time()

    def report(phase: str, done: int, total: int, message: str = "") -> None:
        if on_progress:
            on_progress({
                "phase": phase, "done": done, "total": total, "message": message,
                "progress": round(done / total, 4) if total else 0.0,
            })

    # ---------------------------------------------------- seed checkpoint
    report("seed-eval", 0, 1, "evaluating the seed harness")
    seed_metrics, seed_events, seed_violations = await _eval_all(
        harness, evolver, policy, router, eval_tasks, stream.groups, cost)
    all_events.extend(seed_events)
    all_violations.extend(seed_violations)
    checkpoints.append({
        "t": 0, "label": "seed", "kind": "seed",
        "metrics": seed_metrics,
        "diagnostics": evolver.diagnostics(harness),
        "snapshot": harness.snapshot(),
    })
    report("seed-eval", 1, 1, "seed baseline established")

    if should_cancel and should_cancel():
        return {"cancelled": True, "checkpoints": checkpoints}

    # ----------------------------------------------------------- the stream
    train_tasks = stream.training_tasks()
    if cfg.max_episodes:
        train_tasks = train_tasks[: cfg.max_episodes]
    total = len(train_tasks) + len(stream.blocks) + 1
    done = 1

    block_starts: Dict[str, int] = {}
    for idx, block in enumerate(stream.blocks):
        block_starts[block.group] = idx

    for t, task in enumerate(train_tasks, start=1):
        if should_cancel and should_cancel():
            return {"cancelled": True, "checkpoints": checkpoints}
        inj = evolver.inject(harness, task, router)
        result = await policy.act(task, inj, harness, router)
        detail = envs.grade_detail(task, result.output)
        success = detail["success"]

        await evolver.observe(harness, task, result.action, success, result.trace, router)
        if success:
            # Account what the harness had to write to keep this lesson.
            cost_mod.account_lesson(cost, _last_lesson_of(harness), harness)

        if len(train_log) < 400:
            train_log.append({
                "t": t, "group": task.group, "success": success,
                "source": inj.source, "reason": detail["reason"],
                "output": (result.output or "")[:280],
                "template": task.meta.get("tpl", ""),
            })

        if t % 10 == 0 or t == len(train_tasks):
            report("training", done, total,
                   f"block {stream.group_at(t)} - task {t}/{len(train_tasks)}")

        # block boundary -> checkpoint
        boundary = next((b for b in stream.blocks if b.end == t), None)
        if boundary is not None:
            done += 1
            report("block-end", done, total, f"validating block {boundary.group}")
            gate_ctx = {"validate": _make_validator(harness, evolver, policy, router,
                                                    validation_tasks.get(boundary.group, []))}
            gate = await evolver.on_block_end(harness, boundary, gate_ctx)

            done += 1
            report("eval", done, total, f"evaluating checkpoint at t={t}")
            cp_metrics, cp_events, cp_violations = await _eval_all(
                harness, evolver, policy, router, eval_tasks, stream.groups, cost)
            all_events.extend(cp_events)
            all_violations.extend(cp_violations)
            checkpoints.append({
                "t": t, "label": boundary.group, "kind": "eval",
                "metrics": cp_metrics,
                "diagnostics": {**evolver.diagnostics(harness), "gate": gate},
                "snapshot": harness.snapshot(),
            })

    # -------------------------------------------------------- final metrics
    m = metrics_mod.compute_metrics(checkpoints, stream)
    retrieval = metrics_mod.retrieval_diagnostics(all_events)
    attribution = metrics_mod.prompt_attribution(checkpoints)
    failures = metrics_mod.detect_failure_modes(m, retrieval, attribution)
    guardrails = _guardrail_summary(all_violations)
    cost_out = cost_mod.finalise(cost)
    ledger = cost_mod.efficiency_ledger(cost_out, m.get("terminal_advantage", 0.0))

    return {
        "cancelled": False,
        "checkpoints": checkpoints,
        "metrics": m,
        "retrieval": retrieval,
        "attribution": attribution,
        "failure_modes": failures,
        "guardrails": guardrails,
        "cost": cost_out,
        "efficiency": ledger,
        "train_log": train_log,
        "router": router.state(),
        "final_snapshot": harness.snapshot(),
        "elapsed_s": round(time.time() - started, 2),
        "policy": cfg.policy_mode,
        "model": (cfg.policy_cfg or {}).get("model", ""),
    }


def _last_lesson_of(harness: Harness) -> Any:
    """The most recent lesson the harness recorded, if any."""
    idx = harness.lesson_index or []
    return Lesson.from_dict(idx[-1]) if idx else None


def _guardrail_summary(violations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Which guardrails the agent PROPOSED breaking, and how often."""
    counts: Dict[str, int] = {}
    per_group: Dict[str, Dict[str, int]] = {}
    for v in violations:
        for name in v.get("violations", []):
            counts[name] = counts.get(name, 0) + 1
            g = v.get("group", "?")
            per_group.setdefault(g, {})
            per_group[g][name] = per_group[g].get(name, 0) + 1
    return {
        "total": sum(counts.values()),
        "by_violation": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "by_group": per_group,
        "clean": not counts,
    }


async def _eval_all(harness: Harness, evolver: Evolver, policy, router: Router,
                    eval_tasks: Dict[str, List[envs.Task]],
                    groups: List[str],
                    cost: Optional[Dict[str, Any]] = None) -> tuple:
    """Evaluate a frozen harness on every family's hold-out set."""
    # Evaluate against a COPY of the router: a hold-out query must not be
    # allowed to reshape the agent's own routing.
    eval_router = router.clone()
    metrics: Dict[str, float] = {}
    events: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []
    for g in groups:
        res = await _eval_group(harness, envs.get_group(g), eval_tasks.get(g, []),
                                evolver, policy, eval_router, cost)
        metrics[g] = res["accuracy"]
        events.extend(res["events"])
        violations.extend(res["violations"])
    return metrics, events, violations


def _make_validator(harness: Harness, evolver: Evolver, policy, router: Router,
                    tasks: List[envs.Task]):
    """Async validator used by the Meta-Harness gate: accuracy of a candidate harness."""
    async def validate(candidate: Harness) -> float:
        if not tasks:
            return 0.0
        eval_router = router.clone()
        correct = 0
        for task in tasks:
            inj = evolver.inject(candidate, task, eval_router)
            result = await policy.act(task, inj, candidate, eval_router)
            if envs.grade_detail(task, result.output)["success"]:
                correct += 1
        return correct / len(tasks)
    return validate


def new_run_id() -> str:
    return new_id("run_")
