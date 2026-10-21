"""
Evolvers: the functions that rewrite a harness from experience.

    c_{t+1} = f(c_t, H_t)

Four evolvers are implemented, spanning the two paradigms the paper contrasts
and the three methods it evaluates, plus one that tests the paper's own
recommendation.

  seed           no adaptation. The baseline c_0.
  reasoningbank  experience augmentation. One memory item per episode, top-k
                 retrieval by query similarity. (Ouyang et al., 2025)
  gepa           self-modification of a SHARED strategy prompt, with reflective
                 mutation over collected rollouts and compression of the
                 prompt. (Agrawal et al., 2025)
  metaharness    self-modification of the whole workspace - prompt, memory,
                 skills and controller - accepted only when it improves
                 validation accuracy on the active sub-block. (Lee et al., 2026)
  scoped         experience augmentation with task-scoped routing and workflow
                 conditioning. This is the lab's proposal for the gap the paper
                 identifies: "mechanisms that isolate task-specific updates in
                 the harness and condition retrieval beyond surface query
                 similarity."

The same evolver drives both execution policies:

  * simulated  - a deterministic policy with a noisy skill prior. Lets you
                 design and compare harness mechanics in seconds, with no GPU.
  * live       - a ReAct-style agent calling the local LLM through
                 fox-services.

Both policies consume the same `Injection` produced by `inject()`, so an
evolver's behaviour is identical in the two modes; only the policy's
competence differs.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .harness import Harness, MemoryItem, new_id, strip_provenance
from .retrieval import Router, probe_workflow, workflow_descriptor


@dataclass
class Lesson:
    """One distilled experience, with the scope it is allowed to act in."""
    id: str
    owner: str                     # cluster id (scoped) or group descriptor (global)
    scope: str                     # "global" or "<cluster>|<workflow-fingerprint>"
    env: str
    template_id: str
    query: str
    action: Dict[str, Any]
    text: str
    success: bool = True
    created_at: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "owner": self.owner, "scope": self.scope, "env": self.env,
            "template_id": self.template_id, "query": self.query, "action": self.action,
            "text": self.text, "success": self.success, "created_at": self.created_at,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Lesson":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Injection:
    """What the evolver injects into the prompt for one task."""
    text: str = ""
    lessons: List[Lesson] = field(default_factory=list)
    source: str = "none"
    retrieved_from: str = ""
    similarity: float = 0.0
    note: str = ""


# --------------------------------------------------------------- lesson text

def action_lesson_text(action: Dict[str, Any], task_desc: str = "") -> str:
    """Render a lesson as natural-language guidance for the prompt."""
    kind = action.get("kind")
    if kind == "sql":
        return (
            f"Worked example of a correct query for this kind of question:\n"
            f"```sql\n{action.get('sql', '')}\n```\n"
            f"Reuse this query's shape for similar questions."
        )
    if kind == "clarify":
        return ("When a request is missing a required argument, ASK the user for the "
                "missing argument and make no tool call.")
    if kind == "call":
        args = action.get("args") or {}
        args_txt = ", ".join(f"{k}={v}" for k, v in args.items())
        return (f"When a request has all required arguments, CALL `{action.get('tool')}` "
                f"directly without asking the user. Arguments: {args_txt}.")
    if kind == "triage":
        return (f"Triage reference: acuity {action.get('acuity')}, "
                f"escalate={'yes' if action.get('escalate') else 'no'}. "
                f"Decision support only - never diagnose or prescribe.")
    if kind == "trade":
        return (f"Reference decision: direction {action.get('direction')}, "
                f"size {action.get('size')}. You propose only; the risk engine disposes.")
    if kind == "measure":
        return (f"Reference procedure: call `{action.get('tool')}` with "
                f"{action.get('args')} and report the measured value.")
    return task_desc or "Remember the successful approach."


def generalize(text: str, params: Dict[str, Any]) -> str:
    """
    Turn a concrete action back into a reusable template.

    A successful episode gives a *filled* answer. Storing the filled answer
    would make the lesson useless for the next instance of the same question
    type, so the concrete parameter values are folded back into placeholders.
    This is ordinary pattern extraction, not supervision: the agent only ever
    uses values it has already seen.
    """
    if not text or not params:
        return text
    out = text
    for key, val in sorted(params.items(), key=lambda kv: -len(str(kv[1]))):
        sval = str(val)
        if not sval or sval in ("None", "True", "False") or len(sval) < 2:
            continue
        out = out.replace(sval, "{" + str(key) + "}")
    return out


def generalize_action(action: Dict[str, Any], task) -> Dict[str, Any]:
    """
    Turn a concrete successful action back into a reusable lesson.

    A successful episode hands us *filled* values (this task's user id, this
    task's limit). Storing those would make the lesson replay stale constants
    for the next instance of the same question type - the agent would "learn"
    the last user id it saw. So instance values are folded back into
    placeholders, using the task's own parameter map. The lesson keeps the
    shape of what worked, never the instance it worked on.
    """
    params = (getattr(task, "meta", None) or {}).get("params") or {}
    arg_map = (getattr(task, "meta", None) or {}).get("args_params") or {}
    out = dict(action)
    kind = out.get("kind")
    if kind == "sql":
        # Prefer a template the action already carries (the builtin reference
        # answer is one). Re-deriving it from filled values is lossy for
        # single-character params, which `generalize` deliberately skips.
        tmpl = out.get("sql_template") or ""
        if "{" not in str(tmpl):
            tmpl = generalize(out.get("sql", ""), params)
        out["sql_template"] = tmpl
    elif kind == "call":
        args: Dict[str, Any] = {}
        for k, v in (out.get("args") or {}).items():
            if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                args[k] = v
            elif k in arg_map and arg_map[k] in params:
                args[k] = "{" + str(arg_map[k]) + "}"
            else:
                args[k] = generalize(str(v), params)
        out["args"] = args
    elif kind == "measure":
        out["args"] = {k: (v if isinstance(v, str) and v.startswith("{") and v.endswith("}")
                           else generalize(str(v), params))
                       for k, v in (out.get("args") or {}).items()}
    return out


# ------------------------------------------------------------------ evolvers

class Evolver:
    """Base class. Subclasses decide which surface of the harness they edit."""

    id = "seed"
    name = "Seed harness (no adaptation)"
    paradigm = "none"
    description = "The frozen baseline c_0. Nothing is learned; every checkpoint is the seed."
    surfaces: List[str] = []

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        seed = self.cfg.get("seed")
        # Seeded RNG for lesson ids. The policy salts its perturbation coins
        # with lesson ids, so unseeded ids would make even fully-seeded runs
        # differ from each other. None (default) keeps the old uuid behaviour.
        self.rng = random.Random(seed) if seed is not None else None

    # ------------------------------------------------------------- learning

    async def observe(self, harness: Harness, task, action: Dict[str, Any],
                      success: bool, trace: Dict[str, Any],
                      router: Router) -> Optional[Lesson]:
        """Called after every training episode. Returns the lesson learned, if any."""
        return None

    async def on_block_end(self, harness: Harness, block, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Called at every block boundary. Free to validate, prune or rewrite."""
        return {}

    # ------------------------------------------------------------ injection

    def lesson_list(self, harness: Harness) -> List[Lesson]:
        return [Lesson.from_dict(d) for d in harness.lesson_index]

    def inject(self, harness: Harness, task, router: Router) -> Injection:
        return Injection()

    # ---------------------------------------------------------- diagnostics

    def diagnostics(self, harness: Harness) -> Dict[str, Any]:
        return {}


# --------------------------------------------------------------------- seed

class SeedEvolver(Evolver):
    id = "seed"
    name = "Seed harness (no adaptation)"
    paradigm = "none"
    description = "Baseline. The harness never changes, so every family keeps its seed accuracy."

    def inject(self, harness: Harness, task, router: Router) -> Injection:
        return Injection(text="", lessons=[], source="none",
                         note="no adaptation: the agent runs the seed harness")


# ---------------------------------------------------------- reasoning bank

class ReasoningBankEvolver(Evolver):
    """
    Experience augmentation with similarity-based retrieval.

    Every successful episode is distilled into a memory item. At read time the
    most similar items (top-k) are retrieved and injected. There is no
    similarity threshold by default, and no scope check - exactly the
    configuration the paper reports as causing foreign-memory injection before
    a group is encountered, and retrieval collision when two workflows share a
    region of query space.

    `similarity_threshold` and `scope_mode` are exposed so the failure can be
    reproduced and then tested against a fix.
    """

    id = "reasoningbank"
    name = "ReasoningBank (similarity retrieval)"
    paradigm = "experience"
    surfaces = ["memory"]
    description = ("One memory item per episode; the top-1 item by query similarity is "
                   "injected into later episodes. No scope check by default.")

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__(cfg)
        self.top_k = int(self.cfg.get("top_k", 3))
        self.threshold = float(self.cfg.get("similarity_threshold", 0.0))
        self.scope_mode = self.cfg.get("scope_mode", "global")

    async def observe(self, harness: Harness, task, action: Dict[str, Any],
                      success: bool, trace: Dict[str, Any],
                      router: Router) -> Optional[Lesson]:
        if not success:
            return None
        learned = generalize_action(action, task)
        lesson = Lesson(
            id=new_id("L", self.rng),
            owner=router.assign(task)[0],
            scope="global",
            env=task.env,
            template_id=task.meta.get("tpl", ""),
            query=task.query,
            action=learned,
            text=action_lesson_text(learned),
            success=True,
            meta={"workflow": probe_workflow(task), "group": task.group,
                  "intent": task.meta.get("intent", "")},
        )
        harness.memory.append(MemoryItem(
            id=lesson.id,
            group=task.group,
            query=task.query,
            reflection=lesson.text,
            embedding=[],
            success=True,
            workflow={"fingerprint": lesson.meta["workflow"], "action": learned},
            meta={"template_id": lesson.template_id, "owner": lesson.owner,
                  "intent": lesson.meta["intent"]},
        ))
        harness.lesson_index.append(lesson.to_dict())
        harness.touch()
        return lesson

    def inject(self, harness: Harness, task, router: Router) -> Injection:
        items = harness.memory
        if not items:
            return Injection(source="memory", note="memory store is empty")
        scored = sorted(
            ((router.query_similarity(task.query or "", m.query), m) for m in items),
            key=lambda p: p[0], reverse=True,
        )
        if self.threshold > 0:
            scored = [p for p in scored if p[0] >= self.threshold]
            if not scored:
                return Injection(source="memory",
                                 note=f"no memory above similarity threshold {self.threshold}")
        picked = scored[: self.top_k]
        lessons = [
            Lesson(id=m.id, owner=m.meta.get("owner", "?"), scope="global", env=task.env,
                   template_id=m.meta.get("template_id", ""), query=m.query,
                   action=(m.workflow or {}).get("action", {}), text=m.reflection,
                   meta={"workflow": (m.workflow or {}).get("fingerprint", ""),
                         "intent": m.meta.get("intent", ""),
                         "group": m.group})
            for _, m in picked
        ]
        text = "\n\n".join(l.text for l in lessons)
        return Injection(
            text=text, lessons=lessons, source="memory",
            retrieved_from=lessons[0].id if lessons else "",
            similarity=round(picked[0][0], 4) if picked else 0.0,
            note=f"retrieved {len(lessons)} item(s) from {len(items)} memories",
        )

    def diagnostics(self, harness: Harness) -> Dict[str, Any]:
        return {"memory_count": len(harness.memory),
                "memories": [{"id": m.id, "group": m.group,
                              "template": m.meta.get("template_id", ""),
                              "intent": m.meta.get("intent", ""),
                              "workflow": (m.workflow or {}).get("fingerprint", ""),
                              "query": m.query[:120]} for m in harness.memory]}


# --------------------------------------------------------------------- gepa

class GEPAEvolver(Evolver):
    """
    Reflective prompt evolution over a SHARED strategy prompt.

    The agent reflects on recent rollouts and rewrites one strategy section
    that is injected into *every* subsequent rollout. Rules are compressed to
    stay within a character budget, which is what makes earlier rules lose
    their share of the prompt - the mechanism the paper measures in Figure 3.

    Because the surface is shared, a rule learned in one block is applied to
    every later block: pre-block interference before the block arrives, and
    overwriting once later blocks rewrite the same prompt.
    """

    id = "gepa"
    name = "GEPA (shared prompt evolution)"
    paradigm = "self-modification"
    surfaces = ["strategy_prompt"]
    description = ("Reflective rewrite of a single shared strategy prompt that is injected "
                   "into every rollout. Later blocks overwrite earlier rules.")

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__(cfg)
        self.max_chars = int(self.cfg.get("max_chars", 1400))
        self.update_every = int(self.cfg.get("update_every", 5))
        self.scope_mode = self.cfg.get("scope_mode", "global")
        self._pending: List[Dict[str, Any]] = []

    async def observe(self, harness: Harness, task, action: Dict[str, Any],
                      success: bool, trace: Dict[str, Any],
                      router: Router) -> Optional[Lesson]:
        self._pending.append({
            "query": task.query, "success": success, "env": task.env,
            "task_key": f"{task.group}#{task.id}", "cluster": router.assign(task)[0],
            "action": generalize_action(action, task),
            "intent": task.meta.get("intent", ""),
            "template_id": task.meta.get("tpl", ""),
        })
        if len(self._pending) < self.update_every:
            return None
        return self._flush(harness, router)

    def _flush(self, harness: Harness, router: Router) -> Optional[Lesson]:
        if not self._pending:
            return None
        successes = [p for p in self._pending if p["success"]]
        self._pending = []
        if not successes:
            return None
        # One lesson per distinct intent in the window, in first-seen order.
        # Keeping only the single most recent success would make the prompt's
        # coverage a lottery of window boundaries: on a narrow stream every
        # flush could distill the same intent and the other intents would
        # never be represented, even though the agent solved them.
        by_intent: Dict[str, Dict[str, Any]] = {}
        for p in successes:
            by_intent.setdefault(p.get("intent", "") or p.get("template_id", ""), p)
        label = router.labels().get(successes[-1]["cluster"],
                                    successes[-1]["cluster"])
        last_lesson: Optional[Lesson] = None
        for rep in by_intent.values():
            last_lesson = self._distill(harness, router, rep, len(successes), label)
        return last_lesson

    def _distill(self, harness: Harness, router: Router, rep: Dict[str, Any],
                 n_success: int, label: str) -> Lesson:
        owner = rep["cluster"]
        lesson = Lesson(
            id=new_id("L", self.rng),
            owner=owner,
            scope="global",
            env=rep["env"],
            template_id=rep.get("template_id", ""),
            query=rep["query"],
            action=rep.get("action") or {"kind": "strategy"},
            text=(f"Reflect on recent successful attempts: reuse the approach that worked "
                  f"for requests like: {rep['query'][:120]}"),
            meta={"n_success": n_success, "intent": rep.get("intent", ""),
                  "group": rep.get("task_key", "").split("#")[0]},
        )
        harness.lesson_index.append(lesson.to_dict())
        self._rewrite(harness, lesson, owner, label)
        harness.touch()
        return lesson

    def _rewrite(self, harness: Harness, lesson: Lesson, owner: str, label: str) -> None:
        """
        Add a rule to the SHARED prompt, then compress it to the budget.

        The rules list is the prompt. Compression drops the oldest rules, so a
        family's rule can be overwritten by a later block - the mechanism the
        paper measures as a fall in per-group prompt share (Figure 3).
        """
        rules = list(harness.strategy_rules or [])
        rules.append({
            "lesson_id": lesson.id,
            "owner": owner,
            "intent": lesson.meta.get("intent", ""),
            "text": (f"When the request looks like the ones you just solved successfully "
                     f"({label}), reuse this approach:\n{lesson.text}"),
            "action": lesson.action,
        })
        # Compress: drop the oldest rules until the rendered prompt fits.
        while rules and len(self._render(rules)) > self.max_chars:
            rules.pop(0)
        harness.strategy_rules = rules
        harness.strategy_prompt = self._render(rules)

    @staticmethod
    def _render(rules: List[Dict[str, Any]]) -> str:
        return "\n".join(f"[{r.get('owner', '?')}] {r.get('text', '')}" for r in rules)

    def inject(self, harness: Harness, task, router: Router) -> Injection:
        by_id = {l["id"]: Lesson.from_dict(l) for l in harness.lesson_index}
        lessons: List[Lesson] = []
        for rule in (harness.strategy_rules or []):
            l = by_id.get(rule.get("lesson_id", ""))
            if l is not None:
                lessons.append(l)
        text = strip_provenance(harness.strategy_prompt or "")
        return Injection(text=text, lessons=lessons, source="strategy",
                         retrieved_from=lessons[0].id if lessons else "",
                         similarity=0.0,
                         note=(f"shared strategy prompt, {len(harness.strategy_prompt or '')} chars, "
                               f"{len(lessons)} rule(s) live"))

    def diagnostics(self, harness: Harness) -> Dict[str, Any]:
        return {
            "strategy_chars": len(harness.strategy_prompt or ""),
            "strategy_attribution": harness.strategy_attribution(),
            "rules": [ln for ln in (harness.strategy_prompt or "").splitlines() if ln.strip()],
            "live_rules": [
                {"lesson_id": r.get("lesson_id"), "owner": r.get("owner"),
                 "intent": r.get("intent")} for r in (harness.strategy_rules or [])
            ],
            "scope_mode": self.scope_mode,
        }


# ------------------------------------------------------------- meta-harness

class MetaHarnessEvolver(Evolver):
    """
    Workspace-wide self-modification with a validation gate.

    The proposer may edit the system prompt, the memory file, the skill
    library and the controller. An edit is accepted only when it improves
    validation accuracy on the active sub-block. The gate is what makes
    Meta-Harness the most stable of the three in the paper - and also the
    least plastic, because a rejected edit freezes the workspace.
    """

    id = "metaharness"
    name = "Meta-Harness (gated workspace edits)"
    paradigm = "self-modification"
    surfaces = ["system_prompt", "strategy_prompt", "memory", "skills", "controller"]
    description = ("Proposes edits to prompt, memory, skills and controller; accepts an edit "
                   "only when it improves validation accuracy on the active block.")

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__(cfg)
        self.validation_size = int(self.cfg.get("validation_size", 6))
        self.scope_mode = self.cfg.get("scope_mode", "global")
        self.min_gain = float(self.cfg.get("min_gain", 0.0))
        self.last_proposal: Dict[str, Any] = {}

    async def observe(self, harness: Harness, task, action: Dict[str, Any],
                      success: bool, trace: Dict[str, Any],
                      router: Router) -> Optional[Lesson]:
        if not success:
            return None
        learned = generalize_action(action, task)
        lesson = Lesson(
            id=new_id("L", self.rng),
            owner=router.assign(task)[0],
            scope="global",
            env=task.env,
            template_id=task.meta.get("tpl", ""),
            query=task.query,
            action=learned,
            text=action_lesson_text(learned),
            meta={"workflow": probe_workflow(task), "group": task.group,
                  "intent": task.meta.get("intent", "")},
        )
        harness.lesson_index.append(lesson.to_dict())
        harness.touch()
        return lesson

    async def on_block_end(self, harness: Harness, block, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose a workspace edit, then accept it only if validation improves.

        `ctx["validate"]` is an async callable (candidate harness) -> accuracy in
        [0,1] over the active block's validation split.
        """
        validate = ctx.get("validate")
        if validate is None:
            return {"accepted": False, "reason": "no validation hook"}
        proposal = self._propose(harness, block)
        if not proposal:
            return {"accepted": False, "reason": "nothing to propose"}
        candidate = harness.clone()
        self._apply(candidate, proposal)
        baseline = await validate(harness)
        proposed = await validate(candidate)
        gain = proposed - baseline
        accepted = gain > self.min_gain
        self.last_proposal = {"proposal": proposal, "baseline": baseline,
                              "proposed": proposed, "gain": gain, "accepted": accepted}
        if accepted:
            harness.strategy_prompt = candidate.strategy_prompt
            harness.skills = candidate.skills
            harness.controller = candidate.controller
            harness.system_prompt = candidate.system_prompt
            harness.lesson_index = candidate.lesson_index
            harness.touch()
        return {"accepted": accepted, "gain": round(gain, 4),
                "baseline": round(baseline, 4), "proposed": round(proposed, 4),
                "proposal": proposal["summary"]}

    def _propose(self, harness: Harness, block) -> Optional[Dict[str, Any]]:
        lessons = self.lesson_list(harness)
        recent = [l for l in lessons if l.meta.get("group") == getattr(block, "group", "")]
        if not recent:
            return None
        best = recent[-1]
        return {
            "summary": f"promote the approach learned on {best.meta.get('group')}",
            "strategy_line": f"[{best.owner}] {best.text.splitlines()[0]}",
            "skill": {"name": f"skill:{best.template_id or best.owner}",
                      "description": best.text.splitlines()[0],
                      "source": best.id,
                      "intent": best.meta.get("intent", ""),
                      "action": best.action},
            "controller_line": f"# learned on block {getattr(block, 'group', '')}",
        }

    def _apply(self, harness: Harness, proposal: Dict[str, Any]) -> None:
        line = proposal["strategy_line"]
        lines = [ln for ln in (harness.strategy_prompt or "").splitlines() if ln.strip()]
        if line not in lines:
            lines.append(line)
        harness.strategy_prompt = "\n".join(lines)
        skill = proposal["skill"]
        if not any(s.get("name") == skill["name"] for s in harness.skills):
            harness.skills.append(skill)
        controller = proposal["controller_line"]
        if controller not in (harness.controller or ""):
            harness.controller = ((harness.controller or "") + "\n" + controller).strip()

    def inject(self, harness: Harness, task, router: Router) -> Injection:
        by_id = {l["id"]: Lesson.from_dict(l) for l in harness.lesson_index}
        lessons: List[Lesson] = []
        for skill in harness.skills:
            l = by_id.get(skill.get("source", ""))
            if l is not None:
                lessons.append(l)
        parts = [strip_provenance(harness.strategy_prompt or "")]
        if harness.skills:
            parts.append("\n".join(f"- {s.get('name')}: {s.get('description')}"
                                   for s in harness.skills))
        if lessons:
            parts.append("\n".join(l.text for l in lessons))
        return Injection(text="\n\n".join(p for p in parts if p.strip()),
                         lessons=lessons, source="workspace",
                         retrieved_from=lessons[0].id if lessons else "",
                         note=f"workspace: {len(harness.skills)} skill(s), "
                              f"{len(harness.strategy_prompt or '')} strategy chars")

    def diagnostics(self, harness: Harness) -> Dict[str, Any]:
        return {
            "strategy_chars": len(harness.strategy_prompt or ""),
            "strategy_attribution": harness.strategy_attribution(),
            "skills": harness.skills,
            "controller_lines": len((harness.controller or "").splitlines()),
            "last_proposal": self.last_proposal,
            "scope_mode": self.scope_mode,
        }


# ------------------------------------------------------------------- scoped

class ScopedEvolver(Evolver):
    """
    Task-scoped, workflow-conditioned memory.

    This is the lab's proposal for the gap the paper identifies. Three changes
    to the experience-augmentation paradigm:

    1. SCOPE. Lessons are filed under a *cluster* the agent discovered itself
       (never the ground-truth group label) and are only retrieved for tasks
       that route to the same cluster. A task family the agent has never seen
       therefore inherits nothing.

    2. WORKFLOW CONDITIONING. Lessons are additionally filed under the
       workflow fingerprint of the request (missing argument vs complete
       argument). Two requests can be nearly identical and demand opposite
       behaviour; conditioning on the fingerprint separates them even though
       no similarity threshold can.

    3. APPEND-ONLY SECTIONS. Each scope owns its own prompt section. A later
       block adds to its own section and never compresses or rewrites another
       one's, so nothing is overwritten.

    When a cluster is found to carry two incompatible fingerprints, the
    conflict is recorded in the harness's conflict ledger rather than being
    silently resolved.
    """

    id = "scoped"
    name = "Scoped memory (cluster + workflow conditioned)"
    paradigm = "experience"
    surfaces = ["memory", "scoped_sections", "conflict_ledger"]
    description = ("Lessons are filed per discovered cluster AND per workflow fingerprint, "
                   "in append-only prompt sections. Nothing is overwritten and nothing "
                   "foreign is injected.")

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__(cfg)
        self.threshold = float(self.cfg.get("similarity_threshold", 0.0))
        self.scope_mode = self.cfg.get("scope_mode", "scoped")
        self.max_per_scope = int(self.cfg.get("max_per_scope", 24))
        self.top_k = int(self.cfg.get("top_k", 4))

    def _scope_key(self, task, router: Router) -> str:
        """
        The scope a task belongs to: the cluster the router discovered, plus
        the workflow fingerprint.

        This is the one place the scope is decided, so a subclass can
        replace the information used to derive it without changing anything
        else about retrieval.
        """
        cluster_id, _sim, _ = router.assign(task)
        return f"{cluster_id}|{probe_workflow(task)}"

    async def observe(self, harness: Harness, task, action: Dict[str, Any],
                      success: bool, trace: Dict[str, Any],
                      router: Router) -> Optional[Lesson]:
        if not success:
            return None
        cluster_id, sim, _ = router.assign(task)
        fingerprint = probe_workflow(task)
        scope = self._scope_key(task, router)
        learned = generalize_action(action, task)
        lesson = Lesson(
            id=new_id("L", self.rng),
            owner=cluster_id,
            scope=scope,
            env=task.env,
            template_id=task.meta.get("tpl", ""),
            query=task.query,
            action=learned,
            text=action_lesson_text(learned),
            meta={"workflow": fingerprint, "group": task.group, "cluster": cluster_id,
                  "intent": task.meta.get("intent", "")},
        )
        harness.lesson_index.append(lesson.to_dict())
        harness.memory.append(MemoryItem(
            id=lesson.id, group=task.group, query=task.query, reflection=lesson.text,
            success=True, workflow={"fingerprint": fingerprint, "action": learned},
            meta={"template_id": lesson.template_id, "owner": cluster_id, "scope": scope,
                  "intent": lesson.meta.get("intent", "")},
        ))
        self._render_scope(harness, scope, router)
        self._record_conflicts(harness, router)
        harness.touch()
        return lesson

    def _render_scope(self, harness: Harness, scope: str, router: Router) -> None:
        """Rewrite ONLY this scope's section. Other sections are untouched."""
        lessons = [l for l in self.lesson_list(harness) if l.scope == scope]
        lessons = lessons[-self.max_per_scope:]
        cluster_id, _, fingerprint = scope.partition("|")
        label = router.labels().get(cluster_id, cluster_id)
        heading = (f"Task family: {label} | workflow: {workflow_descriptor(fingerprint)}")
        body = "\n".join(f"- {l.text.splitlines()[0]}" for l in lessons)
        harness.scoped_sections[scope] = f"{heading}\n{body}"

    def _record_conflicts(self, harness: Harness, router: Router) -> None:
        """A cluster holding two opposite workflows is a retrieval-collision risk."""
        for cid, cluster in router.clusters.items():
            if len(cluster.workflows) > 1:
                entry = {
                    "cluster": cid,
                    "workflows": dict(cluster.workflows),
                    "keywords": router.labels().get(cid, cid),
                    "resolution": "isolated into per-workflow sections",
                }
                if not any(e.get("cluster") == cid for e in harness.conflict_ledger):
                    harness.conflict_ledger.append(entry)

    def inject(self, harness: Harness, task, router: Router) -> Injection:
        cluster_id, sim, _ = router.assign(task)
        scope = self._scope_key(task, router)
        candidates = [l for l in self.lesson_list(harness)
                      if l.env == task.env and l.scope == scope]
        if not candidates:
            return Injection(
                source="scoped",
                note=(f"scope '{scope}' has no lessons yet - nothing injected "
                      f"(no foreign memory)"),
                similarity=0.0,
            )
        scored = sorted(((router.query_similarity(task.query or "", l.query), l)
                         for l in candidates), key=lambda p: p[0], reverse=True)
        if self.threshold > 0:
            scored = [p for p in scored if p[0] >= self.threshold]
            if not scored:
                return Injection(source="scoped",
                                 note=f"no lesson in scope '{scope}' above threshold")
        picked = scored[: self.top_k]
        lessons = [l for _, l in picked]
        best_sim = picked[0][0]
        return Injection(
            text="\n\n".join(l.text for l in lessons), lessons=lessons, source="scoped",
            retrieved_from=lessons[0].id if lessons else "",
            similarity=round(best_sim, 4),
            note=(f"scope '{scope}' matched, {len(candidates)} candidate(s); "
                  f"other scopes excluded"),
        )

    def diagnostics(self, harness: Harness) -> Dict[str, Any]:
        return {
            "scoped_sections": {k: len(v) for k, v in harness.scoped_sections.items()},
            "scoped_attribution": harness.scoped_attribution(),
            "conflict_ledger": harness.conflict_ledger,
            "memory_count": len(harness.memory),
            "scope_mode": self.scope_mode,
        }


# ------------------------------------------------------------------ helpers

class OracleEvolver(ScopedEvolver):
    """
    The ceiling: scope isolation done perfectly.

    Not a competitor - a reference. It files and retrieves lessons by the
    task's GROUND-TRUTH family, not by the cluster the router discovered,
    and it matches the intent exactly. So it always knows which family a
    request belongs to and never picks a rule from the wrong one.

    It gives the suite a scale. If scoped scores +16 and the oracle
    scores +19, then scoped captures 84% of what perfect scope isolation
    could deliver, and the remaining 16% is the price of not knowing the
    answer key. Without a ceiling, "+16" has no scale.

    Note that keying on the intent ALONE is not a ceiling: several
    families share an intent (the "flat" trading signal appears in the
    momentum and mean-reversion families), so intent-only retrieval can
    pick a rule from the wrong family and lose to a properly scoped
    method. The family has to be known too.
    """

    id = "oracle"
    name = "Oracle retrieval (reference ceiling)"
    paradigm = "experience"
    surfaces = ["memory", "scoped_sections", "conflict_ledger"]
    description = ("A reference, not a competitor: files and retrieves by the "
                   "ground-truth family, so it always knows which family a "
                   "request belongs to. It bounds what perfect scope isolation "
                   "could achieve.")

    def _scope_key(self, task, router: Router) -> str:
        """The scope by the answer key, not by the discovered cluster."""
        return f"oracle:{task.group}|{probe_workflow(task)}"

    def _oracle_scope(self, task) -> str:
        return self._scope_key(task, None)

    async def observe(self, harness: Harness, task, action: Dict[str, Any],
                      success: bool, trace: Dict[str, Any],
                      router: Router) -> Optional[Lesson]:
        # The oracle inherits scoped's learning and retrieval exactly; only
        # the scope KEY differs, so it is a strict superset of the scoped
        # evolver's information and therefore a true ceiling.
        return await super().observe(harness, task, action, success, trace, router)


EVOLVERS: Dict[str, Any] = {
    "seed": SeedEvolver,
    "reasoningbank": ReasoningBankEvolver,
    "gepa": GEPAEvolver,
    "metaharness": MetaHarnessEvolver,
    "scoped": ScopedEvolver,
    "oracle": OracleEvolver,
}


def get_evolver(evolver_id: str, cfg: Optional[Dict[str, Any]] = None) -> Evolver:
    cls = EVOLVERS.get(evolver_id)
    if cls is None:
        raise KeyError(f"unknown evolver '{evolver_id}'. known: {sorted(EVOLVERS)}")
    return cls(cfg)


def evolver_catalog() -> List[Dict[str, Any]]:
    out = []
    for eid, cls in EVOLVERS.items():
        out.append({
            "id": eid, "name": cls.name, "paradigm": cls.paradigm,
            "description": cls.description, "surfaces": cls.surfaces,
        })
    return out
