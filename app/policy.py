"""
Policies: what actually answers a task.

Two implementations behind one interface.

  SimulatedPolicy  A deterministic policy with a noisy prior. It has a weak
                   default competence and a strong competence when it is
                   following a lesson it learned. Because it consumes the same
                   `Injection` a live agent would, harness mechanics can be
                   designed and compared in seconds, with no GPU and no
                   flakiness.

  LivePolicy       A ReAct-style agent that calls the local LLM through
                   fox-services. It can execute tool calls and SQL, observe the
                   result, and retry.

Both policies return a structured `action` alongside the text. The evolvers
learn from the structured action (that is what actually worked) while the
graders score the text. Both are needed: the grader must see what a real
system would emit.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import envs
from .evolvers import Injection, Lesson, generalize
from .harness import Harness
from .retrieval import Router, probe_workflow


@dataclass
class PolicyResult:
    output: str
    action: Dict[str, Any]
    trace: Dict[str, Any]


# ---------------------------------------------------------------- helpers

def _tool_guess(query: str) -> Tuple[str, Dict[str, Any]]:
    """Heuristic API selection from the request text (the agent's prior)."""
    q = (query or "").lower()
    if "order history" in q or "what i" in q and "bought" in q:
        return "get_order_history", {}
    if "profile" in q:
        return "get_user_profile", {}
    if "stock" in q or "in stock" in q:
        return "check_inventory", {}
    if "ticket" in q or "support" in q:
        return "create_ticket", {}
    if "discount" in q or "promo" in q or "code" in q:
        return "apply_discount", {}
    if "search" in q or "catalogue" in q or "catalog" in q:
        return "search_items", {}
    return "get_order_history", {}


def _red_flag(text: str) -> bool:
    low = (text or "").lower()
    signals = ["chest pain", "crushing", "diaphoresis", "bp 88", "weakness",
               "slurred", "seizure", "fever 39", "confusion", "lactate",
               "hr 118", "bp 96/60"]
    return any(s in low for s in signals)


def _trade_prior(query: str) -> Dict[str, Any]:
    low = (query or "").lower()
    if "stale" in low or "heartbeat" in low:
        return {"direction": "flat", "size": 0, "reason": "signal is stale; default flat"}
    if "gap" in low and ("8%" in low or "6.2 atr" in low):
        return {"direction": "flat", "size": 0, "reason": "dislocated move; default flat"}
    if "flat" in low and "no volume" in low:
        return {"direction": "flat", "size": 0, "reason": "no edge; stand aside"}
    if "above" in low or "breakout" in low:
        return {"direction": "long", "size": 1500, "reason": "breakout with volume"}
    if "below" in low or "breakdown" in low:
        return {"direction": "short", "size": 1500, "reason": "breakdown with volume"}
    return {"direction": "flat", "size": 0, "reason": "no clear signal"}


def _triage_prior(query: str) -> Dict[str, Any]:
    if _red_flag(query):
        return {"acuity": 1, "escalate": True,
                "note": "red flags present - escalate to a licensed clinician immediately"}
    low = (query or "").lower()
    if "mild" in low or "localised" in low or "itchy" in low:
        return {"acuity": 4, "escalate": False, "note": "low acuity - routine pathway"}
    return {"acuity": 3, "escalate": False, "note": "standard acuity - routine pathway"}


# -------------------------------------------------------------- simulated

from . import constants

class SimulatedPolicy:
    """
    Deterministic policy used to exercise harness mechanics without a model.

    Behaviour model
    ---------------
    1. CLASSIFY. The agent reads the request and works out what KIND of
       request it is - its *intent*. In simulated mode the intent comes from
       the environment's keyword lexicon, which is a stand-in for the semantic
       understanding a real LLM brings. The lexicon names the kind of request;
       it never contains the answer.

    2. APPLY. The agent looks at the lessons the harness handed it and picks
       the one for that intent. If the harness holds no lesson for the intent,
       the agent falls back to the most similar lesson it was given. That
       fallback is the interference mechanism: knowledge learned on one family
       gets applied to a family that needs something else.

    3. FALL BACK. With no lesson at all the agent answers from its own prior
       competence, which is imperfect. It mis-reads the workflow some of the
       time, and it only knows some question types.

    4. NOISE. With probability (1 - skill) the action is perturbed, so
       competence grows gradually instead of switching on at the first success.

    The base model's competence is modelled explicitly: for every intent the
    environment declares a probability that a frozen model solves it. The
    simulator realises that probability by drawing the environment's reference
    answer with that probability and a generic wrong answer otherwise. The
    simulator therefore has no privileged access to the *harness* - everything
    the harness can do, it does through the lessons it is given.
    """

    name = "simulated"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.skill = float(cfg.get("skill", constants.DEFAULT_SKILL))
        self.seed = int(cfg.get("seed", 7))
        self.probe_accuracy = dict(
            cfg.get("probe_accuracy", {"missing-arg": constants.PROBE_ACCURACY_MISSING,
                                       "complete-arg": constants.PROBE_ACCURACY_COMPLETE})
        )
        # How often the agent follows a rule that does not fit the request in
        # front of it. The paper's BFCL trace (clarification rate 3.8% -> 29.8%
        # -> 11.2%) says: often enough to hurt, not so often that the agent
        # can never learn. A rule carried over from another ENVIRONMENT is
        # followed less often still - the agent can tell the environments
        # apart from the preamble.
        self.foreign_follow = float(cfg.get("foreign_follow", constants.DEFAULT_FOREIGN_FOLLOW_SAME_ENV))
        self.foreign_follow_cross_env = float(cfg.get("foreign_follow_cross_env", constants.DEFAULT_FOREIGN_FOLLOW_CROSS_ENV))

    def _hit(self, task, salt: str, prob: float) -> bool:
        """
        A deterministic coin for one task.

        The policy is a pure function of (task, harness state). That matters:
        the seed checkpoint and a later checkpoint evaluate the SAME hold-out
        tasks, so any randomness would show up as metric noise and be
        mistaken for learning. The hash keeps the statistical behaviour of a
        noisy model while making every run exactly reproducible.
        """
        if prob <= 0.0:
            return False
        if prob >= 1.0:
            return True
        raw = f"{self.seed}|{salt}|{getattr(task, 'id', '')}|{task.query or ''}"
        digest = hashlib.sha1(raw.encode("utf-8")).digest()
        u = int.from_bytes(digest[:8], "big") / float(1 << 64)
        return u < prob

    async def act(self, task, injection: Injection, harness: Harness,
                  router: Router) -> PolicyResult:
        view = envs.view_for_task(task)
        intent = self._intent_of(task, view)
        lesson = self._governing(task, injection, router, intent)
        foreign = False
        if lesson is not None:
            lesson_intent = lesson.meta.get("intent") or ""
            intent_match = (not intent) or (not lesson_intent) or lesson_intent == intent
            if not intent_match:
                same_env = lesson.env == task.env
                rate = self.foreign_follow if same_env else self.foreign_follow_cross_env
                if not self._hit(task, f"foreign:{lesson.id}", rate):
                    # The agent does not follow a rule that plainly does not
                    # fit, and answers from its own prior instead.
                    lesson = None
                else:
                    foreign = True
        if lesson is not None:
            action = self._action_from_lesson(task, lesson)
        else:
            action = self._prior(task, view, intent)
        action = self._maybe_perturb(task, action, lesson, view)
        output = self._render(task, action)
        return PolicyResult(
            output=output, action=action,
            trace={
                "policy": "simulated",
                "governed_by": lesson.id if lesson else "prior",
                "lesson_scope": lesson.scope if lesson else "",
                "lesson_owner": lesson.owner if lesson else "",
                "intent": intent or "",
                "foreign_knowledge": foreign,
                "injection_source": injection.source,
                "injection_note": injection.note,
                "similarity": injection.similarity,
            },
        )

    # ------------------------------------------------------------ selection

    def _intent_of(self, task, view) -> Optional[str]:
        """Classify the request. Uses only the request text and the preamble."""
        if view is None:
            return None
        return envs.classify_intent(task.query, view.intent_lexicon())

    def _governing(self, task, injection: Injection, router: Router,
                   intent: Optional[str]) -> Optional[Lesson]:
        """
        Pick the lesson that governs this task.

        A lesson for the current intent wins. If none of the injected lessons
        is for this intent, the most similar one is applied anyway - the agent
        has no way of knowing that the knowledge it was handed was learned
        somewhere else. That is exactly the failure the paper reports.
        """
        candidates = list(injection.lessons)
        if not candidates:
            return None
        matching = [l for l in candidates if intent and l.meta.get("intent") == intent]
        pool = matching if matching else candidates
        best, best_score = None, -1.0
        for l in pool:
            score = router.query_similarity(task.query or "", l.query)
            if (score > best_score
                    or (score == best_score and best is not None
                        and l.created_at > best.created_at)):
                best, best_score = l, score
        return best

    def _action_from_lesson(self, task, lesson: Lesson) -> Dict[str, Any]:
        action = dict(lesson.action)
        params = task.meta.get("params") or {}
        kind = action.get("kind")
        if kind == "sql":
            tmpl = action.get("sql_template") or action.get("sql") or ""
            action["sql_template"] = tmpl
            action["sql"] = generalize_inverse(tmpl, params)
        elif kind == "call":
            action["args"] = {k: generalize_inverse(str(v), params)
                              for k, v in (action.get("args") or {}).items()}
        elif kind == "measure":
            action["args"] = {k: generalize_inverse(str(v), params)
                              for k, v in (action.get("args") or {}).items()}
        return action

    # --------------------------------------------------------------- prior

    def _probe(self, task) -> str:
        """
        The agent's own reading of the request's workflow - and it is wrong
        sometimes. `probe_accuracy` says how often the reading is right, which
        is what sets the seed baseline for the two BFCL families (the paper
        measures 53.8% on BFCL-Miss and 77.3% on BFCL-Base).
        """
        truth = probe_workflow(task)
        if truth == "default":
            return "default"
        if self._hit(task, f"probe:{truth}", float(self.probe_accuracy.get(truth, 0.9))):
            return truth
        return "complete-arg" if truth == "missing-arg" else "missing-arg"

    def _prior(self, task, view, intent: Optional[str]) -> Dict[str, Any]:
        """
        Answer without a lesson, using only the agent's own competence.

        For tool use the deciding factor is the workflow reading, which is
        noisy. For every other environment the base model simply knows some
        question types and not others.
        """
        env = task.env
        builtins = view.builtin_actions() if view is not None else {}
        ref = builtins.get(intent) if intent else None
        prob = float(ref["prob"]) if ref else 0.30

        if env == "toolcall":
            reading = self._probe(task)
            if reading == "missing-arg":
                action = dict((ref or {}).get("action") or {})
                if action.get("kind") != "clarify":
                    action = {"kind": "clarify", "missing": "the required argument"}
                return action
            action = dict((ref or {}).get("action") or {})
            if action.get("kind") != "call":
                tool, args = _tool_guess(task.query)
                action = {"kind": "call", "tool": tool, "args": args}
            return self._fill_args(task, action)

        if ref is not None and self._hit(task, f"base:{intent}", prob):
            return self._fill_params(task, dict(ref["action"]))
        return self._generic(task, view)

    def _fill_params(self, task, action: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute the task's parameter values into an action template."""
        params = task.meta.get("params") or {}
        out = dict(action)
        if out.get("kind") == "sql":
            tmpl = out.get("sql_template") or out.get("sql") or ""
            out["sql_template"] = tmpl
            out["sql"] = generalize_inverse(tmpl, params)
        elif out.get("kind") in ("call", "measure"):
            out["args"] = {k: generalize_inverse(str(v), params)
                           for k, v in (out.get("args") or {}).items()}
        return out

    def _fill_args(self, task, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a tool call's arguments from the request.

        The agent can read the argument values out of the request itself; the
        API catalogue tells it which arguments the API takes.
        """
        params = task.meta.get("params") or {}
        args = dict(action.get("args") or {})
        arg_map = task.meta.get("args_params") or {}
        for arg_name, param_name in arg_map.items():
            if param_name in params:
                args[arg_name] = params[param_name]
        action = dict(action)
        action["args"] = args
        return action

    def _generic(self, task, view) -> Dict[str, Any]:
        """
        The agent's answer when its prior competence does not cover the task.

        The wrong answer must be WRONG. A generic fallback that happens to
        match a gold - "flat" for a flat market, "acuity 3" for a standard
        case - silently inflates the base rate and destroys the headroom
        the evaluation needs. So the fallback is chosen to be wrong for
        THIS task: it avoids the gold's direction and its gold's acuity.
        """
        env = task.env
        if env == "sql":
            schema = view.schema_sql() if view is not None else None
            tables = re.findall(r"CREATE TABLE (\w+)", schema or "")
            first = tables[0] if tables else "t"
            sql = f"SELECT * FROM {first}"
            return {"kind": "sql", "sql": sql, "sql_template": sql}
        if env == "toolcall":
            tool, args = _tool_guess(task.query)
            return {"kind": "call", "tool": tool, "args": args}
        if env == "science":
            params = task.meta.get("params") or {}
            return {"kind": "measure", "tool": "measure_property",
                    "args": {"property": "unknown", "sample": params.get("sample", "A"),
                             "temperature_c": params.get("temp", 20)},
                    "value": 0.0}
        if env == "healthcare":
            # Avoid the gold's acuity so the fallback is actually wrong.
            gold = task.gold if isinstance(task.gold, dict) else {}
            try:
                gold_acuity = int(gold.get("acuity", 3))
            except Exception:
                gold_acuity = 3
            wrong = gold_acuity + 1 if gold_acuity < 5 else gold_acuity - 1
            return {"kind": "triage", "acuity": wrong, "escalate": False,
                    "note": "routine pathway - no red flags identified"}
        if env == "trading":
            # Avoid the gold's direction, and never take a position when the
            # reference answer is to stand aside.
            gold = task.gold if isinstance(task.gold, dict) else {}
            gold_dir = str(gold.get("direction", "flat")).lower()
            wrong = {"long": "short", "short": "long", "flat": "long"}.get(gold_dir, "long")
            return {"kind": "trade", "direction": wrong, "size": 0,
                    "reason": "no clear signal"}
        return {"kind": "text", "text": "Unable to answer."}

    def _maybe_perturb(self, task, action: Dict[str, Any], lesson: Optional[Lesson],
                       view=None) -> Dict[str, Any]:
        """Inject noise so competence grows gradually rather than instantly."""
        if lesson is None:
            return action
        salt = f"perturb:{lesson.id}"
        if not self._hit(task, salt, 1.0 - self.skill):
            return action
        kind = action.get("kind")
        out = dict(action)
        if kind == "sql":
            schema = view.schema_sql() if view is not None else None
            tables = re.findall(r"CREATE TABLE (\w+)", schema or "")
            others = [t for t in tables if t != _first_table(out.get("sql", ""))]
            if others:
                out["sql"] = re.sub(r"\b" + re.escape(_first_table(out.get("sql", ""))) + r"\b",
                                    others[0], out.get("sql", ""), count=1)
        elif kind == "call":
            args = dict(out.get("args") or {})
            if args:
                k = sorted(args)[0]
                args[k] = "wrong_value"
                out["args"] = args
        elif kind == "clarify":
            out = {"kind": "call", "tool": _tool_guess(task.query)[0], "args": {}}
        elif kind == "triage":
            out["acuity"] = min(5, int(out.get("acuity", 3)) + 1)
        elif kind == "trade":
            out["direction"] = "short" if out.get("direction") == "long" else "long"
            out["size"] = 0
        elif kind == "measure":
            out["value"] = 0.0
        return out

    def _render(self, task, action: Dict[str, Any]) -> str:
        kind = action.get("kind")
        if kind == "sql":
            return f"```sql\n{action.get('sql', '')}\n```"
        if kind == "clarify":
            return ("Could you confirm the " + str(action.get("missing", "required argument"))
                    + "? I need it before I can run this.")
        if kind == "call":
            return json.dumps({"tool": action.get("tool"), "args": action.get("args") or {}})
        if kind == "measure":
            return json.dumps({"tool": action.get("tool"), "args": action.get("args") or {},
                               "value": action.get("value")})
        if kind == "triage":
            return json.dumps({"acuity": action.get("acuity"),
                               "escalate": bool(action.get("escalate")),
                               "note": action.get("note", ""),
                               "recommendation": "Decision support only; licensed clinician sign-off required."})
        if kind == "trade":
            return json.dumps({"direction": action.get("direction"),
                               "size": action.get("size"),
                               "reason": action.get("reason", "")})
        return str(action.get("text", ""))


def _first_table(sql: str) -> str:
    m = re.search(r"FROM\s+(\w+)", sql or "", re.IGNORECASE)
    return m.group(1) if m else ""


def generalize_inverse(template: str, params: Dict[str, Any]) -> str:
    """Fill `{param}` placeholders in a template with the current task's values."""
    out = template or ""
    for k, v in (params or {}).items():
        out = out.replace("{" + str(k) + "}", str(v))
    return out


# ------------------------------------------------------------------- live

class LivePolicy:
    """
    ReAct-style agent over the local LLM (fox-services -> Ollama).

    The loop: propose an action, execute it if the environment supports it,
    feed the observation back, and repeat until the agent answers or the step
    budget runs out. SQL is executed against the group's schema so the model
    can see its own error and correct it.
    """

    name = "live"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, llm=None):
        cfg = cfg or {}
        self.cfg = cfg
        self.model = cfg.get("model", "fox-managed")
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_steps = int(cfg.get("max_steps", 3))
        self.llm = llm

    # ------------------------------------------------------------ prompting

    def _contract(self, task) -> str:
        env = task.env
        if env == "sql":
            return ("Write one SQL query that answers the question. Reply with only a "
                    "```sql fenced block. The query will be executed against the schema; "
                    "you may be shown the error and asked to fix it.")
        if env == "toolcall":
            return ("Decide what to do. If a required argument is missing from the request, "
                    "ASK the user for it and make NO tool call. If every required argument "
                    "is present, CALL the API immediately without asking. "
                    "Reply with JSON: {\"clarify\": true, \"missing\": \"<arg>\"} or "
                    "{\"tool\": \"<api>\", \"args\": {...}}")
        if env == "science":
            return ("Run the assay. Reply with JSON: {\"tool\": \"<instrument>\", "
                    "\"args\": {...}, \"value\": <measured number>}")
        if env == "healthcare":
            return ("Reply with JSON: {\"acuity\": <1-5>, \"escalate\": <true|false>, "
                    "\"note\": \"<one line>\"}. Decision support only: never diagnose, "
                    "never prescribe.")
        if env == "trading":
            return ("Reply with JSON: {\"direction\": \"long|short|flat\", \"size\": <number>, "
                    "\"reason\": \"<one line>\"}. You propose; the risk engine disposes. "
                    "Never exceed the position cap, and default to flat when the signal "
                    "is stale or dislocated.")
        return "Answer concisely."

    def _task_context(self, task) -> str:
        group = envs.get_group(task.group)
        bits = []
        if group:
            if group.context_md:
                bits.append(group.context_md)
            if group.api_catalog:
                bits.append("APIs: " + "; ".join(
                    f"{a['name']}({', '.join(f'{k}: {v}' for k, v in a['args'].items())})"
                    for a in group.api_catalog))
        bits.append(f"REQUEST: {task.query}")
        return "\n\n".join(bits)

    # --------------------------------------------------------------- acting

    async def act(self, task, injection: Injection, harness: Harness,
                  router: Router) -> PolicyResult:
        assert self.llm is not None, "LivePolicy needs an LLMClient"
        system = harness.build_system_prompt()
        if injection.text.strip():
            system = f"{system}\n\n## What you have learned\n{injection.text.strip()}"
        user = f"{self._task_context(task)}\n\n{self._contract(task)}"

        scratch = ""
        last_action: Dict[str, Any] = {"kind": "text", "text": ""}
        for step in range(self.max_steps):
            messages = [{"role": "system", "content": system}]
            if scratch:
                messages.append({"role": "assistant", "content": scratch.split("REQUEST:")[-1][:2000]})
                messages.append({"role": "user", "content": scratch})
            else:
                messages.append({"role": "user", "content": user})
            try:
                res = await self.llm.chat(messages, model=self.model,
                                          temperature=self.temperature, timeout=self.cfg.get("timeout"))
            except Exception as exc:
                return PolicyResult(output="", action={"kind": "error", "error": str(exc)},
                                    trace={"policy": "live", "error": str(exc)})
            raw = res["content"].strip()
            action = self._parse(task, raw)
            last_action = action
            observation = self._observe(task, action)
            if observation is None:
                break
            scratch = (f"REQUEST: {task.query}\n\nYOUR LAST ATTEMPT:\n{raw}\n\n"
                       f"OBSERVATION:\n{observation}\n\nFix it and give the final answer.")
        output = self._final_text(last_action)
        return PolicyResult(
            output=output, action=last_action,
            trace={"policy": "live", "model": self.model,
                   "injection_source": injection.source, "injection_note": injection.note,
                   "similarity": injection.similarity},
        )

    def _parse(self, task, raw: str) -> Dict[str, Any]:
        env = task.env
        if env == "sql":
            sql = envs.extract_sql(raw)
            return {"kind": "sql", "sql": sql, "sql_template": sql}
        obj = _maybe_json(raw)
        if env == "toolcall":
            if isinstance(obj, dict) and obj.get("clarify"):
                return {"kind": "clarify", "missing": obj.get("missing", "the required argument")}
            if isinstance(obj, dict) and obj.get("tool"):
                return {"kind": "call", "tool": obj["tool"], "args": obj.get("args") or {}}
            calls = envs._extract_tool_calls(raw)
            if calls:
                return {"kind": "call", "tool": calls[0]["tool"], "args": calls[0]["args"]}
            if "?" in raw:
                return {"kind": "clarify", "missing": "the required argument"}
            return {"kind": "text", "text": raw}
        if env == "science":
            if isinstance(obj, dict) and obj.get("tool"):
                return {"kind": "measure", "tool": obj["tool"], "args": obj.get("args") or {},
                        "value": obj.get("value")}
            return {"kind": "measure", "tool": "measure_property", "args": {}, "value": None}
        if env == "healthcare":
            if isinstance(obj, dict) and "acuity" in obj:
                try:
                    acuity = int(obj.get("acuity"))
                except Exception:
                    acuity = 3
                return {"kind": "triage", "acuity": max(1, min(5, acuity)),
                        "escalate": bool(obj.get("escalate")),
                        "note": str(obj.get("note", ""))}
            return {"kind": "triage", "acuity": 3, "escalate": False, "note": raw[:200]}
        if env == "trading":
            if isinstance(obj, dict) and obj.get("direction"):
                try:
                    size = float(obj.get("size", 0) or 0)
                except Exception:
                    size = 0.0
                return {"kind": "trade", "direction": str(obj["direction"]).lower(),
                        "size": size, "reason": str(obj.get("reason", ""))}
            return {"kind": "trade", "direction": "flat", "size": 0, "reason": raw[:200]}
        return {"kind": "text", "text": raw}

    def _observe(self, task, action: Dict[str, Any]) -> Optional[str]:
        """Execute the action if the environment supports it; return the observation."""
        if action.get("kind") == "sql" and task.env == "sql":
            group = envs.get_group(task.group)
            if not group:
                return None
            rows = envs._exec_sql(group.schema_sql or "", action.get("sql", ""))
            if rows is None:
                return "SQL ERROR: the query could not be executed. Check table and column names."
            return "RESULT ROWS:\n" + "\n".join(rows[:20])
        return None

    def _final_text(self, action: Dict[str, Any]) -> str:
        kind = action.get("kind")
        if kind == "sql":
            return f"```sql\n{action.get('sql', '')}\n```"
        if kind == "clarify":
            return "Could you confirm " + str(action.get("missing", "the required argument")) + "?"
        if kind == "call":
            return json.dumps({"tool": action.get("tool"), "args": action.get("args") or {}})
        if kind == "measure":
            return json.dumps({"tool": action.get("tool"), "args": action.get("args") or {},
                               "value": action.get("value")})
        if kind == "triage":
            return json.dumps({"acuity": action.get("acuity"),
                               "escalate": bool(action.get("escalate")),
                               "note": action.get("note", "")})
        if kind == "trade":
            return json.dumps({"direction": action.get("direction"), "size": action.get("size"),
                               "reason": action.get("reason", "")})
        return str(action.get("text", ""))


def _maybe_json(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def build_policy(cfg: Dict[str, Any], llm=None):
    mode = (cfg or {}).get("mode", "simulated")
    if mode == "live":
        return LivePolicy(cfg, llm=llm)
    return SimulatedPolicy(cfg)
