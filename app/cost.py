"""
Cost accounting for a run.

A self-learning agent spends something to store, retrieve and rewrite rules.
A method that buys 2 percentage points while costing ten times the tokens
is a bad trade, and an evaluation suite that cannot see that will recommend
the expensive method every time.

The accounting is deliberately simple and honest about what it measures:

  retrieval cost   how much of the harness had to be read to answer a task
  prompt cost      how many characters of harness text were injected
  memory growth    how many lessons accumulated (and were never used)
  write cost       how many characters of harness text were written

In simulated mode these are counted, not billed: the numbers describe the
SHAPE of the method's overhead, which is what differs between methods. In
live mode the same counters are filled from the actual model calls, so the
two are directly comparable.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Rough price per 1k characters of prompt, used to turn character counts
# into a comparable "prompt units" figure. It is a relative yardstick, not
# a price list - the point is the ratio between methods, not the absolute.
CHARS_PER_UNIT = 1000.0


def empty_cost() -> Dict[str, Any]:
    return {
        "retrieval_chars": 0,
        "prompt_chars": 0,
        "lessons_created": 0,
        "lessons_injected": 0,
        "lessons_used": 0,
        "rules_written": 0,
        "write_chars": 0,
        "scopes": 0,
        "eval_episodes": 0,
    }


def account_episode(cost: Dict[str, Any], injection: Any, used: bool) -> None:
    """One episode: what was injected, and was it used."""
    cost["eval_episodes"] = cost.get("eval_episodes", 0) + 1
    text = getattr(injection, "text", "") or ""
    cost["prompt_chars"] = cost.get("prompt_chars", 0) + len(text)
    lessons = list(getattr(injection, "lessons", []) or [])
    cost["lessons_injected"] = cost.get("lessons_injected", 0) + len(lessons)
    if used:
        cost["lessons_used"] = cost.get("lessons_used", 0) + 1
    # Retrieval cost: the lessons considered, weighted by how far down the
    # ranked list they were (a deep hit is more expensive than a shallow
    # one), plus the harness text that had to be read to find them.
    for i, _l in enumerate(lessons):
        cost["retrieval_chars"] = cost.get("retrieval_chars", 0) + 200 + 20 * i


def account_lesson(cost: Dict[str, Any], lesson: Any, harness: Any = None) -> None:
    """One lesson written to the harness."""
    cost["lessons_created"] = cost.get("lessons_created", 0) + 1
    text = getattr(lesson, "text", "") or ""
    cost["write_chars"] = cost.get("write_chars", 0) + len(text)
    cost["rules_written"] = cost.get("rules_written", 0) + 1
    if harness is not None:
        scope = getattr(lesson, "scope", "") or ""
        scopes = set(cost.get("_scopes", ()))
        if scope:
            scopes.add(scope)
        cost["_scopes"] = scopes


def finalise(cost: Dict[str, Any], n_eval: Optional[int] = None) -> Dict[str, Any]:
    """Turn the counters into the comparable summary the report shows."""
    out = {k: v for k, v in cost.items() if not k.startswith("_")}
    out["scopes"] = len(cost.get("_scopes", ()))
    out["prompt_units"] = round(out.get("prompt_chars", 0) / CHARS_PER_UNIT, 3)
    out["write_units"] = round(out.get("write_chars", 0) / CHARS_PER_UNIT, 3)
    injected = out.get("lessons_injected", 0)
    out["lessons_per_episode"] = round(
        injected / out["eval_episodes"], 3) if out.get("eval_episodes") else 0.0
    used = out.get("lessons_used", 0)
    out["lesson_hit_rate"] = round(used / injected, 3) if injected else 0.0
    # Memory that was written but never injected is dead weight.
    out["unused_lessons"] = max(0, out.get("lessons_created", 0) - used)
    out["total_units"] = round(out.get("prompt_units", 0.0) + out.get("write_units", 0.0), 3)
    if n_eval:
        out["units_per_eval"] = round(out["total_units"] / n_eval, 4)
    return out


def efficiency_ledger(cost: Dict[str, Any], gain: float) -> Dict[str, Any]:
    """
    The trade, stated plainly: what was bought and what it cost.

    `gain` is the terminal advantage in percentage points; `units` is the
    prompt + write volume. The ratio is "percentage points per 100 prompt
    units" - a method that needs a huge harness to gain a little is
    expensive, and the ledger says so.
    """
    units = cost.get("total_units", 0.0) or 0.0
    per_100 = round(gain / (units / 100.0), 3) if units > 0 else None
    return {
        "gain_pp": round(gain, 3),
        "prompt_units": cost.get("prompt_units", 0.0),
        "write_units": cost.get("write_units", 0.0),
        "total_units": units,
        "pp_per_100_units": per_100,
        "lessons_created": cost.get("lessons_created", 0),
        "lesson_hit_rate": cost.get("lesson_hit_rate", 0.0),
        "unused_lessons": cost.get("unused_lessons", 0),
    }
