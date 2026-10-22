"""
The suite runner and the suite report.

The lab's unit of work is a comparison, and a comparison over the whole
suite is 21 scenarios x 5 evolvers x N iterations. Doing that by hand is
the workflow gap this module closes:

  run_suite      launch the whole grid as one job, with progress
  build_report   one table: every cell with mean +/- std, the effect size
                 against the best cell, significance after multiplicity
                 correction, guardrail violations, cost, and the ceiling
  advise         what to do next for a given domain
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

from . import stats as stats_mod

# The metrics the report ranks cells on, in the order a reader wants them.
HEADLINE = ["terminal_advantage", "plasticity", "stability",
            "interference", "forgetting", "forward_transfer",
            "backward_transfer", "area_under_curve"]


def suite_scenarios(scope: str = "all") -> List[str]:
    """The scenarios a suite run covers.

    "core" is the paper's four plus the per-domain drift streams; "all"
    adds the hard shapes, the ordering controls and the single-family
    baselines. "fast" is the cheap subset a user runs first.
    """
    from . import streams as streams_mod
    all_ids = list(streams_mod.SCENARIOS)
    core = ["topic", "workflow", "domain", "compound",
            "healthcare-drift", "trading-drift", "science-drift"]
    fast = ["workflow", "topic", "domain-workflow", "healthcare-drift"]
    if scope == "core":
        return [s for s in core if s in all_ids]
    if scope == "fast":
        return [s for s in fast if s in all_ids]
    return all_ids


def estimate_cost(scenarios: List[str], evolvers: List[str], iterations: int,
                  block_size: int = 20, n_eval: int = 30) -> Dict[str, Any]:
    """What a suite run will cost, before it is launched."""
    from . import streams as streams_mod
    episodes = 0
    eval_episodes = 0
    for sid in scenarios:
        try:
            st = streams_mod.build_stream(sid, block_size=block_size, n_eval=n_eval)
        except KeyError:
            continue
        episodes += st.T
        eval_episodes += st.T * len(st.groups)
    runs = len(scenarios) * len(evolvers) * iterations
    return {
        "runs": runs,
        "scenarios": len(scenarios),
        "evolvers": len(evolvers),
        "iterations": iterations,
        "training_episodes": episodes * len(evolvers) * iterations,
        "evaluation_episodes": eval_episodes * len(evolvers) * iterations,
        "note": "simulated mode: seconds to a couple of minutes. Live mode "
                "multiplies the evaluation episodes by the model's latency.",
    }


def build_report(cells: List[Dict[str, Any]], metric: str = "terminal_advantage",
                 alpha: float = 0.05) -> Dict[str, Any]:
    """
    The suite report: one row per cell, with the evidence for its rank.

    Every cell gets its mean and spread, its effect size against the best
    cell on the headline metric, a significance verdict that accounts for
    the number of comparisons, and its cost. The point is that a reader can
    see not just who won but by how much, at what cost, and whether the
    margin is real.
    """
    by_stream: Dict[str, List[Dict[str, Any]]] = {}
    for c in cells:
        by_stream.setdefault(c.get("stream", "?"), []).append(c)

    rows: List[Dict[str, Any]] = []
    for stream, group in sorted(by_stream.items()):
        # Only cells with the same harness geometry are comparable: a
        # 4-episode probe run and a 20-episode measured run have different
        # variances, and comparing them says nothing.
        by_geom: Dict[Any, List[Dict[str, Any]]] = {}
        for c in group:
            geom = (c.get("block_size", 20), c.get("n_eval", 30))
            by_geom.setdefault(geom, []).append(c)

        for geom, cells_g in sorted(by_geom.items()):
            ranked = stats_mod.rank_cells(cells_g, metric)
            # The best CELL (not the best rank row) is what the others are
            # compared against: rank_cells returns summaries, compare_cells
            # needs the raw values.
            by_evolver = {c.get("evolver"): c for c in cells_g}
            best_evolver = ranked[0]["evolver"] if ranked else None
            best_cell = by_evolver.get(best_evolver)
            # Multiplicity: compare every cell against the best one.
            pvals: Dict[str, float] = {}
            for c in cells_g:
                if best_cell is None or c.get("evolver") == best_evolver:
                    continue
                cmp = stats_mod.compare_cells(best_cell, c, metric)
                if cmp.get("p_value") is not None:
                    pvals[c.get("evolver", "?")] = cmp["p_value"]
            holm = stats_mod.holm_bonferroni(pvals, alpha) if pvals else None

            for r in ranked:
                cell = by_evolver.get(r["evolver"], {})
                gm = cell.get("metrics") or {}
                row = {
                    "stream": stream,
                    "evolver": r["evolver"],
                    "block_size": geom[0], "n_eval": geom[1],
                    "n": r["n"],
                    "rank": r["rank"],
                    "mean": r["mean"],
                    "std": r["std"],
                    "metrics": {k: (gm.get(k) or {}).get("mean") for k in HEADLINE
                                if k in gm},
                    "guardrails": cell.get("guardrails", {}),
                    "cost": cell.get("cost", {}),
                    "efficiency": cell.get("efficiency", {}),
                    "is_reference": r["evolver"] == "oracle",
                }
                if best_cell is not None and r["evolver"] != best_evolver:
                    cmp = stats_mod.compare_cells(best_cell, cell, metric)
                    row["margin_vs_best"] = cmp.get("difference")
                    row["ci_low"] = cmp.get("ci_low")
                    row["ci_high"] = cmp.get("ci_high")
                    row["effect_size"] = cmp.get("effect_size")
                    row["effect_label"] = cmp.get("effect_label")
                    row["p_value"] = cmp.get("p_value")
                    row["significant_raw"] = cmp.get("significant")
                    if holm:
                        row["p_adjusted"] = (holm["adjusted"] or {}).get(r["evolver"])
                        row["significant"] = (holm["significant"] or {}).get(r["evolver"])
                    else:
                        row["significant"] = cmp.get("significant")
                    row["n_comparisons"] = holm["n_comparisons"] if holm else 1
                else:
                    row["margin_vs_best"] = 0.0
                    row["significant"] = True
                    row["is_best"] = True
                rows.append(row)
    rows.sort(key=lambda r: (r["stream"], r["rank"]))
    report = {
        "metric": metric,
        "alpha": alpha,
        "rows": rows,
        "streams": sorted(by_stream),
        "cells": len(rows),
        "note": ("significance is Holm-corrected across the cells of one stream; "
                 "'significant' means the margin survives the correction"),
    }
    report["ceiling"] = _ceiling_capture(rows, metric)
    return report


def _ceiling_capture(rows: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    """
    How much of the achievable gain the best real method captured.

    The oracle row is the ceiling: what perfect scope isolation would
    deliver. When a method captures 100% of it, the remaining gap to a
    perfect agent is in the base model's competence, not in the routing -
    which is a different problem with a different fix.
    """
    by_stream: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_stream.setdefault(r["stream"], []).append(r)
    out = []
    for stream, rs in sorted(by_stream.items()):
        ceiling = next((r for r in rs if r.get("is_reference")), None)
        if ceiling is None:
            continue
        ceil_mean = ceiling["metrics"].get(metric) or 0.0
        real = [r for r in rs if not r.get("is_reference")
                and r.get("metrics", {}).get(metric) is not None]
        if not real:
            continue
        best = max(real, key=lambda r: r["metrics"][metric])
        got = best["metrics"][metric]
        meaningful = ceil_mean >= 5.0
        out.append({
            "stream": stream,
            "ceiling": round(ceil_mean, 3),
            "best_real": best["evolver"],
            "best_mean": round(got, 3),
            # A capture percentage over a near-zero ceiling is noise: the
            # ratio is only meaningful when there is a real gain to capture.
            "captured_pct": round(got / ceil_mean * 100, 1) if (ceil_mean and meaningful) else None,
            "headroom_pp": round(ceil_mean - got, 3),
            "meaningful": meaningful,
        })
    return out


# Which environments a domain's scenarios touch. "general" is not an
# environment, so it is mapped to the non-regulated ones: a generic agent
# works with SQL, tool calls and instruments.
DOMAIN_ENVS: Dict[str, List[str]] = {
    "healthcare": ["healthcare"],
    "trading": ["trading"],
    "finance": ["trading", "sql", "toolcall"],
    "general": ["sql", "toolcall", "science"],
}


def domain_scenarios(domain: str) -> List[str]:
    """The scenarios that exercise a domain."""
    from . import envs, streams as streams_mod
    wanted = set(DOMAIN_ENVS.get(domain.lower(), [domain.lower()]))
    out = []
    for sid, meta in streams_mod.SCENARIOS.items():
        envs_in = {envs.get_group(g).env for g in meta.get("groups", [])
                   if envs.get_group(g)}
        if envs_in & wanted:
            out.append(sid)
    return out


def advise(domain: str = "general", report: Optional[Dict[str, Any]] = None,
           scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    What to do next, for a user who has a domain and a design.

    The advice is derived from the measured report, not asserted: which
    evolver to use for this domain's scenarios, what the caveats are, what
    the suite does not yet measure for them, and what to measure next.
    """
    from . import envs, streams as streams_mod

    if report is None:
        report = {"rows": []}
    dom_scenarios = domain_scenarios(domain)
    # The recommendation is about THIS domain: a method that wins on
    # healthcare streams and loses on trading is not the healthcare answer.
    scope = scenarios if scenarios is not None else dom_scenarios
    rows = [r for r in report.get("rows", []) if r.get("stream") in scope] \
        if scope else list(report.get("rows", []))

    out: Dict[str, Any] = {
        "domain": domain,
        "domain_scenarios": dom_scenarios,
        "scoped_to_domain": bool(scope),
        "recommendation": None,
        "caveats": [],
        "next_steps": [],
        "unmeasured": [],
        "wins": {},
    }

    if rows:
        by_stream: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_stream.setdefault(r["stream"], []).append(r)
        wins: Dict[str, int] = {}
        means: Dict[str, List[float]] = {}
        for rs in by_stream.values():
            top = min(rs, key=lambda r: r["rank"])
            # A "win" has to be a win: the method must have actually learned
            # something. The seed harness wins by not adapting, and its
            # advantage is 0 - that is a floor, not a win.
            top_gain = (top.get("metrics") or {}).get(
                report.get("metric", "terminal_advantage")) or 0.0
            if top_gain > 0.5:
                wins[top["evolver"]] = wins.get(top["evolver"], 0) + 1
            for r in rs:
                m = (r.get("metrics") or {}).get(report.get("metric", "terminal_advantage"))
                if m is not None:
                    means.setdefault(r["evolver"], []).append(m)
        scored = sorted(
            wins.items(),
            key=lambda kv: (-kv[1], -(sum(means.get(kv[0], [0])) /
                              max(1, len(means.get(kv[0], [1]))))))
        if scored:
            out["recommendation"] = scored[0][0]
        out["wins"] = wins

        best_rows = [r for r in rows if r.get("is_best")]
        for r in best_rows:
            if r.get("significant") is False:
                out["caveats"].append(
                    f"{r['stream']}: {r['evolver']} leads but the margin over the "
                    f"runner-up is not significant after correction "
                    f"({r.get('n_comparisons')} comparisons)")
            eff = r.get("efficiency") or {}
            if eff.get("pp_per_100_units") is not None and eff["pp_per_100_units"] < 0:
                out["caveats"].append(
                    f"{r['stream']}: the leading method spends "
                    f"{eff.get('total_units')} prompt units for a negative net gain")
        out["next_steps"] = [
            f"Run the {domain} scenarios the suite covers for this domain "
            f"({', '.join(dom_scenarios) or 'none yet'}) and compare the "
            f"evolvers with the batch launcher.",
            "Check the guardrail audit on each run: a method that wins on "
            "accuracy while proposing unsafe actions is not the method to ship.",
            "Read the cost column: compare the gain against the prompt volume "
            "before choosing.",
        ]

    # What the suite does not measure for this domain yet.
    covered = set(dom_scenarios)
    out["unmeasured"] = [
        "live-mode validation: every number in the suite comes from the "
        "simulated policy, so the ranking is unconfirmed against a real model",
        "a cost ceiling: the suite measures prompt volume but not a budget "
        "the agent must stay under",
    ]
    cal = _calibration_note()
    if cal:
        out["calibration"] = cal
    if not covered:
        out["caveats"].append(
            f"the suite has no scenario for the {domain} domain yet")
    return out


def _calibration_note() -> Optional[Dict[str, Any]]:
    """
    The measured live-vs-simulated gap, if a live validation has been run.

    This is the honest limit on every number in the suite. The simulator is
    calibrated to the baselines the paper reports, so the suite is comparable
    to the paper - but a real model may already have mastered a family, and
    then the headroom the simulator measures there is headroom a real model
    does not have. The rankings transfer; the magnitudes do not.
    """
    import json
    from . import benchmarks as bm
    for path in bm.LIVE_VALIDATION_CANDIDATES:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        return {
            "summary": (f"the local model scored {data['live_baseline_pct']}% on the "
                        f"{data['family']} family against the simulator's "
                        f"{data['simulated_baseline_pct']}% "
                        f"({data['delta_pp']:+.0f}pp)"),
            "reading": data.get("reading", ""),
            "calibration": data.get("calibration", ""),
            "source": path.name,
        }
    return None


def power_report(cells: List[Dict[str, Any]], margin: float = 5.0) -> List[Dict[str, Any]]:
    """
    How many iterations each cell would need to resolve a margin.

    Answers "is 20 enough?" per cell instead of everywhere at once, so a
    user can spend iterations where they buy resolution.
    """
    out = []
    for c in cells:
        gm = (c.get("metrics") or {}).get("terminal_advantage") or {}
        std = gm.get("std", 0.0)
        n = gm.get("n", 0)
        needed = stats_mod.iterations_needed(std, margin) if std else None
        out.append({
            "stream": c.get("stream"), "evolver": c.get("evolver"),
            "n": n, "std": std,
            "power_now": round(stats_mod.power_for_margin(std, n, margin), 3) if std else None,
            "iterations_needed_for_5pp": needed,
        })
    return out
