"""
Statistical comparison of runs, and the suite's coverage report.

A single run's "+2.0" is indistinguishable from noise. These helpers turn a
set of runs into a claim with a number attached:

  compare_cells   two (stream, evolver) cells, with a Welch t-test and a
                  95% confidence interval on the difference.
  rank_cells      the cells of a stream, ranked, with the winner's margin
                  over the runner-up and whether that margin is significant.
  coverage_report which drift types, domains and families the suite
                  actually exercises, and which holes remain.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# Two-sided critical values for the Welch t-test at 95%, by degrees of
# freedom (rounded to the nearest integer). Small table on purpose: the
# suite runs tens of iterations, not thousands.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def _t_crit(df: float) -> float:
    df = max(1, int(round(df)))
    if df in _T95:
        return _T95[df]
    keys = sorted(_T95)
    if df < keys[0]:
        return _T95[keys[0]]
    for k in keys:
        if k >= df:
            return _T95[k]
    return 1.96


def mean_std(xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    if not n:
        return 0.0, 0.0
    mean = sum(xs) / n
    if n < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var)


def welch(a: List[float], b: List[float]) -> Dict[str, Any]:
    """
    Welch's t-test on two samples.

    Returns the difference, a 95% confidence interval and a p-value
    approximation. With n < 2 per side the result is reported as
    inconclusive rather than a false claim of significance.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"difference": (sum(a) / na - sum(b) / nb) if (na and nb) else 0.0,
                "ci_low": None, "ci_high": None, "p_value": None,
                "significant": False,
                "note": "need at least 2 runs per side for a comparison"}
    ma, sa = mean_std(a)
    mb, sb = mean_std(b)
    va, vb = sa * sa, sb * sb
    se2 = va / na + vb / nb
    se = math.sqrt(se2) if se2 > 0 else 0.0
    diff = ma - mb
    if se == 0:
        return {"difference": diff, "ci_low": diff, "ci_high": diff,
                "p_value": 0.0 if diff else 1.0, "significant": diff != 0,
                "note": "zero variance in both samples"}
    df_num = se2 * se2
    df_den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = df_num / df_den if df_den > 0 else na + nb - 2
    t = diff / se
    crit = _t_crit(df)
    return {
        "difference": round(diff, 3),
        "ci_low": round(diff - crit * se, 3),
        "ci_high": round(diff + crit * se, 3),
        "t": round(t, 3),
        "df": round(df, 1),
        "p_value": round(_p_value(t, df), 4),
        "significant": abs(t) > crit,
        "note": "",
    }


def _p_value(t: float, df: float) -> float:
    """Two-sided p-value from a t statistic, via the incomplete beta."""
    try:
        x = df / (df + t * t)
        # Regularised incomplete beta I_x(df/2, 1/2) via a continued fraction.
        a, b = df / 2.0, 0.5
        lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
        front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
        f, c, d = 1.0, 1.0, 0.0
        for i in range(0, 200):
            m = i // 2
            if i == 0:
                num = 1.0
            elif i % 2 == 0:
                num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
            d = 1.0 + num * d
            if abs(d) < 1e-30:
                d = 1e-30
            d = 1.0 / d
            c = 1.0 + num / c
            if abs(c) < 1e-30:
                c = 1e-30
            f *= c * d
            if abs(1.0 - c * d) < 1e-10:
                break
        return max(0.0, min(1.0, 1.0 - front * (f - 1.0)))
    except Exception:
        return 1.0


def compare_cells(cell_a: Dict[str, Any], cell_b: Dict[str, Any],
                  metric: str = "terminal_advantage") -> Dict[str, Any]:
    """Compare two (stream, evolver) cells on one metric."""
    a = cell_a.get("metrics", {}).get(metric, {})
    b = cell_b.get("metrics", {}).get(metric, {})
    xs, ys = a.get("values", []), b.get("values", [])
    out = welch(xs, ys)
    out.update({
        "metric": metric,
        "a": {"label": f"{cell_a.get('evolver')}/{cell_a.get('stream')}",
              "n": a.get("n", len(xs)), "mean": a.get("mean")},
        "b": {"label": f"{cell_b.get('evolver')}/{cell_b.get('stream')}",
              "n": b.get("n", len(ys)), "mean": b.get("mean")},
    })
    return out


def rank_cells(cells: List[Dict[str, Any]],
               metric: str = "terminal_advantage") -> List[Dict[str, Any]]:
    """
    Rank the cells of one stream, with a significance check on the winner.

    "Is the winner actually better, or did it get a lucky draw?" is the
    question a ranked list should answer, so the margin over the runner-up
    carries a confidence interval and a verdict.
    """
    scored = []
    for c in cells:
        m = (c.get("metrics") or {}).get(metric) or {}
        scored.append((m.get("mean", 0.0), m.get("std", 0.0), m.get("n", 0), c))
    scored.sort(key=lambda t: -t[0])
    out = []
    for i, (mean, std, n, c) in enumerate(scored):
        row = {"rank": i + 1, "evolver": c.get("evolver"), "stream": c.get("stream"),
               "n": n, "mean": mean, "std": std, "margin": None,
               "significant": None}
        if i == 0 and len(scored) > 1:
            mean2, std2, n2, c2 = scored[1]
            cmp = compare_cells(c, c2, metric)
            row["margin"] = cmp.get("difference")
            row["ci_low"] = cmp.get("ci_low")
            row["ci_high"] = cmp.get("ci_high")
            row["p_value"] = cmp.get("p_value")
            row["significant"] = cmp.get("significant")
            row["runner_up"] = c2.get("evolver")
        out.append(row)
    return out


def recommend(cells: List[Dict[str, Any]],
              metric: str = "terminal_advantage",
              guard: str = "interference") -> Dict[str, Any]:
    """
    Recommend an evolver for a stream.

    The recommendation is not just the best mean: a method that wins on
    the headline metric but leaks interference or forgets is a bad choice
    for an agent that has to keep working. So the guardrails are checked
    and stated.
    """
    ranked = rank_cells(cells, metric)
    if not ranked:
        return {"recommendation": None, "reason": "no cells"}
    best = ranked[0]
    guards = {}
    for c in cells:
        if c.get("evolver") != best["evolver"]:
            continue
        gm = c.get("metrics") or {}
        guards = {
            "interference": (gm.get(guard) or {}).get("mean"),
            "forgetting": (gm.get("forgetting") or {}).get("mean"),
            "stability": (gm.get("stability") or {}).get("mean"),
            "plasticity": (gm.get("plasticity") or {}).get("mean"),
        }
    caveats = []
    if best.get("significant") is False:
        caveats.append(f"the margin over {best.get('runner_up')} is not significant "
                       f"at 95% - treat the ranking as a tie")
    for k, v in guards.items():
        if v is not None and k in ("interference", "forgetting") and v > 1.0:
            caveats.append(f"{k} is {v:.2f} pp - the method still leaks between families")
    return {
        "recommendation": best["evolver"],
        "metric": metric,
        "mean": best["mean"],
        "margin_over_runner_up": best.get("margin"),
        "significant": best.get("significant"),
        "guards": guards,
        "caveats": caveats,
        "ranking": ranked,
    }


# --------------------------------------------------------------- coverage

def coverage_report(scenarios: Dict[str, Dict[str, Any]],
                    groups_by_id: Dict[str, Any]) -> Dict[str, Any]:
    """
    What the suite exercises, and what it misses.

    A lab that claims coverage should be able to show it. This reports the
    drift types, domains and families every scenario touches, and lists the
    combinations the suite has no scenario for.
    """
    def ginfo(gid: str) -> Dict[str, Any]:
        g = groups_by_id.get(gid)
        return {"id": gid, "env": getattr(g, "env", "?"),
                "drift": getattr(g, "drift", "?"),
                "name": getattr(g, "name", gid)}

    rows = []
    drift_types: Dict[str, int] = {}
    domains: Dict[str, int] = {}
    families_used: Dict[str, int] = {}
    envs_seen: Dict[str, int] = {}

    for sid, meta in scenarios.items():
        gids = list(meta.get("groups") or [])
        drifts = sorted({ginfo(g)["drift"] for g in gids})
        # A compound scenario is labelled as such by the lab, and its
        # constituent drift types are also reported.
        declared = meta.get("drift", "")
        if declared and declared not in drifts:
            drifts = sorted(set(drifts) | {declared})
        envs_ = sorted({ginfo(g)["env"] for g in gids})
        for d in drifts:
            drift_types[d] = drift_types.get(d, 0) + 1
        for e in envs_:
            envs_seen[e] = envs_seen.get(e, 0) + 1
        for g in gids:
            families_used[g] = families_used.get(g, 0) + 1
        domains[sid] = envs_
        rows.append({
            "id": sid, "name": meta.get("name", sid),
            "description": meta.get("description", ""),
            "declared_drift": meta.get("drift", ""),
            "actual_drifts": drifts,
            "families": [ginfo(g) for g in gids],
            "K": len(set(gids)),
            "environments": envs_,
        })

    all_families = set(groups_by_id)
    unused = sorted(all_families - set(families_used))
    all_envs = sorted({ginfo(g)["env"] for g in all_families})

    # Which (drift type) x (domain) combinations exist as families but have
    # no scenario that puts them together.
    combos_needed = set()
    for g in all_families:
        combos_needed.add((ginfo(g)["drift"], ginfo(g)["env"]))
    combos_covered = set()
    for r in rows:
        for d in r["actual_drifts"]:
            for e in r["environments"]:
                combos_covered.add((d, e))

    return {
        "scenarios": rows,
        "totals": {
            "scenarios": len(rows),
            "families": len(all_families),
            "environments": len(all_envs),
            "drift_types": len(drift_types),
        },
        "drift_types": drift_types,
        "environments": envs_seen,
        "families_used": families_used,
        "unused_families": unused,
        "combinations_covered": sorted(f"{d}/{e}" for d, e in combos_covered),
        "combinations_missing": sorted(f"{d}/{e}" for d, e in (combos_needed - combos_covered)),
    }
