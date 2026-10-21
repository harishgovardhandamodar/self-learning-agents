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


def cohens_d(a: List[float], b: List[float]) -> float:
    """
    Cohen's d: the effect size behind a difference.

    A p-value says "probably not zero". This says how big the difference
    is in units of the pooled standard deviation, which is what decides
    whether a statistically significant result is worth acting on.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = mean_std(a)[0], mean_std(b)[0]
    va, vb = mean_std(a)[1] ** 2, mean_std(b)[1] ** 2
    na, nb = len(a), len(b)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled <= 0:
        return 0.0
    return (ma - mb) / math.sqrt(pooled)


def effect_label(d: float) -> str:
    """The conventional reading of a Cohen's d."""
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def power_for_margin(std: float, n: int, margin: float, alpha: float = 0.05) -> float:
    """
    The probability of detecting a margin of this size with this many runs.

    Answers "how many iterations do I need?". With no power, a null result
    is uninterpretable: it may mean "no effect" or "not enough runs".
    """
    if n < 2 or std <= 0 or margin == 0:
        return 1.0 if margin == 0 else 0.0
    # Normal approximation to the two-sample t-test.
    se = std * math.sqrt(2.0 / n)
    if se <= 0:
        return 1.0
    z_alpha = 1.96 if alpha >= 0.05 else 2.576
    lam = abs(margin) / se
    # P(|Z| > z_alpha | non-central shift lam) via the normal CDF.
    from math import erf, sqrt
    def ncdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    power = 1.0 - ncdf(z_alpha - lam) + ncdf(-z_alpha - lam)
    return max(0.0, min(1.0, power))


def iterations_needed(std: float, margin: float, target_power: float = 0.8,
                      alpha: float = 0.05) -> Optional[int]:
    """How many runs per cell are needed to resolve a margin."""
    if std <= 0 or margin == 0:
        return None
    for n in range(2, 501):
        if power_for_margin(std, n, margin, alpha) >= target_power:
            return n
    return None


def holm_bonferroni(pvalues: Dict[str, float], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Multiplicity control across a family of comparisons.

    The suite compares many cells; without correction, some will look
    significant by chance alone. Holm's step-down procedure is the standard
    correction and needs no independence assumption.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Any] = {"adjusted": {}, "significant": {}, "alpha": alpha,
                           "n_comparisons": m}
    running_max = 0.0
    for i, (key, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running_max = max(running_max, adj)
        out["adjusted"][key] = round(running_max, 6)
        out["significant"][key] = running_max < alpha
    return out


def compare_cells(cell_a: Dict[str, Any], cell_b: Dict[str, Any],
                  metric: str = "terminal_advantage") -> Dict[str, Any]:
    """Compare two (stream, evolver) cells on one metric."""
    a = cell_a.get("metrics", {}).get(metric, {})
    b = cell_b.get("metrics", {}).get(metric, {})
    xs, ys = a.get("values", []), b.get("values", [])
    out = welch(xs, ys)
    d = cohens_d(xs, ys)
    out.update({
        "metric": metric,
        "a": {"label": f"{cell_a.get('evolver')}/{cell_a.get('stream')}",
              "n": a.get("n", len(xs)), "mean": a.get("mean")},
        "b": {"label": f"{cell_b.get('evolver')}/{cell_b.get('stream')}",
              "n": b.get("n", len(ys)), "mean": b.get("mean")},
        "effect_size": round(d, 3),
        "effect_label": effect_label(d),
        "pooled_std": round(_pooled_std(xs, ys), 3),
    })
    return out


def _pooled_std(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va, vb = mean_std(a)[1] ** 2, mean_std(b)[1] ** 2
    return math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) /
                     (len(a) + len(b) - 2))


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
