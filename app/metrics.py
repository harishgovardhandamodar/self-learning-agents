"""
Continual-learning metrics, following arXiv:2610.25858 §3.2.

Notation is kept identical to the paper so numbers can be compared directly:

  A[t,k]  accuracy (%) of the frozen harness c_t on the hold-out set of family k
  B_k     A[0,k], the seed baseline for family k
  Dbar    average advantage over all post-seed checkpoints and all families
  Term.   terminal advantage: mean of A[T,k] - B_k over families
  P_k     plasticity of family k: A[tau_k,k] - B_k  (gain during its own block)
  I_k     pre-block interference: mean drop below seed at checkpoints before
          family k's block starts
  F_k     post-block forgetting: drop from the end of family k's block to the
          end of the stream
  S       cross-block stability: -(Ibar + Fbar)

Everything is reported in percentage points (pp.) over the seed baseline.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# Thresholds below which a failure mode is reported as "not observed".
INTERFERENCE_THRESHOLD = 0.5      # pp.
FORGETTING_THRESHOLD = 0.5        # pp.
FOREIGN_RETRIEVAL_THRESHOLD = 0.2  # fraction of eval episodes
COLLISION_THRESHOLD = 0.2         # fraction of cross-group retrievals


def _acc(checkpoints: Sequence[Dict[str, Any]], t: int, group: str) -> Optional[float]:
    for cp in checkpoints:
        if cp["t"] == t:
            m = cp["metrics"].get(group)
            return None if m is None else float(m)
    return None


def compute_metrics(checkpoints: List[Dict[str, Any]], stream) -> Dict[str, Any]:
    """
    Compute the paper's metrics from a run's checkpoints.

    `checkpoints` must contain the seed checkpoint (t=0) plus one checkpoint at
    every block boundary, each with `metrics` = {group_id: accuracy_percent}.
    """
    if not checkpoints:
        return {"error": "no checkpoints"}

    groups = list(stream.groups)
    K = len(groups)
    sigma, tau = stream.sigma, stream.tau
    first_tau = getattr(stream, "first_tau", tau)
    T = stream.T
    post_seed = [cp for cp in checkpoints if cp["t"] > 0]
    post_seed.sort(key=lambda c: c["t"])
    M = len(post_seed)

    # B_k: the seed baseline per family
    seed_cp = next((cp for cp in checkpoints if cp["t"] == 0), None)
    B: Dict[str, float] = {}
    for g in groups:
        v = _acc(checkpoints, 0, g)
        B[g] = float(v) if v is not None else 0.0

    # A[t,k] table
    A: Dict[int, Dict[str, Optional[float]]] = {}
    for cp in checkpoints:
        A[cp["t"]] = {g: cp["metrics"].get(g) for g in groups}

    # Advantage per (checkpoint, family)
    advantage: Dict[int, Dict[str, Optional[float]]] = {
        t: {g: (None if v is None else v - B[g]) for g, v in row.items()}
        for t, row in A.items()
    }

    # Dbar (Eq. 8) and Term.
    dbar_vals = [advantage[t][g] for t in [cp["t"] for cp in post_seed] for g in groups
                 if advantage[t][g] is not None]
    dbar = sum(dbar_vals) / len(dbar_vals) if dbar_vals else 0.0
    term_vals = [advantage.get(T, {}).get(g) for g in groups if advantage.get(T, {}).get(g) is not None]
    term = sum(term_vals) / len(term_vals) if term_vals else 0.0

    # Plasticity (Eq. 9)
    P_k: Dict[str, float] = {}
    for g in groups:
        v = advantage.get(tau.get(g, 0), {}).get(g)
        P_k[g] = float(v) if v is not None else 0.0
    P = sum(P_k.values()) / K if K else 0.0

    # Pre-block interference (Eq. 10)
    I_k: Dict[str, float] = {}
    for g in groups:
        s = sigma.get(g, 0)
        vals = []
        for cp in post_seed:
            if cp["t"] <= s:
                a = advantage.get(cp["t"], {}).get(g)
                if a is not None:
                    vals.append(max(0.0, -a))
        I_k[g] = sum(vals) / len(vals) if vals else 0.0
    I_bar = sum(I_k.values()) / K if K else 0.0

    # Post-block forgetting (Eq. 10)
    F_k: Dict[str, float] = {}
    for g in groups:
        tk = tau.get(g, 0)
        a_tau = advantage.get(tk, {}).get(g)
        a_T = advantage.get(T, {}).get(g)
        if a_tau is None or a_T is None:
            F_k[g] = 0.0
        else:
            F_k[g] = max(0.0, a_tau - a_T)
    F_bar = sum(F_k.values()) / K if K else 0.0

    S = -(I_bar + F_bar)

    # ---------------------------------------------------------- transfer
    # Forward transfer: the gain a family shows at the checkpoint that ends
    # its block, minus what it could have gained from its own block alone.
    # A family that was already above B_k before its block, and is higher
    # at the end of it, was helped by something it learned earlier.
    FT_k: Dict[str, float] = {}
    for g in groups:
        s = sigma.get(g, 0)
        tk = first_tau.get(g, tau.get(g, 0))
        pre = advantage.get(s, {}).get(g)
        at = advantage.get(tk, {}).get(g)
        if pre is None or at is None:
            FT_k[g] = 0.0
        else:
            FT_k[g] = round(max(0.0, at) - max(0.0, pre), 3)
    FT = sum(FT_k.values()) / K if K else 0.0

    # Backward transfer: what happened to a family after its block ended.
    BT_k: Dict[str, float] = {}
    for g in groups:
        tk = first_tau.get(g, tau.get(g, 0))
        a_tau = advantage.get(tk, {}).get(g)
        a_T = advantage.get(T, {}).get(g)
        BT_k[g] = 0.0 if (a_tau is None or a_T is None) else round(a_T - a_tau, 3)
    BT = sum(BT_k.values()) / K if K else 0.0

    # ------------------------------------------------- area under the curve
    # The shape of learning, not just its end state. The trapezoid is scaled
    # by the stream length so streams of different length compare.
    curve: List[Dict[str, Any]] = []
    for cp in post_seed:
        row: Dict[str, Any] = {"t": cp["t"], "label": cp.get("label", ""),
                               "mean_advantage": None, "per_group": {}}
        vals = [advantage[cp["t"]][g] for g in groups
                if advantage[cp["t"]].get(g) is not None]
        if vals:
            row["mean_advantage"] = round(sum(vals) / len(vals), 3)
        for g in groups:
            a = advantage[cp["t"]].get(g)
            if a is not None:
                row["per_group"][g] = round(a, 3)
        curve.append(row)
    auc = 0.0
    if len(curve) >= 2 and all(c["mean_advantage"] is not None for c in curve):
        area = 0.0
        for a, b in zip(curve, curve[1:]):
            dt = b["t"] - a["t"]
            area += dt * (a["mean_advantage"] + b["mean_advantage"]) / 2.0
        auc = area / max(1, T)
    elif curve and curve[0]["mean_advantage"] is not None:
        auc = curve[0]["mean_advantage"]

    return {
        "K": K, "T": T, "M": M,
        "groups": groups,
        "sigma": sigma, "tau": tau, "first_tau": first_tau,
        "checkpoints": [cp["t"] for cp in checkpoints],
        "baseline": B,
        "accuracy": {str(t): row for t, row in A.items()},
        "advantage": {str(t): row for t, row in advantage.items()},
        "terminal_advantage": round(term, 3),
        "average_advantage": round(dbar, 3),
        "plasticity": round(P, 3),
        "plasticity_per_group": {g: round(v, 3) for g, v in P_k.items()},
        "interference": round(I_bar, 3),
        "interference_per_group": {g: round(v, 3) for g, v in I_k.items()},
        "forgetting": round(F_bar, 3),
        "forgetting_per_group": {g: round(v, 3) for g, v in F_k.items()},
        "stability": round(S, 3),
        "forward_transfer": round(FT, 3),
        "forward_transfer_per_group": FT_k,
        "backward_transfer": round(BT, 3),
        "backward_transfer_per_group": BT_k,
        "area_under_curve": round(auc, 3),
        "learning_curve": curve,
        "per_group_terminal": {
            g: (None if advantage.get(T, {}).get(g) is None else round(advantage[T][g], 3))
            for g in groups
        },
    }



# ------------------------------------------------------------ diagnostics

def retrieval_diagnostics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    The retrieval-side mechanism, from paper §4.4 / Figure 4a.

    `events` are per-episode records: {"t", "group", "source", "owner", "similarity"}.

    foreign share per group and checkpoint  - the fraction of evaluation
    episodes that retrieved something learned in a different task group.
    cross-group retrieval matrix             - where the foreign items came from.
    """
    out: Dict[str, Any] = {"foreign_share": {}, "cross_matrix": {}, "events": len(events)}
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    matrix: Dict[str, Dict[str, int]] = {}

    for e in events:
        g = e.get("group", "")
        by_group.setdefault(g, []).append(e)
        owner = e.get("owner") or ""
        source = e.get("source", "")
        if source in ("none", "", "prior"):
            continue
        if owner and owner != g:
            matrix.setdefault(g, {})
            matrix[g][owner] = matrix[g].get(owner, 0) + 1

    for g, evs in by_group.items():
        retrieved = [e for e in evs if e.get("source") not in ("none", "", "prior")]
        foreign = [e for e in retrieved if e.get("owner") and e.get("owner") != g]
        out["foreign_share"][g] = {
            "retrieved": len(retrieved),
            "foreign": len(foreign),
            "share": round(len(foreign) / len(retrieved), 4) if retrieved else 0.0,
        }
    out["cross_matrix"] = matrix
    return out


def prompt_attribution(checkpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Per-group share of the shared prompt, from paper §3.2 / Figure 3a,b.

    A declining share is the visible trace of a later block compressing or
    overwriting an earlier group's rules.

    Returns {"shares": ..., "sizes": ...}. `sizes` holds absolute character
    counts per owner where the evolver reports them (scoped sections). A
    share can fall for a benign reason - a new section appears and dilutes
    the rest - while an absolute shrink means content was actually removed.
    """
    series: Dict[str, Dict[str, float]] = {}
    sizes: Dict[str, Dict[str, float]] = {}
    for cp in checkpoints:
        diag = cp.get("diagnostics") or {}
        attr = diag.get("strategy_attribution") or diag.get("scoped_attribution") or {}
        if not attr:
            continue
        for owner, share in attr.items():
            series.setdefault(owner, {})[str(cp["t"])] = share
        for owner, chars in (diag.get("scoped_sections") or {}).items():
            sizes.setdefault(owner, {})[str(cp["t"])] = float(chars)
    return {"shares": series, "sizes": sizes}


def detect_failure_modes(metrics: Dict[str, Any],
                          retrieval: Optional[Dict[str, Any]] = None,
                          attribution: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Turn the numbers into named diagnoses.

    The paper's four findings are checked for directly, each with the evidence
    that produced it.
    """
    modes: List[Dict[str, Any]] = []

    I = metrics.get("interference", 0.0)
    F = metrics.get("forgetting", 0.0)
    P = metrics.get("plasticity", 0.0)
    S = metrics.get("stability", 0.0)

    if I >= INTERFERENCE_THRESHOLD:
        worst = max(metrics.get("interference_per_group", {}).items(),
                    key=lambda kv: kv[1], default=("", 0))
        modes.append({
            "id": "pre-block-interference",
            "severity": "high" if I >= 2 else "medium",
            "title": "Pre-block interference",
            "detail": (f"Families evaluated before their own block lose {I:.2f} pp on average "
                       f"against the seed harness. Worst: {worst[0]} ({worst[1]:.2f} pp)."),
            "evidence": {"interference": I, "per_group": metrics.get("interference_per_group")},
            "paper_ref": "§4.3 / Fig. 3c",
            "cause": "task-specific rules are written to a surface that every task reads",
        })
    if F >= FORGETTING_THRESHOLD:
        worst = max(metrics.get("forgetting_per_group", {}).items(),
                    key=lambda kv: kv[1], default=("", 0))
        modes.append({
            "id": "post-block-forgetting",
            "severity": "high" if F >= 2 else "medium",
            "title": "Post-block forgetting",
            "detail": (f"Families lose {F:.2f} pp between the end of their own block and the end "
                       f"of the stream. Worst: {worst[0]} ({worst[1]:.2f} pp)."),
            "evidence": {"forgetting": F, "per_group": metrics.get("forgetting_per_group")},
            "paper_ref": "§4.3 / Fig. 3a,b",
            "cause": "later blocks compress or overwrite earlier rules in the shared surface",
        })

    if retrieval:
        worst_g, worst_v = "", 0.0
        for g, d in (retrieval.get("foreign_share") or {}).items():
            if d.get("share", 0) > worst_v:
                worst_g, worst_v = g, d.get("share", 0)
        if worst_v >= FOREIGN_RETRIEVAL_THRESHOLD:
            modes.append({
                "id": "foreign-memory-injection",
                "severity": "high" if worst_v >= 0.5 else "medium",
                "title": "Foreign memory injection",
                "detail": (f"{worst_v*100:.0f}% of evaluated episodes for '{worst_g}' retrieved "
                           f"an experience learned in a different task family."),
                "evidence": retrieval.get("foreign_share"),
                "paper_ref": "§4.4 / Fig. 4a",
                "cause": "retrieval always returns the nearest match, with no scope check",
            })
        matrix = retrieval.get("cross_matrix") or {}
        if matrix:
            total = sum(sum(v.values()) for v in matrix.values())
            if total:
                worst_pair = max(
                    ((g, o, c) for g, row in matrix.items() for o, c in row.items()),
                    key=lambda t: t[2])
                if worst_pair[2] / total >= COLLISION_THRESHOLD:
                    modes.append({
                        "id": "retrieval-collision",
                        "severity": "medium",
                        "title": "Retrieval collision",
                        "detail": (f"Most foreign retrievals cross from '{worst_pair[0]}' to "
                                   f"'{worst_pair[1]}' - similar queries, incompatible workflows."),
                        "evidence": matrix,
                        "paper_ref": "§4.4 / Fig. 4b",
                        "cause": "similarity cannot tell topical overlap from workflow compatibility",
                    })

    if attribution:
        shares = attribution.get("shares", {}) if isinstance(attribution, dict) else {}
        sizes = attribution.get("sizes", {}) if isinstance(attribution, dict) else {}
        declines = []
        for owner, series in shares.items():
            pts = sorted(series.items(), key=lambda kv: int(kv[0]))
            if len(pts) >= 2 and pts[0][1] > 0:
                drop = pts[0][1] - pts[-1][1]
                if drop <= 20:
                    continue
                # A share can fall benignly when a new section appears and
                # dilutes the rest. Only absolute shrinkage is overwriting.
                if owner in sizes:
                    spts = sorted(sizes[owner].items(), key=lambda kv: int(kv[0]))
                    peak = max(v for _, v in spts) if spts else 0
                    if not peak or spts[-1][1] >= 0.8 * peak:
                        continue
                declines.append((owner, round(drop, 2)))
        if declines:
            owner, drop = max(declines, key=lambda t: t[1])
            modes.append({
                "id": "prompt-overwriting",
                "severity": "medium",
                "title": "Prompt overwriting",
                "detail": (f"The share of the shared prompt owned by '{owner}' fell by {drop} "
                           f"percentage points across the stream."),
                "evidence": attribution,
                "paper_ref": "§4.3 / Fig. 3a,b",
                "cause": "a later block rewrote or compressed an earlier block's rules",
            })

    if P > 0 and S < 0:
        modes.append({
            "id": "plasticity-stability-tradeoff",
            "severity": "medium",
            "title": "Plasticity / stability trade-off",
            "detail": (f"Plasticity {P:+.2f} pp with stability {S:+.2f} pp: the harness gains "
                       f"on the block it is training and loses ground on the others."),
            "evidence": {"plasticity": P, "stability": S},
            "paper_ref": "§4.2 / Fig. 2",
            "cause": "adaptation is not isolated per task family",
        })

    if not modes:
        modes.append({
            "id": "no-failure-observed",
            "severity": "none",
            "title": "No failure mode detected",
            "detail": ("Interference, forgetting, foreign retrieval and prompt overwriting are "
                       "all below threshold for this run."),
            "evidence": {"interference": I, "forgetting": F, "stability": S},
            "paper_ref": "",
            "cause": "",
        })
    return modes


def compare_runs(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Table rows for the cross-method comparison (paper Table 1 shape)."""
    rows = []
    for r in runs:
        m = r.get("metrics") or {}
        if not m or "error" in m:
            continue
        rows.append({
            "run_id": r.get("run_id") or r.get("id"),
            "name": r.get("name", ""),
            "evolver": r.get("evolver", ""),
            "policy": r.get("policy", ""),
            "stream": r.get("stream", ""),
            "terminal_advantage": m.get("terminal_advantage"),
            "average_advantage": m.get("average_advantage"),
            "plasticity": m.get("plasticity"),
            "stability": m.get("stability"),
            "interference": m.get("interference"),
            "forgetting": m.get("forgetting"),
        })
    return rows
