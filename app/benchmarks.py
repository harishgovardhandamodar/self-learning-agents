"""
Reference bands: what a healthy run looks like.

The lab's metrics are only interpretable against something. These bands come
from the 20-iteration sweep (`python3 -m tests.sweep --iterations 20
--block 20 --eval 30`), which is the lab's own regression gate: 600 runs
across 6 scenarios x 5 evolvers, each cell reported as mean +/- std.

Two jobs:

  1. REFERENCE. The UI shows a run's numbers against these bands, so a user
     can tell "Term. +12" from "Term. +12 is what scoped does on workflow
     drift" and from "Term. +12 is actually a regression against seed".
  2. REGRESSION. `tests/test_lab.py` asserts the bands still hold, so a
     change to the harness mechanics cannot silently move the numbers.

Regenerate with:
    python3 -m tests.sweep --iterations 20 --block 20 --eval 30 \\
        --out data/sweep20.json
    python3 -m tests.sweep --refresh-benchmarks data/sweep20.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCHMARK_PATH = (Path(__file__).resolve().parent.parent / "data" / "benchmarks.json")

# What each metric means, in the paper's notation. Shown next to the number
# so the dashboard does not require the paper to be read.
METRIC_GLOSSARY: Dict[str, Dict[str, str]] = {
    "terminal_advantage": {
        "short": "Term.",
        "label": "Terminal advantage",
        "meaning": "Mean accuracy gain over the seed harness at the end of the stream. "
                   "Negative means the agent is worse off than it started.",
    },
    "average_advantage": {
        "short": "Dbar",
        "label": "Average advantage",
        "meaning": "Mean gain over the seed across every post-seed checkpoint and family. "
                   "A summary of the whole run, not just its end state.",
    },
    "plasticity": {
        "short": "P",
        "label": "Plasticity",
        "meaning": "Mean gain a family earned during its own block. High P = the agent "
                   "can still learn new things.",
    },
    "stability": {
        "short": "S",
        "label": "Stability",
        "meaning": "-(interference + forgetting). Positive S = families keep what they learned.",
    },
    "interference": {
        "short": "I",
        "label": "Pre-block interference",
        "meaning": "How much a family loses BEFORE its own block, because earlier blocks "
                   "wrote rules into a surface it reads.",
    },
    "forgetting": {
        "short": "F",
        "label": "Post-block forgetting",
        "meaning": "How much a family loses AFTER its own block, because later blocks "
                   "overwrote or compressed its rules.",
    },
    "forward_transfer": {
        "short": "FT",
        "label": "Forward transfer",
        "meaning": "How much a family gained BEFORE its own block started - competence "
                   "carried in from earlier blocks. Positive means learning generalises.",
    },
    "backward_transfer": {
        "short": "BT",
        "label": "Backward transfer",
        "meaning": "How much a family changed AFTER its own block ended. Positive means "
                   "later learning helped old families; negative means it hurt them.",
    },
    "area_under_curve": {
        "short": "AUC",
        "label": "Area under the curve",
        "meaning": "The shape of learning over the horizon, not just its end state. A run "
                   "that spikes early and collapses late scores close to a flat one.",
    },
}

EVOLVER_GLOSSARY: Dict[str, str] = {
    "seed": "No adaptation. The fixed baseline every run is measured against.",
    "reasoningbank": "Stores one memory per successful episode and retrieves the most "
                     "similar ones by request text, with no scope check.",
    "gepa": "Reflectively rewrites one shared strategy prompt, compressed to a character "
            "budget, so later blocks overwrite earlier rules.",
    "metaharness": "Proposes edits to prompt, memory, skills and controller; accepts an "
                   "edit only when it improves validation accuracy.",
    "scoped": "Files lessons per discovered cluster AND per workflow fingerprint, in "
              "append-only sections. Nothing is overwritten, nothing foreign is injected.",
}

# Fallback bands. Regenerated from the sweep; these are the shipped values
# so the UI has reference numbers even with no sweep file present.
DEFAULT_BANDS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "workflow": {
        "seed": {"terminal_advantage": [0.0, 0.0], "plasticity": [0.0, 0.0],
                 "stability": [0.0, 0.0], "interference": [0.0, 0.0],
                 "forgetting": [0.0, 0.0]},
        "reasoningbank": {"terminal_advantage": [12.08, 4.68], "plasticity": [12.58, 4.57],
                          "stability": [-13.17, 4.04], "interference": [12.67, 3.76],
                          "forgetting": [0.5, 1.34]},
        "gepa": {"terminal_advantage": [-11.75, 6.03], "plasticity": [12.42, 4.51],
                 "stability": [-38.08, 4.75], "interference": [13.92, 4.23],
                 "forgetting": [24.17, 4.6]},
        "metaharness": {"terminal_advantage": [-5.5, 7.18], "plasticity": [-2.67, 5.53],
                        "stability": [-13.42, 10.78], "interference": [10.58, 7.93],
                        "forgetting": [2.83, 4.29]},
        "scoped": {"terminal_advantage": [14.08, 5.61], "plasticity": [14.08, 5.61],
                   "stability": [0.0, 0.0], "interference": [0.0, 0.0],
                   "forgetting": [0.0, 0.0]},
    },
}
# Scenarios measured after the suite was expanded. These cells are filled in
# by `python3 -m tests.sweep --refresh-benchmarks data/benchmarks.json`;
# until then they are absent and the UI reports "no reference band".
DEFAULT_BANDS.update({})

BAND_NOTES: Dict[str, str] = {
    "workflow": "Two BFCL families share one API catalogue; the correct workflow flips "
                "between blocks. This is the paper's retrieval-collision case.",
    "topic": "Three SQL databases in sequence. Same task, different schema.",
    "domain": "The environment itself changes: science lab, then tool use, then SQL.",
    "compound": "Topic, domain and workflow shifts in sequence over the longest horizon.",
    "healthcare": "Single-family triage stream. High variance: read the std before "
                  "trusting a single number.",
    "trading": "Single-family signal stream. High variance (ceiling effects): read the "
               "std before trusting a single number.",
}


def load_bands() -> Dict[str, Any]:
    """The shipped bands, preferring a sweep file if one has been generated."""
    if BENCHMARK_PATH.exists():
        try:
            data = json.loads(BENCHMARK_PATH.read_text())
            if data.get("bands"):
                return data
        except Exception:
            pass
    return {"bands": DEFAULT_BANDS, "notes": BAND_NOTES,
            "source": "built-in defaults (20-iteration sweep, block 20 / eval 30)"}


def band_for(scenario: str, evolver: str, metric: str) -> Optional[Dict[str, float]]:
    """The measured mean +/- std for one cell, or None if not measured."""
    data = load_bands()
    cell = (data.get("bands", {}).get(scenario) or {}).get(evolver)
    if not cell or metric not in cell:
        return None
    mean, std = cell[metric]
    return {"mean": mean, "std": std,
            "low": round(mean - std, 2), "high": round(mean + std, 2)}


def assess_run(scenario: str, evolver: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Judge one run against the measured bands.

    `verdict` is one of:
      expected    - inside the measured band for this scenario + evolver
      better      - above the band (usually a good sign, sometimes a bug)
      worse       - below the band
      unknown     - no band for this cell (e.g. a custom stream)
    """
    out: Dict[str, Any] = {"verdict": "unknown", "metrics": {}, "notes": []}
    for metric in ("terminal_advantage", "plasticity", "stability",
                   "interference", "forgetting"):
        b = band_for(scenario, evolver, metric)
        if b is None:
            continue
        # A metric the run does not report is not assessed. Treating a
        # missing metric as 0.0 would invent a regression.
        if metric not in metrics or metrics.get(metric) is None:
            continue
        value = float(metrics.get(metric, 0.0) or 0.0)
        if value < b["low"] or value > b["high"]:
            verdict = "worse" if value < b["low"] else "better"
        else:
            verdict = "expected"
        out["metrics"][metric] = {"value": round(value, 3), "band": b,
                                   "verdict": verdict}
        if verdict == "worse":
            out["notes"].append(
                f"{metric} {value:+.2f} is below the measured band "
                f"[{b['low']:+.2f}, {b['high']:+.2f}] for {evolver} on {scenario}")
    verdicts = {m["verdict"] for m in out["metrics"].values()}
    if "worse" in verdicts:
        out["verdict"] = "worse"
    elif verdicts == {"better"}:
        out["verdict"] = "better"
    elif verdicts and verdicts <= {"expected"}:
        out["verdict"] = "expected"
    return out


def reference_table(scenario: str) -> List[Dict[str, Any]]:
    """The full measured table for one scenario, for the UI's reference panel."""
    data = load_bands()
    bands = data.get("bands", {}).get(scenario) or {}
    rows = []
    for evolver, metrics in bands.items():
        rows.append({
            "evolver": evolver,
            "description": EVOLVER_GLOSSARY.get(evolver, ""),
            "metrics": {m: {"mean": v[0], "std": v[1],
                            "low": round(v[0] - v[1], 2),
                            "high": round(v[0] + v[1], 2)}
                        for m, v in metrics.items()},
        })
    rows.sort(key=lambda r: -r["metrics"].get("terminal_advantage", {}).get("mean", 0))
    return rows
