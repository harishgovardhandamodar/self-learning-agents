"""
Iteration sweep: run every scenario x evolver N times with varied seeds.

A single run is exactly reproducible, but one draw of tasks and policy coins
is still one draw. The sweep varies both the task sampling (train/eval
seeds) and the policy coins across iterations and reports mean ± std per
metric, plus how often each failure mode fires. That turns a qualitative
table into a claim with error bars.

    python3 -m tests.sweep --iterations 20 --block 20 --eval 30
    python3 -m tests.sweep --iterations 20 --scenarios workflow,topic --jobs 8

Each iteration i uses policy seed (policy-seed-base + i), task-train seed
(sample-seed-base + i) and task-eval seed (sample-seed-base + 1000 + i), so
the whole sweep is reproducible from the two bases.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import streams as streams_mod  # noqa: E402
from app.runner import RunConfig, execute  # noqa: E402
from app.evolvers import EVOLVERS  # noqa: E402

METRICS = ["terminal_advantage", "average_advantage", "plasticity",
           "stability", "interference", "forgetting"]
SHORT = {"terminal_advantage": "Term.", "average_advantage": "Dbar",
         "plasticity": "P", "stability": "S", "interference": "I",
         "forgetting": "F"}


async def one(scenario: str, eid: str, block: int, n_eval: int,
              run_seed: int, policy_seed: int, train_seed: int,
              eval_seed: int) -> dict:
    stream = streams_mod.build_stream(scenario, block_size=block, n_eval=n_eval,
                                      train_seed=train_seed, eval_seed=eval_seed)
    cfg = RunConfig(
        design={"spec": {"harness": {"system_prompt": "You are a careful assistant."}}},
        stream=stream,
        evolver=eid,
        evolver_cfg={"router_threshold": 0.40},
        policy_cfg={"mode": "simulated", "skill": 0.9, "seed": policy_seed},
        seed=run_seed,
    )
    t0 = time.time()
    res = await execute(cfg)
    m = res["metrics"]
    return {
        "scenario": scenario, "evolver": eid,
        "run_seed": run_seed, "policy_seed": policy_seed,
        "train_seed": train_seed, "eval_seed": eval_seed,
        "elapsed": round(time.time() - t0, 2),
        "metrics": {k: m.get(k, 0.0) for k in METRICS},
        "modes": sorted(f["id"] for f in res["failure_modes"]),
    }


def mean_std(xs: list) -> tuple:
    n = len(xs)
    if not n:
        return 0.0, 0.0
    mean = sum(xs) / n
    if n < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--eval", type=int, default=30)
    ap.add_argument("--scenarios", default="topic,workflow,domain,compound,healthcare,trading")
    ap.add_argument("--evolvers", default="all")
    ap.add_argument("--policy-seed-base", type=int, default=7)
    ap.add_argument("--sample-seed-base", type=int, default=100)
    ap.add_argument("--run-seed-base", type=int, default=1000,
                    help="base seed for lesson-id generation (bit-reproducibility)")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default="")
    ap.add_argument("--refresh-benchmarks", default="",
                    help="write the measured bands to this JSON file and use it "
                         "as the suite's reference")    args = ap.parse_args()

    scenarios = [s for s in args.scenarios.split(",") if s]
    evolvers = list(EVOLVERS) if args.evolvers == "all" else args.evolvers.split(",")
    sem = asyncio.Semaphore(max(1, args.jobs))

    async def bounded(sc: str, eid: str, i: int) -> dict:
        async with sem:
            return await one(sc, eid, args.block, args.eval,
                             args.run_seed_base + i,
                             args.policy_seed_base + i,
                             args.sample_seed_base + i,
                             args.sample_seed_base + 1000 + i)

    jobs = [bounded(sc, eid, i)
            for sc in scenarios for eid in evolvers for i in range(args.iterations)]
    print(f"sweep: {len(scenarios)} scenarios x {len(evolvers)} evolvers x "
          f"{args.iterations} iterations = {len(jobs)} runs", flush=True)
    t0 = time.time()
    rows = await asyncio.gather(*jobs)
    print(f"finished {len(rows)} runs in {time.time()-t0:.1f}s", flush=True)

    # ---------------------------------------------------------- aggregation
    agg = {}
    for sc in scenarios:
        for eid in evolvers:
            cell = [r for r in rows if r["scenario"] == sc and r["evolver"] == eid]
            stats = {}
            for k in METRICS:
                mean, std = mean_std([r["metrics"][k] for r in cell])
                stats[k] = {"mean": round(mean, 3), "std": round(std, 3),
                            "min": round(min(r["metrics"][k] for r in cell), 3),
                            "max": round(max(r["metrics"][k] for r in cell), 3)}
            mode_freq: dict = {}
            for r in cell:
                for mode in r["modes"]:
                    mode_freq[mode] = mode_freq.get(mode, 0) + 1
            agg[f"{sc}/{eid}"] = {"n": len(cell), "metrics": stats,
                                  "mode_frequency": {m: f"{c}/{len(cell)}"
                                                     for m, c in sorted(mode_freq.items())}}
            line = "  ".join(f"{SHORT[k]} {stats[k]['mean']:+7.2f}±{stats[k]['std']:.2f}"
                             for k in METRICS)
            print(f"{sc:<12} {eid:<14} {line}")

    if args.out:
        json.dump({"args": vars(args), "rows": rows, "aggregate": agg},
                  open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")

    if args.refresh_benchmarks:
        _write_benchmarks(agg, args)
        print(f"refreshed benchmark bands -> {args.refresh_benchmarks}")


BENCHMARK_METRICS = ["terminal_advantage", "average_advantage", "plasticity",
                     "stability", "interference", "forgetting"]


def _write_benchmarks(agg: Dict[str, Any], args) -> None:
    """Write the measured bands in the format app/benchmarks.py loads."""
    from app import benchmarks as bm

    bands: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key, cell in agg.items():
        scenario, evolver = key.rsplit("/", 1)
        bands.setdefault(scenario, {})[evolver] = {
            m: [cell["metrics"][m]["mean"], cell["metrics"][m]["std"]]
            for m in BENCHMARK_METRICS
        }
    payload = {
        "bands": bands,
        "notes": bm.BAND_NOTES,
        "source": (f"measured sweep: {args.iterations} iterations per cell, "
                   f"block {args.block}, eval {args.eval}"),
    }
    with open(args.refresh_benchmarks, "w") as fh:
        json.dump(payload, fh, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
