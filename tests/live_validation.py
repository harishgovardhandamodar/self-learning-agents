"""
Live validation: does the simulated suite's premise hold against a real model?

Every number in the suite comes from the simulated policy. That is a
deliberate trade - a real model makes the suite too slow and too noisy to
iterate on - but it means the suite's ranking is unconfirmed. This script
closes that gap as far as is practical:

  1. run a tiny stream in LIVE mode against the local model
  2. run the same stream in SIMULATED mode
  3. report the seed baseline and the learned gain from each, side by side

It does not try to reproduce the whole suite live: at ~13s per call a
full suite would take days. It answers the question that actually matters
- does the simulated policy's behaviour resemble the real one's closely
enough that conclusions drawn on it transfer?

    python3 -m tests.live_validation [--block 4] [--eval 4]

Skips with a clear message when no model backend is reachable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import envs  # noqa: E402
from app import streams as streams_mod  # noqa: E402
from app.llm import get_client  # noqa: E402
from app.runner import RunConfig, execute  # noqa: E402


async def _health() -> dict:
    async with get_client() as c:
        return await c.health()


async def run_one(mode: str, scenario: str, evolver: str, block: int, n_eval: int,
                  llm=None) -> dict:
    stream = streams_mod.build_stream(scenario, block_size=block, n_eval=n_eval)
    cfg = RunConfig(
        design={"spec": {"harness": {"system_prompt":
                                     "You are a careful, tool-using assistant."}}},
        stream=stream,
        evolver=evolver,
        evolver_cfg={"router_threshold": 0.40},
        policy_cfg={"mode": mode, "model": "fox-managed", "max_steps": 2,
                    "temperature": 0.0},
    )
    t0 = time.time()
    res = await execute(cfg, llm=llm)
    m = res["metrics"]
    return {
        "mode": mode, "evolver": evolver, "scenario": scenario,
        "elapsed_s": round(time.time() - t0, 1),
        "baseline": m.get("baseline", {}),
        "terminal_advantage": m.get("terminal_advantage"),
        "plasticity": m.get("plasticity"),
        "accuracy": m.get("accuracy", {}),
        "guardrails": res.get("guardrails", {}),
        "cost": res.get("cost", {}),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="trading")
    ap.add_argument("--evolver", default="seed")
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--eval", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    h = await _health()
    if not h.get("reachable"):
        print("SKIP: no model backend reachable (fox-services :8210 or Ollama :11434)")
        return 0
    print(f"backend: {'fox-services' if h.get('fox_services') else 'ollama'} "
          f"({len(h.get('models', []))} models)")

    results = []
    # Simulated first: it is the reference the whole suite is built on.
    sim = await run_one("simulated", args.scenario, args.evolver, args.block, args.eval)
    results.append(sim)
    print(f"\nsimulated: baseline={sim['baseline']} "
          f"Term.={sim['terminal_advantage']} ({sim['elapsed_s']}s)")

    live = None
    try:
        async with get_client() as client:
            live = await run_one("live", args.scenario, args.evolver,
                                 args.block, args.eval, llm=client)
    except Exception as e:
        print(f"live run failed: {e}")
    if live:
        results.append(live)
        print(f"live:      baseline={live['baseline']} "
              f"Term.={live['terminal_advantage']} ({live['elapsed_s']}s)")

    if live:
        print("\n=== agreement ===")
        for g, s in sim["baseline"].items():
            l = live["baseline"].get(g)
            if l is None:
                continue
            print(f"  {g:<24} simulated {s:5.1f}%   live {l:5.1f}%   "
                  f"delta {l - s:+5.1f}pp")
        print("\nThe suite's numbers are simulator-relative. This table is the")
        print("evidence for how far they transfer: a live baseline within a")
        print("few points of the simulated one means conclusions drawn on the")
        print("simulator are about the harness, not about the simulator.")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
