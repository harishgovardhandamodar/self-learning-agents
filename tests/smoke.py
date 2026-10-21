"""
Smoke test: run every evolver on every drift scenario with the simulated policy.

    python3 -m tests.smoke [--scenario workflow] [--block 20] [--eval 12]

Prints the paper's metric table (Term., Dbar, P, S) per evolver so the
qualitative behaviour of each method can be checked at a glance.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import streams as streams_mod  # noqa: E402
from app.runner import RunConfig, execute  # noqa: E402
from app.evolvers import EVOLVERS  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="workflow")
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--eval", type=int, default=12)
    ap.add_argument("--evolver", default="all")
    args = ap.parse_args()

    stream = streams_mod.build_stream(args.scenario, block_size=args.block, n_eval=args.eval)
    evolvers = list(EVOLVERS) if args.evolver == "all" else [args.evolver]

    print(f"\n=== scenario: {stream.name}  T={stream.T} K={len(stream.groups)} "
          f"block={args.block} eval={args.eval} ===")
    print(f"{'evolver':<16} {'Term.':>8} {'Dbar':>8} {'P':>8} {'S':>8} "
          f"{'I':>7} {'F':>7}  failure modes")
    print("-" * 100)

    for eid in evolvers:
        cfg = RunConfig(
            design={"spec": {"harness": {"system_prompt": "You are a careful assistant."}}},
            stream=stream,
            evolver=eid,
            evolver_cfg={"router_threshold": 0.40},
            policy_cfg={"mode": "simulated", "skill": 0.9, "seed": 7},
        )
        res = await execute(cfg)
        if res.get("cancelled"):
            print(f"{eid:<16} cancelled")
            continue
        m = res["metrics"]
        modes = ",".join(f["id"] for f in res["failure_modes"] if f["id"] != "no-failure-observed")
        print(f"{eid:<16} {m['terminal_advantage']:>8.2f} {m['average_advantage']:>8.2f} "
              f"{m['plasticity']:>8.2f} {m['stability']:>8.2f} {m['interference']:>7.2f} "
              f"{m['forgetting']:>7.2f}  {modes or '-'}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
