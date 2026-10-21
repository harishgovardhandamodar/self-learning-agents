"""
Scaffold: export a design as a runnable agent project.

The export is a small directory with everything needed to run the agent
outside the lab: the harness JSON the run validated, a runner script that
replays the stream protocol (train -> checkpoint -> evaluate), and the
guardrail checklist the design signed up to. It is a starting point, not a
deployment: the policy still needs a model backend (fox-services or Ollama).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

EXPORT_ROOT = Path(__file__).resolve().parent.parent / "exports"


RUNNER_TEMPLATE = '''"""
Exported agent runner for design "{name}".

Replays the lab protocol outside the lab: load the harness, run the stream,
evolve, checkpoint, evaluate. Requires the lab package on sys.path and a
model backend for live mode (fox-services on :8210 or Ollama on :11434).

    python3 run_agent.py --evolver {evolver} --stream {stream} --mode simulated
    python3 run_agent.py --evolver {evolver} --stream {stream} --mode live
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lab_app"))

from app import streams as streams_mod
from app.runner import RunConfig, execute


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evolver", default="{evolver}")
    ap.add_argument("--stream", default="{stream}")
    ap.add_argument("--mode", default="simulated", choices=["simulated", "live"])
    ap.add_argument("--block", type=int, default={block})
    ap.add_argument("--eval", type=int, default={n_eval})
    args = ap.parse_args()

    design = json.loads(Path("design.json").read_text())
    stream = streams_mod.build_stream(args.stream, block_size=args.block,
                                      n_eval=args.eval)
    cfg = RunConfig(
        design=design,
        stream=stream,
        evolver=args.evolver,
        evolver_cfg=(design.get("spec") or {{}}).get("evolver_cfg", {{}}),
        policy_cfg={{**((design.get("spec") or {{}}).get("policy_cfg", {{}})),
                    "mode": args.mode}},
    )
    res = await execute(cfg)
    print(json.dumps(res["metrics"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
'''

GUARDRAIL_TEMPLATE = """# Guardrail checklist for {name}

Playbook: {playbook}
Domain: {domain}

The lab validated this harness with the guardrails below active. Removing
one invalidates the measurement - re-run the stream before trusting the
agent without it.

{items}
"""


def export_design(design: Dict[str, Any], run_metrics: Dict[str, Any] | None = None,
                  dest: str | Path | None = None) -> Dict[str, Any]:
    """Write the export directory and return its manifest."""
    from . import designs as designs_mod

    spec = design.get("spec") or {}
    name = design.get("name", "agent")
    slug = "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-") or "agent"
    out = Path(dest) if dest else (EXPORT_ROOT / slug)
    if out.exists():
        shutil.rmtree(out)
    (out / "lab_app").mkdir(parents=True)

    # Copy the lab package so the export is self-contained.
    src_app = Path(__file__).resolve().parent
    for f in src_app.glob("*.py"):
        shutil.copy(f, out / "lab_app" / f.name)

    (out / "design.json").write_text(json.dumps(design, indent=2))
    stream = spec.get("stream", "workflow")
    scfg = spec.get("stream_cfg", {}) or {}
    (out / "run_agent.py").write_text(RUNNER_TEMPLATE.format(
        name=name.replace('"', "'"),
        evolver=spec.get("evolver", "scoped"),
        stream=stream,
        block=int(scfg.get("block_size", 20)),
        n_eval=int(scfg.get("n_eval", 30)),
    ))
    guard_ids = spec.get("guardrails", [])
    items = "\n".join(
        f"- [ ] **{designs_mod.GUARDRAILS[g]['title']}**: {designs_mod.GUARDRAILS[g]['rule']}"
        for g in guard_ids if g in designs_mod.GUARDRAILS) or "- (no guardrails recorded)"
    (out / "GUARDRAILS.md").write_text(GUARDRAIL_TEMPLATE.format(
        name=name, playbook=design.get("playbook", "-"),
        domain=design.get("domain", "general"), items=items))
    if run_metrics:
        (out / "validation.json").write_text(json.dumps(run_metrics, indent=2))
    return {"directory": str(out), "files": sorted(p.name for p in out.iterdir())}
