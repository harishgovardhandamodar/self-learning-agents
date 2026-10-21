"""
HTTP API for the Self-Learning Agent Lab.

Design Studio (designs, patterns, playbooks, guardrails) and the run bench
(streams x evolvers x policies) behind one JSON API, with the vanilla-JS
frontend served from /.

Runs execute in background asyncio tasks; progress lands in SQLite so the UI
can poll. Live mode needs fox-services on :8210 (or Ollama on :11434) -
/api/llm/health says which backends answer.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import benchmarks as benchmarks_mod
from . import db, designs as designs_mod, envs, flows as flows_mod
from . import metrics as metrics_mod
from . import scaffold as scaffold_mod
from . import stats as stats_mod
from . import streams as streams_mod
from . import suite as suite_mod
from .evolvers import EVOLVERS
from .llm import get_client
from .runner import RunConfig, execute

app = FastAPI(title="Self-Learning Agent Lab",
              description="Design, build and test self-learning agents.")
db.init_db()

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

_JOBS: Dict[str, asyncio.Task] = {}
_CANCEL: Dict[str, bool] = {}


# ------------------------------------------------------------------ models

class DesignIn(BaseModel):
    name: str = "Untitled agent"
    domain: str = "general"
    description: str = ""
    evolver: str = "scoped"
    stream: str = "workflow"
    policy: str = "simulated"
    patterns: List[str] = Field(default_factory=list)


class DesignUpdate(BaseModel):
    name: Optional[str] = None
    domain: Optional[str] = None
    description: Optional[str] = None
    spec: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class RunIn(BaseModel):
    design_id: Optional[str] = None
    name: str = ""
    stream: str = "workflow"
    evolver: str = "scoped"
    evolver_cfg: Dict[str, Any] = Field(default_factory=dict)
    policy: str = "simulated"
    policy_cfg: Dict[str, Any] = Field(default_factory=dict)
    block_size: int = 20
    n_eval: int = 30
    max_episodes: Optional[int] = None
    # Task-sampling seeds: vary these across repeated runs for an
    # iteration sweep; equal values reproduce the identical stream.
    train_seed: int = 1
    eval_seed: int = 2


class CompareIn(BaseModel):
    run_ids: List[str]


class BatchRunIn(BaseModel):
    """
    A comparison batch: the same stream, many evolvers, N iterations each.

    This is the lab's core workflow - reproduce the paper's Table 1 - and it
    used to require one manual launch per cell. `iterations` repeats each
    cell with varied task draws and policy coins, so the result carries error
    bars instead of a single noisy number.
    """
    design_id: Optional[str] = None
    name: str = ""
    stream: str = "workflow"
    evolvers: List[str] = Field(default_factory=lambda: ["seed", "reasoningbank",
                                                          "gepa", "metaharness", "scoped"])
    policy: str = "simulated"
    policy_cfg: Dict[str, Any] = Field(default_factory=dict)
    evolver_cfg: Dict[str, Any] = Field(default_factory=dict)
    block_size: int = 20
    n_eval: int = 30
    iterations: int = 1
    # Seed bases. Iteration i uses policy_seed_base+i, train_seed_base+i,
    # eval_seed_base+1000+i, run_seed_base+i.
    policy_seed_base: int = 7
    train_seed_base: int = 100
    eval_seed_base: int = 1100
    run_seed_base: int = 1000


class SuiteRunIn(BaseModel):
    """Launch the whole evaluation suite as one job."""
    scope: str = "core"                 # all | core | fast
    scenarios: List[str] = Field(default_factory=list)
    evolvers: List[str] = Field(default_factory=lambda: ["seed", "reasoningbank",
                                                          "gepa", "metaharness", "scoped"])
    iterations: int = 5
    block_size: int = 20
    n_eval: int = 30
    policy: str = "simulated"
    name: str = ""


class ExportIn(BaseModel):
    design_id: str
    run_id: Optional[str] = None


# ------------------------------------------------------------------ health

@app.get("/api/health")
async def health() -> Dict[str, Any]:
    envs.build_envs()
    return {"ok": True, "groups": len(envs.all_groups()),
            "evolvers": sorted(EVOLVERS),
            "scenarios": sorted(streams_mod.SCENARIOS),
            "patterns_available": designs_mod.pattern_catalog()["available"],
            "time": time.time()}


@app.get("/api/llm/health")
async def llm_health() -> Dict[str, Any]:
    async with get_client() as client:
        return await client.health()


# ------------------------------------------------------- patterns & guides

@app.get("/api/patterns")
async def patterns(domain: str = "", limit: int = 12) -> Dict[str, Any]:
    cat = designs_mod.pattern_catalog()
    picked = (designs_mod.patterns_for_domain(domain, limit) if domain
              else cat["patterns"][:limit])
    # Include each pattern's mermaid source so the list can offer a flow
    # without a second round trip.
    out = []
    for p in picked:
        q = dict(p)
        q["has_diagram"] = bool(p.get("diagram"))
        out.append(q)
    return {"patterns": out,
            "categories": cat["categories"],
            "source": cat["source"], "available": cat["available"]}


@app.get("/api/designs/{design_id}/flow")
async def design_flow(design_id: str) -> Dict[str, Any]:
    """
    The mermaid diagrams for one design: the agent flow generated from the
    design itself, and the stream it will be tested on.
    """
    d = db.get_design(design_id)
    if d is None:
        raise HTTPException(404, f"unknown design '{design_id}'")
    spec = d.get("spec") or {}
    stream_id = spec.get("stream", "workflow")
    scfg = spec.get("stream_cfg") or {}
    try:
        stream = streams_mod.build_stream(stream_id, block_size=int(scfg.get("block_size", 20)),
                                          n_eval=int(scfg.get("n_eval", 30)))
        groups = list(stream.groups)
    except KeyError:
        groups = []
    names = {}
    for g in groups:
        grp = envs.get_group(g)
        names[g] = grp.name if grp else g
    return {
        "design_id": design_id,
        "diagrams": [
            flows_mod.agent_flow_diagram(d, groups, names),
            flows_mod.stream_diagram(groups, names, int(scfg.get("block_size", 20))),
        ],
    }


@app.get("/api/patterns/{pattern_id}")
async def pattern_detail(pattern_id: str) -> Dict[str, Any]:
    for p in designs_mod.pattern_catalog()["patterns"]:
        if p.get("id") == pattern_id:
            out = dict(p)
            out["flow"] = flows_mod.pattern_diagram(p)
            return out
    raise HTTPException(404, f"unknown pattern '{pattern_id}'")


@app.get("/api/scenarios/catalog")
async def catalog_scenarios(domain: str = "") -> Dict[str, Any]:
    return {"scenarios": designs_mod.scenarios_for_domain(domain),
            "category_diagrams": flows_mod.category_diagrams(designs_mod.pattern_catalog())}


@app.get("/api/playbooks")
async def playbooks() -> Dict[str, Any]:
    return {"playbooks": list(designs_mod.PLAYBOOKS.values())}


@app.get("/api/playbooks/{playbook_id}")
async def playbook_detail(playbook_id: str) -> Dict[str, Any]:
    pb = designs_mod.playbook(playbook_id)
    if pb is None:
        raise HTTPException(404, f"unknown playbook '{playbook_id}'")
    return pb


@app.get("/api/guardrails")
async def guardrails(domain: str = "general") -> Dict[str, Any]:
    return {"guardrails": designs_mod.guardrails_for_domain(domain),
            "all": list(designs_mod.GUARDRAILS.values())}


@app.get("/api/streams")
async def streams() -> Dict[str, Any]:
    return {"streams": designs_mod.available_streams()}


@app.get("/api/evolvers")
async def evolvers() -> Dict[str, Any]:
    return {"evolvers": designs_mod.available_evolvers()}


# ----------------------------------------------------------------- designs

@app.get("/api/designs")
async def list_designs() -> Dict[str, Any]:
    return {"designs": db.list_designs()}


@app.post("/api/designs")
async def create_design(body: DesignIn) -> Dict[str, Any]:
    d = designs_mod.new_design(
        name=body.name, domain=body.domain, description=body.description,
        evolver=body.evolver, stream=body.stream, policy=body.policy,
        patterns=list(body.patterns))
    return db.save_design(d["id"], d["name"], d["domain"], d["description"],
                          d["spec"], d["tags"])


@app.post("/api/designs/from-playbook/{playbook_id}")
async def design_from_playbook(playbook_id: str, name: str = "") -> Dict[str, Any]:
    try:
        d = designs_mod.design_from_playbook(playbook_id, name or None)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return db.save_design(d["id"], d["name"], d["domain"], d["description"],
                          d["spec"], list(dict.fromkeys(
                              (d.get("tags") or []) + [f"playbook:{d.get('playbook')}"])) )


@app.get("/api/designs/{design_id}")
async def get_design(design_id: str) -> Dict[str, Any]:
    d = db.get_design(design_id)
    if d is None:
        raise HTTPException(404, f"unknown design '{design_id}'")
    return d


@app.put("/api/designs/{design_id}")
async def update_design(design_id: str, body: DesignUpdate) -> Dict[str, Any]:
    d = db.get_design(design_id)
    if d is None:
        raise HTTPException(404, f"unknown design '{design_id}'")
    patch = body.model_dump(exclude_unset=True)
    if "spec" in patch:
        d["spec"] = patch.pop("spec")
    d.update(patch)
    return db.save_design(d["id"], d["name"], d.get("domain", "general"),
                          d.get("description", ""), d["spec"], d.get("tags", []))


@app.delete("/api/designs/{design_id}")
async def delete_design(design_id: str) -> Dict[str, Any]:
    if not db.delete_design(design_id):
        raise HTTPException(404, f"unknown design '{design_id}'")
    return {"deleted": design_id}


# -------------------------------------------------------------------- runs

def _run_row(run_id: str) -> Dict[str, Any]:
    r = db.get_run(run_id)
    if r is None:
        raise HTTPException(404, f"unknown run '{run_id}'")
    return r


async def _execute_and_store(run_id: str, cfg: RunConfig) -> None:
    def on_progress(p: Dict[str, Any]) -> None:
        db.update_run(run_id, progress=float(p.get("progress", 0)),
                       message=str(p.get("message", "")))

    def cancelled() -> bool:
        return _CANCEL.get(run_id, False)

    db.update_run(run_id, status="running", progress=0.0, message="starting")
    try:
        if cfg.policy_mode == "live":
            async with get_client() as client:
                res = await execute(cfg, on_progress=on_progress,
                                    llm=client, should_cancel=cancelled)
        else:
            res = await execute(cfg, on_progress=on_progress,
                                should_cancel=cancelled)
        if res.get("cancelled"):
            db.update_run(run_id, status="cancelled", progress=cfg and 0.0,
                           message="cancelled by user")
            return
        for cp in res.get("checkpoints", []):
            db.add_checkpoint(run_id, cp["t"], cp.get("label", ""),
                              cp.get("kind", "eval"), cp.get("metrics", {}),
                              cp.get("diagnostics", {}), cp.get("snapshot", {}))
        for ev in res.get("train_log", [])[:400]:
            db.add_trace(run_id, ev.get("t", 0), ev.get("group", ""),
                         "episode", ev)
        import json as _json
        db.update_run(run_id, status="done", progress=1.0,
                       message=f"T={res['metrics'].get('T')} "
                               f"Term.={res['metrics'].get('terminal_advantage')}",
                       result=_json.dumps({
                           "metrics": res["metrics"],
                           "failure_modes": res.get("failure_modes", []),
                           "retrieval": res.get("retrieval", {}),
                           "attribution": res.get("attribution", {}),
                           "router": res.get("router", {}),
                           "guardrails": res.get("guardrails", {}),
                           "cost": res.get("cost", {}),
                           "efficiency": res.get("efficiency", {}),
                           "elapsed_s": res.get("elapsed_s"),
                       }))
    except asyncio.CancelledError:
        db.update_run(run_id, status="cancelled", message="cancelled by user")
        raise
    except Exception as e:  # keep the traceback in the record
        import traceback
        db.update_run(run_id, status="error", message=str(e)[:500],
                       error=traceback.format_exc()[-4000:])


def _prepare(body_stream: str, body_block: int, body_n_eval: int,
             body_train_seed: int, body_eval_seed: int) -> Any:
    try:
        return streams_mod.build_stream(body_stream, block_size=body_block,
                                        n_eval=body_n_eval,
                                        train_seed=body_train_seed,
                                        eval_seed=body_eval_seed)
    except KeyError as e:
        raise HTTPException(400, str(e))


def _spawn(run_id: str, cfg: RunConfig) -> None:
    """Start a background run and forget it when it finishes."""
    _CANCEL.pop(run_id, None)
    job = asyncio.create_task(_execute_and_store(run_id, cfg))

    def _forget(_t: asyncio.Task, _rid: str = run_id) -> None:
        _JOBS.pop(_rid, None)
        _CANCEL.pop(_rid, None)

    job.add_done_callback(_forget)
    _JOBS[run_id] = job


@app.post("/api/runs")
async def launch_run(body: RunIn) -> Dict[str, Any]:
    if body.evolver not in EVOLVERS:
        raise HTTPException(400, f"unknown evolver '{body.evolver}'")
    stream = _prepare(body.stream, body.block_size, body.n_eval,
                      body.train_seed, body.eval_seed)
    design_spec: Dict[str, Any] = {}
    if body.design_id:
        d = db.get_design(body.design_id)
        if d is None:
            raise HTTPException(404, f"unknown design '{body.design_id}'")
        design_spec = d["spec"]
    evolver_cfg = {"router_threshold": 0.40,
                   **designs_mod.EVOLVER_DEFAULTS.get(body.evolver, {}),
                   **(body.evolver_cfg or {})}
    policy_cfg = {"mode": body.policy,
                  **((design_spec.get("policy_cfg", {})) if isinstance(design_spec, dict) else {}),
                  **(body.policy_cfg or {})}
    if body.policy == "live":
        llm = await _llm_or_400()
        if llm is None:
            raise HTTPException(400, "no model backend reachable "
                                     "(fox-services :8210 or Ollama :11434)")
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    db.create_run(run_id, body.design_id or "", body.name, body.stream,
                  body.evolver, body.policy,
                  str(policy_cfg.get("model", "")), {
                      "evolver_cfg": evolver_cfg, "policy_cfg": policy_cfg,
                      "block_size": body.block_size, "n_eval": body.n_eval,
                      "train_seed": body.train_seed, "eval_seed": body.eval_seed,
                      "max_episodes": body.max_episodes,
                      "design_spec": design_spec,
                  })
    cfg = RunConfig(design={"spec": design_spec}, stream=stream,
                    evolver=body.evolver, evolver_cfg=evolver_cfg,
                    policy_cfg=policy_cfg, max_episodes=body.max_episodes,
                    seed=body.train_seed)
    _spawn(run_id, cfg)
    return _run_row(run_id)


@app.post("/api/runs/batch")
async def launch_batch(body: BatchRunIn) -> Dict[str, Any]:
    """
    Launch a comparison batch: every evolver x N iterations on one stream.

    The lab's headline result is a comparison, and doing it by hand meant one
    launch per cell. This endpoint creates the whole grid in one call, with
    seeds derived per cell so every run stays reproducible and every cell is
    independent.
    """
    unknown = [e for e in body.evolvers if e not in EVOLVERS]
    if unknown:
        raise HTTPException(400, f"unknown evolver(s): {', '.join(unknown)}")
    if not body.evolvers:
        raise HTTPException(400, "no evolvers selected")
    if body.iterations < 1 or body.iterations > 200:
        raise HTTPException(400, "iterations must be between 1 and 200")
    if body.policy == "live":
        if await _llm_or_400() is None:
            raise HTTPException(400, "no model backend reachable "
                                     "(fox-services :8210 or Ollama :11434)")
    design_spec: Dict[str, Any] = {}
    if body.design_id:
        d = db.get_design(body.design_id)
        if d is None:
            raise HTTPException(404, f"unknown design '{body.design_id}'")
        design_spec = d["spec"]

    created: List[Dict[str, Any]] = []
    for evolver in body.evolvers:
        evolver_cfg = {"router_threshold": 0.40,
                       **designs_mod.EVOLVER_DEFAULTS.get(evolver, {}),
                       **(body.evolver_cfg or {})}
        for i in range(body.iterations):
            train_seed = body.train_seed_base + i
            eval_seed = body.eval_seed_base + i
            policy_cfg = {"mode": body.policy,
                          "seed": body.policy_seed_base + i,
                          **((design_spec.get("policy_cfg", {}))
                             if isinstance(design_spec, dict) else {}),
                          **(body.policy_cfg or {})}
            stream = _prepare(body.stream, body.block_size, body.n_eval,
                              train_seed, eval_seed)
            run_id = f"run_{uuid.uuid4().hex[:10]}"
            label = body.name or f"{body.stream}/{evolver}"
            if body.iterations > 1:
                label += f" #{i + 1}"
            db.create_run(run_id, body.design_id or "", label, body.stream,
                          evolver, body.policy,
                          str(policy_cfg.get("model", "")), {
                              "evolver_cfg": evolver_cfg, "policy_cfg": policy_cfg,
                              "block_size": body.block_size, "n_eval": body.n_eval,
                              "train_seed": train_seed, "eval_seed": eval_seed,
                              "iteration": i, "batch_stream": body.stream,
                              "design_spec": design_spec,
                          })
            cfg = RunConfig(design={"spec": design_spec}, stream=stream,
                            evolver=evolver, evolver_cfg=evolver_cfg,
                            policy_cfg=policy_cfg,
                            seed=body.run_seed_base + i)
            _spawn(run_id, cfg)
            created.append({"run_id": run_id, "evolver": evolver, "iteration": i})
    return {"runs": created, "count": len(created),
            "stream": body.stream, "iterations": body.iterations}


async def _llm_or_400() -> Optional[Any]:
    try:
        async with get_client() as client:
            h = await client.health()
        if h.get("fox_services") or h.get("ollama"):
            return True
        return None
    except Exception:
        return None


@app.get("/api/runs")
async def list_runs(limit: int = 100) -> Dict[str, Any]:
    return {"runs": db.list_runs(limit=limit)}


@app.delete("/api/runs")
async def clear_runs(only_status: str = "") -> Dict[str, Any]:
    """
    Delete runs in bulk - the cleanup a user needs after experimenting.

    `only_status` limits the sweep (e.g. "cancelled" to clear the failures,
    "error" to clear crashed runs). With no filter every run goes.
    """
    rows = db.list_runs(limit=1000)
    removed = []
    for r in rows:
        if only_status and r.get("status") != only_status:
            continue
        db.delete_run(r["id"])
        removed.append(r["id"])
    return {"deleted": len(removed), "run_ids": removed}


@app.get("/api/coverage")
async def coverage() -> Dict[str, Any]:
    """
    What the evaluation suite exercises, and what it still misses.

    A lab that claims coverage should be able to show it: the drift types,
    domains and families every scenario touches, and the combinations that
    have no scenario at all.
    """
    envs.build_envs()
    scen = dict(streams_mod.SCENARIOS)
    for pid, groups in streams_mod.TOPIC_PERMUTATIONS.items():
        scen[pid] = {"name": f"Topic drift, ordering {pid[-1].upper()}",
                     "description": "The same three databases in a different order. "
                                    "A method that only wins on one ordering has not "
                                    "been shown to be better.",
                     "groups": groups, "drift": "topic"}
    return stats_mod.coverage_report(scen, envs.group_by_id())


@app.get("/api/stats/compare")
async def stats_compare(a: str = "", b: str = "", metric: str = "terminal_advantage") -> Dict[str, Any]:
    """
    Compare two cells with a significance test.

    `a` and `b` are cell keys: "stream/evolver" or the full
    "stream/evolver/b{block}e{n_eval}". A prefix resolves to the cell with
    that stream and evolver. The result carries the mean difference, a 95%
    confidence interval, an effect size and a p-value, so "better" is a
    claim with a number attached rather than a glance at two means.
    """
    cells = _cell_summary()

    def resolve(key: str) -> Optional[Dict[str, Any]]:
        if key in cells:
            return cells[key]
        matches = [c for k, c in cells.items()
                   if k == key or k.startswith(key + "/")]
        return matches[0] if len(matches) == 1 else None

    ca, cb = resolve(a), resolve(b)
    if not ca or not cb:
        raise HTTPException(404, f"unknown cell(s): {a!r} / {b!r}")
    return stats_mod.compare_cells(ca, cb, metric)


@app.get("/api/stats/recommend")
async def stats_recommend(stream: str = "", metric: str = "terminal_advantage") -> Dict[str, Any]:
    """
    Recommend an evolver for a stream, with the caveats stated.

    The recommendation is not simply the best mean: a method that wins the
    headline metric while leaking interference is a bad choice for an agent
    that has to keep working, so the guardrails are checked and named.
    """
    cells = [c for c in _cell_summary().values()
             if not stream or c.get("stream") == stream]
    if not cells:
        # No runs yet is a normal state, not an error: say so plainly and
        # list the streams that DO have data.
        available = sorted({c.get("stream") for c in _cell_summary().values()})
        return {"recommendation": None,
                "reason": f"no finished runs for stream {stream!r}",
                "available_streams": available,
                "ranking": []}
    return stats_mod.recommend(cells, metric)


def _cell_summary() -> Dict[str, Any]:
    """
    Aggregate finished runs into cells, keeping the raw values for tests.

    Runs are only comparable when they used the same configuration, so a
    cell is keyed by (stream, evolver) AND the harness geometry (block size,
    hold-out size). Mixing a 4-episode probe run with a 20-episode measured
    run in one cell produces a variance that is not the method's variance,
    and the comparison then says nothing.
    """
    rows = db.list_runs(limit=2000)
    cells: Dict[str, Dict[str, Any]] = {}
    metrics = ("terminal_advantage", "average_advantage", "plasticity", "stability",
               "interference", "forgetting", "forward_transfer", "backward_transfer",
               "area_under_curve")
    for r in rows:
        if r.get("status") != "done" or not r.get("result"):
            continue
        env = r["result"] if isinstance(r["result"], dict) else {}
        m = env.get("metrics", env) if isinstance(env, dict) else {}
        if not m or "terminal_advantage" not in m:
            continue
        cfg = r.get("config") or {}
        block = int(cfg.get("block_size", 20) or 20)
        n_eval = int(cfg.get("n_eval", 30) or 30)
        key = f"{r.get('stream_id')}/{r.get('evolver')}/b{block}e{n_eval}"
        cell = cells.setdefault(key, {
            "stream": r.get("stream_id"), "evolver": r.get("evolver"),
            "block_size": block, "n_eval": n_eval,
            "n": 0, "run_ids": [],
            "metrics": {k: {"values": [], "n": 0} for k in metrics},
            "guardrail_violations": 0,
            "cost_values": [],
            "efficiency_values": [],
        })
        cell["n"] += 1
        cell["run_ids"].append(r["id"])
        for k in metrics:
            v = m.get(k)
            if isinstance(v, (int, float)):
                cell["metrics"][k]["values"].append(float(v))
        gr = env.get("guardrails") or {}
        cell["guardrail_violations"] += int(gr.get("total", 0) or 0)
        c = env.get("cost") or {}
        if c:
            cell["cost_values"].append(c)
        e = env.get("efficiency") or {}
        if e:
            cell["efficiency_values"].append(e)
    # add the summary stats the UI wants
    for cell in cells.values():
        for k, d in cell["metrics"].items():
            xs = d["values"]
            d["n"] = len(xs)
            if xs:
                mean = sum(xs) / len(xs)
                var = (sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
                d["mean"] = round(mean, 3)
                d["std"] = round(var ** 0.5, 3)
            else:
                d["mean"] = 0.0
                d["std"] = 0.0
        # mean cost across the cell's runs
        cv = cell.pop("cost_values", [])
        ev = cell.pop("efficiency_values", [])
        if cv:
            cell["cost"] = {k: round(sum(float(c.get(k, 0) or 0) for c in cv) / len(cv), 3)
                             for k in ("prompt_units", "write_units", "total_units",
                                       "lessons_created", "lesson_hit_rate",
                                       "unused_lessons", "lessons_per_episode")}
        if ev:
            gains = [float(e.get("gain_pp", 0) or 0) for e in ev]
            ppu = [e.get("pp_per_100_units") for e in ev
                   if e.get("pp_per_100_units") is not None]
            cell["efficiency"] = {
                "gain_pp": round(sum(gains) / len(gains), 3) if gains else 0.0,
                "total_units": round(sum(float(e.get("total_units", 0) or 0)
                                         for e in ev) / len(ev), 3),
                "pp_per_100_units": round(sum(ppu) / len(ppu), 3) if ppu else None,
            }
        cell["guardrails"] = {"total": cell.pop("guardrail_violations", 0)}
    return cells


@app.get("/api/suite/scenarios")
async def suite_scenarios(scope: str = "all") -> Dict[str, Any]:
    """The scenarios a suite run covers, and what it will cost."""
    ids = suite_mod.suite_scenarios(scope)
    out = []
    for sid in ids:
        meta = streams_mod.SCENARIOS.get(sid, {})
        out.append({
            "id": sid, "name": meta.get("name", sid),
            "description": meta.get("description", ""),
            "drift": meta.get("drift", ""),
            "order": meta.get("order", "blocked"),
            "groups": meta.get("groups", []),
        })
    return {"scenarios": out, "scope": scope}


@app.post("/api/suite/estimate")
async def suite_estimate(body: SuiteRunIn) -> Dict[str, Any]:
    """What a suite run will cost, before it is launched."""
    ids = body.scenarios or suite_mod.suite_scenarios(body.scope)
    return suite_mod.estimate_cost(ids, body.evolvers, body.iterations,
                                   body.block_size, body.n_eval)


@app.post("/api/suite/run")
async def suite_run(body: SuiteRunIn) -> Dict[str, Any]:
    """
    Launch the whole suite as one job.

    The unit of work in this lab is a comparison, and comparing across the
    suite is 21 scenarios x 5 evolvers x N iterations. Doing that by hand
    was the workflow gap; this launches the whole grid in one call, with
    per-run progress.
    """
    ids = body.scenarios or suite_mod.suite_scenarios(body.scope)
    unknown = [e for e in body.evolvers if e not in EVOLVERS]
    if unknown:
        raise HTTPException(400, f"unknown evolver(s): {', '.join(unknown)}")
    if not ids:
        raise HTTPException(400, "no scenarios selected")
    if body.iterations < 1 or body.iterations > 100:
        raise HTTPException(400, "iterations must be between 1 and 100")
    if body.policy == "live" and await _llm_or_400() is None:
        raise HTTPException(400, "no model backend reachable "
                                 "(fox-services :8210 or Ollama :11434)")

    created = []
    for sid in ids:
        for evolver in body.evolvers:
            for i in range(body.iterations):
                stream = _prepare(sid, body.block_size, body.n_eval,
                                  100 + i, 1100 + i)
                run_id = f"run_{uuid.uuid4().hex[:10]}"
                label = body.name or f"{sid}/{evolver}"
                if body.iterations > 1:
                    label += f" #{i + 1}"
                db.create_run(run_id, "", label, sid, evolver, body.policy, "", {
                    "evolver_cfg": {"router_threshold": 0.40,
                                    **designs_mod.EVOLVER_DEFAULTS.get(evolver, {})},
                    "policy_cfg": {"mode": body.policy, "seed": 7 + i},
                    "block_size": body.block_size, "n_eval": body.n_eval,
                    "train_seed": 100 + i, "eval_seed": 1100 + i,
                    "iteration": i, "suite_scope": body.scope,
                })
                cfg = RunConfig(design={"spec": {}}, stream=stream,
                                evolver=evolver,
                                evolver_cfg={"router_threshold": 0.40,
                                             **designs_mod.EVOLVER_DEFAULTS.get(evolver, {})},
                                policy_cfg={"mode": body.policy, "seed": 7 + i},
                                seed=1000 + i)
                _spawn(run_id, cfg)
                created.append({"run_id": run_id, "stream": sid,
                                "evolver": evolver, "iteration": i})
    return {"runs": created, "count": len(created), "scenarios": ids,
            "iterations": body.iterations}


@app.get("/api/suite/report")
async def suite_report(stream: str = "", metric: str = "terminal_advantage",
                       alpha: float = 0.05) -> Dict[str, Any]:
    """
    The suite report: one row per cell, with the evidence for its rank.

    Mean and spread, the effect size against the best cell, a significance
    verdict that accounts for the number of comparisons, guardrail
    violations and the cost of what was bought.
    """
    cells = _cell_summary()
    rows = list(cells.values())
    if stream:
        rows = [c for c in rows if c.get("stream") == stream]
    if not rows:
        return {"rows": [], "streams": [], "cells": 0,
                "note": "no finished runs yet — launch a suite or a comparison first"}
    report = suite_mod.build_report(rows, metric, alpha)
    report["power"] = suite_mod.power_report(rows)
    return report


@app.get("/api/suite/advice")
async def suite_advice(domain: str = "general",
                       metric: str = "terminal_advantage") -> Dict[str, Any]:
    """
    What to do next for a given domain.

    Derived from the measured report rather than asserted: which evolver
    leads on this domain's scenarios, what the caveats are, and what the
    suite does not yet measure.
    """
    cells = list(_cell_summary().values())
    report = suite_mod.build_report(cells, metric) if cells else {"rows": []}
    return suite_mod.advise(domain, report)


@app.get("/api/runs/summary")
async def runs_summary(stream: str = "", evolver: str = "") -> Dict[str, Any]:
    """
    Aggregate finished runs into a comparison table with error bars.

    Groups by (stream, evolver) and reports mean +/- std across the runs in
    each cell, so a user sees "scoped on workflow: +14.1 +/- 5.6" rather
    than a list of unrelated single numbers.

    NOTE: this route must stay ABOVE `/api/runs/{run_id}` - FastAPI matches
    in declaration order, so a later declaration would shadow it and
    "summary" would be read as a run id.
    """
    rows = db.list_runs(limit=1000)
    cells: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r.get("status") != "done" or not r.get("result"):
            continue
        if stream and r.get("stream_id") != stream:
            continue
        if evolver and r.get("evolver") != evolver:
            continue
        env = r["result"] if isinstance(r["result"], dict) else {}
        metrics = env.get("metrics", env) if isinstance(env, dict) else {}
        if not metrics or "terminal_advantage" not in metrics:
            continue
        key = f"{r.get('stream_id')}/{r.get('evolver')}"
        cell = cells.setdefault(key, {
            "stream": r.get("stream_id"), "evolver": r.get("evolver"),
            "n": 0, "values": {k: [] for k in
                               ("terminal_advantage", "average_advantage",
                                "plasticity", "stability", "interference", "forgetting")},
            "run_ids": [],
        })
        cell["n"] += 1
        cell["run_ids"].append(r["id"])
        for k in cell["values"]:
            v = metrics.get(k)
            if isinstance(v, (int, float)):
                cell["values"][k].append(float(v))
    out = []
    for cell in cells.values():
        row = {"stream": cell["stream"], "evolver": cell["evolver"],
               "n": cell["n"], "run_ids": cell["run_ids"][-50:], "metrics": {}}
        for k, xs in cell["values"].items():
            if not xs:
                continue
            mean = sum(xs) / len(xs)
            std = (sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
            row["metrics"][k] = {"mean": round(mean, 3), "std": round(std, 3),
                                 "n": len(xs)}
        out.append(row)
    out.sort(key=lambda r: (-(r["metrics"].get("terminal_advantage", {}).get("mean", 0)),
                             r["stream"], r["evolver"]))
    return {"cells": out, "total_runs": len(rows)}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> Dict[str, Any]:
    r = _run_row(run_id)
    r["checkpoints"] = db.list_checkpoints(run_id)
    env = r["result"] if isinstance(r.get("result"), dict) else {}
    metrics = env.get("metrics", env) if isinstance(env, dict) else {}
    if metrics and "terminal_advantage" in metrics:
        r["assessment"] = benchmarks_mod.assess_run(r.get("stream_id", ""),
                                                     r.get("evolver", ""),
                                                     metrics)
    return r


@app.get("/api/runs/{run_id}/traces")
async def run_traces(run_id: str, limit: int = 500,
                     kind: str = "") -> Dict[str, Any]:
    _run_row(run_id)
    return {"traces": db.list_traces(run_id, limit=limit, kind=kind or None)}


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> Dict[str, Any]:
    _run_row(run_id)
    _CANCEL[run_id] = True
    job = _JOBS.get(run_id)
    if job and not job.done():
        job.cancel()
    return {"cancelled": run_id}


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str) -> Dict[str, Any]:
    """Remove a run and its checkpoints/traces.

    The lab accumulates cancelled and failed runs; without this the history
    list becomes a graveyard that hides the runs worth looking at.
    """
    _run_row(run_id)
    db.delete_run(run_id)
    return {"deleted": run_id}


@app.post("/api/runs/{run_id}/duplicate")
async def duplicate_run(run_id: str) -> Dict[str, Any]:
    """Re-launch a run with the same configuration but fresh task draws.

    The point of a duplicate is to see whether the result was the method or
    the draw: same evolver, same stream, new seeds.
    """
    src = _run_row(run_id)
    cfg_in = src.get("config") or {}
    body = RunIn(
        design_id=src.get("design_id") or None,
        name=f"{src.get('name', '')} (copy)".strip(),
        stream=src.get("stream_id", "workflow"),
        evolver=src.get("evolver", "seed"),
        evolver_cfg=cfg_in.get("evolver_cfg", {}),
        policy=src.get("policy", "simulated"),
        policy_cfg={k: v for k, v in (cfg_in.get("policy_cfg") or {}).items()
                    if k != "seed"},
        block_size=int(cfg_in.get("block_size", 20)),
        n_eval=int(cfg_in.get("n_eval", 30)),
        train_seed=int(cfg_in.get("train_seed", 1)) + 1,
        eval_seed=int(cfg_in.get("eval_seed", 2)) + 1,
    )
    return await launch_run(body)


@app.get("/api/benchmarks")
async def benchmarks(scenario: str = "") -> Dict[str, Any]:
    """
    The measured reference bands for every scenario x evolver cell.

    These come from the 20-iteration sweep and are what a run's numbers are
    judged against. Without them "Term. +12" is a number, not a result.
    """
    data = benchmarks_mod.load_bands()
    out: Dict[str, Any] = {
        "glossary": benchmarks_mod.METRIC_GLOSSARY,
        "evolvers": benchmarks_mod.EVOLVER_GLOSSARY,
        "notes": data.get("notes", {}),
        "source": data.get("source", ""),
        "scenarios": {},
    }
    for sc in (data.get("bands") or {}):
        if scenario and sc != scenario:
            continue
        out["scenarios"][sc] = {
            "rows": benchmarks_mod.reference_table(sc),
            "note": data.get("notes", {}).get(sc, ""),
        }
    return out


@app.post("/api/assess")
async def assess(body: CompareIn) -> Dict[str, Any]:
    """Judge one or more runs against the measured bands."""
    out = []
    for rid in body.run_ids:
        r = db.get_run(rid)
        if r is None:
            continue
        env = r["result"] if isinstance(r["result"], dict) else {}
        metrics = env.get("metrics", env) if isinstance(env, dict) else {}
        if not metrics:
            continue
        verdict = benchmarks_mod.assess_run(r.get("stream_id", ""),
                                            r.get("evolver", ""), metrics)
        out.append({"run_id": rid, "stream": r.get("stream_id"),
                    "evolver": r.get("evolver"), **verdict})
    return {"assessments": out}


@app.post("/api/compare")
async def compare(body: CompareIn) -> Dict[str, Any]:
    rows = []
    for rid in body.run_ids:
        r = db.get_run(rid)
        if r is None or not r.get("result"):
            continue
        # result is the stored envelope {metrics, failure_modes, ...};
        # accept a bare metrics blob for rows written before the envelope.
        env = r["result"] if isinstance(r["result"], dict) else {}
        metrics = env.get("metrics", env) if isinstance(env, dict) else {}
        modes = [f.get("id", f) if isinstance(f, dict) else f
                 for f in (env.get("failure_modes", []) if isinstance(env, dict) else [])]
        rows.append({"run_id": rid, "name": r.get("name", ""),
                     "evolver": r.get("evolver", ""), "policy": r.get("policy", ""),
                     "stream": r.get("stream_id", ""), "metrics": metrics,
                     "failure_modes": modes})
    out = metrics_mod.compare_runs(rows)
    for row, src in zip(out, rows):
        row["failure_modes"] = src["failure_modes"]
    return {"rows": out}


@app.post("/api/export")
async def export(body: ExportIn) -> Dict[str, Any]:
    d = db.get_design(body.design_id)
    if d is None:
        raise HTTPException(404, f"unknown design '{body.design_id}'")
    metrics = None
    if body.run_id:
        r = db.get_run(body.run_id)
        if r is None:
            raise HTTPException(404, f"unknown run '{body.run_id}'")
        metrics = r.get("result")
    try:
        manifest = scaffold_mod.export_design(d, run_metrics=metrics)
    except Exception as e:
        raise HTTPException(500, f"export failed: {e}")
    return manifest


# ------------------------------------------------------------------ static

@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
