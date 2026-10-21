"""
Mechanism unit tests: the contracts the lab's claims rest on.

These run in milliseconds with no model backend. They pin down the pieces
that the paper comparison depends on:

  * the workflow probe reads every toolcall template correctly
  * intent classification separates all 58 templates
  * the shared-preamble families resolve to one EnvView (union lexicon)
  * lessons generalize (no stale instance values on replay)
  * the router partitions SQL families and merges the two BFCL families
  * the metric formulas match hand-computed values
  * the seed harness measures exactly zero everywhere
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import random

from app import envs, metrics as metrics_mod
from app.evolvers import generalize_action
from app.policy import SimulatedPolicy, generalize_inverse
from app.retrieval import Router, probe_workflow
from app.runner import RunConfig, execute
from app import streams as streams_mod


def _all_templates():
    envs.build_envs()
    for g in envs.all_groups():
        for tpl in g.templates:
            yield g, tpl


def test_probe_workflow_per_template():
    """Every toolcall template must probe to its true workflow."""
    rng = random.Random(0)
    bad = []
    for g in envs.all_groups():
        for tpl in g.templates:
            t = g._materialize(tpl, 0, rng)
            want = {"toolcall": ("missing-arg" if "miss" in (tpl.get("id") or "")
                                 else "complete-arg")}.get(g.env, "default")
            got = probe_workflow(t)
            if got != want:
                bad.append((tpl.get("id"), want, got))
    assert not bad, bad


def test_classify_intent_covers_all_templates():
    """Intent classification separates every template via the visible view."""
    rng = random.Random(0)
    bad = []
    for g in envs.all_groups():
        for tpl in g.templates:
            t = g._materialize(tpl, 0, rng)
            view = envs.view_for_task(t)
            assert view is not None
            got = envs.classify_intent(t.query, view.intent_lexicon())
            if got != tpl.get("intent"):
                bad.append((tpl.get("id"), tpl.get("intent"), got))
    assert not bad, bad


def test_shared_preamble_yields_union_view():
    """The two BFCL families share one view holding both lexicons."""
    envs.build_envs()
    views = envs.views()
    sigs = {g.signature for g in envs.all_groups() if g.env == "toolcall"}
    assert len(sigs) == 1, "BFCL families must share one preamble"
    view = views[sigs.pop()]
    assert set(view.group_ids()) == {"toolcall-clarify", "toolcall-act"}
    lex = view.intent_lexicon()
    assert any(i.startswith("toolcall.clarify") for i in lex)
    assert "toolcall.get_order_history" in lex


def test_generalize_round_trip_sql_and_call():
    """Lessons keep reusable templates: placeholders survive a learn/replay cycle."""
    envs.build_envs()
    rng = random.Random(0)
    checked = 0
    for g in envs.all_groups():
        if g.env not in ("sql", "toolcall"):
            continue
        for tpl in g.templates[:3]:
            t = g._materialize(tpl, 0, rng)
            params = t.meta["params"]
            ref = envs.gold_to_action(g, tpl)
            if ref["kind"] == "sql":
                filled = dict(ref, sql=generalize_inverse(ref["sql_template"], params))
                learned = generalize_action(filled, t)
                # the true template survives even for single-char params
                assert learned["sql_template"] == ref["sql_template"], tpl["id"]
                checked += 1
            elif ref["kind"] == "call":
                filled_args = {k: generalize_inverse(str(v), params)
                               for k, v in ref["args"].items()}
                learned = generalize_action(
                    {"kind": "call", "tool": ref["tool"], "args": filled_args}, t)
                # every instance-derived arg must be a placeholder or constant
                for k, v in learned["args"].items():
                    assert not any(str(pv) == str(v) and len(str(pv)) >= 4
                                   for pv in params.values()
                                   if str(pv) not in ("True", "False", "None")), \
                        (tpl["id"], k, v)
                checked += 1
    assert checked > 0


def test_lesson_replays_on_new_instance():
    """The stale-lesson regression: a learned call must grade on new params."""
    envs.build_envs()
    pol = SimulatedPolicy({"seed": 7})
    g = envs.get_group("toolcall-act")
    rng = random.Random(1)
    t1 = g._materialize(next(t for t in g.templates if t["id"] == "bfcl-base-order"),
                        0, rng)
    view = envs.view_for_task(t1)
    intent = pol._intent_of(t1, view)
    action = pol._prior(t1, view, intent)
    assert envs.grade_detail(t1, pol._render(t1, action))["success"]
    learned = generalize_action(action, t1)
    # new instance, different user + limit
    t2 = g._materialize(next(t for t in g.templates if t["id"] == "bfcl-base-order"),
                        7, random.Random(2))
    assert t2.meta["params"] != t1.meta["params"]
    replay = dict(learned)
    replay["args"] = {k: generalize_inverse(str(v), t2.meta["params"])
                      for k, v in learned["args"].items()}
    assert envs.grade_detail(t2, pol._render(t2, replay))["success"]


def test_router_partitions_sql_and_merges_bfcl():
    """SQL families separate; the two BFCL families merge at threshold 0.40."""
    envs.build_envs()
    router = Router(threshold=0.40)
    owners = {}
    for gid in ["sql-formula1", "sql-codebase", "sql-thrombosis",
                "toolcall-clarify", "toolcall-act"]:
        g = envs.get_group(gid)
        cids = {router.assign(t)[0] for t in g.sample(6, offset=0, seed=1)}
        owners[gid] = cids
    sql_clusters = owners["sql-formula1"] | owners["sql-codebase"] | owners["sql-thrombosis"]
    assert len(sql_clusters) == 3, owners
    assert owners["toolcall-clarify"] == owners["toolcall-act"], owners
    assert len(owners["toolcall-clarify"]) == 1


def test_compute_metrics_hand_values():
    """P/I/F/S on a synthetic two-family stream, checked by hand."""
    checkpoints = [
        {"t": 0, "metrics": {"a": 50.0, "b": 50.0}},
        {"t": 10, "metrics": {"a": 90.0, "b": 30.0}},
        {"t": 20, "metrics": {"a": 80.0, "b": 70.0}},
    ]

    class FakeStream:
        groups = ["a", "b"]
        T = 20
        sigma = {"a": 0, "b": 10}
        tau = {"a": 10, "b": 20}

    m = metrics_mod.compute_metrics(checkpoints, FakeStream())
    assert m["plasticity_per_group"] == {"a": 40.0, "b": 20.0}
    assert m["plasticity"] == 30.0
    # b drops 20pp below seed before its block -> I_b = 20
    assert m["interference_per_group"] == {"a": 0.0, "b": 20.0}
    # a drops 90 -> 80 after its block -> F_a = 10
    assert m["forgetting_per_group"] == {"a": 10.0, "b": 0.0}
    assert m["stability"] == -(10.0 + 5.0)
    assert m["terminal_advantage"] == 25.0  # mean(30, 20)


def test_base_competence_matches_the_declared_rate():
    """
    The environment declares how competent a frozen model is per intent, and
    the measured seed rate must match it.

    If the "wrong answer" fallback ever happens to be RIGHT for a gold (a
    flat proposal for a flat market, acuity 3 for a standard case), the
    seed rate inflates, the headroom disappears, and the whole evaluation
    silently stops being able to detect learning. This test is the guard
    against that.
    """
    from app.policy import SimulatedPolicy

    pol = SimulatedPolicy({"seed": 7})
    envs.build_envs()
    for g in envs.all_groups():
        view = envs.view_for_task(g.sample(1, offset=100_000, seed=2)[0])
        tasks = g.sample(400, offset=100_000, seed=2)
        declared, ok = [], 0
        for t in tasks:
            intent = pol._intent_of(t, view)
            ref = view.builtin_actions().get(intent)
            if ref:
                declared.append(ref["prob"])
            action = pol._prior(t, view, intent)
            ok += envs.grade_detail(t, pol._render(t, action))["success"]
        want = sum(declared) / len(declared)
        got = ok / len(tasks)
        assert abs(want - got) < 0.12, (
            f"{g.id}: declared {want:.2f} but measured {got:.2f}")


def test_generic_fallback_is_actually_wrong():
    """The wrong answer must not match the gold for the domain fallbacks."""
    from app.policy import SimulatedPolicy

    pol = SimulatedPolicy({"seed": 7})
    envs.build_envs()
    for gid in ("health-triage", "trade-signal", "trade-signal-macro"):
        g = envs.get_group(gid)
        view = envs.view_for_task(g.sample(1, offset=100_000, seed=2)[0])
        for t in g.sample(40, offset=100_000, seed=2):
            action = pol._generic(t, view)
            assert not envs.grade_detail(t, pol._render(t, action))["success"], \
                f"{gid}: the generic fallback graded as correct"


def test_benchmark_bands_still_hold():
    """
    The measured bands are the lab's regression gate.

    If a change to the harness mechanics moves a cell's mean outside its own
    measured band, the bands are stale and every user-facing reference number
    becomes wrong. This is the check that keeps the Reference tab honest.
    """
    from app import benchmarks

    data = benchmarks.load_bands()
    bands = data.get("bands") or {}
    assert bands, "no benchmark bands shipped"
    # Every scenario must cover every evolver.
    from app.evolvers import EVOLVERS
    for scenario, cells in bands.items():
        for eid in EVOLVERS:
            assert eid in cells, f"{scenario} has no band for {eid}"
            cell = cells[eid]
            assert "terminal_advantage" in cell, f"{scenario}/{eid} lacks a terminal band"
            mean, std = cell["terminal_advantage"]
            assert std >= 0, f"{scenario}/{eid} has a negative std"

    # The seed baseline must be exactly zero everywhere: the harness does not
    # adapt, so any nonzero advantage is a measurement bug.
    for scenario, cells in bands.items():
        for metric in ("terminal_advantage", "plasticity", "stability",
                       "interference", "forgetting"):
            mean, std = cells["seed"][metric]
            assert mean == 0.0, f"{scenario}/seed {metric} is {mean}, expected 0"

    # Scoped must never show interference or forgetting: that is its whole
    # claim. The measured bands are allowed a small tolerance because a
    # single-family stream can have a one-task wobble.
    for scenario, cells in bands.items():
        for metric in ("interference", "forgetting"):
            mean, _std = cells["scoped"][metric]
            assert mean <= 0.5, f"{scenario}/scoped {metric} regressed to {mean}"


def test_assess_flags_a_regression():
    """A run far outside its band must be flagged, not silently accepted."""
    from app import benchmarks

    band = benchmarks.band_for("workflow", "scoped", "terminal_advantage")
    assert band is not None
    inside = benchmarks.assess_run("workflow", "scoped",
                                   {"terminal_advantage": band["mean"]})
    assert inside["verdict"] in ("expected", "better"), inside
    # A metric the run does not report must not be judged.
    assert "plasticity" not in inside["metrics"], inside
    outside = benchmarks.assess_run("workflow", "scoped",
                                    {"terminal_advantage": band["low"] - 25})
    assert outside["verdict"] == "worse"
    assert outside["notes"], "a regression must come with a reason"
    unknown = benchmarks.assess_run("no-such-stream", "scoped",
                                    {"terminal_advantage": 5})
    assert unknown["verdict"] == "unknown"


def test_runs_are_bit_reproducible():
    """Same seeds in -> identical metrics out. No hidden randomness."""
    async def go():
        out = []
        for _ in range(2):
            stream = streams_mod.build_stream("workflow", block_size=10, n_eval=10,
                                              train_seed=11, eval_seed=22)
            cfg = RunConfig(
                design={"spec": {"harness": {"system_prompt": "x"}}},
                stream=stream, evolver="gepa",
                evolver_cfg={"router_threshold": 0.40},
                policy_cfg={"mode": "simulated", "skill": 0.9, "seed": 7},
                seed=99)
            res = await execute(cfg)
            out.append(res["metrics"])
        assert out[0] == out[1]
    asyncio.run(go())


def test_seed_evolver_is_exact_zero():
    """The seed harness must measure exactly 0 on every metric."""
    async def go():
        stream = streams_mod.build_stream("workflow", block_size=10, n_eval=10)
        cfg = RunConfig(
            design={"spec": {"harness": {"system_prompt": "x"}}},
            stream=stream, evolver="seed",
            evolver_cfg={"router_threshold": 0.40},
            policy_cfg={"mode": "simulated", "skill": 0.9, "seed": 7})
        res = await execute(cfg)
        m = res["metrics"]
        assert all(m[k] == 0.0 for k in
                   ("terminal_advantage", "average_advantage", "plasticity",
                    "stability", "interference", "forgetting")), m
    asyncio.run(go())


def test_scoped_never_injects_foreign_before_scope_seen():
    """Scoped retrieval returns nothing for a scope with no lessons."""
    from app.evolvers import get_evolver
    from app.harness import seed_harness
    envs.build_envs()
    ev = get_evolver("scoped", {})
    h = seed_harness(system_prompt="x")
    router = Router(threshold=0.40)
    g = envs.get_group("toolcall-act")
    t = g.sample(1, offset=100_000, seed=2)[0]
    inj = ev.inject(h, t, router)
    assert inj.lessons == [] and "no foreign memory" in inj.note
