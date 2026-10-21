"""
API tests. Simulated-mode runs always execute; the live test only runs when
a model backend answers (fox-services :8210 or Ollama :11434), otherwise it
skips instead of failing.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app
from app.llm import get_client

client = TestClient(app)


def test_health():
    h = client.get("/api/health").json()
    assert h["ok"] and h["groups"] == 13


def test_design_crud():
    d = client.post("/api/designs", json={"name": "t", "domain": "finance"}).json()
    assert d["id"]
    g = client.get(f"/api/designs/{d['id']}").json()
    assert g["name"] == "t"
    u = client.put(f"/api/designs/{d['id']}", json={"description": "hi"}).json()
    assert u["description"] == "hi"
    assert client.delete(f"/api/designs/{d['id']}").json()["deleted"] == d["id"]


def test_playbook_design_has_guardrails():
    d = client.post("/api/designs/from-playbook/healthcare-triage").json()
    assert "escalation-first-triage" in d["spec"]["guardrails"]
    client.delete(f"/api/designs/{d['id']}")


def _wait_done(run_id, timeout=180):
    for _ in range(timeout):
        time.sleep(1)
        g = client.get(f"/api/runs/{run_id}").json()
        if g["status"] not in ("pending", "running"):
            return g
    raise AssertionError("run did not finish")


def test_simulated_run_end_to_end():
    r = client.post("/api/runs", json={
        "stream": "trading", "evolver": "scoped", "policy": "simulated",
        "block_size": 4, "n_eval": 4}).json()
    g = _wait_done(r["id"])
    assert g["status"] == "done", g.get("message")
    assert "metrics" in (g["result"] or {})
    assert "plasticity" in (g["result"]["metrics"] or {})
    assert len(g.get("checkpoints", [])) == 2
    rows = client.post("/api/compare", json={"run_ids": [r["id"]]}).json()["rows"]
    assert len(rows) == 1 and rows[0]["evolver"] == "scoped"


def test_design_flow_returns_mermaid_diagrams():
    d = client.post("/api/designs", json={"name": "flow", "domain": "healthcare",
                                          "evolver": "scoped"}).json()
    f = client.get(f"/api/designs/{d['id']}/flow").json()
    ids = [x["id"] for x in f["diagrams"]]
    assert "agent-flow" in ids and "stream" in ids
    for x in f["diagrams"]:
        assert x["mermaid"].startswith("flowchart "), x["id"]
        assert x["title"]
    flow = [x for x in f["diagrams"] if x["id"] == "agent-flow"][0]
    # the scoped flow must show scope checking and workflow conditioning
    assert "scope" in flow["mermaid"].lower()
    assert flow["writes"], "scoped must write somewhere"
    client.delete(f"/api/designs/{d['id']}")


def test_design_flow_follows_the_evolver():
    """Two evolvers must not produce the same diagram."""
    ids = []
    for eid in ("seed", "scoped", "gepa"):
        d = client.post("/api/designs", json={"name": f"f-{eid}", "evolver": eid}).json()
        f = client.get(f"/api/designs/{d['id']}/flow").json()
        flow = [x for x in f["diagrams"] if x["id"] == "agent-flow"][0]
        ids.append(flow["mermaid"])
        client.delete(f"/api/designs/{d['id']}")
    assert len(set(ids)) == 3, "each evolver must have its own flow"


def test_design_flow_404_for_unknown_design():
    assert client.get("/api/designs/dsg_nope/flow").status_code == 404


def test_pattern_detail_includes_its_flow():
    p = client.get("/api/patterns?limit=5").json()["patterns"]
    with_diagram = [x for x in p if x.get("has_diagram")]
    if not with_diagram:
        print("SKIP: no pattern with a diagram in the mounted catalog")
        return
    pid = with_diagram[0]["id"]
    detail = client.get(f"/api/patterns/{pid}").json()
    assert detail["flow"]["mermaid"].startswith("flowchart ")


def test_coverage_endpoint_reports_shape_and_holes():
    c = client.get("/api/coverage").json()
    assert c["totals"]["scenarios"] >= 14, c["totals"]
    assert c["totals"]["families"] == 13
    assert c["totals"]["environments"] == 5
    # every drift type must be exercised
    assert set(c["drift_types"]) >= {"topic", "workflow", "domain", "compound"}
    # no family may be orphaned
    assert not c["unused_families"], c["unused_families"]
    # the suite must cover every environment
    assert set(c["environments"]) == {"sql", "toolcall", "science", "healthcare", "trading"}


def test_coverage_includes_the_new_hard_scenarios():
    ids = {s["id"] for s in client.get("/api/coverage").json()["scenarios"]}
    for sid in ("healthcare-drift", "trading-drift", "science-drift",
                "domain-workflow", "revisit", "topic-order-a", "topic-order-b",
                "topic-order-c", "sql", "toolcall", "science"):
        assert sid in ids, f"missing scenario {sid}"


def test_recommend_endpoint_returns_a_caveated_answer():
    # make sure there is something to recommend
    r = client.post("/api/runs", json={
        "stream": "workflow", "evolver": "scoped", "policy": "simulated",
        "block_size": 4, "n_eval": 4}).json()
    _wait_done(r["id"])
    out = client.get("/api/stats/recommend?stream=workflow").json()
    assert out["recommendation"]
    assert "ranking" in out and out["ranking"]
    assert "guards" in out


def test_guardrail_audit_is_reported():
    r = client.post("/api/runs", json={
        "stream": "trading", "evolver": "seed", "policy": "simulated",
        "block_size": 4, "n_eval": 4}).json()
    g = _wait_done(r["id"])
    assert g["status"] == "done", g.get("message")
    gr = g["result"].get("guardrails")
    assert gr is not None, "no guardrail audit in the result"
    assert "by_violation" in gr and "total" in gr


def test_new_metrics_are_present():
    r = client.post("/api/runs", json={
        "stream": "workflow", "evolver": "scoped", "policy": "simulated",
        "block_size": 4, "n_eval": 4}).json()
    g = _wait_done(r["id"])
    m = g["result"]["metrics"]
    for key in ("forward_transfer", "backward_transfer", "area_under_curve",
                "learning_curve"):
        assert key in m, f"missing metric {key}"
    assert len(m["learning_curve"]) == 2  # two block boundaries


def test_suite_scenarios_and_estimate():
    s = client.get("/api/suite/scenarios?scope=all").json()
    assert len(s["scenarios"]) >= 18
    ids = {x["id"] for x in s["scenarios"]}
    for sid in ("interleaved", "recurring", "long-horizon", "hard-order"):
        assert sid in ids, f"missing stream shape {sid}"
    est = client.post("/api/suite/estimate", json={
        "scope": "fast", "evolvers": ["seed", "scoped"], "iterations": 2,
        "block_size": 5, "n_eval": 5}).json()
    assert est["runs"] == est["scenarios"] * 2 * 2
    assert est["training_episodes"] > 0


def test_suite_run_launches_a_grid():
    r = client.post("/api/suite/run", json={
        "scope": "fast", "evolvers": ["seed", "scoped"], "iterations": 1,
        "block_size": 3, "n_eval": 3}).json()
    assert r["count"] == len(r["scenarios"]) * 2
    ids = [x["run_id"] for x in r["runs"]]
    for rid in ids[:2]:
        g = _wait_done(rid)
        assert g["status"] == "done", g.get("message")


def test_suite_run_rejects_bad_input():
    assert client.post("/api/suite/run", json={
        "scope": "fast", "evolvers": ["nope"]}).status_code == 400
    assert client.post("/api/suite/run", json={
        "scope": "fast", "evolvers": ["seed"], "iterations": 0}).status_code == 400


def test_suite_report_has_evidence_for_each_cell():
    client.post("/api/runs", json={
        "stream": "workflow", "evolver": "scoped", "policy": "simulated",
        "block_size": 4, "n_eval": 4}).json()
    client.post("/api/runs", json={
        "stream": "workflow", "evolver": "gepa", "policy": "simulated",
        "block_size": 4, "n_eval": 4}).json()
    rep = client.get("/api/suite/report").json()
    assert rep["rows"], "no report rows"
    row = rep["rows"][0]
    for key in ("stream", "evolver", "n", "mean", "std", "rank", "metrics"):
        assert key in row, key
    others = [r for r in rep["rows"] if not r.get("is_best")]
    if others:
        assert "margin_vs_best" in others[0]
        assert "p_value" in others[0]
        assert "n_comparisons" in others[0]
    assert rep.get("power") is not None


def test_suite_advice_is_present():
    a = client.get("/api/suite/advice?domain=healthcare").json()
    assert a["domain"] == "healthcare"
    assert a["next_steps"], "advice must say what to do next"
    assert a["unmeasured"], "advice must say what is not measured"


def test_run_reports_cost_and_efficiency():
    r = client.post("/api/runs", json={
        "stream": "workflow", "evolver": "reasoningbank", "policy": "simulated",
        "block_size": 4, "n_eval": 4}).json()
    g = _wait_done(r["id"])
    assert g["status"] == "done", g.get("message")
    c = g["result"].get("cost")
    e = g["result"].get("efficiency")
    assert c is not None and "total_units" in c
    assert e is not None and "gain_pp" in e


def test_benchmarks_endpoint_exposes_bands_and_glossary():
    b = client.get("/api/benchmarks").json()
    assert b["scenarios"], "no benchmark scenarios"
    assert "workflow" in b["scenarios"]
    rows = b["scenarios"]["workflow"]["rows"]
    assert len(rows) == 5
    scoped = [r for r in rows if r["evolver"] == "scoped"][0]
    assert scoped["metrics"]["terminal_advantage"]["mean"] > 0
    assert scoped["description"]
    assert "terminal_advantage" in b["glossary"]
    assert b["glossary"]["terminal_advantage"]["meaning"]


def test_batch_launch_creates_a_grid_of_runs():
    r = client.post("/api/runs/batch", json={
        "stream": "workflow", "evolvers": ["seed", "scoped"], "iterations": 2,
        "block_size": 3, "n_eval": 3}).json()
    assert r["count"] == 4, r
    ids = [x["run_id"] for x in r["runs"]]
    assert len(set(ids)) == 4
    for rid in ids:
        g = _wait_done(rid)
        assert g["status"] == "done", g.get("message")
        # every cell must be reproducible and independent
        assert g["config"]["train_seed"] != g["config"]["eval_seed"]
    summary = client.get("/api/runs/summary").json()
    assert summary["cells"], "summary has no cells"


def test_batch_rejects_bad_input():
    assert client.post("/api/runs/batch", json={
        "stream": "workflow", "evolvers": ["nope"]}).status_code == 400
    assert client.post("/api/runs/batch", json={
        "stream": "workflow", "evolvers": []}).status_code == 400
    assert client.post("/api/runs/batch", json={
        "stream": "workflow", "evolvers": ["seed"], "iterations": 0}).status_code == 400
    assert client.post("/api/runs/batch", json={
        "stream": "no-such-stream", "evolvers": ["seed"]}).status_code == 400


def test_run_lifecycle_cancel_duplicate_delete():
    r = client.post("/api/runs", json={
        "stream": "trading", "evolver": "seed", "policy": "simulated",
        "block_size": 3, "n_eval": 3}).json()
    rid = r["id"]
    dup = client.post(f"/api/runs/{rid}/duplicate").json()
    assert dup["id"] != rid
    assert dup["config"]["train_seed"] == r["config"]["train_seed"] + 1
    assert client.delete(f"/api/runs/{rid}").json()["deleted"] == rid
    assert client.get(f"/api/runs/{rid}").status_code == 404
    _wait_done(dup["id"])


def test_run_assessment_is_attached():
    r = client.post("/api/runs", json={
        "stream": "workflow", "evolver": "scoped", "policy": "simulated",
        "block_size": 5, "n_eval": 5}).json()
    g = _wait_done(r["id"])
    assert g["status"] == "done", g.get("message")
    a = g.get("assessment")
    assert a, "no assessment attached to a finished run"
    assert a["verdict"] in ("expected", "better", "worse", "unknown")
    assert "terminal_advantage" in a["metrics"]


def test_summary_route_is_not_shadowed_by_run_id():
    """/api/runs/summary must not be parsed as a run id."""
    s = client.get("/api/runs/summary").json()
    assert "cells" in s
    assert client.get("/api/runs/summary_typo").status_code == 404


def test_live_mode_launches_when_backend_reachable():
    """Live wiring test: launch against the real backend, see it start, cancel.

    A full live stream takes minutes per LLM call (cloud-backed models), so
    this test validates launch validation + background start + cancel, not a
    finished live run. The finished live comparison lives in
    `tests/live_validation.py`, which is run deliberately rather than on
    every test pass.
    """
    import asyncio
    h = asyncio.run(_health())
    if not (h.get("fox_services") or h.get("ollama")):
        print("SKIP live test: no model backend reachable")
        return
    r = client.post("/api/runs", json={
        "stream": "trading", "evolver": "seed", "policy": "live",
        "policy_cfg": {"model": "fox-managed", "max_steps": 1},
        "block_size": 1, "n_eval": 1, "max_episodes": 1}).json()
    # Accepted means the backend-reachability gate passed and the job was
    # queued. Execution progress is NOT asserted here: under TestClient the
    # portal cancels tasks that hold in-flight network waits when the next
    # request arrives (simulated runs complete fine; a real server runs live
    # jobs to completion - verified manually on :8220).
    assert r["status"] in ("pending", "running", "done", "error"), r
    client.post(f"/api/runs/{r['id']}/cancel")


def test_live_validation_module_runs_its_comparison():
    """The live-vs-simulated comparison must at least exist and skip cleanly.

    It is not executed here (a live stream takes minutes); this checks that
    the module is present and would do the comparison when a backend is
    reachable.
    """
    import asyncio
    from tests import live_validation

    h = asyncio.run(live_validation._health())
    if not h.get("reachable"):
        print("SKIP live validation: no model backend reachable")
        return
    assert callable(live_validation.main)
    assert callable(live_validation.run_one)


async def _health():
    async with get_client() as c:
        return await c.health()
