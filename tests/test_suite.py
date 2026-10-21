"""
Tests for the evaluation suite itself: the statistics, the cost accounting,
the new stream shapes, and the suite report.

These check the parts that would otherwise be trusted blindly - a p-value
that is always 0.01, a cost counter that never moves, an interleaved
stream that is secretly still blocked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import math

from app import cost as cost_mod
from app import stats as stats_mod
from app import streams as streams_mod


# ------------------------------------------------------------------ stats

def test_welch_detects_a_real_difference():
    a = [10.0, 11.0, 9.5, 10.5, 10.2, 9.8, 10.1, 10.3]
    b = [5.0, 6.0, 4.5, 5.5, 5.2, 4.8, 5.1, 5.3]
    out = stats_mod.welch(a, b)
    assert out["significant"] is True
    assert out["difference"] > 4.0
    assert out["ci_low"] > 0, out


def test_welch_finds_nothing_in_noise():
    a = [10.0, 10.4, 9.8, 10.2, 10.1, 9.9]
    b = [10.1, 9.7, 10.3, 9.9, 10.2, 9.8]
    out = stats_mod.welch(a, b)
    assert out["significant"] is False


def test_welch_refuses_to_claim_with_one_run():
    out = stats_mod.welch([1.0], [2.0])
    assert out["significant"] is False
    assert out["p_value"] is None
    assert "at least 2 runs" in out["note"]


def test_effect_size_scales_with_the_gap():
    # A 0.5pp gap on a 0.09 spread is a large effect; a 4pp gap on the same
    # spread is larger still. The point is that d scales with the gap and
    # is measured in pooled standard deviations.
    tight_a = [10.0, 10.1, 9.9, 10.05]
    tight_b = [9.5, 9.6, 9.4, 9.55]
    small = stats_mod.cohens_d(tight_a, tight_b)
    large = stats_mod.cohens_d([x + 4 for x in tight_a], tight_b)
    assert large > small > 0
    assert stats_mod.effect_label(large) in ("medium", "large")
    # A wide spread on the same gap means a SMALLER effect.
    wide_a = [10.0, 13.0, 7.0, 10.0]
    wide = stats_mod.cohens_d(wide_a, [9.5, 12.5, 6.5, 9.5])
    assert wide < large


def test_holm_correction_removes_chance_significance():
    # Ten comparisons that are all pure noise: uncorrected, several would
    # look significant; corrected, none should.
    ps = {f"c{i}": 0.04 + i * 0.004 for i in range(10)}
    out = stats_mod.holm_bonferroni(ps, alpha=0.05)
    assert out["n_comparisons"] == 10
    # At most one can survive at alpha=0.05 with Holm.
    survivors = [k for k, v in out["significant"].items() if v]
    assert len(survivors) <= 1, out
    # A single very small p-value must survive.
    strong = stats_mod.holm_bonferroni({"a": 0.0001, "b": 0.9, "c": 0.8}, alpha=0.05)
    assert strong["significant"]["a"] is True


def test_power_grows_with_iterations():
    p5 = stats_mod.power_for_margin(5.0, 5, 2.0)
    p20 = stats_mod.power_for_margin(5.0, 20, 2.0)
    assert p20 > p5
    assert 0.0 <= p5 <= 1.0 and 0.0 <= p20 <= 1.0
    needed = stats_mod.iterations_needed(5.0, 2.0, target_power=0.8)
    assert needed is not None and needed > 1
    if needed:
        assert stats_mod.power_for_margin(5.0, needed, 2.0) >= 0.8


# ------------------------------------------------------------------- cost

def test_cost_accounting_counts_what_was_injected():
    class FakeLesson:
        def __init__(self, text="a lesson"):
            self.text = text
            self.scope = "c1|default"

    class FakeInjection:
        def __init__(self, lessons):
            self.lessons = lessons
            self.text = "PROMPT TEXT " * 10

    c = cost_mod.empty_cost()
    inj = FakeInjection([FakeLesson(), FakeLesson()])
    cost_mod.account_episode(c, inj, used=True)
    cost_mod.account_episode(c, inj, used=False)
    assert c["eval_episodes"] == 2
    assert c["lessons_injected"] == 4
    assert c["lessons_used"] == 1
    assert c["prompt_chars"] > 0
    out = cost_mod.finalise(c)
    assert out["lessons_per_episode"] == 2.0
    assert out["lesson_hit_rate"] == 0.25


def test_efficiency_ledger_reports_gain_per_unit():
    c = {"total_units": 100.0, "prompt_units": 60.0, "write_units": 40.0,
         "lessons_created": 10, "lesson_hit_rate": 0.5, "unused_lessons": 2}
    led = cost_mod.efficiency_ledger(c, 20.0)
    assert led["pp_per_100_units"] == 20.0
    assert led["gain_pp"] == 20.0
    # No units -> no ratio, not a division by zero.
    assert cost_mod.efficiency_ledger({"total_units": 0.0}, 5.0)["pp_per_100_units"] is None


# ------------------------------------------------------------ stream shapes

def test_interleaved_stream_actually_interleaves():
    st = streams_mod.build_stream("interleaved", block_size=10, n_eval=5)
    assert st.order == "interleaved"
    tasks = st.training_tasks()
    # The first arrivals must not be one family.
    first_six = [t.group for t in tasks[:6]]
    assert len(set(first_six)) >= 3, first_six
    # Every family still contributes its quota.
    from collections import Counter
    counts = Counter(t.group for t in tasks)
    assert set(counts.values()) == {10}, counts


def test_blocked_streams_are_still_blocked():
    st = streams_mod.build_stream("topic", block_size=10, n_eval=5)
    tasks = st.training_tasks()
    first_ten = [t.group for t in tasks[:10]]
    assert len(set(first_ten)) == 1, first_ten


def test_long_horizon_has_more_blocks_than_the_paper_scenarios():
    long_st = streams_mod.build_stream("long-horizon", block_size=5, n_eval=5)
    topic_st = streams_mod.build_stream("topic", block_size=5, n_eval=5)
    assert len(long_st.blocks) > len(topic_st.blocks)
    assert len(long_st.blocks) >= 8


def test_recurring_stream_repeats_a_family():
    st = streams_mod.build_stream("recurring", block_size=5, n_eval=5)
    assert st.revisits, "the recurring stream must repeat a family"
    fam, positions = next(iter(st.revisits.items()))
    assert len(positions) >= 3, positions
    # tau is the LAST appearance, first_tau the first.
    assert st.tau[fam] == max(positions)
    assert st.first_tau[fam] == min(positions)


# ------------------------------------------------------------------ report

def _cell(stream, evolver, values, std=1.0):
    return {
        "stream": stream, "evolver": evolver, "n": len(values),
        "metrics": {"terminal_advantage": {"values": values, "n": len(values),
                                           "mean": sum(values) / len(values),
                                           "std": std}},
        "guardrails": {"total": 0}, "cost": {"total_units": 10.0},
        "efficiency": {"gain_pp": sum(values) / len(values), "total_units": 10.0,
                       "pp_per_100_units": 1.0},
    }


def test_report_ranks_and_marks_significance():
    from app import suite as suite_mod
    cells = [
        _cell("s", "scoped", [10.0, 10.5, 9.8, 10.2]),
        _cell("s", "reasoningbank", [6.0, 6.5, 5.8, 6.2]),
    ]
    rep = suite_mod.build_report(cells)
    assert len(rep["rows"]) == 2
    top = rep["rows"][0]
    assert top["evolver"] == "scoped" and top["is_best"]
    other = rep["rows"][1]
    assert other["margin_vs_best"] > 3.0
    assert "effect_size" in other and "p_value" in other


def test_report_says_when_the_margin_is_not_real():
    from app import suite as suite_mod
    cells = [
        _cell("s", "scoped", [10.0, 10.1, 9.9, 10.05], std=0.1),
        _cell("s", "reasoningbank", [10.02, 9.98, 10.01, 9.99], std=0.1),
    ]
    rep = suite_mod.build_report(cells)
    loser = [r for r in rep["rows"] if not r.get("is_best")][0]
    assert loser["significant"] is False, loser


def test_advice_is_derived_not_asserted():
    from app import suite as suite_mod
    cells = [
        _cell("workflow", "scoped", [14.0, 14.5, 13.8, 14.2]),
        _cell("workflow", "gepa", [-10.0, -9.5, -10.5, -10.2]),
    ]
    rep = suite_mod.build_report(cells)
    advice = suite_mod.advise("general", rep)
    assert advice["recommendation"] == "scoped"
    assert advice["next_steps"], "advice must say what to do next"
    assert advice["unmeasured"], "advice must say what is not measured yet"


def test_report_does_not_mix_incomparable_geometry():
    """
    A 4-episode probe run and a 20-episode measured run are not comparable.

    Mixing them in one cell produces a variance that is not the method's
    variance, and the comparison then says nothing at all.
    """
    from app import suite as suite_mod
    same = [
        _cell("s", "scoped", [10.0, 10.5, 9.8, 10.2]),
        _cell("s", "reasoningbank", [6.0, 6.5, 5.8, 6.2]),
    ]
    for c in same:
        c["block_size"] = 20
    rep = suite_mod.build_report(same)
    assert len(rep["rows"]) == 2
    loser = [r for r in rep["rows"] if not r.get("is_best")]
    assert loser and loser[0]["margin_vs_best"] > 3.0

    mixed = [
        _cell("s", "scoped", [10.0, 10.5, 9.8, 10.2]),
        _cell("s", "reasoningbank", [6.0, 6.5, 5.8, 6.2]),
    ]
    mixed[0]["block_size"] = 4
    mixed[1]["block_size"] = 20
    rep2 = suite_mod.build_report(mixed)
    # Each geometry is its own comparison group, so nothing is compared.
    assert len(rep2["rows"]) == 2
    for row in rep2["rows"]:
        assert row.get("is_best") is True, row


def test_domain_scenarios_map_real_streams():
    """A domain must resolve to the scenarios that exercise it."""
    from app import suite as suite_mod
    hc = suite_mod.domain_scenarios("healthcare")
    assert "healthcare-drift" in hc and "healthcare" in hc
    tr = suite_mod.domain_scenarios("trading")
    assert "trading-drift" in tr
    # "general" is not an environment; it maps to the non-regulated ones.
    gen = suite_mod.domain_scenarios("general")
    assert gen and "topic" in gen
    assert set(gen) != set(hc)


def test_seed_never_counts_as_a_win():
    """The seed harness wins by not adapting; that is a floor, not a win."""
    from app import suite as suite_mod
    cells = [
        _cell("s", "seed", [0.0, 0.0, 0.0, 0.0]),
        _cell("s", "scoped", [20.0, 21.0, 19.5, 20.5]),
    ]
    rep = suite_mod.build_report(cells)
    advice = suite_mod.advise("general", rep, scenarios=["s"])
    assert advice["recommendation"] == "scoped"
    assert "seed" not in advice["wins"]


def test_advice_is_scoped_to_the_domain():
    from app import suite as suite_mod
    cells = [
        _cell("healthcare-drift", "scoped", [30.0, 31.0, 29.5, 30.5]),
        _cell("trading-drift", "gepa", [30.0, 31.0, 29.5, 30.5]),
    ]
    rep = suite_mod.build_report(cells)
    advice = suite_mod.advise("healthcare", rep)
    assert advice["scoped_to_domain"] is True
    assert advice["recommendation"] == "scoped", advice["wins"]
    assert "gepa" not in advice["wins"]


def test_ceiling_capture_is_reported():
    """The report must say how much of the achievable gain was captured."""
    from app import suite as suite_mod
    cells = [
        _cell("s", "scoped", [16.0, 16.5, 15.8, 16.2]),
        _cell("s", "oracle", [20.0, 20.5, 19.8, 20.2]),
    ]
    cells[0]["is_reference"] = False
    rep = suite_mod.build_report(cells)
    cap = rep.get("ceiling")
    assert cap, "no ceiling reported"
    row = cap[0]
    assert row["stream"] == "s"
    assert abs(row["ceiling"] - 20.125) < 0.01, row
    assert row["best_real"] == "scoped"
    assert abs(row["best_mean"] - 16.125) < 0.01, row
    assert 79.0 < row["captured_pct"] < 81.0, row
    assert abs(row["headroom_pp"] - 4.0) < 0.01, row


def test_oracle_is_a_true_ceiling():
    """
    The oracle must not be beaten by a real method on the same stream.

    It is scoped with the ground-truth family instead of the discovered
    cluster, so it has strictly more information; anything it loses to means
    the ceiling is not a ceiling and the capture percentage is meaningless.
    """
    import asyncio
    from app import streams as streams_mod
    from app.runner import RunConfig, execute

    async def go():
        stream = streams_mod.build_stream("workflow", block_size=10, n_eval=10)
        out = {}
        for eid in ("scoped", "oracle"):
            cfg = RunConfig(
                design={"spec": {"harness": {"system_prompt": "x"}}},
                stream=stream, evolver=eid,
                evolver_cfg={"router_threshold": 0.40},
                policy_cfg={"mode": "simulated", "skill": 0.9, "seed": 7})
            res = await execute(cfg)
            out[eid] = res["metrics"]["terminal_advantage"]
        # The oracle may not lose by more than sampling noise.
        assert out["oracle"] >= out["scoped"] - 5.0, out
    asyncio.run(go())


def test_estimate_is_honest_about_size():
    from app import suite as suite_mod
    ids = suite_mod.suite_scenarios("fast")
    est = suite_mod.estimate_cost(ids, ["seed", "scoped"], 3, block_size=10, n_eval=10)
    assert est["runs"] == len(ids) * 2 * 3
    assert est["training_episodes"] > 0
    assert est["evaluation_episodes"] > est["training_episodes"]
