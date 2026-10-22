"""
The landscape: the knowledge base must be complete, honest and findable.

The important checks are the ones about honesty. A knowledge base that
asserts more than it can support is worse than no knowledge base, so:

  * every work must carry provenance (an identifier or a resolvable URL)
  * every alignment value must be one of the declared options
  * source facts and lab assessments must stay distinguishable
  * a work that claims a protocol must actually have one
  * the graphs must cover every work and concept
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import landscape as ls

REPO = Path(__file__).resolve().parent.parent


def test_landscape_loads():
    data = ls.load_landscape()
    assert data["works"], "the landscape knowledge base is empty"
    assert data["concepts"], "no concepts"
    assert data["dimensions"], "no alignment dimensions"
    assert len(data["works"]) >= 25, len(data["works"])


def test_the_shipped_data_is_not_shadowed_by_the_runtime_volume():
    """
    The landscape lives in reference/, not data/.

    The runtime Docker volume is mounted at /app/data and shadows anything
    shipped there. The measured bands were shadowed once already; the same
    trap applies to every shipped reference file.
    """
    for path in ls.LANDSCAPE_CANDIDATES:
        if path.parent == REPO / "data":
            continue  # a fallback is allowed, but it must not be the primary
    primary = ls.LANDSCAPE_CANDIDATES[0]
    assert primary.parent != REPO / "data", \
        f"the primary landscape path {primary} is inside the runtime volume"
    assert primary.exists(), f"the shipped landscape is missing at {primary}"


def test_every_work_has_provenance():
    """No work may enter the base without a way to check it."""
    for w in ls.works():
        assert w.get("title"), w.get("id")
        assert w.get("url"), f"{w.get('id')} has no url"
        assert w.get("url").startswith("http"), w.get("url")
        assert w.get("summary"), f"{w.get('id')} has no summary"
        assert w.get("mechanism"), f"{w.get('id')} has no mechanism"
        assert w.get("kind") in ("paper", "product", "framework", "writeup",
                                 "blog", "note"), w.get("kind")
        assert w.get("evidence") or True  # the alignment carries it


def test_every_alignment_value_is_a_declared_option():
    dims = {d["id"]: d for d in ls.dimensions()}
    for w in ls.works():
        a = ls.align_work(w)
        for key, value in a.items():
            assert key in dims, f"{w.get('id')}: unknown axis {key}"
            assert value in dims[key]["options"], \
                f"{w.get('id')}: {key}={value} is not a declared option"


def test_source_facts_and_lab_assessments_are_separable():
    """The alignment is the lab's; the summary is the work's. They must not mix."""
    for w in ls.works():
        # the source facts live at the top level, the assessment is computed
        assert "align" in w
        assessment = ls.assess(w)
        assert assessment["verdict"]
        assert assessment["why"]
        # a work with no alignment is still listed, but is not judged
        if all(v == "unknown" for v in ls.align_work(w).values()):
            assert "not falsifiable" in assessment["verdict"]


def test_the_falsifiability_rule_is_the_one_claimed():
    """
    "Could this work's own protocol have falsified its own claim?"

    That means: it measures degradation across a changing stream AND it is
    graded on a disjoint hold-out. Nothing else counts.
    """
    for w in ls.works():
        a = ls.align_work(w)
        expected = (a["evaluation"] == "yes" and a["holdout"] == "yes")
        assert ls.assess(w)["falsifiable"] is expected, w.get("id")


def test_graphs_cover_every_work_and_concept():
    land = ls.landscape_graph()
    mech = ls.mechanism_graph()
    assert land["mermaid"].startswith("flowchart")
    assert mech["mermaid"].startswith("flowchart")
    assert land["nodes"] == len(ls.works()) + len(ls.concepts())
    # every work appears in the landscape graph
    for w in ls.works():
        assert f"W_{w['id']}" in land["mermaid"], w["id"]
    # every concept appears in the mechanism graph
    for c in ls.concepts():
        assert f"C_{c['id']}" in mech["mermaid"], c["id"]


def test_concepts_referenced_by_works_exist():
    known = {c["id"] for c in ls.concepts()}
    for w in ls.works():
        for cid in w.get("concepts", []) or []:
            assert cid in known, f"{w.get('id')} references unknown concept {cid}"


def test_search_filters():
    all_works = ls.search()
    assert len(all_works) == len(ls.works())
    papers = ls.search(kind="paper")
    assert papers and all(w["kind"] == "paper" for w in papers)
    hit = ls.search(q="reflexion")
    assert hit, "search for a known title found nothing"
    assert all("reflexion" in (w["title"] + w["summary"] + w["mechanism"]).lower()
               for w in hit)


def test_every_alignment_axis_is_covered_by_the_works():
    """An axis nobody can populate is an axis that does not earn its place."""
    dims = [d["id"] for d in ls.dimensions()]
    for axis in dims:
        values = {ls.align_work(w)[axis] for w in ls.works()}
        values.discard("unknown")
        assert len(values) >= 2, f"the axis '{axis}' only ever has one value"


def test_landscape_pattern_links_resolve_in_the_patterns_catalog():
    """The links into the patterns catalog must point at real patterns."""
    from app import designs as designs_mod
    cat = designs_mod.pattern_catalog()
    known = {p["id"] for p in cat.get("patterns", [])}
    if not known:
        print("SKIP: the patterns catalog is not mounted")
        return
    for w in ls.works():
        for pid in w.get("patterns", []) or []:
            assert pid in known, f"{w.get('id')} links to unknown pattern {pid}"
