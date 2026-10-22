"""
The design documents: they must parse, and they must stay true.

Two gates:

  1. Every mermaid source in design/*.md is validated with mermaid's own
     parser, so a document cannot ship a diagram that fails to render.
  2. The documents' factual claims are checked against the code, so they
     cannot drift: the modules named must exist, the tables named must be
     in the schema, the evolvers named must be the evolvers registered.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import design_docs as dd

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "design"


def _all_mermaid() -> list:
    out = []
    for doc in dd.list_documents():
        for b in doc["blocks"]:
            if b["kind"] == "mermaid":
                out.append((doc["id"], b["source"]))
    return out


def test_every_document_parses_into_known_blocks():
    docs = dd.list_documents()
    assert len(docs) >= 6, "the design tab needs its documents"
    for d in docs:
        assert d["title"], d["id"]
        assert d["blocks"], d["id"]
        assert d["outline"], d["id"]


def test_the_documentation_covers_the_requested_diagram_kinds():
    """UML, activity, transaction, module and mermaid diagrams are all present."""
    sources = "\n".join(s for _, s in _all_mermaid())
    assert "classDiagram" in sources, "no UML class diagram"
    assert "sequenceDiagram" in sources, "no transaction/sequence diagram"
    assert "stateDiagram" in sources, "no state/activity diagram"
    assert "erDiagram" in sources, "no data model diagram"
    # a module map: a flowchart whose subgraphs are the layers
    assert "subgraph" in sources, "no module/component map"
    # activity: a flowchart with decision nodes
    assert re.search(r"\{.*\}", sources), "no decision nodes (activity flow)"


def test_every_mermaid_diagram_is_valid():
    """Validate with mermaid's own parser under jsdom."""
    if not (shutil.which("node") and (REPO / "web" / "vendor" / "mermaid.min.js").exists()
            and Path("/tmp/node_modules/jsdom").exists()):
        print("SKIP: node, jsdom or the vendored mermaid build is missing")
        return
    items = _all_mermaid()
    payload = json.dumps([s for _, s in items])
    script = f'''
const {{ JSDOM }} = require("/tmp/node_modules/jsdom");
const dom = new JSDOM("<!doctype html><html><body></body></html>", {{
  pretendToBeVisual: true, url: "http://localhost/",
}});
global.window = dom.window;
global.document = dom.window.document;
global.navigator = dom.window.navigator;
const mermaid = require({json.dumps(str(REPO / "web" / "vendor" / "mermaid.min.js"))});
mermaid.initialize({{ startOnLoad: false, securityLevel: "strict" }});
const items = {payload};
(async () => {{
  let bad = 0;
  for (let i = 0; i < items.length; i++) {{
    try {{ await mermaid.parse(items[i]); }}
    catch (e) {{ bad++; console.log("PARSE FAIL [" + i + "]: " + String(e).slice(0, 300)); }}
  }}
  console.log("checked " + items.length + " diagrams, " + bad + " failed");
  process.exit(bad ? 1 : 0);
}})();
'''
    r = subprocess.run(["node", "-e", script], capture_output=True, text=True,
                       timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out[-4000:]
    assert "0 failed" in out, out[-3000:]


def test_module_dependency_graph_is_generated_from_the_imports():
    """The module map is measured, not drawn: it must match the real imports."""
    from app import design_docs as dd

    graph = dd.module_dependencies()
    assert "main" in graph and "runner" in graph
    # The runner really does depend on the evolvers, the policy and the router.
    for dep in ("evolvers", "policy", "retrieval", "metrics"):
        assert dep in graph["runner"], graph["runner"]
    # And the API really does depend on the suite and the benchmarks.
    for dep in ("suite", "benchmarks", "designs"):
        assert dep in graph["main"], graph["main"]
    # No module may depend on itself.
    for name, deps in graph.items():
        assert name not in deps

    # The generated diagram is injected into the module-map document.
    doc = dd.document("05-evaluation-suite")
    generated = [b for b in doc["blocks"] if b.get("generated")]
    assert generated, "the module-dependency diagram was not injected"
    assert "flowchart" in generated[0]["source"]


def test_the_landscape_document_agrees_with_the_landscape_module():
    """
    The landscape design document must not drift from the code it describes.

    It states the axes, the falsifiability rule, the number of works and
    the verdict counts. Each of those is checkable, so each is checked.
    """
    from app import landscape as ls

    doc = dd.document("07-landscape")
    assert doc, "the landscape subsystem is not documented in the design tab"
    text = "\n".join(
        b.get("text", "") + b.get("source", "") + b.get("title", "")
        for b in doc["blocks"] if b.get("kind") != "table"
    ) + "\n" + "\n".join(
        " ".join(r) for b in doc["blocks"] if b.get("kind") == "table"
        for r in [b.get("header", []) + [c for row in b.get("rows", []) for c in row]]
    )

    # every axis the code knows must be named in the document
    for dim in ls.dimensions():
        assert dim["id"] in text, f"the document never names the {dim['id']} axis"
        assert dim["label"] in text, f"the document never names {dim['label']!r}"

    # the rule must be stated as the code implements it
    assert "evaluation == yes" in text
    assert "holdout == yes" in text

    # the counts must be the real ones
    works = len(ls.works())
    assert f"{works} works" in text, \
        f"the document does not state the real work count ({works})"
    falsifiable = ls.summary()["falsifiable"]
    partial = sum(1 for w in ls.works()
                  if ls.assess(w)["verdict"] == "partially falsifiable")
    assert f"**{falsifiable}** is falsifiable" in text, \
        f"the document states the wrong falsifiable count ({falsifiable})"
    assert f"**{partial}** are partially" in text, \
        f"the document states the wrong partial count ({partial})"

    # the two graphs it names must exist
    assert ls.landscape_graph()["nodes"] > 0
    assert ls.mechanism_graph()["surfaces"]
    assert "landscape graph" in text
    assert "mechanism graph" in text

    # the local table it names must be in the schema
    from app.db import SCHEMA
    assert "CREATE TABLE IF NOT EXISTS landscape_entries" in SCHEMA


def test_documents_name_modules_that_exist():
    """The module map must name the modules that are actually there."""
    from app import (benchmarks, cost, db, designs, envs, evolvers, flows,
                      harness, landscape, llm, main, metrics, policy, retrieval,
                      runner, stats, streams)
    modules = [benchmarks, cost, db, designs, envs, evolvers, flows, harness,
               landscape, llm, main, metrics, policy, retrieval, runner, stats,
               streams]
    text = "\n".join(p.read_text() for p in DOCS.glob("*.md"))
    for m in modules:
        name = m.__name__.split(".")[-1]
        assert (REPO / "app" / (name + ".py")).exists()
        assert f"{name}.py" in text, \
            f"the documents never mention the {name} module"


def test_documents_name_the_real_tables_and_evolvers():
    from app.db import SCHEMA
    from app.evolvers import EVOLVERS
    text = "\n".join(p.read_text() for p in DOCS.glob("*.md"))
    for table in ("designs", "runs", "checkpoints", "traces", "streams"):
        assert table in text, f"the data model never mentions the {table} table"
        assert f"CREATE TABLE IF NOT EXISTS {table}" in SCHEMA, \
            f"the {table} table is not in the schema"
    for eid in EVOLVERS:
        assert eid in text, f"the documents never mention the {eid} evolver"


def test_documents_agree_with_the_api_surface():
    """The API diagram must state the number of endpoints that exist."""
    from app.main import app
    routes = {r.path for r in app.routes if getattr(r, "path", "").startswith("/api")}
    text = (DOCS / "01-system-context.md").read_text()
    assert f"{len(routes)} endpoints" in text, \
        f"the context document does not state the real endpoint count ({len(routes)})"
    assert (DOCS / "02-uml.md").read_text().count(f"{len(routes)} endpoints") == 1


def test_documents_state_the_scenario_and_family_counts():
    from app import envs, streams as streams_mod
    text = "\n".join(p.read_text() for p in DOCS.glob("*.md"))
    assert f"{len(envs.all_groups())} families" in text, \
        "the documents state a family count that is now wrong"
    assert f"{len(streams_mod.SCENARIOS)} scenarios" in text, \
        "the documents state a scenario count that is now wrong"
    # the ordering permutations are counted separately, not folded in
    assert f"{len(streams_mod.TOPIC_PERMUTATIONS)} orderings" in text
