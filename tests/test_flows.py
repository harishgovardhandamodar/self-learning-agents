"""
Diagram tests: every mermaid source the lab generates must be valid.

A diagram that fails to render is worse than no diagram, so the generator is
checked here rather than in the browser. The syntax is validated by
mermaid's own parser when it is available (node + the vendored mermaid),
and by structural checks always.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import designs as designs_mod, flows as flows_mod

REPO = Path(__file__).resolve().parent.parent


def _all_diagrams():
    """Every diagram the lab can produce, across all evolvers and streams."""
    out = []
    for eid in flows_mod.EVOLVER_FLOW:
        for stream in ("workflow", "topic", "compound", "healthcare", "trading"):
            d = designs_mod.new_design(name="t", domain="general", evolver=eid,
                                       stream=stream)
            out.append(flows_mod.agent_flow_diagram(d, ["a", "b"], {"a": "A", "b": "B"}))
            out.append(flows_mod.stream_diagram(["a", "b"], {"a": "A", "b": "B"}, 20))
    return out


def _nodes_used(src: str) -> set:
    """Every node id that appears in an edge (directives are not nodes)."""
    used = set()
    # edges:  A --> B   A -->|label| B   A -.-> B   A ==> B
    for m in re.finditer(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*[-.=]+>\|?[^|\n]*\|?\s*([A-Za-z_][A-Za-z0-9_]*)",
            src):
        used.add(m.group(1))
        used.add(m.group(2))
    # classDef / class / style are directives, not nodes.
    return used - {"classDef", "class", "style", "linkStyle", "click"}


def _nodes_defined(src: str) -> set:
    """Every node id that has a shape (a label) somewhere in the source."""
    defined = set()
    # A node is defined when an id is followed by a shape:  ID["..."]  ID{"..."}
    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*(\[\[|\[\(|\[|\{\{|\{)", src):
        defined.add(m.group(1))
    return defined - {"classDef", "class", "style", "linkStyle", "click"}


def test_generated_diagrams_are_structurally_sound():
    for d in _all_diagrams():
        src = d["mermaid"]
        assert src.startswith("flowchart "), d["id"]
        # Every node referenced by an edge must also be defined.
        missing = _nodes_used(src) - _nodes_defined(src)
        assert not missing, f"{d['id']} references undefined nodes: {missing}"
        # No `&` chaining (a common mermaid parse failure).
        assert " & " not in src, f"{d['id']} uses edge chaining"
        # Balanced brackets on every line.
        for line in src.splitlines():
            assert line.count("[") == line.count("]"), (d["id"], line)
            assert line.count("{") == line.count("}"), (d["id"], line)
            assert line.count('"') % 2 == 0, (d["id"], line)


def test_agent_flow_differs_per_evolver():
    """The diagram must reflect the design, not be a static picture."""
    seen = set()
    for eid in flows_mod.EVOLVER_FLOW:
        d = designs_mod.new_design(name="t", domain="general", evolver=eid,
                                   stream="workflow")
        src = flows_mod.agent_flow_diagram(d)["mermaid"]
        seen.add(src)
    assert len(seen) == len(flows_mod.EVOLVER_FLOW), \
        "every evolver must produce its own flow"


def test_seed_flow_has_no_write_path():
    d = designs_mod.new_design(name="t", domain="general", evolver="seed")
    out = flows_mod.agent_flow_diagram(d)
    assert out["writes"] == []
    assert "No adaptation" in out["mermaid"]


def test_scoped_flow_shows_scope_and_workflow_conditioning():
    d = designs_mod.new_design(name="t", domain="general", evolver="scoped")
    src = flows_mod.agent_flow_diagram(d)["mermaid"]
    assert "workflow" in src.lower()
    assert "conflict" in src.lower()


def test_labels_are_escaped():
    """A group name with quotes or brackets must not break the diagram."""
    out = flows_mod.stream_diagram(["x"], {"x": 'Weird "name" [with] {braces}'}, 10)
    for line in out["mermaid"].splitlines():
        assert line.count("[") == line.count("]"), line
        assert line.count("{") == line.count("}"), line
        assert line.count('"') % 2 == 0, line


def _mermaid_available() -> bool:
    node = shutil.which("node")
    mermaid_js = REPO / "web" / "vendor" / "mermaid.min.js"
    jsdom = Path("/tmp/node_modules/jsdom")
    return bool(node) and mermaid_js.exists() and jsdom.exists()


def test_mermaid_parser_accepts_every_generated_diagram():
    """Validate with mermaid's own parser when node + jsdom + the build exist.

    mermaid's browser build needs a DOM, so the check runs it under jsdom
    rather than pulling in a headless browser for a syntax check.
    """
    if not _mermaid_available():
        print("SKIP: node, jsdom or vendored mermaid not available")
        return
    diagrams = _all_diagrams()
    payload = json.dumps([d["mermaid"] for d in diagrams])
    script = f'''
const {{ JSDOM }} = require("/tmp/node_modules/jsdom");
const dom = new JSDOM("<!doctype html><html><body></body></html>", {{
  pretendToBeVisual: true, url: "http://localhost/",
}});
global.window = dom.window;
global.document = dom.window.document;
global.navigator = dom.window.navigator;
global.DOMPurify = undefined;
const mermaid = require({json.dumps(str(REPO / "web" / "vendor" / "mermaid.min.js"))});
mermaid.initialize({{ startOnLoad: false, securityLevel: "strict" }});
const diagrams = {payload};
(async () => {{
  let bad = 0;
  for (let i = 0; i < diagrams.length; i++) {{
    try {{ await mermaid.parse(diagrams[i]); }}
    catch (e) {{ bad++; console.log("PARSE FAIL " + i + ": " + String(e).slice(0, 400)); }}
  }}
  console.log("checked " + diagrams.length + " diagrams, " + bad + " failed");
  process.exit(bad ? 1 : 0);
}})();
'''
    r = subprocess.run(["node", "-e", script], capture_output=True, text=True,
                       timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out[-4000:]
    assert "0 failed" in out, out[-3000:]
