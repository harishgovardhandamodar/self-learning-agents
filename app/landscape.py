"""
The landscape: the academic and industry field of self-learning agents,
aligned with the axes this lab measures.

Two kinds of fact are kept strictly apart:

  * SOURCE FACT - what the work says about itself: title, venue, one line
    on what it does, one on the mechanism. Stated conservatively, and
    always with provenance.
  * LAB ASSESSMENT - the alignment columns. These are this lab's reading
    of the work, and the UI labels them as such. A reader must never
    mistake an alignment for a claim the paper makes.

The shipped file is the seed of the knowledge base; entries a user adds
through the API live in SQLite and are merged in on read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# The reference data lives outside the runtime volume, for the same reason
# the measured bands do: the volume would shadow it.
LANDSCAPE_CANDIDATES = [
    Path(os.environ.get("SLA_LANDSCAPE_DIR",
                        str(Path(__file__).resolve().parent.parent / "reference")))
    / "landscape.json",
    Path(__file__).resolve().parent.parent / "data" / "landscape.json",
]

_CACHE: Optional[Dict[str, Any]] = None


def load_landscape() -> Dict[str, Any]:
    """The shipped knowledge base."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    for path in LANDSCAPE_CANDIDATES:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if data.get("works"):
            data["source_path"] = str(path)
            _CACHE = data
            return data
    _CACHE = {"works": [], "concepts": [], "dimensions": [], "version": "empty",
               "source_path": ""}
    return _CACHE


def dimensions() -> List[Dict[str, Any]]:
    return load_landscape().get("dimensions", [])


def concepts() -> List[Dict[str, Any]]:
    return load_landscape().get("concepts", [])


def works(include_added: bool = True) -> List[Dict[str, Any]]:
    """Every work: the shipped seed plus whatever the lab's users added."""
    out = [dict(w) for w in load_landscape().get("works", [])]
    if include_added:
        try:
            from . import db
            for row in db.list_landscape_entries():
                entry = dict(row)
                entry.setdefault("kind", "note")
                entry["added_locally"] = True
                out.append(entry)
        except Exception:
            pass
    return out


def concept_index() -> Dict[str, Dict[str, Any]]:
    return {c["id"]: c for c in concepts()}


def align_work(work: Dict[str, Any]) -> Dict[str, Any]:
    """The alignment, with the missing axes filled in as 'unknown'."""
    dims = dimensions()
    given = work.get("align") or {}
    out = {}
    for d in dims:
        key = d["id"]
        value = given.get(key)
        # "unknown" is always a legal value: a work that has not been placed
        # on an axis is placed on it as unknown, not as a wrong answer.
        if value in (d.get("options") or []) or value == "unknown":
            out[key] = value
        else:
            out[key] = "unknown"
    return out


def assess(work: Dict[str, Any]) -> Dict[str, Any]:
    """
    Could this work's own protocol have falsified its own claim?

    A work that claims to learn but measures nothing under drift, or grades
    only on what it trained on, cannot be contradicted by its own results.
    That is the distinction the tab exists to make visible.
    """
    a = align_work(work)
    falsifiable = (a["evaluation"] == "yes" and a["holdout"] == "yes")
    partial = (a["evaluation"] in ("yes", "partial") and a["holdout"] == "yes")
    return {
        "verdict": ("falsifiable" if falsifiable else
                    "partially falsifiable" if partial else
                    "not falsifiable by its own protocol"),
        "falsifiable": falsifiable,
        "why": _why(a, work),
    }


def _why(a: Dict[str, Any], work: Dict[str, Any] = None) -> str:
    bits = []
    if a["evaluation"] != "yes":
        if a["evaluation"] == "partial":
            bits.append("it measures some effect of learning, but not interference or "
                        "forgetting across a changing stream")
        else:
            bits.append("it does not measure interference or forgetting across a changing stream")
    if a["holdout"] != "yes":
        bits.append("it is not graded on instances it was not trained on")
    if not bits:
        return ("it measures degradation across a changing stream on a disjoint hold-out, "
                "so a change that made things worse would have been caught")
    return "because " + " and ".join(bits)


def summary() -> Dict[str, Any]:
    """The headline numbers for the tab's header."""
    ws = works()
    dims = dimensions()
    by_kind: Dict[str, int] = {}
    by_evidence: Dict[str, int] = {}
    by_surface: Dict[str, int] = {}
    falsifiable = 0
    for w in ws:
        by_kind[w.get("kind", "other")] = by_kind.get(w.get("kind", "other"), 0) + 1
        a = align_work(w)
        by_evidence[a.get("evidence", "unknown")] = by_evidence.get(a.get("evidence", "unknown"), 0) + 1
        by_surface[a.get("surface", "unknown")] = by_surface.get(a.get("surface", "unknown"), 0) + 1
        if assess(w)["falsifiable"]:
            falsifiable += 1
    return {
        "works": len(ws),
        "concepts": len(concepts()),
        "by_kind": by_kind,
        "by_evidence": by_evidence,
        "by_surface": by_surface,
        "falsifiable": falsifiable,
        "dimensions": len(dims),
        "version": load_landscape().get("version", ""),
    }


def search(q: str = "", kind: str = "", surface: str = "",
           evidence: str = "") -> List[Dict[str, Any]]:
    """Filter the knowledge base."""
    query = (q or "").strip().lower()
    out = []
    for w in works():
        a = align_work(w)
        if kind and w.get("kind") != kind:
            continue
        if surface and a.get("surface") != surface:
            continue
        if evidence and a.get("evidence") != evidence:
            continue
        if query:
            hay = " ".join(str(w.get(k, "")) for k in
                           ("title", "authors", "org", "venue", "summary",
                            "mechanism", "id")).lower()
            if query not in hay:
                continue
        out.append({
            "id": w.get("id"), "kind": w.get("kind"), "title": w.get("title"),
            "authors": w.get("authors"), "venue": w.get("venue"),
            "year": w.get("year"), "url": w.get("url"),
            "summary": w.get("summary"), "mechanism": w.get("mechanism"),
            "align": a, "assessment": assess(w),
            "concepts": w.get("concepts", []),
            "patterns": w.get("patterns", []),
            "added_locally": bool(w.get("added_locally")),
        })
    out.sort(key=lambda w: (w.get("year") or 0, w.get("title") or ""), reverse=True)
    return out


# ------------------------------------------------------------------ graphs

def landscape_graph() -> Dict[str, Any]:
    """
    The landscape graph: the mechanisms, and the works that implement them.

    The shape matters as much as the content. Laid out as "each work
    points at the concepts it implements", thirty-six works and fifteen
    concepts render as a single column over five thousand pixels tall - a
    picture nobody can read, and one an enlarged view cannot rescue.

    So each concept gets a box of its own: the concept heads it, the
    works that implement it sit below it, and an edge runs from the
    concept to each of them. The picture answers a question the flat form
    cannot - *what implements this mechanism?* - and the aspect comes out
    close to a page, which is what makes the enlarged view worth having.

    A work that implements three mechanisms appears in three boxes, under
    a node id of its own each time. Reusing one id would make mermaid
    merge the definitions and print the node's id in place of its label,
    which is exactly the kind of quietly wrong picture this lab does not
    ship.

    Mermaid source is returned alongside the data so the diagram and the
    table cannot disagree.
    """
    idx = concept_index()
    ws = works()

    # which works implement which concept
    by_concept: Dict[str, List[Dict[str, Any]]] = {}
    for w in ws:
        for cid in w.get("concepts", []) or []:
            if cid in idx:
                by_concept.setdefault(cid, []).append(w)

    lines = ["flowchart TB"]
    work_nodes: List[str] = []
    edges = 0

    for c in concepts():
        cid = c["id"]
        members = by_concept.get(cid, [])
        if not members:
            continue
        label = esc(c.get("label", cid))
        kind = esc(c.get("kind", ""))
        lines.append(f'  subgraph G_{cid}[" "]')
        lines.append("    direction TB")
        lines.append(f'    C_{cid}["{label}<br/><small>{kind}</small>"]')
        for w in members:
            wid = w.get("id", "")
            node = f"W_{wid}__{cid}"
            work_nodes.append(node)
            title = esc(w.get("title", wid)[:28])
            wkind = esc(w.get("kind", ""))
            lines.append(f'    {node}["{title}<br/><small>{wkind}</small>"]')
        lines.append("  end")
        lines.append(f"  style G_{cid} fill:#141922,stroke:#2a3140,color:#d7dee9")
        for w in members:
            lines.append(f"  C_{cid} --> W_{w.get('id','')}__{cid}")
            edges += 1

    # the chain holds the boxes in a readable order
    spine = [c["id"] for c in concepts() if by_concept.get(c["id"])]
    for a, b in zip(spine, spine[1:]):
        lines.append(f"  G_{a} ~~~ G_{b}")

    lines.append("  classDef work fill:#16283a,stroke:#5aa9ff,color:#d7dee9;")
    lines.append("  classDef paradigm fill:#1d3a2a,stroke:#4cc38a,color:#d7dee9;")
    lines.append("  classDef mechanism fill:#3a2a1d,stroke:#ffb020,color:#d7dee9;")
    lines.append("  classDef problem fill:#3a1d1d,stroke:#ff6b6b,color:#d7dee9;")
    lines.append("  classDef practice fill:#2a2340,stroke:#b98cff,color:#d7dee9;")
    if work_nodes:
        lines.append("  class " + ",".join(work_nodes) + " work;")
    by_concept_kind: Dict[str, List[str]] = {}
    for cid in spine:
        c = idx[cid]
        by_concept_kind.setdefault(c.get("kind", "mechanism"), []).append(f"C_{cid}")
    for kind, members in by_concept_kind.items():
        lines.append("  class " + ",".join(members) + f" {kind};")

    return {
        "title": "The landscape: the mechanisms, and the works that implement them",
        "mermaid": "\n".join(lines),
        "nodes": len(work_nodes) + len(spine),
        "edges": edges,
        "concepts": concepts(),
        "by_concept": {cid: [w.get("id") for w in m]
                       for cid, m in by_concept.items()},
    }



def mechanism_graph() -> Dict[str, Any]:
    """
    The mechanism graph: concepts grouped by what they rewrite.

    It answers the question the alignment matrix answers a row at a time:
    what kind of thing is actually being changed when an agent improves.
    """
    idx = concept_index()
    surfaces: Dict[str, List[str]] = {}
    for w in works():
        a = align_work(w)
        surface = a.get("surface", "unknown")
        for cid in w.get("concepts", []) or []:
            c = idx.get(cid)
            if not c:
                continue
            bucket = surfaces.setdefault(surface, [])
            if cid not in bucket:
                bucket.append(cid)
    lines = ["flowchart TD"]
    for surface, cids in sorted(surfaces.items()):
        lines.append(f'  S_{surface}["{esc(surface)}"]')
        for cid in cids:
            lines.append(f"  S_{surface} --> C_{cid}")
    for c in concepts():
        lines.append(f'  C_{c["id"]}["{esc(c.get("label", c["id"]))}<br/>'
                     f'<small>{esc(c.get("kind", ""))}</small>"]')
    lines.append("  classDef surface fill:#3a2a1d,stroke:#ffb020,color:#d7dee9;")
    lines.append("  classDef mechanism fill:#16283a,stroke:#5aa9ff,color:#d7dee9;")
    lines.append("  classDef paradigm fill:#1d3a2a,stroke:#4cc38a,color:#d7dee9;")
    lines.append("  classDef problem fill:#3a1d1d,stroke:#ff6b6b,color:#d7dee9;")
    lines.append("  classDef practice fill:#2a2340,stroke:#b98cff,color:#d7dee9;")
    for surface in surfaces:
        lines.append(f"  class S_{surface} surface;")
    by_kind: Dict[str, List[str]] = {}
    for c in concepts():
        by_kind.setdefault(c.get("kind", "mechanism"), []).append(f"C_{c['id']}")
    for kind, members in by_kind.items():
        lines.append("  class " + ",".join(members) + f" {kind};")
    return {
        "title": "Mechanisms grouped by what they rewrite",
        "mermaid": "\n".join(lines),
        "surfaces": sorted(surfaces),
        "concepts": concepts(),
    }


def alignment_matrix() -> Dict[str, Any]:
    """The matrix: one row per work, one column per axis."""
    dims = dimensions()
    rows = []
    for w in search():
        row = {"id": w["id"], "kind": w["kind"], "title": w["title"],
               "year": w.get("year"), "venue": w.get("venue"),
               "url": w.get("url"), "added_locally": w.get("added_locally"),
               "align": w["align"], "assessment": w["assessment"],
               "summary": w.get("summary"), "mechanism": w.get("mechanism"),
               "patterns": w.get("patterns", [])}
        rows.append(row)
    return {"dimensions": dims, "rows": rows}


def esc(text: Any) -> str:
    return (str(text or "")
            .replace('"', "'")
            .replace("\n", " ")
            .replace("[", "(")
            .replace("]", ")")
            .replace("|", "/")
            .replace("<", "(")
            .replace(">", ")"))
