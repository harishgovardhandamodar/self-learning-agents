# Landscape curation — the quarterly sweep

The knowledge base (`reference/landscape.json`, currently v2026-11: 51 works,
21 concepts, 9 dimensions) decays. The 2025–26 field moves too fast for a
one-off curation to stay honest, so the base is maintained by a quarterly
sweep, not by memory.

## Feeds (in order)

1. The self-evolving-agents survey lineage (arXiv:2507.21046 and its
   references) — the canonical feed for memory/policy evolution work.
2. The Awesome-Self-Evolving-Agents and Awesome-Agent-Memory lists —
   the feed for benchmarks and open-source systems.
3. The lab's own runs — any evolver or mechanism the lab implements that
   the field also implements gets a row pointing at the field instance.

## Rules (enforced by `tests/test_landscape.py`)

- Every work carries provenance: an arXiv id or a resolvable URL.
  `tests/test_landscape_provenance.py` verifies every arXiv id against
  arXiv itself. Prefer arXiv ids for papers; use canonical repo/vendor
  URLs for frameworks, products and benchmarks.
- Every alignment value must be a declared option on its axis.
  `unknown` is always legal — a guess is worse than a gap.
- New axes must earn their place: at least two distinct non-`unknown`
  values across the base, or the axis is decoration.
- Source facts (title/summary/mechanism) are the work's; alignment and
  verdict are this lab's, and the API marks which is which.
- Benchmarks are first-class works (`kind: benchmark`), because a
  protocol that can falsify a claim is the unit the tab exists to track.

## Checklist per sweep

- [ ] Pull the feeds; add works, concepts, and (rarely) dimensions.
- [ ] Bump `version` (YYYY-MM).
- [ ] Run `pytest tests/test_landscape.py tests/test_landscape_provenance.py`.
- [ ] Confirm the graphs still cover every work and concept (tested).
