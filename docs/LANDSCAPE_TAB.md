# Landscape tab — what is being built

A new tab that collects the academic and industry landscape of self-improving
and self-learning agents, and **aligns** it with the axes this lab measures.

## The problem it solves

The lab measures six things: what a method rewrites, whether its learning is
isolated per task family, how it matches a stored experience to a task,
whether it measures continual degradation, whether it uses a disjoint
hold-out, and what kind of evidence backs it.

The literature has hundreds of papers and products that say "the agent
learns" and "the agent improves". Almost none of them are measured on
those axes, because those axes are not usually the question being asked.

So the tab is not a reading list. It is an **alignment matrix**: every
work, placed on the lab's axes, with a derived column that says whether
the work's own protocol could have falsified its own claim.

## Two kinds of fact, kept apart

| kind | source | who says it |
|---|---|---|
| **Source fact** | the paper, the product spec, the writeup | the work's authors |
| **Lab assessment** | the alignment columns | this lab, and it is opinion |

The UI shows which is which. A reader must never mistake the lab's
alignment for something the paper claims. The source facts are
deliberately minimal — title, venue, one line on what it does, one on the
mechanism — because that is what can be stated safely and checked.

## Where the data comes from

`data/landscape.json` ships with the app: a curated knowledge base of
papers, products, frameworks, writeups and blogs, each with provenance
(an arXiv id or a canonical URL, plus the kind of evidence). It is
*reference data*, like the measured bands, and it lives outside the runtime
volume for the same reason.

The tab also **collects**: `POST /api/landscape/works` adds an entry the
user found, and it is stored in SQLite and merged into the knowledge base
alongside the shipped one. The shipped file is the seed; the database
holds what the lab's users have added.

## The knowledge graphs

Two, both generated from the same data:

1. **The landscape graph** — works and the concepts they implement, with
   typed edges (implements, extends, contradicts, requires). It answers
   "what is this area actually made of".
2. **The mechanism graph** — the concepts grouped by the surface they
   rewrite. It answers "what kind of thing is being changed".

Both are Mermaid, rendered live, and both are derived from the data rather
than drawn, so they cannot disagree with the table.

## What it deliberately does not do

- It does not scrape. A crawler against arXiv and vendor sites is
  fragile, slow and unauditable; a curated set with provenance is
  checkable, and the API accepts additions so the set can grow.
- It does not rank works. Popularity is not quality, and the lab's own
  standard is the protocol, not the citation count.
- It does not claim a work is good. It places it on the axes and says
  whether its protocol could have falsified its own claim.
