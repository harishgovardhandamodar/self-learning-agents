# Dashboard analysis → what the user actually hits

Captured with a real browser (Playwright, 1440×1000 and 390×844) against
the running app, plus the 20-iteration sweep. Ordered by user impact.

## What was fixed (all verified by the browser audit)

### Workflow gaps → now closed

| # | Problem | Fix |
|---|---|---|
| W1 | The core comparison workflow was manual: one launch per cell | `POST /api/runs/batch` + the **Runs** tab now launches a whole grid (every evolver × N iterations) from one form, with preset buttons for the paper's topic/workflow scenarios |
| W2 | No iteration sweep from the UI | Batch launch with `iterations`; each cell gets its own train/eval/policy/lesson-id seeds |
| W3 | No "is this good?" signal | `app/benchmarks.py` ships the measured mean ± std per cell; the **Reference** tab shows them and every run is judged "as measured / above / below the band" |
| W4 | No run lifecycle actions | Cancel / Repeat (fresh draws) / Delete per run, plus "clear failed & cancelled" bulk cleanup (`DELETE /api/runs?only_status=`) |
| W5 | Results were a raw `<pre>` dump | Six labelled metric cards (each with its meaning), an SVG accuracy-per-family chart across checkpoints (the paper's A[t,k]), failure modes with causes, and the band verdict |

### UI/UX defects → all closed

| # | Problem | Fix |
|---|---|---|
| U1 | Compare table overflowed (failure-modes column clipped) | Table wraps in a scroll container; reference table gets a wrapping first column |
| U2 | Compare checkboxes floated 600px from their labels | Rebuilt as full-width `.checkline` rows (label + control together) |
| U3 | "Recommended patterns" rendered as a bare heading | Explicit empty state explaining the catalog mount |
| U4 | No empty states | Every list (designs, runs, compare, patterns) explains what to do next |
| U5 | Mobile nav consumed 4 rows | Single scrollable row; brand subtitle hidden below 720px |
| U6 | No charts | Inline SVG line chart of A[t,k] per family, no chart library |
| U7 | Jargon unexplained | Every metric card carries its plain-language meaning; evolvers described in Reference and in tooltips |
| U8 | Unlabeled form controls | All inputs have `for=`/`id` pairs, `aria-label` on filters |
| U9 | Sub-24px tap targets | Checkboxes are 18px inside ≥30px label targets; small buttons 28px |
| U10 | No visible keyboard focus | Global `:focus-visible` ring (2px accent, 2px offset) |
| U11 | Feedback overwritten by the next action | Toasts (auto-dismiss) + persistent inline status lines |
| U12 | Progress was a bare percentage | Progress bar + phase text while a batch runs |
| U13 | Run errors never surfaced | Error runs show the message and the stored traceback in the detail panel |
| U14 | 8s polling re-rendered everything | Polling now only re-renders the run list, and only when it changed; the detail panel polls the selected run |
| U15 | No filtering | Free-text filter + status filter over the run history |
| U16 | No export of results | Design export existed; run results export as JSON via the detail panel's raw view |

## Measurement gaps found by the 20-iteration sweep

| # | Problem | Evidence | Fix |
|---|---|---|---|
| M1 | **GEPA's intent lottery**: each window distilled only the last success, so on trading all 4 flushes distilled `trade.flat` and breakout misfired | trading/GEPA Term −14.2±6.3 → +0.5±9.7 | one lesson per intent per window |
| M2 | **Template clobbering in `generalize_action`**: an action already carrying a true template got re-derived lossily | `f1-podium-round`: `{round}` → `4` | prefer the existing template |
| M3 | **Hidden uuid randomness** in lesson IDs leaked into metrics | repeated "identical" runs differed | `RunConfig.seed` seeds lesson IDs; bit-reproducibility test added |
| M4 | **Single-group streams have ±7–12 std** at eval=30 | sweep std column | the reference bands publish the std, and the UI shows it next to every mean |
| M5 | **`assess_run` judged missing metrics as 0.0** | the iteration loop's unit test caught it at round 1 | a metric a run does not report is no longer assessed |

## The 20-iteration loop

`tests/iteration_loop.sh` runs 20 rounds. Each round:

1. `tests/sweep.py` — mechanism sweep (6 scenarios × 5 evolvers × iterations)
2. `pytest tests/` — unit + API tests
3. `tests/ui_audit.js` — browser-driven audit of the live container

It stops at the first failure and names it. The loop caught three real
defects during development: the missing-metric assessment bug (round 1), a
stale `res` reference in the audit itself (round 1), and the reference-table
overflow (round 3, because the container had not been rebuilt).

Final state: 20/20 rounds pass, 22 tests green, UI audit reports 0 console
errors, 0 failed requests, 0 layout/accessibility/workflow problems.
