# Development

How the code is organized and tested.

Work from the repository root and preserve unrelated local changes. Use uv for Python commands and keep prose free of em dashes. Report presentation belongs in the generators; numerical results belong in analysis JSON. The renderer must not query Lichess or rerun estimates.

## Implementation map

| Responsibility | Files |
| --- | --- |
| PGN parsing, canonical boards, selected moves, chapter regions and first entries | [graph.py](../repertoire_score/graph.py), [board_cache.py](../repertoire_score/board_cache.py) |
| Authenticated evidence, validation and persistent cache | [explorer.py](../repertoire_score/explorer.py) |
| The `repertoire` command, scoring orchestration, evidence model and calculated uncertainty | [cli.py](../repertoire_score/cli.py), [score.py](../repertoire_score/score.py), [model.py](../repertoire_score/model.py), [uncertainty.py](../repertoire_score/uncertainty.py) |
| Backward values, forward probability, entry baselines, depth and chapter transitions | [evaluate.py](../repertoire_score/evaluate.py), [baseline.py](../repertoire_score/baseline.py), [depth.py](../repertoire_score/depth.py), [transitions.py](../repertoire_score/transitions.py) |
| Cache-only empirical traversal, stopping ledger, depth distribution and entry-route examples | [preparation.py](../repertoire_score/preparation.py), [insights.py](../repertoire_score/insights.py) |
| Position reach, first gaps, reuse, reply variety, WDL and recursive branch spread | [character.py](../repertoire_score/character.py), [gaps.py](../repertoire_score/gaps.py), [sharpness.py](../repertoire_score/sharpness.py), [spread.py](../repertoire_score/spread.py) |
| Gain/drag comparisons, ratings, opening flows and source attribution | [vulnerabilities.py](../repertoire_score/vulnerabilities.py), [ratings.py](../repertoire_score/ratings.py), [openings.py](../repertoire_score/openings.py), [attribution.py](../repertoire_score/attribution.py) |
| Saved gain intervals and spread/entry presentation data | [report_insights.py](../repertoire_score/report_insights.py) |
| Correlations | [position_correlations.py](../repertoire_score/position_correlations.py), [rating_correlations.py](../repertoire_score/rating_correlations.py), with shared helpers in [stats.py](../repertoire_score/stats.py) |
| Summary, full report, chapter and opening pages, exit points, Lichess links, cross-page link resolution and nested contents | [consolidated.py](../repertoire_score/consolidated.py), [render.py](../repertoire_score/render.py), [layout.py](../repertoire_score/layout.py). `report.py` supplies core score/event helpers and legacy rendering utilities. |
| Lichess study export | [studies.py](../repertoire_score/studies.py) |
| Incremental stage orchestration | [build.py](../repertoire_score/build.py) |

Numerical analyses produce saved JSON; `consolidated.py` combines matching saved results and does not rerun estimates. Keep network access out of the renderer and cache-only metrics.

The complete build follows this order:

`scores -> vulnerabilities -> preparation -> character -> ratings -> openings -> report insights -> both correlations -> consolidated rendering`

Ratings depend on preparation, character and vulnerabilities. Report insights depend on preparation, character, vulnerabilities and openings. Companion manifests contain source-score hashes and, where applicable, supporting-analysis hashes and cache provenance. A changed companion may require rebuilding its dependents even when the headline score is unchanged. Strict rendering checks provenance; do not edit hashes or numerical JSON by hand to bypass a mismatch. Standalone commands can render an intermediate report with pending analyses; the batch defers rendering until all stages succeed.

## Model and report conventions

- Canonical board identity includes pieces, turn, castling rights and legal en passant, excluding move counters. Merge exact transpositions; displayed move sequences are representative routes, not exclusive historical line frequencies.
- Get move text, resulting positions, side to move and terminal results from the cached helpers in [board_cache.py](../repertoire_score/board_cache.py) (`children`, `san`, `move_text`, `turn`, `owner_outcome`) instead of building a `chess.Board` inside loops; boards are the main cost in every stage. Saved analysis JSON is written compactly with `layout.data_json`.
- Overall own-move conflicts use explicit overrides, then first PGN variation and earlier chapter order. Each chapter comparison prefers its own first moves and retains compatible merged continuations. Keep every alternative chapter visible and distinguish its conditional comparison from the selected overall policy.
- Our selected moves do not inherit their database popularity. Opponent frequencies retain deviations and valid residual mass. Expand cached replies even after the last recorded PGN move; immediate transpositions into known preparation continue. Unknown continuations stop rather than searching for later re-entry.
- Unprepared replies use the cached parent move row's score, counts and rating. Prepared scores use recursive continuation values. Comparison reports screen replies from those parent rows without requesting candidate child positions.
- Chapter reach is first entry anywhere in its region, across all routes. Score, entry baseline, depth and chapter metrics use the same normalized first-entry mixture and policy. Overlapping positions, chapters, opening categories and gain comparisons are not additive.
- Missing or zero evidence remains unresolved; API failure is not successful zero-data evidence. Sparse filtering applies to strengths and vulnerabilities, not to probability conservation or the repertoire score. Reuse shared samples at canonical transpositions.
- Scores and CP favor the repertoire owner for both colors. Main tables use empirical **Repertoire score**; outcome volatility retains the `sharpness` JSON field while score tables use recursive branch spread. CP appears only beside headline deltas. Opponent ratings describe chapters or lines, never a repertoire-wide average or a score adjustment.
- The summary leads each color with five exit points and five own moves to review, plus opening names alongside lines. The position tree, chapter comparisons, costly replies, strongest moves and preparation metrics are expandable. Separate opening rankings, repeat-gap-share tables and position-contribution tables remain in the full report; chapter detail lives on chapter pages.
- Pages link to each other with in-page anchors and `@page` placeholders that `generate` resolves to relative paths, so a section can move between pages without broken links. Caveats belong in the glossary (`def-*` anchors); tables link to them rather than repeating paragraphs.

The evaluator has no network dependency. `insights.py` implements traversal-derived metrics; `report_insights.py` prepares saved gain intervals and spread/entry presentation data. Keep these responsibilities distinct.

## Testing

Sanity checks enforce conservation at every evaluated node and reproduce the root value from weighted stopping contributions. Tests cover forced moves, beneficial and harmful deviations relative to an explicit leaf baseline, duplicate chapters, shared leaves, transpositions, own-move conflicts, sparse and missing evidence, residual buckets, inconsistent responses, first-entry weighting, cycles, color reversal and fixed-seed reproducibility.

```sh
uv run --no-sync pytest -q
uv run --no-sync ruff check
```

GitHub Actions runs both on Ubuntu and Windows for every push to `main` and every pull request. Shared pytest fixtures live in `tests/conftest.py`. Tests use synthetic data and cache fixtures; a live token is not required. On Windows, if pytest cannot write to the default temp folder:

```powershell
$env:TMP = Join-Path $PWD '.cache/tmp'
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Path $env:TMP -Force | Out-Null
```

For a focused change, choose the relevant checks:

| Change | Useful test command |
| --- | --- |
| Presentation and summary | `uv run --no-sync pytest tests/test_consolidated.py tests/test_report_insights.py -q` |
| Traversal, reach, chapters, depth or uncertainty | `uv run --no-sync pytest tests/test_model.py tests/test_uncertainty.py tests/test_endpoint_traversal.py tests/test_regions.py tests/test_chapter_policies.py tests/test_depth.py tests/test_gaps.py -q` |
| Explorer or incremental reuse | `uv run --no-sync pytest tests/test_explorer.py tests/test_build.py -q` |
| Broad numerical or dependency changes | `uv run --no-sync pytest -q` |

Before publishing regenerated reports, check probability conservation, saved sanity checks, matching source and evidence hashes, legal representative lines, table columns and links. The full report's contents are generated from headings and anchors.

For presentation-only changes, confirm that score and companion JSON hashes remain unchanged and that no requests were made. If only the summary changed, the full report should remain unchanged too. Keep the five-row summary limits and sparse filters intact. The presentation tests check that every relative link on every generated page reaches an existing file and anchor. Check the final diff with `git diff --check`; documentation-only edits do not need a scoring run.

The supplied PGN source files are never modified.

## Design notes

[design.md](design.md) describes the statistical foundations: the probability model, evidence handling, uncertainty, chapter semantics and validation invariants.
