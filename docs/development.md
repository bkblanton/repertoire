# Development

How the code is organized, the conventions it follows, and how it is tested. The statistical model is described in [Model](design.md).

## Setup and checks

```sh
uv sync --locked
uv run pytest -q
uv run ruff check
uv run ruff format --check
uv run mypy
```

`uv run ruff format` applies the formatting; it keeps each string's existing quote style. mypy requires every function to be annotated. The saved score file is described by TypedDicts in `repertoire/schema.py`; companion analyses and the rows the report renders are plain JSON objects, typed as `JsonObject`, and a loaded color is a `report.bundle.Bundle`. GitHub Actions runs all four checks on Ubuntu and Windows for every push to `main` and every pull request. Tests use synthetic PGNs and Explorer tables, so they need no token or network access.

`.git-blame-ignore-revs` lists the commit that reformatted the code base. GitHub skips it in blame views; locally, run `git config blame.ignoreRevsFile .git-blame-ignore-revs`.

## Code map

All code is in the `repertoire` package.

| Area | Modules |
| --- | --- |
| Command line | `cli.py` dispatches each `repertoire` command to its module's `main`. |
| Build | `build.py` runs the stages incrementally; `fetch.py` fetches every table a build needs before any stage runs; `studies.py` exports Lichess studies. |
| PGN and positions | `graph.py` parses PGNs into the merged position graph and finds automatic chapter entries; `board_cache.py` caches move text, child positions and results. |
| Evidence | `explorer.py` fetches, validates and caches Explorer tables. `evals.py` imports Lichess cloud evaluations from the downloaded export file into a local SQLite store; it makes no requests. |
| Scoring | `score.py` plans and scores one color; `model.py` turns evidence into probabilities; `evaluate.py` traverses the graph; `uncertainty.py` computes posterior means, variances and intervals; `baseline.py`, `depth.py` and `transitions.py` compute entry baselines, prepared depth and chapter transitions. |
| Analysis stages | `vulnerabilities.py`, `preparation.py` (with `routes.py`), `character.py` (with `gaps.py`, `sharpness.py` and `spread.py`), `ratings.py`, `openings.py` (with `opening_names.py`), `report_insights.py`, `position_correlations.py` and `rating_correlations.py`, sharing `context.py` for loading saved scores and `stats.py` for weighted statistics. |
| Attribution | `attribution.py` links each line to the chapters that contain it. |
| Comparisons | `alternatives.py` finds decision points and builds hypothetical repertoires; `compare.py` is the `compare` command. |
| Reports | The `report` package renders every page from saved JSON: `bundle` loads matching analyses, `derive` and `tables` build rows, `format`, `markdown` and `links` format numbers, tables and lines, `sections` and `pages` lay out the pages, `generate` writes them and resolves links between them, and `comparison` renders comparison pages. `definitions.md` is the glossary. `render.py` is the `report` command. |
| Saved data | `schema.py` describes the saved score JSON as TypedDicts; `status.py` holds status values; `ledger.py` builds the end-event ledger saved with each score; `layout.py` decides where outputs go and writes JSON. |
| Data | `data/chess-openings/` is the bundled opening-name dataset (see its README for updating it). |

## Build pipeline

A full build runs these stages in order:

`export → fetch → scores → vulnerabilities → preparation → character → ratings → openings → insights → correlations → rating correlations → saved comparisons → render`

Only `export` and `fetch` use the network. Every later stage reads the cache and saved JSON, so it can be rerun offline. Ratings depend on vulnerabilities, preparation and character; insights also depend on openings; the correlations depend on vulnerabilities and ratings.

Each stage saves a manifest with the hashes of its inputs: the source PGN, the score it builds on, the analyses it depends on and the cache entries it read. `build` reruns a stage only when one of these, the code or the settings changed, and checkpoints progress in `reports/data/.build-state.json`. The renderer refuses to combine analyses whose hashes do not match. Never edit hashes or numerical JSON by hand to get past a mismatch.

## Conventions

**One evaluation engine.** `model.prepare_node` builds each position's branches (every legal reply in sorted UCI order, then any games with no listed move) and is the only place evidence becomes probabilities. `evaluate.py` traverses those nodes:

- `fold` computes values from the leaves up: `backward` for scores, and the recursive WDL in `sharpness.py` and spread in `spread.py`.
- `forward` propagates probability to end events and first entries.
- `reaches` gives the probability arriving at each position.
- `best_routes` gives representative lines. Exact ties go to the earlier root, then earlier moves, so labels never depend on traversal order.

`preparation.Evaluator` is a lazy view over the same nodes for the analysis stages. Get one from `preparation.Evaluators`, which shares an evaluator between chapters with the same moves. Add new traversals to the engine instead of copying a propagation loop into an analysis module.

**Positions and moves.**

- A position is identified by its pieces, side to move, castling rights and legal en passant square, without move counters. Transpositions are merged, and displayed lines are representative routes.
- Get move text, child positions, side to move and results from `board_cache.py` (`children`, `san`, `move_text`, `route_line`, `turn`, `owner_outcome`) rather than building a `chess.Board` inside a loop; board construction is the main cost in every stage.

**Move selection.** Explicit policy overrides win. Otherwise `evaluate.select_alternatives` plays the highest-scoring of competing chapter moves, and the first recorded move elsewhere. The score stage saves the winners and `context.selected_policy` hands them to every later stage, so never resolve the overall policy from the configuration alone. Each chapter is evaluated with its own first moves and the overall policy elsewhere; keep alternative chapters visible and label them as such.

**Evidence.**

- Your moves never inherit their database popularity. Opponent reply probabilities keep deviations and games with no listed move.
- Replies are expanded after the last recorded move, and a reply that transposes straight into preparation continues it. Unknown positions end preparation; there is no search for a later return.
- Unprepared replies use the parent table's move row. Analyses never fetch the tables of unprepared positions.
- Positions where you have a prepared move have no table. Their database games are the opponent move rows leading to them (`model.position_counts`), and a move's database score is the table after it. Analyses read only the tables the score used (`AnalysisContext.read_evidence`), so leftover cache entries never change a result.
- Missing evidence stays unresolved; a failed request is never zero data. Sparse filtering applies to strength and vulnerability rankings, never to probabilities or the score.

**Reports.**

- Numerical results belong in analysis JSON. The `report` package only reads saved results: it never queries Lichess or recomputes an estimate. Change the generators rather than editing generated Markdown.
- Pages link to each other through in-page anchors and `@page` placeholders that `generate` resolves to relative paths, so sections can move between pages without breaking links.
- Caveats belong in the glossary (`def-*` anchors in `report/definitions.md`); tables link to them instead of repeating paragraphs.
- Scores and CP favor the repertoire owner for both colors. Opponent ratings describe chapters or lines; they never form a repertoire-wide average or adjust a score.
- Saved JSON is written compactly with `layout.data_json`.
- Documentation and report text avoid em dashes.

## Tests

Shared helpers for small PGN graphs, Explorer tables and cached runs are in `tests/helpers.py`, and pytest fixtures in `tests/conftest.py`. Sanity checks enforce probability conservation at every evaluated position and reproduce the root value from the end events; the full list of covered cases is under [Validation](design.md#validation).

For a focused change, run the relevant tests:

| Change | Tests |
| --- | --- |
| Reports and summary | `uv run pytest tests/test_consolidated.py tests/test_report_insights.py -q` |
| Traversal, reach, chapters, depth or uncertainty | `uv run pytest tests/test_model.py tests/test_uncertainty.py tests/test_endpoint_traversal.py tests/test_entries.py tests/test_chapter_policies.py tests/test_depth.py tests/test_gaps.py -q` |
| Explorer access or incremental builds | `uv run pytest tests/test_explorer.py tests/test_build.py -q` |
| Anything broader | `uv run pytest -q` |

The report tests check that every relative link on every generated page reaches an existing file and anchor.

### Checking that results are unchanged

A refactor or performance change should leave every score and table the same. With a populated cache, build into a scratch directory before and after the change and compare the JSON, ignoring timestamps, paths and hashes:

```sh
uv run repertoire build --offline --force --directory path/to/scratch/data
```

For a presentation-only change, the score and analysis JSON should be byte-identical, and only the Markdown should differ.
