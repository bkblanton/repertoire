# Repertoire score

Python 3.12+ application implementing `design.md`, managed with uv. It estimates a population-based expected chess score under a fixed repertoire policy. It is not an individual performance prediction or an engine evaluation.

## Run

```powershell
uv sync
$env:LICHESS_TOKEN = (Get-Content -LiteralPath 'C:/path/to/lichess_token.txt' -Raw).Trim()
uv run repertoire-score inspect 'C:/path/to/repertoire.pgn' --color white --output reports/inspection
uv run repertoire-score run 'C:/path/to/repertoire.pgn' --color white --config configs/white.json --output reports/white
Remove-Item Env:LICHESS_TOKEN
```

The token is read only from `LICHESS_TOKEN`; it is never written to reports or cache files. Network errors abort the run and do not become zero-game evidence. Cache files permit resuming a run. Cached responses do not expire automatically; use `--refresh` for a fresh database snapshot. To recompute without network or a token:

```powershell
uv run repertoire-score run 'C:/path/to/repertoire.pgn' --color white --config configs/white.json --output reports/white --offline
uv run pytest -q
```

If Windows prevents uv from accessing its default cache or managed Python folder, set `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` to writable directories before running uv. A compatible installed interpreter can be specified with `uv sync --python C:/path/to/python.exe`.

## Configuration

All ratings are selected explicitly: 0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500. Speeds are blitz, rapid and classical, with the full supported date range. The API covers indexed rated games, not every game ever played on Lichess. These defaults follow the [official Explorer endpoint specification](https://github.com/lichess-org/api/blob/master/doc/specs/tags/openingexplorer/lichess.yaml).

Configuration is a JSON object:

```json
{
  "policy": {"canonical four-field FEN": "e2e4"},
  "chapter_regions": {
    "chapter-id": {
      "anchors": [{"path": ["e4", "e5", "Nc3", "Nf6", "g3", "Nc6"]}],
      "description": "Quiet System after both black knights develop"
    }
  },
  "filters": {"ratings": "0,1000,1200,1400,1600,1800,2000,2200,2500", "speeds": "blitz,rapid,classical", "since": "1952-01", "until": "3000-12"},
  "exclude": []
}
```

Policy keys are canonical positions: piece placement, turn, castling and legal en passant, without counters. Values are one UCI move or a move-to-weight map summing to one. Own-move conflicts fail until configured. All PGN variations count as repertoire content; annotations are not instructions and do not remove lines.

The supplied configurations use **chapter regions**. Each chapter has explicit subject anchors, accepting canonical FEN strings or path objects containing SAN/UCI moves and an optional `root_fen`. Its region contains those anchors and all descendants through moves recorded in that chapter, including positions shared with other chapters. Repeated introductory moves before the anchors are excluded. Comments are not executed or automatically interpreted as membership rules. The exact subject anchors and region membership are recorded in the report.

Entry probability means **chapter reach probability**: first arrival anywhere in the region before the model stops, counting each modeled game once per chapter. A path may bypass an early anchor and enter at a shared descendant through a later transposition. The program finds possible first arrivals by walking the resolved repertoire from its roots and stopping on region entry, then propagates first-arrival probability mass. It does not simply sum unrestricted position frequencies or discard a late entry because it descends from an earlier one. After entry, scoring follows the complete merged repertoire, including other chapters' continuations. Those continuations do not automatically become members of the source chapter's region.

The Quiet System starts at `1.e4 e5 2.Nc3 Nf6 3.g3 Nc6`, Mieses at `1.e4 e5 2.Nc3 Nf6 3.g3`, and Paulsen at `1.e4 e5 2.Nc3 Nc6 3.g3`. Several other shared opening subjects have explicit earlier anchors; other chapters retain their previous opening anchors and now include shared descendants. These are documented definitions of where preparation becomes relevant, not exclusive opening classifications. Adjust `chapter_regions` to change a subject boundary. Updated PGN descendants are included automatically on rerun; missing anchors fail validation.

For compatibility, `entries` still accepts a map from chapter IDs to exact entry positions or paths. A chapter cannot have both `entries` and `chapter_regions`. Without either, only an unambiguous singleton chapter-specific frontier is inferred; ambiguous entries require configuration. That legacy uniqueness heuristic does not generally capture a system's full reach.

Full reports also include directed **chapter transition probabilities**: conditional on first entering a source chapter, how often does the model reach the destination at or after that point? Shared or simultaneous entry counts. The JSON retains all ordered chapter pairs, including zeros and undefined results. This is different from an unordered intersection, because the destination may have been visited only before the source. Overlapping chapter frequencies and transition rows are not additive. The model does not follow deviations through unknown positions to possible later re-entry.

`exclude` can contain chapter IDs or strings of the form `chapter-id:canonical-position:uci` to omit a branch and its descendants from that chapter. `root_weights` may map canonical root positions to weights summing to one. Disconnected custom-FEN chapters do not receive absolute reach probabilities without a connecting route or explicit root weights. Reachable cycles fail with a position sequence for diagnosis.

## Statistics and missing evidence

Our selected moves have probability one, or configured mixture weights. Opponent moves use their share of all games at that parent, including deviations. Leaves use position results; deviations use parent move-row results. Legal unobserved moves remain in the posterior model. Valid residual results form a separate no-recorded-continuation bucket.

The default stopping-score prior is Dirichlet(0.5, 0.5, 0.5), in owner win/draw/loss order. At opponent nodes its total strength is divided across all legal moves and any observed residual bucket. Joint move-by-result sampling preserves the dependency between move probability and deviation score. The same samples are reused across all paths into a position.

Default Monte Carlo settings are 2,000 simulations and seed 20260928. Change these with `--simulations`, `--seed`, and `--prior W D L`. Reports include sensitivity to symmetric priors of 0.1 and 2 per result category. `--sparse-threshold` defaults to 30 observations. `--tolerance` defaults to a 1 percentage point interval-width target; it is a reporting flag, not a pruning threshold.

Zero-data stopping scores are unresolved in empirical results. A zero-data opponent distribution stops unresolved at that position, without uniform play or parent-score fallback. Reports give conditional bounds [resolved contribution, resolved contribution + unresolved mass]. Sparse sensitivity assigns all flagged stopping events any score from zero to one. These are conditional sensitivity bounds, not credible intervals.

The approximate 95% interval and posterior statistics retained in JSON use explicit prior completion for unresolved scores. They must be read alongside unresolved mass and conservative bounds. They do not account for all dependence from games appearing in multiple position aggregates, selection bias, or population mismatch. Posterior unresolved mass includes prior probability assigned to legal but unobserved moves and may exceed empirical unresolved mass. Main tables label the raw empirical estimate as **Repertoire score** and omit the posterior mean.

## Outputs and implementation

Each one-color report and study description includes only that color's standard starting-position baseline and the overall repertoire score minus that baseline, in percentage points. The combined summary shows both colors and their respective deltas. The reference uses ordinary database play under the same Explorer filters before forcing repertoire moves, with counts and retrieval provenance recorded in JSON. It does not alter the repertoire calculation. JSON retains both color references for compatibility.

Each chapter also includes an empirical entry baseline and the repertoire score minus that baseline in percentage points. Multiple entries use their normalized first-entry probabilities, matching the chapter's empirical repertoire calculation. Database sample counts are not used as mixture weights. JSON retains the component weights, scores, counts and provenance; the table shows one weighted baseline per chapter. Missing entry evidence remains unresolved. A positive difference means higher modeled score, not demonstrated causal improvement or statistical significance.

- `.json`: complete event ledger, overall and chapter scores, sensitivity, sample counts, prior influence, policy diagnostics, and manifest with input hash, filters, cache keys, and retrieval timestamps.
- `.md`: overall score, chapter table, exact entries, largest contributions and uncertainty priorities.
- `.study-description.md`: concise text to paste into a private Lichess study description, with the overall score, starting reference, population and all chapter reach/score/baseline/difference lines. It contains no local paths, technical run details or posterior means. Simple bullets keep it readable in the [Lichess description renderer](https://github.com/lichess-org/lila/blob/master/ui/analyse/src/study/description.ts).
- `summary.md`: combined overall and chapter tables, regenerated from the latest registered result for each color in that output folder. Each analysis run refreshes it automatically, including runs for just one color. The folder's `.report-index.json` records the latest result filename per color; standard `white.json` and `black.json` files are discovered on the first run.
- `.inspection.json`: parsed chapters, conflicts and entry candidates.

All Markdown generation is automated. To change presentation without fetching evidence or recalculating scores, render the saved results:

```powershell
uv run repertoire-report reports/white.json reports/black.json --summary reports/summary.md
```

This command regenerates both detailed reports, both study descriptions and the combined summary. It leaves JSON results unchanged, requires no token and makes no network requests. Normal `repertoire-score run` calls generate these outputs automatically after saving their results, so `run-repertoires.ps1` also refreshes them. Generation produces local files for copying into Lichess; it does not edit the live studies. Before-and-after comparisons and archived reports remain historical artifacts rather than current summary inputs.

`graph.py` parses and resolves the shared graph and constructs chapter regions and their entry frontiers. `explorer.py` handles evidence and caching. `model.py` prepares evidence and samples posteriors. `evaluate.py` performs backward value and forward probability passes. `transitions.py` computes directed chapter transitions. `report.py` renders outputs. The evaluator has no network dependency.

Sanity checks enforce conservation at every evaluated node and reproduce the root value from weighted stopping contributions. Tests cover forced moves, beneficial and harmful deviations relative to an explicit leaf baseline, duplicate chapters, shared leaves, transpositions, own-move conflicts, sparse and missing evidence, residual buckets, inconsistent responses, first-entry weighting, cycles, color reversal and fixed-seed reproducibility.

The supplied PGN source files are never modified. `prefetch.py` can warm the cache before policy selection, but normal runs fetch all required data themselves.
