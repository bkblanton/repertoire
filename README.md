# Repertoire score

Analyze White and Black Lichess study PGNs to see how often your preparation is reached, how it scores, and where the most common gaps remain. The program follows your chosen moves, weights the opponent's replies using Lichess opening statistics, and combines exact-position transpositions across chapters.

It produces a [summary](reports/summary.md) for everyday review, a [full report](reports/report.md) for detailed analysis, and one page per chapter. The score describes database outcomes under a fixed repertoire policy; it is not an engine evaluation or a prediction of your personal rating gain.

The application uses Python 3.12+ and [uv](https://docs.astral.sh/uv/). [design.md](design.md) explains the statistical foundations and some earlier presentation plans. This README and the tests describe the current behavior.

## Contents

- [Getting started](#getting-started)
  - [Analyze other PGN files](#analyze-other-pgn-files)
- [Reading the reports](#reading-the-reports)
- [Updating results](#updating-results)
- [Files and cache](#files-and-cache)
- [Configuration](#configuration)
  - [Move selection](#move-selection)
  - [Chapter subjects and transpositions](#chapter-subjects-and-transpositions)
- [How the score is calculated](#how-the-score-is-calculated)
- [Metric reference](#metric-reference)
  - [Scores, baselines and conversions](#scores-baselines-and-conversions)
  - [Prepared depth](#prepared-depth)
  - [Equivalent gap reach](#equivalent-gap-reach)
  - [Where preparation ends](#where-preparation-ends)
  - [Score spread and outcome volatility](#score-spread-and-outcome-volatility)
  - [Reuse, reply variety and position profiles](#reuse-reply-variety-and-position-profiles)
  - [Opponent ratings](#opponent-ratings)
- [Analysis commands](#analysis-commands)
  - [Preparation and entry routes](#preparation-and-entry-routes)
  - [Repertoire character](#repertoire-character)
  - [Strengths and vulnerabilities](#strengths-and-vulnerabilities)
  - [Rating contexts](#rating-contexts)
  - [Opening names and reach](#opening-names-and-reach)
  - [Correlations](#correlations)
  - [Report insights and attribution](#report-insights-and-attribution)
  - [Refreshing saved analyses](#refreshing-saved-analyses)
- [Comparing alternative preparation](#comparing-alternative-preparation)
- [Development](#development)
  - [Implementation map](#implementation-map)
  - [Model and report conventions](#model-and-report-conventions)
  - [Testing](#testing)
  - [Troubleshooting](#troubleshooting)

## Getting started

Run commands from the repository root. The program reads Lichess studies but never edits them, and it never changes PGN files you supply.

### Setup

```powershell
uv sync --locked
```

If Windows restricts uv's default folders, set these before running the setup command:

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.uv-python'
```

You can select an existing compatible interpreter with `uv sync --python C:/path/to/python.exe`. After setup, `uv run --no-sync` uses the installed environment without another dependency synchronization.

### Generate both reports

The studies to analyze are listed in [studies.json](studies.json):

```json
{
  "white": "https://lichess.org/study/abcd1234",
  "black": "https://lichess.org/study/mnop3456"
}
```

A study URL, a chapter URL (the whole study is exported) or a bare 8-character study ID all work. One command exports both studies and rebuilds every report:

```powershell
./run-repertoires.ps1 -TokenFile 'path/to/lichess_token.txt'
```

The token needs the `study:read` scope to export private studies; the same token authenticates Explorer requests. The wrapper reads it into `LICHESS_TOKEN` for the run and restores the previous environment afterward. The Python program reads credentials only from that variable. Tokens are never written to reports, exports or cache files.

The export step writes `studies/white.pgn` and `studies/black.pgn` with all chapters, variations and comments. Each download is validated by parsing it before the previous export is replaced, and a file is left untouched when only its `Date` headers changed, so an unchanged study reuses every saved stage. The exports are tracked in Git, so `git diff studies/` shows what changed in your preparation since the last commit.

The build then scores both colors, generates the supporting analyses, and writes `reports/report.md`, `reports/summary.md`, and the chapter and opening pages under `reports/chapters/` and `reports/openings/`. Existing Explorer responses are reused; new or extended lines require only the missing tables. Network failures abort the build and retain completed work for a later retry.

Without the wrapper, the same run is:

```powershell
$env:LICHESS_TOKEN = (Get-Content -LiteralPath 'C:/path/to/lichess_token.txt' -Raw).Trim()
uv run repertoire-build
```

`uv run repertoire-fetch` exports the studies without building. `repertoire-build --no-fetch` (wrapper `-NoFetch`) builds from the last export without contacting the study API. `--sources` and `--studies` select a different study list and export folder.

Commands later in this README refer to the exported files:

```powershell
$whiteStudy = 'studies/white.pgn'
$blackStudy = 'studies/black.pgn'
```

### Analyze other PGN files

Pass two PGN paths to analyze files you already have instead of the configured studies; nothing is downloaded:

```powershell
./run-repertoires.ps1 -WhitePgn 'C:/path/to/white.pgn' -BlackPgn 'C:/path/to/black.pgn' -TokenFile 'C:/path/to/lichess_token.txt'
uv run repertoire-build 'C:/path/to/white.pgn' 'C:/path/to/black.pgn' --offline
```

Both or neither path must be given. Supported inputs:

- **Multi-chapter study exports**, such as a study downloaded from the Lichess UI. Each PGN game is a chapter; `ChapterURL` headers supply the chapter IDs used by the configuration files.
- **A plain PGN** with one or more games and no study headers. Chapters are numbered `1`, `2`, ... in file order, and the variations are the repertoire.
- **A PGN with an entry point**: a game with `SetUp`/`FEN` headers starts from that position. If the position is reachable from other chapters, it joins the merged repertoire there. A disconnected custom-FEN chapter is scored conditionally on reaching its root, unless `root_weights` assigns the roots weights (see [Other options](#other-options)). `entries` or `chapter_regions` in a configuration file can also place a chapter's entry at a later position.

The configuration files are keyed by the chapter IDs of the configured studies. With a different PGN, pass a matching configuration via `--white-config`/`--black-config`, or start without overrides; `repertoire-score inspect` lists the chapter IDs and entry candidates. Analyzing a different PGN replaces the saved results in `reports/data/` and the generated reports, so pass `--directory` to keep separate results.

### Work offline

If all required positions are already cached:

```powershell
uv run repertoire-build --offline
```

Offline mode needs no token and makes no requests, so it builds from the last study exports in `studies/`. A required cache miss stops the build with a diagnostic; it is not treated as a position with zero games.

### Inspect or score one repertoire

Inspection shows chapter IDs, move conflicts and entry candidates without querying Lichess:

```powershell
uv run repertoire-score inspect $whiteStudy --color white --config configs/white.json --output reports/data/inspection
```

To score only White using cached evidence:

```powershell
uv run repertoire-score run $whiteStudy --color white --config configs/white.json --output reports/data/white --offline
```

For Black, use `$blackStudy`, `--color black`, `configs/black.json` and the `reports/data/black` output prefix. A score-only run can leave additional report sections pending; use the complete build when regenerating all reports.

## Reading the reports

Start with [summary.md](reports/summary.md). After the headline scores, each color leads with what to work on:

- **Where preparation ends** groups every unprepared reply by the last prepared position before it, so one study task is one row. *Games leaving prep here* is the share of all games with that color whose preparation ends at that position; *share of games at this position* separates a chapter that simply stops (100%) from rare sidelines at a busy position. Unlike other rankings, these rows do not overlap.
- **Own moves to review** ranks selected moves by move reach times drag against the parent database score.

Collapsed sections follow: the most common positions as a nested tree, chapter comparisons, costly unprepared replies, strongest moves, and preparation and variability. Evidence and definitions appear at the end. Changed-source and missing-analysis notices stay visible above the scores. The summary uses one decimal place and compact game counts.

[report.md](reports/report.md) is the index for detailed analysis: per-color chapter tables, exit points, positions, openings, vulnerabilities, strengths, gap priorities, depth distributions, correlations, and a glossary (`Definitions and evidence`) with one anchor per metric. Tables state their main caveat in a sentence and link to the glossary entry instead of repeating it. Each chapter has its own page in `reports/chapters/` (W1, W2, ... and B1, B2, ...) with its exit points, positions, vulnerabilities, strengths, gaps, depth, entry positions and routes, plus links to the previous and next chapter and to the Lichess study chapter. Per-opening entry evidence is in `reports/openings/white.md` and `reports/openings/black.md`.

| Column | How to read it |
| --- | --- |
| Repertoire score | Expected points from the repertoire owner's perspective: a win is 1, a draw is 0.5, a loss is 0. |
| Baseline / delta | Ordinary database score at the start or chapter entry, and the repertoire score minus that reference. |
| Position reach | Probability of encountering a canonical position before preparation stops, including transpositions. Chapter pages use games after chapter entry. |
| Move reach | Probability of reaching the parent position and then playing that move. It can be lower than the reach of the resulting position when other routes transpose into it. |
| Games leaving prep here | Probability that preparation ends right after that prepared position, combining all of its unprepared replies. |
| Gap reach | Probability of first reaching a position without a prepared reply. It equals that unanswered position's reach. |
| Score spread | How much continuation scores vary across the branches, including later replies. Unprepared replies end preparation, so their tables omit it. |
| Per 1,000 games | Reach-weighted drag, gain, or contribution expressed as score points per 1,000 games with that color (or entering the chapter). |
| Games | Games observed at that position or in the cached parent move row; this is not the sample size of the entire continuation score. |
| Opponent rating | A local or chapter-level description of the database cohort, not a rating adjustment to the score. |
| Chapter / opening source | Where the preparation was recorded and the most common opening label carried into the position. The share appears only when other labels also contribute. |

A displayed line is a legal example route to the position, not an exclusive historical sequence. Each exact board uses one representative overall-policy route as its label in every table; move rows add their move to the parent board's label. Line links open the Lichess analysis board at the position where the repertoire owner decides: after the opponent's reply, or before our move. Positions immediately before a guaranteed prepared reply are collapsed into the position after that reply; unanswered positions remain visible. Position rows and chapters can overlap along a game, so their reach and contributions must not be added.

Tables show percentages only; centipawn equivalents appear beside headline deltas (overview rows and chapter headlines). Differences such as `+5.00%` mean five percentage points, not a relative percentage increase. Columns that are identical in every row, such as all-zero depth endings, are omitted. The [metric reference](#metric-reference) explains the formulas and uncertainty.

### Display options

All Markdown generation is automated. To render matching saved results without recalculating scores or fetching data:

```powershell
uv run repertoire-report reports/data/white.json reports/data/black.json --require-complete
```

`--output` and `--summary` select destinations; chapter and opening pages go in `chapters/` and `openings/` beside the full report, and pages for chapters or colors that are no longer rendered are removed. `--top` (default 10) controls overall rankings, `--chapter-top` (default 5) controls chapter rankings (chapter exit tables show at least 10 rows), and `--position-top` (default 20) controls the prepared-position and unprepared-reply tables for each color and chapter. The summary keeps five rows in its exit and review tables and twelve positions in its tree. Filtering happens before the display limit; complete rankings remain in JSON.

Unprepared positions use their cached parent-move outcomes and counts, or cached position outcomes for recorded endpoints. Transposed arrivals are combined using modeled reach. Pooled parent counts can overlap and are marked with a dagger (†). All reached canonical boards and exact FENs remain available in the character JSON.

## Updating results

Choose the workflow that matches what changed. Rebuilding from updated PGNs, rendering existing results, and refreshing database evidence are separate operations.

| What changed | What to run | What it reads |
| --- | --- | --- |
| The Lichess studies | `./run-repertoires.ps1 -TokenFile ...` | Fresh study exports, configuration and cached evidence; fetches missing required tables. |
| Move choices or chapter subjects in the configs | `uv run repertoire-build --offline`, or the authenticated wrapper if tables are missing. | The last study exports, configuration and cached evidence. |
| Report wording, layout or display limits | `uv run repertoire-report reports/data/white.json reports/data/black.json --require-complete` after editing the generator. | Saved JSON only; no Explorer cache reads, score recalculation or requests. |
| Numerical analysis code | The incremental build, offline when possible. | Changed stages and their dependencies; unchanged results are reused. |
| Lichess statistics themselves | An explicit score run with `--refresh`, then the complete build. | New responses replace the requested cache entries. |
| A hypothetical chapter | A separate comparison from a matched entry position. | Matching cached tables, plus any missing required evidence. See [comparisons](#comparing-alternative-preparation). |

The incremental build runs in one Python process and checks PGN contents, configuration, numerical settings, program and dependency versions, relevant cache files, supporting analyses and output contents. Unrelated cache additions do not invalidate results. Successful stages are reused after an interrupted build. Timings and checkpoints are saved in `reports/data/.build-state.json`.

Use `repertoire-build --force` or the wrapper's `-Force` to rebuild every stage using the cache. Cached responses do not expire automatically, and **force does not refresh Lichess statistics**. The batch has no refresh flag. For fresh White evidence, for example:

```powershell
$env:LICHESS_TOKEN = (Get-Content -LiteralPath 'C:/path/to/lichess_token.txt' -Raw).Trim()
try {
    uv run repertoire-score run $whiteStudy --color white --config configs/white.json --output reports/data/white --refresh
} finally {
    Remove-Item Env:LICHESS_TOKEN
}
uv run repertoire-build --offline
```

To refresh Black too, run the corresponding Black score command inside the `try` block before rebuilding. This manual example clears its token variable afterward; the wrapper preserves a previously set token.

Saved results describe a particular PGN and evidence snapshot. A request for current results requires exporting the studies again, which the default build does. Rendering an older result is supported, but its saved-snapshot notice must remain visible.

Companion report hashes, input hashes, colors, and filters must match. Stale analyses cause the explicit rendering command to fail rather than silently mix snapshots. `--require-complete` also requires all companion analysis families and matching future preparation gain correlations. Without it, missing analyses are clearly marked pending. A changed or missing source PGN produces a prominent saved-snapshot notice; rendering an archived result does not pretend it describes the current study. Historical improvements and hypothetical comparisons remain separate because they use different policies or snapshots.

Standalone scoring and analysis commands refresh the reports automatically. During that process, stale companions are omitted and marked pending. The complete build renders once, after all stages succeed, and requires matching analyses. The output registry in `reports/data/.report-index.json` records the latest score filenames; standard `white.json` and `black.json` are discovered on the first run.

## Files and cache

| Location | Purpose |
| --- | --- |
| [studies.json](studies.json) | The White and Black Lichess study URLs exported by the default build. |
| `studies/` | The latest study exports, `white.pgn` and `black.pgn`, tracked in Git. |
| [configs/white.json](configs/white.json), [configs/black.json](configs/black.json) | Maintained policy overrides and chapter subject anchors, keyed by Lichess chapter ID. Preserve explicit choices when importing newer PGNs. |
| `.cache/explorer/` | Persistent raw Explorer responses, keyed by endpoint, canonical board and query filters. Each file wraps `identity`, `retrieved_at`, and `data`; score manifests identify the relevant cache keys. |
| `reports/data/` | Score snapshots, companion JSON, correlation results, `.build-state.json` checkpoints and `.report-index.json` output registration. |
| [reports/report.md](reports/report.md), [reports/summary.md](reports/summary.md) | The current generated full report and summary. |
| `reports/chapters/`, `reports/openings/` | Generated chapter pages (`W1.md`, `B1.md`, ...) and per-color opening evidence pages, linked from the report and summary. |
| `reports/comparisons/`, `reports/positions/` | Separate hypothetical comparisons and focused position reports; these can describe different snapshots. |

Explorer cache and generated analysis JSON are ignored by Git because they can grow very large. Readable Markdown reports remain tracked, so a fresh checkout may include reports without the local evidence needed to regenerate them. Preserve local data when changing Git tracking; clearing the cache is not a routine repair.

Saved score JSON contains the complete event ledger, scores, sensitivity results, sample counts, policy diagnostics and a manifest. Check `manifest.input_path`, `input_sha256`, `configuration`, `filters` and `evidence` before reusing a snapshot. Cache files record `identity`, `retrieved_at` and `data`; manifests identify the relevant keys and timestamps.

Inspection creates `.inspection.json` diagnostics. Custom score output prefixes are supported; a prefix inside a `data` directory renders the two Markdown reports in its parent. Color-specific and analysis-family Markdown are no longer generated. Comparisons and focused position reports remain separate because they may describe different policies or snapshots.

Treat credentials as secrets: keep token files out of Git and never print their contents or include them in artifacts. Change the Python generators rather than hand-editing generated Markdown.

## Configuration

The maintained color settings are [configs/white.json](configs/white.json) and [configs/black.json](configs/black.json). They define move overrides, chapter subjects and Explorer filters.

### Explorer filters

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

### Move selection

Policy keys are canonical positions: piece placement, turn, castling and legal en passant, without counters. Values are one UCI move or a move-to-weight map summing to one. Without an explicit override, own-move conflicts choose the first recorded move: PGN main variation before side variations, and earlier chapters before later chapters. Conflicting alternatives are never averaged or assigned simultaneous probability one. All PGN variations remain available as repertoire content; annotations are not instructions and do not remove lines.

Every chapter remains in the chapter report, including alternatives excluded from the overall policy. For each chapter comparison, its first recorded own moves take precedence and the overall policy applies elsewhere. This retains compatible preparation split across multiple chapters. Score, baseline, expected prepared depth, entry probability, transitions from that chapter, and vulnerabilities all use that same comparison policy. Alternative rows are labeled; a separate overall-policy region reach shows how often the selected overall repertoire enters that region. Shared region reach does not imply that the alternative own move was selected. Reordering chapters changes overall priority without discarding the alternatives' comparisons. Saved JSON records the exact policy overrides.

### Chapter subjects and transpositions

The supplied configurations use **chapter regions**. Each chapter has explicit subject anchors, accepting canonical FEN strings or path objects containing SAN/UCI moves and an optional `root_fen`. Its region contains those anchors and all descendants through moves recorded in that chapter, including positions shared with other chapters. Repeated introductory moves before the anchors are excluded. Comments are not executed or automatically interpreted as membership rules. The exact subject anchors and region membership are recorded in the report.

Entry probability means **chapter reach probability**: first arrival anywhere in the region before the model stops, counting each modeled game once per chapter. A path may bypass an early anchor and enter at a shared descendant through a later transposition. The program finds possible first arrivals by walking the resolved repertoire from its roots and stopping on region entry, then propagates first-arrival probability mass. It does not simply sum unrestricted position frequencies or discard a late entry because it descends from an earlier one. After entry, scoring follows the complete merged repertoire, including other chapters' continuations. Those continuations do not automatically become members of the source chapter's region.

The maintained White configuration includes Vienna subject anchors at `1.e4 e5 2.Nc3 Nf6 3.g3 Nc6`, `1.e4 e5 2.Nc3 Nf6 3.g3`, and `1.e4 e5 2.Nc3 Nc6 3.g3`. Several other shared opening subjects have explicit earlier anchors; chapter-owned shared descendants are included in their regions. These are definitions of where preparation becomes relevant, not exclusive opening classifications. Match the configuration's chapter IDs against the current PGN rather than relying on older chapter names. Adjust `chapter_regions` to change a subject boundary. Updated PGN descendants are included automatically on rerun; missing anchors fail validation.

For compatibility, `entries` still accepts a map from chapter IDs to exact entry positions or paths. A chapter cannot have both `entries` and `chapter_regions`. Without either, all first chapter-unique positions become anchors, with chapter-owned descendants defining the region. If none can be entered under that chapter's policy, the first opponent reply on its mainline (or its PGN root) is used. Explicit subject anchors are preferable for comparing alternatives from a common position, such as `1.e4 e6` for both Advance and Tarrasch French chapters; automatic entries can describe different conditional subtrees.

Full reports also include directed **chapter transition probabilities**: conditional on first entering a source chapter, how often does the model reach the destination at or after that point? Shared or simultaneous entry counts. The JSON retains all ordered chapter pairs, including zeros and undefined results. This is different from an unordered intersection, because the destination may have been visited only before the source. Overlapping chapter frequencies and transition rows are not additive. The model does not follow deviations through unknown positions to possible later re-entry.

### Other options

`exclude` can contain chapter IDs or strings of the form `chapter-id:canonical-position:uci` to omit a branch and its descendants from that chapter. `root_weights` may map canonical root positions to weights summing to one. Disconnected custom-FEN chapters do not receive absolute reach probabilities without a connecting route or explicit root weights. Reachable cycles fail with a position sequence for diagnosis.

## How the score is calculated

Our selected moves have probability one, or configured mixture weights. Opponent moves use their share of all games at that parent, including deviations. This includes cached replies after the last recorded PGN move: replies that immediately reach a known repertoire board resume preparation. An unanswered own-turn leaf uses position results; an unprepared opponent reply uses its parent move-row results without fetching the child. Scores, position reach, prepared depth, chapter entries, and opening sources follow the same transitions. Legal unobserved moves remain in the posterior model. Valid residual results form a separate no-recorded-continuation bucket.

### Sampling and uncertainty

The default stopping-score prior is Dirichlet(0.5, 0.5, 0.5), in owner win/draw/loss order. At opponent nodes its total strength is divided across all legal moves and any observed residual bucket. Joint move-by-result sampling preserves the dependency between move probability and deviation score. The same samples are reused across all paths into a position.

Default Monte Carlo settings are 2,000 simulations and seed 20260928. Change these with `--simulations`, `--seed`, and `--prior W D L`. Reports include sensitivity to symmetric priors of 0.1 and 2 per result category. `--sparse-threshold` defaults to 30 observations. `--tolerance` defaults to a 1 percentage point interval-width target; it is a reporting flag, not a pruning threshold.

Zero-data stopping scores are unresolved in empirical results. A zero-data opponent distribution stops unresolved at that position, without uniform play or parent-score fallback. Reports give conditional bounds [resolved contribution, resolved contribution + unresolved mass]. Sparse sensitivity assigns all flagged stopping events any score from zero to one. These are conditional sensitivity bounds, not credible intervals.

The approximate 95% interval and posterior statistics retained in JSON use explicit prior completion for unresolved scores. They must be read alongside unresolved mass and conservative bounds. They do not account for all dependence from games appearing in multiple position aggregates, selection bias, or population mismatch. Posterior unresolved mass includes prior probability assigned to legal but unobserved moves and may exceed empirical unresolved mass. Main tables label the raw empirical estimate as **Repertoire score** and omit the posterior mean.

## Metric reference

All scores favor the repertoire owner, for both White and Black. Whole-repertoire metrics start at the repertoire root; chapter metrics use that chapter's normalized first-entry mixture and comparison policy.

### Scores, baselines and conversions

Each color's section includes only that color's standard starting-position baseline. The overview and summary show both colors and their respective deltas, in percentage points. Both also include a combined row with 50% weight for White and 50% for Black, provided both use matching Explorer filters and standard starting positions. The row averages scores, starting baselines and prepared depths; chapter counts are summed across colors. Each overview row includes an **Elo equivalent**: `400 * log10(score / (1 - score)) - 400 * log10(baseline / (1 - baseline))`. This translates the modeled score edge to an Elo scale; it is not a measured rating gain. One-color reports omit the combined row. The reference uses ordinary database play under the same Explorer filters before forcing repertoire moves, with counts and retrieval provenance recorded in JSON. It does not alter the repertoire calculation. JSON retains both color references for compatibility.

Headline deltas (overview rows and chapter headlines) also show a whole-number **centipawn equivalent (CP)**; every other table shows percentages only, so three units for one quantity do not crowd the tables. Use `C(p) = ln(p / (1 - p)) / 0.00368208`, the inverse [Lichess score curve](https://lichess.org/page/accuracy). For example, Black's 52.50% score is about +27 cp from Black's perspective. **CP delta** is `C(after) - C(before)`, or score CP minus entry/starting-baseline CP. Negative changes indicate a worse score. Scores include half a point for draws. These are human-results conversions rather than engine evaluations. Convert weighted mixtures after averaging their scores, and score intervals by converting both endpoints. Reach, frequency, and contribution percentages are not expected scores. Missing scores remain unresolved and conversion at 0% or 100% is unavailable.

Each chapter also includes an empirical entry baseline and the repertoire score minus that baseline in percentage points. Reports display score deltas, gains, drag, and weighted contributions with `%` in the value rather than `pp` in the heading. For example, 55% minus 50% displays as `+5.00%`, a five percentage point difference rather than a relative change. JSON retains the existing percentage-point units and field names. Multiple entries use their normalized first-entry probabilities, matching the chapter's empirical repertoire calculation. Database sample counts are not used as mixture weights. JSON retains the component weights, scores, counts and provenance; the table shows one weighted baseline per chapter. Missing entry evidence remains unresolved. A positive difference means higher modeled score, not demonstrated causal improvement or statistical significance.

### Prepared depth

**Expected prepared depth** is the expected number of remaining repertoire-owner moves before leaving theory or reaching a theory leaf or terminal outcome. Each selected own move contributes one; opponent moves contribute no unit themselves and weight subsequent prepared moves by their empirical frequencies. An available own move at the starting position is included, but previous moves and entry itself earn no bonus. There is no cutoff, discount, or tunable coverage parameter. Overall depth starts at the repertoire root; chapter depth uses the same first-entry mixture as its score and baseline and follows the complete merged repertoire thereafter.

The evaluator computes depth backward through the DAG: an own-move node has depth `1 + weighted child depth`, an opponent node has depth `sum(reply probability * child depth)`, and a deviation or leaf has depth zero. Shared prefixes, duplicate lines, and transpositions do not create additional per-game moves. Deeper or broader preparation earns credit according to its probability of being used. Missing leaf outcome counts do not affect depth; missing opponent distributions preserve lower/upper bounds using the finite remaining repertoire, and missing first-entry weights leave chapter depth unresolved. Bounds are conditional on the resolved frequencies, not confidence intervals. JSON stores `prepared_depth` under `overall` and each chapter's `score`; wherever depth is displayed, its unit is own moves. Historical JSON without the metric requires reanalysis.

`repertoire-preparation` automatically saves a **prepared-depth distribution** for each color and chapter. It propagates probability over both canonical board and elapsed own-move count, so paths that transpose into a shared board retain their different remaining-depth histories. The full report shows cumulative probabilities of preparing at least each depth and exact-depth endings at prepared endpoints, unprepared replies, and other stops. The survival probabilities from depth 1 sum to expected prepared depth. The summary links the full distribution and reports its median. Zero-data leaf outcomes do not obscure depth; missing opponent distributions retain finite structural bounds and are labeled unresolved. There is no depth cutoff or discount parameter.

### Equivalent gap reach

**Equivalent gap reach** summarizes recurring first unprepared positions: `R = sqrt(sum(p_i ** 2))`, where `p_i` is the probability of first reaching canonical board `i` with no prepared own reply. Combine all exact transpositions into one gap before squaring. `R ** 2` is the probability that two independent modeled games first encounter the same gap; `R` is the reach of one gap with the same repeat probability. Lower values indicate less concentrated or less likely gaps. Probabilities use the full scope, without conditioning on reaching a gap, omitting sparse rows, or imposing a depth cutoff.

The cache-only character analysis follows selected own moves and cached opponent reply frequencies. After a prepared endpoint, it uses that endpoint's cached reply table to identify the first unanswered positions. A reply that transposes into a prepared board continues through the merged repertoire. It never fetches unprepared child tables. An unanswered board's position reach equals its first-gap probability, combining all transposed arrivals; every overall and chapter scope validates this equality. Terminal games produce no gap. Missing or zero response tables, unnamed residuals, and closed canonical cycles retain unresolved probability, with conservative bounds rather than a falsely exact value. The standalone gap calculation can solve cyclic components with exits as absorbing Markov chains; full repertoire scoring still rejects reachable cycles.

Both consolidated files show each color's equivalent gap reach. Chapter tables include **Equivalent gap reach after entry** and **Weighted gap reach contribution**, calculated as chapter entry probability times the conditional value. The conditional distribution uses the same normalized first-entry mixture and comparison policy as the chapter score. These weighted values are on the full-repertoire probability scale, but are not additive: chapters and gap boards can overlap, and equivalent gap reach is nonlinear. Character JSON retains every canonical gap probability, resolved/terminal/unresolved mass, bounds, and conservation checks under `gap_coverage`.

**Gap priorities** show each canonical first gap's share of repeat-gap probability, `p_i ** 2 / sum(p_j ** 2)`. Transposed arrivals are combined before squaring, and displayed rankings retain the full distribution as their denominator. These shares explain which gaps dominate equivalent gap reach. Unknown gap mass is labeled separately.

### Where preparation ends

**Exit points** group the saved stopping ledger by the last prepared board before each stop: the board where the opponent chose an unprepared reply (including database results with no individual move row), or an own-turn board with no recorded move. **Games leaving prep here** is the summed first-gap probability of those stops, and **share of games at this position** divides it by the board's own reach. Each modeled game stops once, so exit rows partition the stopping mass and, unlike position rows, do not overlap; finished games and missing opponent data are excluded. The database score is the reach-weighted mean of the cached parent-row results of those replies. Exit tables appear in the summary, each color's section of the full report, and every chapter page; they need no new queries. A board where many games leave through many rare replies, such as a chapter that ends one move early, appears as one row instead of being split across reply rows of about 1% each.

### Score spread and outcome volatility

**Branch score spread** is now the primary variability metric in score tables. It is computed recursively: `B(s) = sum(p * (B(child) + (score(child) - score(s))**2))`, displayed as `100 * sqrt(B(s))` in percentage points. Known stopping scores have B=0, and forced own moves inherit their continuation. Exact transpositions reuse one continuation. Entry and combined-color mixtures include differences between their mean scores; standard deviations are never averaged. The result matches the complete stopping-event variance. **Reply** beneath a position's spread gives its immediate opponent-reply spread using recursive continuation scores. It is unavailable at own turns, stopping boundaries, or incomplete named reply tables. Sparse evidence stays included; unresolved outcomes remain unavailable. **Prep ends** indicates the model boundary, rather than a certain game result. The full report retains the stopping-board and arrival-cohort breakdown and shows the branch share of total outcome variance. Score spread replaces the previous Sharpness column; before/after spreads share one column, and the summary chapter table has one fewer column.

**Outcome volatility (formerly sharpness):** the existing recursive WDL metric remains on a 0%-100% scale, calculated as `400 * (W + D/4 - (W + D/2)**2)`. It includes both branch-score variance and game-result variation within stopping outcomes. Use the same selected moves, empirical replies, stopping rules and canonical transpositions as repertoire score. Mix WDL before calculating volatility, including first entries and the 50/50 color mixture. Prepared continuations use recursive WDL; unprepared replies use their cached parent rows, and terminal results are exact. Missing results remain unresolved. It is 100% for equal wins and losses, 10% for 5% wins / 90% draws / 5% losses, and zero for a certain result. Mostly decisive database games keep it near 90% to 95% for every chapter, so it no longer appears in headlines; the branch spread breakdowns retain it. The `outcomes.sharpness` JSON field and underlying WDL are preserved.

Score tables show branch score spread; outcome volatility appears only in detailed breakdowns. The `outcomes.sharpness` JSON field remains for compatibility. The two metrics answer different questions: spread describes differences between continuation scores, while volatility also includes variation in final game results within each stopping outcome.

### Reuse, reply variety and position profiles

**Expected reuse:** for a distinct own position/move decision with modeled encounter probability `p`, `N*p` is its expected encounters in N independent games and `1-(1-p)^N` is its probability of being seen at least once. Summing these yields total encounters and distinct decisions encountered; their difference gives repeat encounters. Exact transpositions and duplicate chapter providers share one decision. The curve defaults to 10, 50, 100 and 500 games; customize it with `--games`. Chapter curves count games entering that chapter, not all games. The denominator includes only selected decisions with positive empirical reach. Exposure is not memory retention.

**Reply predictability:** entropy over observed named opponent replies at each active decision, with effective replies `2^H`. The scope summary exponentiates the mean entropy weighted by position reach and the recorded continuation fraction. Unrecorded continuation mass is reported as missing coverage, never invented as another chess move. The accumulated information in bits per game is shown separately from the average per decision. Zero observations produce unavailable predictability. Sparse samples are flagged using the existing scoring threshold without excluding them. Leaves are not queried for further replies.

**Position profiles:** aggregate board features at the boundary of preparation, using prepared leaf boards and the board after each unprepared opponent reply. Show queens, current king wings, bishop pairs, isolated/doubled/passed pawns, isolated d-pawns and exact pawn skeleton frequencies. Effective skeleton count is `2^H` over the weighted skeleton distribution. JSON also includes material, pawn and rook counts and profiles conditional on stopping type. Current king files are not treated as proof of castling history. Unresolved opponent distributions stop at their known board and are flagged; downstream reuse is unknown. These describe preparation boundaries rather than eventual middlegames or personal outcomes.

The consolidated report includes chapter comparison tables and per-chapter details. Its `--top` and `--chapter-top` options control displayed ranking lengths; JSON preserves every row. Descriptive empirical estimates have no sampling confidence intervals. The three measures do not assign a combined quality score or an arbitrary depth discount.

### Opponent ratings

Explorer `averageRating` is the move maker's rating. At our turn, use the previous opponent move's parent-table row. At the opponent's turn, weight the current response rows by game counts, including the Black repertoire's starting-board row where White is to move. The White starting-board row has no preceding opponent move and is labeled n/a. These are local position contexts; neither creates a whole-repertoire rating average. Transposed arrivals use modeled reach rather than database counts. Specific opponent vulnerabilities use the reply row; our selected moves and cached alternatives use the resulting board's opponent response rows. Unprepared child tables are never fetched.

Chapter score-evidence averages weight ratings once at the stopping outcomes. Entry-baseline context instead uses the chapter's first-entry mixture, including the incoming edges under its comparison policy. Local line ratings describe the exact position or move context. Chapter pawn groups can combine stopping evidence; whole-repertoire groups show only the individual example's rating. Missing ratings and residual outcome buckets remain unavailable, and partial coverage is disclosed. These descriptive ratings do not modify scores or rankings. There is no White, Black, combined-study or whole-study baseline rating average.

Opponent reply rows also show **Rating Δ vs parent**: the reply's move-maker average minus the parent's game-weighted opponent response average. Transposed replies combine these paired differences by modeled arrival reach. Missing comparisons remain unavailable; a current-position response average without a specific opponent reply to compare is marked n/a. Partial parent-rating and paired-arrival coverage are disclosed. This describes the reply cohort and does not change any score or ranking.

## Analysis commands

The complete build runs these analyses in dependency order. The commands below are useful when updating one analysis or investigating a result. Unless noted otherwise, they use saved scores and existing cache entries, without a token or Lichess requests. Cache-only analyses that reconstruct the repertoire require the source PGN and evidence to match the saved snapshot.

### Preparation and entry routes

After scoring and vulnerability generation, run:

```powershell
uv run repertoire-preparation reports/data/white.json reports/data/black.json
```

This command is cache-only. It writes `.preparation.json` beside each score result and refreshes the consolidated report and summary. It also runs automatically in `run-repertoires.ps1`. Missing candidate evidence is reported, never fetched silently or treated as a zero score.

The same analysis saves actual **first-entry route examples** under each chapter's comparison policy. It stops every root-to-entry path when it first reaches any chapter-region position, merges all arriving probability at the exact board, and keeps the most likely single route as an example. Entry-position weights include every first-arrival route; the separately displayed example weight covers just that route. Both are conditional on reaching any position in the chapter. Examples are validated as legal and cannot pass an earlier chapter position. These explanations preserve the existing transposition-inclusive chapter scores and reach; ordinary position-table lines remain representative board labels. Older results without these fields explicitly request a cache-only preparation refresh.

### Repertoire character

```powershell
uv run repertoire-character reports/data/white.json reports/data/black.json
```

This cache-only command writes `data/white.character.json` and `data/black.character.json`, then refreshes the consolidated report and summary. It also runs automatically in `run-repertoires.ps1`. No token or API requests are needed. Source PGN hashes and cached scoring evidence must match the saved scores. Overall and chapter scores and expected prepared depths are independently reproduced before writing these metrics. Chapter scopes retain weighted first-entry mixtures and chapter-local policy, including unselected alternatives.

### Strengths and vulnerabilities

Generate overall and chapter rankings from saved scores and cached parent-position tables:

```powershell
uv run repertoire-vulnerabilities reports/data/white.json reports/data/black.json
```

This defaults to **cache-only** and requires no token. If own decision positions were not needed by an earlier score run, add `--fetch-missing` with `LICHESS_TOKEN` set. Only missing own-parent tables are fetched, once per canonical position and filter set. Every candidate reply or alternative is read from its parent's cached move rows. Candidate child endpoints are never requested for screening. Existing score evidence must match the saved report's cache keys and retrieval timestamps; a changed PGN or refreshed evaluation cache requires regenerating scores first.

Outputs are `data/white.vulnerabilities.json` and `data/black.vulnerabilities.json`. The consolidated report has separate tables for unprepared opponent replies, prepared opponent replies, and selected own moves. Opponent replies rank by weighted drag; own moves rank by the direct deficit against the parent database score. Each category is filtered before its display limit. Set displayed ranking lengths with `repertoire-report --top` and `--chapter-top`; JSON always retains all rankings and signed comparisons. `run-repertoires.ps1` generates this analysis after scoring both colors, fetching missing parent tables unless `-Offline` is set. Rendering alone does not recalculate vulnerabilities.

#### Gain and drag

**Opponent reply drag** is `repertoire value before reply - value after reply`, in percentage points. **Weighted drag** multiplies that local drop by `parent reach * reply probability` and determines opponent rankings. Prepared replies use the full merged continuation; deviations use the parent move row's empirical score. **Our move drag** is `ordinary parent database score - repertoire continuation score after our move`, expressed in percentage points without multiplying by reach. The strengths section ranks the opposite difference, `repertoire continuation score - parent database score`. Weighted drag, weighted gain and position contributions are displayed as score points per 1,000 games (ten times the reach-weighted percentage points), so a weighted drag of `0.0317%` reads as `0.32`. Our move's historical popularity is never applied. The benchmarks differ, so the two rankings remain separate. Historical same-table alternatives remain in JSON as separate screening information with their sample sizes; they compare database outcomes and are not substituted for prepared continuation scores. They are not evaluated replacement policies or recommendations, and the maximum observed score can exaggerate sampling noise.

**Own-move gains** are split into the selected move's database score minus its parent's database score, and its prepared continuation score minus the selected move's database score. Both components sum to total gain. The selected move's database evidence comes from the cached parent row. Local gain and drag intervals reuse the saved model's joint move/result sampling and shared transposition values; the parent and selected database scores share one joint table. They are approximate prior-completed model intervals, excluding game overlap and population selection effects. Headline intervals and sparse-evidence sensitivity remain distinct.

Reach sums incoming mass across exact-position transpositions. Each position/move is counted once per ranking. Chapter rankings use the same normalized first-entry mixture and comparison policy as chapter scoring. Opponent weighted drag and all reported chapter reach are conditional on chapter entry. Own drag is a direct score subtraction at its parent. JSON also retains reach-weighted own comparisons and comparisons weighted again by chapter entry probability. For alternative chapters it is counterfactual, not an impact on the selected overall repertoire. A move that enters the chapter belongs to the overall or upstream ranking. Representative lines are legal route labels, not exclusive historical sequence probabilities. Reports retain the owner's overall baseline and delta, and each chapter's weighted entry baseline, score, delta and expected prepared depth for context.

The summary's own-move highlights rank by **move reach times local continuation gain or drag**, after the existing sparse filter. Full-report own-move rankings retain their local comparisons. Every highlighted gain includes the later prepared continuation, so these weighted comparisons overlap and must not be added. Line tables also show **Avg games per encounter**, `1 / reach`, for independent modeled games. In chapter tables this means games that enter the chapter; in overall tables it means games with that color. Zero reach is never encountered under the policy, rather than a finite waiting interval.

#### Position contributions and stopping outcomes

The strengths section appears overall and per chapter. It shows our largest positive continuation-score differences against the parent database score, followed by **all reached prepared positions and unprepared opponent replies** ranked by contribution: reach times score. Prepared positions use their full repertoire continuation score; unprepared replies use cached parent-row outcomes or recorded endpoint outcomes. Exact boards combine all transposed arrivals. Boards immediately before guaranteed own replies and the standard starting board are omitted, as in the common-positions section. Intermediate positions count, so White's `1.e4` carries the entire repertoire score. These contributions overlap along a game and must not be summed; they show score carried through a position, rather than incremental improvement. Chapter reach and contributions are conditional on entry. Sparse and unresolved positions are filtered before display limits. Pooled parent counts may overlap and are marked with a dagger. No scoring or new Explorer queries are needed to render this ranking from the saved character data.

The preparation JSON retains a separate stopping-outcome ledger and baseline-relative contributions for analysis. Each modeled game stops once, so its complete stopping contributions still reproduce the resolved score without repeatedly counting intermediate positions.

#### Sparse evidence and ranking limits

Positive drag highlights below-reference branches. Negative signed changes are preserved in JSON. Nested lines and overlapping chapters must not be summed, and these screening measures do not decompose the overall baseline delta or estimate causal improvement. No-data comparisons remain unresolved rather than becoming zero scores. Strengths and vulnerability tables, and their summary highlights, omit rows flagged sparse in their local, parent, or immediate endpoint evidence. Filtering happens before each display limit. Position contributions exclude missing or sparse local counts; merged unprepared positions are omitted if any parent-row arrival is sparse, even when pooled counts exceed the threshold. JSON retains all rows, and the score and probability models still include all evidence under the existing sparse threshold (normally 30 games). Opponent move counts describe reply frequency; prepared values can depend on different downstream samples. Residual non-move stopping buckets are validated but not ranked as chess moves. Validation reproduces saved overall and chapter scores, checks probability conservation, and checks that signed opponent deviations, including residuals, balance around their parent means.

### Rating contexts

```powershell
uv run repertoire-ratings reports/data/white.json reports/data/black.json
uv run repertoire-report reports/data/white.json reports/data/black.json --require-complete
```

Run ratings after the vulnerability, preparation and character commands. It reads only existing Explorer cache entries, writes `data/white.ratings.json` and `data/black.ratings.json`, and refreshes the same consolidated report and summary. The runner includes this step automatically. Source, score, supporting-analysis and cache hashes identify the evidence. Refresh ratings whenever a supporting analysis changes; strict assembly rejects stale ledgers.

### Opening names and reach

```powershell
uv run repertoire-openings reports/data/white.json reports/data/black.json
```

This cache-only command creates `data/white.openings.json` and `data/black.openings.json` and refreshes the consolidated report and summary. It downloads no opening dataset and never requests unprepared child positions. Only cached Explorer `opening` fields on repertoire boards supply names and ECO codes. A named canonical board gives every arrival its exact cached current name. At an unnamed board, each route retains its last name and probability. For example, a shared unnamed board reached with 1% probability through Alekhine and 20% through Vienna carries those separate masses, rather than counting its entire 21% reach for both openings. Unclassified routes remain unclassified. Structural potential labels from all recorded variations are saved separately and cannot introduce probability from unused alternatives.

Broader family membership recognizes only other cached names matching at colon or comma boundaries, for example Sicilian Defense and Sicilian Defense: Accelerated Dragon. A transition between unrelated names, such as Smith-Morra and Open Sicilian, does not create a parent relation. Unnamed positions with no known name upstream remain unclassified. The saved catalog also identifies names that cannot be reached under the selected policy.

Each color has a full-report opening table containing every reached category. The summary uses opening sources on individual position rows, without a separate opening table. Identical opening names across multiple ECO codes share one category, retaining all cached codes and each board's exact cached label in JSON. Reach is first arrival at an exact cached name or a known more specific named variation. Inheritance cannot introduce a new opening: a route already passed the name's entry. Named transpositions and later named variations still count as entries, including routes bypassing earlier family roots. Multiple entries and returns count once per category. Different categories overlap and their reach cannot be summed. All opening comparisons use the overall selected policy rather than each chapter's alternative policy.

At every reached board, JSON stores current-name probability masses and conditional shares, with unclassified mass kept separately. It also stores the joint reach contributed by games that previously entered each opening, and the contribution divided by total board reach. These historical origin shares remain available after a later exact name changes the current classification. Opening details show the most common positions reached through that opening, alongside total board reach and its share of arrivals. Histories can include several opening categories, so origin shares overlap; current-name shares plus unclassified share form a normalized partition. Opening schema version 2 rejects older results whose structural unions inflated reach.

Line tables with chapter sources also show the **most common opening source**, followed by its share of arrivals. The winner is the last cached name carried by the largest incoming probability, rather than an overlapping historical opening family. An exact cached name replaces earlier labels; unclassified arrivals compete as a separate source. Position rows combine all transposed routes, while move comparisons use only arrivals through their specific parent move. Chapter tables follow that chapter's comparison policy, including alternatives; entry rows use only first arrivals. Equal shares use alphabetical order with named sources before unclassified ones. The opening companion saves these scope-specific sources without querying any additional positions.

Opening repertoire scores, recursive WDL sharpness, entry baselines, deltas, remaining prepared own moves, and equivalent gap reach use the same normalized first-entry weights. Gap distributions are merged by canonical board before taking the square root of the sum of squared probabilities. Unprepared entry boards use the parent response's results and depth zero. Missing outcome or baseline evidence stays unresolved. Scores and deltas include CP equivalents. Expandable details show exact or inherited names, real first-entry examples, combined entry weights, reach, game counts, and evidence sources. A displayed route can represent only part of its board's first-entry mass. Pooled parent counts may contain overlapping historical games and are marked with a dagger. Opening groups have no aggregate opponent rating, preserving the rule that ratings apply only at chapter or line level. The original overall and chapter score files are unchanged.

### Correlations

Both report correlations are reach-weighted. They examine different relationships: future preparation depth versus continuation gain, and opponent rating versus score within the same parent position.

Measure the association between future prepared depth and future preparation gain, weighted by decision reach:

```powershell
uv run repertoire-correlations reports/data/white.json reports/data/black.json
```

This writes `prepared-depth-gain-correlation.json` beside the scores and refreshes the consolidated report. Each canonical selected own move is one observation under the overall policy. Its depth is the expected number of prepared own moves after that move, excluding the selected move itself. Its gain is the recursive continuation score minus the selected move's database score from the cached parent table. Weight is the probability of playing that decision, merged across all transpositions. The main table shows reach-weighted linear and rank correlations and the weighted gain slope. Unweighted results and a positive-depth-only check are available in expandable details.

Approximate 95% model-based intervals propagate the saved Dirichlet evidence model jointly through the graph. Each canonical table is sampled once per draw and reused through shared continuations and transpositions. Depths, scores, baseline scores and reach weights are recomputed together. Defaults are 2,000 draws and seed 20261005; `--simulations` and `--seed` control reproducibility and numerical precision. The analysis uses bounded batches to limit memory. Intervals are withheld when the empirical statistic is undefined or more than 1% of draws are undefined.

These intervals describe finite-database uncertainty for the fixed repertoire under its saved prior. Correlation is nonlinear: uncertainty in continuation scores and the prior can shift the sampled correlations, so a model interval need not contain the observed-count point estimate. Distinct board tables are still treated as independent, even though historical games can overlap. The result describes association, not the causal gain from adding preparation. Sparse, unreachable and unresolved decisions are excluded. This replaces the earlier chapter-level depth/delta analysis in the report; its utility module remains available for legacy saved analyses.

No Lichess requests or token are needed. The PGN files are read only to reconstruct dependencies and must still match the saved input hashes. Correlations run automatically in `run-repertoires.ps1` or through this separate command. Presentation-only `repertoire-report` combines the matching saved correlations without recalculating them.

To analyze the association between opponent rating and score using saved evidence:

```powershell
uv run python -m repertoire_score.rating_correlations reports/data/white.json reports/data/black.json
```

This writes `reports/comparisons/opponent-rating-score.md` and ignored supporting JSON in `reports/data/`. It compares replies within canonical parent boards, using reach weights, recursive prepared scores, cached unprepared reply scores, and parent-bootstrap intervals. It also compares chapter scores and baseline deltas with chapter stopping-evidence opponent ratings, grouping overlapping chapter regions for bootstrap intervals. Sparse replies are excluded, and a 1,000-game sensitivity check is included. These are descriptive cohort associations, not causal rating effects or predictions at a target rating. No Lichess requests are made.

### Report insights and attribution

To add report insights to matching saved analyses without fetching data or changing existing scores:

```powershell
uv run repertoire-insights reports/data/white.json reports/data/black.json
```

This writes `white.insights.json` and `black.insights.json` and refreshes the two Markdown reports. It requires matching preparation, character, vulnerability, and opening analyses, and verifies the source PGN and cached evidence. Supporting hashes prevent old insight calculations from being combined with newer analyses.

All report tables containing individual lines now include linked chapter attribution. A recorded move lists its exact position/move providers, including every shared source. A position lists the chapters containing that canonical board. An unprepared reply is labeled unprepared and lists its parent chapters as context; an unrecorded move that transposes into preparation lists the destination chapters. Representative routes may combine chapters. These relationships are stored separately from the older parent-membership `chapters` fields in JSON.

To refresh attribution in saved score, vulnerability, preparation and character reports without recomputing estimates or querying Lichess:

```powershell
uv run repertoire-attribution reports/data/white.json reports/data/black.json
```

The command validates source hashes and companion-report hashes before writing, preserves all numerical results and updates companion source-report hashes. Normal analysis commands also populate attribution automatically.

### Refreshing saved analyses

These targeted compatibility commands update specific fields in matching saved analyses. They do not bring an older score up to date with an edited PGN. Use the complete build for current study results.

To update own-move comparison definitions from matching saved continuation analysis without parsing a changed PGN or requesting any data:

```powershell
uv run repertoire-vulnerabilities reports/data/white.json reports/data/black.json --refresh-saved
uv run repertoire-report reports/data/white.json reports/data/black.json --require-complete
```

This refresh checks score and character provenance, updates only vulnerability comparisons, and preserves all existing rating contexts while updating their supporting comparison hash. It does not rescore newer source edits; the report explicitly flags changed or missing PGNs. Normal analysis still requires the current PGN to match its saved score snapshot.

To add or refresh only these differences using the already matching rating ledgers and identical cached evidence:

```powershell
uv run repertoire-ratings reports/data/white.json reports/data/black.json --differences-only
```

Normal rating generation also includes these differences automatically. The differences-only refresh can use a saved score snapshot after its PGN changes, because it does not parse or rescore that PGN. The consolidated report flags the snapshot, and matching score, analysis and cache hashes remain required.

To add recursive spreads while preserving saved score estimates and intervals:

```powershell
uv run repertoire-insights reports/data/white.json reports/data/black.json --refresh-spread
```

This reads existing Explorer tables offline and makes no requests.

Each chapter shows its leading first-entry boards and their opening-source mixtures near the headline. Saved report insights retain `opening_summary_groups` for compatibility with earlier summary layouts. Those groups merge names only when their weighted first-entry board distributions match along guaranteed own moves and their scores agree, using the downstream opening's baseline and delta and counting reach once. The current summary shows opening sources alongside positions instead of a separate opening table; full opening categories remain separate. Summary score and CP cells are paired, chapter source ranges are compact, and rankings show up to five rows. To refresh the saved groups from matching analyses, run `uv run repertoire-insights reports/data/white.json reports/data/black.json --refresh-opening-groups`.

## Comparing alternative preparation

First rebuild the actual repertoire from its current PGN. Compare the candidate and actual preparation from the same entry board, for the same color, with matching filters and evidence. Report the conditional score at that entry separately from its full-repertoire reach and impact.

Keep hypothetical results separate from the actual score snapshots and current reports. The comparison tools leave source PGNs unchanged, write hypothetical PGNs under `.cache/`, and put readable results under `reports/comparisons/`.

[scripts/compare_vienna.py](scripts/compare_vienna.py) compares candidate Vienna chapters from `1.e4 e5 2.Nc3` and checks combinations of improving chapter blocks. [scripts/compare_french.py](scripts/compare_french.py) is specifically the Schlechter `4.Bd3` comparison, not a generic French importer. Both expect matching saved actual results and support `--candidate`, `--offline`, and `--token-file`. Read their scenario definitions before adapting them to a different opening; chapter order changes the selected policy. There is no generic candidate-comparison CLI that automatically fits every study.

## Development

Work from the repository root and preserve unrelated local changes. Use uv for Python commands and keep prose free of em dashes. Report presentation belongs in the generators; numerical results belong in analysis JSON. The renderer must not query Lichess or rerun estimates.

### Implementation map

| Responsibility | Files |
| --- | --- |
| PGN parsing, canonical boards, selected moves, chapter regions and first entries | [graph.py](repertoire_score/graph.py), [board_cache.py](repertoire_score/board_cache.py) |
| Authenticated evidence, validation and persistent cache | [explorer.py](repertoire_score/explorer.py) |
| Scoring orchestration and posterior model | [\_\_main\_\_.py](repertoire_score/__main__.py), [model.py](repertoire_score/model.py) |
| Backward values, forward probability, entry baselines, depth and chapter transitions | [evaluate.py](repertoire_score/evaluate.py), [baseline.py](repertoire_score/baseline.py), [depth.py](repertoire_score/depth.py), [transitions.py](repertoire_score/transitions.py) |
| Cache-only empirical traversal, stopping ledger, depth distribution and entry-route examples | [preparation.py](repertoire_score/preparation.py), [insights.py](repertoire_score/insights.py) |
| Position reach, first gaps, reuse, reply variety, WDL and recursive branch spread | [character.py](repertoire_score/character.py), [gaps.py](repertoire_score/gaps.py), [sharpness.py](repertoire_score/sharpness.py), [spread.py](repertoire_score/spread.py) |
| Gain/drag comparisons, ratings, opening flows and source attribution | [vulnerabilities.py](repertoire_score/vulnerabilities.py), [ratings.py](repertoire_score/ratings.py), [openings.py](repertoire_score/openings.py), [attribution.py](repertoire_score/attribution.py) |
| Saved gain intervals and spread/entry presentation data | [report_insights.py](repertoire_score/report_insights.py) |
| Current correlations | [position_correlations.py](repertoire_score/position_correlations.py), [rating_correlations.py](repertoire_score/rating_correlations.py). `correlations.py` retains legacy chapter-level utilities. |
| Summary, full report, chapter and opening pages, exit points, Lichess links, cross-page link resolution and nested contents | [consolidated.py](repertoire_score/consolidated.py), [render.py](repertoire_score/render.py), [layout.py](repertoire_score/layout.py). `report.py` supplies core score/event helpers and legacy rendering utilities. |
| Lichess study export | [studies.py](repertoire_score/studies.py) |
| Incremental stage orchestration | [build.py](repertoire_score/build.py), [run-repertoires.ps1](run-repertoires.ps1) |

Numerical analyses produce saved JSON; `consolidated.py` combines matching saved results and does not rerun estimates. Keep network access out of the renderer and cache-only metrics.

The complete build follows this order:

`scores -> vulnerabilities -> preparation -> character -> ratings -> openings -> report insights -> both correlations -> consolidated rendering`

Ratings depend on preparation, character and vulnerabilities. Report insights depend on preparation, character, vulnerabilities and openings. Companion manifests contain source-score hashes and, where applicable, supporting-analysis hashes and cache provenance. A changed companion may require rebuilding its dependents even when the headline score is unchanged. Strict rendering checks provenance; do not edit hashes or numerical JSON by hand to bypass a mismatch. Standalone commands can render an intermediate report with pending analyses; the batch defers rendering until all stages succeed.

### Model and report conventions

- Canonical board identity includes pieces, turn, castling rights and legal en passant, excluding move counters. Merge exact transpositions; displayed move sequences are representative routes, not exclusive historical line frequencies.
- Overall own-move conflicts use explicit overrides, then first PGN variation and earlier chapter order. Each chapter comparison prefers its own first moves and retains compatible merged continuations. Keep every alternative chapter visible and distinguish its conditional comparison from the selected overall policy.
- Our selected moves do not inherit their database popularity. Opponent frequencies retain deviations and valid residual mass. Expand cached replies even after the last recorded PGN move; immediate transpositions into known preparation continue. Unknown continuations stop rather than searching for later re-entry.
- Unprepared replies use the cached parent move row's score, counts and rating. Prepared scores use recursive continuation values. Comparison reports screen replies from those parent rows without requesting candidate child positions.
- Chapter reach is first entry anywhere in its region, across all routes. Score, entry baseline, depth and chapter metrics use the same normalized first-entry mixture and policy. Overlapping positions, chapters, opening categories and gain comparisons are not additive.
- Missing or zero evidence remains unresolved; API failure is not successful zero-data evidence. Sparse filtering applies to strengths and vulnerabilities, not to probability conservation or the repertoire score. Reuse shared samples at canonical transpositions.
- Scores and CP favor the repertoire owner for both colors. Main tables use empirical **Repertoire score**; outcome volatility retains the `sharpness` JSON field while score tables use recursive branch spread. CP appears only beside headline deltas. Opponent ratings describe chapters or lines, never a repertoire-wide average or a score adjustment.
- The summary leads each color with five exit points and five own moves to review, plus opening names alongside lines. The position tree, chapter comparisons, costly replies, strongest moves and preparation metrics are expandable. Separate opening rankings, repeat-gap-share tables and position-contribution tables remain in the full report; chapter detail lives on chapter pages. Trimming code and subsections have been removed.
- Pages link to each other with in-page anchors and `@page` placeholders that `generate` resolves to relative paths, so a section can move between pages without broken links. Caveats belong in the glossary (`def-*` anchors); tables link to them rather than repeating paragraphs.

The evaluator has no network dependency. `insights.py` implements traversal-derived metrics; `report_insights.py` prepares saved gain intervals and spread/entry presentation data. Keep these responsibilities distinct.

### Testing

Sanity checks enforce conservation at every evaluated node and reproduce the root value from weighted stopping contributions. Tests cover forced moves, beneficial and harmful deviations relative to an explicit leaf baseline, duplicate chapters, shared leaves, transpositions, own-move conflicts, sparse and missing evidence, residual buckets, inconsistent responses, first-entry weighting, cycles, color reversal and fixed-seed reproducibility.

```powershell
uv run --no-sync pytest -q
```

Tests use synthetic data and cache fixtures; a live token is not required. On Windows, if pytest cannot write to the default temp folder:

```powershell
$env:TMP = Join-Path $PWD '.cache/tmp'
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Path $env:TMP -Force | Out-Null
```

For a focused change, choose the relevant checks:

| Change | Useful test command |
| --- | --- |
| Presentation and summary | `uv run --no-sync pytest tests/test_consolidated.py tests/test_report_insights.py -q` |
| Traversal, reach, chapters or depth | `uv run --no-sync pytest tests/test_model.py tests/test_endpoint_traversal.py tests/test_regions.py tests/test_chapter_policies.py tests/test_depth.py tests/test_gaps.py -q` |
| Explorer or incremental reuse | `uv run --no-sync pytest tests/test_explorer.py tests/test_build.py -q` |
| Broad numerical or dependency changes | `uv run --no-sync pytest -q` |

Before publishing regenerated reports, check probability conservation, saved sanity checks, matching source and evidence hashes, legal representative lines, table columns and links. The full report's contents are generated from headings and anchors.

For presentation-only changes, confirm that score and companion JSON hashes remain unchanged and that no requests were made. If only the summary changed, the full report should remain unchanged too. Keep the five-row summary limits and sparse filters intact. The presentation tests check that every relative link on every generated page reaches an existing file and anchor. Check the final diff with `git diff --check`; documentation-only edits do not need a scoring run.

The supplied PGN source files are never modified. `prefetch.py` can warm the cache before policy selection, but normal runs fetch all required data themselves.

### Troubleshooting

| Symptom | What to check |
| --- | --- |
| `Offline cache miss` or missing parent comparison evidence | Cache path, filters and required board; an authenticated run can collect missing required tables. Keep missing data unresolved instead of inventing results. |
| Companion belongs to a different snapshot, or supporting hashes changed | Rebuild the affected analysis and its dependents. [Targeted saved-analysis refreshes](#refreshing-saved-analyses) do not rescore newer PGNs. |
| Scoring or a build stage fails | Read the CLI diagnostic and any `.error.json`. Completed checkpoints and cache remain available; rerun after resolving the error. Existing Markdown can still describe the previous snapshot. |
| Study export fails with HTTP 401, 403 or 404 | Check the URL in `studies.json` and that the token has `study:read`; private studies are visible only to their owner and members. The previous export is kept. |
| Unknown chapter ID or missing configured anchor | Compare the new PGN's inspection with the maintained config; removed or recreated chapters may have different IDs. Update intended subject definitions explicitly. |
| Chapter defining position has less than 100% reach after entry | Inspect first-entry boards and routes: some games may enter through later transpositions and bypass that position. |
| Cache is complete but a batch is slow | Inspect `reports/data/.build-state.json` stage timings and reused/built counts. Changes limited to `consolidated.py` or `render.py` should rebuild only the render stage. |
