# Usage

Run commands from the repository root. The program reads Lichess studies but never edits them, and it never changes PGN files you supply. For metric definitions, see [metrics.md](metrics.md).

## Contents

- [Setup](#setup)
- [Commands](#commands)
- [Generate both reports](#generate-both-reports)
- [Analyze other PGN files](#analyze-other-pgn-files)
- [Work offline](#work-offline)
- [Inspect or score one repertoire](#inspect-or-score-one-repertoire)
- [Display options](#display-options)
- [Updating results](#updating-results)
- [Files and cache](#files-and-cache)
- [Configuration](#configuration)
  - [Explorer filters](#explorer-filters)
  - [Move selection](#move-selection)
  - [Chapter subjects and transpositions](#chapter-subjects-and-transpositions)
  - [Other options](#other-options)
- [Analysis commands](#analysis-commands)
  - [Preparation and entry routes](#preparation-and-entry-routes)
  - [Repertoire character](#repertoire character)
  - [Strengths and vulnerabilities](#strengths-and-vulnerabilities)
  - [Rating contexts](#rating-contexts)
  - [Opening names and reach](#opening-names-and-reach)
  - [Correlations](#correlations)
  - [Report insights and attribution](#report-insights-and-attribution)
- [Comparing alternative preparation](#comparing-alternative-preparation)
- [Troubleshooting](#troubleshooting)

## Setup

```sh
uv sync --locked
```

If Windows restricts uv's default folders, set these in PowerShell before running the setup command:

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.uv-python'
```

You can select an existing compatible interpreter with `uv sync --python C:/path/to/python.exe`. After setup, `uv run --no-sync` uses the installed environment without another dependency synchronization.

## Commands

Everything runs through one command, `repertoire <command>`. `uv run repertoire --help` lists the commands and `uv run repertoire <command> --help` shows each one's options. Most people only need `build`; the others run single stages and are described under [Analysis commands](#analysis-commands).

| Command | Purpose |
| --- | --- |
| `build` | Export the studies, score both colors and write every report. |
| `fetch` | Export the studies without building. |
| `score` | Inspect or score one repertoire PGN. |
| `report` | Render the Markdown reports from saved results, offline. |
| `vulnerabilities`, `preparation`, `character`, `ratings`, `openings`, `insights`, `correlations`, `rating-correlations` | Run one analysis stage from saved results. |

## Generate both reports

The studies to analyze are listed in [studies.json](../studies.json):

```json
{
  "white": "https://lichess.org/study/abcd1234",
  "black": "https://lichess.org/study/mnop3456"
}
```

A study URL, a chapter URL (the whole study is exported) or a bare 8-character study ID all work. One command exports both studies and rebuilds every report:

```sh
uv run repertoire build --token-file path/to/lichess_token.txt
```

Create a personal token at [lichess.org/account/oauth/token](https://lichess.org/account/oauth/token). It needs the `study:read` scope to export private studies; the same token authenticates Explorer requests. Instead of `--token-file`, you can set the `LICHESS_TOKEN` environment variable. The program never writes the token to reports, exports or cache files.

The export step writes `studies/white.pgn` and `studies/black.pgn` with all chapters, variations and comments. Each download is validated by parsing it before the previous export is replaced, and a file is left untouched when only its `Date` headers changed, so an unchanged study reuses every saved stage. The exports are tracked in Git, so `git diff studies/` shows what changed in your preparation since the last commit.

The build then scores both colors, generates the supporting analyses, and writes `reports/report.md`, `reports/summary.md`, and the chapter and opening pages under `reports/chapters/` and `reports/openings/`. Existing Explorer responses are reused; new or extended lines require only the missing tables. Network failures abort the build and retain completed work for a later retry.

`uv run repertoire fetch` exports the studies without building. `repertoire build --no-fetch` builds from the last export without contacting the study API. `--sources` and `--studies` select a different study list and export folder.

## Analyze other PGN files

Pass two PGN paths to analyze files you already have instead of the configured studies; nothing is downloaded:

```sh
uv run repertoire build path/to/white.pgn path/to/black.pgn --token-file path/to/lichess_token.txt
uv run repertoire build path/to/white.pgn path/to/black.pgn --offline
```

Both or neither path must be given. Supported inputs:

- **Multi-chapter study exports**, such as a study downloaded from the Lichess UI. Each PGN game is a chapter; `ChapterURL` headers supply the chapter IDs used by the configuration files.
- **A plain PGN** with one or more games and no study headers. Chapters are numbered `1`, `2`, ... in file order, and the variations are the repertoire.
- **A PGN with an entry point**: a game with `SetUp`/`FEN` headers starts from that position. If the position is reachable from other chapters, it joins the merged repertoire there. A disconnected custom-FEN chapter is scored conditionally on reaching its root, unless `root_weights` assigns the roots weights (see [Other options](#other-options)). `entries` or `chapter_regions` in a configuration file can also place a chapter's entry at a later position.

The configuration files are keyed by the chapter IDs of the configured studies. With a different PGN, pass a matching configuration via `--white-config`/`--black-config`, or start without overrides; `repertoire score inspect` lists the chapter IDs and entry candidates. Analyzing a different PGN replaces the saved results in `reports/data/` and the generated reports, so pass `--directory` to keep separate results.

## Work offline

If all required positions are already cached:

```sh
uv run repertoire build --offline
```

Offline mode needs no token and makes no requests, so it builds from the last study exports in `studies/`. A required cache miss stops the build with a diagnostic; it is not treated as a position with zero games.

## Inspect or score one repertoire

Inspection shows chapter IDs, move conflicts and entry candidates without querying Lichess:

```sh
uv run repertoire score inspect studies/white.pgn --color white --config configs/white.json --output reports/data/inspection
```

To score only White using cached evidence:

```sh
uv run repertoire score run studies/white.pgn --color white --config configs/white.json --output reports/data/white --offline
```

For Black, use `studies/black.pgn`, `--color black`, `configs/black.json` and the `reports/data/black` output prefix. A score-only run can leave additional report sections pending; use the complete build when regenerating all reports.

## Display options

All Markdown generation is automated. To render matching saved results without recalculating scores or fetching data:

```sh
uv run repertoire report reports/data/white.json reports/data/black.json --require-complete
```

`--output` and `--summary` select destinations; chapter and opening pages go in `chapters/` and `openings/` beside the full report, and pages for chapters or colors that are no longer rendered are removed. `--top` (default 10) controls overall rankings, `--chapter-top` (default 5) controls chapter rankings (chapter exit tables show at least 10 rows), and `--position-top` (default 20) controls the prepared-position and unprepared-reply tables for each color and chapter. The summary keeps five rows in its exit and review tables and twelve positions in its tree. Filtering happens before the display limit; complete rankings remain in JSON.

Unprepared positions use their cached parent-move outcomes and counts, or cached position outcomes for recorded endpoints. Transposed arrivals are combined using modeled reach. Pooled parent counts can overlap and are marked with a dagger (†). All reached canonical boards and exact FENs remain available in the character JSON.

## Updating results

Choose the workflow that matches what changed. Rebuilding from updated PGNs, rendering existing results, and refreshing database evidence are separate operations.

| What changed | What to run | What it reads |
| --- | --- | --- |
| The Lichess studies | `uv run repertoire build --token-file ...` | Fresh study exports, configuration and cached evidence; fetches missing required tables. |
| Move choices or chapter subjects in the configs | `uv run repertoire build --offline`, or `--token-file` instead of `--offline` if tables are missing. | The last study exports, configuration and cached evidence. |
| Report wording, layout or display limits | `uv run repertoire report reports/data/white.json reports/data/black.json --require-complete` after editing the generator. | Saved JSON only; no Explorer cache reads, score recalculation or requests. |
| Numerical analysis code | The incremental build, offline when possible. | Changed stages and their dependencies; unchanged results are reused. |
| Lichess statistics themselves | An explicit score run with `--refresh`, then the complete build. | New responses replace the requested cache entries. |
| A hypothetical chapter | A separate comparison from a matched entry position. | Matching cached tables, plus any missing required evidence. See [comparisons](#comparing-alternative-preparation). |

The incremental build runs in one Python process and checks PGN contents, configuration, numerical settings, program and dependency versions, relevant cache files, supporting analyses and output contents. Unrelated cache additions do not invalidate results. Successful stages are reused after an interrupted build. Timings and checkpoints are saved in `reports/data/.build-state.json`.

Use `repertoire build --force` to rebuild every stage using the cache. Cached responses do not expire automatically, and **force does not refresh Lichess statistics**. The batch has no refresh flag. For fresh White evidence, for example:

```sh
uv run repertoire score run studies/white.pgn --color white --config configs/white.json --output reports/data/white --refresh --token-file path/to/lichess_token.txt
uv run repertoire build --offline
```

To refresh Black too, run the corresponding Black score command before rebuilding.

Saved results describe a particular PGN and evidence snapshot. A request for current results requires exporting the studies again, which the default build does. Rendering an older result is supported, but its saved-snapshot notice must remain visible.

Companion report hashes, input hashes, colors, and filters must match. Stale analyses cause the explicit rendering command to fail rather than silently mix snapshots. `--require-complete` also requires all companion analysis families and matching future preparation gain correlations. Without it, missing analyses are clearly marked pending. A changed or missing source PGN produces a prominent saved-snapshot notice; rendering an archived result does not pretend it describes the current study. Historical improvements and hypothetical comparisons remain separate because they use different policies or snapshots.

Standalone scoring and analysis commands refresh the reports automatically. During that process, stale companions are omitted and marked pending. The complete build renders once, after all stages succeed, and requires matching analyses. The output registry in `reports/data/.report-index.json` records the latest score filenames; standard `white.json` and `black.json` are discovered on the first run.

## Files and cache

| Location | Purpose |
| --- | --- |
| [studies.json](../studies.json) | The White and Black Lichess study URLs exported by the default build. |
| `studies/` | The latest study exports, `white.pgn` and `black.pgn`, tracked in Git. |
| [configs/white.json](../configs/white.json), [configs/black.json](../configs/black.json) | Maintained policy overrides and chapter subject anchors, keyed by Lichess chapter ID. Preserve explicit choices when importing newer PGNs. |
| `.cache/explorer/` | Persistent raw Explorer responses, keyed by endpoint, canonical board and query filters. Each file wraps `identity`, `retrieved_at`, and `data`; score manifests identify the relevant cache keys. |
| `reports/data/` | Score snapshots, companion JSON, correlation results, `.build-state.json` checkpoints and `.report-index.json` output registration. |
| [reports/report.md](../reports/report.md), [reports/summary.md](../reports/summary.md) | The current generated full report and summary. |
| `reports/chapters/`, `reports/openings/` | Generated chapter pages (`W1.md`, `B1.md`, ...) and per-color opening evidence pages, linked from the report and summary. |
| `reports/comparisons/`, `reports/positions/` | Separate hypothetical comparisons and focused position reports; these can describe different snapshots. |

Explorer cache and generated analysis JSON are ignored by Git because they can grow very large. Readable Markdown reports remain tracked, so a fresh checkout may include reports without the local evidence needed to regenerate them. Preserve local data when changing Git tracking; clearing the cache is not a routine repair.

Saved score JSON contains the complete event ledger, scores, sensitivity results, sample counts, policy diagnostics and a manifest. Check `manifest.input_path`, `input_sha256`, `configuration`, `filters` and `evidence` before reusing a snapshot. Cache files record `identity`, `retrieved_at` and `data`; manifests identify the relevant keys and timestamps.

Inspection creates `.inspection.json` diagnostics. Custom score output prefixes are supported; a prefix inside a `data` directory renders the two Markdown reports in its parent. Comparisons and focused position reports remain separate because they may describe different policies or snapshots.

Treat credentials as secrets: keep token files out of Git and never print their contents or include them in artifacts. Change the Python generators rather than hand-editing generated Markdown.

## Configuration

The maintained color settings are [configs/white.json](../configs/white.json) and [configs/black.json](../configs/black.json). They define move overrides, chapter subjects and Explorer filters.

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

Instead of a region, `entries` can map a chapter ID to exact entry positions or paths; the chapter is then entered only at those boards. A chapter cannot have both `entries` and `chapter_regions`. Without either, all first chapter-unique positions become anchors, with chapter-owned descendants defining the region. If none can be entered under that chapter's policy, the first opponent reply on its mainline (or its PGN root) is used. Explicit subject anchors are preferable for comparing alternatives from a common position, such as `1.e4 e6` for both Advance and Tarrasch French chapters; automatic entries can describe different conditional subtrees.

Full reports also include directed **chapter transition probabilities**: conditional on first entering a source chapter, how often does the model reach the destination at or after that point? Shared or simultaneous entry counts. The JSON retains all ordered chapter pairs, including zeros and undefined results. This is different from an unordered intersection, because the destination may have been visited only before the source. Overlapping chapter frequencies and transition rows are not additive. The model does not follow deviations through unknown positions to possible later re-entry.

### Other options

`exclude` can contain chapter IDs or strings of the form `chapter-id:canonical-position:uci` to omit a branch and its descendants from that chapter. `root_weights` may map canonical root positions to weights summing to one. Disconnected custom-FEN chapters do not receive absolute reach probabilities without a connecting route or explicit root weights. Reachable cycles fail with a position sequence for diagnosis.

## Analysis commands

The complete build runs these analyses in dependency order. The commands below are useful when updating one analysis or investigating a result. Unless noted otherwise, they use saved scores and existing cache entries, without a token or Lichess requests. Cache-only analyses that reconstruct the repertoire require the source PGN and evidence to match the saved snapshot.

### Preparation and entry routes

After scoring and vulnerability generation, run:

```sh
uv run repertoire preparation reports/data/white.json reports/data/black.json
```

This command is cache-only. It writes `.preparation.json` beside each score result and refreshes the consolidated report and summary. It also runs automatically in `repertoire build`. Missing candidate evidence is reported, never fetched silently or treated as a zero score.

The same analysis saves actual **first-entry route examples** under each chapter's comparison policy. It stops every root-to-entry path when it first reaches any chapter-region position, merges all arriving probability at the exact board, and keeps the most likely single route as an example. Entry-position weights include every first-arrival route; the separately displayed example weight covers just that route. Both are conditional on reaching any position in the chapter. Examples are validated as legal and cannot pass an earlier chapter position. These explanations preserve the existing transposition-inclusive chapter scores and reach; ordinary position-table lines remain representative board labels.

### Repertoire character

```sh
uv run repertoire character reports/data/white.json reports/data/black.json
```

This cache-only command writes `data/white.character.json` and `data/black.character.json`, then refreshes the consolidated report and summary. It also runs automatically in `repertoire build`. No token or API requests are needed. Source PGN hashes and cached scoring evidence must match the saved scores. Overall and chapter scores and expected prepared depths are independently reproduced before writing these metrics. Chapter scopes retain weighted first-entry mixtures and chapter-local policy, including unselected alternatives.

### Strengths and vulnerabilities

Generate overall and chapter rankings from saved scores and cached parent-position tables:

```sh
uv run repertoire vulnerabilities reports/data/white.json reports/data/black.json
```

This defaults to **cache-only** and requires no token. If own decision positions were not needed by an earlier score run, add `--fetch-missing` with `--token-file` or `LICHESS_TOKEN`. Only missing own-parent tables are fetched, once per canonical position and filter set. Every candidate reply or alternative is read from its parent's cached move rows. Candidate child endpoints are never requested for screening. Existing score evidence must match the saved report's cache keys and retrieval timestamps; a changed PGN or refreshed evaluation cache requires regenerating scores first.

Outputs are `data/white.vulnerabilities.json` and `data/black.vulnerabilities.json`. The consolidated report has separate tables for unprepared opponent replies, prepared opponent replies, and selected own moves. Opponent replies rank by weighted drag; own moves rank by the direct deficit against the parent database score. Each category is filtered before its display limit. Set displayed ranking lengths with `repertoire report --top` and `--chapter-top`; JSON always retains all rankings and signed comparisons. `repertoire build` generates this analysis after scoring both colors, fetching missing parent tables unless `--offline` is set. Rendering alone does not recalculate vulnerabilities.

The metrics behind these rankings are defined under [Strengths and vulnerabilities](metrics.md#strengths-and-vulnerabilities).

### Rating contexts

```sh
uv run repertoire ratings reports/data/white.json reports/data/black.json
uv run repertoire report reports/data/white.json reports/data/black.json --require-complete
```

Run ratings after the vulnerability, preparation and character commands. It reads only existing Explorer cache entries, writes `data/white.ratings.json` and `data/black.ratings.json`, and refreshes the same consolidated report and summary. The runner includes this step automatically. Source, score, supporting-analysis and cache hashes identify the evidence. Refresh ratings whenever a supporting analysis changes; strict assembly rejects stale ledgers.

### Opening names and reach

```sh
uv run repertoire openings reports/data/white.json reports/data/black.json
```

This cache-only command creates `data/white.openings.json` and `data/black.openings.json` and refreshes the consolidated report and summary. It downloads no opening dataset and never requests unprepared child positions. Only cached Explorer `opening` fields on repertoire boards supply names and ECO codes. A named canonical board gives every arrival its exact cached current name. At an unnamed board, each route retains its last name and probability. For example, a shared unnamed board reached with 1% probability through Alekhine and 20% through Vienna carries those separate masses, rather than counting its entire 21% reach for both openings. Unclassified routes remain unclassified. Structural potential labels from all recorded variations are saved separately and cannot introduce probability from unused alternatives.

Broader family membership recognizes only other cached names matching at colon or comma boundaries, for example Sicilian Defense and Sicilian Defense: Accelerated Dragon. A transition between unrelated names, such as Smith-Morra and Open Sicilian, does not create a parent relation. Unnamed positions with no known name upstream remain unclassified. The saved catalog also identifies names that cannot be reached under the selected policy.

Each color has a full-report opening table containing every reached category. The summary uses opening sources on individual position rows, without a separate opening table. Identical opening names across multiple ECO codes share one category, retaining all cached codes and each board's exact cached label in JSON. Reach is first arrival at an exact cached name or a known more specific named variation. Inheritance cannot introduce a new opening: a route already passed the name's entry. Named transpositions and later named variations still count as entries, including routes bypassing earlier family roots. Multiple entries and returns count once per category. Different categories overlap and their reach cannot be summed. All opening comparisons use the overall selected policy rather than each chapter's alternative policy.

At every reached board, JSON stores current-name probability masses and conditional shares, with unclassified mass kept separately. It also stores the joint reach contributed by games that previously entered each opening, and the contribution divided by total board reach. These historical origin shares remain available after a later exact name changes the current classification. Opening details show the most common positions reached through that opening, alongside total board reach and its share of arrivals. Histories can include several opening categories, so origin shares overlap; current-name shares plus unclassified share form a normalized partition.

Line tables with chapter sources also show the **most common opening source**, followed by its share of arrivals. The winner is the last cached name carried by the largest incoming probability, rather than an overlapping historical opening family. An exact cached name replaces earlier labels; unclassified arrivals compete as a separate source. Position rows combine all transposed routes, while move comparisons use only arrivals through their specific parent move. Chapter tables follow that chapter's comparison policy, including alternatives; entry rows use only first arrivals. Equal shares use alphabetical order with named sources before unclassified ones. The opening companion saves these scope-specific sources without querying any additional positions.

Opening repertoire scores, recursive WDL sharpness, entry baselines, deltas, remaining prepared own moves, and equivalent gap reach use the same normalized first-entry weights. Gap distributions are merged by canonical board before taking the square root of the sum of squared probabilities. Unprepared entry boards use the parent response's results and depth zero. Missing outcome or baseline evidence stays unresolved. Scores and deltas include CP equivalents. Expandable details show exact or inherited names, real first-entry examples, combined entry weights, reach, game counts, and evidence sources. A displayed route can represent only part of its board's first-entry mass. Pooled parent counts may contain overlapping historical games and are marked with a dagger. Opening groups have no aggregate opponent rating, preserving the rule that ratings apply only at chapter or line level. The original overall and chapter score files are unchanged.

### Correlations

Both report correlations are reach-weighted. They examine different relationships: future preparation depth versus continuation gain, and opponent rating versus score within the same parent position.

Measure the association between future prepared depth and future preparation gain, weighted by decision reach:

```sh
uv run repertoire correlations reports/data/white.json reports/data/black.json
```

This writes `prepared-depth-gain-correlation.json` beside the scores and refreshes the consolidated report. Each canonical selected own move is one observation under the overall policy. Its depth is the expected number of prepared own moves after that move, excluding the selected move itself. Its gain is the recursive continuation score minus the selected move's database score from the cached parent table. Weight is the probability of playing that decision, merged across all transpositions. The main table shows reach-weighted linear and rank correlations and the weighted gain slope. Unweighted results and a positive-depth-only check are available in expandable details.

These are point estimates from the observed counts, without intervals. The result describes association, not the causal gain from adding preparation. Sparse, unreachable and unresolved decisions are excluded.

No Lichess requests or token are needed. The PGN files are read only to reconstruct dependencies and must still match the saved input hashes. Correlations run automatically in `repertoire build` or through this separate command. Presentation-only `repertoire report` combines the matching saved correlations without recalculating them.

To analyze the association between opponent rating and score using saved evidence:

```sh
uv run repertoire rating-correlations reports/data/white.json reports/data/black.json
```

This writes `reports/comparisons/opponent-rating-score.md` and ignored supporting JSON in `reports/data/`. It compares replies within canonical parent boards, using reach weights, recursive prepared scores and cached unprepared reply scores. It also compares chapter scores and baseline deltas with chapter stopping-evidence opponent ratings, grouping overlapping chapter regions. Both are point estimates without intervals. Sparse replies are excluded, and a 1,000-game sensitivity check is included. These are descriptive cohort associations, not causal rating effects or predictions at a target rating. No Lichess requests are made.

### Report insights and attribution

To add report insights to matching saved analyses without fetching data or changing existing scores:

```sh
uv run repertoire insights reports/data/white.json reports/data/black.json
```

This writes `white.insights.json` and `black.insights.json` and refreshes the two Markdown reports. It requires matching preparation, character, vulnerability, and opening analyses, and verifies the source PGN and cached evidence. Supporting hashes prevent old insight calculations from being combined with newer analyses.

All report tables containing individual lines include linked chapter attribution. A recorded move lists its exact position/move providers, including every shared source. A position lists the chapters containing that canonical board. An unprepared reply is labeled unprepared and lists its parent chapters as context; an unrecorded move that transposes into preparation lists the destination chapters. Representative routes may combine chapters.

Every analysis command adds this attribution automatically.

## Comparing alternative preparation

First rebuild the actual repertoire from its current PGN. Compare the candidate and actual preparation from the same entry board, for the same color, with matching filters and evidence. Report the conditional score at that entry separately from its full-repertoire reach and impact.

Keep hypothetical results separate from the actual score snapshots and current reports. The comparison tools leave source PGNs unchanged, write hypothetical PGNs under `.cache/`, and put readable results under `reports/comparisons/`.

[scripts/compare_vienna.py](../scripts/compare_vienna.py) compares candidate Vienna chapters from `1.e4 e5 2.Nc3` and checks combinations of improving chapter blocks. [scripts/compare_french.py](../scripts/compare_french.py) is specifically the Schlechter `4.Bd3` comparison, not a generic French importer. Both expect matching saved actual results and support `--candidate`, `--offline`, and `--token-file`. Read their scenario definitions before adapting them to a different opening; chapter order changes the selected policy. There is no generic candidate-comparison CLI that automatically fits every study.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `Offline cache miss` or missing parent comparison evidence | Cache path, filters and required board; an authenticated run can collect missing required tables. Keep missing data unresolved instead of inventing results. |
| Companion belongs to a different snapshot, supporting hashes changed, or a file was written by a different program version | Rebuild with `repertoire build`; it reruns only the affected analyses and their dependents. |
| Scoring or a build stage fails | Read the CLI diagnostic and any `.error.json`. Completed checkpoints and cache remain available; rerun after resolving the error. Existing Markdown can still describe the previous snapshot. |
| Study export fails with HTTP 401, 403 or 404 | Check the URL in `studies.json` and that the token has `study:read`; private studies are visible only to their owner and members. The previous export is kept. |
| Unknown chapter ID or missing configured anchor | Compare the new PGN's inspection with the maintained config; removed or recreated chapters may have different IDs. Update intended subject definitions explicitly. |
| Chapter defining position has less than 100% reach after entry | Inspect first-entry boards and routes: some games may enter through later transpositions and bypass that position. |
| Cache is complete but a batch is slow | Inspect `reports/data/.build-state.json` stage timings and reused/built counts. Changes limited to `render.py` or the `report` package should rebuild only the render stage. |
