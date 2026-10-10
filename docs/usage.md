# Usage

Run every command from the repository root. Repertoire reads your Lichess studies but never edits them, and it never changes PGN files you give it. For what the numbers in the reports mean, see [Reports and metrics](metrics.md).

## Contents

- [Setup](#setup)
- [Commands](#commands)
- [Building the reports](#building-the-reports)
- [Long runs](#long-runs)
- [Analyzing other PGN files](#analyzing-other-pgn-files)
- [Keeping results up to date](#keeping-results-up-to-date)
- [Configuration](#configuration)
- [Comparing alternative preparation](#comparing-alternative-preparation)
- [Engine evaluations](#engine-evaluations)
- [Running single stages](#running-single-stages)
- [Files](#files)
- [Troubleshooting](#troubleshooting)

## Setup

```sh
uv sync --locked
```

You need a [Lichess personal API token](https://lichess.org/account/oauth/token) for anything that contacts Lichess. Give it the `study:read` scope to export private studies. Pass it with `--token-file path/to/token.txt` or set the `LICHESS_TOKEN` environment variable. A file named `lichess_token.txt` in the repository root is ignored by Git. The token is never written to reports, exports or the cache.

## Commands

Everything runs through `uv run repertoire <command>`. Add `--help` to any command to list its options. Most of the time you only need `build`.

| Command | What it does |
| --- | --- |
| `build` | Export the studies, fetch the Explorer tables, score both colors and write every report. |
| `export` | Export the studies without building. |
| `fetch` | Fetch the Explorer tables both repertoires need, without building. |
| `compare` | Compare a candidate study or PGN with your repertoire. See [Comparing alternative preparation](#comparing-alternative-preparation). |
| `evals` | Import Lichess cloud evaluations for your positions from the downloaded export. See [Engine evaluations](#engine-evaluations). |
| `report` | Render the Markdown reports from saved results, offline. |
| `score` | Inspect or score one repertoire PGN. |
| `vulnerabilities`, `preparation`, `character`, `ratings`, `openings`, `insights`, `engine`, `correlations`, `rating-correlations` | Run one analysis stage from saved results. See [Running single stages](#running-single-stages). |

## Building the reports

List your studies in `studies.json` in the repository root:

```json
{
  "white": "https://lichess.org/study/<white-study-id>",
  "black": "https://lichess.org/study/<black-study-id>"
}
```

A study URL, a chapter URL (the whole study is exported) or a bare 8-character study ID all work. Then run:

```sh
uv run repertoire build --token-file path/to/lichess_token.txt
```

The build runs in three steps:

1. **Export.** Both studies are downloaded to `studies/white.pgn` and `studies/black.pgn`, with every chapter, variation and comment. A download replaces the previous export only after it parses cleanly, and a file whose only change is its `Date` headers is left alone, so an unchanged study reuses every saved result.
2. **Fetch.** Every Opening Explorer table either repertoire needs is fetched in one pass, with a count and time estimate up front. Tables already in the cache are reused.
3. **Analyze.** Everything after the fetch runs offline: both colors are scored, the supporting analyses run, and the reports are written.

The reports are:

| File | Contents |
| --- | --- |
| `reports/summary.md` | The headline scores and what to work on next. Start here. |
| `reports/report.md` | The full report for both colors, with a glossary. |
| `reports/chapters/` | One page per chapter: `W1.md`, `W2.md`, ... and `B1.md`, `B2.md`, .... |
| `reports/openings/` | Opening-by-opening evidence for each color. |

Useful options:

- `--dry-run` exports the studies, prints the number of tables to fetch and a minimum time, then stops.
- `--offline` builds from the last export and the cache, with no token and no network access. A table missing from the cache stops the build with a message; it is never treated as a position with no games.
- `--no-export` builds from the last export but still fetches missing tables.
- `--force` reruns every stage. It uses the cached tables and does not refresh Lichess statistics.
- `--sources` and `--studies` choose a different study list and export folder.
- `--white-config` and `--black-config` choose different [configuration](#configuration) files.

`uv run repertoire export` runs only the export step, and `uv run repertoire fetch` only the fetch step.

## Long runs

Every position where the opponent is to move needs its own Opening Explorer table, as do the starting position, chapter entries and positions where your preparation ends; your own prepared moves need none. The build fetches them all before it analyzes anything. Requests go out one at a time, about one per second. Lichess answers sustained use with HTTP 429, which pauses the run for a minute at a time, so a first run on a large repertoire takes hours: some 60 chapters across both colors need about 1,300 tables. Later runs request only tables that are not cached, so editing a few lines costs a few requests.

Check the size of a run before you start it:

```sh
uv run repertoire build --dry-run --token-file path/to/lichess_token.txt
```

```text
white and black: 1270 Explorer tables: 260 cached, 1010 to fetch (at least 17m)
```

"At least" assumes one request per second and no rate limiting, so real runs are slower. `fetch --dry-run` gives the same count for the last export without needing a token.

While fetching, a progress line is printed about every 15 seconds. Its estimate uses the recent rate, including rate-limit pauses, so it settles after the first few minutes:

```text
white and black: 87/1010 fetched, ~1h 10m left
Explorer: rate-limited (HTTP 429); waiting 60s (87/1010 fetched)
```

The run looks after itself:

- **Rate limits** are waited out for as long as they last.
- **Server errors and dropped connections** are retried, with waits growing to a minute. A table that still fails after 30 minutes stops the run with an error.
- **Stopping** with Ctrl+C or by closing the terminal loses nothing already fetched. Each table is saved as soon as it arrives, and each build stage is saved when it finishes. Run the same command again to continue.

Keep the computer awake during an unattended run, because sleep drops the connection. On Windows, set the sleep timeout to Never in the power settings. On macOS, prefix the command with `caffeinate -i`; on Linux, with `systemd-inhibit`.

To do the slow part separately, for example overnight, fetch first and build later without network access:

```sh
uv run repertoire fetch --token-file path/to/lichess_token.txt
uv run repertoire build --offline
```

## Analyzing other PGN files

Pass two PGN paths, White then Black, to analyze files you already have. Nothing is downloaded from your studies:

```sh
uv run repertoire build path/to/white.pgn path/to/black.pgn --token-file path/to/lichess_token.txt
```

Give both paths or neither. Supported inputs:

- **A study export**, such as a study downloaded from the Lichess website. Each game is a chapter, and the `ChapterURL` headers supply the chapter IDs used in configuration files.
- **A plain PGN** with one or more games. Chapters are numbered `1`, `2`, ... in file order, and every variation is part of the repertoire.
- **A PGN that starts from a position**, using `SetUp` and `FEN` headers. If that position is reachable from other chapters, the chapter joins the repertoire there. Otherwise it is scored on the condition that its starting position is reached, unless [`root_weights`](#other-options) gives the starting positions weights.

Configuration files are keyed by chapter ID, so a different PGN needs its own configuration (or none). `uv run repertoire score inspect path/to/white.pgn --color white` lists a PGN's chapter IDs. Results for a different PGN replace the saved results and reports; pass `--directory` to keep them separate.

## Keeping results up to date

| What changed | What to run |
| --- | --- |
| Your Lichess studies | `uv run repertoire build --token-file ...` |
| Your configuration files | `uv run repertoire build --offline`, or with `--token-file` if new tables are needed. |
| Display limits, such as rows per table | `uv run repertoire report reports/data/white.json reports/data/black.json --require-complete` |
| Lichess statistics themselves | Refresh with `score run --refresh` (below), then `uv run repertoire build --offline`. |

The build is incremental. It checks the PGN contents, configuration, settings, program and dependency versions, the cache files each stage used, and each stage's inputs, and reruns only the stages whose inputs changed. Unrelated additions to the cache do not invalidate anything.

Cached tables never expire, and neither `build` nor `--force` refreshes them. To refetch the tables for one color:

```sh
uv run repertoire score run studies/white.pgn --color white --config configs/white.json --output reports/data/white --refresh --token-file path/to/lichess_token.txt
uv run repertoire build --offline
```

Every saved analysis records hashes of the PGN, the cached evidence and the analyses it depends on. Analyses from different snapshots are never combined: a stale one is rebuilt by `build`, or shown as pending by the other commands. If a source PGN has changed since it was scored, the reports say that they describe the earlier version.

## Configuration

Each color has a configuration file, `configs/white.json` and `configs/black.json`. They are created with no overrides the first time you run `build` or `fetch`, and are needed only for exceptions. A configuration file is a JSON object with these optional keys:

```json
{
  "policy": {"<canonical FEN>": "e2e4"},
  "entries": {"<chapter-id>": [{"path": ["e4", "e5", "Nc3", "Nf6", "g3", "Nc6"]}]},
  "filters": {"ratings": "0,1000,1200,1400,1600,1800,2000,2200,2500", "speeds": "blitz,rapid,classical", "since": "1952-01", "until": "3000-12"},
  "exclude": [],
  "root_weights": {}
}
```

Chapter IDs come from the study's chapter URLs; `uv run repertoire score inspect studies/white.pgn --color white` lists them with each chapter's automatic entries. Unknown chapter IDs, and positions outside the repertoire, fail validation.

### Explorer filters

By default the Explorer is queried for every rating band (0, 1000, 1200, 1400, 1600, 1800, 2000, 2200 and 2500), for blitz, rapid and classical games, over the full date range. These follow the [Opening Explorer API specification](https://github.com/lichess-org/api/blob/master/doc/specs/tags/openingexplorer/lichess.yaml). The Explorer covers rated games in its index, not every game played on Lichess. Override any of them with `filters`.

### Move selection

When your repertoire has more than one move in a position, Repertoire chooses one:

- **Competing chapters are decided by score.** Where chapters record different first moves in the same position, each move is scored with the best choices after it, and the highest-scoring one is played. To test an idea, put the alternative in its own chapter and rebuild: the summary and full report list every competing position with each alternative's score and its difference from the move played. An exact tie keeps the earlier chapter.
- **Otherwise the first recorded move is played:** the PGN main line before side variations. Side variations of your own moves within one chapter do not compete; move one into its own chapter to have it scored.

`policy` overrides both rules. Its keys are canonical FENs: piece placement, side to move, castling rights and a legal en passant square, without the move counters. Its values are a UCI move, or a map of UCI moves to weights that sum to one. An override is never compared with the alternatives.

Picking the best of several estimated scores favors moves that scored well by chance, so treat a small winning margin with care. Every PGN variation stays part of the repertoire; comments and annotations are not read as instructions.

Every chapter still gets its own page and row, including chapters whose moves lost to an alternative. A chapter is always scored with its own first moves, and with the overall repertoire everywhere else, so each alternative can be judged on its own terms.

### Chapter entries

A chapter is reached at its **entries**. By default they are found automatically: walking each of the chapter's lines from its start, following its own first moves, an entry is the first position that no other chapter continues from.

In a full repertoire this places each chapter where it starts to differ from the others. A chapter covering one branch of another chapter's opening, such as a 3...Nc6 chapter inside a 3.g3 chapter, takes that branch, and the broader chapter keeps the rest. A line that ends where another chapter continues hands that position over. If other chapters continue from every position in a chapter, as with an exact duplicate, its first own-turn main-line position is used.

`entries` overrides the automatic entries. It maps a chapter ID to a list of positions, each a canonical FEN or a `path` of SAN or UCI moves with an optional `root_fen`. Use it when a chapter covers only part of an opening (a chapter that shares no moves with any other chapter is entered at its start), or to give alternatives the same entry, such as `1.e4 e6` for both an Advance and a Tarrasch French chapter. See [Chapter reach and entries](metrics.md#chapter-reach-and-entries) for how entries affect the reports.

### Other options

- `exclude` leaves lines out of the repertoire. An entry is a chapter ID, to drop the whole chapter, or a `chapter-id:canonical-FEN:uci` string, to drop one move and everything after it from that chapter.
- `root_weights` maps the starting positions of custom-FEN chapters to weights that sum to one, so chapters with no connecting route get absolute reach probabilities.

A repertoire whose moves can return to an earlier position fails with the positions in the cycle.

## Comparing alternative preparation

`repertoire compare` answers "would this other preparation do better than mine?" for a candidate study or PGN. Build your repertoire first; the comparison reads its saved score and cached tables and never changes your studies, PGNs or reports.

```sh
uv run repertoire compare path/to/vienna-gambit.pgn --color white
uv run repertoire compare https://lichess.org/study/<candidate-study-id> --token-file path/to/lichess_token.txt
```

Inputs are any number of PGN files and Lichess study or chapter URLs, in priority order. A study URL exports the whole study and a chapter URL only that chapter. Exports are saved to `studies/candidates/`; `--no-export` reuses the saved copy. `--color` is needed only when the input has no `Orientation` header, as with a PGN downloaded from the Lichess website.

### How candidates are compared

Each candidate chapter is followed along its own first moves until it plays a different move from your repertoire, or adds a move where you have none. That position is a **decision point**, and its options are your move and each distinct candidate move:

- Chapters choosing the same move share one option. If they later disagree, that position becomes a decision point inside the option.
- Chapters choosing different moves, such as an Advance and a Schlechter chapter against your Tarrasch, compete at one decision point.

Adopting an option places its candidate lines ahead of your chapters: they decide every position they record, and your own preparation continues wherever they end or transpose into it. Three scenarios are scored:

- **Your repertoire**, reproduced exactly from its saved score.
- **Improving alternatives only:** at each decision point, the option with the highest score, which may be your own move.
- **All alternatives:** every decision point switches to the candidate. Where candidate chapters compete, the higher-scoring one is used.

Each scenario is scored for the whole color and from the **entry**, the last position all decision points share (`1.e4 e5 2.Nc3` for a Vienna study). Choose another with `--entry "1.e4 e5 2.Nc3"` or a FEN. Decision points whose lines transpose into each other are chosen together over every combination; the others are chosen one at a time.

### The comparison page

`reports/comparisons/<name>.md` leads with the verdict and the three scenarios, then lists each option at each decision point with its score, its change at that position and for the whole color, and a paired 95% interval. It warns when candidate scores come from much weaker or stronger opponents, when alternatives interact through transpositions, and when candidate chapters compete. Each decision point then has its own section: the database score of every move, how much the preparation adds, the main replies, and where each option's preparation most often ends.

`reports/comparisons/<name>.adopt.pgn` holds the candidate lines of the improving choice, ready to import into your study.

Missing tables are fetched with the usual count and progress; `--dry-run` stops after the count, and `--offline` lists them instead. `--name` sets the output name and `--title` the page title.

### Saved comparisons

Add `--save` to keep a comparison in `comparisons.json`:

```sh
uv run repertoire compare https://lichess.org/study/<candidate-study-id> --color white --save --token-file path/to/lichess_token.txt
```

```json
{
  "comparisons": [
    {"name": "vienna-gambit", "color": "white", "sources": ["https://lichess.org/study/<candidate-study-id>"]}
  ]
}
```

`build` then exports saved candidate studies along with your own, fetches their tables in the same pass, and reruns each comparison when your score or the candidate changes. A failing comparison is reported and skipped without blocking the main reports. The summary lists every saved comparison with its verdict and marks any made from an older score. `uv run repertoire compare` with no inputs reruns every saved comparison. A saved PGN from outside the repository is copied into `studies/candidates/`.

## Engine evaluations

`repertoire evals` keeps a local store of [Lichess cloud evaluations](https://database.lichess.org/#evals): Stockfish evaluations that Lichess users' browsers have shared, in centipawns or moves to mate from White's side. They describe positions objectively and are kept separate from the database score, which they never change. Build your repertoire first; the commands read the saved scores (`reports/data/white.json` and `black.json` by default) to find your positions.

The evaluations come only from the export Lichess publishes, a single file of several hundred million positions (about 21 GB compressed). Download [lichess_db_eval.jsonl.zst](https://database.lichess.org/lichess_db_eval.jsonl.zst) to `.cache/evals/`; the download can take hours, since Lichess limits its speed. Then run:

```sh
uv run repertoire evals import
uv run repertoire evals status
```

**`import`** searches the export for every position in your studies, every position one move beyond them (`--plies` changes how far) and every position where preparation ends. The file lists positions in no particular order, so it cannot be searched for one position: each import reads it from start to end, which takes about ten minutes, and keeps only the positions it is looking for. The store remembers which positions it has looked for in that file, so running `import` again after editing your studies reads the file again only if there are new positions, and looks only for those. Pass `--export` for a file elsewhere. A newer export is a different file, and importing it looks for every position again, keeping the deeper evaluation of each.

Both commands end with the coverage for each color, which **`status`** prints on its own: how many of the positions where preparation ends have an evaluation, weighted by how often games end there, and how many of the positions games reach. Positions nobody has analyzed on Lichess, usually rare replies deep in a line, have no evaluation. Nothing is requested or estimated in their place.

For each position the store keeps the deepest search in the export, with every line it reports, in `.cache/evals/evals.sqlite`.

`build` then adds an engine view to the reports (see [Engine view](metrics.md#engine-view)) and reruns it whenever the store changes. The export is optional: without a store, each report page says once that no engine evaluations are available, the engine column and sections are left out, and everything else is unchanged.

## Running single stages

`build` runs every stage in order and is the normal way to update results. The stage commands are useful for investigating one analysis.

### Inspecting and scoring one color

```sh
uv run repertoire score inspect studies/white.pgn --color white
uv run repertoire score run studies/white.pgn --color white --config configs/white.json --output reports/data/white --offline
```

`score inspect` prints chapter IDs, own-move conflicts and each chapter's automatic entries, without contacting Lichess. `score run` scores one color, fetching missing tables unless `--offline` is given. Other options:

- `--prior W D L` sets the win, draw and loss prior (default 0.5 each). See [Uncertainty](metrics.md#uncertainty).
- `--sparse-threshold` sets the game count below which evidence is flagged sparse (default 30).
- `--refresh` refetches every table the score needs.
- `--tolerance` records in the JSON whether the 95% interval is narrower than this many percentage points (default 1).

A score run alone leaves the other analyses pending. Run `build` to complete them.

### Analysis stages

These read saved scores and the cache and refresh the reports when they finish. They need no token and make no requests, and they check that the source PGNs and cached evidence still match the saved scores. Pass the saved scores, for example `uv run repertoire preparation reports/data/white.json reports/data/black.json`.

| Command | Writes (in `reports/data/`) | Notes |
| --- | --- | --- |
| `vulnerabilities` | `<color>.vulnerabilities.json` | Gains and drags of your moves and the opponent's replies. |
| `preparation` | `<color>.preparation.json` | Stopping outcomes, prepared-depth distributions and entry routes. |
| `character` | `<color>.character.json` | `--games` sets the reuse curve (default 10 50 100 500). |
| `ratings` | `<color>.ratings.json` | Needs vulnerabilities, preparation and character. |
| `openings` | `<color>.openings.json` | |
| `insights` | `<color>.insights.json` | Needs preparation, character, vulnerabilities and openings. |
| `engine` | `<color>.engine.json` | Needs vulnerabilities and preparation; reads the [evaluation store](#engine-evaluations). |
| `correlations` | `prepared-depth-gain-correlation.json` | Needs vulnerabilities. |
| `rating-correlations` | `opponent-rating-score-correlation.json` | Needs ratings and vulnerabilities. Also writes `reports/comparisons/opponent-rating-score.md`. |

### Rendering the reports

```sh
uv run repertoire report reports/data/white.json reports/data/black.json --require-complete
```

`report` renders saved results without recalculating or fetching anything. `--require-complete` fails if any analysis is missing or stale, instead of marking it pending. Display limits:

- `--top` (default 10): rows in overall rankings.
- `--chapter-top` (default 5): rows in chapter rankings. Chapter exit tables always show at least 10.
- `--position-top` (default 20): rows in the prepared-position and unprepared-reply tables for each color and chapter.

Rows are filtered before the limit is applied, and the JSON keeps every row. `--output` and `--summary` choose where the reports go.

## Files

| Location | Contents |
| --- | --- |
| `studies.json` | The White and Black study URLs. |
| `studies/` | The latest study exports, and candidate studies in `studies/candidates/`. |
| `configs/` | Your [configuration](#configuration) files. |
| `comparisons.json` | [Saved comparisons](#saved-comparisons). |
| `.cache/explorer/` | Every fetched Explorer table, keyed by position, endpoint and filters, with its retrieval time. |
| `.cache/evals/` | The cloud evaluation store and, if you keep it there, the downloaded export. |
| `reports/` | The generated reports, chapter and opening pages, and comparison pages. |
| `reports/data/` | Saved scores and analyses as JSON, and the build's progress in `.build-state.json`. |

All of these are ignored by Git, so your repertoire stays on your computer. Back up `studies.json`, `configs/` and `comparisons.json` yourself if you want their history. Deleting the cache means fetching every table again, and deleting `.cache/evals/` means downloading and importing the evaluation export again.

Saved scores include the full model: every stopping event, score, sensitivity result, sample count and policy decision, plus a `manifest` recording the input file, its hash, the configuration, the filters and the cache entries used. A failed score run writes its error to `<output>.error.json`.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `rate-limited (HTTP 429); waiting 60s` again and again | Normal on long runs; the run resumes after each pause. Avoid running other Opening Explorer clients on the same token or network at the same time. |
| The running estimate is far above the dry run's "at least" | Expected: the dry run assumes no rate limiting. |
| `still unavailable after 30 minutes of retries` | Lichess or your connection was down, or the computer slept. Rerun the same command; fetched tables are kept. |
| A run was stopped | Rerun the same command. Fetched tables and finished stages are kept. |
| Study export fails with HTTP 401, 403 or 404 | Check the URL in `studies.json`, and that the token has `study:read`. Private studies are visible only to their owner and members. The previous export is kept. |
| `Offline cache miss` | Run without `--offline`, with a token, to fetch the missing tables. |
| An analysis belongs to a different snapshot or program version | Run `build`; it reruns only the affected stages. |
| Unknown chapter ID, or a configured entry not in the repertoire | Chapters that were deleted and recreated get new IDs. Compare `score inspect` output with your configuration and update it. |
| A stage fails | Read the message and any `.error.json` file, fix the cause and rerun. Fetched tables and finished stages are kept; the previous reports stay in place. |
