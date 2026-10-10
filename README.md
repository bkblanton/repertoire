# Repertoire

Find out how your chess opening repertoire scores against real games, and where it runs out.

Repertoire reads your White and Black [Lichess studies](https://lichess.org/study), plays your prepared moves, and weights every opponent reply by how often it is played in the [Lichess Opening Explorer](https://lichess.org/analysis#explorer). Transpositions are merged across chapters, so a position counts once however it is reached. The result is a set of Markdown reports: a short summary, a full report, and one page per chapter.

The score describes database results when you play your repertoire's moves. It is not an engine evaluation or a prediction of your own results.

## What the reports show

- **Expected score** for each color and chapter, next to the ordinary database score from the same starting point, with approximate 95% intervals.
- **Where preparation ends:** the positions where games most often leave your preparation, one row per study task.
- **Moves to review:** your own moves that score below the database average for their position.
- **Where your edge comes from:** the delta split move by move into parts that add up, so you can see which decisions earn it.
- **Chapter reach and depth:** how often each chapter is reached, by which move orders, and how many prepared moves you play on average.
- **Competing alternatives:** where chapters play different moves in the same position, each choice is scored and the best one is used.
- **Candidate comparisons:** whether another study would improve your repertoire, decision by decision.

## Requirements

- Python 3.12 or later and [uv](https://docs.astral.sh/uv/).
- A [Lichess personal API token](https://lichess.org/account/oauth/token). It authenticates Opening Explorer requests and study exports; give it the `study:read` scope if your studies are private.

## Quick start

Clone the repository and install the dependencies:

```sh
git clone https://github.com/bkblanton/repertoire.git
cd repertoire
uv sync --locked
```

Create `studies.json` in the repository root. A study URL, a chapter URL or a bare study ID all work:

```json
{
  "white": "https://lichess.org/study/<white-study-id>",
  "black": "https://lichess.org/study/<black-study-id>"
}
```

Export both studies, score them and write the reports:

```sh
uv run repertoire build --token-file path/to/lichess_token.txt
```

The token can also come from the `LICHESS_TOKEN` environment variable. It is never written to reports, exports or the cache.

Open `reports/summary.md` when the build finishes. [Reading the reports](docs/metrics.md#reading-the-reports) explains each section.

### The first run is slow

The build needs one Opening Explorer table for every position in your repertoire: about 2,000 for 60 chapters across both colors. Lichess rate-limits sustained use, so a first run can take hours. Add `--dry-run` to see the count and a minimum time before you start.

You can leave the run unattended. It prints progress with an estimate of the time left, waits out rate limits and short outages, and saves every table as it arrives. If you stop it, run the same command again to continue. Later runs fetch only new positions and finish in a few minutes. See [Long runs](docs/usage.md#long-runs).

## Common commands

```sh
# Count the tables to fetch and estimate the time, without fetching.
uv run repertoire build --dry-run --token-file path/to/lichess_token.txt

# Rebuild from the last export and cached tables, without network access.
uv run repertoire build --offline

# Analyze PGN files you already have instead of your studies.
uv run repertoire build path/to/white.pgn path/to/black.pgn --token-file path/to/lichess_token.txt

# Compare a candidate study with your repertoire.
uv run repertoire compare https://lichess.org/study/<candidate-study-id> --token-file path/to/lichess_token.txt
```

Your repertoire stays on your computer. `studies.json`, the study exports, your configuration, saved comparisons and the generated reports are all ignored by Git.

## Documentation

- [Usage](docs/usage.md): every command, configuration, updating results, comparisons and troubleshooting.
- [Reports and metrics](docs/metrics.md): how to read the reports and what each number means.
- [Model](docs/design.md): the probability model, evidence handling and uncertainty.
- [Development](docs/development.md): code layout, conventions and tests.

## Development

```sh
uv sync --locked
uv run pytest -q
uv run ruff check
uv run ruff format --check
uv run mypy
```

Tests use synthetic data, so they need no token or network access.

## License

[GPL-3.0-or-later](LICENSE).

Opening statistics and study exports come from the [Lichess API](https://lichess.org/api). Opening names come from [lichess-org/chess-openings](https://github.com/lichess-org/chess-openings) (CC0).
