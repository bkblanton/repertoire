# Repertoire score

Analyze White and Black Lichess study PGNs to see how often your preparation is reached, how it scores, and where the most common gaps remain. The program follows your chosen moves, weights the opponent's replies using Lichess opening statistics, and combines exact-position transpositions across chapters.

It produces a summary for everyday review, a full report for detailed analysis, and one page per chapter, as Markdown under `reports/`. The score describes database outcomes under a fixed repertoire policy; it is not an engine evaluation or a prediction of your personal rating gain.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/).
- A [Lichess personal API token](https://lichess.org/account/oauth/token). It authenticates Opening Explorer requests; give it the `study:read` scope to export private studies.

## Quick start

Install the dependencies:

```sh
uv sync --locked
```

Create `studies.json` in the repository root, pointing at your White and Black studies. A study URL, a chapter URL or a bare study ID all work:

```json
{
  "white": "https://lichess.org/study/<white-study-id>",
  "black": "https://lichess.org/study/<black-study-id>"
}
```

No other setup is needed. You can add move choices and chapter subjects later in `configs/white.json` and `configs/black.json`; see [Configuration](docs/usage.md#configuration).

Your repertoire stays on your computer: `studies.json`, the study exports in `studies/`, `configs/`, saved comparisons in `comparisons.json` and the generated `reports/` are all ignored by Git.

Export both studies, score them and write every report:

```sh
uv run repertoire build --token-file path/to/lichess_token.txt
```

The token can also come from the `LICHESS_TOKEN` environment variable. It is never written to reports, exports or the cache.

**The first run can take hours.** The build needs one Opening Explorer table per position in your repertoire (about 2,000 for some 60 chapters across both colors), requested one at a time, and Lichess rate-limits sustained use with one-minute pauses. To see the count and a minimum time before committing to a run, add `--dry-run`. During the fetch, progress and an estimate of the time left are printed every few seconds. Rate limits and short outages are waited out automatically, so the run can be left unattended; keep the computer from sleeping. Press Ctrl+C at any time: every fetched table is kept in `.cache/explorer/`, and running the same command again continues where it stopped. Later runs fetch only positions they have not seen, then take a couple of minutes to build. See [Long runs](docs/usage.md#long-runs).

Other common runs:

```sh
# Count the tables to fetch and estimate the time, without fetching.
uv run repertoire build --dry-run --token-file path/to/lichess_token.txt

# Fetch the tables only (for example overnight), then build later without network access.
uv run repertoire fetch --token-file path/to/lichess_token.txt

# Rebuild from the last export and cached evidence, without network access.
uv run repertoire build --offline

# Analyze PGN files you already have instead of the configured studies.
uv run repertoire build path/to/white.pgn path/to/black.pgn --token-file path/to/lichess_token.txt

# Compare a candidate study with your repertoire, alternative by alternative.
uv run repertoire compare https://lichess.org/study/<candidate-study-id> --token-file path/to/lichess_token.txt

# List chapter IDs, move conflicts and entry candidates for a PGN.
uv run repertoire score inspect path/to/white.pgn --color white --output reports/data/inspection
```

## Documentation

- [Usage](docs/usage.md): setup details, every command, configuration, files and cache, and troubleshooting.
- [Reports and metrics](docs/metrics.md): how to read the reports, how the score is calculated, and what each metric means.
- [Development](docs/development.md): code layout, conventions and tests.
- [Design](docs/design.md): the statistical foundations.

## Development

```sh
uv run pytest -q
uv run ruff check
uv run ruff format --check
```

Tests use synthetic data and cache fixtures, so they need no token or network access.

## License

[GPL-3.0-or-later](LICENSE). Opening statistics and study exports come from the [Lichess API](https://lichess.org/api).
