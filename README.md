# Repertoire score

Analyze White and Black Lichess study PGNs to see how often your preparation is reached, how it scores, and where the most common gaps remain. The program follows your chosen moves, weights the opponent's replies using Lichess opening statistics, and combines exact-position transpositions across chapters.

It produces a [summary](reports/summary.md) for everyday review, a [full report](reports/report.md) for detailed analysis, and one page per chapter. The score describes database outcomes under a fixed repertoire policy; it is not an engine evaluation or a prediction of your personal rating gain.

## Example

This repository includes the author's own repertoires as a worked example: the study exports in [studies/](studies), their configuration in [configs/](configs), and the generated [reports/](reports). Start with the [example summary](reports/summary.md).

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/).
- A [Lichess personal API token](https://lichess.org/account/oauth/token). It authenticates Opening Explorer requests; give it the `study:read` scope to export private studies.

## Quick start

Install the dependencies:

```sh
uv sync --locked
```

Point [studies.json](studies.json) at your White and Black studies. A study URL, a chapter URL or a bare study ID all work:

```json
{
  "white": "https://lichess.org/study/abcd1234",
  "black": "https://lichess.org/study/mnop3456"
}
```

The files in [configs/](configs) are keyed by the example studies' chapter IDs, so replace each with `{}` before analyzing your own studies. You can add move choices and chapter subjects later; see [Configuration](docs/usage.md#configuration).

Export both studies, score them and write every report:

```sh
uv run repertoire build --token-file path/to/lichess_token.txt
```

The token can also come from the `LICHESS_TOKEN` environment variable. It is never written to reports, exports or the cache. Opening Explorer responses are cached in `.cache/explorer/`, so later runs request only positions they have not seen. Requests are made one at a time and back off when Lichess rate-limits.

Other common runs:

```sh
# Rebuild from the last export and cached evidence, without network access.
uv run repertoire build --offline

# Analyze PGN files you already have instead of the configured studies.
uv run repertoire build path/to/white.pgn path/to/black.pgn --token-file path/to/lichess_token.txt

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
