"""Build both repertoires in one process: fetch every Explorer table first, then run each stage offline,
reusing verified unchanged analyses."""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from . import (
    character,
    compare,
    fetch,
    openings,
    position_correlations,
    preparation,
    rating_correlations,
    ratings,
    report_insights,
    studies,
    vulnerabilities,
)
from . import score as scoring
from .context import DEFAULT_CACHE
from .explorer import add_token_option, apply_token_file, observe_cache
from .layout import data_json, report_directory
from .render import defer_report_outputs
from .report.generate import generate, page_names

# Comparisons never affect the repertoire's own analyses; only their own step depends on this code.
COMPARISON_CODE = ('compare.py', 'alternatives.py')
FAMILIES = ('vulnerabilities', 'preparation', 'character', 'ratings', 'openings', 'insights')
DEPENDENCIES = {
    'ratings': ('preparation', 'character', 'vulnerabilities'),
    'insights': ('preparation', 'character', 'vulnerabilities', 'openings'),
}
MODULES = dict(
    vulnerabilities=vulnerabilities,
    preparation=preparation,
    character=character,
    ratings=ratings,
    openings=openings,
    insights=report_insights,
)


# File hashes for the current build run, keyed by path, size and modification time. One build
# checks the same inputs and roughly a thousand cache files for every stage. Builder clears this at
# the start of a run and forgets each stage's outputs after rewriting them, so a file replaced
# within the clock's resolution is never mistaken for its previous version.
_digests = {}


def forget(paths):
    names = {str(Path(p).resolve()) for p in paths}
    for identity in [i for i in _digests if i[0] in names]:
        del _digests[identity]


def digest(path):
    path = Path(path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    if not path.is_file():
        return None
    identity = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if identity not in _digests:
        _digests[identity] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _digests[identity]


def file_inputs(paths):
    return {str(Path(path).resolve()): digest(path) for path in paths}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(data_json(value), encoding='utf-8')
    temporary.replace(path)


def code_inputs(presentation=False):
    """Source files whose changes invalidate saved analyses, or with `presentation` also the rendered pages."""
    package = Path(__file__).parent
    rendering = [package / 'render.py', *sorted((package / 'report').glob('*.py'))]
    # Study export, table fetching and the command dispatcher never affect analyses or rendering.
    excluded = ('studies.py', 'fetch.py', 'cli.py', 'render.py', *COMPARISON_CODE)
    files = [p for p in sorted(package.glob('*.py')) if p.name not in excluded]
    # Bundled opening names label the opening analyses.
    files += sorted((package / 'data').rglob('*.tsv'))
    if presentation:
        files += rendering
    # Changes to pinned dependencies invalidate numerical analyses as well.
    files += [p for p in (package.parent / 'uv.lock', package.parent / 'pyproject.toml') if p.exists()]
    return file_inputs(files)


class Builder:
    """Checkpoint completed stages; failed stages never acquire a valid record."""

    def __init__(self, state_path, force=False):
        self.path, self.force = Path(state_path), force
        try:
            self.state = json.loads(self.path.read_text(encoding='utf-8'))
            if self.state.get('version') != 1 or not isinstance(self.state.get('steps'), dict):
                self.state = {}
        except (FileNotFoundError, ValueError):
            self.state = {}
        self.state.setdefault('version', 1)
        self.state.setdefault('steps', {})
        self.timings = []
        _digests.clear()

    def step(self, name, inputs, outputs, action):
        started = time.perf_counter()
        current = inputs()
        old = self.state['steps'].get(name, {})
        output_hashes = file_inputs(outputs)
        if (
            not self.force
            and old.get('inputs') == current
            and old.get('outputs') == output_hashes
            and all(output_hashes.values())
            and old.get('cache') == file_inputs(old.get('cache', {}))
        ):
            status = 'reused'
            print(f'Reusing {name}', flush=True)
        else:
            print(f'Building {name}', flush=True)
            # Drop the record before executing: a failed or interrupted write
            # must not be trusted on a subsequent invocation.
            self.state['steps'].pop(name, None)
            write_json(self.path, self.state)
            with observe_cache() as watched:
                action()
            forget(outputs)
            if current != inputs():
                raise ValueError(f'Inputs changed during {name}; rerun the batch')
            output_hashes = file_inputs(outputs)
            if not all(output_hashes.values()):
                raise ValueError(f'{name} did not produce all required outputs')
            self.state['steps'][name] = dict(inputs=current, outputs=output_hashes, cache=file_inputs(sorted(watched)))
            write_json(self.path, self.state)
            status = 'built'
        self.timings.append(dict(step=name, status=status, seconds=time.perf_counter() - started))
        print(f'{name}: {status} in {self.timings[-1]["seconds"]:.2f}s', flush=True)


def build(
    white_pgn,
    black_pgn,
    *,
    white_config=None,
    black_config=None,
    directory='reports/data',
    cache=DEFAULT_CACHE,
    offline=False,
    force=False,
    comparisons=None,
):
    started = time.perf_counter()
    pgns, configs = dict(white=white_pgn, black=black_pgn), dict(white=white_config, black=black_config)
    # Saved comparisons to rerun; the command line passes comparisons.json.
    saved = compare.load_registry(comparisons) if comparisons else []
    if not offline:
        # All network work happens here, so the stages below never wait on Lichess.
        for entry in saved:
            compare.load_inputs(entry['sources'], entry['name'], export=True)
        fetch.fetch(pgns, configs, cache, extra=compare.registry_tables(saved, pgns, configs))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    runner = Builder(directory / '.build-state.json', force)
    analysis_code, presentation_code = code_inputs(), code_inputs(presentation=True)
    package = Path(__file__).parent
    comparison_code = dict(presentation_code, **file_inputs([package / name for name in COMPARISON_CODE]))
    paths = [directory / f'{color}.json' for color in ('white', 'black')]
    cache = str(Path(cache).resolve())
    options = dict(
        command='run',
        cache=cache,
        offline=True,
        refresh=False,
        prior=[0.5, 0.5, 0.5],
        sparse_threshold=30,
        tolerance=1.0,
    )
    with defer_report_outputs():
        for color, pgn, config, path in zip(
            ('white', 'black'), (white_pgn, black_pgn), (white_config, black_config), paths
        ):
            args = SimpleNamespace(
                **options,
                color=color,
                pgn=str(Path(pgn).resolve()),
                config=str(Path(config).resolve()) if config else None,
                output=str(path.with_suffix('')),
            )
            runner.step(
                f'{color}.score',
                lambda args=args: dict(
                    code=analysis_code,
                    files=file_inputs([args.pgn] + ([args.config] if args.config else [])),
                    color=args.color,
                    cache=cache,
                    prior=args.prior,
                    sparse_threshold=args.sparse_threshold,
                    tolerance=args.tolerance,
                ),
                [path, path.with_suffix('.inspection.json')],
                lambda args=args: scoring.analyze(args),
            )
        for family in FAMILIES:
            for path in paths:
                required = [path] + [path.with_suffix(f'.{name}.json') for name in DEPENDENCIES.get(family, ())]

                def analyze(path=path, family=family):
                    value = MODULES[family].analyze(path, cache)
                    write_json(path.with_suffix(f'.{family}.json'), value)

                runner.step(
                    f'{path.stem}.{family}',
                    lambda required=required: dict(code=analysis_code, files=file_inputs(required), cache=cache),
                    [path.with_suffix(f'.{family}.json')],
                    analyze,
                )
        correlation = directory / 'prepared-depth-gain-correlation.json'
        preparation_inputs = paths + [path.with_suffix('.vulnerabilities.json') for path in paths]
        runner.step(
            'correlations',
            lambda: dict(code=analysis_code, files=file_inputs(preparation_inputs), cache=cache),
            [correlation],
            lambda: position_correlations.analyze(paths, correlation, cache=cache),
        )
        rating_correlation = directory / 'opponent-rating-score-correlation.json'
        rating_inputs = paths + [
            path.with_suffix(f'.{family}.json') for path in paths for family in ('ratings', 'vulnerabilities')
        ]
        runner.step(
            'rating-correlations',
            lambda: dict(code=analysis_code, files=file_inputs(rating_inputs)),
            [rating_correlation],
            lambda: write_json(rating_correlation, rating_correlations.analyze(paths)),
        )
    folder = report_directory(paths[0])
    results = []
    for entry in saved:
        results += compare_step(runner, entry, directory, folder, cache, offline, comparison_code)
    render_inputs = (
        paths
        + [path.with_suffix(f'.{family}.json') for path in paths for family in FAMILIES]
        + [correlation, rating_correlation]
        + results
    )
    registry = directory / '.report-index.json'

    def render():
        generate(paths, require_complete=True)
        write_json(registry, {path.stem: path.name for path in paths})

    # Chapter and opening pages are outputs too, so a deleted page triggers a fresh render.
    pages = [folder / name for path in paths for name in page_names(json.loads(path.read_text(encoding='utf-8')))]
    runner.step(
        'render',
        lambda: dict(code=presentation_code, files=file_inputs(render_inputs)),
        [folder / 'report.md', folder / 'summary.md', registry, *pages],
        render,
    )
    result = dict(
        created_at=datetime.now(UTC).isoformat(),
        seconds=time.perf_counter() - started,
        built=sum(t['status'] == 'built' for t in runner.timings),
        reused=sum(t['status'] == 'reused' for t in runner.timings),
        steps=runner.timings,
    )
    runner.state['last_run'] = result
    write_json(runner.path, runner.state)
    print(f'Batch complete in {result["seconds"]:.2f}s: {result["built"]} built, {result["reused"]} reused', flush=True)
    return result


def compare_step(runner, entry, directory, folder, cache, offline, code):
    """Rerun one saved comparison when its repertoire score or candidate changes. A failure is reported and
    skipped, so it never blocks the main reports; the previous comparison page is kept."""
    name, score = entry['name'], Path(directory) / f"{entry['color']}.json"
    data, page = Path(directory) / 'comparisons' / f'{name}.json', Path(folder) / 'comparisons' / f'{name}.md'

    def inputs():
        files = [score, *compare.source_files(entry['sources'], name)]
        return dict(code=code, entry=entry, files=file_inputs(files), cache=cache)

    try:
        runner.step(
            f'comparison.{name}',
            inputs,
            [data, page],
            lambda: compare.run(
                entry['sources'],
                color=entry['color'],
                score=score,
                entry=entry.get('entry'),
                name=name,
                cache=cache,
                offline=offline,
                export=False,
            ),
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f'Comparison {name} skipped: {exc}', flush=True)
        return []
    return [data]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('white_pgn', nargs='?', help='Exported White PGN; omit both to use the Lichess studies')
    parser.add_argument('black_pgn', nargs='?')
    parser.add_argument('--sources', default=studies.SOURCES, help='JSON file with white and black study URLs')
    parser.add_argument('--studies', default=studies.DIRECTORY, help='Folder for exported study PGNs')
    parser.add_argument('--no-export', action='store_true', help='Use the previously exported study PGNs')
    parser.add_argument('--white-config', help='Configuration (default: configs/white.json, created if missing)')
    parser.add_argument('--black-config', help='Configuration (default: configs/black.json, created if missing)')
    parser.add_argument('--directory', default='reports/data')
    parser.add_argument('--cache', default=DEFAULT_CACHE)
    parser.add_argument('--offline', action='store_true', help='Use only cached Explorer tables and the last export')
    parser.add_argument('--force', action='store_true', help='Rebuild all analyses using existing cached evidence')
    parser.add_argument('--comparisons', default=compare.REGISTRY, help='Saved comparisons to rerun')
    parser.add_argument(
        '--dry-run', action='store_true', help='Export the studies, then count the tables to fetch and stop'
    )
    add_token_option(parser)
    args = parser.parse_args()
    apply_token_file(parser, args)
    if (args.white_pgn is None) != (args.black_pgn is None):
        parser.error('give both PGN paths, or neither to use the exported studies')
    options = vars(args)
    sources, folder, no_export = options.pop('sources'), options.pop('studies'), options.pop('no_export')
    dry_run = options.pop('dry_run')
    options.pop('token_file')
    for color in ('white', 'black'):
        options[f'{color}_config'] = scoring.config_path(color, options[f'{color}_config'])
    try:
        if args.white_pgn is None:
            # Offline builds never contact Lichess, so they reuse the last export.
            paths = studies.default_paths(folder) if no_export or args.offline else studies.export(sources, folder)
            options.update(white_pgn=paths['white'], black_pgn=paths['black'])
        if dry_run:
            pgns = dict(white=options['white_pgn'], black=options['black_pgn'])
            configs = dict(white=options['white_config'], black=options['black_config'])
            saved = compare.load_registry(options['comparisons'])
            extra = compare.registry_tables(saved, pgns, configs)
            fetch.fetch(pgns, configs, options['cache'], dry_run=True, extra=extra)
            return
        build(**options)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(
            1,
            f'Build failed: {exc}\nFetched tables and completed stages are kept; rerun the same command to continue.\n',
        )


if __name__ == '__main__':
    main()
