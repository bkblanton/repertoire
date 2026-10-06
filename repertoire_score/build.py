"""Build both repertoires in one process, reusing verified unchanged analyses."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from . import __main__ as scoring
from . import character, openings, position_correlations, preparation, rating_correlations, ratings, report_insights, studies, vulnerabilities
from .consolidated import generate, page_names
from .explorer import observe_cache
from .layout import report_directory
from .render import defer_report_outputs

FAMILIES = ('vulnerabilities', 'preparation', 'character', 'ratings', 'openings', 'insights')
DEPENDENCIES = {
    'ratings': ('preparation', 'character', 'vulnerabilities'),
    'insights': ('preparation', 'character', 'vulnerabilities', 'openings'),
}
MODULES = dict(vulnerabilities=vulnerabilities, preparation=preparation, character=character,
               ratings=ratings, openings=openings, insights=report_insights)


def digest(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def file_inputs(paths):
    return {str(Path(path).resolve()): digest(path) for path in paths}


def write_json(path, value):
    # Preserve the companion format and its existing provenance contracts.
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def code_inputs(presentation=False):
    package = Path(__file__).parent
    # Study export code never affects analyses or rendering.
    files = [p for p in sorted(package.glob('*.py')) if p.name != 'studies.py'
             and (presentation or p.name not in ('consolidated.py', 'render.py'))]
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

    def step(self, name, inputs, outputs, action):
        started = time.perf_counter()
        current = inputs()
        old = self.state['steps'].get(name, {})
        output_hashes = file_inputs(outputs)
        if (not self.force and old.get('inputs') == current
                and old.get('outputs') == output_hashes and all(output_hashes.values())
                and old.get('cache') == file_inputs(old.get('cache', {}))):
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
            if current != inputs():
                raise ValueError(f'Inputs changed during {name}; rerun the batch')
            output_hashes = file_inputs(outputs)
            if not all(output_hashes.values()):
                raise ValueError(f'{name} did not produce all required outputs')
            self.state['steps'][name] = dict(inputs=current, outputs=output_hashes,
                                            cache=file_inputs(sorted(watched)))
            write_json(self.path, self.state)
            status = 'built'
        self.timings.append(dict(step=name, status=status, seconds=time.perf_counter() - started))
        print(f'{name}: {status} in {self.timings[-1]["seconds"]:.2f}s', flush=True)


def build(white_pgn, black_pgn, *, white_config='configs/white.json', black_config='configs/black.json',
          directory='reports/data', cache='.cache/explorer', offline=False, force=False,
          simulations=2000, seed=20260928, repetitions=20000):
    started = time.perf_counter()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    runner = Builder(directory / '.build-state.json', force)
    analysis_code, presentation_code = code_inputs(), code_inputs(presentation=True)
    paths = [directory / f'{color}.json' for color in ('white', 'black')]
    cache = str(Path(cache).resolve())
    options = dict(command='run', cache=cache, offline=offline, refresh=False,
                   simulations=simulations, seed=seed, prior=[.5, .5, .5], sparse_threshold=30, tolerance=1.)
    with defer_report_outputs():
        for color, pgn, config, path in zip(('white', 'black'), (white_pgn, black_pgn),
                                            (white_config, black_config), paths):
            args = SimpleNamespace(**options, color=color, pgn=str(Path(pgn).resolve()),
                config=str(Path(config).resolve()) if config else None, output=str(path.with_suffix('')))
            runner.step(f'{color}.score',
                lambda args=args: dict(code=analysis_code, files=file_inputs([args.pgn] + ([args.config] if args.config else [])),
                    color=args.color, cache=cache, simulations=simulations, seed=seed,
                    prior=args.prior, sparse_threshold=args.sparse_threshold, tolerance=args.tolerance),
                [path, path.with_suffix('.inspection.json')], lambda args=args: scoring.analyze(args))
        for family in FAMILIES:
            for path in paths:
                required = [path] + [path.with_suffix(f'.{name}.json') for name in DEPENDENCIES.get(family, ())]
                def analyze(path=path, family=family):
                    kwargs = dict(fetch_missing=not offline) if family == 'vulnerabilities' else {}
                    value = MODULES[family].analyze(path, cache, **kwargs)
                    write_json(path.with_suffix(f'.{family}.json'), value)
                runner.step(f'{path.stem}.{family}',
                    lambda required=required: dict(code=analysis_code, files=file_inputs(required), cache=cache),
                    [path.with_suffix(f'.{family}.json')], analyze)
        correlation = directory / 'prepared-depth-gain-correlation.json'
        preparation_inputs = paths + [path.with_suffix('.vulnerabilities.json') for path in paths]
        runner.step('correlations',
            lambda: dict(code=analysis_code, files=file_inputs(preparation_inputs), cache=cache,
                         simulations=simulations, seed=20261005),
            [correlation], lambda: position_correlations.analyze(paths, correlation, cache=cache,
                                                               simulations=simulations, seed=20261005))
        rating_correlation = directory / 'opponent-rating-score-correlation.json'
        rating_inputs = paths + [path.with_suffix(f'.{family}.json') for path in paths
                                for family in ('ratings', 'vulnerabilities')]
        runner.step('rating-correlations',
            lambda: dict(code=analysis_code, files=file_inputs(rating_inputs), repetitions=repetitions, seed=20261004),
            [rating_correlation], lambda: write_json(rating_correlation,
                rating_correlations.analyze(paths, repetitions=repetitions, seed=20261004)))
    folder = report_directory(paths[0])
    render_inputs = paths + [path.with_suffix(f'.{family}.json') for path in paths for family in FAMILIES] + [correlation, rating_correlation]
    registry = directory / '.report-index.json'
    def render():
        generate(paths, require_complete=True)
        write_json(registry, {path.stem: path.name for path in paths})
    # Chapter and opening pages are outputs too, so a deleted page triggers a fresh render.
    pages = [folder / name for path in paths for name in page_names(json.loads(path.read_text(encoding='utf-8')))]
    runner.step('render', lambda: dict(code=presentation_code, files=file_inputs(render_inputs)),
                [folder / 'report.md', folder / 'summary.md', registry, *pages], render)
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter() - started,
                  built=sum(t['status'] == 'built' for t in runner.timings),
                  reused=sum(t['status'] == 'reused' for t in runner.timings), steps=runner.timings)
    runner.state['last_run'] = result
    write_json(runner.path, runner.state)
    print(f'Batch complete in {result["seconds"]:.2f}s: {result["built"]} built, {result["reused"]} reused', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('white_pgn', nargs='?', help='Exported White PGN; omit both to use the Lichess studies')
    parser.add_argument('black_pgn', nargs='?')
    parser.add_argument('--sources', default=studies.SOURCES, help='JSON file with white and black study URLs')
    parser.add_argument('--studies', default=studies.DIRECTORY, help='Folder for exported study PGNs')
    parser.add_argument('--no-fetch', action='store_true', help='Use the previously exported study PGNs')
    parser.add_argument('--white-config', default='configs/white.json')
    parser.add_argument('--black-config', default='configs/black.json')
    parser.add_argument('--directory', default='reports/data')
    parser.add_argument('--cache', default='.cache/explorer')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--force', action='store_true', help='Rebuild all analyses using existing cached evidence')
    args = parser.parse_args()
    if (args.white_pgn is None) != (args.black_pgn is None):
        parser.error('give both PGN paths, or neither to use the exported studies')
    options = vars(args)
    sources, folder, no_fetch = options.pop('sources'), options.pop('studies'), options.pop('no_fetch')
    try:
        if args.white_pgn is None:
            # Offline builds never contact Lichess, so they reuse the last export.
            paths = (studies.default_paths(folder) if no_fetch or args.offline
                     else studies.fetch(sources, folder))
            options.update(white_pgn=paths['white'], black_pgn=paths['black'])
        build(**options)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(1, f'Batch failed: {exc}\nCompleted checkpoints and Explorer cache are preserved.\n')


if __name__ == '__main__':
    main()
