"""Generate one consolidated report and summary from saved JSON, offline."""
import argparse
import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from .consolidated import generate
from .layout import report_directory


_deferred = ContextVar('deferred_report_render', default=False)


@contextmanager
def defer_report_outputs():
    """A batch owns the final render; standalone commands still render eagerly."""
    token = _deferred.set(True)
    try:
        yield
    finally:
        _deferred.reset(token)


def load_report(path):
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    if report.get('color') not in ('white', 'black') or not all(k in report for k in ('overall', 'chapters', 'manifest', 'events')):
        raise ValueError(f'Not a repertoire result file: {path}')
    return report


def update_report_outputs(result_path):
    """Register the latest result per color in this folder and refresh all outputs."""
    if _deferred.get():
        return
    result_path = Path(result_path)
    report = load_report(result_path)
    directory = result_path.parent
    index_path = directory / '.report-index.json'
    index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else {}
    # Pick up the standard files produced before the output registry existed.
    for color in ('white', 'black'):
        candidate = directory / f'{color}.json'
        if color not in index and candidate.exists() and load_report(candidate)['color'] == color:
            index[color] = candidate.name
    index[report['color']] = result_path.name
    paths = []
    for color in ('white', 'black'):
        name = index.get(color)
        if name is None:
            continue
        if Path(name).name != name or Path(name).is_absolute():
            raise ValueError('Report registry filenames must stay within their output folder')
        path = directory / name
        if not path.exists():
            del index[color]
            continue
        if load_report(path)['color'] != color:
            raise ValueError(f'Report registry color mismatch: {path}')
        paths.append(path)
    # During a scoring refresh, old companions are pending rather than mixed in.
    generate(paths, summary_path=report_directory(result_path) / 'summary.md', strict=False)
    index_path.write_text(json.dumps(index, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', help='Saved repertoire JSON results')
    parser.add_argument('--output', help='Full report (default: first report folder/report.md)')
    parser.add_argument('--summary', help='Summary (default: full report folder/summary.md)')
    parser.add_argument('--top', type=int, default=10, help='Rows per overall ranking')
    parser.add_argument('--chapter-top', type=int, default=5, help='Rows per chapter ranking (default: 5)')
    parser.add_argument('--position-top', type=int, default=20, help='Rows per category in each color and chapter common-position ranking')
    parser.add_argument('--require-complete', action='store_true', help='Require every companion analysis and current correlations')
    args = parser.parse_args()
    try:
        generate(args.reports, args.output, args.summary, require_complete=args.require_complete,
                 top=args.top, chapter_top=args.chapter_top, position_top=args.position_top)
    except ValueError as exc:
        parser.exit(1, f'Report generation failed: {exc}\n')
    output = Path(args.output) if args.output else report_directory(args.reports[0]) / 'report.md'
    summary = Path(args.summary) if args.summary else output.parent / 'summary.md'
    print(f'Full report: {output.resolve()}\nSummary: {summary.resolve()}\nNetwork requests: 0')


if __name__ == '__main__':
    main()
