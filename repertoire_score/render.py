"""Regenerate reports and study descriptions from saved JSON, without network access."""
import argparse
import json
from pathlib import Path
from .report import markdown, study_description, summary_markdown


def load_report(path):
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    if report.get('color') not in ('white', 'black') or not all(k in report for k in ('overall', 'chapters', 'manifest', 'events')):
        raise ValueError(f'Not a repertoire result file: {path}')
    return report


def render_reports(paths, summary_path):
    paths = [Path(p) for p in paths]
    reports = [load_report(p) for p in paths]
    for path, report in zip(paths, reports):
        path.with_suffix('.md').write_text(markdown(report), encoding='utf-8')
        path.with_suffix('.study-description.md').write_text(study_description(report), encoding='utf-8')
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_markdown(reports, paths), encoding='utf-8')
    return reports


def update_report_outputs(result_path):
    """Register the latest result per color in this folder and refresh all outputs."""
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
    render_reports(paths, directory / 'summary.md')
    index_path.write_text(json.dumps(index, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', help='Saved repertoire JSON results')
    parser.add_argument('--summary', help='Combined Markdown output (default: first report folder/summary.md)')
    args = parser.parse_args()
    target = Path(args.summary) if args.summary else Path(args.reports[0]).parent / 'summary.md'
    render_reports(args.reports, target)
    print(f'Regenerated {len(args.reports)} detailed reports and study descriptions; summary: {target.resolve()}')


if __name__ == '__main__':
    main()
