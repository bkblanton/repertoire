"""Offline associations between opponent reply ratings and repertoire scores."""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .report.format import score_cell, score_points
from .stats import connected_groups, correlation, rank
from .status import Status


def finite(value):
    return isinstance(value, (int, float)) and np.isfinite(value)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def point(moment, centered):
    """Correlation and score-per-100-rating slope from sufficient moments."""
    a = np.asarray(moment, dtype=float)
    if centered:
        xx, xy, yy = a[..., 0], a[..., 1], a[..., 2]
    else:
        w, wx, wy, wxx, wxy, wyy = np.moveaxis(a, -1, 0)
        xx, xy, yy = wxx - wx * wx / w, wxy - wx * wy / w, wyy - wy * wy / w
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where((xx > 1e-15) & (yy > 1e-15), xy / np.sqrt(np.maximum(xx * yy, 0)), np.nan)
        slope = np.where(xx > 1e-15, 10000 * xy / xx, np.nan)
    return np.clip(r, -1, 1), slope


def estimate(moments, centered):
    """Pooled correlation and slope from per-group sufficient moments."""
    a = np.asarray(moments, dtype=float)
    if not len(a):
        return dict(correlation=None, slope_percent_per_100_rating=None, clusters=0, status=Status.TOO_FEW_GROUPS)
    r, slope = point(a.sum(axis=0), centered)
    return dict(
        correlation=float(r) if finite(float(r)) else None,
        slope_percent_per_100_rating=float(slope) if finite(float(slope)) else None,
        clusters=len(a),
        status=Status.CALCULATED,
    )


def within_parent(rows):
    """Demean both variables within each canonical parent board."""
    parents = defaultdict(list)
    for row in rows:
        parents[row['position']].append(row)
    moments, eligible, details = [], [], []
    for position, replies in sorted(parents.items()):
        if len(replies) < 2:
            continue
        x = np.array([r['rating'] for r in replies], dtype=float)
        y = np.array([r['score'] for r in replies], dtype=float)
        w = np.array([r['weight'] for r in replies], dtype=float)
        dx, dy = x - np.average(x, weights=w), y - np.average(y, weights=w)
        xx, xy, yy = (float(np.sum(w * v)) for v in (dx * dx, dx * dy, dy * dy))
        if xx <= 1e-15:
            continue
        moments.append([xx, xy, yy])
        eligible.extend(replies)
        details.append(
            dict(position=position, replies=len(replies), weight=float(w.sum()), centered_moments=[xx, xy, yy])
        )
    result = estimate(moments, True)
    result.update(
        replies=len(eligible),
        parents=len(moments),
        parent_moments=details,
        eligible_reply_weight=sum(r['weight'] for r in eligible),
    )
    return result


def chapter_association(rows, field, weighted):
    groups = defaultdict(list)
    origin_x = np.mean([r['rating'] for r in rows]) if rows else 0
    origin_y = np.mean([r[field] for r in rows]) if rows else 0
    for row in rows:
        x, y = row['rating'] - origin_x, row[field] - origin_y
        w = row['reach'] if weighted else 1.0
        groups[row['cluster']].append(np.array([w, w * x, w * y, w * x * x, w * x * y, w * y * y]))
    result = estimate([np.sum(groups[k], axis=0) for k in sorted(groups)], False)
    result.update(
        chapters=len(rows),
        weighting='chapter entry reach' if weighted else 'equal chapter weight',
        rank_correlation=correlation(
            rank(np.array([r['rating'] for r in rows])), rank(np.array([r[field] for r in rows]))
        )
        if len(rows) > 1
        else None,
    )
    if result['rank_correlation'] is not None and not finite(result['rank_correlation']):
        result['rank_correlation'] = None
    return result


def load_color(path):
    path = Path(path)
    paths = dict(
        score=path, ratings=path.with_suffix('.ratings.json'), vulnerabilities=path.with_suffix('.vulnerabilities.json')
    )
    hashes = {name: digest(p) for name, p in paths.items()}
    saved = {name: json.loads(p.read_bytes()) for name, p in paths.items()}
    score, ratings, vulnerabilities = (saved[k] for k in ('score', 'ratings', 'vulnerabilities'))
    m = score['manifest']
    source = Path(m['input_path'])
    if digest(source) != m['input_sha256']:
        raise ValueError('Source PGN changed since scoring; saved associations would be stale')
    for name in ('ratings', 'vulnerabilities'):
        companion = saved[name]
        if companion['color'] != score['color'] or any(
            companion['manifest'].get(k) != expected
            for k, expected in (
                ('report_sha256', hashes['score']),
                ('input_sha256', m['input_sha256']),
                ('filters', m['filters']),
            )
        ):
            raise ValueError(f'{name} does not match the saved score snapshot')
    if ratings['manifest']['supporting_sha256']['vulnerabilities'] != hashes['vulnerabilities']:
        raise ValueError('Ratings reference a different vulnerability snapshot')
    scopes = {s['id']: s for s in ratings['scopes']}
    chapter_rows, position_sets = [], []
    excluded_chapters = []
    for chapter in score['chapters']:
        context = scopes.get(chapter['id'], {}).get('score_evidence', {})
        values = dict(
            rating=context.get('mean'),
            score=chapter['score'].get('raw_empirical_score'),
            delta=chapter['entry_baseline'].get('difference_pp'),
            reach=chapter['score'].get('entry_probability'),
        )
        if (
            context.get('known_coverage', 0) < 1 - 1e-9
            or not all(finite(v) for v in values.values())
            or values['reach'] <= 0
        ):
            excluded_chapters.append(chapter['id'])
            continue
        values.update(
            id=chapter['id'],
            name=chapter['name'],
            delta=values['delta'] / 100,
            baseline=chapter['entry_baseline']['raw_score'],
            rating_coverage=context['known_coverage'],
        )
        chapter_rows.append(values)
        position_sets.append(set(chapter['prepared_positions']))
    for number, group in enumerate(connected_groups(position_sets), 1):
        for i in group:
            chapter_rows[i]['cluster'] = f"{score['color']}-{number}"
    threshold = m['sparse_threshold']
    replies, excluded = [], defaultdict(int)
    seen = set()
    rating_rows = scopes['overall']['moves']
    for row in vulnerabilities['overall']['all_signed_rows']:
        if row['kind'] != 'opponent':
            continue
        if row['id'] in seen:
            raise ValueError('Duplicate canonical parent/reply pair')
        seen.add(row['id'])
        if row.get('sparse') or row['sample_count'] < threshold:
            excluded['sparse'] += 1
            continue
        rating = rating_rows.get(row['id'], {}).get('mean')
        if not finite(rating) or not finite(row.get('move_score')) or row['branch_reach'] <= 0:
            excluded['missing_rating_score_or_reach'] += 1
            continue
        replies.append(
            dict(
                id=row['id'],
                position=row['position'],
                rating=rating,
                score=row['move_score'],
                weight=row['branch_reach'],
                games=row['sample_count'],
                prepared=row['prepared'],
                parent_score=row['reference_score'],
                parent_rating=rating_rows[row['id']].get('parent_mean'),
                line=row['line'],
                chapters=row['chapters'],
            )
        )
    del saved, ratings, vulnerabilities
    return dict(
        color=score['color'],
        chapters=chapter_rows,
        replies=replies,
        exclusions=dict(chapters=excluded_chapters, replies=dict(excluded)),
        provenance=dict(
            paths={k: str(v.resolve()) for k, v in paths.items()},
            hashes=hashes,
            input_path=str(source),
            input_sha256=m['input_sha256'],
            filters=m['filters'],
            sparse_threshold=threshold,
        ),
    )


def cell(result, key='correlation', slope=False):
    value = result.get(key)
    return 'unavailable' if value is None else f"{value:+.3f}{'%' if slope else ''}"


def markdown(result):
    lines = [
        '# Opponent rating and repertoire score',
        '',
        'This cache-only analysis measures associations between the mean '
        'ratings of players choosing replies and our modeled continuation '
        'scores. It does not measure the causal effect of rating or predict an '
        'individual player\'s result.',
        '',
        '## Replies from the same position',
        '',
        'Both rating and continuation score are centered on their '
        'reach-weighted means among eligible replies from the same board. '
        'Weights are parent reach times reply frequency. Positive slopes favor '
        'the repertoire owner; negative slopes mean replies associated with '
        'stronger opponents leave us with lower scores. Prepared replies use '
        'the recursive repertoire score, and unprepared replies use the cached '
        'parent move row. Transposed boards and replies occur only once in '
        'this analysis.',
        '',
        '| Repertoire | Reply set | Replies / parent boards | Within-position '
        'correlation | Score association per +100 rating |',
        '|---|---|---:|---:|---:|',
    ]
    labels = {
        'all': 'All eligible replies',
        'prepared': 'Prepared replies',
        'unprepared': 'Unprepared replies',
        'at_least_1000_games': 'Replies with at least 1,000 games',
    }
    for color, data in result['results'].items():
        for key, estimate in data['reply_associations'].items():
            lines.append(
                f"| {color.title()} | {labels[key]} | {estimate['replies']:,} / {estimate['parents']:,} | "
                f"{cell(estimate)} | {cell(estimate, 'slope_percent_per_100_rating', True)} |"
            )
    lines += [
        '',
        'These are descriptive point estimates without intervals. Repeated '
        'games and shared downstream evidence connect different boards. The '
        'slope describes differences between reply cohorts, not the effect of '
        'raising one opponent\'s rating.',
        '',
        '## Across chapters',
        '',
        'The rating is the chapter\'s modeled, once-per-game stopping-evidence '
        'opponent mean, paired with the chapter score. Delta is score minus '
        'entry baseline. Equal chapter weighting is the primary comparison; '
        'reach weighting shows sensitivity to commonly reached chapters. '
        'Chapters preparing shared boards are resampled together, including '
        'indirect transpositions. There is no whole-repertoire average rating.',
        '',
        '| Repertoire | Comparison | Chapters / transposition groups | Correlation | Rank correlation |',
        '|---|---|---:|---:|---:|',
    ]
    for color, data in result['results'].items():
        for key, estimate in data['chapter_associations'].items():
            label = key.replace('_', ' ').capitalize()
            rank_value = estimate['rank_correlation']
            rank_text = f'{rank_value:+.3f}' if rank_value is not None else 'unavailable'
            lines.append(
                f"| {color.title()} | {label} | {estimate['chapters']} / "
                f"{estimate['clusters']} | {cell(estimate)} | {rank_text} |"
            )
    lines += [
        '',
        'Across-chapter results also reflect which openings and positions the '
        'repertoire selects. Few transposition groups, overlap outside prepared '
        'chapter boards, and shared historical games limit their interpretation.',
        '',
        '## Chapter inputs',
        '',
    ]
    for color, data in result['results'].items():
        lines += [
            f'### {color.title()} repertoire',
            '',
            '| Chapter | Entry reach | Average opponent rating | Repertoire score / CP | Delta vs entry baseline |',
            '|---|---:|---:|---:|---:|',
        ]
        for row in data['chapters']:
            name = row['name'].replace('|', '&#124;')
            lines.append(
                f"| {name} | {100 * row['reach']:.2f}% | {row['rating']:,.1f} | "
                f"{score_cell(row['score'])} | "
                f"{score_points(100 * row['delta'], signed=True)} |"
            )
    lines += [
        '',
        '## Evidence and reproduction',
        '',
        '- Filters: standard chess, all cached rating buckets, blitz, rapid '
        'and classical. Scores include half a point for draws.',
        '- Unprepared child positions are not queried. Only saved score, rating and vulnerability JSON files are read.',
        '- CP uses the same score-scale conversion as the consolidated report and is not an engine evaluation.',
        '- Reproduce: `uv run repertoire rating-correlations reports/data/white.json reports/data/black.json`.',
        '',
    ]
    for color, data in result['results'].items():
        lines.append(
            f"{color.title()}: sparse threshold "
            f"{data['provenance']['sparse_threshold']} games; excluded replies "
            f"{data['exclusions']['replies']}; excluded chapters "
            f"{len(data['exclusions']['chapters'])}."
        )
    return '\n'.join(lines) + '\n'


def analyze(paths):
    result = dict(created_at=datetime.now(UTC).isoformat(), network_requests=0, results={})
    for path in paths:
        data = load_color(path)
        if data['color'] in result['results']:
            raise ValueError('Supply one score file per color')
        print(
            f"{data['color']}: {len(data['chapters'])} chapters, {len(data['replies']):,} non-sparse replies",
            flush=True,
        )
        replies = data.pop('replies')
        subsets = dict(
            all=replies,
            prepared=[r for r in replies if r['prepared']],
            unprepared=[r for r in replies if not r['prepared']],
            at_least_1000_games=[r for r in replies if r['games'] >= 1000],
        )
        data['reply_associations'] = {name: within_parent(rows) for name, rows in subsets.items()}
        data['chapter_associations'] = {
            name: chapter_association(data['chapters'], field, weighted)
            for name, field, weighted in (
                ('score', 'score', False),
                ('delta', 'delta', False),
                ('reach_weighted_score', 'score', True),
                ('reach_weighted_delta', 'delta', True),
            )
        }
        result['results'][data['color']] = data
        for name, source in data['provenance']['paths'].items():
            if digest(source) != data['provenance']['hashes'][name]:
                raise ValueError('An input changed during analysis')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+')
    parser.add_argument('--output', default='reports/data/opponent-rating-score-correlation.json')
    parser.add_argument('--markdown', default='reports/comparisons/opponent-rating-score.md')
    args = parser.parse_args()
    result = analyze(args.reports)
    for target, content in (
        (args.output, json.dumps(result, indent=2, allow_nan=False) + '\n'),
        (args.markdown, markdown(result)),
    ):
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        print(f'Saved {path.resolve()}', flush=True)


if __name__ == '__main__':
    main()
