"""Display rows derived from saved analyses: combined colors, exit points and position contributions."""

from collections.abc import Iterable, Sequence
from typing import cast

from ..schema import JsonObject, Position
from ..sharpness import FIELDS as OUTCOME_FIELDS
from ..sharpness import summarize as summarize_outcomes
from ..spread import mixture as spread_mixture
from .bundle import Bundle
from .format import centipawn_delta, centipawn_equivalent, elo_equivalent


def combined_overall(bundles: Iterable[Bundle]) -> JsonObject | None:
    """A 50/50 color mixture, averaging scores before applying conversions."""
    reports = {b['report']['color']: b['report'] for b in bundles}
    if set(reports) != {'white', 'black'}:
        return None
    white, black = reports['white'], reports['black']
    if white['manifest']['filters'] != black['manifest']['filters']:
        return {'unavailable': 'the colors use different Explorer filters'}
    if any(r['manifest'].get('overall_basis') != 'standard starting position' for r in (white, black)):
        return {'unavailable': 'both colors must be evaluated from the standard starting position'}

    def average(values: Sequence[float | None]) -> float | None:
        return None if any(v is None for v in values) else sum(cast('Sequence[float]', values)) / 2

    score = average([r['overall'].get('raw_empirical_score') for r in (white, black)])
    baseline = average([r.get('starting_position_reference', {}).get('owner_score') for r in (white, black)])
    outcomes = [r['overall'].get('outcomes') for r in (white, black)]
    combined_outcomes = (
        summarize_outcomes(cast('list[float]', [average([o[field] for o in outcomes]) for field in OUTCOME_FIELDS]))
        if all(outcomes)
        else None
    )
    spreads = [r['overall'].get('branch_score_spread') for r in (white, black)]
    combined_spread = (
        spread_mixture([(0.5, spread) for spread in spreads], score)
        if all(s and 'within_stop_outcome_variance' in s for s in spreads)
        else None
    )
    return {
        'weights': {'white': 0.5, 'black': 0.5},
        'repertoire_score': score,
        'starting_baseline': baseline,
        'difference_pp': None if score is None or baseline is None else 100 * (score - baseline),
        'elo_equivalent': elo_equivalent(score, baseline),
        'centipawn_equivalent': centipawn_equivalent(score),
        'baseline_centipawn_equivalent': centipawn_equivalent(baseline),
        'centipawn_delta': centipawn_delta(score, baseline),
        'expected_prepared_depth': average(
            [r['overall'].get('prepared_depth', {}).get('expected_moves') for r in (white, black)]
        ),
        'outcomes': combined_outcomes,
        'branch_score_spread': combined_spread,
        'chapters': sum(len(r['chapters']) for r in (white, black)),
    }


def non_sparse_rows(rows: Iterable[JsonObject]) -> list[JsonObject]:
    """Filter local, parent, and immediate endpoint evidence before display limits."""
    return [
        r
        for r in rows
        if not any(r.get(field, False) for field in ('sparse', 'parent_sparse', 'continuation_endpoint_sparse'))
    ]


def own_priorities(rows: Iterable[JsonObject], strongest: bool = False) -> list[JsonObject]:
    field = 'local_gain_pp' if strongest else 'local_drop_pp'
    return sorted(
        [r for r in non_sparse_rows(rows) if r.get(field) is not None and r[field] > 0 and r['branch_reach'] > 0],
        key=lambda r: (-r['branch_reach'] * r[field], -r['branch_reach'], r['line']),
    )


def visible_positions(scope: JsonObject | None) -> list[JsonObject]:
    """Keep one board after a guaranteed own reply, plus unanswered boards."""
    rows = [
        r
        for r in (scope or {}).get('positions', [])
        if not r['is_starting_position'] and r['reach'] > 0 and r['kind'] != 'own_move'
    ]
    return sorted(rows, key=lambda r: (-r['reach'], len(r['line'].split()), r['line'], r['position']))


def unanswered_position(row: JsonObject, color: str) -> bool:
    return (
        bool(row.get('unprepared_origins'))
        or row['kind'] == 'unprepared_reply'
        or (row['kind'] == 'theory_leaf' and row['to_move'] == color)
    )


def position_contributions(scope: JsonObject | None, color: str, sparse_threshold: int = 30) -> list[JsonObject]:
    """Score carried through every reached board; nested rows are not additive."""
    rows: list[JsonObject] = []
    for row in visible_positions(scope):
        unanswered = unanswered_position(row, color)
        value = row.get('database_score' if unanswered else 'repertoire_score')
        if value is None:
            continue
        counts = [r.get('games') for r in row.get('unprepared_origins', [])] or [row.get('games')]
        sparse = row.get('sparse', False) or any(n is None or n < sparse_threshold for n in counts)
        rows.append(
            dict(
                row,
                score=value,
                score_basis='database' if unanswered else 'repertoire',
                contribution_pp=100 * row['reach'] * value,
                sparse=sparse,
            )
        )
    return sorted(
        rows, key=lambda r: (-r['contribution_pp'], -r['reach'], len(r['line'].split()), r['line'], r['position'])
    )


def exit_points(scope: JsonObject | None) -> list[JsonObject]:
    """Group first unprepared positions by the last prepared board, so one study task is one row.

    Every modeled game stops once, so these groups partition the stopping mass and do not overlap.
    """
    positions = {r['position']: r for r in (scope or {}).get('positions', [])}
    groups: dict[Position, JsonObject] = {}
    for stop in (scope or {}).get('stopping_outcomes', []):
        # Finished games and missing opponent data are not places where preparation runs out.
        finished = stop['type'] == 'theory_leaf' and (
            positions.get(stop['position'], {}).get('kind') == 'terminal'
            or (stop.get('sample_count') == 0 and stop.get('score') is not None)
        )
        if (
            stop['type'] not in ('deviation', 'theory_leaf', 'no_recorded_continuation')
            or finished
            or not stop['reach'] > 0
        ):
            continue
        board = stop.get('parent_position') or stop['position']
        group = groups.setdefault(
            board, dict(position=board, reach=0.0, known=0.0, score_mass=0.0, replies=[], leaf=False, unrecorded=0.0)
        )
        group['reach'] += stop['reach']
        if stop['type'] == 'theory_leaf':
            group['leaf'] = True
        elif stop['type'] == 'no_recorded_continuation':
            group['unrecorded'] += stop['reach']  # database results without an individual move row
        else:
            group['replies'].append(stop)
        if stop.get('score') is not None:
            group['known'] += stop['reach']
            group['score_mass'] += stop['reach'] * stop['score']
    rows: list[JsonObject] = []
    for group in groups.values():
        board = positions.get(group['position'], {})
        node = board.get('reach')
        rows.append(
            dict(
                group,
                line=board.get('line', ''),
                kind=board.get('kind'),
                node_reach=node,
                share=group['reach'] / node if node else None,
                score=group['score_mass'] / group['known'] if group['known'] > 0 else None,
                is_starting_position=board.get('is_starting_position', False),
                chapter_attribution=board.get('chapter_attribution'),
                replies=sorted(group['replies'], key=lambda r: (-r['reach'], r['line'])),
            )
        )
    return sorted(rows, key=lambda r: (-r['reach'], r['line'], r['position']))
