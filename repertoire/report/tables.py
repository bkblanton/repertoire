"""Reusable tables of moves, replies, positions, chapters and openings."""

from .format import (
    count,
    delta_points,
    escape,
    gain_split,
    gap_percentage,
    interval_cell,
    line,
    move_reach_label,
    number,
    opponent_rating,
    per_thousand,
    percentage,
    position_reach_label,
    rating_difference,
    reach_cell,
    score_points,
    spread_display,
)
from .links import linked_line
from .markdown import SourceCell, drop_uniform, table


def chapter_table(report, refs, detailed=True):
    """One row per chapter: its page link, Lichess study link, and entry-conditional metrics."""
    alternatives = any(c.get('policy_overrides') for c in report['chapters'])
    rows = []
    for c in report['chapters']:
        s, base = c['score'], c.get('entry_baseline', {})
        index = refs.entries[c['id']][0]
        title = f"[{report['color'][0].upper()}{index}. {escape(c['name'])}](#{refs.anchor(c['id'])})"
        if c.get('policy_overrides'):
            title += ' **(alternative)**'
        if refs.study_link(c['id']):
            title += '<br>' + refs.study_link(c['id'])
        reach = percentage(s.get('entry_probability'))
        if alternatives:
            reach += '<br>Overall ' + percentage(s.get('overall_policy_entry_probability', s.get('entry_probability')))
        gaps = gap_percentage(c.get('gap_coverage'))
        if detailed:
            gaps += '<br>Weighted ' + gap_percentage(c.get('gap_coverage'), weighted=True)
        rows.append(
            [
                title,
                reach,
                percentage(base.get('raw_score')),
                percentage(s.get('raw_empirical_score')),
                score_points(base.get('difference_pp'), signed=True),
                number(s.get('prepared_depth', {}).get('expected_moves'), 1),
                spread_display(s.get('branch_score_spread'), False),
                gaps,
                opponent_rating(c.get('opponent_ratings', {}).get('score_evidence')),
            ]
        )
    return table(
        [
            'Chapter',
            'Chapter reach' + ('<br>Overall-policy reach' if alternatives else ''),
            'Entry baseline',
            'Repertoire score',
            'Delta',
            'Prepared depth',
            'Score spread',
            'Equivalent gap reach' + ('<br>Weighted contribution' if detailed else ''),
            'Opponent rating',
        ],
        rows,
    )


def own_move_table(rows, scope, refs, strongest=False):
    """Our moves against the database score of the position they are played from."""
    refs = refs.for_scope(scope.get('id'))
    return table(
        [
            'Line',
            'Chapter source',
            move_reach_label(scope),
            'Database score<br>Position / move',
            'Repertoire score',
            'Score spread',
            'Gain' if strongest else 'Drag',
            'Gain split',
            'Games',
            'Opponent rating',
        ],
        [
            [
                refs.move_cell(r),
                refs.sources(r),
                reach_cell(r['branch_reach']),
                percentage(r['reference_score']) + ' / ' + percentage(r.get('move_database_score')),
                percentage(r['move_score']),
                spread_display(r.get('move_spread')),
                delta_points(r['move_score'], r['reference_score'], drag=not strongest)
                + '<br>95%: '
                + interval_cell(r.get('local_gain_interval_pp' if strongest else 'local_drop_interval_pp')),
                gain_split(r),
                count(r['parent_sample_count']),
                opponent_rating(r.get('opponent_rating')),
            ]
            for r in rows
        ],
    )


def reply_table(rows, scope, refs, prepared):
    """Unprepared replies stop preparation, so only the spread before the reply is informative."""
    spread_header = 'Score spread<br>Before / after' if prepared else 'Score spread before reply'

    def spread(r):
        before = spread_display(r.get('reference_spread'), False)
        return before + '<br>' + spread_display(r.get('move_spread'), False) if prepared else before

    return table(
        [
            'Line',
            'Chapter source',
            move_reach_label(scope),
            'Reply frequency',
            'Before reply (repertoire)',
            'After reply (repertoire)' if prepared else 'After reply (database)',
            spread_header,
            'Drag',
            'Drag per 1,000 games',
            'Games',
            'Opponent rating',
            'Rating Δ vs parent',
        ],
        [
            [
                refs.move_cell(r),
                refs.sources(r),
                reach_cell(r['branch_reach']),
                percentage(r['branch_probability']),
                percentage(r['reference_score']),
                percentage(r['move_score']),
                spread(r),
                score_points(r['local_drop_pp']) + '<br>95%: ' + interval_cell(r.get('local_drop_interval_pp')),
                per_thousand(r['weighted_drag_pp']),
                count(r['sample_count']),
                opponent_rating(r.get('opponent_rating')),
                rating_difference(r.get('opponent_rating')),
            ]
            for r in rows
        ],
    )


def position_label(row, color, refs=None):
    text = refs.position_cell(row) if refs else line(row['line'])
    if row['kind'] == 'terminal':
        return text + '<br>Game over'
    if row['kind'] in ('theory_leaf', 'unprepared_reply') and row['to_move'] == color:
        return text + f'<br>{color.title()} to move; no prepared reply'
    return text


def position_games(row):
    games = row.get('games')
    if games is None:
        return 'unavailable'
    return count(games) + ('†' if len(row.get('unprepared_origins', [])) > 1 else '')


def position_contribution_table(rows, scope, refs):
    refs = refs.for_scope(scope.get('id'))

    def kind(row):
        if row['kind'] == 'terminal':
            return 'Game over'
        if row['score_basis'] == 'database':
            return 'Unprepared reply'
        return 'Prepared endpoint' if row['kind'] == 'theory_leaf' else 'Prepared position'

    headers, values = drop_uniform(
        [
            'Position',
            'Chapter source',
            'Position type',
            position_reach_label(scope),
            'Score',
            'Score spread',
            'Contribution per 1,000 games',
            'Games',
            'Opponent rating',
        ],
        [
            [
                position_label(r, refs.color, refs),
                refs.sources(r),
                kind(r),
                reach_cell(r['reach']),
                percentage(r['score']),
                spread_display(r.get('branch_score_spread')),
                per_thousand(r['contribution_pp']),
                position_games(r),
                opponent_rating(r.get('opponent_rating')),
            ]
            for r in rows
        ],
        {'Position type'},
    )
    return table(headers, values)


def leading_entries(chapter, scope, refs, top=3):
    refs = refs.for_scope(chapter['id'])
    entries = {e['position']: e for e in chapter['entries']}
    routes = (scope or {}).get('entry_routes', {}).get('positions', [])
    selected = sorted(routes, key=lambda r: -r['conditional_first_entry_weight'])[:top]
    if not selected:
        return []
    rows = []
    for route in selected:
        original = dict(entries[route['position']], _opening_entry=True)
        cell = refs.sources(original)
        mixture = chapter.get('entry_opening_sources', {}).get(route['position'], [])
        if mixture:
            labels = [refs.opening_label(source) for source in mixture]
            cell = SourceCell(str(cell), labels[0], 'Also: ' + '; '.join(labels[1:]) if len(labels) > 1 else '')
        rows.append(
            [
                linked_line(route['example']['line'], route['position']),
                cell,
                percentage(route['conditional_first_entry_weight']),
                opponent_rating(original.get('opponent_rating')),
            ]
        )
    return [
        *table(['Entry position (example route)', 'Chapter source', 'Share of entries', 'Opponent rating'], rows),
        'Each line is one real route to its entry position; the share counts every move order. '
        f'[All entry positions and routes](#{refs.anchor(chapter["id"])}-entries).',
        '',
    ]


def summary_own_priorities(rows, refs, strongest=False, show_components=True):
    field, label = ('local_gain_pp', 'Gain') if strongest else ('local_drop_pp', 'Drag')
    return table(
        [
            'Line',
            'Chapter source',
            'Move reach',
            'Repertoire score',
            'Score spread',
            label,
            label + ' per 1,000 games',
            'Opponent rating',
        ],
        [
            [
                refs.move_cell(r),
                refs.sources(r),
                reach_cell(r['branch_reach']),
                percentage(r['move_score']),
                spread_display(r.get('move_spread'), show_components),
                delta_points(r['move_score'], r['reference_score'], drag=not strongest)
                + (
                    '<br>Move '
                    + score_points(r['database_move_gain_pp'], signed=True)
                    + ', prep '
                    + score_points(r['continuation_gain_pp'], signed=True)
                    if show_components and r.get('database_move_gain_pp') is not None
                    else ''
                )
                + '<br>95%: '
                + interval_cell(r.get('local_gain_interval_pp' if strongest else 'local_drop_interval_pp')),
                per_thousand(r['branch_reach'] * r[field], signed=strongest),
                opponent_rating(r.get('opponent_rating')),
            ]
            for r in rows
        ],
    )


def opening_table(rows, refs, anchors):
    values = []
    for row in rows:
        base, repertoire = row['entry_baseline']['raw_score'], row['repertoire_score']
        values.append(
            [
                f"[{escape(row['name'])}](#{anchors[row['id']]})",
                escape(row['eco']),
                percentage(row['reach']),
                percentage(base),
                percentage(repertoire),
                score_points(row['difference_pp'], signed=True),
                spread_display(row.get('branch_score_spread'), False),
                number(row['expected_prepared_moves'], 1),
                gap_percentage(row['gap_coverage']),
                refs.sources({'chapter_attribution': {'source_ids': row['chapter_ids']}}),
            ]
        )
    return table(
        [
            'Opening',
            'ECO',
            'Reach',
            'Entry baseline',
            'Repertoire score',
            'Delta',
            'Score spread',
            'Prepared depth',
            'Equivalent gap reach',
            'Chapters',
        ],
        values,
    )
