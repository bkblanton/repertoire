"""Report sections shared by the full report, the summary and the chapter pages."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from ..board_cache import san
from ..preparation import line_text
from ..schema import JsonObject
from ..stats import cell
from ..status import Status
from .bundle import Bundle, scope_by_id
from .derive import (
    combined_overall,
    exit_points,
    non_sparse_rows,
    position_contributions,
    unanswered_position,
    visible_positions,
)
from .format import (
    _display,
    count,
    elo_equivalent,
    escape,
    evidence_date,
    gap_percentage,
    headline_delta,
    interval_cell,
    line,
    number,
    opponent_rating,
    per_thousand,
    percentage,
    plies,
    population_text,
    position_reach_label,
    rating_cell,
    rating_difference,
    reach_cell,
    score_points,
    spread_display,
)
from .links import Chapters, analysis_url, bundle_refs, exit_reply, linked_line
from .markdown import Cell, about, drop_uniform, section, table
from .tables import (
    opening_table,
    own_move_table,
    position_contribution_table,
    position_games,
    position_label,
    reply_table,
)


def evidence_snapshot(bundles: Sequence[Bundle]) -> list[str]:
    rows: list[list[str]] = []
    for bundle in bundles:
        overall = bundle['report']['overall']

        def score_range(bounds: Sequence[float] | None) -> str:
            return ' to '.join(percentage(v) for v in bounds) if bounds else 'unavailable'

        rows.append(
            [
                bundle['report']['color'].title(),
                score_range(overall.get('posterior', {}).get('credible_interval_95')),
                score_range(overall.get('sparse_sensitivity')),
                percentage(overall.get('sparse_mass')),
            ]
        )
    return [
        '**Score uncertainty**',
        '',
        'The 95% interval covers sampling in the database games. The sparse-evidence range lets every thinly '
        'sampled outcome take any score from 0% to 100%: a stress test, not an interval. '
        f'{about("evidence", "Limits and definitions")}.',
        '',
        *table(
            ['Repertoire', 'Approximate 95% score interval', 'Sparse-evidence sensitivity', 'Sparse probability'], rows
        ),
    ]


def overview(bundles: Sequence[Bundle], headline_only: bool = False) -> list[str]:
    """Headline rows; the delta keeps its CP translation and Elo uses one fewer decimal than scores."""
    elo_digits = max(0, _display.get()['digits'] - 1)

    def elo_text(value: float | None) -> str:
        return 'unavailable' if value is None else number(value, elo_digits, signed=True)

    rows: list[list[Cell]] = []
    for b in bundles:
        r = b['report']
        o = r['overall']
        base = r.get('starting_position_reference', {}).get('owner_score')
        score = o.get('raw_empirical_score')
        rows.append(
            [
                r['color'].title(),
                percentage(base),
                percentage(score),
                headline_delta(score, base),
                elo_text(elo_equivalent(score, base)),
                number(o.get('prepared_depth', {}).get('expected_moves'), 1),
                spread_display(o.get('branch_score_spread'), False),
                len(r['chapters']),
            ]
        )
    combined = combined_overall(bundles)
    if combined and 'unavailable' not in combined:
        rows.append(
            [
                '**Combined**',
                percentage(combined['starting_baseline']),
                percentage(combined['repertoire_score']),
                headline_delta(combined['repertoire_score'], combined['starting_baseline']),
                elo_text(combined['elo_equivalent']),
                number(combined['expected_prepared_depth'], 1),
                spread_display(combined.get('branch_score_spread'), False),
                combined['chapters'],
            ]
        )
    headers = [
        'Repertoire',
        'Starting baseline',
        'Repertoire score',
        'Delta',
        'Elo equivalent',
        'Prepared depth (own moves)',
        'Score spread',
        'Chapters',
    ]
    if headline_only:
        columns = (0, 1, 2, 3, 4, 5)
        headers = [headers[i] for i in columns]
        rows = [[row[i] for i in columns] for row in rows]
    return table(headers, rows)


def overview_notes(bundles: Sequence[Bundle]) -> str:
    text = (
        'Delta is the repertoire score minus the database score from the starting '
        'position, in percentage points; its cp and Elo equivalents put it on familiar scales but are not engine '
        'evaluations or rating forecasts. '
    )
    combined = combined_overall(bundles)
    if combined:
        if 'unavailable' in combined:
            text += 'Combined score unavailable: ' + combined['unavailable'] + '. '
        else:
            text += 'Combined gives White and Black equal weight (50% each). '
    return text + f'Prepared depth counts your own moves before preparation ends. {about("score", "Definitions")}.'


def snapshot_notes(bundles: Sequence[Bundle]) -> list[str]:
    text: list[str] = []
    for b in bundles:
        r = b['report']
        color = r['color'].title()
        if not b['current_source']:
            state = 'has changed' if Path(r['manifest']['input_path']).exists() else 'is missing'
            text += [
                f'**{color} snapshot notice:** the source PGN {state}. These results '
                'describe the saved analysis, not the current PGN.',
                '',
            ]
        timestamp = r['manifest'].get('created_at', 'date unavailable')
        try:
            timestamp = datetime.fromisoformat(timestamp).astimezone(UTC).strftime('%Y-%m-%d %H:%M UTC')
        except ValueError:
            pass
        text += [
            f"{color}: scored {timestamp}; database evidence retrieved {evidence_date(r)}. {population_text(r)}",
            '',
        ]
        if b['unavailable']:
            text += [
                f"**{color} analysis pending:** " + '; '.join(f'{k}: {v}' for k, v in b['unavailable'].items()) + '.',
                '',
            ]
    return text


def gap_section(
    scope: JsonObject | None,
    level: str = '###',
    refs: Chapters | None = None,
    top: int = 10,
    anchor: str | None = None,
) -> list[str]:
    metrics = (scope or {}).get('gap_coverage')
    if not metrics:
        return [
            *section(f'{level} Equivalent gap reach', anchor),
            'Unavailable; regenerate the cache-only character analysis.',
            '',
        ]
    text = [
        *section(f'{level} Equivalent gap reach', anchor),
        f'**{gap_percentage(metrics)}**: the reach of a single gap that would repeat as often as all of your '
        f'first gaps together. Lower means gaps are rarer or more spread out. {about("gap-reach")}.',
        '',
    ]
    if metrics.get('unresolved_mass', 0):
        text += [
            f"First-gap reach is unresolved for {percentage(metrics['unresolved_mass'])} of games in this scope. "
            'The displayed range bounds the missing response data; it is not a sampling confidence interval.',
            '',
        ]
    priorities = (scope or {}).get('gap_priorities', {})
    positions = {r['position']: r for r in (scope or {}).get('positions', [])}
    selected = priorities.get('priorities', [])[:top]
    if selected and refs is not None:
        refs = refs.for_scope(cast(JsonObject, scope).get('id'))
        rows: list[list[str]] = []
        for priority in selected:
            row = positions[priority['position']]
            rows.append(
                [
                    refs.position_cell(row),
                    refs.sources(row),
                    percentage(priority['reach']),
                    percentage(priority['repeat_probability_share']),
                    percentage(row.get('database_score')),
                    position_games(row),
                    rating_cell(row.get('opponent_rating')),
                ]
            )
        text += [
            '**Gaps driving repeat encounters**',
            '',
            *table(
                [
                    'First-gap position',
                    'Chapter source',
                    position_reach_label(cast(JsonObject, scope)),
                    'Repeat-gap share',
                    'Database score',
                    'Games',
                    'Opponent rating',
                ],
                rows,
            ),
            f"These {len(selected)} gaps account for "
            f"**{percentage(selected[-1]['cumulative_repeat_probability_share'])}** "
            f"of repeat-gap probability ({priorities['share_basis']}). "
            f"{about('gap-priorities', 'Gap priority definition')}.",
            '',
        ]
    return text


def branch_spread_section(scope: JsonObject | None, level: str = '###', anchor: str | None = None) -> list[str]:
    spread = (scope or {}).get('branch_score_spread')
    if not spread:
        return []
    return [
        *section(f'{level} Branch score spread', anchor),
        'How much expected scores differ between the branches of your preparation, including later replies. '
        'Outcome volatility also counts how much game results vary once preparation ends. '
        f'{about("spread")}.',
        '',
        *table(
            ['Measure', 'Value'],
            [
                [name, percentage(spread.get(field))]
                for name, field in (
                    ('Total branch spread', 'standard_deviation'),
                    ('Between stopping positions', 'between_position_deviation'),
                    ('Between arrival cohorts at the same position', 'within_position_deviation'),
                )
            ]
            + [
                [
                    'Outcome volatility (0%-100%)',
                    percentage(4 * spread['outcome_variance'])
                    if spread.get('outcome_variance') is not None
                    else 'unresolved',
                ],
                [
                    'Branch share of outcome variance',
                    percentage(spread['variance'] / spread['outcome_variance'])
                    if spread.get('outcome_variance') and spread.get('variance') is not None
                    else 'n/a',
                ],
            ],
        ),
    ]


def character_section(
    scope: JsonObject | None, refs: Chapters, top: int, level: str = '###', anchor: str | None = None
) -> list[str]:
    if not scope or 'reuse' not in scope:
        return []
    refs = refs.for_scope(scope.get('id'))
    reuse, p = scope['reuse'], scope['predictability']
    profile = scope['position_profiles']['all']
    text = [
        *section(f'{level} Preparation, replies, and resulting positions', anchor),
        f"**{reuse['reachable_distinct_decisions']:,} distinct own decisions**, "
        f"**{number(p['effective_replies'], 1)} effective opponent replies**, and "
        f"**{number(profile['effective_pawn_structures'], 1)} effective pawn structures** where preparation ends. "
        f"Reply data cover {percentage(p['recorded_reply_coverage'])} of reached opponent opportunities; "
        f"{percentage(p['sparse_recorded_opportunity_fraction'])} of recorded opportunities have sparse samples.",
        '',
    ]
    text += table(
        ['Games entering this scope', 'Distinct decisions encountered', 'Still unseen', 'Repeat encounters'],
        [
            [
                f"{row['games']:,}",
                number(row['expected_distinct_decisions'], 1),
                number(row['expected_unseen_decisions'], 1),
                f"{row['expected_repeat_encounters']:,.0f}",
            ]
            for row in reuse['curve']
        ],
    )
    text += ['Opponent positions with the widest choice of replies, weighted by reach:', '']
    text += table(
        [
            'Position',
            'Chapter source',
            'Reach',
            'Effective replies',
            'Most common reply',
            'Games',
            'Opponent rating',
        ],
        [
            [
                refs.position_cell(r),
                refs.sources(r),
                percentage(r['reach']),
                number(r['effective_replies'], 1),
                _common_reply(r['replies'][0]) if r['replies'] else 'unavailable',
                count(r['recorded_reply_observations']) + (' *' if r['sparse'] else ''),
                opponent_rating(r.get('opponent_rating')),
            ]
            for r in p['positions'][:top]
        ],
    )
    text += ['The positions where preparation ends:', '']
    features: list[list[str]] = []
    for name in ('queens', 'king_placement'):
        features.extend(
            [escape(r['value']).capitalize(), percentage(r['probability'])]
            for r in profile['distributions'].get(name, [])
        )
    from ..character import FEATURE_LABELS

    features.extend([FEATURE_LABELS[name], percentage(value)] for name, value in profile['features'].items())
    text += table(['Feature', 'Share of stopping positions'], features)
    text += table(
        ['Stopping type', 'Probability'],
        [
            [kind.replace('_', ' ').capitalize(), percentage(sub['scope_mass'])]
            for kind, sub in scope['position_profiles'].items()
            if kind != 'all'
        ],
    )
    text += table(
        ['Frequent pawn structure', 'Share', 'Example route', 'Chapter source', 'Example opponent rating']
        + (['Group opponent rating'] if scope.get('id') != 'overall' else []),
        [
            [
                escape(r['value']),
                percentage(r['probability']),
                linked_line(r['example_line'], r.get('example_position')),
                refs.sources(
                    {'position': r.get('example_position'), 'chapter_attribution': r.get('example_chapter_attribution')}
                ),
                opponent_rating(r.get('opponent_rating')),
            ]
            + ([opponent_rating(r.get('group_opponent_rating'))] if scope.get('id') != 'overall' else [])
            for r in profile['distributions'].get('pawn_structure', [])[:top]
        ],
    )
    return text


def _common_reply(reply: JsonObject) -> str:
    text = escape(reply['san']) + ' (' + percentage(reply['probability_given_recorded_reply']) + ')'
    difference = rating_difference(reply.get('opponent_rating'))
    return text if difference.startswith(('n/a', 'unavailable')) else f'{text}<br>{difference} rating vs parent'


def vulnerabilities_section(
    scope: JsonObject | None, refs: Chapters, top: int, level: str = '###', anchor: str | None = None
) -> list[str]:
    if not scope:
        return []
    refs = refs.for_scope(scope.get('id'))
    text = [
        *section(f'{level} Vulnerabilities', anchor),
        'Drag is how far a move lowers the expected score. Opponent replies rank by drag per 1,000 games, which '
        'also counts how often they come up; your moves rank by drag against the database score of the position '
        f'they are played from. Rows overlap, so do not add them. {about("drag")}.',
        '',
    ]
    for prepared, name in ((False, 'Unprepared opponent replies'), (True, 'Prepared opponent replies')):
        rows = [r for r in non_sparse_rows(scope.get('rankings', {}).get('opponent', [])) if r['prepared'] == prepared][
            :top
        ]
        if not rows:
            continue
        text += [f'**{name}**', '', *reply_table(rows, scope, refs, prepared)]
    rows = non_sparse_rows(scope.get('rankings', {}).get('own', []))[:top]
    if rows:
        text += [
            '**Your moves**',
            '',
            'Drag = database score of the position − repertoire score after your move. '
            'Gain split shows how much comes from the move itself and how much from later preparation.',
            '',
        ]
        text += own_move_table(rows, scope, refs)
    if scope.get('unresolved_rows'):
        text += [
            f"{len(scope['unresolved_rows'])} reachable comparisons remain "
            "unresolved and are excluded from these rankings.",
            '',
        ]
    return text


def strengths_section(
    moves: JsonObject | None,
    positions: JsonObject | None,
    refs: Chapters,
    top: int,
    level: str = '###',
    sparse_threshold: int = 30,
    anchor: str | None = None,
) -> list[str]:
    if not moves and not positions:
        return []
    text = section(f'{level} Strengths', anchor)
    rows = non_sparse_rows((moves or {}).get('strengths', []))[:top]
    if rows:
        text += [
            '**Your strongest moves**',
            '',
            'Gain = repertoire score after your move − database score of the position.',
            '',
        ]
        text += own_move_table(rows, cast(JsonObject, moves), refs, strongest=True)
    rows = non_sparse_rows(position_contributions(positions, refs.color, sparse_threshold))[:top]
    if rows:
        text += [
            '**Positions contributing the most to the repertoire score**',
            '',
            'Contribution = reach × score, in points per 1,000 games, for prepared positions and unprepared replies. '
            f'Rows overlap, so do not add them. {about("contribution")}.',
            '',
        ]
        text += position_contribution_table(rows, cast(JsonObject, positions), refs)
    elif positions is not None and 'positions' not in positions:
        text += [
            'Position contributions are unavailable in the saved results. '
            'Regenerate the character analysis to include this ranking.',
            '',
        ]
    return text


def edge_rows(scope: JsonObject | None, gains: bool) -> list[JsonObject]:
    """Your moves by their share of the edge, largest first; thinly sampled comparisons are left out."""
    rows = [
        r
        for r in non_sparse_rows((scope or {}).get('all_signed_rows', []))
        if r['kind'] == 'own' and r.get('edge_pp') is not None and (r['edge_pp'] > 0 if gains else r['edge_pp'] < 0)
    ]
    return sorted(rows, key=lambda r: (-abs(r['edge_pp']), r['line']))


def edge_table(rows: Sequence[JsonObject], scope: JsonObject, refs: Chapters) -> list[str]:
    def move_gain(r: JsonObject) -> str:
        gain = r.get('database_move_gain_pp')
        if gain is None:
            gain = 100 * (r['move_database_score'] - r['reference_score'])
        return score_points(gain, signed=True) + '<br>95%: ' + interval_cell(r.get('database_move_gain_interval_pp'))

    refs = refs.for_scope(scope.get('id'))
    return table(
        ['Line', 'Chapter source', 'Move reach', 'Move gain', 'Edge per 1,000 games'],
        [
            [
                refs.move_cell(r),
                refs.sources(r),
                reach_cell(r['branch_reach']),
                move_gain(r),
                per_thousand(r['edge_pp'], signed=True),
            ]
            for r in rows
        ],
    )


def other_edge_pp(ledger: JsonObject) -> float:
    """Everything but your moves: move orders at transpositions, theory leaves and finished games."""
    return ledger['move_orders_pp'] + ledger['theory_leaves_pp'] + ledger['finished_games_pp']


def edge_totals(ledger: JsonObject) -> list[str]:
    rows = [[f"Your moves ({ledger['decision_count']:,})", per_thousand(ledger['decisions_pp'], signed=True)]]
    for key, label in (
        ('move_orders_pp', 'Move orders at transpositions'),
        ('theory_leaves_pp', 'Theory leaves'),
        ('finished_games_pp', 'Finished games'),
    ):
        if abs(ledger[key]) > 1e-12:
            rows.append([label, per_thousand(ledger[key], signed=True)])
    rows.append(['**Total: the delta**', '**' + per_thousand(ledger['delta_pp'], signed=True) + '**'])
    return table(['Part', 'Per 1,000 games'], rows)


def edge_sentence(ledger: JsonObject, shares: bool = True) -> str:
    """What your moves cost, and with `shares` how concentrated the edge is; shares need a positive edge, and
    move numbers mean little on chapter pages, which start part-way into a game."""
    costs = (
        f"{ledger['costing_count']:,} of your moves score below their position, costing "
        f"{per_thousand(-ledger['costs_pp'])} per 1,000 games in all."
    )
    delta = ledger['delta_pp']
    if not shares or delta <= 0:
        return costs
    middle = sum(g['edge_pp'] for g in ledger['by_move_number'] if g['moves'] in ('2-3', '4-6'))
    return (
        f"Your top {ledger['top_decisions']} moves earn {percentage(ledger['top_decisions_pp'] / delta)} of the "
        f'edge, and your moves 2 to 6 earn {percentage(middle / delta)}. {costs}'
    )


def summary_edge_section(scope: JsonObject | None, refs: Chapters, color: str, top: int = 5) -> list[str]:
    ledger = (scope or {}).get('edge') or {}
    rows = edge_rows(scope, gains=True)[:top]
    if ledger.get('status') != Status.RESOLVED or not rows:
        return []
    return [
        '<details>',
        '<summary>Where your edge comes from</summary>',
        '',
        'Each of your moves earns how often you play it times its move gain: the database score after the move '
        f'minus the database score of the position. Unlike other tables, these rows add up. {about("edge")}.',
        '',
        *edge_table(rows, cast(JsonObject, scope), refs),
        f"Your {ledger['decision_count']:,} moves add up to {per_thousand(ledger['decisions_pp'], signed=True)} "
        f'per 1,000 games and move orders at transpositions to {per_thousand(other_edge_pp(ledger), signed=True)}, '
        f"for the delta of {per_thousand(ledger['delta_pp'], signed=True)}. {edge_sentence(ledger)}",
        '',
        f'[All of your edge, move by move](#{color}-edge).',
        '',
        '</details>',
        '',
    ]


def edge_section(
    scope: JsonObject | None,
    refs: Chapters,
    top: int,
    level: str = '###',
    anchor: str | None = None,
    detailed: bool = True,
) -> list[str]:
    """The delta split into parts that add up; `detailed` adds the move-number and transposition tables."""
    ledger = (scope or {}).get('edge') or {}
    if ledger.get('status') != Status.RESOLVED:
        return []
    scope = cast(JsonObject, scope)
    text = [
        *section(f'{level} Where your edge comes from', anchor),
        'The delta splits into parts that add up. Each of your moves earns how often you play it times its move '
        'gain: the database score after the move minus the database score of the position. Unlike the gains in '
        'the strengths tables, these do not overlap. Move orders at transpositions account for the database '
        'pooling every move order into a position while your repertoire arrives in its own proportions. '
        f'{about("edge")}.',
        '',
        *edge_totals(ledger),
        edge_sentence(ledger, shares=detailed),
        '',
    ]
    if detailed:
        text += table(
            ['Your move number', 'Per 1,000 games'],
            [[g['moves'], per_thousand(g['edge_pp'], signed=True)] for g in ledger['by_move_number']],
        )
    for gains, name in ((True, 'Moves earning the most'), (False, 'Moves costing the most')):
        rows = edge_rows(scope, gains)[:top]
        if rows:
            text += [f'**{name}**', '', *edge_table(rows, scope, refs)]
    shown = ledger['transpositions'][:top] if detailed else []
    if shown:
        text += [
            '<details>',
            '<summary>Move orders at transpositions</summary>',
            '',
            'Positions your repertoire reaches through several move orders in other proportions than database '
            "games do. Each arrival shows its share of the position's reach and the database score of its move.",
            '',
            *table(
                ['Position', 'Arrivals', 'Effect per 1,000 games'],
                [
                    [
                        refs.position_cell(t),
                        '<br>'.join(arrival_text(a, t, refs) for a in t['arrivals']),
                        per_thousand(t['effect_pp'], signed=True),
                    ]
                    for t in shown
                ],
            ),
            '</details>',
            '',
        ]
    return text


def arrival_text(arrival: JsonObject, transposition: JsonObject, refs: Chapters) -> str:
    total = sum(a['reach'] for a in transposition['arrivals'])
    share = percentage(arrival['reach'] / total) if total else 'n/a'
    if arrival['parent_position'] is None:
        label = 'chapter entry'
    else:
        parent = arrival['parent_position']
        label = line(
            refs.move_route(dict(position=parent, move_san=san(parent, arrival['move']), line=arrival['line']))
        )
    return f"{label}: {share} at {percentage(arrival['score'])}"


def rare_moves_metric(scope: JsonObject | None) -> list[str] | None:
    """The summary row on moves met less than once per 1,000 games, as a share of the edge when it is positive."""
    effort = (scope or {}).get('effort')
    if not effort:
        return None
    rare, total, delta = effort['rare_decisions'], effort['decisions'], effort.get('delta_pp')
    earned = (
        f'{percentage(effort["rare_edge_pp"] / delta)} of the edge'
        if delta is not None and delta > 0
        else f'{per_thousand(effort["rare_edge_pp"], signed=True)} per 1,000 games'
    )
    games = f'{round(1 / effort["rare_reach"]):,}'
    return [f'Moves met less than once in {games} games', f'{rare:,} of {total:,}, earning {earned}']


def effort_section(
    bundle: Bundle, refs: Chapters, top: int, level: str = '###', anchor: str | None = None
) -> list[str]:
    scope = bundle.get('vulnerabilities', {}).get('overall') or {}
    effort = scope.get('effort')
    if not effort:
        return []
    rows = {r['id']: r for r in scope.get('all_signed_rows', [])}
    color = refs.color
    text = [
        *section(f'{level} Effort and value', anchor),
        'What your preparation earns for the moves it takes to know. Edge per 1,000 games is the same as in '
        f'[where your edge comes from](#{color}-edge). {about("effort")}.',
        '',
    ]
    chapters = [c for c in effort.get('chapters', []) if c.get('edge_pp') is not None and c['decisions']]
    if chapters:
        text += [
            '**Chapters by edge per move**',
            '',
            "A chapter's edge is its chance of being entered times its delta; its moves are the ones it records "
            'that games reach. A move recorded in several chapters counts in each.',
            '',
            *table(
                ['Chapter', 'Moves', 'Edge per 1,000 games', 'Per move'],
                [
                    [
                        refs.label(c['id'], True),
                        f"{c['decisions']:,}",
                        per_thousand(c['edge_pp'], signed=True),
                        per_thousand(c['edge_per_decision_pp'], signed=True),
                    ]
                    for c in sorted(chapters, key=lambda c: (-c['edge_per_decision_pp'], c['id']))
                ],
            ),
        ]
    pruning = [rows[i] for i in effort.get('pruning', []) if i in rows][:top]
    if pruning:
        text += [
            '**Lines to consider pruning**',
            '',
            'Each row is one of your moves with every move that can only be reached through it. Value is what '
            'dropping them would lose, reach × gain, and rows are ranked by value per move. Rows do not overlap. '
            f'Lines with at least {effort["prune_min_decisions"]} moves are shown; moves that lose score on their '
            f'own are in [vulnerabilities](#{color}-vulnerabilities).',
            '',
            *table(
                ['Line', 'Chapter source', 'Move reach', 'Moves', 'Value per 1,000 games'],
                [
                    [
                        refs.move_cell(r),
                        refs.sources(r),
                        reach_cell(r['branch_reach']),
                        f"{r['decisions_dropped']:,}",
                        per_thousand(r['drop_value_pp'], signed=True),
                    ]
                    for r in pruning
                ],
            ),
        ]
    review = [rows[i] for i in effort.get('review', []) if i in rows][:top]
    if review:
        games = effort['recent_games']
        text += [
            '**Valuable moves you rarely play**',
            '',
            'Moves worth reviewing because they come up too seldom to stay fresh through play: ranked by reach × gain '
            f'times the chance that none of your last {games} games reached them.',
            '',
            *table(
                ['Line', 'Chapter source', 'Move reach', 'Gain', f'Not met in {games} games'],
                [
                    [
                        refs.move_cell(r),
                        refs.sources(r),
                        reach_cell(r['branch_reach']),
                        score_points(r['local_gain_pp'], signed=True)
                        + '<br>95%: '
                        + interval_cell(r.get('local_gain_interval_pp')),
                        percentage((1 - r['branch_reach']) ** games),
                    ]
                    for r in review
                ],
            ),
        ]
    return text


def transposing_cell(row: JsonObject, refs: Chapters) -> str:
    """Your move after the reply, linked to the prepared position it reaches, and the chapters holding it."""
    route = refs.move_route(row)
    number = plies(route) // 2 + 1
    token = (
        f"{number}.{row['transposing_san']}"
        if row['exit_position'].split()[1] == 'w'
        else (f"{number}...{row['transposing_san']}")
    )
    chapters = refs.chapter_sources({'chapter_attribution': {'source_ids': row['target_chapters']}})
    link = analysis_url(row['transposition_target'], f'{route} {token}')
    return f'[{line(token)}]({link})<br>to {chapters}'


def change_cell(row: JsonObject) -> str:
    return (
        score_points(row['change_pp'], signed=True)
        + f"<br>Move {score_points(row['move_change_pp'], signed=True)}, "
        + f"prep {score_points(row['preparation_change_pp'], signed=True)}"
        + '<br>95%: '
        + interval_cell(row['change_interval_pp'])
    )


def free_transpositions_section(
    scope: JsonObject | None,
    refs: Chapters,
    top: int,
    level: str = '###',
    anchor: str | None = None,
    chapter: str | None = None,
) -> list[str]:
    """Unprepared replies after which one of your moves reaches prepared positions; on a chapter page, the
    replies that leave that chapter."""
    rows = [r for r in (scope or {}).get('free_transpositions', []) if not r.get('sparse')]
    if chapter is not None:
        rows = [r for r in rows if chapter in (r.get('chapter_attribution') or {}).get('context_ids', [])]
    if not rows:
        return []
    text = [
        *section(f'{level} Free transpositions', anchor),
        'Unprepared replies after which one of your moves reaches a position you have prepared. Change is your '
        'score there minus the database score after the reply, which is how the reports count these games now; it '
        'splits into the move and your preparation after it. The comparison is with average play after the reply, '
        'not your best move there, so a negative change is a reason not to transpose and a positive one only '
        f'says it beats average play. {about("free-transpositions")}.',
        '',
    ]
    for raise_score, name in ((True, 'Transposing raises your score'), (False, 'Transposing lowers your score')):
        shown = [r for r in rows if (r['change_pp'] > 0) == raise_score][:top]
        if not shown:
            continue
        text += [
            f'**{name}**',
            '',
            *table(
                ['Line', 'Chapter source', 'Games leaving prep here', 'Transpose with', 'Change'],
                [
                    [
                        refs.move_cell(r),
                        refs.sources(r),
                        reach_cell(r['reach']),
                        transposing_cell(r, refs),
                        change_cell(r),
                    ]
                    for r in shown
                ],
            ),
        ]
    return text


def alternatives_section(
    report: JsonObject, refs: Chapters, level: str = '###', anchor: str | None = None
) -> list[str]:
    """Boards where chapters compete: every alternative's score there, and which one the repertoire plays."""
    rows: list[list[str]] = []
    for alternative in report.get('alternatives', []):
        played = next(o for o in alternative['options'] if o['move'] == alternative['selected'])
        route = line_text(alternative['path'])
        for i, option in enumerate(sorted(alternative['options'], key=lambda o: o['move'] != alternative['selected'])):
            difference = (
                None if option['score'] is None or played['score'] is None else option['score'] - played['score']
            )
            weighted = None if difference is None or alternative['reach'] is None else alternative['reach'] * difference
            rows.append(
                [
                    refs.position_cell({'position': alternative['position'], 'line': route}) if i == 0 else '',
                    percentage(alternative['reach']) if i == 0 else '',
                    escape(option['san']) + (' **(played)**' if option is played else ''),
                    ', '.join(refs.label(cid) for cid in option['chapters']),
                    percentage(option['score']),
                    ''
                    if option is played
                    else score_points(None if difference is None else 100 * difference, signed=True),
                    '' if option is played else per_thousand(None if weighted is None else 100 * weighted, signed=True),
                ]
            )
    if not rows:
        return []
    color = report['color']
    return [
        *section(f'{level} {color.title()} competing alternatives', anchor or f'{color}-alternatives'),
        'Where chapters record different first moves, each alternative is scored with the best choices after it '
        'and the highest-scoring one is played. The other rows show what each alternative would change: at the '
        f'position, and in points per 1,000 {color.title()} games. {about("competing")}.',
        '',
        *table(
            [
                'Position',
                'Position reach',
                'Your move',
                'Chapters',
                'Repertoire score',
                'Difference vs played',
                'Points per 1,000 games',
            ],
            rows,
        ),
    ]


def load_comparisons(bundles: Sequence[Bundle]) -> list[JsonObject]:
    """Saved comparison results beside the scores, each marked current when made from the same score file."""
    if not bundles:
        return []
    digests = {b['report']['color']: b['digest'] for b in bundles}
    result: list[JsonObject] = []
    for path in sorted((bundles[0]['path'].parent / 'comparisons').glob('*.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except ValueError:
            continue
        if data.get('kind') != 'comparison' or data.get('color') not in digests:
            continue
        data['current'] = data['manifest']['report_sha256'] == digests[data['color']]
        result.append(data)
    return result


def comparisons_section(bundles: Sequence[Bundle], level: str = '##') -> list[str]:
    """One row per saved comparison: its verdict and what the improving choice would change."""
    from .comparison import headline, verdict

    rows: list[list[str]] = []
    for data in load_comparisons(bundles):
        scenarios = data['scenarios']
        status = (
            verdict(data)
            if data['current']
            else '**Out of date:** your repertoire changed; rerun `uv run repertoire compare` to refresh it.'
        )
        rows.append(
            [
                f"[{escape(data['title'])}](@comparisons/{data['name']})",
                data['color'].title(),
                status,
                headline(scenarios['improving']['score'], scenarios['current']['score']),
                headline(scenarios['all']['score'], scenarios['current']['score']),
            ]
        )
    if not rows:
        return []
    return [
        *section(f'{level} Comparisons', 'comparisons'),
        'Candidate studies compared with your repertoire, alternative by alternative. Changes are to the '
        'whole color: adopting only the improving alternatives, or every alternative.',
        '',
        *table(['Comparison', 'Color', 'Verdict', 'Improving only', 'All alternatives'], rows),
    ]


def correlations_section(result: JsonObject | None, reason: str | None) -> list[str]:
    text = section('### Future preparation gain', 'preparation-correlation')
    if not result:
        return text + [f'Correlation analysis unavailable: {reason}.', '']
    text += [
        'Does more preparation after our selected move tend to improve its '
        'score? Each observation is one selected own move under the overall '
        'repertoire policy. '
        '**Future depth** is the expected number of prepared own moves '
        'remaining after that move, excluding the move itself. '
        '**Future preparation gain** is the recursive repertoire score after '
        'the move minus that move\'s database score from the cached parent '
        'response table. '
        'Both scores use the repertoire owner\'s perspective.',
        '',
        'Observations are weighted by their probability of being played. '
        'Canonical decisions are counted once and incoming reach is merged '
        'across transpositions, '
        'so many rare branches do not outweigh common decisions simply because '
        'the tree branches. A game can encounter several decisions, so these '
        'weights need not sum to 100%. '
        'Sparse parent/move samples or continuation endpoints, unresolved '
        'scores and unreachable decisions are excluded. '
        'These are point estimates from the observed database counts.',
        '',
    ]
    primary = ('reach_weighted_pearson', 'reach_weighted_spearman', 'slope_pp_per_move')
    text += table(
        ['Repertoire', 'Own decisions', 'Linear correlation', 'Rank correlation', 'Gain slope (% per future own move)'],
        [[color.title(), f"{r['n']:,}", *[cell(r, k) for k in primary]] for color, r in result['results'].items()],
    )
    text += [
        'Positive correlation means decisions with deeper future preparation '
        'tend to have larger gains over their own move baseline. '
        'The slope is the associated gain per additional expected future own '
        'move. It compares different positions, so it is not a causal estimate '
        'of what adding a move would gain.',
        '',
        '<details>',
        '<summary>Unweighted and positive-depth sensitivity checks</summary>',
        '',
    ]
    text += table(
        [
            'Repertoire',
            'Unweighted linear correlation',
            'Unweighted rank correlation',
            'Positive-depth decisions',
            'Reach-weighted linear, excluding zero depth',
        ],
        [
            [
                color.title(),
                cell(r, 'pearson'),
                cell(r, 'spearman'),
                f"{r['sensitivity_without_zero_depth']['n']:,}",
                (
                    'undefined'
                    if r['sensitivity_without_zero_depth']['reach_weighted_pearson'] is None
                    else f"{r['sensitivity_without_zero_depth']['reach_weighted_pearson']:.3f}"
                ),
            ]
            for color, r in result['results'].items()
        ],
    )
    text += [
        'The positive-depth check excludes decisions with no remaining '
        'prepared own moves and is a point estimate only. '
        'Weighted ranks use the cumulative reach distribution with midpoint ranks for ties.',
        '',
        '</details>',
        '',
        'These describe association across this repertoire\'s decisions. Deeper preparation tends to go with larger '
        'gains, but the slope is not a forecast of the gain from adding preparation.',
        '',
    ]
    return text


MODERATE_CORRELATION = 0.3


def preparation_gain_sentence(result: JsonObject | None, color: str) -> str | None:
    """The summary's one-line reading of the preparation correlation, only when the data supports it.

    Both reach-weighted correlations must be at least moderate, the slope positive, and the association must
    survive dropping decisions with no remaining preparation, so the claim is not carried by the zero-depth cluster.
    """
    r = ((result or {}).get('results') or {}).get(color)
    if not r:
        return None
    pearson = r.get('reach_weighted_pearson')
    spearman = r.get('reach_weighted_spearman')
    slope = r.get('slope_pp_per_move')
    positive_depth = (r.get('sensitivity_without_zero_depth') or {}).get('reach_weighted_pearson')
    if pearson is None or spearman is None or slope is None or positive_depth is None:
        return None
    if min(pearson, spearman) < MODERATE_CORRELATION or slope <= 0 or positive_depth <= 0:
        return None
    return (
        f'Deeper preparation goes with larger gains over the database score (linear correlation {pearson:.2f}; '
        f'about +{percentage(slope / 100)} per additional prepared own move). This is an association across '
        'your decisions, not a forecast of what adding a move would gain '
        '([Future preparation gain](#preparation-correlation)).'
    )


def rating_correlations_section(result: JsonObject | None, reason: str | None) -> list[str]:
    from ..rating_correlations import cell as rating_cell

    text = section('### Opponent rating and score improvement', 'rating-correlations')
    if not result:
        return text + [
            f'Rating correlation analysis unavailable: {reason}. '
            'Generate it with `uv run repertoire rating-correlations` and the saved score files.',
            '',
        ]
    text += [
        'This compares opponent replies from the same parent position. Rating '
        'delta is the reply cohort\'s average opponent rating minus the '
        'reach-weighted mean rating of eligible replies at that parent. Score '
        'delta is its continuation score minus the '
        'corresponding reach-weighted mean continuation score. Prepared replies use the recursive repertoire score; '
        'unprepared replies use their cached parent move results. Weights are parent reach × reply frequency. '
        'Canonical parent/reply pairs are counted once across transpositions; sparse replies are excluded.',
        '',
        'Negative correlation means that higher-rated reply cohorts tend to '
        'leave the repertoire with lower continuation scores. '
        'The slope expresses the associated score difference for +100 rating, in percentage points.',
        '',
    ]
    headers = ['Repertoire', 'Replies / parent positions', 'Linear correlation', 'Score Δ per +100 rating']

    def row(color: str, estimate: JsonObject) -> list[str]:
        return [
            color.title(),
            f"{estimate['replies']:,} / {estimate['parents']:,}",
            rating_cell(estimate),
            rating_cell(estimate, 'slope_percent_per_100_rating', True),
        ]

    text += table(headers, [row(color, data['reply_associations']['all']) for color, data in result['results'].items()])
    text += ['<details>', '<summary>Prepared replies, unprepared replies, and larger samples</summary>', '']
    labels = {
        'prepared': 'Prepared replies',
        'unprepared': 'Unprepared replies',
        'at_least_1000_games': 'Replies with at least 1,000 games',
    }
    text += table(
        [headers[0], 'Reply set', *headers[1:]],
        [
            [color.title(), label, *row(color, data['reply_associations'][key])[1:]]
            for color, data in result['results'].items()
            for key, label in labels.items()
        ],
    )
    text += [
        '</details>',
        '',
        'These are point estimates. Shared downstream evidence and overlapping '
        'historical games can link different parents. '
        'This describes associations between move-maker cohorts, rather than '
        'the effect of changing an individual opponent\'s rating. '
        'The baseline is the local reply mean, and these scores are from the repertoire owner\'s perspective.',
        '',
    ]
    return text


def common_positions_for_scope(
    scope: JsonObject | None, refs: Chapters, limit: int = 20, level: str = '###', anchor: str | None = None
) -> list[str]:
    scope = scope or {}
    refs = refs.for_scope(scope.get('id'))
    color = refs.color
    text = section(f'{level} Most common positions', anchor)
    text += [
        'Reach counts every move order, and a position\'s reach includes the positions after it, so do not add '
        'rows. Prepared positions show the repertoire score; unprepared replies show the database score. '
        f'A position where you always play the same move appears after that move. {about("position-reach")}.',
        '',
    ]
    if 'positions' not in scope:
        return text + [
            'Position reach is not available in the saved results. Regenerate the '
            'character analysis to include this ranking.',
            '',
        ]
    if scope.get('unresolved_opponent_distribution_mass', 0) > 0:
        text += [
            'Only known reach is ranked. Missing opponent distributions leave '
            'downstream routes unresolved, so the ranking can change when evidence '
            'becomes available.',
            '',
        ]
    positions = visible_positions(scope)
    prepared = [r for r in positions if not unanswered_position(r, color)]
    unprepared = [r for r in positions if unanswered_position(r, color)]
    reach = position_reach_label(scope)
    for title, rows in (('Prepared positions', prepared), ('Unprepared opponent replies', unprepared)):
        if not rows:
            continue
        text += [f'{level}# {title}', '']
        if rows is prepared:
            text += table(
                [
                    'Position',
                    'Chapter source',
                    reach,
                    'Repertoire score',
                    'Score spread',
                    'Games',
                    'Opponent rating',
                ],
                [
                    [
                        position_label(r, color, refs),
                        refs.sources(r),
                        reach_cell(r['reach']),
                        percentage(r.get('repertoire_score')),
                        spread_display(r.get('branch_score_spread')),
                        position_games(r),
                        opponent_rating(r.get('opponent_rating')),
                    ]
                    for r in rows[:limit]
                ],
            )
        else:
            # Preparation stops at these boards, so branch spread is zero by definition and is not shown.
            text += table(
                [
                    'Position',
                    'Chapter source',
                    reach,
                    'Database score',
                    'Games',
                    'Opponent rating',
                    'Rating Δ vs parent',
                ],
                [
                    [
                        position_label(r, color, refs),
                        refs.sources(r),
                        reach_cell(r['reach']),
                        percentage(r.get('database_score')),
                        position_games(r),
                        opponent_rating(r.get('opponent_rating')),
                        rating_difference(r.get('opponent_rating')),
                    ]
                    for r in rows[:limit]
                ],
            )
        text += [f'Showing {min(limit, len(rows))} of {len(rows):,} {title.lower()}.', '']
    return text


def common_positions_section(bundles: Sequence[Bundle], limit: int = 20) -> list[str]:
    text: list[str] = []
    for bundle in bundles:
        report = bundle['report']
        scope = scope_by_id(bundle.get('character')).get('overall', {})
        text += common_positions_for_scope(
            scope, bundle_refs(bundle), limit, anchor=f'{report["color"]}-common-positions'
        )
    return text


def exits_section(
    scope: JsonObject | None, refs: Chapters, top: int, level: str = '###', anchor: str | None = None
) -> list[str]:
    scope = scope or {}
    refs = refs.for_scope(scope.get('id'))
    color = refs.color
    overall = scope.get('id', 'overall') == 'overall'
    text = section(f'{level} Where preparation ends', anchor)
    if 'stopping_outcomes' not in scope:
        return text + ['Unavailable; regenerate the cache-only character analysis.', '']
    rows = exit_points(scope)
    if not rows:
        return text + ['No modeled game leaves preparation in this scope.', '']
    shown = rows[:top]
    text += [
        'The prepared positions where games leave your preparation, each combining every unprepared reply from it. '
        f'Every game leaves preparation once, so unlike other tables these rows do not overlap. {about("exits")}.',
        '',
    ]
    values: list[list[str]] = []
    for row in shown:
        route = refs.route(row['position'], row['line'])
        if row['is_starting_position']:
            label = f'[Starting position]({analysis_url(row["position"])})'
        else:
            label = linked_line(route, row['position'])
        replies = row['replies']
        listed = '<br>'.join(exit_reply(r, route) + ' ' + percentage(r['reach']) for r in replies[:3])
        if len(replies) > 3:
            listed += f'<br>and {len(replies) - 3:,} more'
        if row['unrecorded']:
            listed += ('<br>' if listed else '') + f"Unrecorded replies {percentage(row['unrecorded'])}"
        if row['leaf']:
            listed = f'No prepared {color.title()} move'
        values.append(
            [
                label,
                refs.sources(row),
                reach_cell(row['reach']),
                percentage(row['share']),
                listed or 'n/a',
                percentage(row['score']),
            ]
        )
    reach = 'Games leaving prep here' + ('' if overall else ' after chapter entry')
    text += table(
        [
            'Preparation ends after',
            'Chapter source',
            reach,
            'Of games at this position',
            'Most common unprepared replies',
            'Database score after leaving',
        ],
        values,
    )
    games = f'{color.title()} games' if overall else 'games entering this chapter'
    text += [
        f'These {len(shown)} positions are where '
        f'**{percentage(sum(r["reach"] for r in shown))}** of {games} leave '
        'preparation. '
        f'Preparation ends at {len(rows):,} positions in all.',
        '',
    ]
    return text


class _Node(TypedDict):
    row: JsonObject
    tokens: list[str]
    children: list['_Node']
    parent: NotRequired['_Node | None']


def position_tree(scope: JsonObject | None, refs: Chapters, limit: int = 12) -> list[str]:
    """Most common prepared positions as a nested list; each node shows only the moves since its parent."""
    scope = scope or {}
    refs = refs.for_scope(scope.get('id'))
    rows = [r for r in visible_positions(scope) if not unanswered_position(r, refs.color)][:limit]
    nodes: list[_Node] = [dict(row=r, tokens=refs.route(r['position'], r['line']).split(), children=[]) for r in rows]
    roots: list[_Node] = []
    # Assign parents from shorter routes first, so a transposition with more reach still nests under its prefix.
    for node in sorted(nodes, key=lambda n: len(n['tokens'])):
        parent = max(
            (
                n
                for n in nodes
                if len(n['tokens']) < len(node['tokens']) and node['tokens'][: len(n['tokens'])] == n['tokens']
            ),
            key=lambda n: len(n['tokens']),
            default=None,
        )
        node['parent'] = parent
        (parent['children'] if parent else roots).append(node)
    order = {id(n): i for i, n in enumerate(nodes)}
    text: list[str] = []

    def visit(node: _Node, depth: int) -> None:
        row = node['row']
        start = len(node['parent']['tokens']) if node['parent'] else 0
        moves = line(' '.join(node['tokens'][start:]))
        opening = refs.opening_source(row)
        text.append(
            '  ' * depth + f"- **[{moves}]({analysis_url(row['position'], ' '.join(node['tokens']))})**: "
            f"{percentage(row['reach'])} of games, score {percentage(row.get('repertoire_score'))}"
            + ('' if opening == 'unavailable' else f' · {opening}')
        )
        for child in sorted(node['children'], key=lambda n: order[id(n)]):
            visit(child, depth + 1)

    for root in sorted(roots, key=lambda n: order[id(n)]):
        visit(root, 0)
    return text + ['']


def depth_section(scope: JsonObject | None, level: str = '###', anchor: str | None = None) -> list[str]:
    text = section(f'{level} Prepared-depth distribution', anchor)
    distribution = (scope or {}).get('depth_distribution', {})
    if not distribution.get('survival'):
        return text + ['Depth distribution is unavailable; refresh the cache-only preparation analysis.', '']
    overall = cast(JsonObject, scope).get('id', 'overall') == 'overall'
    text += [
        (
            'How many of your own moves games stay in preparation for. '
            if overall
            else 'How many of your own moves games stay in preparation for, among games entering this chapter. '
        )
        + '*At least this many* is cumulative; the other columns show where games end at exactly that depth. '
        f'Columns that are zero at every depth are left out. {about("depth-distribution")}.',
        '',
    ]
    bounds = distribution['expected_bounds']
    mean = (
        number(distribution['expected_moves'], 1)
        if distribution['expected_moves'] is not None
        else ' to '.join(number(bound, 1) for bound in bounds)
    )
    median = distribution.get('median_moves')
    text += [f'Average **{mean} own moves**; median **{median if median is not None else "unresolved"}**.', '']
    endings = {r['own_moves']: r for r in distribution['endings']}
    rows: list[list[Cell]] = []
    for r in distribution['survival']:
        stop = endings.get(r['own_moves'], {})
        probability = (
            percentage(r['probability']) if r['probability'] is not None else ' to '.join(map(percentage, r['bounds']))
        )
        rows.append(
            [
                r['own_moves'],
                probability,
                percentage(stop.get('prepared_endpoint', 0)),
                percentage(stop.get('unprepared_reply', 0)),
                percentage(stop.get('game_over', 0) + stop.get('other_stop', 0)),
                percentage(stop.get('unresolved_distribution', 0)),
            ]
        )
    headers, rows = drop_uniform(
        [
            'Own moves prepared',
            'At least this many',
            'Ends at a prepared endpoint',
            'Ends at an unprepared reply',
            'Ends otherwise',
            'Unresolved from here',
        ],
        rows,
        {'Ends at a prepared endpoint', 'Ends otherwise', 'Unresolved from here'},
    )
    return text + table(headers, rows)


def entry_routes_section(chapter: JsonObject, scope: JsonObject | None, refs: Chapters, level: str = '##') -> list[str]:
    refs = refs.for_scope(chapter['id'])
    text = [
        *section(f'{level} Exact first-entry positions', f'{refs.anchor(chapter["id"])}-entries'),
        'Every position where games first enter this chapter, as a share of the games that enter it; each game '
        'counts once. The example is the most likely route to the position, and *share of this route* counts that '
        f'route alone. {about("first-entry")}.',
        '',
    ]
    routes = (scope or {}).get('entry_routes', {})
    entries = {e['position']: e for e in chapter['entries']}
    if routes.get('status') == Status.RESOLVED:
        rows: list[list[str]] = []
        for r in routes['positions']:
            original = entries[r['position']]
            example = r['example']
            rows.append(
                [
                    linked_line(example['line'], r['position']),
                    refs.sources(dict(original, _opening_entry=True)),
                    percentage(r['conditional_first_entry_weight']),
                    percentage(example['conditional_probability']),
                    opponent_rating(original.get('opponent_rating')),
                ]
            )
        text += table(
            [
                'Entry position (example route)',
                'Chapter source',
                'Share of entries',
                'Share of this route',
                'Opponent rating',
            ],
            rows,
        )
        zero = len(entries) - len(rows)
        if zero:
            text += [f'{zero} additional configured entry positions have zero modeled arrival.', '']
        return text
    text += [
        'Actual first-arrival examples are unavailable. The lines below label '
        'exact boards and do not imply that games followed those move orders.',
        '',
    ]
    return text + table(
        ['Entry position', 'Chapter source', 'Share of entries', 'Opponent rating'],
        [
            [
                linked_line(' '.join(e['path']), e['position']),
                refs.sources(dict(e, _opening_entry=True)),
                percentage(e.get('conditional_first_entry_weight')),
                opponent_rating(e.get('opponent_rating')),
            ]
            for e in chapter['entries']
        ],
    )


def openings_section(bundle: Bundle, refs: Chapters) -> list[str]:
    """Opening table for the full report; per-opening evidence lives on its own page."""
    data = bundle.get('openings')
    color = bundle['report']['color']
    text = section('### Openings reached', f'{color}-openings')
    if data is None:
        return text + ['Opening analysis pending. Run `repertoire openings` with the saved score results.', '']
    rows = data['openings']
    anchors = {row['id']: f'{color}-opening-{i}' for i, row in enumerate(rows, 1)}
    text += [
        'Reach counts the first arrival at a named opening or one of its named variations, by any move order. '
        f'Openings and their variations overlap, so do not add their reach. {about("opening-names")}.',
        '',
    ]
    if not rows:
        return text + ['No cached opening names are reachable in this repertoire.', '']
    text += opening_table(rows[:20], refs, anchors)
    coverage = data['coverage']
    text += [
        f"{len(rows)} openings and variations are reached; "
        f"{coverage['named_repertoire_positions']:,} repertoire positions have an exact opening name. "
        f"{percentage(coverage['ever_classified_probability'])} of games reach a named opening.",
        '',
    ]
    if len(rows) > 20:
        text += [
            '<details>',
            f'<summary>Remaining {len(rows) - 20} openings</summary>',
            '',
            *opening_table(rows[20:], refs, anchors),
            '</details>',
            '',
        ]
    return text + [
        f'Entry positions and evidence for every opening: [{color.title()} opening details](@openings/{color}).',
        '',
    ]
