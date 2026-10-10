"""The full report, the summary, and the chapter and opening pages."""

from collections.abc import Sequence
from typing import Any

from ..schema import JsonObject
from .bundle import Bundle, scope_by_id
from .definitions import methods
from .derive import combined_overall, non_sparse_rows, own_priorities, unanswered_position, visible_positions
from .format import (
    count,
    delta_points,
    encounters,
    escape,
    gap_percentage,
    headline_delta,
    interval_cell,
    number,
    opponent_rating,
    per_thousand,
    percentage,
    rating_difference,
    reach_cell,
    spread_display,
)
from .links import Chapters, bundle_refs, linked_line
from .markdown import Cell, SourceCell, about, drop_uniform, report_navigation, section, table
from .sections import (
    alternatives_section,
    branch_spread_section,
    chapter_engine_fact,
    character_section,
    common_positions_for_scope,
    common_positions_section,
    comparisons_section,
    correlations_section,
    depth_section,
    edge_section,
    effort_section,
    engine_section,
    engine_sentence,
    entry_routes_section,
    evidence_snapshot,
    exits_section,
    free_transpositions_section,
    gap_section,
    openings_section,
    overview,
    overview_notes,
    position_tree,
    preparation_gain_sentence,
    rare_moves_metric,
    rating_correlations_section,
    snapshot_notes,
    strengths_section,
    summary_edge_section,
    vulnerabilities_section,
)
from .tables import chapter_table, leading_entries, summary_own_priorities

LINKS_NOTE = (
    'Line links open the position on the Lichess analysis board, where you choose your move; '
    'chapter codes such as W1 and B1 open the chapter pages.'
)


def opening_details_page(bundle: Bundle, refs: Chapters, warnings: Sequence[str] = ()) -> list[str]:
    """One page per color with each opening's first entries and the positions reached through it."""
    data = bundle.get('openings') or {}
    color = bundle['report']['color']
    rows = data.get('openings', [])
    anchors = {row['id']: f'{color}-opening-{i}' for i, row in enumerate(rows, 1)}
    text = [
        f'# {color.title()} openings',
        '',
        f'[Summary](@summary) · [Full report](@report) · [{color.title()} openings table](#{color}-openings)',
        '',
        *[item for warning in warnings for item in (warning, '')],
        f'How games first reach each opening in your {color.title()} repertoire, and the most common positions '
        'reached through it. *Share of entries* combines every route to the same position; each line is one real '
        'route. Game counts describe the entry position, not the whole continuation, and † marks pooled counts that '
        'may include the same games more than once. Opening groups have no aggregate opponent rating. '
        f'{about("opening-names")}.',
        '',
    ]
    reached = {row['id']: row for row in rows}
    catalog = {row['id']: row for row in data.get('catalog', [])}
    sparse = bundle['report']['manifest']['sparse_threshold']
    positions = visible_positions(scope_by_id(bundle.get('character')).get('overall', {}))
    for row in rows:
        text += section('## ' + escape(row['name']) + ' (' + escape(row['eco']) + ')', anchors[row['id']])
        parents = [p for p in row['parent_ids'] if p in reached]
        if parents:
            text += [
                'Part of ' + ', '.join(f"[{escape(catalog[p]['name'])}](#{anchors[p]})" for p in parents) + '.',
                '',
            ]
        values: list[list[Cell]] = []
        for entry in row['entries']:
            labels = data['positions'][entry['position']]
            games = (
                f"{count(entry['games'])}"
                + ('†' if len(entry['origins']) > 1 else '')
                + (' (sparse)' if entry['games'] < sparse else '')
            )
            if entry['games_source'] != 'position':
                games += f"<br>from {escape(entry['games_source'])}"
            source = refs.sources({'chapter_attribution': {'source_ids': entry['chapter_ids']}})
            values.append(
                [
                    linked_line(entry['example']['line'], entry['position']),
                    SourceCell(
                        str(source),
                        refs.opening_source({'position': entry['position']}),
                        '' if labels['exact_name'] else 'Name inherited from an earlier position',
                    ),
                    percentage(entry['conditional_weight']),
                    percentage(entry['entry_probability']),
                    percentage(entry['baseline_score']),
                    percentage(entry['repertoire_score']),
                    spread_display(entry.get('branch_score_spread')),
                    games,
                    opponent_rating(entry.get('opponent_rating')),
                    rating_difference(entry.get('opponent_rating')),
                ]
            )
        text += table(
            [
                'Entry position (example route)',
                'Chapter source',
                'Share of entries',
                'Position reach',
                'Entry baseline',
                'Repertoire score',
                'Score spread',
                'Games',
                'Opponent rating',
                'Rating Δ vs parent',
            ],
            values,
        )
        origins: list[tuple[JsonObject, float, float]] = []
        for position in positions:
            origin = data['positions'].get(position['position'], {})
            contribution = origin.get('opening_reach_contributions', {}).get(row['id'], 0.0)
            if contribution:
                origins.append((position, contribution, origin['opening_reach_fractions'][row['id']]))
        if origins:
            headers, values = drop_uniform(
                [
                    'Position',
                    'Chapter source',
                    'Position reach',
                    'Reach through this opening',
                    'Share of arrivals',
                ],
                [
                    [
                        refs.position_cell(position),
                        refs.sources(position),
                        percentage(position['reach']),
                        percentage(contribution),
                        percentage(fraction),
                    ]
                    for position, contribution, fraction in sorted(
                        origins, key=lambda item: (-item[1], -item[0]['reach'])
                    )[:5]
                ],
                {'Share of arrivals'},
            )
            text += ['**Most common positions reached through this opening**', '', *table(headers, values)]
    return text


def chapter_page(
    bundle: Bundle,
    chapter: JsonObject,
    refs: Chapters,
    warnings: Sequence[str],
    chapter_top: int,
    position_top: int,
) -> list[str]:
    """Everything about one chapter on its own page; scores are conditional on first entry."""
    r = bundle['report']
    color = r['color']
    cid = chapter['id']
    characters = scope_by_id(bundle.get('character'))
    preparation = scope_by_id(bundle.get('preparation'))
    vulnerabilities = scope_by_id(bundle.get('vulnerabilities'), 'chapters')
    index = refs.entries[cid][0]
    code = f'{color[0].upper()}{index}'
    anchor = refs.anchor(cid)
    s = chapter['score']
    base = chapter.get('entry_baseline', {})
    char = characters.get(cid)
    links = ['[Summary](@summary)', '[Full report](@report)', f'[{color.title()} chapters](#{color}-chapters)']
    if refs.study_link(cid):
        links.append(refs.study_link(cid, 'Lichess study'))
    neighbors = [c['id'] for c in r['chapters']]
    position = neighbors.index(cid)
    if position > 0:
        links.append('Previous: ' + refs.label(neighbors[position - 1]))
    if position + 1 < len(neighbors):
        links.append('Next: ' + refs.label(neighbors[position + 1]))
    contents = [
        ('starts', 'Where this chapter starts'),
        ('exits', 'Where preparation ends'),
        ('free-transpositions', 'Free transpositions'),
        ('common-positions', 'Most common positions'),
        ('vulnerabilities', 'Vulnerabilities'),
        ('strengths', 'Strengths'),
        ('edge', 'Where your edge comes from'),
        ('gaps', 'Equivalent gap reach'),
        ('prepared-depth', 'Prepared-depth distribution'),
        ('spread', 'Branch score spread'),
        ('character', 'Preparation, replies, and resulting positions'),
        ('entries', 'Exact first-entry positions'),
    ]
    text = [
        f'<a id="{anchor}"></a>',
        '',
        f'# {code}. {escape(chapter["name"])}',
        '',
        ' · '.join(links),
        '',
        *[item for warning in warnings for item in (warning, '')],
    ]
    if chapter.get('policy_overrides'):
        text += [
            "**Alternative comparison.** This chapter prefers its own first moves. "
            "Region reach under the overall policy: "
            f"{percentage(s.get('overall_policy_entry_probability'))}. {about('alternatives')}.",
            '',
        ]
    depth = s.get('prepared_depth', {}).get('expected_moves')
    reach = s.get('entry_probability')
    text += table(
        [
            'Chapter reach',
            'Entry baseline',
            'Repertoire score',
            'Delta',
            'Prepared depth (own moves)',
            'Score spread',
            'Equivalent gap reach',
        ],
        [
            [
                percentage(reach) + (f'<br>{encounters(reach, color.title())}' if reach else ''),
                percentage(base.get('raw_score')),
                percentage(s.get('raw_empirical_score')),
                headline_delta(s.get('raw_empirical_score'), base.get('raw_score')),
                number(depth, 1),
                spread_display(s.get('branch_score_spread'), False),
                gap_percentage(chapter.get('gap_coverage')),
            ]
        ],
    )
    text += [
        'Chapter reach counts games that reach the chapter by any move order; '
        f'everything else on this page counts only those games. {about("entry")}.',
        '',
    ]
    facts: list[str] = []
    ratings = chapter.get('opponent_ratings', {})
    evidence, entry = ratings.get('score_evidence') or {}, ratings.get('entry_baseline') or {}
    coverage = [
        f'{label} {percentage(context["known_coverage"])} rated'
        for label, context in (('scored games', evidence), ('entry', entry))
        if context.get('known_coverage') is not None and context['known_coverage'] < 1 - 1e-9
    ]
    facts.append(
        f"**Opponents:** average rating {opponent_rating(ratings.get('score_evidence'))} in the scored games, "
        f"{opponent_rating(ratings.get('entry_baseline'))} at entry"
        + (f" ({'; '.join(coverage)})" if coverage else '')
        + '.'
    )
    if char and 'reuse' in char:
        facts.append(
            f"**Preparation:** {char['reuse']['reachable_distinct_decisions']:,} own decisions to know; "
            f"{number(char['predictability']['effective_replies'], 1)} effective opponent replies; "
            f"{number(char['position_profiles']['all']['effective_pawn_structures'], 1)} effective pawn structures "
            'where preparation ends.'
        )
    facts.append(
        f"**Gaps:** adds {gap_percentage(chapter.get('gap_coverage'), weighted=True)} to the "
        f"{color.title()} equivalent gap reach. {about('gap-reach')}."
    )
    engine = chapter_engine_fact(bundle, cid)
    if engine:
        facts.append(engine)
    interval = s.get('posterior', {}).get('credible_interval_95')
    if interval:
        facts.append(
            f"**Evidence:** approximate 95% score interval {' to '.join(map(percentage, interval))}; "
            f"sparse-evidence range {' to '.join(map(percentage, s.get('sparse_sensitivity', [])))}"
            + (f"; unresolved {percentage(s['unresolved_mass'])}" if s.get('unresolved_mass') else '')
            + f". {about('evidence')}."
        )
    text += [*[f'- {fact}' for fact in facts], '']
    text += ['On this page: ' + ' · '.join(f'[{title}](#{anchor}-{key})' for key, title in contents), '']
    text += section('## Where this chapter starts', f'{anchor}-starts')
    text += leading_entries(chapter, preparation.get(cid), refs) or ['Entry routes are unavailable.', '']
    text += exits_section(char, refs, max(chapter_top, 10), level='##', anchor=f'{anchor}-exits')
    free = free_transpositions_section(
        bundle.get('vulnerabilities', {}).get('overall'),
        refs,
        chapter_top,
        level='##',
        anchor=f'{anchor}-free-transpositions',
        chapter=cid,
    )
    text += free or section('## Free transpositions', f'{anchor}-free-transpositions') + [
        'No well-sampled unprepared reply in this chapter has a move back into your preparation.',
        '',
    ]
    text += common_positions_for_scope(
        char or {'id': cid}, refs, position_top, level='##', anchor=f'{anchor}-common-positions'
    )
    text += vulnerabilities_section(
        vulnerabilities.get(cid), refs, chapter_top, level='##', anchor=f'{anchor}-vulnerabilities'
    ) or section('## Vulnerabilities', f'{anchor}-vulnerabilities') + ['Vulnerability analysis pending.', '']
    text += strengths_section(
        vulnerabilities.get(cid),
        char,
        refs,
        chapter_top,
        level='##',
        sparse_threshold=r['manifest']['sparse_threshold'],
        anchor=f'{anchor}-strengths',
    ) or section('## Strengths', f'{anchor}-strengths') + ['Strength analysis pending.', '']
    text += edge_section(
        vulnerabilities.get(cid), refs, chapter_top, level='##', anchor=f'{anchor}-edge', detailed=False
    ) or section('## Where your edge comes from', f'{anchor}-edge') + ['Edge ledger pending.', '']
    text += gap_section(char, level='##', refs=refs, top=chapter_top, anchor=f'{anchor}-gaps')
    text += depth_section(preparation.get(cid), level='##', anchor=f'{anchor}-prepared-depth')
    text += branch_spread_section(char, level='##', anchor=f'{anchor}-spread') or section(
        '## Branch score spread', f'{anchor}-spread'
    ) + ['Branch spread pending.', '']
    text += character_section(char, refs, chapter_top, level='##', anchor=f'{anchor}-character') or section(
        '## Preparation, replies, and resulting positions', f'{anchor}-character'
    ) + ['Character analysis pending.', '']
    transitions = [
        t
        for t in r.get('chapter_transitions', [])
        if t['source_id'] == cid and t.get('conditional_probability') is not None and t['conditional_probability'] > 0
    ]
    if transitions:
        text += [
            '**Chapters reached after entering this one**',
            '',
            'The chance of reaching each chapter at or after entering this one, including a shared entry. '
            'Rows overlap.',
            '',
        ]
        text += table(
            ['Destination', 'Conditional probability'],
            [
                [refs.label(t['destination_id'], True), percentage(t['conditional_probability'])]
                for t in sorted(transitions, key=lambda t: t['conditional_probability'], reverse=True)[:5]
            ],
        )
    text += entry_routes_section(chapter, preparation.get(cid), refs)
    return text


def full_report(
    bundles: Sequence[Bundle],
    correlations: JsonObject | None,
    correlation_reason: str | None,
    top: int = 10,
    chapter_top: int = 5,
    position_top: int = 20,
    rating_correlations: JsonObject | None = None,
    rating_correlation_reason: str | None = 'not generated',
) -> dict[str, str]:
    """The full report index plus one page per chapter and one opening page per color, keyed by relative path."""
    combined = combined_overall(bundles)
    has_combined = combined is not None and 'unavailable' not in combined
    warnings = [row for row in snapshot_notes(bundles) if row.startswith('**')]
    pages: dict[str, Any] = {}
    text = [
        '# Repertoire report',
        '',
        'Every table behind the [summary](summary.md), for each color, followed by correlations and definitions. '
        f'Scores are from your side and count draws as half a point. {LINKS_NOTE} '
        f'{about("links", "About labels and links")}.',
        '',
        '<!-- report-navigation -->',
        '',
        *snapshot_notes(bundles),
        *section(
            '## Combined repertoire' if has_combined else '## Score overview',
            'combined' if has_combined else 'overview',
        ),
        *overview(bundles),
        overview_notes(bundles),
        '',
    ]
    for b in bundles:
        r = b['report']
        color = r['color']
        refs = bundle_refs(b)
        o = r['overall']
        characters = scope_by_id(b.get('character'))
        preparation = scope_by_id(b.get('preparation'))
        text += section(f'## {color.title()} repertoire', color)
        text += section(f'### {color.title()} chapters ({len(r["chapters"])})', f'{color}-chapters')
        text += [
            'Chapter reach is the chance that a game reaches one of the chapter\'s entry positions, by any move '
            'order; every other column counts only the games that enter the chapter. Each chapter name opens its '
            'own page. Chapters overlap only where one chapter\'s line transposes onto another\'s entry position, '
            f'and some games reach no chapter, so reaches and scores are not additive. {about("entry")}.',
            '',
            *chapter_table(r, refs),
        ]
        text += alternatives_section(r, refs)
        text += exits_section(characters.get('overall'), refs, top, anchor=f'{color}-exits')
        text += free_transpositions_section(
            b.get('vulnerabilities', {}).get('overall'), refs, top, anchor=f'{color}-free-transpositions'
        )
        text += common_positions_section([b], position_top)
        text += openings_section(b, refs)
        text += vulnerabilities_section(b.get('vulnerabilities', {}).get('overall'), refs, top)
        text += strengths_section(
            b.get('vulnerabilities', {}).get('overall'),
            characters.get('overall'),
            refs,
            top,
            sparse_threshold=r['manifest']['sparse_threshold'],
        )
        text += edge_section(b.get('vulnerabilities', {}).get('overall'), refs, top, anchor=f'{color}-edge')
        text += effort_section(b, refs, top, anchor=f'{color}-effort')
        text += engine_section(b, refs, top, anchor=f'{color}-engine')
        text += gap_section(characters.get('overall'), refs=refs, top=top)
        text += depth_section(preparation.get('overall'), anchor=f'{color}-prepared-depth')
        text += branch_spread_section(characters.get('overall'))
        text += character_section(characters.get('overall'), refs, min(top, 5))
        text += [
            '### Score and evidence limits',
            '',
            f'How far the score could move with more evidence, and how games end in the model. {about("evidence")}.',
            '',
        ]
        interval = o.get('posterior', {}).get('credible_interval_95')
        text += table(
            ['Measure', 'Value'],
            [
                [
                    'Approximate 95% score interval',
                    ' to '.join(map(percentage, interval)) if interval else 'unavailable',
                ],
                ['Unresolved probability', percentage(o['unresolved_mass'])],
                ['Conditional score bounds', ' to '.join(map(percentage, o['conditional_bounds']))],
                ['Sparse probability', percentage(o['sparse_mass'])],
                ['Sparse-evidence range', ' to '.join(map(percentage, o['sparse_sensitivity']))],
            ],
        )
        text += table(
            ['Stopping type', 'Probability'],
            [[k.replace('_', ' ').capitalize(), percentage(v)] for k, v in o['masses'].items()],
        )
        text += section('### Uncertainty priorities and prior sensitivity', f'{color}-uncertainty-prior-sensitivity')
        text += [
            'Where more database games would narrow the score most, and how the score responds to the prior. '
            'Priority is the reach of a stopping route times the width of its score interval, in points per '
            '1,000 games; it is a guide to review, not a share of the variance.',
            '',
            '<details>',
            '<summary>Show the routes and priors</summary>',
            '',
        ]
        text += table(
            [
                'Stopping route',
                'Chapter source',
                'Priority per 1,000 games',
                'Games',
                'Opponent rating',
                'Rating Δ vs parent',
            ],
            [
                [
                    linked_line(' '.join(e['representative_path_san']), e.get('position')),
                    refs.sources(e),
                    per_thousand(100 * e['uncertainty_priority']),
                    count(e['sample_count']),
                    opponent_rating(e.get('opponent_rating')),
                    rating_difference(e.get('opponent_rating')),
                ]
                for e in sorted(r['events'], key=lambda e: e['uncertainty_priority'], reverse=True)[:top]
            ],
        )
        text += table(
            ['Owner W/D/L prior', 'Approximate 95% score interval'],
            [
                [str(p['prior']), ' to '.join(map(percentage, p['overall']['posterior']['credible_interval_95']))]
                for p in r.get('prior_sensitivity', [])
            ],
        )
        text += ['</details>', '']
        for chapter in r['chapters']:
            pages[f'chapters/{refs.page(chapter["id"])}'] = chapter_page(
                b, chapter, refs, warnings, chapter_top, position_top
            )
        if b.get('openings') is not None:
            pages[f'openings/{color}.md'] = opening_details_page(b, refs, warnings)
    text += section('## Correlations', 'correlations')
    text += [
        'Two associations across the whole repertoire. Both main analyses are reach-weighted: each selected move or '
        'opponent reply counts by how often it is played under the repertoire. '
        'Unweighted sensitivity checks are labeled explicitly.',
        '',
    ]
    text += correlations_section(correlations, correlation_reason)
    text += rating_correlations_section(rating_correlations, rating_correlation_reason) + methods(bundles)
    pages = {
        'report.md': '\n'.join(report_navigation(text)) + '\n',
        **{name: '\n'.join(row.rstrip() for row in lines) + '\n' for name, lines in pages.items()},
    }
    return pages


def summary_report(bundles: Sequence[Bundle], correlations: JsonObject | None = None) -> str:
    """Headline, then what to work on: where preparation ends and own moves to review; the rest collapses."""
    snapshot = snapshot_notes(bundles)
    warnings = [row for row in snapshot if row.startswith('**')]
    metadata = [row for row in snapshot if row and not row.startswith('**')]
    comparisons = comparisons_section(bundles)
    links = ['[Full report](@report)']
    if comparisons:
        links.append('[Comparisons](#comparisons)')
    links.append('[Definitions](#methods)')
    text = [
        '# Repertoire summary',
        '',
        ' · '.join(links),
        '',
        'How your repertoire scores in Lichess database games when you play its moves, and where to work on it '
        'next. Scores are from your side and count draws as half a point.',
        '',
        *[item for warning in warnings for item in (warning, '')],
        *overview(bundles, headline_only=True),
        'Delta is the repertoire score minus the database score from the starting position. Its cp and Elo '
        'equivalents are translations of that difference, not engine evaluations or rating forecasts. '
        f'Combined weights White and Black equally (50% each). {about("score", "Definitions and evidence")}.',
        '',
        LINKS_NOTE,
        '',
    ]
    for b in bundles:
        r = b['report']
        color = r['color']
        refs = bundle_refs(b)
        char = scope_by_id(b.get('character')).get('overall', {})
        moves = b.get('vulnerabilities', {}).get('overall', {})
        prep = scope_by_id(b.get('preparation')).get('overall', {}).get('depth_distribution', {})
        depth = r['overall'].get('prepared_depth', {}).get('expected_moves')
        median = prep.get('median_moves')
        chapters = len(r['chapters'])
        text += [
            f'## {color.title()} repertoire',
            '',
            f"Preparation lasts **{number(depth, 1)} own moves** on average"
            + (f' (median {median})' if median is not None else '')
            + f" across {chapters} chapter{'' if chapters == 1 else 's'}. "
            f"Equivalent gap reach **{gap_percentage(char.get('gap_coverage'))}** "
            f"([recurring gaps](#{color}-equivalent-gap-reach)).",
            '',
        ]
        gain = preparation_gain_sentence(correlations, color)
        if gain:
            text += [gain, '']
        engine = engine_sentence(b)
        if engine:
            text += [engine, '']
        text += exits_section(char, refs, 5)
        text += [f'[More exit points](#{color}-exits) · [All positions and gaps](#{color}-common-positions)', '']
        own_bad = own_priorities(moves.get('rankings', {}).get('own', []))
        replies = [x for x in non_sparse_rows(moves.get('rankings', {}).get('opponent', [])) if not x['prepared']][:5]
        if own_bad:
            text += [
                '### Own moves to review',
                '',
                'Your moves that score below the database score of the position they are played from, ranked by '
                'move reach × drag. Drag is a comparison, not a promised gain; thinly sampled comparisons are left '
                f'out. {about("drag")}.',
                '',
                *summary_own_priorities(own_bad[:5], refs, show_components=False),
                f'[All vulnerabilities and prepared replies](#{color}-vulnerabilities).',
                '',
            ]
        positions = [p for p in visible_positions(char) if not unanswered_position(p, color)]
        if positions:
            text += [
                '<details>',
                '<summary>Most common positions</summary>',
                '',
                'Prepared positions by reach, nested by move order; each line shows the moves since the position '
                f'above it. Reach counts every move order, and nested positions overlap. {about("position-reach")}.',
                '',
                *position_tree(char, refs),
                f'Showing {min(12, len(positions))} of {len(positions):,} prepared '
                f'positions. [All positions and gaps](#{color}-common-positions).',
                '',
                '</details>',
                '',
            ]
        text += [
            '<details>',
            f'<summary>Chapter comparisons ({chapters} chapter{"" if chapters == 1 else "s"})</summary>',
            '',
            'Chapter reach counts games that reach a chapter by any move order; the other columns count only those '
            'games. Chapters can overlap through transpositions and some games reach no chapter, so they are not '
            f'additive. {about("entry")}.',
            '',
            *chapter_table(r, refs, detailed=False),
            '</details>',
            '',
        ]
        competing = alternatives_section(r, refs, level='####', anchor=f'{color}-summary-alternatives')
        if competing:
            text += [
                '<details>',
                f'<summary>Competing alternatives ({len(r["alternatives"])})</summary>',
                '',
                *competing,
                '</details>',
                '',
            ]
        if replies:
            text += [
                '<details>',
                '<summary>Costly unprepared replies</summary>',
                '',
                'Opponent replies you have not prepared, ranked by drag per 1,000 games: how far the score falls '
                f'from where it stood before the reply, times how often the reply comes up. {about("drag")}.',
                '',
                *table(
                    [
                        'Line',
                        'Chapter source',
                        'Move reach',
                        'Score before → after',
                        'Score spread before reply',
                        'Drag',
                        'Drag per 1,000 games',
                        'Games',
                        'Opponent rating',
                        'Rating Δ vs parent',
                    ],
                    [
                        [
                            refs.move_cell(x),
                            refs.sources(x),
                            reach_cell(x['branch_reach']),
                            percentage(x['reference_score']) + ' → ' + percentage(x['move_score']),
                            spread_display(x.get('reference_spread'), False),
                            delta_points(x['move_score'], x['reference_score'], drag=True)
                            + '<br>95%: '
                            + interval_cell(x.get('local_drop_interval_pp')),
                            per_thousand(x['weighted_drag_pp']),
                            count(x['sample_count']),
                            opponent_rating(x.get('opponent_rating')),
                            rating_difference(x.get('opponent_rating')),
                        ]
                        for x in replies
                    ],
                ),
                f'[All vulnerabilities](#{color}-vulnerabilities).',
                '',
                '</details>',
                '',
            ]
        text += summary_edge_section(b.get('vulnerabilities', {}).get('overall'), refs, color)
        metrics = [['Expected prepared depth', number(depth, 1) + ' own moves']]
        if median is not None:
            metrics.append(['Median prepared depth', f'{median} own moves'])
        if 'reuse' in char:
            reuse, pred = char['reuse'], char['predictability']
            curve = next((x for x in reuse['curve'] if x['games'] == 100), None)
            metrics.append(['Distinct own decisions', f"{reuse['reachable_distinct_decisions']:,}"])
            if curve:
                metrics.append(
                    [
                        'Distinct decisions met in 100 games',
                        number(curve['expected_distinct_decisions'], 0),
                    ]
                )
            metrics.append(['Effective opponent replies', number(pred['effective_replies'], 1)])
        rare = rare_moves_metric(b.get('vulnerabilities', {}).get('overall'))
        if rare:
            metrics.append(rare)
        metrics.append(['Equivalent gap reach', gap_percentage(char.get('gap_coverage'))])
        metrics.append(
            ['Branch score spread', percentage((char.get('branch_score_spread') or {}).get('standard_deviation'))]
        )
        detail_links = [f'[Prepared-depth distribution](#{color}-prepared-depth)']
        if 'reuse' in char:
            detail_links.append(
                f'[Preparation reuse and reply variety](#{color}-preparation-replies-and-resulting-positions)'
            )
        if char.get('branch_score_spread'):
            detail_links.append(f'[Branch spread and outcome volatility](#{color}-branch-score-spread)')
        if rare:
            detail_links.append(f'[Effort and value](#{color}-effort)')
        text += [
            '<details>',
            '<summary>Preparation and variability</summary>',
            '',
            *table(['Measure', 'Value'], metrics),
            ' · '.join(detail_links),
            '',
            '</details>',
            '',
        ]
    text += comparisons
    text += [
        '<details>',
        '<summary>Evidence and definitions</summary>',
        '',
        *evidence_snapshot(bundles),
        '**Data snapshot**',
        '',
        *[item for description in metadata for item in (description, '')],
        '**Reading the tables**',
        '',
        f'- **Games leaving prep here:** the share of games whose preparation ends right after that position. '
        f'Each game leaves once, so these rows add up. {about("exits", "Exit points")}.',
        '- **Equivalent gap reach:** the reach of a single gap that would repeat as often as all first gaps '
        f'together. Lower is better. {about("gap-reach", "Definition and chapter weighting")}.',
        '- **Delta, gain and drag:** positive delta and gain are good; positive drag is a shortfall. '
        '*Per 1,000 games* multiplies a difference by how often it comes up.',
        '- **1 in N games:** how often a position or move comes up, assuming independent games.',
        '- Rows in other tables overlap and cannot be added. Opponent ratings describe the database games and '
        'never adjust a score. The 95% ranges are approximate model intervals, not proof of a causal gain.',
        '',
    ]
    if any(c.get('policy_overrides') for b in bundles for c in b['report']['chapters']):
        text += [
            '**Alternative** chapters prefer their own first moves; where chapters compete, overall plays '
            'the highest-scoring move, and otherwise the earliest chapter and first PGN choice.',
            '',
        ]
    text += [
        '</details>',
        '',
        'Chapter entries, every position, openings and evidence details are in the [full report](@report).',
        '',
    ]
    return '\n'.join(row.rstrip() for row in text)
