"""The full report, the summary, and the chapter and opening pages."""

from .bundle import scope_by_id
from .definitions import methods
from .derive import combined_overall, non_sparse_rows, own_priorities, unanswered_position, visible_positions
from .format import (
    count,
    delta_points,
    display,
    escape,
    games_per_encounter,
    gap_percentage,
    headline_delta,
    interval_cell,
    number,
    opponent_rating,
    per_thousand,
    percentage,
    rating_difference,
    spread_display,
)
from .links import bundle_refs, linked_line
from .markdown import about, drop_uniform, report_navigation, section, table
from .sections import (
    alternatives_section,
    branch_spread_section,
    character_section,
    common_positions_for_scope,
    common_positions_section,
    comparisons_section,
    correlations_section,
    depth_section,
    entry_routes_section,
    evidence_snapshot,
    exits_section,
    gap_section,
    openings_section,
    overview,
    overview_notes,
    position_tree,
    rating_correlations_section,
    snapshot_notes,
    strengths_section,
    vulnerabilities_section,
)
from .tables import chapter_table, leading_entries, summary_own_priorities


def opening_details_page(bundle, refs, warnings=()):
    """One page per color with each opening's first entries and the positions reached through it."""
    data = bundle.get('openings') or {}
    color = bundle['report']['color']
    rows = data.get('openings', [])
    anchors = {row['id']: f'{color}-opening-{i}' for i, row in enumerate(rows, 1)}
    text = [
        f'# {color.title()} openings: entry positions and evidence',
        '',
        f'[Summary](@summary) · [Full report](@report) · [{color.title()} openings table](#{color}-openings)',
        '',
        *[item for warning in warnings for item in (warning, '')],
        'Entry weight combines all first-arrival routes to the exact board; '
        'each example is one real first-arrival route. '
        'Games describe entry evidence, not the sample size of the full '
        'recursive score; † marks pooled counts whose historical games can '
        'overlap. '
        f'Opening groups have no aggregate opponent rating. {about("opening-names")}.',
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
                'Broader categories: '
                + ', '.join(f"[{escape(catalog[p]['name'])}](#{anchors[p]})" for p in parents)
                + '.',
                '',
            ]
        values = []
        for entry in row['entries']:
            labels = data['positions'][entry['position']]
            values.append(
                [
                    linked_line(entry['example']['line'], entry['position']),
                    refs.sources({'chapter_attribution': {'source_ids': entry['chapter_ids']}}),
                    refs.opening_source({'position': entry['position']}),
                    'Exact' if labels['exact_name'] else 'Inherited',
                    percentage(entry['conditional_weight']),
                    percentage(entry['entry_probability']),
                    percentage(entry['baseline_score']),
                    percentage(entry['repertoire_score']),
                    spread_display(entry.get('branch_score_spread')),
                    f"{entry['games']:,}"
                    + ('†' if len(entry['origins']) > 1 else '')
                    + (' (sparse)' if entry['games'] < sparse else ''),
                    entry['games_source'],
                    opponent_rating(entry.get('opponent_rating')),
                    rating_difference(entry.get('opponent_rating')),
                ]
            )
        text += table(
            [
                'First-entry example',
                'Chapter source / context',
                'Most common opening source',
                'Name source',
                'Entry weight',
                'Position reach',
                'Entry baseline',
                'Repertoire score',
                'Score spread',
                'Entry games',
                'Evidence source',
                'Avg opponent rating',
                'Rating Δ vs parent',
            ],
            values,
        )
        origins = []
        for position in positions:
            origin = data['positions'].get(position['position'], {})
            contribution = origin.get('opening_reach_contributions', {}).get(row['id'], 0.0)
            if contribution:
                origins.append((position, contribution, origin['opening_reach_fractions'][row['id']]))
        if origins:
            headers, values = drop_uniform(
                [
                    'Position',
                    'Chapter source / context',
                    'Total position reach',
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


def chapter_page(bundle, chapter, refs, warnings, chapter_top, position_top):
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
        links.append(refs.study_link(cid, 'Open in Lichess study'))
    neighbors = [c['id'] for c in r['chapters']]
    position = neighbors.index(cid)
    if position > 0:
        links.append('Previous: ' + refs.label(neighbors[position - 1]))
    if position + 1 < len(neighbors):
        links.append('Next: ' + refs.label(neighbors[position + 1]))
    contents = [
        ('exits', 'Where preparation ends'),
        ('common-positions', 'Most common positions'),
        ('vulnerabilities', 'Vulnerabilities'),
        ('strengths', 'Strengths'),
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
    text += [
        f"Chapter reach **{percentage(s.get('entry_probability'))}**; "
        f"repertoire score **{percentage(s.get('raw_empirical_score'))}**; "
        f"entry baseline **{percentage(base.get('raw_score'))}**; delta "
        f"**{headline_delta(s.get('raw_empirical_score'), base.get('raw_score'))}**; "
        f"prepared depth **{number(s.get('prepared_depth', {}).get('expected_moves'))} own moves**. "
        f"Values on this page are conditional on entering the chapter. {about('entry')}.",
        '',
    ]
    text += leading_entries(chapter, preparation.get(cid), refs)
    ratings = chapter.get('opponent_ratings', {})
    text += [
        f"Score-evidence opponent rating **{opponent_rating(ratings.get('score_evidence'))}**; "
        f"entry-baseline opponent rating **{opponent_rating(ratings.get('entry_baseline'))}**. "
        f"Rated coverage: continuation {percentage((ratings.get('score_evidence') or {}).get('known_coverage'))}; "
        f"first entry {percentage((ratings.get('entry_baseline') or {}).get('known_coverage'))}.",
        '',
    ]
    if char and char.get('branch_score_spread'):
        text += [
            "Branch score spread "
            f"**{percentage(char['branch_score_spread'].get('standard_deviation'))}**. "
            f"{about('spread')}.",
            '',
        ]
    text += [
        f"Equivalent gap reach after entry **{gap_percentage(chapter.get('gap_coverage'))}**; "
        f"weighted gap reach contribution **{gap_percentage(chapter.get('gap_coverage'), weighted=True)}**.",
        '',
    ]
    if char and 'reuse' in char:
        text += [
            f"{char['reuse']['reachable_distinct_decisions']} distinct own "
            f"decisions; {number(char['predictability']['effective_replies'])} "
            "effective replies; "
            f"{number(char['position_profiles']['all']['effective_pawn_structures'])} "
            "effective boundary pawn structures.",
            '',
        ]
    text += ['On this page: ' + ' · '.join(f'[{title}](#{anchor}-{key})' for key, title in contents), '']
    text += exits_section(char, refs, max(chapter_top, 10), level='##', anchor=f'{anchor}-exits')
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
            '**Most likely chapter transitions after entry**',
            '',
            'Conditional reach of the destination at or after source entry, '
            'including shared or simultaneous entry. Rows overlap.',
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
    interval = s.get('posterior', {}).get('credible_interval_95')
    if interval:
        text += [
            '## Score interval and evidence limits',
            '',
            "Approximate 95% score interval: "
            f"{' to '.join(map(percentage, interval))}; unresolved probability: "
            f"{percentage(s.get('unresolved_mass'))}; "
            "sparse-score sensitivity: "
            f"{' to '.join(map(percentage, s.get('sparse_sensitivity', [])))}. "
            f"{about('evidence')}.",
            '',
        ]
    return text


def full_report(
    bundles,
    correlations,
    correlation_reason,
    top=10,
    chapter_top=5,
    position_top=20,
    rating_correlations=None,
    rating_correlation_reason='not generated',
):
    """The full report index plus one page per chapter and one opening page per color, keyed by relative path."""
    combined = combined_overall(bundles)
    has_combined = combined is not None and 'unavailable' not in combined
    warnings = [row for row in snapshot_notes(bundles) if row.startswith('**')]
    pages = {}
    text = [
        '# Repertoire report',
        '',
        'Scores, chapter comparisons, vulnerabilities, position contributions, and repertoire character. '
        'All scores are from the repertoire owner\'s perspective. Each chapter '
        'has its own page, linked from the chapter tables; '
        f'line links open the Lichess analysis board. {about("links", "About labels and links")}.',
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
            'Chapter reach is the chance of first reaching any position in a '
            'chapter through any move order; the other values are conditional on '
            'that entry. '
            'Each chapter name opens its own page with positions, gaps and entry routes. '
            f'Chapters overlap, so their values are not additive. {about("entry")}.',
            '',
            *chapter_table(r, refs),
        ]
        text += alternatives_section(r, refs)
        text += exits_section(characters.get('overall'), refs, top, anchor=f'{color}-exits')
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
        text += gap_section(characters.get('overall'), refs=refs, top=top)
        text += depth_section(preparation.get('overall'), anchor=f'{color}-prepared-depth')
        text += branch_spread_section(characters.get('overall'))
        text += character_section(characters.get('overall'), refs, min(top, 5))
        text += ['### Score and evidence limits', '']
        interval = o.get('posterior', {}).get('credible_interval_95')
        text += ['<details>', '<summary>Score and evidence limits</summary>', '']
        text += table(
            ['Measure', 'Value'],
            [
                [
                    'Approximate model-based 95% score interval',
                    ' to '.join(map(percentage, interval)) if interval else 'unavailable',
                ],
                ['Unresolved probability', percentage(o['unresolved_mass'])],
                ['Conditional score bounds', ' to '.join(map(percentage, o['conditional_bounds']))],
                ['Sparse probability', percentage(o['sparse_mass'])],
                ['Sparse-score sensitivity', ' to '.join(map(percentage, o['sparse_sensitivity']))],
            ],
        )
        text += table(
            ['Stopping type', 'Probability'],
            [[k.replace('_', ' ').capitalize(), percentage(v)] for k, v in o['masses'].items()],
        )
        text += ['</details>', '']
        text += section('### Uncertainty priorities and prior sensitivity', f'{color}-uncertainty-prior-sensitivity')
        text += [
            '<details>',
            '<summary>Uncertainty priorities and prior sensitivity</summary>',
            '',
            'Priority is posterior mean reach multiplied by the stopping-score '
            'interval width, in points per 1,000 games. '
            'This is a review heuristic, not additive variance attribution.',
            '',
        ]
        text += table(
            [
                'Stopping route',
                'Chapter source / context',
                'Priority per 1,000 games',
                'Games',
                'Avg opponent rating',
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
        'Both main analyses are reach-weighted: each selected move or opponent '
        'reply is weighted by its probability of being encountered under the '
        'repertoire policy. '
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


def summary_report(bundles):
    """Headline, then what to work on: where preparation ends and own moves to review; the rest collapses."""
    with display(digits=1, compact_counts=True):
        return _summary_report(bundles)


def _summary_report(bundles):
    snapshot = snapshot_notes(bundles)
    warnings = [row for row in snapshot if row.startswith('**')]
    metadata = [row for row in snapshot if row and not row.startswith('**')]
    text = [
        '# Repertoire summary',
        '',
        '[Complete report](@report) · [Definitions](#methods)',
        '',
        *[item for warning in warnings for item in (warning, '')],
        *overview(bundles, headline_only=True),
        'Scores count draws as half a point. Delta is the repertoire score '
        'minus the starting baseline; its CP and Elo equivalents '
        'are score-scale translations, not engine evaluations or rating '
        'forecasts. Combined weights White and Black equally (50% each). '
        f'{about("score", "Definitions and evidence")}.',
        '',
        'Each color starts with where preparation ends and the own moves most worth reviewing. '
        'Line links open the Lichess analysis board at the position where you '
        'choose a move; chapter links open each chapter page.',
        '',
    ]
    for b in bundles:
        r = b['report']
        color = r['color']
        refs = bundle_refs(b)
        refs.compact = True
        char = scope_by_id(b.get('character')).get('overall', {})
        moves = b.get('vulnerabilities', {}).get('overall', {})
        prep = scope_by_id(b.get('preparation')).get('overall', {}).get('depth_distribution', {})
        depth = r['overall'].get('prepared_depth', {}).get('expected_moves')
        median = prep.get('median_moves')
        text += [
            f'## {color.title()} repertoire',
            '',
            f"Preparation lasts **{number(depth)} own moves** on average"
            + (f' (median {median})' if median is not None else '')
            + f". Equivalent gap reach **{gap_percentage(char.get('gap_coverage'))}** "
            f"([recurring gaps](#{color}-equivalent-gap-reach)).",
            '',
        ]
        text += exits_section(char, refs, 5)
        text += [f'[More exit points](#{color}-exits) · [All positions and gaps](#{color}-common-positions)', '']
        own_bad = own_priorities(moves.get('rankings', {}).get('own', []))
        own_good = own_priorities(moves.get('strengths', []), strongest=True)
        replies = [x for x in non_sparse_rows(moves.get('rankings', {}).get('opponent', [])) if not x['prepared']][:5]
        if own_bad:
            text += [
                '### Own moves to review',
                '',
                'Ranked by move reach × drag against the parent database score. These '
                'are comparison deficits, not promised gains; '
                f'sparse comparisons are omitted. {about("drag")}.',
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
                'Prepared positions by reach, nested by move order; each line shows '
                'the moves since the position above it. '
                f'Reach includes transpositions and nested positions overlap. {about("position-reach")}.',
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
            f'<summary>Chapter comparisons ({len(r["chapters"])} '
            f'chapter{"" if len(r["chapters"]) == 1 else "s"})</summary>',
            '',
            'Chapter scores and gap reach are conditional on first entry through '
            'any move order; overlapping chapters are not additive. '
            f'{about("entry")}.',
            '',
            *chapter_table(r, refs),
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
                'Opponent replies without a prepared answer, ranked by drag per 1,000 '
                'games: the score drop from the repertoire value before the reply. '
                f'{about("drag")}.',
                '',
                *table(
                    [
                        'Line',
                        'Chapter context',
                        'Move reach',
                        'Scores before / after',
                        'Score spread before reply',
                        'Drag',
                        'Drag per 1,000 games',
                        'Reply games / Avg opponent rating',
                    ],
                    [
                        [
                            refs.move_cell(x),
                            refs.sources(x),
                            percentage(x['branch_reach'])
                            + '<br>1 per '
                            + games_per_encounter(x['branch_reach'])
                            + ' games',
                            percentage(x['reference_score']) + '<br>' + percentage(x['move_score']),
                            spread_display(x.get('reference_spread'), False),
                            delta_points(x['move_score'], x['reference_score'], drag=True)
                            + '<br>95%: '
                            + interval_cell(x.get('local_drop_interval_pp')),
                            per_thousand(x['weighted_drag_pp']),
                            f"{count(x['sample_count'])} games<br>Rating "
                            + opponent_rating(x.get('opponent_rating'))
                            + '; Δ '
                            + rating_difference(x.get('opponent_rating')),
                        ]
                        for x in replies
                    ],
                ),
                f'[All vulnerabilities](#{color}-vulnerabilities).',
                '',
                '</details>',
                '',
            ]
        if own_good:
            text += [
                '<details>',
                '<summary>Strongest moves</summary>',
                '',
                'Ranked by move reach × gain against the parent database score. Move '
                'gain and later preparation gain appear separately. '
                'These overlapping comparisons are not additive.',
                '',
                *summary_own_priorities(own_good[:5], refs, strongest=True),
                f'[All strengths and position contributions](#{color}-strengths).',
                '',
                '</details>',
                '',
            ]
        metrics = [['Expected prepared depth', number(depth) + ' own moves']]
        if median is not None:
            metrics.append(['Median prepared depth', f'{median} own moves'])
        if 'reuse' in char:
            reuse, pred = char['reuse'], char['predictability']
            curve = next((x for x in reuse['curve'] if x['games'] == 100), None)
            metrics.append(['Distinct own decisions', f"{reuse['reachable_distinct_decisions']:,}"])
            if curve:
                metrics.append(
                    [
                        'Expected distinct decisions after 100 modeled games',
                        number(curve['expected_distinct_decisions'], 0),
                    ]
                )
            metrics.append(['Effective opponent replies', number(pred['effective_replies'])])
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
        text += [
            '<details>',
            '<summary>Preparation and variability</summary>',
            '',
            *table(['Measure', 'Value'], metrics),
            ' | '.join(detail_links) + '.',
            '',
            '</details>',
            '',
        ]
    text += comparisons_section(bundles)
    text += [
        '<details>',
        '<summary>Evidence and definitions</summary>',
        '',
        *evidence_snapshot(bundles),
        '**Data snapshot**',
        '',
        *[item for description in metadata for item in (description, '')],
        '**Metric definitions**',
        '',
        'Games leaving prep here is the share of games whose first unprepared '
        'position follows that prepared position; those rows do not overlap. '
        f'{about("exits", "Exit points")}.',
        '',
        'Equivalent gap reach is the reach of one gap that would produce the '
        'same repeat probability as all first gaps combined. '
        'Lower values indicate less concentrated recurring gaps. '
        f'{about("gap-reach", "Definition and chapter weighting")}.',
        '',
        'Positive gain and delta are favorable; positive drag is a deficit. '
        'Per 1,000 games values multiply a comparison by its reach. '
        'Avg games per encounter is 1 / reach for games with that color. Line comparisons overlap and cannot be added. '
        'Ratings describe local opponents and do not adjust scores. Local 95% '
        'bands are approximate prior-completed model intervals, not causal '
        'gain intervals.',
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
        'Chapter entry explanations, strengths, vulnerabilities and evidence details: [complete report](@report).',
        '',
    ]
    return '\n'.join(row.rstrip() for row in text)
