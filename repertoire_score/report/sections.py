"""Report sections shared by the full report, the summary and the chapter pages."""
from datetime import datetime, timezone
from pathlib import Path

from ..stats import cell
from .bundle import scope_by_id
from .derive import combined_overall, exit_points, non_sparse_rows, position_contributions, unanswered_position, visible_positions
from .format import _display, elo_equivalent, escape, evidence_date, games_per_encounter, gap_percentage, headline_delta, line, number, opponent_rating, percentage, population_text, position_reach_label, rating_difference, spread_display
from .links import analysis_url, bundle_refs, exit_reply, linked_line
from .markdown import about, drop_uniform, section, table
from .tables import opening_table, own_move_table, position_contribution_table, position_games, position_label, reply_table


def evidence_snapshot(bundles):
    rows = []
    for bundle in bundles:
        overall = bundle['report']['overall']
        def score_range(bounds):
            return ' to '.join(percentage(v) for v in bounds) if bounds else 'unavailable'
        rows.append([bundle['report']['color'].title(), score_range(overall.get('posterior', {}).get('credible_interval_95')),
                     score_range(overall.get('sparse_sensitivity')), percentage(overall.get('sparse_mass'))])
    return ['**Score uncertainty**', '',
            'Model intervals describe sampling under the saved population and priors. Sparse sensitivity lets flagged outcomes take any score from 0 to 1; '
            f'it is a separate evidence stress test. {about("evidence", "Limits and definitions")}.', '',
            *table(['Repertoire', 'Approximate 95% score interval', 'Sparse-evidence sensitivity', 'Sparse probability'], rows)]


def overview(bundles, headline_only=False):
    """Headline rows; the delta keeps its CP translation and Elo uses one fewer decimal than scores."""
    elo_digits = max(0, _display.get()['digits'] - 1)
    def elo_text(value):
        return 'unavailable' if value is None else number(value, elo_digits, signed=True)
    rows = []
    for b in bundles:
        r = b['report']; o = r['overall']; base = r.get('starting_position_reference', {}).get('owner_score')
        score = o.get('raw_empirical_score')
        rows.append([r['color'].title(), percentage(base), percentage(score), headline_delta(score, base),
                     elo_text(elo_equivalent(score, base)), number(o.get('prepared_depth', {}).get('expected_moves')),
                     spread_display(o.get('branch_score_spread'), False), len(r['chapters'])])
    combined = combined_overall(bundles)
    if combined and 'unavailable' not in combined:
        rows.append(['**Combined**', percentage(combined['starting_baseline']), percentage(combined['repertoire_score']),
                     headline_delta(combined['repertoire_score'], combined['starting_baseline']),
                     elo_text(combined['elo_equivalent']), number(combined['expected_prepared_depth']),
                     spread_display(combined.get('branch_score_spread'), False), combined['chapters']])
    headers = ['Repertoire', 'Starting baseline', 'Repertoire score', 'Delta', 'Elo equivalent',
               'Prepared depth (own moves)', 'Score spread', 'Chapters']
    if headline_only:
        columns = (0, 1, 2, 3, 4, 5)
        headers = [headers[i] for i in columns]
        rows = [[row[i] for i in columns] for row in rows]
    return table(headers, rows)


def overview_notes(bundles):
    text = ('Delta is repertoire score minus its starting baseline, in percentage points. The CP and Elo equivalents translate that delta '
            'to familiar scales; they are not engine evaluations or rating forecasts. Scores include half a point for draws. ')
    combined = combined_overall(bundles)
    if combined:
        if 'unavailable' in combined:
            text += 'Combined score unavailable: ' + combined['unavailable'] + '. '
        else:
            text += 'Combined gives White and Black equal weight (50% each). '
    return text + f'Prepared depth counts remaining own moves. {about("score", "Scores, deltas and conversions")}.'


def snapshot_notes(bundles):
    text = []
    for b in bundles:
        r = b['report']; color = r['color'].title()
        if not b['current_source']:
            state = 'has changed' if Path(r['manifest']['input_path']).exists() else 'is missing'
            text += [f'**{color} snapshot notice:** the source PGN {state}. These results describe the saved analysis, not the current PGN.', '']
        timestamp = r['manifest'].get('created_at', 'date unavailable')
        try:
            timestamp = datetime.fromisoformat(timestamp).astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        except ValueError:
            pass
        text += [f"{color}: scored {timestamp}; database evidence retrieved {evidence_date(r)}. {population_text(r)}", '']
        if b['unavailable']:
            text += [f"**{color} analysis pending:** " + '; '.join(f'{k}: {v}' for k, v in b['unavailable'].items()) + '.', '']
    return text


def gap_section(scope, level='###', refs=None, top=10, anchor=None):
    metrics = (scope or {}).get('gap_coverage')
    if not metrics:
        return [*section(f'{level} Equivalent gap reach', anchor), 'Unavailable; regenerate the cache-only character analysis.', '']
    text = [*section(f'{level} Equivalent gap reach', anchor),
            f'**{gap_percentage(metrics)}**: the reach of one gap with the same repeat probability as all first unprepared positions combined. '
            f'Lower is better. {about("gap-reach")}.', '']
    if metrics.get('unresolved_mass', 0):
        text += [f"First-gap reach is unresolved for {percentage(metrics['unresolved_mass'])} of games in this scope. "
                 'The displayed range bounds the missing response data; it is not a sampling confidence interval.', '']
    priorities = (scope or {}).get('gap_priorities', {})
    positions = {r['position']: r for r in (scope or {}).get('positions', [])}
    selected = priorities.get('priorities', [])[:top]
    if selected and refs is not None:
        refs = refs.for_scope(scope.get('id'))
        rows = []
        for priority in selected:
            row = positions[priority['position']]
            rating = opponent_rating(row.get('opponent_rating'))
            diff = rating_difference(row.get('opponent_rating'))
            rows.append([refs.position_cell(row), refs.sources(row), percentage(priority['reach']),
                         percentage(priority['repeat_probability_share']), percentage(row.get('database_score')),
                         position_games(row), rating + '<br>Δ ' + diff])
        text += ['**Gaps driving repeat encounters**', '',
                 *table(['First-gap position', 'Chapter source', position_reach_label(scope), 'Repeat-gap share',
                         'Database score', 'Games', 'Avg opponent rating'], rows),
                 f"These {len(selected)} gaps account for **{percentage(selected[-1]['cumulative_repeat_probability_share'])}** "
                 f"of repeat-gap probability ({priorities['share_basis']}). {about('gap-priorities', 'Gap priority definition')}.", '']
    return text


def branch_spread_section(scope, level='###', anchor=None):
    spread = (scope or {}).get('branch_score_spread')
    if not spread:
        return []
    return [*section(f'{level} Branch score spread', anchor),
        'Recursive spread of expected scores across preparation branches. Outcome volatility also includes game-result variation after preparation ends. '
        f'{about("spread")}.', '',
        *table(['Measure', 'Value'], [[name, percentage(spread.get(field))] for name, field in (
            ('Total branch spread', 'standard_deviation'), ('Between stopping positions', 'between_position_deviation'),
            ('Between arrival cohorts at the same position', 'within_position_deviation'))]
            + [['Outcome volatility (0%-100%)', percentage(4 * spread['outcome_variance']) if spread.get('outcome_variance') is not None else 'unresolved'],
               ['Branch share of outcome variance', percentage(spread['variance'] / spread['outcome_variance'])
                if spread.get('outcome_variance') and spread.get('variance') is not None else 'n/a']])]


def character_section(scope, refs, top, level='###', anchor=None):
    if not scope or 'reuse' not in scope:
        return []
    refs = refs.for_scope(scope.get('id'))
    reuse, p = scope['reuse'], scope['predictability']
    profile = scope['position_profiles']['all']
    text = [*section(f'{level} Preparation, replies, and resulting positions', anchor),
            f"**{reuse['reachable_distinct_decisions']} distinct own decisions**, "
            f"**{number(p['effective_replies'])} effective opponent replies**, and "
            f"**{number(profile['effective_pawn_structures'])} effective pawn structures** where preparation ends. "
            f"Reply data cover {percentage(p['recorded_reply_coverage'])} of reached opponent opportunities; "
            f"{percentage(p['sparse_recorded_opportunity_fraction'])} of recorded opportunities have sparse samples.", '']
    text += table(['Games entering this scope', 'Distinct decisions encountered', 'Still unseen', 'Repeat encounters'],
                  [[row['games'], number(row['expected_distinct_decisions']), number(row['expected_unseen_decisions']),
                    number(row['expected_repeat_encounters'])] for row in reuse['curve']])
    text += ['Opponent positions contributing the most reply information:', '']
    text += table(['Position reached by', 'Chapter source', 'Reach', 'Effective replies', 'Common reply (share)', 'Games', 'Avg opponent rating'],
                  [[refs.position_cell(r), refs.sources(r), percentage(r['reach']), number(r['effective_replies']),
                    escape(r['replies'][0]['san']) + ' (' + percentage(r['replies'][0]['probability_given_recorded_reply']) + ')'
                    + '<br>Rating Δ vs parent: ' + rating_difference(r['replies'][0].get('opponent_rating'))
                    if r['replies'] else 'unavailable', f"{r['recorded_reply_observations']:,}" + (' *' if r['sparse'] else ''), opponent_rating(r.get('opponent_rating'))]
                   for r in p['positions'][:top]])
    text += ['Positions at the boundary of preparation:', '']
    features = []
    for name in ('queens', 'king_placement'):
        features.extend([escape(r['value']).capitalize(), percentage(r['probability'])] for r in profile['distributions'].get(name, []))
    from ..character import FEATURE_LABELS
    features.extend([FEATURE_LABELS[name], percentage(value)] for name, value in profile['features'].items())
    text += table(['Feature', 'Share of stopping positions'], features)
    text += table(['Stopping type', 'Probability'], [[kind.replace('_', ' ').capitalize(), percentage(sub['scope_mass'])]
                  for kind, sub in scope['position_profiles'].items() if kind != 'all'])
    text += table(['Frequent pawn structure', 'Share', 'Example route', 'Chapter source / context', 'Example opponent rating'] + (['Group opponent rating'] if scope.get('id') != 'overall' else []),
                  [[escape(r['value']), percentage(r['probability']), linked_line(r['example_line'], r.get('example_position')),
                    refs.sources({'position': r.get('example_position'),
                                  'chapter_attribution': r.get('example_chapter_attribution')}), opponent_rating(r.get('opponent_rating'))]
                    + ([opponent_rating(r.get('group_opponent_rating'))] if scope.get('id') != 'overall' else [])
                   for r in profile['distributions'].get('pawn_structure', [])[:top]])
    return text


def vulnerabilities_section(scope, refs, top, level='###', anchor=None):
    if not scope:
        return []
    refs = refs.for_scope(scope.get('id'))
    text = [*section(f'{level} Vulnerabilities', anchor),
            'Opponent replies rank by drag per 1,000 games; our moves rank by the deficit against the parent database score. '
            f'Rows overlap and cannot be added. {about("drag")}.', '']
    for prepared, name in ((False, 'Unprepared opponent replies'), (True, 'Prepared opponent replies')):
        rows = [r for r in non_sparse_rows(scope.get('rankings', {}).get('opponent', [])) if r['prepared'] == prepared][:top]
        if not rows:
            continue
        text += [f'**{name}**', '', *reply_table(rows, scope, refs, prepared)]
    rows = non_sparse_rows(scope.get('rankings', {}).get('own', []))[:top]
    if rows:
        text += ['**Our selected moves**', '',
                 'Drag = parent database score − repertoire score after our move. Reach is context, not a multiplier in this ranking.', '']
        text += own_move_table(rows, scope, refs)
    if scope.get('unresolved_rows'):
        text += [f"{len(scope['unresolved_rows'])} reachable comparisons remain unresolved and are excluded from these rankings.", '']
    return text


def strengths_section(moves, positions, refs, top, level='###', sparse_threshold=30, anchor=None):
    if not moves and not positions:
        return []
    text = section(f'{level} Strengths', anchor)
    rows = non_sparse_rows((moves or {}).get('strengths', []))[:top]
    if rows:
        text += ['**Our strongest moves**', '',
                 'Gain = repertoire score after our move − parent database score. Ranked by this direct difference.', '']
        text += own_move_table(rows, moves, refs, strongest=True)
    rows = non_sparse_rows(position_contributions(positions, refs.color, sparse_threshold))[:top]
    if rows:
        text += ['**Positions contributing the most to the repertoire score**', '',
                 'Contribution = reach × score, in points per 1,000 games, for prepared positions and unprepared replies. '
                 f'Rows overlap and must not be added. {about("contribution")}.', '']
        text += position_contribution_table(rows, positions, refs)
    elif positions is not None and 'positions' not in positions:
        text += ['Position contributions are unavailable in the saved results. Regenerate the character analysis to include this ranking.', '']
    return text


def correlations_section(result, reason):
    text = section('### Future preparation gain', 'preparation-correlation')
    if not result:
        return text + [f'Correlation analysis unavailable: {reason}.', '']
    text += ['Does more preparation after our selected move tend to improve its score? Each observation is one selected own move under the overall repertoire policy. '
             '**Future depth** is the expected number of prepared own moves remaining after that move, excluding the move itself. '
             '**Future preparation gain** is the recursive repertoire score after the move minus that move\'s database score from the cached parent response table. '
             'Both scores use the repertoire owner\'s perspective.', '',
             'Observations are weighted by their probability of being played. Canonical decisions are counted once and incoming reach is merged across transpositions, '
             'so many rare branches do not outweigh common decisions simply because the tree branches. A game can encounter several decisions, so these weights need not sum to 100%. '
             'Sparse parent/move samples or continuation endpoints, unresolved scores and unreachable decisions are excluded. '
             'These are point estimates from the observed database counts.', '']
    primary = ('reach_weighted_pearson', 'reach_weighted_spearman', 'slope_pp_per_move')
    text += table(['Repertoire', 'Own decisions', 'Linear correlation', 'Rank correlation',
                   'Gain slope (% per future own move)'],
                  [[color.title(), f"{r['n']:,}", *[cell(r, k) for k in primary]] for color, r in result['results'].items()])
    text += ['Positive correlation means decisions with deeper future preparation tend to have larger gains over their own move baseline. '
             'The slope is the associated gain per additional expected future own move; it does not estimate the effect of adding a move.', '',
             '<details>', '<summary>Unweighted and positive-depth sensitivity checks</summary>', '']
    text += table(['Repertoire', 'Unweighted linear correlation', 'Unweighted rank correlation',
                   'Positive-depth decisions', 'Reach-weighted linear, excluding zero depth'],
                  [[color.title(), cell(r, 'pearson'), cell(r, 'spearman'),
                    f"{r['sensitivity_without_zero_depth']['n']:,}",
                    ('undefined' if r['sensitivity_without_zero_depth']['reach_weighted_pearson'] is None else
                     f"{r['sensitivity_without_zero_depth']['reach_weighted_pearson']:.3f}")]
                   for color, r in result['results'].items()])
    text += ['The positive-depth check excludes decisions with no remaining prepared own moves and is a point estimate only. '
             'Weighted ranks use the cumulative reach distribution with midpoint ranks for ties.', '', '</details>', '',
             'These describe association across this repertoire\'s decisions, not the causal gain from adding preparation.', '']
    return text


def rating_correlations_section(result, reason):
    from ..rating_correlations import cell as rating_cell
    text = section('### Opponent rating and score improvement', 'rating-correlations')
    if not result:
        return text + [f'Rating correlation analysis unavailable: {reason}. '
                       'Generate it with `uv run repertoire rating-correlations` and the saved score files.', '']
    text += ['This compares opponent replies from the same parent position. Rating delta is the reply cohort\'s average opponent rating minus the '
             'reach-weighted mean rating of eligible replies at that parent. Score delta is its continuation score minus the '
             'corresponding reach-weighted mean continuation score. Prepared replies use the recursive repertoire score; '
             'unprepared replies use their cached parent move results. Weights are parent reach × reply frequency. '
             'Canonical parent/reply pairs are counted once across transpositions; sparse replies are excluded.', '',
             'Negative correlation means that higher-rated reply cohorts tend to leave the repertoire with lower continuation scores. '
             'The slope expresses the associated score difference for +100 rating, in percentage points.', '']
    headers = ['Repertoire', 'Replies / parent positions', 'Linear correlation', 'Score Δ per +100 rating']
    def row(color, estimate):
        return [color.title(), f"{estimate['replies']:,} / {estimate['parents']:,}", rating_cell(estimate),
                rating_cell(estimate, 'slope_percent_per_100_rating', True)]
    text += table(headers, [row(color, data['reply_associations']['all']) for color, data in result['results'].items()])
    text += ['<details>', '<summary>Prepared replies, unprepared replies, and larger samples</summary>', '']
    labels = {'prepared': 'Prepared replies', 'unprepared': 'Unprepared replies',
              'at_least_1000_games': 'Replies with at least 1,000 games'}
    text += table([headers[0], 'Reply set', *headers[1:]],
                  [[color.title(), label, *row(color, data['reply_associations'][key])[1:]]
                   for color, data in result['results'].items() for key, label in labels.items()])
    text += ['</details>', '',
             'These are point estimates. Shared downstream evidence and overlapping historical games can link different parents. '
             'This describes associations between move-maker cohorts, rather than the effect of changing an individual opponent\'s rating. '
             'The baseline is the local reply mean, and these scores are from the repertoire owner\'s perspective.', '']
    return text


def common_positions_for_scope(scope, refs, limit=20, level='###', anchor=None):
    scope = scope or {}
    refs = refs.for_scope(scope.get('id'))
    color = refs.color
    text = section(f'{level} Most common positions', anchor)
    text += ['Reach includes every transposed route, and nested positions overlap, so do not add their reach. '
             'Prepared rows show continuation scores; unprepared replies show cached database scores. '
             f'Boards immediately before our prepared move are omitted. {about("position-reach")}.', '']
    if 'positions' not in scope:
        return text + ['Position reach is not available in the saved results. Regenerate the character analysis to include this ranking.', '']
    if scope.get('unresolved_opponent_distribution_mass', 0) > 0:
        text += ['Only known reach is ranked. Missing opponent distributions leave downstream routes unresolved, so the ranking can change when evidence becomes available.', '']
    positions = visible_positions(scope)
    prepared = [r for r in positions if not unanswered_position(r, color)]
    unprepared = [r for r in positions if unanswered_position(r, color)]
    reach = position_reach_label(scope)
    for title, rows in (('Prepared positions', prepared), ('Unprepared opponent replies', unprepared)):
        if not rows:
            continue
        text += [f'{level}# {title}', '']
        if rows is prepared:
            text += table(['Position (representative line)', 'Chapter source', reach, 'Avg games per encounter', 'Repertoire score',
                           'Score spread', 'Games at position / reply', 'Avg opponent rating'],
                          [[position_label(r, color, refs), refs.sources(r), percentage(r['reach']), games_per_encounter(r['reach']),
                            percentage(r.get('repertoire_score')), spread_display(r.get('branch_score_spread')), position_games(r),
                            opponent_rating(r.get('opponent_rating'))] for r in rows[:limit]])
        else:
            # Preparation stops at these boards, so branch spread is zero by definition and is not shown.
            text += table(['Position (representative line)', 'Chapter source', reach, 'Avg games per encounter', 'Database score',
                           'Games at position / reply', 'Avg opponent rating', 'Rating Δ vs parent'],
                          [[position_label(r, color, refs), refs.sources(r), percentage(r['reach']), games_per_encounter(r['reach']),
                            percentage(r.get('database_score')), position_games(r), opponent_rating(r.get('opponent_rating')),
                            rating_difference(r.get('opponent_rating'))] for r in rows[:limit]])
        text += [f'Showing {min(limit, len(rows))} of {len(rows):,} {title.lower()}.', '']
    return text


def common_positions_section(bundles, limit=20):
    text = []
    for bundle in bundles:
        report = bundle['report']
        scope = scope_by_id(bundle.get('character')).get('overall', {})
        text += common_positions_for_scope(scope, bundle_refs(bundle), limit, anchor=f'{report["color"]}-common-positions')
    return text


def exits_section(scope, refs, top, level='###', anchor=None):
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
    text += ['Each row is a prepared position where games leave preparation, combining every unprepared reply from it. '
             f'Unlike other rankings, these rows do not overlap. {about("exits")}.', '']
    values = []
    for row in shown:
        route = refs.route(row['position'], row['line'])
        if row['is_starting_position']:
            label = f'[Starting position]({analysis_url(row["position"])})'
        else:
            label = linked_line(route, row['position'])
        if row['leaf']:
            label += f'<br>{color.title()} to move; no prepared reply'
        replies = row['replies']
        listed = ', '.join(exit_reply(r, route) + ' ' + percentage(r['reach']) for r in replies[:3])
        if len(replies) > 3:
            listed += f'<br>and {len(replies) - 3:,} more'
        if row['unrecorded']:
            listed += ('<br>' if listed else '') + f"Unrecorded replies {percentage(row['unrecorded'])}"
        number_of_replies = 'Your move' if row['leaf'] else f'{len(replies):,}' + (' + unrecorded' if row['unrecorded'] else '')
        values.append([label, refs.sources(row), percentage(row['reach']), games_per_encounter(row['reach']),
                       percentage(row['share']), number_of_replies, listed or 'n/a', percentage(row['score'])])
    reach = 'Games leaving prep here' + ('' if overall else ' after chapter entry')
    text += table(['Preparation ends after', 'Chapter source', reach, 'Avg games per encounter', 'Share of games at this position',
                   'Unprepared replies', 'Most common unprepared replies', 'Database score after leaving'], values)
    games = f'{color.title()} games' if overall else 'games entering this chapter'
    text += [f'These {len(shown)} positions are where **{percentage(sum(r["reach"] for r in shown))}** of {games} leave preparation. '
             f'Preparation ends at {len(rows):,} positions in all.', '']
    return text


def position_tree(scope, refs, limit=12):
    """Most common prepared positions as a nested list; each node shows only the moves since its parent."""
    scope = scope or {}
    refs = refs.for_scope(scope.get('id'))
    rows = [r for r in visible_positions(scope) if not unanswered_position(r, refs.color)][:limit]
    nodes = [dict(row=r, tokens=refs.route(r['position'], r['line']).split(), children=[]) for r in rows]
    roots = []
    # Assign parents from shorter routes first, so a transposition with more reach still nests under its prefix.
    for node in sorted(nodes, key=lambda n: len(n['tokens'])):
        parent = max((n for n in nodes if len(n['tokens']) < len(node['tokens'])
                      and node['tokens'][:len(n['tokens'])] == n['tokens']), key=lambda n: len(n['tokens']), default=None)
        node['parent'] = parent
        (parent['children'] if parent else roots).append(node)
    order = {id(n): i for i, n in enumerate(nodes)}
    text = []
    def visit(node, depth):
        row = node['row']
        start = len(node['parent']['tokens']) if node['parent'] else 0
        moves = line(' '.join(node['tokens'][start:]))
        opening = refs.opening_source(row)
        text.append('  ' * depth + f"- **[{moves}]({analysis_url(row['position'], ' '.join(node['tokens']))})**: "
                    f"{percentage(row['reach'])} of games, score {percentage(row.get('repertoire_score'))}"
                    + ('' if opening == 'unavailable' else f' · {opening}'))
        for child in sorted(node['children'], key=lambda n: order[id(n)]):
            visit(child, depth + 1)
    for root in sorted(roots, key=lambda n: order[id(n)]):
        visit(root, 0)
    return text + ['']


def depth_section(scope, level='###', anchor=None):
    text = section(f'{level} Prepared-depth distribution', anchor)
    distribution = (scope or {}).get('depth_distribution', {})
    if not distribution.get('survival'):
        return text + ['Depth distribution is unavailable; refresh the cache-only preparation analysis.', '']
    overall = scope.get('id', 'overall') == 'overall'
    text += [('Probabilities count games with this color. ' if overall else
              'Probabilities are conditional on first reaching any position in this chapter, through any move order. ')
             + 'Depth counts remaining own prepared moves; the second column is cumulative and ending columns stop at exactly that depth. '
             f'Ending columns that are zero at every depth are omitted. {about("depth-distribution")}.', '']
    bounds = distribution['expected_bounds']
    mean = number(distribution['expected_moves']) if distribution['expected_moves'] is not None else ' to '.join(map(number, bounds))
    median = distribution.get('median_moves')
    text += [f'Mean **{mean} own moves**; median **{median if median is not None else "unresolved"}**.', '']
    endings = {r['own_moves']: r for r in distribution['endings']}
    rows = []
    for r in distribution['survival']:
        stop = endings.get(r['own_moves'], {})
        probability = (percentage(r['probability']) if r['probability'] is not None
                       else ' to '.join(map(percentage, r['bounds'])))
        rows.append([r['own_moves'], probability, percentage(stop.get('prepared_endpoint', 0)),
                     percentage(stop.get('unprepared_reply', 0)),
                     percentage(stop.get('game_over', 0) + stop.get('other_stop', 0)),
                     percentage(stop.get('unresolved_distribution', 0))])
    headers, rows = drop_uniform(['Own moves prepared', 'At least this many', 'Prepared endpoint at this depth',
                                  'Unprepared reply at this depth', 'Other ending at this depth', 'Unresolved from this depth'], rows,
                                 {'Prepared endpoint at this depth', 'Other ending at this depth', 'Unresolved from this depth'})
    return text + table(headers, rows)


def entry_routes_section(chapter, scope, refs, level='##'):
    refs = refs.for_scope(chapter['id'])
    text = [*section(f'{level} Exact first-entry positions', f'{refs.anchor(chapter["id"])}-entries'),
            'Weights are conditional on first reaching any position in this chapter, through any move order; each game counts once. '
            'Entry-position weight combines every first-arrival route to the same board; example-route weight belongs only to the displayed route. '
            f'{about("first-entry")}.', '']
    routes = (scope or {}).get('entry_routes', {})
    entries = {e['position']: e for e in chapter['entries']}
    if routes.get('status') == 'resolved':
        rows = []
        for r in routes['positions']:
            original = entries[r['position']]
            example = r['example']
            rows.append([linked_line(example['line'], r['position']), percentage(r['conditional_first_entry_weight']),
                         percentage(example['conditional_probability']), refs.sources(dict(original, _opening_entry=True)),
                         opponent_rating(original.get('opponent_rating'))])
        text += table(['First-entry example', 'Entry-position weight', 'Example-route weight', 'Entry-position sources',
                       'Entry-position opponent rating'], rows)
        zero = len(entries) - len(rows)
        if zero: text += [f'{zero} additional configured entry positions have zero modeled arrival.', '']
        return text
    text += ['Actual first-arrival examples are unavailable. The lines below label exact boards and do not imply that games followed those move orders.', '']
    return text + table(['Entry position (representative line)', 'First-entry weight', 'Chapter sources', 'Avg opponent rating'],
                        [[linked_line(' '.join(e['path']), e['position']), percentage(e.get('conditional_first_entry_weight')),
                          refs.sources(dict(e, _opening_entry=True)), opponent_rating(e.get('opponent_rating'))] for e in chapter['entries']])


def openings_section(bundle, refs):
    """Opening table for the full report; per-opening evidence lives on its own page."""
    data = bundle.get('openings')
    color = bundle['report']['color']
    text = section('### Openings reached', f'{color}-openings')
    if data is None:
        return text + ['Opening analysis pending. Run `repertoire openings` with the saved score results.', '']
    rows = data['openings']
    anchors = {row['id']: f'{color}-opening-{i}' for i, row in enumerate(rows, 1)}
    text += ['Reach counts first arrival at a named opening or its named variations, across all move orders under the selected repertoire. '
             f'Families and variations overlap, so their reaches must not be added. {about("opening-names")}.', '']
    if not rows:
        return text + ['No cached opening names are reachable in this repertoire.', '']
    text += opening_table(rows[:20], refs, anchors)
    coverage = data['coverage']
    text += [f"{len(rows)} reached categories; {coverage['named_repertoire_positions']} repertoire boards have exact cached names. "
             f"A known name is reached in {percentage(coverage['ever_classified_probability'])} of modeled games.", '']
    if len(rows) > 20:
        text += ['<details>', f'<summary>Remaining {len(rows) - 20} openings</summary>', '',
                 *opening_table(rows[20:], refs, anchors), '</details>', '']
    return text + [f'Entry positions and evidence for every opening: [{color.title()} opening details](@openings/{color}).', '']
