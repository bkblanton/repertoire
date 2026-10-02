"""One readable report and summary from matching saved analysis JSONs.

Rendering never reads Explorer tables, recalculates metrics, or uses the network.
Companion hashes prevent mixing results from different repertoire snapshots.
"""
import hashlib
import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from .attribution import write_text
from .correlations import METRICS, cell
from .report import evidence_date, population_text
from .layout import report_directory


FAMILIES = ('vulnerabilities', 'preparation', 'character', 'ratings')
LICHESS_WIN_CHANCE_COEFFICIENT = 0.00368208


def percentage(value):
    if value is None: return 'unresolved'
    return '<0.01%' if 0 < 100 * value < .005 else f'{100 * value:.2f}%'


def number(value, digits=2, signed=False):
    if value is None:
        return 'unresolved'
    if signed and 0 < abs(value) < .5 * 10 ** -digits:
        return ('+<' if value > 0 else '-<') + f'{10 ** -digits:.{digits}f}'
    return format(value, f'{"+" if signed else ""}.{digits}f')


def escape(value):
    return html.escape(str(value), quote=False).replace('|', '&#124;').replace('\n', ' ')


def line(value):
    """Display repeated PGN move numbers as conventional paired moves."""
    value = ' '.join(value.split())
    value = re.sub(r'(?<!\S)(\d+)\.(?!\.)(\S+)\s+\1\.\.\.(\S+)', r'\1. \2 \3', value)
    value = re.sub(r'(?<!\S)(\d+)\.\.\.(?=\S)', r'\1... ', value)
    value = re.sub(r'(?<!\S)(\d+)\.(?!\.)', r'\1. ', value)
    return escape(' '.join(value.split()) or '(starting position)')


def table(headers, rows):
    if not rows:
        return []
    text_columns = ('Line', 'Position', 'Chapter', 'Source', 'Your move', 'Common reply', 'Observed alternative',
                    'Feature', 'Stopping type', 'Measure', 'Destination', 'Frequent pawn')
    return ['| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join('---' if i == 0 or h.startswith(text_columns) else '---:' for i, h in enumerate(headers)) + ' |',
            *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows], '']


def section(title, anchor=None):
    return ([f'<a id="{anchor}"></a>', ''] if anchor else []) + [title, '']


def load(paths, strict=True, require_complete=False):
    bundles, seen = [], set()
    for value in paths:
        path = Path(value)
        report = json.loads(path.read_text(encoding='utf-8'))
        color = report.get('color')
        if color not in ('white', 'black') or not all(k in report for k in ('overall', 'chapters', 'manifest', 'events')):
            raise ValueError(f'Not a repertoire result file: {path}')
        if color in seen:
            raise ValueError('Supply one score report per color')
        seen.add(color)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = report['manifest']
        source = Path(manifest['input_path'])
        current = source.exists() and hashlib.sha256(source.read_bytes()).hexdigest() == manifest['input_sha256']
        bundle = dict(path=path, report=report, digest=digest, current_source=current, unavailable={})
        for family in FAMILIES:
            companion = path.with_suffix(f'.{family}.json')
            reason = None
            if not companion.exists():
                reason = 'not generated'
            else:
                data = json.loads(companion.read_text(encoding='utf-8'))
                m = data.get('manifest', {})
                if data.get('color') != color or any(m.get(key) != expected for key, expected in (
                        ('report_sha256', digest), ('input_sha256', manifest['input_sha256']), ('filters', manifest['filters']))):
                    reason = 'belongs to a different score snapshot'
                    if strict:
                        raise ValueError(f'{companion}: {reason}; regenerate this analysis before combining')
                else:
                    if family == 'vulnerabilities' and m.get('own_score_basis') != 'prepared repertoire continuation':
                        reason = 'own-move comparisons need a saved continuation refresh'
                        if strict: raise ValueError(f'{companion}: {reason}; use --refresh-saved')
                    if family == 'vulnerabilities' and m.get('continuation_source_sha256'):
                        source_analysis = path.with_suffix('.character.json')
                        if not source_analysis.exists() or hashlib.sha256(source_analysis.read_bytes()).hexdigest() != m['continuation_source_sha256']:
                            reason = 'continuation analysis changed since comparison generation'
                            if strict: raise ValueError(f'{companion}: {reason}; refresh comparisons before combining')
                    if family == 'ratings':
                        for supporting, expected_hash in m.get('supporting_sha256', {}).items():
                            supporting_path = path.with_suffix(f'.{supporting}.json')
                            if (not supporting_path.exists() or hashlib.sha256(supporting_path.read_bytes()).hexdigest() != expected_hash
                                    or supporting not in bundle):
                                reason = 'supporting analysis changed since rating generation'
                                if strict:
                                    raise ValueError(f'{companion}: {reason}; regenerate ratings before combining')
                                break
                        if set(m.get('supporting_sha256', {})) != {'preparation', 'character', 'vulnerabilities'}:
                            reason = 'missing rating provenance'
                            if strict: raise ValueError(f'{companion}: {reason}')
                    if reason is None:
                        bundle[family] = data
            if reason:
                bundle['unavailable'][family] = reason
        if require_complete and bundle['unavailable']:
            raise ValueError(f'{color.title()} analyses incomplete: {bundle["unavailable"]}')
        if 'ratings' in bundle:
            from .ratings import attach
            attach(bundle)
        bundles.append(bundle)
    return sorted(bundles, key=lambda b: b['report']['color'] != 'white')


def load_correlations(bundles, strict=True, require_complete=False):
    path = bundles[0]['path'].parent / 'depth-delta-correlation.json'
    if not path.exists():
        if require_complete:
            raise ValueError('Generate depth-delta correlations before combining the reports')
        return None, 'not generated'
    result = json.loads(path.read_text(encoding='utf-8'))
    expected = {b['report']['color']: b for b in bundles}
    provenance = result.get('provenance', {})
    matches = set(provenance) == set(expected) and all(
        p.get('report_sha256') == expected[c]['digest']
        and p.get('input_sha256') == expected[c]['report']['manifest']['input_sha256']
        and p.get('filters') == expected[c]['report']['manifest']['filters']
        for c, p in provenance.items())
    if not matches:
        if strict:
            raise ValueError(f'{path}: correlations belong to different score snapshots; regenerate them before combining')
        return None, 'belongs to different score snapshots'
    return result, None


class Chapters:
    """Compact clickable references preserve every source without giant cells."""
    def __init__(self, report):
        self.color = report['color']
        self.catalog = report['chapters']
        self.entries = {c['id']: (i, c) for i, c in enumerate(self.catalog, 1)}

    def anchor(self, cid):
        return f'{self.color}-chapter-{self.entries[cid][0]}'

    def label(self, cid, full=False):
        if cid not in self.entries:
            return escape(cid)
        index, chapter = self.entries[cid]
        text = escape(chapter['name']) if full else f'{self.color[0].upper()}{index}'
        return f'[{text}](#{self.anchor(cid)})'

    def sources(self, row):
        attribution = row.get('chapter_attribution') or {}
        ids, prefix = attribution.get('source_ids', []), ''
        if not ids:
            ids = attribution.get('transposition_ids', [])
            prefix = 'Transposition: '
        if not ids:
            ids = attribution.get('context_ids', [])
            prefix = 'Unprepared; context: '
        if not ids:
            return 'None'
        if set(ids) == set(self.entries) and len(ids) > 2:
            return prefix + f'[All {len(ids)} {self.color.title()} chapters](#{self.color}-chapters)'
        return prefix + ', '.join(self.label(cid) for cid in ids)


def elo_equivalent(score, baseline):
    """Difference of logistic Elo score equivalents; boundaries are undefined."""
    if score is None or baseline is None or not (0 < score < 1 and 0 < baseline < 1):
        return None
    return 400 * (math.log10(score / (1 - score)) - math.log10(baseline / (1 - baseline)))


def centipawn_equivalent(score):
    """Direct inverse Lichess score equivalent, from the repertoire owner's view."""
    if score is None or not 0 < score < 1:
        return None
    return math.log(score / (1 - score)) / LICHESS_WIN_CHANCE_COEFFICIENT


def centipawn_delta(after, before):
    """Convert both scores first, then subtract before from after."""
    converted = [centipawn_equivalent(p) for p in (after, before)]
    return None if any(p is None for p in converted) else converted[0] - converted[1]


def cp(score):
    value = centipawn_equivalent(score)
    return 'unavailable' if value is None else number(value, signed=True)


def cp_change(after, before):
    value = centipawn_delta(after, before)
    return 'unavailable' if value is None else number(value, signed=True)


def cp_interval(bounds):
    return ' to '.join(map(cp, bounds)) if bounds else 'unavailable'


def combined_overall(bundles):
    """A 50/50 color mixture, averaging scores before applying conversions."""
    reports = {b['report']['color']: b['report'] for b in bundles}
    if set(reports) != {'white', 'black'}:
        return None
    white, black = reports['white'], reports['black']
    if white['manifest']['filters'] != black['manifest']['filters']:
        return {'unavailable': 'the colors use different Explorer filters'}
    if any(r['manifest'].get('overall_basis') != 'standard starting position' for r in (white, black)):
        return {'unavailable': 'both colors must be evaluated from the standard starting position'}

    def average(values):
        return None if any(v is None for v in values) else sum(values) / 2

    score = average([r['overall'].get('raw_empirical_score') for r in (white, black)])
    baseline = average([r.get('starting_position_reference', {}).get('owner_score') for r in (white, black)])
    return {'weights': {'white': .5, 'black': .5}, 'repertoire_score': score, 'starting_baseline': baseline,
            'difference_pp': None if score is None or baseline is None else 100 * (score - baseline),
            'elo_equivalent': elo_equivalent(score, baseline),
            'centipawn_equivalent': centipawn_equivalent(score),
            'baseline_centipawn_equivalent': centipawn_equivalent(baseline),
            'centipawn_delta': centipawn_delta(score, baseline),
            'expected_prepared_depth': average([r['overall'].get('prepared_depth', {}).get('expected_moves') for r in (white, black)]),
            'chapters': sum(len(r['chapters']) for r in (white, black))}


def overview(bundles):
    rows = []
    for b in bundles:
        r = b['report']; o = r['overall']; base = r.get('starting_position_reference', {}).get('owner_score')
        score = o.get('raw_empirical_score')
        delta = None if base is None or score is None else 100 * (score - base)
        elo = elo_equivalent(score, base)
        rows.append([r['color'].title(), percentage(base), cp(base), percentage(score), cp(score), number(delta, signed=True),
                     cp_change(score, base),
                     'unavailable' if elo is None else number(elo, signed=True),
                     number(o.get('prepared_depth', {}).get('expected_moves')), len(r['chapters'])])
    combined = combined_overall(bundles)
    if combined and 'unavailable' not in combined:
        rows.append(['**Combined**', percentage(combined['starting_baseline']), cp(combined['starting_baseline']),
                     percentage(combined['repertoire_score']), cp(combined['repertoire_score']),
                     number(combined['difference_pp'], signed=True), cp_change(combined['repertoire_score'], combined['starting_baseline']),
                     'unavailable' if combined['elo_equivalent'] is None else number(combined['elo_equivalent'], signed=True),
                     number(combined['expected_prepared_depth']), combined['chapters']])
    return table(['Repertoire', 'Starting baseline', 'Baseline CP', 'Repertoire score', 'Score CP', 'Delta (pp)',
                  'CP delta', 'Elo equivalent', 'Prepared depth (own moves)', 'Chapters'], rows)


def overview_notes(bundles):
    text = 'Delta is repertoire score minus its starting baseline. Elo equivalent converts that score difference to the 400-point logistic Elo scale; it is not a measured rating gain. '
    text += 'CP is the direct centipawn equivalent of the adjacent score using the inverse [Lichess score curve](https://lichess.org/page/accuracy). CP delta is score CP minus baseline CP; positive values favor the repertoire owner. Scores include half a point for draws. These are score-scale conversions rather than engine evaluations. '
    combined = combined_overall(bundles)
    if combined:
        if 'unavailable' in combined:
            text += 'Combined score unavailable: ' + combined['unavailable'] + '. '
        else:
            text += 'Combined gives White and Black equal weight (50% each), averaging their scores and baselines before both conversions. '
    return text + 'Prepared depth counts remaining own moves. Strengths and vulnerability rankings omit sparse rows; scores still include all evidence.'


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

def opponent_rating(context):
    if not context or context.get('mean') is None:
        if context and context.get('reason') == 'no_preceding_opponent_move':
            return 'n/a (no preceding opponent move)'
        return 'unavailable'
    result = f"{context['mean']:,.0f}"
    if context['known_coverage'] < 1 - 1e-9:
        result += f" ({100 * context['known_coverage']:.1f}% rated)"
    return result


def rating_difference(context):
    if context and context.get('reason') == 'no_preceding_opponent_move':
        return 'n/a (no preceding opponent move)'
    if context and context.get('basis') == 'current opponent response rows' and 'difference_vs_parent' not in context:
        return 'n/a'
    if not context or context.get('difference_vs_parent') is None:
        return 'unavailable'
    result = f"{context['difference_vs_parent']:+,.0f}"
    notes = []
    if context.get('comparison_coverage', 1.) < 1 - 1e-9:
        notes.append(f"{100 * context['comparison_coverage']:.1f}% compared")
    if context.get('parent_known_coverage', 1.) < 1 - 1e-9:
        notes.append(f"parent {100 * context['parent_known_coverage']:.1f}% rated")
    return result + (' (' + '; '.join(notes) + ')' if notes else '')


def chapter_table(report, refs, link=True):
    alternatives = any(c.get('policy_overrides') for c in report['chapters'])
    headers = ['Chapter', 'Chapter reach (incl. transpositions)', 'Entry baseline', 'Baseline CP', 'Repertoire score',
               'Score CP', 'Delta (pp)', 'CP delta', 'Prepared depth', 'Avg opponent rating']
    if alternatives:
        headers.insert(2, 'Overall-policy reach')
    rows = []
    for c in report['chapters']:
        s, base = c['score'], c.get('entry_baseline', {})
        index = refs.entries[c['id']][0]
        title = f"{report['color'][0].upper()}{index}. {escape(c['name'])}"
        if link:
            title = f'[{title}](#{refs.anchor(c["id"])})'
        if c.get('policy_overrides'):
            title += ' **(alternative)**'
        row = [title, percentage(s.get('entry_probability')), percentage(base.get('raw_score')), cp(base.get('raw_score')),
               percentage(s.get('raw_empirical_score')), cp(s.get('raw_empirical_score')), number(base.get('difference_pp'), signed=True),
               cp_change(s.get('raw_empirical_score'), base.get('raw_score')),
               number(s.get('prepared_depth', {}).get('expected_moves')),
               opponent_rating(c.get('opponent_ratings', {}).get('score_evidence'))]
        if alternatives:
            row.insert(2, percentage(s.get('overall_policy_entry_probability', s.get('entry_probability'))))
        rows.append(row)
    return table(headers, rows)


def character_section(scope, refs, top, level='###'):
    if not scope or 'reuse' not in scope:
        return []
    reuse, p = scope['reuse'], scope['predictability']
    profile = scope['position_profiles']['all']
    text = [f'{level} Preparation, replies, and resulting positions', '',
            f"**{reuse['reachable_distinct_decisions']} distinct own decisions**, "
            f"**{number(p['effective_replies'])} effective opponent replies**, and "
            f"**{number(profile['effective_pawn_structures'])} effective pawn structures** where preparation ends. "
            f"Reply data cover {percentage(p['recorded_reply_coverage'])} of reached opponent opportunities; "
            f"{percentage(p['sparse_recorded_opportunity_fraction'])} of recorded opportunities have sparse samples.", '']
    text += table(['Games entering this scope', 'Distinct decisions encountered', 'Still unseen', 'Repeat encounters'],
                  [[row['games'], number(row['expected_distinct_decisions']), number(row['expected_unseen_decisions']),
                    number(row['expected_repeat_encounters'])] for row in reuse['curve']])
    text += ['Most frequently used own decisions:', '']
    text += table(['Position reached by', 'Your move', 'Chapter source', reach_label(scope), 'Avg opponent rating'],
                  [[line(r['line']), escape(r['san']), refs.sources(r), percentage(r['reach']), opponent_rating(r.get('opponent_rating'))] for r in reuse['decisions'][:top]])
    text += ['Opponent positions contributing the most reply information:', '']
    text += table(['Position reached by', 'Chapter source', 'Reach', 'Effective replies', 'Common reply (share)', 'Games', 'Avg opponent rating'],
                  [[line(r['line']), refs.sources(r), percentage(r['reach']), number(r['effective_replies']),
                    escape(r['replies'][0]['san']) + ' (' + percentage(r['replies'][0]['probability_given_recorded_reply']) + ')'
                    + '<br>Rating Δ vs parent: ' + rating_difference(r['replies'][0].get('opponent_rating'))
                    if r['replies'] else 'unavailable', f"{r['recorded_reply_observations']:,}" + (' *' if r['sparse'] else ''), opponent_rating(r.get('opponent_rating'))]
                   for r in p['positions'][:top]])
    text += ['Positions at the boundary of preparation:', '']
    features = []
    for name in ('queens', 'king_placement'):
        features.extend([escape(r['value']).capitalize(), percentage(r['probability'])] for r in profile['distributions'].get(name, []))
    from .character import FEATURE_LABELS
    features.extend([FEATURE_LABELS[name], percentage(value)] for name, value in profile['features'].items())
    text += table(['Feature', 'Share of stopping positions'], features)
    text += table(['Stopping type', 'Probability'], [[kind.replace('_', ' ').capitalize(), percentage(sub['scope_mass'])]
                  for kind, sub in scope['position_profiles'].items() if kind != 'all'])
    text += table(['Frequent pawn structure', 'Share', 'Example route', 'Chapter source / context', 'Example opponent rating'] + (['Group opponent rating'] if scope.get('id') != 'overall' else []),
                  [[escape(r['value']), percentage(r['probability']), line(r['example_line']),
                    refs.sources({'chapter_attribution': r.get('example_chapter_attribution')}), opponent_rating(r.get('opponent_rating'))]
                    + ([opponent_rating(r.get('group_opponent_rating'))] if scope.get('id') != 'overall' else [])
                   for r in profile['distributions'].get('pawn_structure', [])[:top]])
    return text


def reach_label(scope):
    return 'Reach in repertoire' if scope.get('id', 'overall') == 'overall' else 'Reach after chapter entry'


def non_sparse_rows(rows):
    """Filter local, parent, and immediate endpoint evidence before display limits."""
    return [r for r in rows if not any(r.get(field, False)
            for field in ('sparse', 'parent_sparse', 'continuation_endpoint_sparse'))]


def own_move_table(rows, scope, refs, strongest=False):
    field = 'local_gain_pp' if strongest else 'local_drop_pp'
    return table(['Line', 'Chapter source', reach_label(scope), 'Parent database score', 'Parent CP', 'Repertoire score', 'Score CP', 'CP delta',
                  'Gain (pp)' if strongest else 'Drag (pp)', 'Parent games', 'Avg opponent rating'],
                 [[line(r['line']), refs.sources(r), percentage(r['branch_reach']), percentage(r['reference_score']), cp(r['reference_score']),
                   percentage(r['move_score']) + (' (sparse endpoint)' if r.get('continuation_endpoint_sparse') else ''), cp(r['move_score']),
                   cp_change(r['move_score'], r['reference_score']), number(r[field], signed=strongest),
                   f"{r['parent_sample_count']:,}" + (' (sparse)' if r.get('parent_sparse') else ''),
                   opponent_rating(r.get('opponent_rating'))] for r in rows])


def vulnerabilities_section(scope, refs, top, level='###'):
    if not scope:
        return []
    text = [f'{level} Vulnerabilities', '',
            ('Opponent replies rank by weighted drag; our moves rank by the direct deficit against the parent database score. '
             'Drag is the local score drop; weighted drag multiplies it by reply reach. CP delta is after minus before, so negative values indicate a worse score. Comparisons overlap and cannot be added.' if level == '###'
             else '[Definitions: drag and reach](#methods).'), '']
    for prepared, name in ((False, 'Unprepared opponent replies'), (True, 'Prepared opponent replies')):
        rows = [r for r in non_sparse_rows(scope.get('rankings', {}).get('opponent', [])) if r['prepared'] == prepared][:top]
        if not rows:
            continue
        text += [f'**{name}**', '']
        text += table(['Line', 'Chapter source / context', reach_label(scope), 'Reply frequency at parent',
                       'Before reply (repertoire)', 'Before CP', 'After reply (repertoire)' if prepared else 'After reply (database)',
                       'After CP', 'CP delta', 'Drag (pp)', 'Weighted drag (pp)', 'Reply games', 'Avg opponent rating', 'Rating Δ vs parent'],
                      [[line(r['line']), refs.sources(r), percentage(r['branch_reach']), percentage(r['branch_probability']),
                        percentage(r['reference_score']), cp(r['reference_score']), percentage(r['move_score']), cp(r['move_score']),
                        cp_change(r['move_score'], r['reference_score']), number(r['local_drop_pp']), number(r['weighted_drag_pp'], 4),
                        f"{r['sample_count']:,}" + (' (sparse)' if r['sparse'] else ''), opponent_rating(r.get('opponent_rating')),
                        rating_difference(r.get('opponent_rating'))] for r in rows])
    rows = non_sparse_rows(scope.get('rankings', {}).get('own', []))[:top]
    if rows:
        text += ['**Our selected moves**', '',
                 'Drag = parent database score − repertoire score after our move. Reach is context, not a multiplier in this ranking.', '']
        text += own_move_table(rows, scope, refs)
    if scope.get('unresolved_rows'):
        text += [f"{len(scope['unresolved_rows'])} reachable comparisons remain unresolved and are excluded from these rankings.", '']
    return text

def stopping_contributions(scope):
    """Non-overlapping stopping mass, merged by exact board and evidence type."""
    from .ratings import mixture, unavailable
    groups = {}
    for row in (scope or {}).get('stops', []):
        if row['reach'] <= 0:
            continue
        identity = (row['position'], row['type'])
        # Residual buckets do not identify an exact next board.
        if row['type'] not in ('theory_leaf', 'deviation', 'terminal'):
            identity += (row.get('parent_position'), row.get('move'))
        groups.setdefault(identity, []).append(row)
    merged = []
    for rows in groups.values():
        main = max(rows, key=lambda r: r['reach'])
        total = sum(r['reach'] for r in rows)
        known = sum(r['reach'] for r in rows if r['score'] is not None)
        contribution = sum(r['contribution_pp'] for r in rows if r['contribution_pp'] is not None)
        row = dict(main, reach=total, contribution_pp=contribution, known_score_mass=known,
                   score=contribution / (100 * total) if known >= total - 1e-12 else None,
                   sample_count=sum(r['sample_count'] for r in rows), pooled=len(rows) > 1,
                   sparse=any(r['sparse'] for r in rows),
                   opponent_rating=mixture([(r['reach'], r.get('opponent_rating') or unavailable()) for r in rows], 'stopping position arrivals'))
        attribution = {}
        for field in ('source_ids', 'transposition_ids', 'context_ids'):
            attribution[field] = list(dict.fromkeys(cid for r in rows for cid in (r.get('chapter_attribution') or {}).get(field, [])))
        row['chapter_attribution'] = attribution
        merged.append(row)
    return sorted(merged, key=lambda r: (-r['contribution_pp'], r['line'], r['position']))


def visible_positions(scope):
    """Keep one board after a guaranteed own reply, plus unanswered boards."""
    rows = [r for r in (scope or {}).get('positions', [])
            if not r['is_starting_position'] and r['reach'] > 0 and r['kind'] != 'own_move']
    return sorted(rows, key=lambda r: (-r['reach'], len(r['line'].split()), r['line'], r['position']))


def unanswered_position(row, color):
    return (bool(row.get('unprepared_origins')) or row['kind'] == 'unprepared_reply'
            or (row['kind'] == 'theory_leaf' and row['to_move'] == color))


def position_label(row, color):
    text = line(row['line'])
    if row['kind'] == 'terminal':
        return text + '<br>Game over'
    if row['kind'] in ('theory_leaf', 'unprepared_reply') and row['to_move'] == color:
        return text + f'<br>{color.title()} to move; no prepared reply'
    return text


def position_games(row):
    count = row.get('games')
    if count is None: return 'unavailable'
    return f'{count:,}' + ('†' if len(row.get('unprepared_origins', [])) > 1 else '')


def position_contributions(scope, color, sparse_threshold=30):
    """Score carried through every reached board; nested rows are not additive."""
    rows = []
    for row in visible_positions(scope):
        unanswered = unanswered_position(row, color)
        value = row.get('database_score' if unanswered else 'repertoire_score')
        if value is None:
            continue
        counts = [r.get('games') for r in row.get('unprepared_origins', [])] or [row.get('games')]
        sparse = row.get('sparse', False) or any(n is None or n < sparse_threshold for n in counts)
        rows.append(dict(row, score=value, score_basis='database' if unanswered else 'repertoire',
                         contribution_pp=100 * row['reach'] * value, sparse=sparse))
    return sorted(rows, key=lambda r: (-r['contribution_pp'], -r['reach'], len(r['line'].split()), r['line'], r['position']))


def position_contribution_table(rows, scope, refs):
    def kind(row):
        if row['kind'] == 'terminal': return 'Game over'
        if row['score_basis'] == 'database': return 'Unprepared reply'
        return 'Prepared endpoint' if row['kind'] == 'theory_leaf' else 'Prepared position'

    return table(['Position (representative line)', 'Chapter source / context', 'Position type', reach_label(scope),
                  'Score', 'Score CP', 'Contribution (pp)', 'Games at position / reply', 'Avg opponent rating'],
                 [[position_label(r, refs.color), refs.sources(r), kind(r), percentage(r['reach']), percentage(r['score']),
                   cp(r['score']), number(r['contribution_pp'], 4), position_games(r), opponent_rating(r.get('opponent_rating'))]
                  for r in rows])


def strengths_section(moves, positions, refs, top, level='###', sparse_threshold=30):
    if not moves and not positions:
        return []
    text = [f'{level} Strengths', '']
    rows = non_sparse_rows((moves or {}).get('strengths', []))[:top]
    if rows:
        text += ['**Our strongest moves**', '',
                 'Gain = repertoire score after our move − parent database score. Ranked by this direct difference.', '']
        text += own_move_table(rows, moves, refs, strongest=True)
    rows = non_sparse_rows(position_contributions(positions, refs.color, sparse_threshold))[:top]
    if rows:
        text += ['**Positions contributing the most to the repertoire score**', '',
                 ('Contribution = reach × score for any prepared position or unprepared opponent reply, including intermediate boards. '
                  'Prepared positions use repertoire continuation scores; unprepared replies use cached database scores. '
                  'Guaranteed own replies are collapsed and transposed arrivals share one board. Rows overlap and must not be added; this measures score carried through a position, not improvement over a baseline.' if level == '###'
                  else '[Definition: position contribution](#methods).'), '']
        text += position_contribution_table(rows, positions, refs)
    elif positions is not None and 'positions' not in positions:
        text += ['Position contributions are unavailable in the saved results. Regenerate the character analysis to include this ranking.', '']
    return text


def scope_by_id(data, key='scopes'):
    return {s['id']: s for s in data.get(key, [])} if data else {}


def correlations_section(result, reason):
    text = section('## Prepared depth and score improvement', 'correlations')
    if not result:
        return text + [f'Correlation analysis unavailable: {reason}.', '']
    text += ['Depth and delta are both measured from chapter entry. These correlations compare chapters, rather than estimating the effect of adding one move. '
             'Brackets contain approximate 95% cluster-bootstrap confidence intervals.', '']
    primary = list(METRICS)[:4]
    text += table(['Repertoire', 'Chapters / groups', *[METRICS[k] for k in primary]],
                  [[color.title(), f"{r['n']} / {r['cluster_count']}", *[cell(r, k) for k in primary]] for color, r in result['results'].items()])
    text += ['<details>', '<summary>Other correlation variants and sensitivity checks</summary>', '']
    text += table(['Measure', *[c.title() for c in result['results']]],
                  [[label, *[cell(r, metric) for r in result['results'].values()]] for metric, label in list(METRICS.items())[4:]])
    for name, r in result.get('sensitivity', {}).items():
        text += [f"{escape(name.replace('_', ' ').capitalize())}: linear correlation {cell(r, 'pearson')}; rank correlation {cell(r, 'spearman')}.", '']
    text += ['</details>', '',
             'Baseline adjustment correlates the residuals after fitting both depth and delta against entry baseline and an intercept; pooled fits also include color. '
             'It is a sensitivity check for a different question. The bootstrap resamples transposition groups within each color, recomputing ranks, weights, and adjustments. '
             f"{result['bootstrap']['repetitions']:,} replicates were used. Intervals describe chapter-group variation conditional on the saved estimates; they do not propagate Explorer count uncertainty or establish causation.", '']
    return text


def methods(bundles):
    text = section('## Definitions and evidence', 'methods')
    text += ['- **Repertoire score:** wins count as 1 and draws as 0.5. Your selected moves are forced; opponent replies use all database replies, including departures from preparation. Evaluation stops at a theory leaf, deviation, or other stopping outcome. Exact transpositions share a position.',
             '- **Baseline and delta:** overall uses the standard starting-position database score for the same color. Chapter baseline is the weighted database score at first-entry positions, using the same entry weights as chapter score. Delta is repertoire score minus baseline, in percentage points (pp). These population comparisons do not establish personal or causal improvement.',
             '- **Elo equivalent:** use `E(p) = 400 * log10(p / (1 - p))` and report `E(repertoire score) - E(starting baseline)`, following the [logistic expected-score equation](https://tec.fide.com/wp-content/uploads/2025/07/Statistical_model_for_chess_tournament_simulations.pdf). Combined scores use 50% White and 50% Black with matching filters and standard starting positions. Average the scores first, rather than averaging the separate Elo equivalents. Conversion is unavailable at scores of 0% or 100%; no clipping or prior is substituted. This is a score-scale translation, not a personal rating forecast or a measured causal gain.',
             '- **Centipawn equivalent:** CP columns use `C(p) = ln(p / (1 - p)) / 0.00368208`, the inverse [Lichess score curve](https://lichess.org/page/accuracy), applied directly to the adjacent expected score (wins plus half draws). Positive CP favors the repertoire owner, including Black. CP delta is `C(after) - C(before)`, or score CP minus entry/starting-baseline CP in overview tables. Convert the two scores separately; converting their percentage-point difference is incorrect. Weighted or transposed mixtures, including combined colors and chapter entries, average the scores before conversion. Score intervals convert their endpoints. Missing values and boundaries of 0% or 100% are unavailable. Reach, reply frequency, and contribution percentages are not expected scores and receive no CP conversion. This calibration translates human results to a centipawn scale; it is not an engine evaluation.',
             '- **Position reach:** probability of visiting a canonical repertoire board before preparation ends, under the overall selected policy. Incoming probabilities from every transposed route are combined into one row. Prepared endpoints and the first unprepared opponent replies are included. Unprepared reply reach comes from the cached parent response table; incoming mass from transposed routes is combined, and parent chapters provide context. No child queries are needed. Prepared rows include the repertoire continuation score; unprepared rows show cached database outcomes. Games count board or parent-move observations and are not the sample size of a continuation score. For merged unprepared arrivals, scores use incoming reach weights; pooled parent counts (†) may overlap. The displayed ranking omits boards immediately before a prepared own move, retaining the board after the reply; own-turn endpoints with no reply are labeled. Missing opponent distributions leave only known incoming mass, with downstream reach unresolved. The representative route follows selected own moves and observed opponent replies. Source links identify chapters containing the board, including shared introductory positions.',
             '- **Entry probability:** first arrival anywhere in the chapter region before the model stops, counting each modeled game once per chapter, including transpositions. Chapters overlap; entry probabilities, scores, and deltas are not additive. Representative lines show one route; reach combines all modeled routes.',
             '- **Alternative chapters:** overall uses the first PGN move in the earliest chapter unless explicitly overridden. An alternative chapter prefers its own first moves, with the overall policy elsewhere. Its score, reach, baseline, and depth use that comparison policy. Overall-policy reach is disclosed separately and does not mean an alternative move was selected.',
             '- **Expected prepared depth:** remaining prepared own moves averaged over entry routes and opponent replies. It includes an available own move at entry and compatible continuations from other chapters, with no discount or cutoff.',
             '- **Vulnerability drag:** local drag is the before score minus the after score, in percentage points. Opponent weighted drag multiplies this by reply reach. Our move drag is parent database score minus repertoire continuation score after the selected move, without a reach multiplier. Own strengths use the opposite difference. CP delta always subtracts before from after, so its sign is opposite to positive drag. The benchmarks differ. Nested rows overlap; drag is a screen, not an additive decomposition or promised gain. Cached alternatives are ranked observed outcomes, not recommendations.',
             '- **Position contribution:** reach × score, in percentage points, across all reached prepared positions and first unprepared opponent replies. Prepared positions use the repertoire continuation score; unprepared replies use the cached parent-row score or recorded endpoint database score. Exact transpositions share one board. Boards before guaranteed own replies and the standard starting board are omitted, as in common positions. These rows carry overlapping downstream results, so they must not be summed and are not a decomposition or a measure of improvement. Chapter values are conditional on entry. Unresolved scores are excluded. The separate stopping-outcome ledger in JSON remains additive because each modeled game stops once. † marks pooled parent counts that may overlap.',
             '- **Reuse:** over N independent games, an own decision with probability p has Np expected encounters and probability 1−(1−p)^N of appearing at least once. Exact transpositions share a memorization decision. This measures exposure, not retention. Chapter curves mean games entering that chapter.',
             '- **Effective replies:** 2 raised to the reach-and-recorded-fraction-weighted mean reply entropy. Missing or zero reply evidence is unavailable, not perfect predictability. Effective pawn structures similarly measure frequency-weighted diversity of exact pawn squares at the end of preparation, not future middlegame plans.',
             '- **Position features:** describe the board where preparation ends, including after an unprepared reply. King wings describe current files, not castling history. Isolated/doubled/passed shares mean at least one such pawn or file. Profile features can overlap.',
             '- **Evidence and uncertainty:** strengths and vulnerabilities exclude rows flagged sparse in their local, parent, or immediate endpoint evidence, before applying display limits. Position contributions exclude missing or sparse local game counts; a merged unprepared position is excluded if any parent-row arrival has sparse evidence, even if pooled counts exceed the threshold. Summary highlights use the same filter. All rows remain in JSON and the score, depth, reach, and uncertainty calculations still include sparse evidence. Elsewhere, (sparse) or * marks counts below the scoring sparse threshold. Parent games describe the local database benchmark, not the sample size of a longer continuation. Prepared reply counts measure frequency; their continuation scores may rely on other downstream samples. Missing scores remain unresolved. Approximate score intervals complete missing evidence with the configured prior and exclude unknown historical-game overlap, population mismatch, and selection bias. Conservative bounds and sparse sensitivity are different from these intervals.',
             '- **Opponent rating:** Explorer averageRating describes the move maker. At an opponent-turn board, local rating is the game-count-weighted mean of its cached response rows, including the Black repertoire’s starting-board row where White is to move. At our turn, it comes from the preceding opponent move row, combining transposed arrivals by modeled reach. At the White starting board there is no preceding opponent move, so its local context is marked n/a. Reply vulnerabilities use the specific reply. Rating Δ vs parent is that opponent reply’s mean minus the parent’s game-weighted opponent response mean; positive values describe a higher-rated reply cohort. Transposed replies combine paired differences by arrival reach, with comparison and parent coverage disclosed when partial. This is a difference between move-maker cohorts, not a White-minus-Black rating gap. Our-move rows use the resulting board’s opponent response mean. An unavailable alternative has no cached child evidence. Chapter averages weight rating evidence at stopping outcomes once per modeled game; entry-baseline ratings use the first-entry mixture. These can describe different populations and do not adjust any score. Partial coverage is shown as a rated percentage; missing ratings and unnamed residual outcomes are not zero. Pawn-group averages are limited to chapters. There are no rating averages for a repertoire color or the combined study.',
             '- **Chapter sources:** numbered chapter links identify every source of the exact final move or stopping position. Unprepared replies show parent context; immediate transpositions identify destination chapters. A representative route may pass through several chapters.', '']
    text += ['<details>', '<summary>Saved analysis files and validation</summary>', '']
    for b in bundles:
        r = b['report']; m = r['manifest']; color = r['color'].title()
        text += [f'### {color} evidence', '',
                 f"Analyzed {m['created_at']}. PGN: `{escape(m['input_path'])}`.", '',
                 f"Input SHA-256: `{m['input_sha256']}`. Score SHA-256: `{b['digest']}`.", '',
                 f"{m['positions']:,} graph positions; {m['evaluated_positions']:,} evaluated positions; "
                 f"{m['simulations']:,} score simulations; seed {m['seed']}; owner W/D/L prior {m['prior']}; sparse threshold {m['sparse_threshold']}.", '',
                 'Data files: ' + ' | '.join(f'[{label}]({path.name})' for label, path in
                     [('Scores', b['path'])] + [(family.title(), b['path'].with_suffix(f'.{family}.json')) for family in FAMILIES if family in b]) + '.', '']
        checks = ['Scoring sanity checks passed' if r.get('diagnostics', {}).get('sanity_checks_passed') else 'Scoring sanity checks unavailable']
        if 'vulnerabilities' in b:
            checks.append(f"vulnerability candidate-child queries: {b['vulnerabilities']['manifest']['candidate_child_queries']}")
        if 'preparation' in b or 'character' in b:
            checks.append('preparation and character use cached evidence only')
        text += ['; '.join(checks) + '. Companion score hashes and filters were checked before assembly.', '']
    return text + ['</details>', '']


def common_positions_section(bundles, limit=20):
    text = []
    introduction = ['Reach counts games with this color and includes all transposed routes. Prepared rows show continuation scores; unanswered replies show cached database scores. '
                    'Boards immediately before our prepared move are omitted. Positions overlap; do not add their reach. [Score sources and game counts](#methods).', '']
    for bundle in bundles:
        report = bundle['report']; color = report['color']; refs = Chapters(report)
        text += section('### Most common positions', f'{color}-common-positions')
        text += introduction
        scope = scope_by_id(bundle.get('character')).get('overall', {})
        if 'positions' not in scope:
            text += ['Position reach is not available in the saved results. Regenerate the character analysis to include this ranking.', '']
            continue
        if scope.get('unresolved_opponent_distribution_mass', 0) > 0:
            text += ['Only known reach is ranked. Missing opponent distributions leave downstream routes unresolved, so the ranking can change when evidence becomes available.', '']
        positions = visible_positions(scope)
        prepared = [r for r in positions if not unanswered_position(r, color)]
        unprepared = [r for r in positions if unanswered_position(r, color)]
        for title, rows, score_key, score_label in [
                ('Prepared positions', prepared, 'repertoire_score', 'Repertoire score'),
                ('Unprepared opponent replies', unprepared, 'database_score', 'Database score')]:
            if not rows:
                continue
            text += [f'#### {title}', '']
            text += table(['Position (representative line)', 'Chapter source', 'Reach in repertoire', score_label, 'Score CP', 'Games at position / reply', 'Avg opponent rating'] + (['Rating Δ vs parent'] if score_key == 'database_score' else []),
                          [[position_label(r, color), refs.sources(r), percentage(r['reach']), percentage(r.get(score_key)), cp(r.get(score_key)), position_games(r), opponent_rating(r.get('opponent_rating'))]
                           + ([rating_difference(r.get('opponent_rating'))] if score_key == 'database_score' else []) for r in rows[:limit]])
            text += [f'Showing {min(limit, len(rows))} of {len(rows):,} {title.lower()}.', '']
        text += ['Exact FENs and all position reach data are retained in the character JSON.', '']
    return text


def full_report(bundles, correlations, correlation_reason, top=10, chapter_top=2, position_top=20):
    combined = combined_overall(bundles)
    has_combined = combined is not None and 'unavailable' not in combined
    text = ['# Repertoire report', '',
            'Scores, chapter comparisons, vulnerabilities, position contributions, and repertoire character in one place. '
            'All scores are from the repertoire owner\'s perspective.', '',
            '[Summary](summary.md) | ' + ('[Combined](#combined) | ' if has_combined else '')
            + ' | '.join(f'[{b["report"]["color"].title()}](#{b["report"]["color"]})' for b in bundles)
            + ' | ' + ' | '.join(f'[{b["report"]["color"].title()} positions](#{b["report"]["color"]}-common-positions)' for b in bundles)
            + ' | [Depth correlations](#correlations) | [Definitions and evidence](#methods)', '',
            *snapshot_notes(bundles),
            *section('## Combined repertoire' if has_combined else '## Score overview', 'combined' if has_combined else 'overview'),
            *overview(bundles), overview_notes(bundles), '']
    for b in bundles:
        r = b['report']; color = r['color']; refs = Chapters(r); o = r['overall']
        characters = scope_by_id(b.get('character'))
        vulnerabilities = scope_by_id(b.get('vulnerabilities'), 'chapters')
        text += section(f'## {color.title()} repertoire', color)
        text += ['### Score and evidence limits', '']
        interval = o.get('posterior', {}).get('credible_interval_95')
        text += ['<details>', '<summary>Score and evidence limits</summary>', '']
        text += table(['Measure', 'Value', 'Centipawn equivalent (cp)'], [
            ['Approximate model-based 95% score interval', ' to '.join(map(percentage, interval)) if interval else 'unavailable', cp_interval(interval)],
            ['Unresolved probability', percentage(o['unresolved_mass']), 'n/a'],
            ['Conditional score bounds', ' to '.join(map(percentage, o['conditional_bounds'])), cp_interval(o['conditional_bounds'])],
            ['Sparse probability', percentage(o['sparse_mass']), 'n/a'],
            ['Sparse-score sensitivity', ' to '.join(map(percentage, o['sparse_sensitivity'])), cp_interval(o['sparse_sensitivity'])]])
        text += table(['Stopping type', 'Probability'], [[k.replace('_', ' ').capitalize(), percentage(v)] for k, v in o['masses'].items()])
        text += ['</details>', '']
        text += common_positions_section([b], position_top)
        text += section(f'### {color.title()} chapters ({len(r["chapters"])})', f'{color}-chapters')
        text += ['Entry probability is the chance of reaching a chapter; all other metrics below are conditional on first entry. '
                 'Alternative chapters use their own comparison policy. Chapters can overlap.', '', *chapter_table(r, refs)]
        text += vulnerabilities_section(b.get('vulnerabilities', {}).get('overall'), refs, top)
        text += strengths_section(b.get('vulnerabilities', {}).get('overall'), characters.get('overall'), refs, top,
                                  sparse_threshold=r['manifest']['sparse_threshold'])
        text += character_section(characters.get('overall'), refs, min(top, 5))
        text += ['<details>', '<summary>Uncertainty priorities and prior sensitivity</summary>', '',
                 'Priority is posterior mean reach multiplied by the stopping-score interval width. This is a review heuristic, not additive variance attribution.', '']
        text += table(['Stopping route', 'Chapter source / context', 'Priority (pp)', 'Games', 'Avg opponent rating', 'Rating Δ vs parent'],
                      [[line(' '.join(e['representative_path_san'])), refs.sources(e), number(100 * e['uncertainty_priority'], 4),
                        f"{e['sample_count']:,}", opponent_rating(e.get('opponent_rating')), rating_difference(e.get('opponent_rating'))] for e in sorted(r['events'], key=lambda e: e['uncertainty_priority'], reverse=True)[:top]])
        text += table(['Owner W/D/L prior', 'Approximate 95% score interval', 'Centipawn interval (cp)'],
                      [[str(p['prior']), ' to '.join(map(percentage, p['overall']['posterior']['credible_interval_95'])),
                        cp_interval(p['overall']['posterior']['credible_interval_95'])]
                       for p in r.get('prior_sensitivity', [])])
        text += ['</details>', '']
        for c in r['chapters']:
            cid = c['id']; s = c['score']; base = c.get('entry_baseline', {})
            title = f"{color[0].upper()}{refs.entries[cid][0]}. {escape(c['name'])}"
            text += section('#### ' + title, refs.anchor(cid))
            text += [f"Entry probability **{percentage(s.get('entry_probability'))}**; repertoire score **{percentage(s.get('raw_empirical_score'))} ({cp(s.get('raw_empirical_score'))} cp)**; "
                     f"entry baseline **{percentage(base.get('raw_score'))} ({cp(base.get('raw_score'))} cp)**; delta **{number(base.get('difference_pp'), signed=True)} pp / {cp_change(s.get('raw_empirical_score'), base.get('raw_score'))} cp**; "
                     f"prepared depth **{number(s.get('prepared_depth', {}).get('expected_moves'))} own moves**.", '']
            ratings = c.get('opponent_ratings', {})
            text += [f"Score-evidence opponent rating **{opponent_rating(ratings.get('score_evidence'))}**; "
                     f"entry-baseline opponent rating **{opponent_rating(ratings.get('entry_baseline'))}**. "
                     f"Rated coverage: continuation {percentage((ratings.get('score_evidence') or {}).get('known_coverage'))}; "
                     f"first entry {percentage((ratings.get('entry_baseline') or {}).get('known_coverage'))}.", '']
            if c.get('policy_overrides'):
                text += [f"**Alternative comparison.** Region reach under the overall policy: {percentage(s.get('overall_policy_entry_probability'))}.", '']
            char = characters.get(cid)
            if char and 'reuse' in char:
                text += [f"{char['reuse']['reachable_distinct_decisions']} distinct own decisions; {number(char['predictability']['effective_replies'])} effective replies; "
                         f"{number(char['position_profiles']['all']['effective_pawn_structures'])} effective boundary pawn structures.", '']
            text += ['<details>', f'<summary>Analysis, lines, and entry routes for {color[0].upper()}{refs.entries[cid][0]}</summary>', '']
            interval = s.get('posterior', {}).get('credible_interval_95')
            if interval:
                text += [f"Approximate 95% score interval: {' to '.join(map(percentage, interval))}; unresolved probability: {percentage(s.get('unresolved_mass'))}; "
                         f"sparse-score sensitivity: {' to '.join(map(percentage, s.get('sparse_sensitivity', [])))}.", '']
            text += vulnerabilities_section(vulnerabilities.get(cid), refs, chapter_top, level='#####')
            text += strengths_section(vulnerabilities.get(cid), char, refs, chapter_top, level='#####',
                                      sparse_threshold=r['manifest']['sparse_threshold'])
            text += character_section(char, refs, chapter_top, level='#####')
            transitions = [t for t in r.get('chapter_transitions', []) if t['source_id'] == cid
                           and t.get('conditional_probability') is not None and t['conditional_probability'] > 0]
            if transitions:
                text += ['**Most likely chapter transitions after entry**', '',
                         'Conditional reach of the destination at or after source entry, including shared or simultaneous entry. Rows overlap.', '']
                text += table(['Destination', 'Conditional probability'],
                              [[refs.label(t['destination_id'], True), percentage(t['conditional_probability'])]
                               for t in sorted(transitions, key=lambda t: t['conditional_probability'], reverse=True)[:5]])
            text += ['**Exact first-entry positions**', '',
                     'Weights are conditional on reaching this chapter. Each path is one representative route to an exact position.', '']
            text += table(['Entry route', 'First-entry weight', 'Chapter sources', 'Avg opponent rating'],
                          [[line(' '.join(e['path'])), percentage(e.get('conditional_first_entry_weight')), refs.sources(e), opponent_rating(e.get('opponent_rating'))] for e in c['entries']])
            text += ['</details>', '']
    text += correlations_section(correlations, correlation_reason) + methods(bundles)
    return '\n'.join(text) + '\n'


def summary_report(bundles, full_path, correlations, correlation_reason):
    text = ['# Repertoire summary', '', f'[Complete report]({Path(full_path).name})', '',
            *snapshot_notes(bundles), *overview(bundles), overview_notes(bundles), '',
            'The model follows your selected moves and database-weighted opponent replies. Scores describe the saved Lichess population.', '']
    for b in bundles:
        r = b['report']; color = r['color']; refs = Chapters(r)
        char = scope_by_id(b.get('character')).get('overall', {})
        moves = b.get('vulnerabilities', {}).get('overall', {})
        text += [f'## {color.title()} repertoire', '']
        own_bad = non_sparse_rows(moves.get('rankings', {}).get('own', []))
        own_good = non_sparse_rows(moves.get('strengths', []))
        if own_good:
            best = own_good[0]
            text += [f"Largest own-move strength: **{line(best['line'])}** ({refs.sources(best)}), "
                     f"**{number(best['local_gain_pp'], signed=True)} pp ({cp_change(best['move_score'], best['reference_score'])} cp)** above its parent database score; reach {percentage(best['branch_reach'])}.", '']
        if own_bad:
            worst = own_bad[0]
            text += [f"Largest own-move deficit: **{line(worst['line'])}** ({refs.sources(worst)}), "
                     f"**{number(worst['local_drop_pp'])} pp** below its parent database score; CP delta **{cp_change(worst['move_score'], worst['reference_score'])} cp**; reach {percentage(worst['branch_reach'])}.", '']
        for title, rows in [
                ('Most common unprepared replies', sorted([x for x in char.get('positions', []) if x['kind'] == 'unprepared_reply'], key=lambda x: -x['reach'])[:3]),
                ('Unprepared replies with the largest weighted drag', [x for x in non_sparse_rows(moves.get('rankings', {}).get('opponent', [])) if not x['prepared']][:3])]:
            if not rows:
                continue
            text += [f'### {title}', '']
            drag = 'branch_reach' in rows[0]
            text += table(['Line', 'Chapter context', 'Reach in repertoire']
                          + (['Before reply (repertoire)', 'Before CP'] if drag else [])
                          + ['After reply (database)' if drag else 'Database score', 'After CP' if drag else 'Score CP']
                          + (['CP delta', 'Drag (pp)', 'Weighted drag (pp)'] if drag else ['Reply games'])
                          + ['Avg opponent rating', 'Rating Δ vs parent'],
                          [[line(x['line']), refs.sources(x), percentage(x['branch_reach'] if drag else x['reach'])]
                           + ([percentage(x['reference_score']), cp(x['reference_score'])] if drag else [])
                           + [percentage(x['move_score'] if drag else x.get('database_score')), cp(x['move_score'] if drag else x.get('database_score'))]
                           + ([cp_change(x['move_score'], x['reference_score']), number(x['local_drop_pp']), number(x['weighted_drag_pp'], 4)] if drag else
                              [(f"{x['games']:,}" + ('†' if len(x.get('unprepared_origins', [])) > 1 else '') if x.get('games') is not None else 'unavailable')])
                           + [
                            opponent_rating(x.get('opponent_rating')), rating_difference(x.get('opponent_rating'))] for x in rows])
        contributions = non_sparse_rows(position_contributions(char, color, r['manifest']['sparse_threshold']))
        if contributions:
            text += ['### Largest position contributions', '',
                     'Reach × score across prepared positions and unprepared replies. These rows overlap; do not add them.', '',
                     *position_contribution_table(contributions[:3], char, refs)]
        masses = r['overall']['masses']
        text += [f"Preparation ends at an existing endpoint in **{percentage(masses.get('theory_leaf', 0))}** of modeled games "
                 f"and after an unprepared opponent reply in **{percentage(masses.get('deviation', 0))}**. "
                 + (f"Other stopping or unresolved cases account for {percentage(sum(v for k, v in masses.items() if k not in ('theory_leaf', 'deviation')))}."
                    if any(v > 0 for k, v in masses.items() if k not in ('theory_leaf', 'deviation')) else ''), '']
        if 'reuse' in char:
            reuse, p = char['reuse'], char['predictability']
            curve = next((x for x in reuse['curve'] if x['games'] == 100), None)
            text += [f"{reuse['reachable_distinct_decisions']} distinct own decisions; **{number(p['effective_replies'])} effective opponent replies**. "
                     + (f"After 100 modeled games, expect to encounter about {number(curve['expected_distinct_decisions'], 0)} of those decisions." if curve else ''), '']
        text += ['<details>', f'<summary>All {len(r["chapters"])} {color.title()} chapters</summary>', '',
                 'Chapter reach counts first arrival across all transposed routes. Scores, entry baselines, and depth are conditional on entry; chapters overlap.', '',
                 *chapter_table(r, refs), '</details>', '']
    text += ['Positive gain and delta are favorable; positive drag is a deficit. Own-move gains and deficits compare the full prepared continuation with the parent database score. '
             'Opponent drag weights the drop from its parent repertoire value by reply reach. These overlapping comparisons cannot be added.', '',
             'Chapter opponent rating describes score evidence at the preparation boundary, weighted once per modeled game. Line ratings describe their local opponent cohort. '
             'Positive rating Δ means a higher-rated opponent cohort than at the parent; ratings do not adjust scores. Partial coverage and sparse evidence are disclosed in the full report.', '']
    if any(c.get('policy_overrides') for b in bundles for c in b['report']['chapters']):
        text += ['**Alternative** chapters prefer their own first moves; overall uses the earliest chapter and first PGN choice. Overall-policy reach is shown separately.', '']
    text += [f'Per-chapter strengths and vulnerabilities, common positions, preparation character and evidence limits: [complete report]({Path(full_path).name}).', '']
    text = [re.sub(r'\]\(#((?:white|black)-[^)]+)\)', r'](' + Path(full_path).name + r'#\1)', x) for x in text]
    return '\n'.join(row.rstrip() for row in text)

def generate(paths, full_path=None, summary_path=None, *, strict=True, require_complete=False, top=10, chapter_top=2, position_top=20):
    if not paths or min(top, chapter_top, position_top) < 1:
        raise ValueError('Supply reports and positive table lengths')
    bundles = load(paths, strict, require_complete)
    correlations, reason = load_correlations(bundles, strict, require_complete)
    full_path = Path(full_path) if full_path else report_directory(bundles[0]['path']) / 'report.md'
    summary_path = Path(summary_path) if summary_path else full_path.parent / 'summary.md'
    if full_path.resolve() == summary_path.resolve():
        raise ValueError('Report and summary must be different files')
    if any(p.suffix.lower() != '.md' for p in (full_path, summary_path)):
        raise ValueError('Report and summary destinations must be Markdown (.md) files')
    full_text = full_report(bundles, correlations, reason, top, chapter_top, position_top)
    summary_text = summary_report(bundles, full_path, correlations, reason)
    full_text = full_text.replace('[Summary](summary.md)', f'[Summary]({summary_path.name})')
    # Use relative paths even when callers place the two outputs in different folders.
    import os
    full_text = full_text.replace(f'[Summary]({summary_path.name})', f'[Summary]({Path(os.path.relpath(summary_path, full_path.parent)).as_posix()})')
    summary_text = summary_text.replace(f']({full_path.name}', f']({Path(os.path.relpath(full_path, summary_path.parent)).as_posix()}')
    for b in bundles:
        for data in [b['path']] + [b['path'].with_suffix(f'.{family}.json') for family in FAMILIES if family in b]:
            full_text = full_text.replace(f']({data.name})', f']({Path(os.path.relpath(data, full_path.parent)).as_posix()})')
    for target, content in ((full_path, full_text), (summary_path, summary_text)):
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text(target, content)
    return bundles
