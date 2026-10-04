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

import chess
from .attribution import write_text
from .correlations import METRICS, cell
from .report import evidence_date, population_text
from .layout import report_directory
from .sharpness import FIELDS as OUTCOME_FIELDS, stopping_wdl, summarize as summarize_outcomes
from .spread import mixture as spread_mixture


FAMILIES = ('vulnerabilities', 'preparation', 'character', 'ratings', 'openings', 'insights')
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


def score_points(value, digits=2, signed=False):
    """Display an already percentage-scaled score difference or contribution."""
    return 'unresolved' if value is None else number(value, digits, signed) + '%'


def games_per_encounter(probability):
    """Mean waiting interval in independent games with the displayed reach basis."""
    if probability is None: return 'unavailable'
    if probability <= 0: return 'never'
    value = 1 / probability
    return f'{value:,.0f}' if value >= 10 else f'{value:.1f}'


def own_priorities(rows, strongest=False):
    field = 'local_gain_pp' if strongest else 'local_drop_pp'
    return sorted([r for r in non_sparse_rows(rows)
                   if r.get(field) is not None and r[field] > 0 and r['branch_reach'] > 0],
                  key=lambda r: (-r['branch_reach'] * r[field], -r['branch_reach'], r['line']))


def escape(value):
    return html.escape(str(value), quote=False).replace('|', '&#124;').replace('\n', ' ')


def line(value):
    """Display repeated PGN move numbers as conventional paired moves."""
    value = ' '.join(value.split())
    value = re.sub(r'(?<!\S)(\d+)\.(?!\.)(\S+)\s+\1\.\.\.(\S+)', r'\1. \2 \3', value)
    value = re.sub(r'(?<!\S)(\d+)\.\.\.(?=\S)', r'\1... ', value)
    value = re.sub(r'(?<!\S)(\d+)\.(?!\.)', r'\1. ', value)
    return escape(' '.join(value.split()) or '(starting position)')


class SourceCell(str):
    """Chapter links carry their neighboring opening cell through table rendering."""
    def __new__(cls, text, opening):
        result = super().__new__(cls, text)
        result.opening = opening
        return result


def table(headers, rows):
    if not rows:
        return []
    if 'Most common opening source' not in headers:
        source_columns = [i for i in range(len(headers)) if any(isinstance(row[i], SourceCell) for row in rows)
                          and headers[i] != 'Chapters']
        for i in reversed(source_columns):
            headers = [*headers[:i + 1], 'Most common opening source', *headers[i + 1:]]
            rows = [[*row[:i + 1], row[i].opening if isinstance(row[i], SourceCell) else 'unavailable', *row[i + 1:]] for row in rows]
    text_columns = ('Line', 'Position', 'Chapter', 'Source', 'Your move', 'Common reply', 'Observed alternative',
                    'Opening', 'Most common opening', 'ECO', 'Current names', 'Name source', 'Evidence source',
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
                    if family == 'character' and require_complete and m.get('outcome_schema_version') != 1:
                        reason = 'recursive WDL and sharpness not generated; regenerate character analysis'
                    if family == 'character' and require_complete and m.get('position_outcome_schema_version') != 1:
                        reason = 'position WDL and sharpness not generated; regenerate character analysis'
                    if family == 'openings' and m.get('schema_version') != 2:
                        reason = 'unsupported opening schema; regenerate opening analysis'
                        if strict: raise ValueError(f'{companion}: {reason}')
                    if family == 'openings' and m.get('source_schema_version') != 1:
                        reason = 'chapter opening sources not generated; regenerate opening analysis'
                        if strict: raise ValueError(f'{companion}: {reason}')
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
                    if family == 'insights':
                        if m.get('schema_version') != 1:
                            reason = 'unsupported report insight schema'
                        elif require_complete and m.get('recursive_spread_schema_version') != 1:
                            reason = 'recursive score spread not generated; use repertoire-insights --refresh-spread'
                        elif set(m.get('supporting_sha256', {})) != {'preparation', 'character', 'vulnerabilities', 'openings'}:
                            reason = 'missing report insight provenance'
                        elif any(name not in bundle or not path.with_suffix(f'.{name}.json').exists()
                                 or hashlib.sha256(path.with_suffix(f'.{name}.json').read_bytes()).hexdigest() != digest
                                 for name, digest in m['supporting_sha256'].items()):
                            reason = 'supporting analysis changed since insight generation'
                        if reason and strict:
                            raise ValueError(f'{companion}: {reason}; regenerate report insights')
                    if reason is None:
                        bundle[family] = data
            if reason:
                bundle['unavailable'][family] = reason
        if require_complete and bundle['unavailable']:
            raise ValueError(f'{color.title()} analyses incomplete: {bundle["unavailable"]}')
        if 'ratings' in bundle:
            from .ratings import attach
            attach(bundle)
        gap_scopes = scope_by_id(bundle.get('character'))
        report['gap_coverage'] = gap_scopes.get('overall', {}).get('gap_coverage')
        report['overall']['outcomes'] = gap_scopes.get('overall', {}).get('outcomes')
        for chapter in report['chapters']:
            chapter['gap_coverage'] = gap_scopes.get(chapter['id'], {}).get('gap_coverage')
            chapter['score']['outcomes'] = gap_scopes.get(chapter['id'], {}).get('outcomes')
        attach_outcomes(bundle)
        attach_insights(bundle)
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
    def __init__(self, report, openings=None):
        self.color = report['color']
        self.catalog = report['chapters']
        self.entries = {c['id']: (i, c) for i, c in enumerate(self.catalog, 1)}
        self.openings = openings or {}
        self.scope = 'overall'
        self.compact = False
        self.source_scopes = {s['id']: s for s in self.openings.get('source_scopes', [])}
        self.opening_anchors = {r['id']: f'{self.color}-opening-{i}' for i, r in enumerate(self.openings.get('openings', []), 1)}

    def for_scope(self, scope):
        import copy
        result = copy.copy(self)
        result.scope = scope or 'overall'
        return result

    def opening_source(self, row):
        if not self.openings:
            return 'unavailable'
        def canonical(position):
            return ' '.join(position.split()[:4]) if isinstance(position, str) else None
        position = canonical(row.get('position') or row.get('example_position'))
        move = row.get('move')
        parent = canonical(row.get('parent_position') or (row.get('position') if move else None))
        if parent and move:
            try:
                board = chess.Board(parent + ' 0 1')
                board.push_uci(move)
            except ValueError:
                return 'unavailable'
            position = ' '.join(board.fen(en_passant='legal').split()[:4])
        named = self.openings.get('positions', {}).get(position, {}).get('exact_name')
        if named:
            source = dict(id=named, share=1.)
        else:
            scope = self.source_scopes.get(self.scope, {})
            source = scope.get('entry_sources' if row.get('_opening_entry') else 'positions', {}).get(parent if parent and move else position)
        if source is None:
            return 'unavailable'
        identity = source['id']
        label = escape(identity) if identity else 'Unclassified'
        if identity in self.opening_anchors:
            label = f'[{label}](#{self.opening_anchors[identity]})'
        return f"{label} ({percentage(source['share'])})"

    def anchor(self, cid):
        return f'{self.color}-chapter-{self.entries[cid][0]}'

    def label(self, cid, full=False):
        if cid not in self.entries:
            return escape(cid)
        index, chapter = self.entries[cid]
        text = escape(chapter['name']) if full else f'{self.color[0].upper()}{index}'
        return f'[{text}](#{self.anchor(cid)})'

    def sources(self, row):
        return SourceCell(self.chapter_sources(row), self.opening_source(row))

    def chapter_sources(self, row):
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
        if not self.compact:
            return prefix + ', '.join(self.label(cid) for cid in ids)
        ordered = sorted(ids, key=lambda cid: self.entries.get(cid, (float('inf'),))[0])
        groups = []
        for cid in ordered:
            if groups and cid in self.entries and groups[-1][-1] in self.entries and self.entries[cid][0] == self.entries[groups[-1][-1]][0] + 1:
                groups[-1].append(cid)
            else:
                groups.append([cid])
        return prefix + ', '.join(self.label(g[0]) + ('-' + self.label(g[-1]) if len(g) >= 3 else
            (', ' + self.label(g[-1]) if len(g) == 2 else '')) for g in groups)


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
    outcomes = [r['overall'].get('outcomes') for r in (white, black)]
    combined_outcomes = (summarize_outcomes([average([o[field] for o in outcomes]) for field in OUTCOME_FIELDS])
                         if all(outcomes) else None)
    spreads = [r['overall'].get('branch_score_spread') for r in (white, black)]
    combined_spread = (spread_mixture([(.5, spread) for spread in spreads], score)
                       if all(s and 'within_stop_outcome_variance' in s for s in spreads) else None)
    return {'weights': {'white': .5, 'black': .5}, 'repertoire_score': score, 'starting_baseline': baseline,
            'difference_pp': None if score is None or baseline is None else 100 * (score - baseline),
            'elo_equivalent': elo_equivalent(score, baseline),
            'centipawn_equivalent': centipawn_equivalent(score),
            'baseline_centipawn_equivalent': centipawn_equivalent(baseline),
            'centipawn_delta': centipawn_delta(score, baseline),
            'expected_prepared_depth': average([r['overall'].get('prepared_depth', {}).get('expected_moves') for r in (white, black)]),
            'outcomes': combined_outcomes,
            'branch_score_spread': combined_spread,
            'chapters': sum(len(r['chapters']) for r in (white, black))}


def sharpness_display(outcomes):
    if not outcomes:
        return 'unavailable'
    return score_points(outcomes.get('sharpness'))


def spread_display(spread, immediate=True):
    if not spread:
        return 'unavailable'
    result = percentage(spread.get('standard_deviation'))
    if immediate and spread.get('immediate_standard_deviation') is not None:
        result += '<br>Reply ' + percentage(spread['immediate_standard_deviation'])
    elif immediate and spread.get('basis') == 'stopping score; no modeled future replies':
        result += '<br>Prep ends'
    return result


def attach_outcomes(bundle):
    """Join saved recursive position WDL to comparison rows without new queries."""
    character = scope_by_id(bundle.get('character'))
    vulnerabilities = bundle.get('vulnerabilities', {})
    scopes = [vulnerabilities.get('overall', {}), *vulnerabilities.get('chapters', [])]
    color = bundle['report']['color'] == 'white'
    for scope in scopes:
        positions = {r['position']: r for r in character.get(scope.get('id', 'overall'), {}).get('positions', [])}
        rows = list(scope.get('all_signed_rows', [])) + list(scope.get('strengths', []))
        for ranking in scope.get('rankings', {}).values():
            rows.extend(ranking)
        for row in rows:
            row['reference_outcomes'] = (positions.get(row['position'], {}).get('outcomes')
                                         if row['kind'] == 'opponent' else None)
            if row['prepared']:
                row['move_outcomes'] = positions.get(row['target'], {}).get('outcomes')
            else:
                fixed = row['move_score'] if row.get('score_basis') == 'terminal result' else None
                row['move_outcomes'] = summarize_outcomes(stopping_wdl(row['counts_white_draw_black'], color, fixed))


def attach_insights(bundle):
    insights = scope_by_id(bundle.get('insights'))
    characters = scope_by_id(bundle.get('character'))
    bundle['report']['overall']['branch_score_spread'] = insights.get('overall', {}).get('branch_score_spread')
    for chapter in bundle['report']['chapters']:
        chapter['score']['branch_score_spread'] = insights.get(chapter['id'], {}).get('branch_score_spread')
        chapter['entry_opening_sources'] = insights.get(chapter['id'], {}).get('entry_opening_sources', {})
    for sid, scope in characters.items():
        if sid in insights:
            scope['gap_priorities'] = insights[sid]['gaps']
            scope['branch_score_spread'] = insights[sid]['branch_score_spread']
            for row in scope.get('positions', []):
                row['branch_score_spread'] = insights[sid].get('position_spreads', {}).get(row['position'])
    for row in bundle.get('openings', {}).get('openings', []):
        added = bundle.get('insights', {}).get('opening_spreads', {}).get(row['id'], {})
        row['branch_score_spread'] = added.get('branch_score_spread')
        for entry in row['entries']:
            entry['branch_score_spread'] = added.get('entries', {}).get(entry['position'])
    for scope in [bundle.get('vulnerabilities', {}).get('overall', {}), *bundle.get('vulnerabilities', {}).get('chapters', [])]:
        added = insights.get(scope.get('id', 'overall'), {}).get('moves', {})
        rows = list(scope.get('all_signed_rows', [])) + list(scope.get('strengths', []))
        for values in scope.get('rankings', {}).values():
            rows.extend(values)
        for row in rows:
            row.update(added.get(row['id'], {}))


def score_cell(value):
    return percentage(value) + ' (' + cp(value) + ' cp)'


def delta_cell(after, before, *, drag=False):
    delta = None if after is None or before is None else 100 * (before-after if drag else after-before)
    return score_points(delta, signed=True) + ' (' + cp_change(after, before) + ' cp)'


def interval_cell(bounds):
    return ' to '.join(score_points(v, signed=True) for v in bounds) if bounds else 'unavailable'


def gain_split(row):
    if row.get('database_move_gain_pp') is None:
        return 'unavailable'
    return ('Move ' + score_points(row['database_move_gain_pp'], signed=True) +
            ' (' + cp_change(row.get('move_database_score'), row.get('reference_score')) + ' cp)<br>' +
            'Prep ' + score_points(row['continuation_gain_pp'], signed=True) +
            ' (' + cp_change(row.get('move_score'), row.get('move_database_score')) + ' cp)')


def evidence_snapshot(bundles):
    rows = []
    for bundle in bundles:
        overall = bundle['report']['overall']
        def score_range(bounds):
            return '<br>'.join(score_cell(v) for v in bounds) if bounds else 'unavailable'
        rows.append([bundle['report']['color'].title(), score_range(overall.get('posterior', {}).get('credible_interval_95')),
                     score_range(overall.get('sparse_sensitivity')), percentage(overall.get('sparse_mass'))])
    return ['### Score uncertainty', '',
            'Model intervals describe sampling under the saved population and priors. Sparse sensitivity lets flagged outcomes take any score from 0 to 1; '
            'it is a separate evidence stress test. [Limits and definitions](#methods).', '',
            *table(['Repertoire', 'Approximate 95% score interval', 'Sparse-evidence sensitivity', 'Sparse probability'], rows)]


def overview(bundles, compact=False):
    rows = []
    for b in bundles:
        r = b['report']; o = r['overall']; base = r.get('starting_position_reference', {}).get('owner_score')
        score = o.get('raw_empirical_score')
        delta = None if base is None or score is None else 100 * (score - base)
        elo = elo_equivalent(score, base)
        rows.append([r['color'].title(), percentage(base), cp(base), percentage(score), cp(score), score_points(delta, signed=True),
                     cp_change(score, base),
                     'unavailable' if elo is None else number(elo, signed=True),
                     number(o.get('prepared_depth', {}).get('expected_moves')), spread_display(o.get('branch_score_spread'), False), len(r['chapters'])])
    combined = combined_overall(bundles)
    if combined and 'unavailable' not in combined:
        rows.append(['**Combined**', percentage(combined['starting_baseline']), cp(combined['starting_baseline']),
                     percentage(combined['repertoire_score']), cp(combined['repertoire_score']),
                     score_points(combined['difference_pp'], signed=True), cp_change(combined['repertoire_score'], combined['starting_baseline']),
                     'unavailable' if combined['elo_equivalent'] is None else number(combined['elo_equivalent'], signed=True),
                     number(combined['expected_prepared_depth']), spread_display(combined.get('branch_score_spread'), False), combined['chapters']])
    if compact:
        return table(['Repertoire', 'Starting baseline', 'Repertoire score', 'Delta', 'Elo equivalent',
                      'Prepared depth', 'Score spread', 'Chapters'],
                     [[r[0], f'{r[1]} ({r[2]} cp)', f'{r[3]} ({r[4]} cp)', f'{r[5]} ({r[6]} cp)',
                       r[7], r[8], r[9], r[10]] for r in rows])
    return table(['Repertoire', 'Starting baseline', 'Baseline CP', 'Repertoire score', 'Score CP', 'Delta',
                  'CP delta', 'Elo equivalent', 'Prepared depth (own moves)', 'Score spread', 'Chapters'], rows)


def overview_notes(bundles):
    text = 'Delta is repertoire score minus its starting baseline. Score differences use % for percentage points: 55% minus 50% is +5%, rather than a relative percentage change. Elo equivalent converts that score difference to the 400-point logistic Elo scale; it is not a measured rating gain. '
    text += 'CP is the direct centipawn equivalent of the adjacent score using the inverse [Lichess score curve](https://lichess.org/page/accuracy). CP delta is score CP minus baseline CP; positive values favor the repertoire owner. Scores include half a point for draws. These are score-scale conversions rather than engine evaluations. '
    combined = combined_overall(bundles)
    if combined:
        if 'unavailable' in combined:
            text += 'Combined score unavailable: ' + combined['unavailable'] + '. '
        else:
            text += 'Combined gives White and Black equal weight (50% each), averaging their scores and baselines before both conversions. '
    return text + 'Prepared depth counts remaining own moves. Score spread measures recursive variation between prepared branch scores, in percentage points. A second line labeled Reply shows immediate opponent-reply spread using recursive continuation scores. Outcome volatility (formerly sharpness) remains the normalized WDL variance and appears in headlines and detailed breakdowns. Mix entry means and variances before calculating spread, including for combined colors. Strengths and vulnerability rankings omit sparse rows; scores still include all evidence.'


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


def gap_percentage(metrics, weighted=False):
    metrics = metrics or {}
    prefix = 'weighted_' if weighted else ''
    value = metrics.get(prefix + 'equivalent_gap_reach')
    if value is not None:
        return percentage(value)
    bounds = metrics.get(prefix + 'equivalent_gap_reach_bounds')
    return ' to '.join(map(percentage, bounds)) + ' (bounds)' if bounds is not None else 'unavailable'


def gap_section(scope, level='###', refs=None, top=10, compact=False):
    metrics = (scope or {}).get('gap_coverage')
    if not metrics:
        return [f'{level} Equivalent gap reach', '', 'Unavailable; regenerate the cache-only character analysis.', '']
    text = [f'{level} Equivalent gap reach', '',
            f'**{gap_percentage(metrics)}**. The reach of one gap that would produce the same repeat probability as all first unprepared positions combined. '
            'Lower values indicate less concentrated recurring gaps. [Definition and chapter weighting](#methods).', '']
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
            reach = percentage(priority['reach'])
            if compact:
                reach += '<br>1 per ' + games_per_encounter(priority['reach']) + ' games'
            rows.append([line(row['line']), refs.sources(row), reach,
                         percentage(priority['repeat_probability_share']), score_cell(row.get('database_score')),
                         spread_display(row.get('branch_score_spread')), position_games(row), rating + '<br>Δ ' + diff])
        text += ['**Gaps driving repeat encounters**', '',
                 *table(['First-gap position', 'Chapter source', reach_label(scope), 'Repeat-gap share',
                         'Database score', 'Score spread', 'Games', 'Avg opponent rating'], rows),
                 f"These {len(selected)} gaps account for **{percentage(selected[-1]['cumulative_repeat_probability_share'])}** "
                 f"of repeat-gap probability ({priorities['share_basis']}). "
                 + ('[Gap priority definition](#methods).' if compact else
                    'Share is gap reach squared divided by the sum of all gap reaches squared. All transposed arrivals are combined; sparse gaps retain their probability.'), '']
    return text


def branch_spread_section(scope, level='###'):
    spread = (scope or {}).get('branch_score_spread')
    if not spread:
        return []
    return [f'{level} Branch score spread', '',
        'Recursive spread of expected scores across preparation branches. Immediate reply differences and deeper continuation variance are combined before taking the square root. '
        'Shared transpositions reuse their continuation; entry and arrival cohorts retain their weights. Outcome volatility includes game-result variation after preparation ends. '
        '[Definition](#methods).', '',
        *table(['Measure', 'Value'], [[name, percentage(spread.get(field))] for name, field in (
            ('Total branch spread', 'standard_deviation'), ('Between stopping positions', 'between_position_deviation'),
            ('Between arrival cohorts at the same position', 'within_position_deviation'))]
            + [['Outcome volatility (0%-100%)', percentage(4 * spread['outcome_variance']) if spread.get('outcome_variance') is not None else 'unresolved'],
               ['Branch share of outcome variance', percentage(spread['variance'] / spread['outcome_variance'])
                if spread.get('outcome_variance') and spread.get('variance') is not None else 'n/a']])]


def chapter_table(report, refs, link=True, compact=False):
    alternatives = any(c.get('policy_overrides') for c in report['chapters'])
    headers = ['Chapter', 'Chapter reach (incl. transpositions)', 'Entry baseline', 'Baseline CP', 'Repertoire score',
               'Score CP', 'Delta', 'CP delta', 'Prepared depth', 'Score spread',
               'Weighted gap reach contribution', 'Equivalent gap reach after entry', 'Avg opponent rating']
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
               percentage(s.get('raw_empirical_score')), cp(s.get('raw_empirical_score')), score_points(base.get('difference_pp'), signed=True),
               cp_change(s.get('raw_empirical_score'), base.get('raw_score')),
               number(s.get('prepared_depth', {}).get('expected_moves')),
               spread_display(s.get('branch_score_spread'), False),
               gap_percentage(c.get('gap_coverage'), weighted=True), gap_percentage(c.get('gap_coverage')),
               opponent_rating(c.get('opponent_ratings', {}).get('score_evidence'))]
        if alternatives:
            row.insert(2, percentage(s.get('overall_policy_entry_probability', s.get('entry_probability'))))
        rows.append(row)
    if compact:
        cells = []
        for row, c in zip(rows, report['chapters']):
            s, base = c['score'], c.get('entry_baseline', {})
            reach = percentage(s.get('entry_probability'))
            if alternatives:
                reach += '<br>Overall ' + percentage(s.get('overall_policy_entry_probability', s.get('entry_probability')))
            cells.append([row[0], reach, score_cell(base.get('raw_score')), score_cell(s.get('raw_empirical_score')),
                delta_cell(s.get('raw_empirical_score'), base.get('raw_score')),
                number(s.get('prepared_depth', {}).get('expected_moves')), spread_display(s.get('branch_score_spread'), False),
                gap_percentage(c.get('gap_coverage')) + '<br>Weighted ' + gap_percentage(c.get('gap_coverage'), weighted=True),
                opponent_rating(c.get('opponent_ratings', {}).get('score_evidence'))])
        return table(['Chapter', 'Chapter reach (incl. transpositions)' + ('<br>Overall-policy reach' if alternatives else ''),
            'Entry baseline', 'Repertoire score', 'Delta', 'Prepared depth', 'Score spread',
            'Equivalent gap reach after entry<br>Weighted gap reach contribution', 'Avg opponent rating'], cells)
    return table(headers, rows)


def character_section(scope, refs, top, level='###'):
    if not scope or 'reuse' not in scope:
        return []
    refs = refs.for_scope(scope.get('id'))
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
                    refs.sources({'position': r.get('example_position'),
                                  'chapter_attribution': r.get('example_chapter_attribution')}), opponent_rating(r.get('opponent_rating'))]
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
    refs = refs.for_scope(scope.get('id'))
    field = 'local_gain_pp' if strongest else 'local_drop_pp'
    return table(['Line', 'Chapter source', reach_label(scope), 'Parent database score', 'Move database score',
                  'Repertoire score', 'Score spread', 'Gain / CP delta' if strongest else 'Drag / CP delta',
                  'Gain split: move / prep', 'Parent games', 'Avg opponent rating'],
                 [[line(r['line']), refs.sources(r), percentage(r['branch_reach']), score_cell(r['reference_score']),
                   score_cell(r.get('move_database_score')), score_cell(r['move_score']),
                   spread_display(r.get('move_spread')), delta_cell(r['move_score'], r['reference_score'], drag=not strongest)
                   + '<br>95%: ' + interval_cell(r.get('local_gain_interval_pp' if strongest else 'local_drop_interval_pp')),
                   gain_split(r), f"{r['parent_sample_count']:,}", opponent_rating(r.get('opponent_rating'))] for r in rows])


def vulnerabilities_section(scope, refs, top, level='###'):
    if not scope:
        return []
    refs = refs.for_scope(scope.get('id'))
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
                       'Before reply (repertoire)', 'After reply (repertoire)' if prepared else 'After reply (database)',
                       'Score spread<br>Before / after', 'CP delta', 'Drag', 'Weighted drag', 'Reply games', 'Avg opponent rating'],
                      [[line(r['line']), refs.sources(r), percentage(r['branch_reach']) + '<br>1 per ' + games_per_encounter(r['branch_reach']) + ' games',
                        percentage(r['branch_probability']), score_cell(r['reference_score']), score_cell(r['move_score']),
                        spread_display(r.get('reference_spread'), False) + '<br>' + spread_display(r.get('move_spread'), False),
                        cp_change(r['move_score'], r['reference_score']), score_points(r['local_drop_pp'])
                        + '<br>95%: ' + interval_cell(r.get('local_drop_interval_pp')), score_points(r['weighted_drag_pp'], 4),
                        f"{r['sample_count']:,}", opponent_rating(r.get('opponent_rating'))
                        + '<br>Δ ' + rating_difference(r.get('opponent_rating'))] for r in rows])
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
    refs = refs.for_scope(scope.get('id'))
    def kind(row):
        if row['kind'] == 'terminal': return 'Game over'
        if row['score_basis'] == 'database': return 'Unprepared reply'
        return 'Prepared endpoint' if row['kind'] == 'theory_leaf' else 'Prepared position'

    return table(['Position (representative line)', 'Chapter source / context', 'Position type', reach_label(scope),
                  'Avg games per encounter', 'Score', 'Score CP', 'Score spread', 'Contribution', 'Games at position / reply', 'Avg opponent rating'],
                 [[position_label(r, refs.color), refs.sources(r), kind(r), percentage(r['reach']), games_per_encounter(r['reach']), percentage(r['score']),
                   cp(r['score']), spread_display(r.get('branch_score_spread')), score_points(r['contribution_pp'], 4), position_games(r), opponent_rating(r.get('opponent_rating'))]
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
    text += ['- **Opening names and reach:** use only exact opening names in cached repertoire-position responses. At a named board, all arrivals take its exact current name. At an unnamed board, each incoming route retains its last name and probability; merging routes does not give an opening the other routes’ mass. Unclassified routes stay unclassified. Structural potential labels from recorded alternatives are retained in JSON but never create probability. Reach absorbs flow on the first arrival at an exact opening name or a known named variation; inheritance alone cannot introduce an opening that a route has not entered. Broader categories match existing cached names at colon or comma boundaries, so parent and variation rows overlap; unrelated names are transitions rather than parent relationships. Per-board origin contributions track the joint probability of reaching that board after previously entering each opening, even after a later name reset. Divide by total board reach to obtain that opening’s share of arrivals. These overlapping origins are not an exclusive partition. Opening scores, baselines, depth and gap distributions share the same normalized first-entry weights and overall selected policy. No external dataset or unprepared child queries are used.',
             '- **Repertoire score:** wins count as 1 and draws as 0.5. Your selected moves are forced; opponent replies use all cached database replies, including after the last recorded PGN move. Replies that immediately transpose into a known repertoire board resume preparation. Evaluation stops at an unanswered own-turn board, unprepared reply, terminal result, or unresolved evidence. Unprepared replies use the cached parent move row without fetching the child. Exact transpositions share a position; scores, reach, prepared depth, chapter entries and opening sources follow the same transitions.',
             '- **Outcome volatility (formerly sharpness):** normalized variance of the eventual game score, `400 * (W + D/4 - (W + D/2)**2)`, on a 0%-100% scale. Owner-relative WDL follows the same selected moves, empirical opponent replies, transpositions and stopping rules as repertoire score; W + D/2 reproduces that score. First-entry and combined-color WDL are mixed before calculating volatility. It is 100% for equal wins and losses, 10% for 5% wins / 90% draws / 5% losses, and zero for a certain result. It describes game-result variation, rather than tactical difficulty or reply sensitivity. Headlines and detailed branch breakdowns retain this secondary metric; the JSON field remains `sharpness` for compatibility. Missing outcome evidence stays unresolved. No prior, sparse filter or new query completes it.',
             '- **Baseline and delta:** overall uses the standard starting-position database score for the same color. Chapter baseline is the weighted database score at first-entry positions, using the same entry weights as chapter score. Delta is repertoire score minus baseline. Score differences and contributions display with %, using percentage points rather than relative percentage changes. These population comparisons do not establish personal or causal improvement.',
             '- **Elo equivalent:** use `E(p) = 400 * log10(p / (1 - p))` and report `E(repertoire score) - E(starting baseline)`, following the [logistic expected-score equation](https://tec.fide.com/wp-content/uploads/2025/07/Statistical_model_for_chess_tournament_simulations.pdf). Combined scores use 50% White and 50% Black with matching filters and standard starting positions. Average the scores first, rather than averaging the separate Elo equivalents. Conversion is unavailable at scores of 0% or 100%; no clipping or prior is substituted. This is a score-scale translation, not a personal rating forecast or a measured causal gain.',
             '- **Centipawn equivalent:** CP columns use `C(p) = ln(p / (1 - p)) / 0.00368208`, the inverse [Lichess score curve](https://lichess.org/page/accuracy), applied directly to the adjacent expected score (wins plus half draws). Positive CP favors the repertoire owner, including Black. CP delta is `C(after) - C(before)`, or score CP minus entry/starting-baseline CP in overview tables. Convert the two scores separately; converting their percentage-point difference is incorrect. Weighted or transposed mixtures, including combined colors and chapter entries, average the scores before conversion. Score intervals convert their endpoints. Missing values and boundaries of 0% or 100% are unavailable. Reach, reply frequency, and contribution percentages are not expected scores and receive no CP conversion. This calibration translates human results to a centipawn scale; it is not an engine evaluation.',
             '- **Position reach:** probability of visiting a canonical repertoire board before preparation ends, under the overall selected policy. Incoming probabilities from every transposed route are combined into one row. Prepared endpoints and all first unprepared opponent replies are included, even after the last recorded PGN move. An unanswered own-turn board has the same reach as its first-gap probability, summed over all transposed arrivals. Unprepared reply reach comes from the cached parent response table; incoming mass from transposed routes is combined, and parent chapters provide context. No child queries are needed. Prepared rows include the repertoire continuation score; unprepared rows show cached database outcomes. Games count board or parent-move observations and are not the sample size of a continuation score. For merged unprepared arrivals, scores use incoming reach weights; pooled parent counts (†) may overlap. The displayed ranking omits boards immediately before a prepared own move, retaining the board after the reply; own-turn endpoints with no reply are labeled. Missing opponent distributions leave only known incoming mass, with downstream reach unresolved. The representative route follows selected own moves and observed opponent replies. Source links identify chapters containing the board, including shared introductory positions.',
             '- **Entry probability:** first arrival anywhere in the chapter region before the model stops, counting each modeled game once per chapter, including transpositions. Chapters overlap; entry probabilities, scores, and deltas are not additive. Representative lines show one route; reach combines all modeled routes.',
             '- **Alternative chapters:** overall uses the first PGN move in the earliest chapter unless explicitly overridden. An alternative chapter prefers its own first moves, with the overall policy elsewhere. Its score, reach, baseline, and depth use that comparison policy. Overall-policy reach is disclosed separately and does not mean an alternative move was selected.',
             '- **Expected prepared depth:** remaining prepared own moves averaged over entry routes and opponent replies. It includes an available own move at entry and compatible continuations from other chapters, with no discount or cutoff.',
             '- **Equivalent gap reach:** `sqrt(sum(p_i ** 2))`, where `p_i` is the probability of first reaching an exact position with no prepared own reply. All transposed arrivals to the same gap are combined before squaring. Its square is the probability that two independent modeled games first encounter the same gap; the value itself is the reach of one gap with that repeat probability. Lower values mean gaps are less concentrated or less likely. Prepared endpoints use their cached opponent response table, and replies that transpose into preparation continue through the merged graph. Terminal games produce no gap. No unprepared child tables are fetched. All known gap probabilities are included, without sparse filtering, a depth cutoff, or normalization over gaps. Missing response distributions, unnamed residuals, and closed canonical cycles leave unresolved probability; displayed bounds allow that mass to spread over unseen gaps or join the largest known gap. These are conditional bounds, not sampling confidence intervals. Chapter equivalent gap reach uses the same normalized first-entry mixture and comparison policy as its score. Weighted gap reach contribution is chapter entry probability times its equivalent gap reach after entry. Chapters and gap boards overlap, and the square-root calculation is nonlinear, so these contributions are not additive. The same complete traversal is used for position reach, scores and prepared depth; unanswered position reach is validated against first-gap probability in every overall and chapter scope.',
             '- **Vulnerability drag:** local drag is the before score minus the after score, in percentage points. Opponent weighted drag multiplies this by reply reach. Our move drag is parent database score minus repertoire continuation score after the selected move, without a reach multiplier. Own strengths use the opposite difference. CP delta always subtracts before from after, so its sign is opposite to positive drag. The benchmarks differ. Nested rows overlap; drag is a screen, not an additive decomposition or promised gain. Cached alternatives are ranked observed outcomes, not recommendations.',
             '- **Position contribution:** reach × score, in percentage points, across all reached prepared positions and first unprepared opponent replies. Prepared positions use the repertoire continuation score; unprepared replies use the cached parent-row score or recorded endpoint database score. Exact transpositions share one board. Boards before guaranteed own replies and the standard starting board are omitted, as in common positions. These rows carry overlapping downstream results, so they must not be summed and are not a decomposition or a measure of improvement. Chapter values are conditional on entry. Unresolved scores are excluded. The separate stopping-outcome ledger in JSON remains additive because each modeled game stops once. † marks pooled parent counts that may overlap.',
             '- **Reuse:** over N independent games, an own decision with probability p has Np expected encounters and probability 1−(1−p)^N of appearing at least once. Exact transpositions share a memorization decision. This measures exposure, not retention. Chapter curves mean games entering that chapter.',
             '- **Encounter frequency and summary priorities:** avg games per encounter is 1 / modeled reach, assuming independent games. Chapter rows mean games entering that chapter, not all games with that color. Summary own comparisons rank reach times local continuation gain or drag; full-report own tables retain local rankings. Their continuation comparisons include later preparation and overlap, so weighted gains and drag must not be added or interpreted as the isolated causal value of a move.',
             '- **Prepared-depth distribution:** each probability counts remaining own prepared moves from the same roots or normalized first-entry mixture as the score. The survival column is the chance of at least that many moves; ending columns stop at exactly that depth. Sum the survival probabilities from depth 1 to reproduce expected prepared depth. A transposed board can receive histories with different elapsed depths; these remain distinct for this calculation. Missing reply distributions retain finite bounds from the longest legal prepared continuation, while missing leaf outcome counts do not obscure depth. Median is the smallest depth containing at least half the stopping probability.',
             '- **First-entry examples:** chapter reach counts first arrival anywhere in its region through any move order. Entry-position weight combines all first-arrival paths to that exact board. The displayed example is the most likely single actual first-arrival route, with its own smaller or equal weight. It stops on reaching the chapter and cannot pass an earlier chapter entry. Ordinary position-table lines remain representative board labels; they do not imply that every game followed that sequence.',
             '- **Effective replies:** 2 raised to the reach-and-recorded-fraction-weighted mean reply entropy. Missing or zero reply evidence is unavailable, not perfect predictability. Effective pawn structures similarly measure frequency-weighted diversity of exact pawn squares at the end of preparation, not future middlegame plans.',
             '- **Position features:** describe the board where preparation ends, including after an unprepared reply. King wings describe current files, not castling history. Isolated/doubled/passed shares mean at least one such pawn or file. Profile features can overlap.',
             '- **Evidence and uncertainty:** strengths and vulnerabilities exclude rows flagged sparse in their local, parent, or immediate endpoint evidence, before applying display limits. Position contributions exclude missing or sparse local game counts; a merged unprepared position is excluded if any parent-row arrival has sparse evidence, even if pooled counts exceed the threshold. Summary highlights use the same filter. All rows remain in JSON and the score, depth, reach, and uncertainty calculations still include sparse evidence. Elsewhere, (sparse) or * marks counts below the scoring sparse threshold. Parent games describe the local database benchmark, not the sample size of a longer continuation. Prepared reply counts measure frequency; their continuation scores may rely on other downstream samples. Missing scores remain unresolved. Approximate score intervals complete missing evidence with the configured prior and exclude unknown historical-game overlap, population mismatch, and selection bias. Conservative bounds and sparse sensitivity are different from these intervals.',
             '- **Opponent rating:** Explorer averageRating describes the move maker. At an opponent-turn board, local rating is the game-count-weighted mean of its cached response rows, including the Black repertoire’s starting-board row where White is to move. At our turn, it comes from the preceding opponent move row, combining transposed arrivals by modeled reach. At the White starting board there is no preceding opponent move, so its local context is marked n/a. Reply vulnerabilities use the specific reply. Rating Δ vs parent is that opponent reply’s mean minus the parent’s game-weighted opponent response mean; positive values describe a higher-rated reply cohort. Transposed replies combine paired differences by arrival reach, with comparison and parent coverage disclosed when partial. This is a difference between move-maker cohorts, not a White-minus-Black rating gap. Our-move rows use the resulting board’s opponent response mean. An unavailable alternative has no cached child evidence. Chapter averages weight rating evidence at stopping outcomes once per modeled game; entry-baseline ratings use the first-entry mixture. These can describe different populations and do not adjust any score. Partial coverage is shown as a rated percentage; missing ratings and unnamed residual outcomes are not zero. Pawn-group averages are limited to chapters. There are no rating averages for a repertoire color or the combined study.',
             '- **Chapter sources:** numbered chapter links identify every source of the exact final move or stopping position. Unprepared replies show parent context; immediate transpositions identify destination chapters. A representative route may pass through several chapters.',
             '- **Most common opening source:** the last cached opening name carried by the largest share of arrivals, followed by that share in parentheses. An exact cached name replaces earlier names. Unnamed transpositions retain each route’s probability, and unclassified arrivals compete as a separate source. Move comparisons use arrivals through that specific parent move; position rows combine all routes. Chapter tables use their comparison policy, and entry rows use first arrivals. Equal shares use alphabetical order, with named sources before unclassified ones. This is a share of arrivals, not reach in the whole repertoire.', '']
    text += [
        '- **Gap priorities:** a canonical gap with reach p contributes p² to repeat-gap probability. Its share is p² divided by the sum over every gap. All transposed arrivals are combined before squaring. Shares sum to 100% over known gaps; when response data are unresolved, they describe known repeat probability only. The five summary rows retain the full-distribution denominator, including sparse gaps.',
        '- **Move gain and preparation gain:** selected move database score minus parent database score, plus recursive repertoire score after the move minus its database score. The database move score comes from the already cached parent move row. Their sum equals total gain; drag is its negative. CP components subtract the corresponding converted scores and also sum to total CP delta. These compare sampled cohorts and preparation, rather than identifying causal benefits.',
        '- **Branch score spread:** `B(s) = sum(p_i * (B(child_i) + (score(child_i) - score(s))**2))`, displayed as `100 * sqrt(B(s))` in percentage points. Known stopping scores have B=0; forced own moves inherit their child. The recursion uses the scorer\'s exact empirical replies and shared canonical continuations, including preparation resumed through transpositions. It reproduces the complete stopping-event score variance. First-entry and combined-color mixtures include variance between their mean scores; standard deviations are never averaged. A second cell line labeled **Reply** is immediate opponent-reply spread, using recursive child scores. It is unavailable at own turns, stopping boundaries, or incomplete named reply distributions; pooled residuals still follow the scorer\'s stopping rules. **Prep ends** means no future branch spread is modeled, not a certain game result. Unprepared positions use their saved WDL mixture as a stopping outcome, with no child queries. Sparse samples remain included; unresolved scores or entry weights remain unavailable. The detailed breakdown separates stopping boards and incoming evidence cohorts. Total outcome variance equals branch variance plus weighted outcome variance within stopping events; branch share divides branch variance by total outcome variance. Variances add, standard deviations do not.',
        '- **Local comparison intervals:** approximate prior-completed 95% model intervals for gain or drag, using the saved simulation count, seed and priors. Opponent move frequencies and outcomes are sampled jointly, with the same sampled continuation value reused across transposed routes. Own move and parent database scores share one joint move/result table. Policies and population are fixed; weighted rankings retain empirical reach. Historical game overlap, selection effects and population mismatch are not modeled. These are local model intervals, not confidence intervals for causal improvement.',
        '- **Summary opening groups and chapter entries:** opening labels share a summary row only when their weighted first-entry board distributions coincide along guaranteed own moves and their repertoire scores match. This includes successive variations with sibling names. No opponent move is assumed. The downstream member supplies its baseline and delta, and reach is counted once; full reports retain every category. Equality of aggregate reach alone never groups categories. Chapter headlines show leading first-entry boards and their complete first-arrival opening-source mixture under the chapter comparison policy, combining all transposed arrivals. Example lines are individual routes, not exclusive move orders.', '']
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


def common_positions_for_scope(scope, refs, limit=20, level='###', anchor=None):
    scope = scope or {}
    refs = refs.for_scope(scope.get('id'))
    color = refs.color
    text = section(f'{level} Most common positions', anchor)
    denominator = ('Reach counts games with this color' if scope.get('id', 'overall') == 'overall'
                   else 'Reach is conditional on first reaching any position in this chapter under its comparison policy')
    text += [denominator + ' and includes all transposed routes. A displayed line labels an exact board; games can reach it through other move orders. '
             + ('Some games enter this chapter later and never visit its defining board. ' if scope.get('id', 'overall') != 'overall' else '')
             + 'Prepared rows show continuation scores; unanswered replies show cached database scores. '
             'Boards immediately before our prepared move are omitted. Positions overlap; do not add their reach. [Score sources and game counts](#methods).', '']
    if 'positions' not in scope:
        return text + ['Position reach is not available in the saved results. Regenerate the character analysis to include this ranking.', '']
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
        text += [f'{level}# {title}', '']
        text += table(['Position (representative line)', 'Chapter source', reach_label(scope), 'Avg games per encounter', score_label, 'Score CP', 'Score spread', 'Games at position / reply', 'Avg opponent rating'] + (['Rating Δ vs parent'] if score_key == 'database_score' else []),
                      [[position_label(r, color), refs.sources(r), percentage(r['reach']), games_per_encounter(r['reach']), percentage(r.get(score_key)), cp(r.get(score_key)), spread_display(r.get('branch_score_spread')), position_games(r), opponent_rating(r.get('opponent_rating'))]
                       + ([rating_difference(r.get('opponent_rating'))] if score_key == 'database_score' else []) for r in rows[:limit]])
        text += [f'Showing {min(limit, len(rows))} of {len(rows):,} {title.lower()}.', '']
    return text + ['Exact FENs and all position reach data are retained in the character JSON.', '']


def common_positions_section(bundles, limit=20):
    text = []
    for bundle in bundles:
        report = bundle['report']
        scope = scope_by_id(bundle.get('character')).get('overall', {})
        text += common_positions_for_scope(scope, Chapters(report, bundle.get('openings')), limit,
                                           anchor=f'{report["color"]}-common-positions')
    return text


def depth_section(scope, level='###', anchor=None):
    text = section(f'{level} Prepared-depth distribution', anchor)
    distribution = (scope or {}).get('depth_distribution', {})
    if not distribution.get('survival'):
        return text + ['Depth distribution is unavailable; refresh the cache-only preparation analysis.', '']
    overall = scope.get('id', 'overall') == 'overall'
    text += [('Probabilities count games with this color.' if overall else
              'Probabilities are conditional on first reaching any position in this chapter, through any move order.'), '',
             'Depth counts remaining own prepared moves. The second column is cumulative; ending columns show the probability of stopping at exactly that depth. '
             'Transposed histories retain their different elapsed depths. Unknown reply distributions produce bounds; missing leaf results do not obscure depth.', '']
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
    return text + table(['Own moves prepared', 'At least this many', 'Prepared endpoint at this depth',
                         'Unprepared reply at this depth', 'Other ending at this depth', 'Unresolved from this depth'], rows)


def entry_routes_section(chapter, scope, refs):
    refs = refs.for_scope(chapter['id'])
    text = [f'<a id="{refs.anchor(chapter["id"])}-entries"></a>', '', '**Exact first-entry positions**', '',
            'The denominator is games first reaching any position in this chapter, through any move order. '
            'Each game counts once for this chapter. Entry-position weight combines every first-arrival route to the same board; '
            'example-route weight belongs only to the displayed route. Both weights are conditional on chapter entry.', '']
    routes = (scope or {}).get('entry_routes', {})
    entries = {e['position']: e for e in chapter['entries']}
    if routes.get('status') == 'resolved':
        rows = []
        for r in routes['positions']:
            original = entries[r['position']]
            example = r['example']
            rows.append([line(example['line']), percentage(r['conditional_first_entry_weight']),
                         percentage(example['conditional_probability']), refs.sources(dict(original, _opening_entry=True)),
                         opponent_rating(original.get('opponent_rating'))])
        text += ['Each example is an actual first-arrival path under the chapter comparison policy, chosen as the most likely single route to its entry board. '
                 'It has not passed an earlier chapter position. Entry-position opponent ratings combine all arrivals, rather than just the example route.', '']
        text += table(['First-entry example', 'Entry-position weight', 'Example-route weight', 'Entry-position sources',
                       'Entry-position opponent rating'], rows)
        zero = len(entries) - len(rows)
        if zero: text += [f'{zero} additional configured entry positions have zero modeled arrival.', '']
        return text
    text += ['Actual first-arrival examples are unavailable. The lines below label exact boards and do not imply that games followed those move orders.', '']
    return text + table(['Entry position (representative line)', 'First-entry weight', 'Chapter sources', 'Avg opponent rating'],
                        [[line(' '.join(e['path'])), percentage(e.get('conditional_first_entry_weight')), refs.sources(dict(e, _opening_entry=True)),
                          opponent_rating(e.get('opponent_rating'))] for e in chapter['entries']])


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
            labels = []
            for source in mixture:
                name = escape(source['id']) if source['id'] else 'Unclassified'
                if source['id'] in refs.opening_anchors:
                    name = f'[{name}](#{refs.opening_anchors[source["id"]]})'
                labels.append(name + ' (' + percentage(source['share']) + ')')
            cell = SourceCell(str(cell), labels[0] + ('<br>Also: ' + '; '.join(labels[1:]) if len(labels) > 1 else ''))
        rows.append([line(route['example']['line']), percentage(route['conditional_first_entry_weight']),
                     cell, opponent_rating(original.get('opponent_rating'))])
    return ['**Where this chapter starts**', '',
            'Leading first-entry boards, weighted over every transposed arrival. Opening sources describe those first arrivals; '
            'the example line is one route. Later entry boards can bypass the main defining position.', '',
            *table(['First-entry example', 'Entry-position weight', 'Chapter source', 'Avg opponent rating'], rows),
            f'[All entry positions and routes](#{refs.anchor(chapter["id"])}-entries).', '']


def summary_own_priorities(rows, refs, strongest=False):
    field, label = ('local_gain_pp', 'Gain') if strongest else ('local_drop_pp', 'Drag')
    return table(['Line', 'Chapter source', 'Reach in repertoire', 'Repertoire score', 'Score spread', label + ' / CP delta',
                  'Weighted ' + label.lower(), 'Avg opponent rating'],
                 [[line(r['line']), refs.sources(r), percentage(r['branch_reach']) + '<br>1 per ' + games_per_encounter(r['branch_reach']) + ' games', score_cell(r['move_score']),
                   spread_display(r.get('move_spread')),
                   delta_cell(r['move_score'], r['reference_score'], drag=not strongest)
                   + ('<br>Move ' + score_points(r['database_move_gain_pp'], signed=True) + '; prep '
                      + score_points(r['continuation_gain_pp'], signed=True) if r.get('database_move_gain_pp') is not None else '')
                   + '<br>95%: ' + interval_cell(r.get('local_gain_interval_pp' if strongest else 'local_drop_interval_pp')),
                   score_points(r['branch_reach'] * r[field], 4, signed=strongest),
                   opponent_rating(r.get('opponent_rating'))] for r in rows])


def opening_table(rows, refs, anchors, compact=False):
    if compact:
        lookup = {r['id']: r for r in refs.openings.get('openings', [])}
        values = []
        for row in rows:
            labels = []
            previous = None
            for identity in row.get('_summary_ids', [row['id']]):
                member = lookup.get(identity, row)
                name = member['name']
                short = name
                if previous:
                    for separator in (': ', ', '):
                        if name.startswith(previous + separator):
                            short = name[len(previous + separator):]
                            break
                    if short == name and ': ' in previous and ': ' in name:
                        family, variation = name.split(': ', 1)
                        if family == previous.split(': ', 1)[0]:
                            short = variation
                labels.append(('↳ ' if previous else '') + f'[{escape(short)}](#{anchors[identity]})')
                previous = name
            base, score = row['entry_baseline']['raw_score'], row['repertoire_score']
            values.append(['<br>'.join(labels), percentage(row['reach']), score_cell(base), score_cell(score),
                           delta_cell(score, base), spread_display(row.get('branch_score_spread'), False)])
        return table(['Opening family / variation', 'Reach', 'Entry baseline', 'Repertoire score', 'Delta', 'Score spread'], values)
    headers = ['Opening', 'ECO', 'Reach', 'Entry baseline', 'Baseline CP', 'Repertoire score', 'Score CP',
               'Delta', 'CP delta', 'Score spread']
    if not compact:
        headers += ['Prepared own moves after entry', 'Equivalent gap reach after entry', 'Chapters']
    values = []
    for row in rows:
        base, repertoire = row['entry_baseline']['raw_score'], row['repertoire_score']
        cells = [f"[{escape(row['name'])}](#{anchors[row['id']]})", escape(row['eco']), percentage(row['reach']),
                 percentage(base), cp(base), percentage(repertoire), cp(repertoire),
                 score_points(row['difference_pp'], signed=True), cp_change(repertoire, base), spread_display(row.get('branch_score_spread'), False)]
        if not compact:
            cells += [number(row['expected_prepared_moves']), gap_percentage(row['gap_coverage']),
                      refs.sources({'chapter_attribution': {'source_ids': row['chapter_ids']}})]
        values.append(cells)
    return table(headers, values)


def openings_section(bundle, refs, compact=False):
    data = bundle.get('openings')
    color = bundle['report']['color']
    text = section('### Openings reached', f'{color}-openings' if not compact else None)
    if data is None:
        return text + ['Opening analysis pending. Run `repertoire-openings` with the saved score results.', '']
    rows = data['openings']
    anchors = {row['id']: f'{color}-opening-{i}' for i, row in enumerate(rows, 1)}
    if compact:
        text += ['Opening names linked by guaranteed own moves share a row when their complete entry cohorts match. '
                 'The downstream opening supplies its baseline and delta; reach is counted once. '
                 'Opening reaches overlap. [Definitions](#methods).', '']
        lookup = {r['id']: r for r in rows}
        groups = bundle.get('insights', {}).get('opening_summary_groups') or [dict(ids=[r['id']], representative=r['id']) for r in rows]
        display = [dict(lookup[g['representative']], _summary_ids=g['ids']) for g in groups]
        display.sort(key=lambda r: (-r['reach'], r['name']))
        text += opening_table(display[:5], refs, anchors, True)
        return text + [f"{len(rows)} opening categories reached. [All openings and entry routes](#{color}-openings).", '']
    text += ['Reach counts first arrival at a named opening or its named variations, across all move orders under the selected repertoire. '
             'At unnamed transpositions, each opening keeps only its own incoming probability. Families and variations overlap; their reaches must not be added. '
             'Scores and entry baselines use the same weighted first entries. Identical names share one row with all cached ECO codes.', '']
    if not rows:
        return text + ['No cached opening names are reachable in this repertoire.', '']
    text += opening_table(rows[:5 if compact else 20], refs, anchors, compact)
    if compact:
        return text + [f"{len(rows)} opening categories reached. [All openings and entry routes](#{color}-openings).", '']
    coverage = data['coverage']
    text += [f"{len(rows)} reached categories; {coverage['named_repertoire_positions']} repertoire boards have exact cached names. "
             f"A known name is reached in {percentage(coverage['ever_classified_probability'])} of modeled games. "
             'Unnamed boards retain the name and probability share of each arriving route; a cached exact name replaces their current names. '
             'Origin contributions still show how much reach came through earlier openings. '
             'Family membership uses existing cached name prefixes. No new dataset or child-position queries are used.', '']
    if len(rows) > 20:
        text += ['<details>', f'<summary>Remaining {len(rows) - 20} openings</summary>', '',
                 *opening_table(rows[20:], refs, anchors), '</details>', '']
    text += ['<details>', '<summary>Opening entry positions and evidence</summary>', '',
             'Entry weight combines all first-arrival routes to the exact board. Each example is one real first-arrival route, '
             'and may represent only part of that board’s weight. Games describe entry evidence, not the sample size of the full recursive score. '
             'Unprepared entries use cached parent move rows; † marks pooled counts whose historical games can overlap. '
             'Ratings describe each entry board’s local opponent cohort, weighted by first-arrival flow. '
             'Opening groups have no aggregate opponent rating.', '']
    reached = {row['id']: row for row in rows}
    catalog = {row['id']: row for row in data['catalog']}
    sparse = bundle['report']['manifest']['sparse_threshold']
    for row in rows:
        text += section('#### ' + escape(row['name']) + ' (' + escape(row['eco']) + ')', anchors[row['id']])
        parents = [p for p in row['parent_ids'] if p in reached]
        if parents:
            text += ['Broader categories: ' + ', '.join(f"[{escape(catalog[p]['name'])}](#{anchors[p]})" for p in parents) + '.', '']
        values = []
        for entry in row['entries']:
            labels = data['positions'][entry['position']]
            values.append([line(entry['example']['line']), refs.sources({'chapter_attribution': {'source_ids': entry['chapter_ids']}}),
                refs.opening_source({'position': entry['position']}),
                'Exact' if labels['exact_name'] else 'Inherited', percentage(entry['conditional_weight']),
                percentage(entry['entry_probability']), percentage(entry['baseline_score']), cp(entry['baseline_score']),
                percentage(entry['repertoire_score']), cp(entry['repertoire_score']), spread_display(entry.get('branch_score_spread')),
                f"{entry['games']:,}" + ('†' if len(entry['origins']) > 1 else '') + (' (sparse)' if entry['games'] < sparse else ''),
                entry['games_source'], opponent_rating(entry.get('opponent_rating')), rating_difference(entry.get('opponent_rating'))])
        text += table(['First-entry example', 'Chapter source / context', 'Most common opening source', 'Name source', 'Entry weight', 'Reach in repertoire',
                       'Entry baseline', 'Baseline CP', 'Repertoire score', 'Score CP', 'Score spread', 'Entry games', 'Evidence source',
                       'Avg opponent rating', 'Rating Δ vs parent'], values)
        origins = []
        scope = scope_by_id(bundle.get('character')).get('overall', {})
        for position in visible_positions(scope):
            origin = data['positions'].get(position['position'], {})
            contribution = origin.get('opening_reach_contributions', {}).get(row['id'], 0.)
            if contribution:
                origins.append((position, contribution, origin['opening_reach_fractions'][row['id']]))
        if origins:
            text += ['**Most common positions reached through this opening**', '',
                     'Total position reach includes every route. Reach through this opening includes only games that first entered it. '
                     'Their ratio shows this opening’s share of arrivals, including after a later name change.', '']
            text += table(['Position', 'Chapter source / context', 'Total position reach', 'Reach through this opening', 'Share of arrivals'],
                [[line(position['line']), refs.sources(position), percentage(position['reach']), percentage(contribution), percentage(fraction)]
                 for position, contribution, fraction in sorted(origins, key=lambda item: (-item[1], -item[0]['reach']))[:5]])
    return text + ['</details>', '']


def full_report(bundles, correlations, correlation_reason, top=10, chapter_top=5, position_top=20):
    combined = combined_overall(bundles)
    has_combined = combined is not None and 'unavailable' not in combined
    text = ['# Repertoire report', '',
            'Scores, chapter comparisons, vulnerabilities, position contributions, and repertoire character in one place. '
            'All scores are from the repertoire owner\'s perspective.', '',
            '[Summary](summary.md) | ' + ('[Combined](#combined) | ' if has_combined else '')
            + ' | '.join(f'[{b["report"]["color"].title()}](#{b["report"]["color"]})' for b in bundles)
            + ' | ' + ' | '.join(f'[{b["report"]["color"].title()} positions](#{b["report"]["color"]}-common-positions)' for b in bundles)
            + ' | ' + ' | '.join(f'[{b["report"]["color"].title()} openings](#{b["report"]["color"]}-openings)' for b in bundles)
            + ' | [Depth correlations](#correlations) | [Definitions and evidence](#methods)', '',
            *snapshot_notes(bundles),
            *section('## Combined repertoire' if has_combined else '## Score overview', 'combined' if has_combined else 'overview'),
            *overview(bundles), overview_notes(bundles), '', *evidence_snapshot(bundles)]
    for b in bundles:
        r = b['report']; color = r['color']; refs = Chapters(r, b.get('openings')); o = r['overall']
        characters = scope_by_id(b.get('character'))
        preparation = scope_by_id(b.get('preparation'))
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
        text += gap_section(characters.get('overall'), refs=refs, top=top)
        text += branch_spread_section(characters.get('overall'))
        text += common_positions_section([b], position_top)
        text += openings_section(b, refs)
        text += depth_section(preparation.get('overall'), anchor=f'{color}-prepared-depth')
        text += section(f'### {color.title()} chapters ({len(r["chapters"])})', f'{color}-chapters')
        text += ['Entry probability is the chance of first reaching any position in a chapter through any move order. Scores, baselines, depth, and equivalent gap reach after entry are conditional on that entry. '
                 'Weighted gap reach contribution is chapter entry probability times equivalent gap reach after entry. '
                 'Alternative chapters use their own comparison policy. Chapters and gap positions can overlap, so weighted contributions are not additive.', '', *chapter_table(r, refs)]
        text += vulnerabilities_section(b.get('vulnerabilities', {}).get('overall'), refs, top)
        text += strengths_section(b.get('vulnerabilities', {}).get('overall'), characters.get('overall'), refs, top,
                                  sparse_threshold=r['manifest']['sparse_threshold'])
        text += character_section(characters.get('overall'), refs, min(top, 5))
        text += ['<details>', '<summary>Uncertainty priorities and prior sensitivity</summary>', '',
                 'Priority is posterior mean reach multiplied by the stopping-score interval width. This is a review heuristic, not additive variance attribution.', '']
        text += table(['Stopping route', 'Chapter source / context', 'Priority', 'Games', 'Avg opponent rating', 'Rating Δ vs parent'],
                      [[line(' '.join(e['representative_path_san'])), refs.sources(e), score_points(100 * e['uncertainty_priority'], 4),
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
                     f"entry baseline **{percentage(base.get('raw_score'))} ({cp(base.get('raw_score'))} cp)**; delta **{score_points(base.get('difference_pp'), signed=True)} / {cp_change(s.get('raw_empirical_score'), base.get('raw_score'))} cp**; "
                     f"prepared depth **{number(s.get('prepared_depth', {}).get('expected_moves'))} own moves**; "
                     f"outcome volatility **{sharpness_display(s.get('outcomes'))}**.", '']
            text += leading_entries(c, preparation.get(cid), refs)
            ratings = c.get('opponent_ratings', {})
            text += [f"Score-evidence opponent rating **{opponent_rating(ratings.get('score_evidence'))}**; "
                     f"entry-baseline opponent rating **{opponent_rating(ratings.get('entry_baseline'))}**. "
                     f"Rated coverage: continuation {percentage((ratings.get('score_evidence') or {}).get('known_coverage'))}; "
                     f"first entry {percentage((ratings.get('entry_baseline') or {}).get('known_coverage'))}.", '']
            if c.get('policy_overrides'):
                text += [f"**Alternative comparison.** Region reach under the overall policy: {percentage(s.get('overall_policy_entry_probability'))}.", '']
            char = characters.get(cid)
            if char and char.get('branch_score_spread'):
                text += [f"Branch score spread **{percentage(char['branch_score_spread'].get('standard_deviation'))}**. [Definition](#methods).", '']
            text += [f"Equivalent gap reach after entry **{gap_percentage(c.get('gap_coverage'))}**; "
                     f"weighted gap reach contribution **{gap_percentage(c.get('gap_coverage'), weighted=True)}**.", '']
            if char and 'reuse' in char:
                text += [f"{char['reuse']['reachable_distinct_decisions']} distinct own decisions; {number(char['predictability']['effective_replies'])} effective replies; "
                         f"{number(char['position_profiles']['all']['effective_pawn_structures'])} effective boundary pawn structures.", '']
            text += ['<details>', f'<summary>Analysis, lines, and entry routes for {color[0].upper()}{refs.entries[cid][0]}</summary>', '']
            interval = s.get('posterior', {}).get('credible_interval_95')
            if interval:
                text += [f"Approximate 95% score interval: {' to '.join(map(percentage, interval))}; unresolved probability: {percentage(s.get('unresolved_mass'))}; "
                         f"sparse-score sensitivity: {' to '.join(map(percentage, s.get('sparse_sensitivity', [])))}.", '']
            text += gap_section(char, level='#####', refs=refs, top=chapter_top)
            text += branch_spread_section(char, level='#####')
            text += common_positions_for_scope(char or {'id': cid}, refs, position_top, level='#####',
                                               anchor=refs.anchor(cid) + '-common-positions')
            text += depth_section(preparation.get(cid), level='#####', anchor=refs.anchor(cid) + '-prepared-depth')
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
            text += entry_routes_section(c, preparation.get(cid), refs)
            text += ['</details>', '']
    text += correlations_section(correlations, correlation_reason) + methods(bundles)
    return '\n'.join(text) + '\n'


def summary_report(bundles, full_path, correlations, correlation_reason):
    text = ['# Repertoire summary', '', f'[Complete report]({Path(full_path).name})', '',
            *snapshot_notes(bundles), *overview(bundles, compact=True),
            'Scores include half a point for draws. CP and Elo are score-scale equivalents; Elo is not a measured rating gain. Combined weights White and Black equally (50% each). '
            f'[Definitions and evidence]({Path(full_path).name}#methods).', '', *evidence_snapshot(bundles)]
    for b in bundles:
        r = b['report']; color = r['color']; refs = Chapters(r, b.get('openings'))
        refs.compact = True
        char = scope_by_id(b.get('character')).get('overall', {})
        moves = b.get('vulnerabilities', {}).get('overall', {})
        text += [f'## {color.title()} repertoire', '',
                 f"**Equivalent gap reach: {gap_percentage(char.get('gap_coverage'))}.** "
                 f"Branch score spread **{percentage((char.get('branch_score_spread') or {}).get('standard_deviation'))}**. "
                 f"Outcome volatility **{sharpness_display(r['overall'].get('outcomes'))}**. "
                 f'[Definitions]({Path(full_path).name}#methods).', '']
        text += openings_section(b, refs, compact=True)
        text += gap_section(char, refs=refs, top=5, compact=True)
        text += ['Own comparisons rank by reach × gain or drag. Move and prep components sum to the total gain; '
                 'the components describe historical cohorts and prepared continuations. Local rankings and model limits are in the full report.', '']
        own_bad = own_priorities(moves.get('rankings', {}).get('own', []))
        own_good = own_priorities(moves.get('strengths', []), strongest=True)
        if own_good:
            text += ['### Own moves with the largest weighted gain', '', *summary_own_priorities(own_good[:5], refs, strongest=True)]
        if own_bad:
            text += ['### Own moves with the largest weighted drag', '', *summary_own_priorities(own_bad[:5], refs)]
        replies = [x for x in non_sparse_rows(moves.get('rankings', {}).get('opponent', [])) if not x['prepared']][:5]
        if replies:
            text += ['### Unprepared replies with the largest weighted drag', '',
                     *table(['Line', 'Chapter context', 'Reach in repertoire', 'Scores before / after (CP)',
                             'Score spread<br>Before / after', 'Drag / CP delta', 'Weighted drag', 'Reply games', 'Avg opponent rating'],
                            [[line(x['line']), refs.sources(x), percentage(x['branch_reach'])
                              + '<br>1 per ' + games_per_encounter(x['branch_reach']) + ' games',
                              score_cell(x['reference_score']) + '<br>' + score_cell(x['move_score']),
                              spread_display(x.get('reference_spread'), False) + '<br>' + spread_display(x.get('move_spread'), False),
                              delta_cell(x['move_score'], x['reference_score'], drag=True)
                              + '<br>95%: ' + interval_cell(x.get('local_drop_interval_pp')),
                              score_points(x['weighted_drag_pp'], 4), f"{x['sample_count']:,}",
                              opponent_rating(x.get('opponent_rating')) + '<br>Δ ' + rating_difference(x.get('opponent_rating'))]
                             for x in replies])]
        contributions = non_sparse_rows(position_contributions(char, color, r['manifest']['sparse_threshold']))[:5]
        if contributions:
            text += ['### Largest position contributions', '',
                     'Reach × score carried through prepared positions and first gaps. These overlapping rows are not additive.', '',
                     *table(['Position', 'Chapter context', 'Reach in repertoire', 'Score', 'Score spread', 'Contribution', 'Avg opponent rating'],
                            [[position_label(x, color), refs.sources(x), percentage(x['reach']), score_cell(x['score']),
                              spread_display(x.get('branch_score_spread')), score_points(x['contribution_pp'], 4),
                              opponent_rating(x.get('opponent_rating'))] for x in contributions])]
        prep = scope_by_id(b.get('preparation')).get('overall', {}).get('depth_distribution', {})
        median = prep.get('median_moves')
        if median is not None:
            text += [f"Median prepared depth **{median} own moves**. [Full distribution]({Path(full_path).name}#{color}-prepared-depth).", '']
        if 'reuse' in char:
            reuse, pred = char['reuse'], char['predictability']
            curve = next((x for x in reuse['curve'] if x['games'] == 100), None)
            text += [f"{reuse['reachable_distinct_decisions']} distinct own decisions; **{number(pred['effective_replies'])} effective opponent replies**. "
                     + (f"After 100 modeled games, expect about {number(curve['expected_distinct_decisions'], 0)} distinct decisions." if curve else ''), '']
        text += ['<details>', f'<summary>All {len(r["chapters"])} {color.title()} chapters</summary>', '',
                 'Chapter scores and gap reach are conditional on first entry through any move order. '
                 'Weighted gap reach contribution multiplies the conditional value by chapter reach; overlapping chapters are not additive.', '',
                 *chapter_table(r, refs, compact=True), '</details>', '']
    text += ['Positive gain and delta are favorable; positive drag is a deficit. CP delta is after minus before. '
             'Avg games per encounter is 1 / reach for games with that color. Line comparisons overlap and cannot be added. '
             'Ratings describe local opponents and do not adjust scores. Local 95% bands are approximate prior-completed model intervals, not causal gain intervals.', '']
    if any(c.get('policy_overrides') for b in bundles for c in b['report']['chapters']):
        text += ['**Alternative** chapters prefer their own first moves; overall uses the earliest chapter and first PGN choice.', '']
    text += [f'Chapter entry explanations, strengths, vulnerabilities and evidence details: [complete report]({Path(full_path).name}).', '']
    text = [re.sub(r'\]\(#([^)]+)\)', r'](' + Path(full_path).name + r'#\1)', x) for x in text]
    return '\n'.join(row.rstrip() for row in text)


def generate(paths, full_path=None, summary_path=None, *, strict=True, require_complete=False, top=10, chapter_top=5, position_top=20):
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
