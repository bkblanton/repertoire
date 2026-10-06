"""Readable summary, report, chapter and opening pages from matching saved analysis JSONs.

Rendering never reads Explorer tables, recalculates metrics, or uses the network.
Companion hashes prevent mixing results from different repertoire snapshots.
"""
import hashlib
import html
import json
import math
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

import chess
from .attribution import write_text
from .correlations import cell
from .report import evidence_date, population_text
from .layout import report_directory
from .sharpness import FIELDS as OUTCOME_FIELDS, stopping_wdl, summarize as summarize_outcomes
from .spread import mixture as spread_mixture


FAMILIES = ('vulnerabilities', 'preparation', 'character', 'ratings', 'openings', 'insights')
LICHESS_WIN_CHANCE_COEFFICIENT = 0.00368208
LICHESS_ANALYSIS = 'https://lichess.org/analysis/standard/'
CHAPTER_PAGE = re.compile(r'[WB]\d+\.md')

# The summary uses one decimal and compact game counts; the full report keeps two decimals.
_display = ContextVar('display', default={'digits': 2, 'compact_counts': False})


@contextmanager
def display(**options):
    token = _display.set(dict(_display.get(), **options))
    try:
        yield
    finally:
        _display.reset(token)


def percentage(value, digits=None):
    if value is None: return 'unresolved'
    digits = _display.get()['digits'] if digits is None else digits
    smallest = 10 ** -digits
    return f'<{smallest:.{digits}f}%' if 0 < 100 * value < smallest / 2 else f'{100 * value:.{digits}f}%'


def number(value, digits=2, signed=False):
    if value is None:
        return 'unresolved'
    if signed and 0 < abs(value) < .5 * 10 ** -digits:
        return ('+<' if value > 0 else '-<') + f'{10 ** -digits:.{digits}f}'
    return format(value, f'{"+" if signed else ""}.{digits}f')


def score_points(value, digits=None, signed=False):
    """Display an already percentage-scaled score difference or contribution."""
    digits = _display.get()['digits'] if digits is None else digits
    return 'unresolved' if value is None else number(value, digits, signed) + '%'


def per_thousand(value_pp, signed=False):
    """Reach-weighted percentage points as score points per 1,000 games with that color or scope."""
    if value_pp is None:
        return 'unresolved'
    value = 10 * value_pp
    digits = 0 if abs(value) >= 100 else 1 if abs(value) >= 1 else 2
    return number(value, digits, signed)


def count(value):
    if value is None:
        return 'unavailable'
    if not _display.get()['compact_counts'] or value < 10_000:
        return f'{value:,}'
    for size, suffix in ((1e9, 'B'), (1e6, 'M')):
        if value >= size:
            return f'{value / size:.1f}{suffix}'
    return f'{value / 1e3:.0f}k'


def plies(route):
    route = (route or '').strip()
    return 0 if route in ('', '(PGN root)') else len(route.split())


def analysis_url(position, route=''):
    """Open an exact canonical board on the Lichess analysis board and opening explorer."""
    fields = position.split()[:4]
    return LICHESS_ANALYSIS + '_'.join(fields + ['0', str(plies(route) // 2 + 1)])


def linked_line(route, position):
    text = line(route)
    return f'[{text}]({analysis_url(position, route)})' if position else text


def games_per_encounter(probability):
    """Mean waiting interval in independent games with the displayed reach basis."""
    if probability is None: return 'unavailable'
    if probability <= 0: return 'never'
    value = 1 / probability
    return f'{value:,.0f}' if value >= 10 else f'{value:.1f}'


def about(key, text='Definitions'):
    """Caveats live once in the glossary; tables link to the relevant entry."""
    return f'[{text}](#def-{key})'


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


def compressed_columns(headers, rows):
    """Keep related values together without discarding any table evidence."""
    headers, rows = list(headers), [list(row) for row in rows]
    if any(len(row) != len(headers) for row in rows):
        raise ValueError('Table header and row lengths differ')

    def combine(primary, secondary, title, format_cell):
        if primary not in headers or secondary not in headers:
            return
        first, second = headers.index(primary), headers.index(secondary)
        headers[first] = title
        for row in rows:
            row[first] = format_cell(row[first], row[second])
            del row[second]
        del headers[second]

    def paired_cp(value, equivalent):
        if equivalent == 'n/a':
            return value
        first, separator, rest = str(value).partition('<br>')
        endpoints, converted = first.split(' to '), str(equivalent).split(' to ')
        combined = (' to '.join(f'{p} ({c} cp)' for p, c in zip(endpoints, converted))
                    if len(endpoints) > 1 and len(endpoints) == len(converted)
                    else first + f' ({equivalent} cp)')
        return combined + (separator + rest if separator else '')

    for secondary, primaries in (
            ('Baseline CP', ('Starting baseline', 'Entry baseline')),
            ('Score CP', ('Repertoire score', 'Database score', 'Score')),
            ('Before CP', ('Before reply (repertoire)',)),
            ('After CP', ('After reply (repertoire)', 'After reply (database)')),
            ('Parent CP', ('Parent database score',)),
            ('Centipawn equivalent (cp)', ('Value',)),
            ('Centipawn interval (cp)', ('Approximate 95% score interval',))):
        for primary in primaries:
            combine(primary, secondary, primary, paired_cp)
    if 'Delta' in headers:
        combine('Delta', 'CP delta', 'Delta', paired_cp)
    elif 'Drag' in headers:
        combine('Drag', 'CP delta', 'Drag / CP delta', paired_cp)
    combine('Opening', 'ECO', 'Opening / ECO', lambda opening, eco: f'{opening}<br>{eco}')
    combine('Chapter reach (incl. transpositions)', 'Overall-policy reach',
            'Chapter reach (incl. transpositions)<br>Overall-policy reach',
            lambda reach, overall: f'{reach}<br>Overall {overall}')
    combine('Equivalent gap reach after entry', 'Weighted gap reach contribution',
            'Equivalent gap reach after entry<br>Weighted gap reach contribution',
            lambda reach, weighted: f'{reach}<br>Weighted {weighted}')
    for reach in ('Position reach', 'Position reach after chapter entry', 'Move reach', 'Move reach after chapter entry',
                  'Games leaving prep here', 'Games leaving prep here after chapter entry'):
        combine(reach, 'Avg games per encounter', reach + '<br>Avg games per encounter',
                lambda probability, games: f'{probability}<br>1 per {games} games')
    combine('Avg opponent rating', 'Rating Δ vs parent', 'Avg opponent rating<br>Rating Δ vs parent',
            lambda rating, difference: f'{rating}<br>Δ {difference}')
    return headers, rows


def table(headers, rows):
    if not rows:
        return []
    headers, rows = compressed_columns(headers, rows)
    if 'Most common opening source' not in headers:
        source_columns = [i for i in range(len(headers)) if any(isinstance(row[i], SourceCell) for row in rows)
                          and headers[i] != 'Chapters']
        for i in reversed(source_columns):
            headers = [*headers[:i + 1], 'Most common opening source', *headers[i + 1:]]
            rows = [[*row[:i + 1], row[i].opening if isinstance(row[i], SourceCell) else 'unavailable', *row[i + 1:]] for row in rows]
    text_columns = ('Line', 'Position', 'Chapter', 'Source', 'Your move', 'Common reply', 'Observed alternative',
                    'Opening', 'Most common opening', 'ECO', 'Current names', 'Name source', 'Evidence source',
                    'Feature', 'Stopping type', 'Measure', 'Destination', 'Frequent pawn', 'Preparation ends',
                    'Most common unprepared')
    return ['| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join('---' if i == 0 or (h.startswith(text_columns) and not h.startswith('Position reach')) else '---:'
                              for i, h in enumerate(headers)) + ' |',
            *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows], '']


def drop_uniform(headers, rows, optional):
    """Remove optional columns that carry the same value in every row, such as all-zero endings."""
    keep = [i for i, h in enumerate(headers)
            if h not in optional or len(rows) < 2 or len({str(row[i]) for row in rows}) > 1]
    return [headers[i] for i in keep], [[row[i] for i in keep] for row in rows]


def section(title, anchor=None):
    return ([f'<a id="{anchor}"></a>', ''] if anchor else []) + [title, '']


def report_navigation(text):
    """Build a nested contents list from actual second- and third-level headings."""
    existing = set(re.findall(r'<a id="([^"]+)"', '\n'.join(text)))
    body, headings, parent = [], [], None
    for item in text:
        heading = re.fullmatch(r'(##|###) (.+)', item)
        if heading:
            level, title = heading.groups()
            previous = next((line for line in reversed(body) if line.strip()), '')
            explicit = re.fullmatch(r'<a id="([^"]+)"></a>', previous)
            if explicit:
                anchor = explicit[1]
            else:
                slug = re.sub(r'[^a-z0-9]+', '-', html.unescape(title).lower()).strip('-')
                stem = f'{parent}-{slug}' if level == '###' and parent else slug
                anchor, suffix = stem, 2
                while anchor in existing:
                    anchor, suffix = f'{stem}-{suffix}', suffix + 1
                existing.add(anchor)
                body += [f'<a id="{anchor}"></a>', '']
            if level == '##':
                parent = anchor
            headings.append((level, title, anchor, parent))
        body.append(item)
    navigation = ['**Table of contents**', '', '- [Summary](summary.md)']
    for level, title, anchor, parent in headings:
        label = 'Chapter-by-chapter table' if anchor in ('white-chapters', 'black-chapters') else title
        indent = '    ' if level == '###' and parent else ''
        navigation.append(f'{indent}- [{label}](#{anchor})')
    navigation.append('')
    index = body.index('<!-- report-navigation -->')
    body[index:index+1] = navigation
    return body


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
    path = bundles[0]['path'].parent / 'prepared-depth-gain-correlation.json'
    if not path.exists():
        if require_complete:
            raise ValueError('Generate future preparation gain correlations before combining the reports')
        return None, 'not generated'
    result = json.loads(path.read_text(encoding='utf-8'))
    expected = {b['report']['color']: b for b in bundles}
    provenance = result.get('provenance', {})
    matches = result.get('schema_version') == 1 and set(result.get('results', {})) == set(expected)
    matches &= set(provenance) == set(expected) and all(
        p.get('report_sha256') == expected[c]['digest']
        and p.get('input_sha256') == expected[c]['report']['manifest']['input_sha256']
        and p.get('filters') == expected[c]['report']['manifest']['filters']
        and p.get('prior') == expected[c]['report']['manifest']['prior']
        and p.get('sparse_threshold') == expected[c]['report']['manifest']['sparse_threshold']
        and expected[c]['path'].with_suffix('.vulnerabilities.json').exists()
        and p.get('vulnerabilities_sha256') == hashlib.sha256(
            expected[c]['path'].with_suffix('.vulnerabilities.json').read_bytes()).hexdigest()
        for c, p in provenance.items())
    if not matches:
        if strict:
            raise ValueError(f'{path}: correlations belong to different score snapshots; regenerate them before combining')
        return None, 'belongs to different score snapshots'
    return result, None


def load_rating_correlations(bundles, strict=True):
    path = bundles[0]['path'].parent / 'opponent-rating-score-correlation.json'
    if not path.exists():
        return None, 'not generated'
    result = json.loads(path.read_text(encoding='utf-8'))
    expected = {b['report']['color']: b for b in bundles}
    matches = set(result.get('results', {})) == set(expected)
    for color, bundle in expected.items():
        provenance = result.get('results', {}).get(color, {}).get('provenance', {})
        hashes = provenance.get('hashes', {})
        matches &= (hashes.get('score') == bundle['digest']
                    and provenance.get('input_sha256') == bundle['report']['manifest']['input_sha256']
                    and provenance.get('filters') == bundle['report']['manifest']['filters']
                    and provenance.get('sparse_threshold') == bundle['report']['manifest']['sparse_threshold'])
        for family in ('ratings', 'vulnerabilities'):
            companion = bundle['path'].with_suffix(f'.{family}.json')
            matches &= (family in bundle and companion.exists()
                        and hashes.get(family) == hashlib.sha256(companion.read_bytes()).hexdigest())
    if not matches:
        if strict:
            raise ValueError(f'{path}: rating correlations belong to different analysis snapshots; regenerate them before combining')
        return None, 'belongs to different analysis snapshots'
    return result, None


class Chapters:
    """Compact clickable references preserve every source without giant cells."""
    def __init__(self, report, openings=None, boards=None):
        self.color = report['color']
        self.catalog = report['chapters']
        self.entries = {c['id']: (i, c) for i, c in enumerate(self.catalog, 1)}
        self.openings = openings or {}
        # One representative overall-policy route per exact board keeps labels identical across tables.
        self.boards = boards or {}
        self.scope = 'overall'
        self.compact = False
        self.source_scopes = {s['id']: s for s in self.openings.get('source_scopes', [])}
        self.opening_anchors = {r['id']: f'{self.color}-opening-{i}' for i, r in enumerate(self.openings.get('openings', []), 1)}

    def route(self, position, fallback=''):
        return self.boards.get(position, fallback)

    def position_cell(self, row, route=None):
        """Board label linked to the Lichess analysis board at that exact position."""
        route = self.route(row.get('position'), row.get('line', '')) if route is None else route
        return linked_line(route, row.get('position'))

    def move_route(self, row):
        """The parent board's shared label plus this move, so a board reads the same in every table."""
        parent, san = row.get('position'), row.get('move_san')
        if parent not in self.boards or not san:
            return row['line']
        prefix = self.boards[parent]
        number = plies(prefix) // 2 + 1
        token = f'{number}.{san}' if parent.split()[1] == 'w' else f'{number}...{san}'
        return token if plies(prefix) == 0 else f'{prefix} {token}'

    def move_cell(self, row):
        """Links open the board where the repertoire owner chooses: before our move, after their reply."""
        route = self.move_route(row)
        if row.get('kind') == 'opponent' and row.get('target'):
            return linked_line(route, row['target'])
        parent_route = ' '.join(route.split()[:-1])
        text = line(route)
        return f'[{text}]({analysis_url(row["position"], parent_route)})' if row.get('position') else text

    def page(self, cid):
        return f'{self.color[0].upper()}{self.entries[cid][0]}.md'

    def study_link(self, cid, text='Lichess study'):
        url = self.entries.get(cid, (None, {}))[1].get('url')
        return f'[{text}]({url})' if url else ''

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
        return self.opening_label(source)

    def opening_label(self, source):
        """Opening link, with its arrival share only when other sources also contribute."""
        identity = source['id']
        label = escape(identity) if identity else 'Unclassified'
        if identity in self.opening_anchors:
            label = f'[{label}](#{self.opening_anchors[identity]})'
        return label if source['share'] >= 1 - 1e-9 else f"{label} ({percentage(source['share'])})"

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


def delta_points(after, before, *, drag=False):
    delta = None if after is None or before is None else 100 * (before-after if drag else after-before)
    return score_points(delta, signed=True)


def headline_delta(after, before):
    """Headline deltas alone keep a centipawn-scale translation; other tables show percentages only."""
    value = centipawn_delta(after, before)
    return delta_points(after, before) + ('' if value is None else f' ({number(value, 0, signed=True)} cp)')


def interval_cell(bounds):
    return ' to '.join(score_points(v, signed=True) for v in bounds) if bounds else 'unavailable'


def gain_split(row):
    if row.get('database_move_gain_pp') is None:
        return 'unavailable'
    return ('Move ' + score_points(row['database_move_gain_pp'], signed=True) + '<br>'
            'Prep ' + score_points(row['continuation_gain_pp'], signed=True))


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


def overview(bundles, compact=False, focused=False):
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
    if compact and focused:
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


def gap_section(scope, level='###', refs=None, top=10, compact=False, anchor=None):
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
            reach = percentage(priority['reach'])
            if compact:
                reach += '<br>1 per ' + games_per_encounter(priority['reach']) + ' games'
            rows.append([refs.position_cell(row), refs.sources(row), reach,
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


def chapter_table(report, refs, link=True, compact=False):
    """One row per chapter: its page link, Lichess study link, and entry-conditional metrics."""
    alternatives = any(c.get('policy_overrides') for c in report['chapters'])
    rows = []
    for c in report['chapters']:
        s, base = c['score'], c.get('entry_baseline', {})
        index = refs.entries[c['id']][0]
        title = f"{report['color'][0].upper()}{index}. {escape(c['name'])}"
        if link:
            title = f'[{title}](#{refs.anchor(c["id"])})'
        if c.get('policy_overrides'):
            title += ' **(alternative)**'
        if refs.study_link(c['id']):
            title += '<br>' + refs.study_link(c['id'])
        reach = percentage(s.get('entry_probability'))
        if alternatives:
            reach += '<br>Overall ' + percentage(s.get('overall_policy_entry_probability', s.get('entry_probability')))
        rows.append([title, reach, percentage(base.get('raw_score')), percentage(s.get('raw_empirical_score')),
                     score_points(base.get('difference_pp'), signed=True),
                     number(s.get('prepared_depth', {}).get('expected_moves')), spread_display(s.get('branch_score_spread'), False),
                     gap_percentage(c.get('gap_coverage')) + '<br>Weighted ' + gap_percentage(c.get('gap_coverage'), weighted=True),
                     opponent_rating(c.get('opponent_ratings', {}).get('score_evidence'))])
    return table(['Chapter', 'Chapter reach (incl. transpositions)' + ('<br>Overall-policy reach' if alternatives else ''),
                  'Entry baseline', 'Repertoire score', 'Delta', 'Prepared depth', 'Score spread',
                  'Equivalent gap reach after entry<br>Weighted gap reach contribution', 'Avg opponent rating'], rows)


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
    from .character import FEATURE_LABELS
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


def position_reach_label(scope):
    """Board reach combines every transposed arrival."""
    return 'Position reach' if scope.get('id', 'overall') == 'overall' else 'Position reach after chapter entry'


def move_reach_label(scope):
    """Move reach counts only arrivals at the parent that continue with this move."""
    return 'Move reach' if scope.get('id', 'overall') == 'overall' else 'Move reach after chapter entry'


def non_sparse_rows(rows):
    """Filter local, parent, and immediate endpoint evidence before display limits."""
    return [r for r in rows if not any(r.get(field, False)
            for field in ('sparse', 'parent_sparse', 'continuation_endpoint_sparse'))]


def own_move_table(rows, scope, refs, strongest=False):
    refs = refs.for_scope(scope.get('id'))
    return table(['Line', 'Chapter source', move_reach_label(scope), 'Parent database score', 'Move database score',
                  'Repertoire score', 'Score spread', 'Gain' if strongest else 'Drag',
                  'Gain split: move / prep', 'Parent games', 'Avg opponent rating'],
                 [[refs.move_cell(r), refs.sources(r), percentage(r['branch_reach']), percentage(r['reference_score']),
                   percentage(r.get('move_database_score')), percentage(r['move_score']),
                   spread_display(r.get('move_spread')), delta_points(r['move_score'], r['reference_score'], drag=not strongest)
                   + '<br>95%: ' + interval_cell(r.get('local_gain_interval_pp' if strongest else 'local_drop_interval_pp')),
                   gain_split(r), count(r['parent_sample_count']), opponent_rating(r.get('opponent_rating'))] for r in rows])


def reply_table(rows, scope, refs, prepared):
    """Unprepared replies stop preparation, so only the spread before the reply is informative."""
    spread_header = 'Score spread<br>Before / after' if prepared else 'Score spread before reply'
    def spread(r):
        before = spread_display(r.get('reference_spread'), False)
        return before + '<br>' + spread_display(r.get('move_spread'), False) if prepared else before
    return table(['Line', 'Chapter source / context', move_reach_label(scope), 'Reply frequency at parent',
                  'Before reply (repertoire)', 'After reply (repertoire)' if prepared else 'After reply (database)',
                  spread_header, 'Drag', 'Drag per 1,000 games', 'Reply games', 'Avg opponent rating'],
                 [[refs.move_cell(r), refs.sources(r), percentage(r['branch_reach']) + '<br>1 per ' + games_per_encounter(r['branch_reach']) + ' games',
                   percentage(r['branch_probability']), percentage(r['reference_score']), percentage(r['move_score']), spread(r),
                   score_points(r['local_drop_pp']) + '<br>95%: ' + interval_cell(r.get('local_drop_interval_pp')),
                   per_thousand(r['weighted_drag_pp']), count(r['sample_count']), opponent_rating(r.get('opponent_rating'))
                   + '<br>Δ ' + rating_difference(r.get('opponent_rating'))] for r in rows])


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


def position_label(row, color, refs=None):
    text = refs.position_cell(row) if refs else line(row['line'])
    if row['kind'] == 'terminal':
        return text + '<br>Game over'
    if row['kind'] in ('theory_leaf', 'unprepared_reply') and row['to_move'] == color:
        return text + f'<br>{color.title()} to move; no prepared reply'
    return text


def position_games(row):
    games = row.get('games')
    if games is None: return 'unavailable'
    return count(games) + ('†' if len(row.get('unprepared_origins', [])) > 1 else '')


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

    headers, values = drop_uniform(
        ['Position (representative line)', 'Chapter source / context', 'Position type', position_reach_label(scope),
         'Avg games per encounter', 'Score', 'Score spread', 'Contribution per 1,000 games', 'Games at position / reply', 'Avg opponent rating'],
        [[position_label(r, refs.color, refs), refs.sources(r), kind(r), percentage(r['reach']), games_per_encounter(r['reach']), percentage(r['score']),
          spread_display(r.get('branch_score_spread')), per_thousand(r['contribution_pp']), position_games(r), opponent_rating(r.get('opponent_rating'))]
         for r in rows], {'Position type'})
    return table(headers, values)


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


def scope_by_id(data, key='scopes'):
    return {s['id']: s for s in data.get(key, [])} if data else {}


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
             'Points use observed database counts; brackets contain approximate 95% model-based intervals.', '']
    primary = ('reach_weighted_pearson', 'reach_weighted_spearman', 'slope_pp_per_move')
    text += table(['Repertoire', 'Own decisions', 'Linear correlation (95% interval)', 'Rank correlation (95% interval)',
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
             f"Intervals use {result['simulations']:,} joint Dirichlet evidence draws for this fixed repertoire, with the score model's saved prior. "
             'Each canonical table is sampled once per draw and reused at every transposition and shared continuation; scores, depths, reach weights and selected-move baselines are recomputed together. '
             'The 2.5th and 97.5th percentiles describe finite-database uncertainty under this model. Distinct board tables are still treated as independent, '
             'although their historical games can overlap. These are individual model-based intervals, not chapter-bootstrap intervals or evidence of causal improvement.', '']
    text += ['The interval describes correlations of jointly sampled underlying tables. Uncertainty in continuation scores and the saved prior can shift this nonlinear statistic, '
             'so the interval need not contain the observed-count point estimate.', '']
    return text


def rating_correlations_section(result, reason):
    from .rating_correlations import cell as rating_cell
    text = section('### Opponent rating and score improvement', 'rating-correlations')
    if not result:
        return text + [f'Rating correlation analysis unavailable: {reason}. '
                       'Generate it with `uv run python -m repertoire_score.rating_correlations` and the saved score files.', '']
    text += ['This compares opponent replies from the same parent position. Rating delta is the reply cohort\'s average opponent rating minus the '
             'reach-weighted mean rating of eligible replies at that parent. Score delta is its continuation score minus the '
             'corresponding reach-weighted mean continuation score. Prepared replies use the recursive repertoire score; '
             'unprepared replies use their cached parent move results. Weights are parent reach × reply frequency. '
             'Canonical parent/reply pairs are counted once across transpositions; sparse replies are excluded.', '',
             'Negative correlation means that higher-rated reply cohorts tend to leave the repertoire with lower continuation scores. '
             'The slope expresses the associated score difference for +100 rating, in percentage points.', '']
    headers = ['Repertoire', 'Replies / parent positions', 'Linear correlation (95% interval)',
               'Score Δ per +100 rating (95% interval)']
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
             f"Brackets contain approximate 95% parent-bootstrap confidence intervals from {result['bootstrap_repetitions']:,} replicates. "
             'Each replicate keeps all replies from a parent board together. Shared downstream evidence and overlapping historical games can '
             'still link different parents. Intervals condition on the saved scores and ratings; they do not propagate database count uncertainty. '
             'This describes associations between move-maker cohorts, rather than the effect of changing an individual opponent\'s rating. '
             'The baseline is the local reply mean, and these scores are from the repertoire owner\'s perspective.', '']
    return text


def methods(bundles):
    text = section('## Definitions and evidence', 'methods')
    text += ['- <a id="def-opening-names"></a>**Opening names and reach:** use only exact opening names in cached repertoire-position responses. At a named board, all arrivals take its exact current name. At an unnamed board, each incoming route retains its last name and probability; merging routes does not give an opening the other routes’ mass. Unclassified routes stay unclassified. Structural potential labels from recorded alternatives are retained in JSON but never create probability. Reach absorbs flow on the first arrival at an exact opening name or a known named variation; inheritance alone cannot introduce an opening that a route has not entered. Broader categories match existing cached names at colon or comma boundaries, so parent and variation rows overlap; unrelated names are transitions rather than parent relationships. Per-board origin contributions track the joint probability of reaching that board after previously entering each opening, even after a later name reset. Divide by total board reach to obtain that opening’s share of arrivals. These overlapping origins are not an exclusive partition. Opening scores, baselines, depth and gap distributions share the same normalized first-entry weights and overall selected policy. No external dataset or unprepared child queries are used.',
             '- <a id="def-score"></a>**Repertoire score:** wins count as 1 and draws as 0.5. Your selected moves are forced; opponent replies use all cached database replies, including after the last recorded PGN move. Replies that immediately transpose into a known repertoire board resume preparation. Evaluation stops at an unanswered own-turn board, unprepared reply, terminal result, or unresolved evidence. Unprepared replies use the cached parent move row without fetching the child. Exact transpositions share a position; scores, reach, prepared depth, chapter entries and opening sources follow the same transitions.',
             '- <a id="def-volatility"></a>**Outcome volatility (formerly sharpness):** normalized variance of the eventual game score, `400 * (W + D/4 - (W + D/2)**2)`, on a 0%-100% scale. Owner-relative WDL follows the same selected moves, empirical opponent replies, transpositions and stopping rules as repertoire score; W + D/2 reproduces that score. First-entry and combined-color WDL are mixed before calculating volatility. It is 100% for equal wins and losses, 10% for 5% wins / 90% draws / 5% losses, and zero for a certain result. It describes game-result variation, rather than tactical difficulty or reply sensitivity. Because amateur database games are mostly decisive, it varies little between chapters, so only the branch spread breakdowns show it; the JSON field remains `sharpness` for compatibility. Missing outcome evidence stays unresolved. No prior, sparse filter or new query completes it.',
             '- <a id="def-baseline"></a>**Baseline and delta:** overall uses the standard starting-position database score for the same color. Chapter baseline is the weighted database score at first-entry positions, using the same entry weights as chapter score. Delta is repertoire score minus baseline. Score differences and contributions display with %, using percentage points rather than relative percentage changes. These population comparisons do not establish personal or causal improvement.',
             '- <a id="def-elo"></a>**Elo equivalent:** use `E(p) = 400 * log10(p / (1 - p))` and report `E(repertoire score) - E(starting baseline)`, following the [logistic expected-score equation](https://tec.fide.com/wp-content/uploads/2025/07/Statistical_model_for_chess_tournament_simulations.pdf). Combined scores use 50% White and 50% Black with matching filters and standard starting positions. Average the scores first, rather than averaging the separate Elo equivalents. Conversion is unavailable at scores of 0% or 100%; no clipping or prior is substituted. This is a score-scale translation, not a personal rating forecast or a measured causal gain.',
             '- <a id="def-cp"></a>**Centipawn equivalent:** CP values appear only beside headline deltas (overview rows and chapter headlines); every other table shows percentages. They use `C(p) = ln(p / (1 - p)) / 0.00368208`, the inverse [Lichess score curve](https://lichess.org/page/accuracy), applied directly to the expected score (wins plus half draws). Positive CP favors the repertoire owner, including Black. CP delta is `C(after) - C(before)`, or score CP minus entry/starting-baseline CP in overview tables. Convert the two scores separately; converting their percentage-point difference is incorrect. Weighted or transposed mixtures, including combined colors and chapter entries, average the scores before conversion. Score intervals convert their endpoints. Missing values and boundaries of 0% or 100% are unavailable. Reach, reply frequency, and contribution percentages are not expected scores and receive no CP conversion. This calibration translates human results to a centipawn scale; it is not an engine evaluation.',
             '- <a id="def-position-reach"></a>**Position reach:** probability of visiting a canonical repertoire board before preparation ends, under the overall selected policy. Incoming probabilities from every transposed route are combined into one row. Prepared endpoints and all first unprepared opponent replies are included, even after the last recorded PGN move. An unanswered own-turn board has the same reach as its first-gap probability, summed over all transposed arrivals. Unprepared reply reach comes from the cached parent response table; incoming mass from transposed routes is combined, and parent chapters provide context. No child queries are needed. Prepared rows include the repertoire continuation score; unprepared rows show cached database outcomes. Games count board or parent-move observations and are not the sample size of a continuation score. For merged unprepared arrivals, scores use incoming reach weights; pooled parent counts (†) may overlap. The displayed ranking omits boards immediately before a prepared own move, retaining the board after the reply; own-turn endpoints with no reply are labeled. Missing opponent distributions leave only known incoming mass, with downstream reach unresolved. The representative route follows selected own moves and observed opponent replies. Source links identify chapters containing the board, including shared introductory positions.',
             '- <a id="def-entry"></a>**Entry probability:** first arrival anywhere in the chapter region before the model stops, counting each modeled game once per chapter, including transpositions. Chapters overlap; entry probabilities, scores, and deltas are not additive. Representative lines show one route; reach combines all modeled routes.',
             '- <a id="def-alternatives"></a>**Alternative chapters:** overall uses the first PGN move in the earliest chapter unless explicitly overridden. An alternative chapter prefers its own first moves, with the overall policy elsewhere. Its score, reach, baseline, and depth use that comparison policy. Overall-policy reach is disclosed separately and does not mean an alternative move was selected.',
             '- <a id="def-depth"></a>**Expected prepared depth:** remaining prepared own moves averaged over entry routes and opponent replies. It includes an available own move at entry and compatible continuations from other chapters, with no discount or cutoff.',
             '- <a id="def-gap-reach"></a>**Equivalent gap reach:** `sqrt(sum(p_i ** 2))`, where `p_i` is the probability of first reaching an exact position with no prepared own reply. All transposed arrivals to the same gap are combined before squaring. Its square is the probability that two independent modeled games first encounter the same gap; the value itself is the reach of one gap with that repeat probability. Lower values mean gaps are less concentrated or less likely. Prepared endpoints use their cached opponent response table, and replies that transpose into preparation continue through the merged graph. Terminal games produce no gap. No unprepared child tables are fetched. All known gap probabilities are included, without sparse filtering, a depth cutoff, or normalization over gaps. Missing response distributions, unnamed residuals, and closed canonical cycles leave unresolved probability; displayed bounds allow that mass to spread over unseen gaps or join the largest known gap. These are conditional bounds, not sampling confidence intervals. Chapter equivalent gap reach uses the same normalized first-entry mixture and comparison policy as its score. Weighted gap reach contribution is chapter entry probability times its equivalent gap reach after entry. Chapters and gap boards overlap, and the square-root calculation is nonlinear, so these contributions are not additive. The same complete traversal is used for position reach, scores and prepared depth; unanswered position reach is validated against first-gap probability in every overall and chapter scope.',
             '- <a id="def-drag"></a>**Vulnerability drag:** local drag is the before score minus the after score, in percentage points. Opponent weighted drag multiplies this by reply reach. Our move drag is parent database score minus repertoire continuation score after the selected move, without a reach multiplier. Own strengths use the opposite difference. CP delta always subtracts before from after, so its sign is opposite to positive drag. The benchmarks differ. Nested rows overlap; drag is a screen, not an additive decomposition or promised gain. Cached alternatives are ranked observed outcomes, not recommendations.',
             '- <a id="def-contribution"></a>**Position contribution:** reach × score, in percentage points, across all reached prepared positions and first unprepared opponent replies. Prepared positions use the repertoire continuation score; unprepared replies use the cached parent-row score or recorded endpoint database score. Exact transpositions share one board. Boards before guaranteed own replies and the standard starting board are omitted, as in common positions. These rows carry overlapping downstream results, so they must not be summed and are not a decomposition or a measure of improvement. Chapter values are conditional on entry. Unresolved scores are excluded. The separate stopping-outcome ledger in JSON remains additive because each modeled game stops once. † marks pooled parent counts that may overlap.',
             '- <a id="def-reuse"></a>**Reuse:** over N independent games, an own decision with probability p has Np expected encounters and probability 1−(1−p)^N of appearing at least once. Exact transpositions share a memorization decision. This measures exposure, not retention. Chapter curves mean games entering that chapter.',
             '- <a id="def-encounter"></a>**Encounter frequency and summary priorities:** avg games per encounter is 1 / modeled reach, assuming independent games. Chapter rows mean games entering that chapter, not all games with that color. Summary own comparisons rank reach times local continuation gain or drag; full-report own tables retain local rankings. Their continuation comparisons include later preparation and overlap, so weighted gains and drag must not be added or interpreted as the isolated causal value of a move.',
             '- <a id="def-depth-distribution"></a>**Prepared-depth distribution:** each probability counts remaining own prepared moves from the same roots or normalized first-entry mixture as the score. The survival column is the chance of at least that many moves; ending columns stop at exactly that depth. Sum the survival probabilities from depth 1 to reproduce expected prepared depth. A transposed board can receive histories with different elapsed depths; these remain distinct for this calculation. Missing reply distributions retain finite bounds from the longest legal prepared continuation, while missing leaf outcome counts do not obscure depth. Median is the smallest depth containing at least half the stopping probability.',
             '- <a id="def-first-entry"></a>**First-entry examples:** chapter reach counts first arrival anywhere in its region through any move order. Entry-position weight combines all first-arrival paths to that exact board. The displayed example is the most likely single actual first-arrival route, with its own smaller or equal weight. It stops on reaching the chapter and cannot pass an earlier chapter entry. Ordinary position-table lines remain representative board labels; they do not imply that every game followed that sequence.',
             '- <a id="def-effective-replies"></a>**Effective replies:** 2 raised to the reach-and-recorded-fraction-weighted mean reply entropy. Missing or zero reply evidence is unavailable, not perfect predictability. Effective pawn structures similarly measure frequency-weighted diversity of exact pawn squares at the end of preparation, not future middlegame plans.',
             '- <a id="def-position-features"></a>**Position features:** describe the board where preparation ends, including after an unprepared reply. King wings describe current files, not castling history. Isolated/doubled/passed shares mean at least one such pawn or file. Profile features can overlap.',
             '- <a id="def-evidence"></a>**Evidence and uncertainty:** strengths and vulnerabilities exclude rows flagged sparse in their local, parent, or immediate endpoint evidence, before applying display limits. Position contributions exclude missing or sparse local game counts; a merged unprepared position is excluded if any parent-row arrival has sparse evidence, even if pooled counts exceed the threshold. Summary highlights use the same filter. All rows remain in JSON and the score, depth, reach, and uncertainty calculations still include sparse evidence. Elsewhere, (sparse) or * marks counts below the scoring sparse threshold. Parent games describe the local database benchmark, not the sample size of a longer continuation. Prepared reply counts measure frequency; their continuation scores may rely on other downstream samples. Missing scores remain unresolved. Approximate score intervals complete missing evidence with the configured prior and exclude unknown historical-game overlap, population mismatch, and selection bias. Conservative bounds and sparse sensitivity are different from these intervals.',
             '- <a id="def-rating"></a>**Opponent rating:** Explorer averageRating describes the move maker. At an opponent-turn board, local rating is the game-count-weighted mean of its cached response rows, including the Black repertoire’s starting-board row where White is to move. At our turn, it comes from the preceding opponent move row, combining transposed arrivals by modeled reach. At the White starting board there is no preceding opponent move, so its local context is marked n/a. Reply vulnerabilities use the specific reply. Rating Δ vs parent is that opponent reply’s mean minus the parent’s game-weighted opponent response mean; positive values describe a higher-rated reply cohort. Transposed replies combine paired differences by arrival reach, with comparison and parent coverage disclosed when partial. This is a difference between move-maker cohorts, not a White-minus-Black rating gap. Our-move rows use the resulting board’s opponent response mean. An unavailable alternative has no cached child evidence. Chapter averages weight rating evidence at stopping outcomes once per modeled game; entry-baseline ratings use the first-entry mixture. These can describe different populations and do not adjust any score. Partial coverage is shown as a rated percentage; missing ratings and unnamed residual outcomes are not zero. Pawn-group averages are limited to chapters. There are no rating averages for a repertoire color or the combined study.',
             '- <a id="def-chapter-sources"></a>**Chapter sources:** numbered chapter links identify every source of the exact final move or stopping position. Unprepared replies show parent context; immediate transpositions identify destination chapters. A representative route may pass through several chapters.',
             '- <a id="def-opening-source"></a>**Most common opening source:** the last cached opening name carried by the largest share of arrivals, followed by that share in parentheses. An exact cached name replaces earlier names. Unnamed transpositions retain each route’s probability, and unclassified arrivals compete as a separate source. Move comparisons use arrivals through that specific parent move; position rows combine all routes. Chapter tables use their comparison policy, and entry rows use first arrivals. Equal shares use alphabetical order, with named sources before unclassified ones. This is a share of arrivals, not reach in the whole repertoire.', '']
    text += [
        '- <a id="def-gap-priorities"></a>**Gap priorities:** a canonical gap with reach p contributes p² to repeat-gap probability. Its share is p² divided by the sum over every gap. All transposed arrivals are combined before squaring. Shares sum to 100% over known gaps; when response data are unresolved, they describe known repeat probability only. The five summary rows retain the full-distribution denominator, including sparse gaps.',
        '- <a id="def-gain"></a>**Move gain and preparation gain:** selected move database score minus parent database score, plus recursive repertoire score after the move minus its database score. The database move score comes from the already cached parent move row. Their sum equals total gain; drag is its negative. CP components subtract the corresponding converted scores and also sum to total CP delta. These compare sampled cohorts and preparation, rather than identifying causal benefits.',
        '- <a id="def-spread"></a>**Branch score spread:** `B(s) = sum(p_i * (B(child_i) + (score(child_i) - score(s))**2))`, displayed as `100 * sqrt(B(s))` in percentage points. Known stopping scores have B=0; forced own moves inherit their child. The recursion uses the scorer\'s exact empirical replies and shared canonical continuations, including preparation resumed through transpositions. It reproduces the complete stopping-event score variance. First-entry and combined-color mixtures include variance between their mean scores; standard deviations are never averaged. A second cell line labeled **Reply** is immediate opponent-reply spread, using recursive child scores. It is unavailable at own turns, stopping boundaries, or incomplete named reply distributions; pooled residuals still follow the scorer\'s stopping rules. **Prep ends** means no future branch spread is modeled, not a certain game result. Unprepared positions use their saved WDL mixture as a stopping outcome, with no child queries. Sparse samples remain included; unresolved scores or entry weights remain unavailable. The detailed breakdown separates stopping boards and incoming evidence cohorts. Total outcome variance equals branch variance plus weighted outcome variance within stopping events; branch share divides branch variance by total outcome variance. Variances add, standard deviations do not.',
        '- <a id="def-intervals"></a>**Local comparison intervals:** approximate prior-completed 95% model intervals for gain or drag, using the saved simulation count, seed and priors. Opponent move frequencies and outcomes are sampled jointly, with the same sampled continuation value reused across transposed routes. Own move and parent database scores share one joint move/result table. Policies and population are fixed; weighted rankings retain empirical reach. Historical game overlap, selection effects and population mismatch are not modeled. These are local model intervals, not confidence intervals for causal improvement.',
        '- <a id="def-opening-groups"></a>**Summary opening groups and chapter entries:** opening labels share a summary row only when their weighted first-entry board distributions coincide along guaranteed own moves and their repertoire scores match. This includes successive variations with sibling names. No opponent move is assumed. The downstream member supplies its baseline and delta, and reach is counted once; full reports retain every category. Equality of aggregate reach alone never groups categories. Chapter headlines show leading first-entry boards and their complete first-arrival opening-source mixture under the chapter comparison policy, combining all transposed arrivals. Example lines are individual routes, not exclusive move orders.',
        '- <a id="def-exits"></a>**Where preparation ends:** every modeled game leaves preparation once, at its first unprepared position. Those stops are grouped by the last prepared board before them: the board where the opponent chose an unprepared reply, or an own-turn board with no recorded move. **Games leaving prep here** sums the first-gap probability of every unprepared reply from that board, including database results grouped without an individual move row (shown as unrecorded replies), so all rows together add to 100% of games apart from finished games and unresolved evidence; unlike most tables, these rows do not overlap. **Share of games at this position** divides that by the board\'s own reach: 100% means preparation always stops there (often a chapter that simply ends), while a small share means rare sidelines at a busy position. The database score averages the cached parent-row results of those replies by reach. Rows come from the saved stopping ledger, without new queries.',
        '- <a id="def-move-reach"></a>**Position reach and move reach:** position reach is the probability of reaching an exact board by any move order. Move reach is the probability of reaching the parent board and then playing that particular move, so the board after a move can have higher position reach than the move itself when other routes transpose into it. Position tables show position reach; move comparisons show move reach.',
        '- <a id="def-per-thousand"></a>**Points per 1,000 games:** reach-weighted values (weighted drag, weighted gain, and position contribution) are shown as score points per 1,000 games with that color, or per 1,000 games entering the chapter on chapter pages. One point is one win; a draw is half a point. A weighted drag of 0.3 means about 0.3 points lost per 1,000 games, relative to the comparison score.',
        '- <a id="def-links"></a>**Board labels and links:** each exact board uses one representative overall-policy route as its label in every table, with move rows adding their move to the parent board\'s label. Actual first-entry routes keep their own move orders. Line links open the Lichess analysis board at the position where the repertoire owner decides: after the opponent\'s reply, or before our move. Chapter names link to their Lichess study chapters.', '']
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


def board_routes(bundle):
    """Overall-policy route per exact board, shared by every table that labels that board."""
    scope = scope_by_id(bundle.get('character')).get('overall', {})
    return {r['position']: r['line'] for r in scope.get('positions', [])}


def bundle_refs(bundle):
    return Chapters(bundle['report'], bundle.get('openings'), board_routes(bundle))


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


def exit_points(scope):
    """Group first unprepared positions by the last prepared board, so one study task is one row.

    Every modeled game stops once, so these groups partition the stopping mass and do not overlap.
    """
    positions = {r['position']: r for r in (scope or {}).get('positions', [])}
    groups = {}
    for stop in (scope or {}).get('stopping_outcomes', []):
        # Finished games and missing opponent data are not places where preparation runs out.
        finished = stop['type'] == 'theory_leaf' and (positions.get(stop['position'], {}).get('kind') == 'terminal'
                                                       or (stop.get('sample_count') == 0 and stop.get('score') is not None))
        if (stop['type'] not in ('deviation', 'theory_leaf', 'no_recorded_continuation') or finished
                or not stop['reach'] > 0):
            continue
        board = stop.get('parent_position') or stop['position']
        group = groups.setdefault(board, dict(position=board, reach=0., known=0., score_mass=0., replies=[],
                                              leaf=False, unrecorded=0.))
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
    rows = []
    for group in groups.values():
        board = positions.get(group['position'], {})
        node = board.get('reach')
        rows.append(dict(group, line=board.get('line', ''), kind=board.get('kind'), node_reach=node,
                         share=group['reach'] / node if node else None,
                         score=group['score_mass'] / group['known'] if group['known'] > 0 else None,
                         is_starting_position=board.get('is_starting_position', False),
                         chapter_attribution=board.get('chapter_attribution'),
                         replies=sorted(group['replies'], key=lambda r: (-r['reach'], r['line']))))
    return sorted(rows, key=lambda r: (-r['reach'], r['line'], r['position']))


def exit_reply(stop, parent_route):
    """Linked reply label numbered from the parent board's shared route."""
    try:
        board = chess.Board(stop['parent_position'] + ' 0 1')
        san = board.san(chess.Move.from_uci(stop['move']))
    except (TypeError, ValueError, KeyError):
        return line(stop['line'].split()[-1]) if stop.get('line') else 'other'
    number = plies(parent_route) // 2 + 1
    token = f'{number}.{san}' if board.turn else f'{number}...{san}'
    route = token if plies(parent_route) == 0 else f'{parent_route} {token}'
    return f'[{line(token)}]({analysis_url(stop["position"], route)})'


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
            cell = SourceCell(str(cell), labels[0] + ('<br>Also: ' + '; '.join(labels[1:]) if len(labels) > 1 else ''))
        rows.append([linked_line(route['example']['line'], route['position']), percentage(route['conditional_first_entry_weight']),
                     cell, opponent_rating(original.get('opponent_rating'))])
    return ['**Where this chapter starts**', '',
            'Leading first-entry boards, weighted over every transposed arrival; the example line is one actual route. '
            'Later entry boards can bypass the main defining position.', '',
            *table(['First-entry example', 'Entry-position weight', 'Chapter source', 'Avg opponent rating'], rows),
            f'[All entry positions and routes](#{refs.anchor(chapter["id"])}-entries).', '']


def summary_own_priorities(rows, refs, strongest=False, show_components=True):
    field, label = ('local_gain_pp', 'Gain') if strongest else ('local_drop_pp', 'Drag')
    return table(['Line', 'Chapter source', 'Move reach', 'Repertoire score', 'Score spread', label,
                  label + ' per 1,000 games', 'Avg opponent rating'],
                 [[refs.move_cell(r), refs.sources(r), percentage(r['branch_reach']) + '<br>1 per ' + games_per_encounter(r['branch_reach']) + ' games',
                   percentage(r['move_score']), spread_display(r.get('move_spread'), show_components),
                   delta_points(r['move_score'], r['reference_score'], drag=not strongest)
                   + ('<br>Move ' + score_points(r['database_move_gain_pp'], signed=True) + '; prep '
                      + score_points(r['continuation_gain_pp'], signed=True) if show_components and r.get('database_move_gain_pp') is not None else '')
                   + '<br>95%: ' + interval_cell(r.get('local_gain_interval_pp' if strongest else 'local_drop_interval_pp')),
                   per_thousand(r['branch_reach'] * r[field], signed=strongest),
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
            values.append(['<br>'.join(labels), percentage(row['reach']), percentage(base), percentage(score),
                           delta_points(score, base), spread_display(row.get('branch_score_spread'), False)])
        return table(['Opening family / variation', 'Reach', 'Entry baseline', 'Repertoire score', 'Delta', 'Score spread'], values)
    values = []
    for row in rows:
        base, repertoire = row['entry_baseline']['raw_score'], row['repertoire_score']
        values.append([f"[{escape(row['name'])}](#{anchors[row['id']]})", escape(row['eco']), percentage(row['reach']),
                       percentage(base), percentage(repertoire), score_points(row['difference_pp'], signed=True),
                       spread_display(row.get('branch_score_spread'), False), number(row['expected_prepared_moves']),
                       gap_percentage(row['gap_coverage']), refs.sources({'chapter_attribution': {'source_ids': row['chapter_ids']}})])
    return table(['Opening', 'ECO', 'Reach', 'Entry baseline', 'Repertoire score', 'Delta', 'Score spread',
                  'Prepared own moves after entry', 'Equivalent gap reach after entry', 'Chapters'], values)


def openings_section(bundle, refs):
    """Opening table for the full report; per-opening evidence lives on its own page."""
    data = bundle.get('openings')
    color = bundle['report']['color']
    text = section('### Openings reached', f'{color}-openings')
    if data is None:
        return text + ['Opening analysis pending. Run `repertoire-openings` with the saved score results.', '']
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


def opening_details_page(bundle, refs, warnings=()):
    """One page per color with each opening's first entries and the positions reached through it."""
    data = bundle.get('openings') or {}
    color = bundle['report']['color']
    rows = data.get('openings', [])
    anchors = {row['id']: f'{color}-opening-{i}' for i, row in enumerate(rows, 1)}
    text = [f'# {color.title()} openings: entry positions and evidence', '',
            f'[Summary](@summary) · [Full report](@report) · [{color.title()} openings table](#{color}-openings)', '',
            *[item for warning in warnings for item in (warning, '')],
            'Entry weight combines all first-arrival routes to the exact board; each example is one real first-arrival route. '
            'Games describe entry evidence, not the sample size of the full recursive score; † marks pooled counts whose historical games can overlap. '
            f'Opening groups have no aggregate opponent rating. {about("opening-names")}.', '']
    reached = {row['id']: row for row in rows}
    catalog = {row['id']: row for row in data.get('catalog', [])}
    sparse = bundle['report']['manifest']['sparse_threshold']
    positions = visible_positions(scope_by_id(bundle.get('character')).get('overall', {}))
    for row in rows:
        text += section('## ' + escape(row['name']) + ' (' + escape(row['eco']) + ')', anchors[row['id']])
        parents = [p for p in row['parent_ids'] if p in reached]
        if parents:
            text += ['Broader categories: ' + ', '.join(f"[{escape(catalog[p]['name'])}](#{anchors[p]})" for p in parents) + '.', '']
        values = []
        for entry in row['entries']:
            labels = data['positions'][entry['position']]
            values.append([linked_line(entry['example']['line'], entry['position']),
                refs.sources({'chapter_attribution': {'source_ids': entry['chapter_ids']}}),
                refs.opening_source({'position': entry['position']}),
                'Exact' if labels['exact_name'] else 'Inherited', percentage(entry['conditional_weight']),
                percentage(entry['entry_probability']), percentage(entry['baseline_score']),
                percentage(entry['repertoire_score']), spread_display(entry.get('branch_score_spread')),
                f"{entry['games']:,}" + ('†' if len(entry['origins']) > 1 else '') + (' (sparse)' if entry['games'] < sparse else ''),
                entry['games_source'], opponent_rating(entry.get('opponent_rating')), rating_difference(entry.get('opponent_rating'))])
        text += table(['First-entry example', 'Chapter source / context', 'Most common opening source', 'Name source', 'Entry weight',
                       'Position reach', 'Entry baseline', 'Repertoire score', 'Score spread', 'Entry games', 'Evidence source',
                       'Avg opponent rating', 'Rating Δ vs parent'], values)
        origins = []
        for position in positions:
            origin = data['positions'].get(position['position'], {})
            contribution = origin.get('opening_reach_contributions', {}).get(row['id'], 0.)
            if contribution:
                origins.append((position, contribution, origin['opening_reach_fractions'][row['id']]))
        if origins:
            headers, values = drop_uniform(
                ['Position', 'Chapter source / context', 'Total position reach', 'Reach through this opening', 'Share of arrivals'],
                [[refs.position_cell(position), refs.sources(position), percentage(position['reach']), percentage(contribution), percentage(fraction)]
                 for position, contribution, fraction in sorted(origins, key=lambda item: (-item[1], -item[0]['reach']))[:5]],
                {'Share of arrivals'})
            text += ['**Most common positions reached through this opening**', '', *table(headers, values)]
    return text


def chapter_page(bundle, chapter, refs, warnings, chapter_top, position_top):
    """Everything about one chapter on its own page; scores are conditional on first entry."""
    r = bundle['report']; color = r['color']; cid = chapter['id']
    characters = scope_by_id(bundle.get('character'))
    preparation = scope_by_id(bundle.get('preparation'))
    vulnerabilities = scope_by_id(bundle.get('vulnerabilities'), 'chapters')
    index = refs.entries[cid][0]
    code = f'{color[0].upper()}{index}'
    anchor = refs.anchor(cid)
    s = chapter['score']; base = chapter.get('entry_baseline', {})
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
    contents = [('exits', 'Where preparation ends'), ('common-positions', 'Most common positions'),
                ('vulnerabilities', 'Vulnerabilities'), ('strengths', 'Strengths'), ('gaps', 'Equivalent gap reach'),
                ('prepared-depth', 'Prepared-depth distribution'), ('spread', 'Branch score spread'),
                ('character', 'Preparation, replies, and resulting positions'), ('entries', 'Exact first-entry positions')]
    text = [f'<a id="{anchor}"></a>', '', f'# {code}. {escape(chapter["name"])}', '', ' · '.join(links), '',
            *[item for warning in warnings for item in (warning, '')]]
    if chapter.get('policy_overrides'):
        text += [f"**Alternative comparison.** This chapter prefers its own first moves. Region reach under the overall policy: "
                 f"{percentage(s.get('overall_policy_entry_probability'))}. {about('alternatives')}.", '']
    text += [f"Chapter reach **{percentage(s.get('entry_probability'))}**; repertoire score **{percentage(s.get('raw_empirical_score'))}**; "
             f"entry baseline **{percentage(base.get('raw_score'))}**; delta **{headline_delta(s.get('raw_empirical_score'), base.get('raw_score'))}**; "
             f"prepared depth **{number(s.get('prepared_depth', {}).get('expected_moves'))} own moves**. "
             f"Values on this page are conditional on entering the chapter. {about('entry')}.", '']
    text += leading_entries(chapter, preparation.get(cid), refs)
    ratings = chapter.get('opponent_ratings', {})
    text += [f"Score-evidence opponent rating **{opponent_rating(ratings.get('score_evidence'))}**; "
             f"entry-baseline opponent rating **{opponent_rating(ratings.get('entry_baseline'))}**. "
             f"Rated coverage: continuation {percentage((ratings.get('score_evidence') or {}).get('known_coverage'))}; "
             f"first entry {percentage((ratings.get('entry_baseline') or {}).get('known_coverage'))}.", '']
    if char and char.get('branch_score_spread'):
        text += [f"Branch score spread **{percentage(char['branch_score_spread'].get('standard_deviation'))}**. {about('spread')}.", '']
    text += [f"Equivalent gap reach after entry **{gap_percentage(chapter.get('gap_coverage'))}**; "
             f"weighted gap reach contribution **{gap_percentage(chapter.get('gap_coverage'), weighted=True)}**.", '']
    if char and 'reuse' in char:
        text += [f"{char['reuse']['reachable_distinct_decisions']} distinct own decisions; {number(char['predictability']['effective_replies'])} effective replies; "
                 f"{number(char['position_profiles']['all']['effective_pawn_structures'])} effective boundary pawn structures.", '']
    text += ['On this page: ' + ' · '.join(f'[{title}](#{anchor}-{key})' for key, title in contents), '']
    text += exits_section(char, refs, max(chapter_top, 10), level='##', anchor=f'{anchor}-exits')
    text += common_positions_for_scope(char or {'id': cid}, refs, position_top, level='##', anchor=f'{anchor}-common-positions')
    text += (vulnerabilities_section(vulnerabilities.get(cid), refs, chapter_top, level='##', anchor=f'{anchor}-vulnerabilities')
             or section('## Vulnerabilities', f'{anchor}-vulnerabilities') + ['Vulnerability analysis pending.', ''])
    text += (strengths_section(vulnerabilities.get(cid), char, refs, chapter_top, level='##',
                               sparse_threshold=r['manifest']['sparse_threshold'], anchor=f'{anchor}-strengths')
             or section('## Strengths', f'{anchor}-strengths') + ['Strength analysis pending.', ''])
    text += gap_section(char, level='##', refs=refs, top=chapter_top, anchor=f'{anchor}-gaps')
    text += depth_section(preparation.get(cid), level='##', anchor=f'{anchor}-prepared-depth')
    text += (branch_spread_section(char, level='##', anchor=f'{anchor}-spread')
             or section('## Branch score spread', f'{anchor}-spread') + ['Branch spread pending.', ''])
    text += (character_section(char, refs, chapter_top, level='##', anchor=f'{anchor}-character')
             or section('## Preparation, replies, and resulting positions', f'{anchor}-character') + ['Character analysis pending.', ''])
    transitions = [t for t in r.get('chapter_transitions', []) if t['source_id'] == cid
                   and t.get('conditional_probability') is not None and t['conditional_probability'] > 0]
    if transitions:
        text += ['**Most likely chapter transitions after entry**', '',
                 'Conditional reach of the destination at or after source entry, including shared or simultaneous entry. Rows overlap.', '']
        text += table(['Destination', 'Conditional probability'],
                      [[refs.label(t['destination_id'], True), percentage(t['conditional_probability'])]
                       for t in sorted(transitions, key=lambda t: t['conditional_probability'], reverse=True)[:5]])
    text += entry_routes_section(chapter, preparation.get(cid), refs)
    interval = s.get('posterior', {}).get('credible_interval_95')
    if interval:
        text += ['## Score interval and evidence limits', '',
                 f"Approximate 95% score interval: {' to '.join(map(percentage, interval))}; unresolved probability: {percentage(s.get('unresolved_mass'))}; "
                 f"sparse-score sensitivity: {' to '.join(map(percentage, s.get('sparse_sensitivity', [])))}. {about('evidence')}.", '']
    return text


def full_report(bundles, correlations, correlation_reason, top=10, chapter_top=5, position_top=20,
                rating_correlations=None, rating_correlation_reason='not generated'):
    """The full report index plus one page per chapter and one opening page per color, keyed by relative path."""
    combined = combined_overall(bundles)
    has_combined = combined is not None and 'unavailable' not in combined
    warnings = [row for row in snapshot_notes(bundles) if row.startswith('**')]
    pages = {}
    text = ['# Repertoire report', '',
            'Scores, chapter comparisons, vulnerabilities, position contributions, and repertoire character. '
            'All scores are from the repertoire owner\'s perspective. Each chapter has its own page, linked from the chapter tables; '
            f'line links open the Lichess analysis board. {about("links", "About labels and links")}.', '',
            '<!-- report-navigation -->', '',
            *snapshot_notes(bundles),
            *section('## Combined repertoire' if has_combined else '## Score overview', 'combined' if has_combined else 'overview'),
            *overview(bundles), overview_notes(bundles), '']
    for b in bundles:
        r = b['report']; color = r['color']; refs = bundle_refs(b); o = r['overall']
        characters = scope_by_id(b.get('character'))
        preparation = scope_by_id(b.get('preparation'))
        text += section(f'## {color.title()} repertoire', color)
        text += section(f'### {color.title()} chapters ({len(r["chapters"])})', f'{color}-chapters')
        text += ['Chapter reach is the chance of first reaching any position in a chapter through any move order; the other values are conditional on that entry. '
                 'Each chapter name opens its own page with positions, gaps and entry routes. '
                 f'Chapters overlap, so their values are not additive. {about("entry")}.', '', *chapter_table(r, refs)]
        text += exits_section(characters.get('overall'), refs, top, anchor=f'{color}-exits')
        text += common_positions_section([b], position_top)
        text += openings_section(b, refs)
        text += vulnerabilities_section(b.get('vulnerabilities', {}).get('overall'), refs, top)
        text += strengths_section(b.get('vulnerabilities', {}).get('overall'), characters.get('overall'), refs, top,
                                  sparse_threshold=r['manifest']['sparse_threshold'])
        text += gap_section(characters.get('overall'), refs=refs, top=top)
        text += depth_section(preparation.get('overall'), anchor=f'{color}-prepared-depth')
        text += branch_spread_section(characters.get('overall'))
        text += character_section(characters.get('overall'), refs, min(top, 5))
        text += ['### Score and evidence limits', '']
        interval = o.get('posterior', {}).get('credible_interval_95')
        text += ['<details>', '<summary>Score and evidence limits</summary>', '']
        text += table(['Measure', 'Value'], [
            ['Approximate model-based 95% score interval', ' to '.join(map(percentage, interval)) if interval else 'unavailable'],
            ['Unresolved probability', percentage(o['unresolved_mass'])],
            ['Conditional score bounds', ' to '.join(map(percentage, o['conditional_bounds']))],
            ['Sparse probability', percentage(o['sparse_mass'])],
            ['Sparse-score sensitivity', ' to '.join(map(percentage, o['sparse_sensitivity']))]])
        text += table(['Stopping type', 'Probability'], [[k.replace('_', ' ').capitalize(), percentage(v)] for k, v in o['masses'].items()])
        text += ['</details>', '']
        text += section('### Uncertainty priorities and prior sensitivity', f'{color}-uncertainty-prior-sensitivity')
        text += ['<details>', '<summary>Uncertainty priorities and prior sensitivity</summary>', '',
                 'Priority is posterior mean reach multiplied by the stopping-score interval width, in points per 1,000 games. '
                 'This is a review heuristic, not additive variance attribution.', '']
        text += table(['Stopping route', 'Chapter source / context', 'Priority per 1,000 games', 'Games', 'Avg opponent rating', 'Rating Δ vs parent'],
                      [[linked_line(' '.join(e['representative_path_san']), e.get('position')), refs.sources(e),
                        per_thousand(100 * e['uncertainty_priority']), count(e['sample_count']), opponent_rating(e.get('opponent_rating')),
                        rating_difference(e.get('opponent_rating'))] for e in sorted(r['events'], key=lambda e: e['uncertainty_priority'], reverse=True)[:top]])
        text += table(['Owner W/D/L prior', 'Approximate 95% score interval'],
                      [[str(p['prior']), ' to '.join(map(percentage, p['overall']['posterior']['credible_interval_95']))]
                       for p in r.get('prior_sensitivity', [])])
        text += ['</details>', '']
        for chapter in r['chapters']:
            pages[f'chapters/{refs.page(chapter["id"])}'] = chapter_page(b, chapter, refs, warnings, chapter_top, position_top)
        if b.get('openings') is not None:
            pages[f'openings/{color}.md'] = opening_details_page(b, refs, warnings)
    text += section('## Correlations', 'correlations')
    text += ['Both main analyses are reach-weighted: each selected move or opponent reply is weighted by its probability of being encountered under the repertoire policy. '
             'Unweighted sensitivity checks are labeled explicitly.', '']
    text += correlations_section(correlations, correlation_reason)
    text += rating_correlations_section(rating_correlations, rating_correlation_reason) + methods(bundles)
    pages = {'report.md': '\n'.join(report_navigation(text)) + '\n',
             **{name: '\n'.join(row.rstrip() for row in lines) + '\n' for name, lines in pages.items()}}
    return pages


def summary_report(bundles, full_path, correlations, correlation_reason):
    """Headline, then what to work on: where preparation ends and own moves to review; the rest collapses."""
    with display(digits=1, compact_counts=True):
        return _summary_report(bundles)


def _summary_report(bundles):
    snapshot = snapshot_notes(bundles)
    warnings = [row for row in snapshot if row.startswith('**')]
    metadata = [row for row in snapshot if row and not row.startswith('**')]
    text = ['# Repertoire summary', '', '[Complete report](@report) · [Definitions](#methods)', '',
            *[item for warning in warnings for item in (warning, '')],
            *overview(bundles, compact=True, focused=True),
            'Scores count draws as half a point. Delta is the repertoire score minus the starting baseline; its CP and Elo equivalents '
            'are score-scale translations, not engine evaluations or rating forecasts. Combined weights White and Black equally (50% each). '
            f'{about("score", "Definitions and evidence")}.', '',
            'Each color starts with where preparation ends and the own moves most worth reviewing. '
            'Line links open the Lichess analysis board at the position where you choose a move; chapter links open each chapter page.', '']
    for b in bundles:
        r = b['report']; color = r['color']; refs = bundle_refs(b)
        refs.compact = True
        char = scope_by_id(b.get('character')).get('overall', {})
        moves = b.get('vulnerabilities', {}).get('overall', {})
        prep = scope_by_id(b.get('preparation')).get('overall', {}).get('depth_distribution', {})
        depth = r['overall'].get('prepared_depth', {}).get('expected_moves')
        median = prep.get('median_moves')
        text += [f'## {color.title()} repertoire', '',
                 f"Preparation lasts **{number(depth)} own moves** on average"
                 + (f' (median {median})' if median is not None else '')
                 + f". Equivalent gap reach **{gap_percentage(char.get('gap_coverage'))}** "
                 f"([recurring gaps](#{color}-equivalent-gap-reach)).", '']
        text += exits_section(char, refs, 5)
        text += [f'[More exit points](#{color}-exits) · [All positions and gaps](#{color}-common-positions)', '']
        own_bad = own_priorities(moves.get('rankings', {}).get('own', []))
        own_good = own_priorities(moves.get('strengths', []), strongest=True)
        replies = [x for x in non_sparse_rows(moves.get('rankings', {}).get('opponent', [])) if not x['prepared']][:5]
        if own_bad:
            text += ['### Own moves to review', '',
                     'Ranked by move reach × drag against the parent database score. These are comparison deficits, not promised gains; '
                     f'sparse comparisons are omitted. {about("drag")}.', '',
                     *summary_own_priorities(own_bad[:5], refs, show_components=False),
                     f'[All vulnerabilities and prepared replies](#{color}-vulnerabilities).', '']
        positions = [p for p in visible_positions(char) if not unanswered_position(p, color)]
        if positions:
            text += ['<details>', '<summary>Most common positions</summary>', '',
                     'Prepared positions by reach, nested by move order; each line shows the moves since the position above it. '
                     f'Reach includes transpositions and nested positions overlap. {about("position-reach")}.', '',
                     *position_tree(char, refs),
                     f'Showing {min(12, len(positions))} of {len(positions):,} prepared positions. [All positions and gaps](#{color}-common-positions).', '',
                     '</details>', '']
        text += ['<details>', f'<summary>Chapter comparisons ({len(r["chapters"])} chapter{"" if len(r["chapters"]) == 1 else "s"})</summary>', '',
                 'Chapter scores and gap reach are conditional on first entry through any move order; overlapping chapters are not additive. '
                 f'{about("entry")}.', '',
                 *chapter_table(r, refs), '</details>', '']
        if replies:
            text += ['<details>', '<summary>Costly unprepared replies</summary>', '',
                     'Opponent replies without a prepared answer, ranked by drag per 1,000 games: the score drop from the repertoire value before the reply. '
                     f'{about("drag")}.', '',
                     *table(['Line', 'Chapter context', 'Move reach', 'Scores before / after', 'Score spread before reply',
                             'Drag', 'Drag per 1,000 games', 'Reply games / Avg opponent rating'],
                            [[refs.move_cell(x), refs.sources(x), percentage(x['branch_reach'])
                              + '<br>1 per ' + games_per_encounter(x['branch_reach']) + ' games',
                              percentage(x['reference_score']) + '<br>' + percentage(x['move_score']),
                              spread_display(x.get('reference_spread'), False),
                              delta_points(x['move_score'], x['reference_score'], drag=True)
                              + '<br>95%: ' + interval_cell(x.get('local_drop_interval_pp')),
                              per_thousand(x['weighted_drag_pp']), f"{count(x['sample_count'])} games<br>Rating "
                              + opponent_rating(x.get('opponent_rating')) + '; Δ ' + rating_difference(x.get('opponent_rating'))]
                             for x in replies]),
                     f'[All vulnerabilities](#{color}-vulnerabilities).', '', '</details>', '']
        if own_good:
            text += ['<details>', '<summary>Strongest moves</summary>', '',
                     'Ranked by move reach × gain against the parent database score. Move gain and later preparation gain appear separately. '
                     'These overlapping comparisons are not additive.', '',
                     *summary_own_priorities(own_good[:5], refs, strongest=True),
                     f'[All strengths and position contributions](#{color}-strengths).', '', '</details>', '']
        metrics = [['Expected prepared depth', number(depth) + ' own moves']]
        if median is not None:
            metrics.append(['Median prepared depth', f'{median} own moves'])
        if 'reuse' in char:
            reuse, pred = char['reuse'], char['predictability']
            curve = next((x for x in reuse['curve'] if x['games'] == 100), None)
            metrics.append(['Distinct own decisions', f"{reuse['reachable_distinct_decisions']:,}"])
            if curve:
                metrics.append(['Expected distinct decisions after 100 modeled games', number(curve['expected_distinct_decisions'], 0)])
            metrics.append(['Effective opponent replies', number(pred['effective_replies'])])
        metrics.append(['Branch score spread', percentage((char.get('branch_score_spread') or {}).get('standard_deviation'))])
        detail_links = [f'[Prepared-depth distribution](#{color}-prepared-depth)']
        if 'reuse' in char:
            detail_links.append(f'[Preparation reuse and reply variety](#{color}-preparation-replies-and-resulting-positions)')
        if char.get('branch_score_spread'):
            detail_links.append(f'[Branch spread and outcome volatility](#{color}-branch-score-spread)')
        text += ['<details>', '<summary>Preparation and variability</summary>', '',
                 *table(['Measure', 'Value'], metrics),
                 ' | '.join(detail_links) + '.', '', '</details>', '']
    text += ['<details>', '<summary>Evidence and definitions</summary>', '',
             *evidence_snapshot(bundles),
             '**Data snapshot**', '', *[item for description in metadata for item in (description, '')],
             '**Metric definitions**', '',
             'Games leaving prep here is the share of games whose first unprepared position follows that prepared position; those rows do not overlap. '
             f'{about("exits", "Exit points")}.', '',
             'Equivalent gap reach is the reach of one gap that would produce the same repeat probability as all first gaps combined. '
             f'Lower values indicate less concentrated recurring gaps. {about("gap-reach", "Definition and chapter weighting")}.', '',
             'Positive gain and delta are favorable; positive drag is a deficit. Per 1,000 games values multiply a comparison by its reach. '
             'Avg games per encounter is 1 / reach for games with that color. Line comparisons overlap and cannot be added. '
             'Ratings describe local opponents and do not adjust scores. Local 95% bands are approximate prior-completed model intervals, not causal gain intervals.', '']
    if any(c.get('policy_overrides') for b in bundles for c in b['report']['chapters']):
        text += ['**Alternative** chapters prefer their own first moves; overall uses the earliest chapter and first PGN choice.', '']
    text += ['</details>', '', 'Chapter entry explanations, strengths, vulnerabilities and evidence details: [complete report](@report).', '']
    return '\n'.join(row.rstrip() for row in text)


def resolve_links(pages, aliases):
    """Point in-page anchors and @page placeholders at whichever generated file holds them."""
    owners, homes = {}, {}
    for path, text in pages.items():
        found = re.findall(r'<a id="([^"]+)"></a>', text)
        for anchor in found:
            owners.setdefault(anchor, path)
        if path.parent.name == 'chapters' and found:
            homes[path] = found[0]
    def relative(target, source):
        return Path(os.path.relpath(target, source.parent)).as_posix()
    resolved = {}
    for path, text in pages.items():
        def anchor_link(match):
            anchor = match[1]
            target = owners.get(anchor)
            if target is None or target == path:
                return match[0]
            return f']({relative(target, path)})' if homes.get(target) == anchor else f']({relative(target, path)}#{anchor})'
        def page_link(match):
            target = aliases.get(match[1])
            return match[0] if target is None else f']({relative(target, path)}{match[2] or ""})'
        text = re.sub(r'\]\(#([^)\s]+)\)', anchor_link, text)
        resolved[path] = re.sub(r'\]\(@([\w/.-]+)(#[^)\s]*)?\)', page_link, text)
    return resolved


def generate(paths, full_path=None, summary_path=None, *, strict=True, require_complete=False, top=10, chapter_top=5, position_top=20):
    if not paths or min(top, chapter_top, position_top) < 1:
        raise ValueError('Supply reports and positive table lengths')
    bundles = load(paths, strict, require_complete)
    correlations, reason = load_correlations(bundles, strict, require_complete)
    rating_correlations, rating_reason = load_rating_correlations(bundles, strict)
    full_path = Path(full_path) if full_path else report_directory(bundles[0]['path']) / 'report.md'
    summary_path = Path(summary_path) if summary_path else full_path.parent / 'summary.md'
    if full_path.resolve() == summary_path.resolve():
        raise ValueError('Report and summary must be different files')
    if any(p.suffix.lower() != '.md' for p in (full_path, summary_path)):
        raise ValueError('Report and summary destinations must be Markdown (.md) files')
    rendered = full_report(bundles, correlations, reason, top, chapter_top, position_top, rating_correlations, rating_reason)
    folder = full_path.parent
    pages = {full_path: rendered.pop('report.md'), summary_path: summary_report(bundles, full_path, correlations, reason) + '\n',
             **{folder / name: content for name, content in rendered.items()}}
    aliases = {'report': full_path, 'summary': summary_path,
               **{name[:-3]: folder / name for name in rendered}}
    pages[full_path] = pages[full_path].replace('](summary.md)', '](@summary)')
    # Use relative paths even when callers place the outputs in different folders.
    for b in bundles:
        for data in [b['path']] + [b['path'].with_suffix(f'.{family}.json') for family in FAMILIES if family in b]:
            pages[full_path] = pages[full_path].replace(f']({data.name})', f']({Path(os.path.relpath(data, folder)).as_posix()})')
    pages = resolve_links(pages, aliases)
    for target, content in pages.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text(target, content)
    # Remove pages for chapters or colors that no longer exist; only generator-owned names are touched.
    for directory, pattern in ((folder / 'chapters', CHAPTER_PAGE), (folder / 'openings', re.compile(r'(white|black)\.md'))):
        if directory.is_dir():
            for stale in directory.iterdir():
                if pattern.fullmatch(stale.name) and stale not in pages:
                    stale.unlink()
    return bundles


def page_names(report):
    """Relative chapter and opening page paths that a render of this score result writes."""
    color = report['color']
    return [f'chapters/{color[0].upper()}{i}.md' for i in range(1, len(report['chapters']) + 1)] + [f'openings/{color}.md']
