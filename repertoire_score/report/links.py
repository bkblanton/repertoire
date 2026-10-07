"""Board labels, Lichess analysis links and chapter attribution cells."""
import chess

from .bundle import scope_by_id
from .format import escape, line, percentage, plies
from .markdown import SourceCell


LICHESS_ANALYSIS = 'https://lichess.org/analysis/standard/'


def analysis_url(position, route=''):
    """Open an exact canonical board on the Lichess analysis board and opening explorer."""
    fields = position.split()[:4]
    return LICHESS_ANALYSIS + '_'.join(fields + ['0', str(plies(route) // 2 + 1)])


def linked_line(route, position):
    text = line(route)
    return f'[{text}]({analysis_url(position, route)})' if position else text


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


def board_routes(bundle):
    """Overall-policy route per exact board, shared by every table that labels that board."""
    scope = scope_by_id(bundle.get('character')).get('overall', {})
    return {r['position']: r['line'] for r in scope.get('positions', [])}


def bundle_refs(bundle):
    return Chapters(bundle['report'], bundle.get('openings'), board_routes(bundle))


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
