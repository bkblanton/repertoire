"""Lichess cloud evaluations for the repertoire's positions, kept in a local store.

The store is filled only from the downloaded database export (https://database.lichess.org/#evals): hundreds of
millions of positions, one JSON line each, in no useful order. It cannot be searched for one position, so `import`
reads the whole file once and keeps every position the reports use and every position one move beyond them. The
store remembers which positions it looked for in each export file, so a rerun scans again only for new positions.
Positions the export does not have stay without an evaluation; nothing is requested or estimated in their place.

Evaluations are stored as published: from White's side, in centipawns or moves to mate, keeping the deepest
search for each position. Nothing here changes a score.
"""

import argparse
import json
import queue
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from datetime import UTC, datetime
from itertools import compress
from pathlib import Path
from typing import Any, TypedDict

import chess
import zstandard

from .board_cache import canonical, children
from .board_cache import position_of as node_position
from .context import DEFAULT_CACHE, AnalysisContext
from .evaluate import by_position, reaches
from .explorer import Progress, duration
from .graph import resolve, topology
from .model import empirical, prepare
from .schema import Position

DIRECTORY = Path('.cache/evals')
STORE = DIRECTORY / 'evals.sqlite'
EXPORT = DIRECTORY / 'lichess_db_eval.jsonl.zst'
EXPORT_URL = 'https://database.lichess.org/lichess_db_eval.jsonl.zst'
REPORTS = ('reports/data/white.json', 'reports/data/black.json')
# Decompressed bytes handed from the reading thread to the matching loop at a time.
CHUNK = 1 << 26
# Pieces, side to move and castling rights of each export line. The en passant field is left out of the match
# and settled with python-chess, since the export may write a square that no pawn can capture on.
LINE_KEY = re.compile(rb'\{"fen":"([^ "]+ [wb] [^ "]+) ')


class Evaluation(TypedDict):
    depth: int
    knodes: int
    pvs: list[dict[str, Any]]  # each with 'cp' or 'mate' (White's side) and 'line' (UCI moves)
    source: str  # 'export'
    retrieved_at: str  # when the export file was saved


SCHEMA = '''
create table if not exists evals (
    position text primary key, depth integer not null, knodes integer not null, pvs text not null,
    source text not null, retrieved_at text not null
);
create table if not exists exports (
    id integer primary key, name text not null, size integer not null, modified text not null,
    scanned_at text, unique (name, size, modified)
);
create table if not exists searched (
    export integer not null, position text not null, primary key (export, position)
) without rowid;
'''


def deeper(a: Evaluation, b: Evaluation | None) -> bool:
    """Whether `a` comes from a deeper search than `b`; Lichess recommends the deepest evaluation."""
    return b is None or (a['depth'], a['knodes']) > (b['depth'], b['knodes'])


class Store:
    """Evaluations by position and the positions each export file was searched for.

    One SQLite file rather than a file per position: an import adds tens of thousands of positions at once."""

    def __init__(self, path: str | Path = STORE, read_only: bool = False) -> None:
        if read_only:
            # Analyses only read the store, and must not touch the file the build hashes.
            self.db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def __enter__(self) -> 'Store':
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self.db.in_transaction:
            self.db.commit()
        self.db.close()

    def commit(self) -> None:
        self.db.commit()

    def get(self, position: Position) -> Evaluation | None:
        row = self.db.execute(
            'select depth, knodes, pvs, source, retrieved_at from evals where position = ?', (position,)
        ).fetchone()
        if row is None:
            return None
        depth, knodes, pvs, source, retrieved_at = row
        return Evaluation(depth=depth, knodes=knodes, pvs=json.loads(pvs), source=source, retrieved_at=retrieved_at)

    def read(self, positions: Iterable[Position]) -> dict[Position, Evaluation]:
        return {k: e for k in positions if (e := self.get(k)) is not None}

    def positions(self) -> set[Position]:
        return {k for (k,) in self.db.execute('select position from evals')}

    def put(self, position: Position, evaluation: Evaluation) -> bool:
        """Keep `evaluation` if it is deeper than the stored one; returns whether it was kept."""
        if not deeper(evaluation, self.get(position)):
            return False
        self.db.execute(
            'insert or replace into evals values (?, ?, ?, ?, ?, ?)',
            (
                position,
                evaluation['depth'],
                evaluation['knodes'],
                json.dumps(evaluation['pvs'], separators=(',', ':')),
                evaluation['source'],
                evaluation['retrieved_at'],
            ),
        )
        return True

    def export(self, path: Path) -> int:
        """The id of an export file, identified by name, size and modification time, so a moved copy matches."""
        stat = path.stat()
        identity = (path.name, stat.st_size, modified(path))
        self.db.execute('insert or ignore into exports (name, size, modified) values (?, ?, ?)', identity)
        row = self.db.execute('select id from exports where name = ? and size = ? and modified = ?', identity)
        return int(row.fetchone()[0])

    def searched(self, export: int) -> set[Position]:
        return {k for (k,) in self.db.execute('select position from searched where export = ?', (export,))}

    def mark_searched(self, export: int, positions: Iterable[Position]) -> None:
        self.db.executemany('insert or ignore into searched values (?, ?)', ((export, k) for k in positions))
        self.db.execute('update exports set scanned_at = ? where id = ?', (now(), export))


def now() -> str:
    return datetime.now(UTC).isoformat()


def modified(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()


def from_export(line: Mapping[str, Any], retrieved_at: str) -> Evaluation:
    best = max(line['evals'], key=lambda e: (e['depth'], e.get('knodes', 0)))
    return Evaluation(
        depth=best['depth'], knodes=best.get('knodes', 0), pvs=best['pvs'], source='export', retrieved_at=retrieved_at
    )


def position_of(fen: str) -> Position:
    return canonical(chess.Board(fen + ' 0 1'))


def blocks(path: Path, size: int, done: Callable[[int], None]) -> Iterator[bytes]:
    """Decompressed blocks of the export, read on a second thread so that decompression overlaps matching.

    `done` receives the compressed bytes read so far after each block."""
    handoff: queue.Queue[bytes | BaseException | None] = queue.Queue(maxsize=4)

    def read() -> None:
        try:
            with path.open('rb') as raw:
                reader = zstandard.ZstdDecompressor().stream_reader(raw, read_size=1 << 22)
                while block := reader.read(size):
                    handoff.put(block)
                    done(raw.tell())
            handoff.put(None)
        except BaseException as exc:  # noqa: BLE001 - re-raised by the consumer
            handoff.put(exc)

    threading.Thread(target=read, daemon=True).start()
    while (item := handoff.get()) is not None:
        if isinstance(item, BaseException):
            raise item
        yield item


def wanted_lines(data: bytes, end: int, prefixes: set[bytes]) -> Iterator[bytes]:
    """The complete lines in data[:end] whose pieces, side and castling are in `prefixes`.

    The keys are extracted by one regular-expression pass and the lines picked out without a Python loop, since
    popular positions match in nearly every block."""
    keys = LINE_KEY.findall(data, 0, end)
    hits = prefixes.intersection(keys)
    if not hits:
        return
    lines = data[:end].split(b'\n')
    if lines and not lines[-1]:
        lines.pop()
    if len(lines) == len(keys):
        yield from compress(lines, map(hits.__contains__, keys))
        return
    # A line in an unexpected format: match each line on its own.
    for line in lines:
        match = LINE_KEY.match(line)
        if match and match.group(1) in hits:
            yield line


def scan(
    path: Path, wanted: Collection[Position], size: int = CHUNK, label: str = 'Evaluation export'
) -> Iterator[tuple[Position, Evaluation]]:
    """Every evaluation in the export for a position in `wanted`, in file order. A position can appear more
    than once, with different en passant fields."""
    prefixes = {k.rsplit(' ', 1)[0].encode() for k in wanted}
    retrieved_at = modified(path)
    total = path.stat().st_size
    progress = dict(read=0, found=0)
    started = printed = time.monotonic()

    def report(final: bool = False) -> None:
        share = progress['read'] / total if total else 1.0
        elapsed = time.monotonic() - started
        left = f', ~{duration(elapsed * (1 - share) / share)} left' if 0 < share < 1 and not final else ''
        print(f'{label}: {100 * share:.0f}% read, {progress["found"]:,} positions found{left}', flush=True)

    def complete_lines() -> Iterator[tuple[bytes, int]]:
        """Each block with the end of its last complete line; the partial line is carried into the next."""
        tail = b''
        for block in blocks(path, size, lambda n: progress.__setitem__('read', n)):
            data = tail + block
            end = data.rfind(b'\n') + 1
            tail = data[end:]
            yield data, end
        yield tail, len(tail)

    for data, end in complete_lines():
        for line in wanted_lines(data, end, prefixes):
            item = json.loads(line)
            if (k := position_of(item['fen'])) in wanted:
                progress['found'] += 1
                yield k, from_export(item, retrieved_at)
        if time.monotonic() - printed >= Progress.INTERVAL:
            printed = time.monotonic()
            report()
    report(final=True)


def import_export(store: Store, path: Path, wanted: Collection[Position], size: int = CHUNK) -> int:
    """Search the export for the wanted positions it has not been searched for; returns how many were added."""
    export = store.export(path)
    pending = set(wanted) - store.searched(export)
    if not pending:
        print(f'Evaluation export: all {len(wanted):,} positions were already searched in {path.name}', flush=True)
        return 0
    print(f'Evaluation export: searching {path.name} for {len(pending):,} positions', flush=True)
    added = set()
    for k, evaluation in scan(path, pending, size):
        if store.put(k, evaluation):
            added.add(k)
    store.mark_searched(export, pending)
    store.commit()
    return len(added)


def frequencies(analysis: AnalysisContext, cache: str | Path) -> tuple[dict[Position, float], dict[Position, float]]:
    """How often games reach each position of the overall repertoire, and how often preparation ends at each
    position: after an unprepared reply, or where you have no prepared move. Finished games are left out."""
    graph, color, manifest, roots = analysis.graph, analysis.color, analysis.manifest, analysis.roots
    transitions = resolve(graph, color, analysis.policy)
    order = topology(transitions, list(roots))
    evidence = analysis.read_evidence(cache, list(manifest['evidence']), required=True)
    model = prepare(graph, transitions, order, color, evidence)
    sampled = empirical(model, color)
    reach = reaches(model, order, sampled, roots)
    exits: dict[Position, float] = {}
    for node_key in order:
        node, r, k = model[node_key], reach[node_key], node_position(node_key)
        if r <= 0:
            continue
        if node.mode == 'opponent':
            moves = children(k)
            for b, (p, _) in zip(node.branches, sampled[node_key]):
                if b.kind == 'deviation' and b.move is not None and b.fixed_score is None and p > 0:
                    exits[moves[b.move]] = exits.get(moves[b.move], 0.0) + r * p
        elif node.mode == 'stop' and node.branches[0].kind == 'theory_leaf' and node.branches[0].fixed_score is None:
            exits[k] = exits.get(k, 0.0) + r
    return {k: r for k, r in by_position(reach).items() if r > 0}, exits


def successors(position: Position) -> list[Position]:
    """Positions one legal move away, without filling the analysis caches with thousands of boards."""
    board = chess.Board(position + ' 0 1')
    result = []
    for move in board.legal_moves:
        board.push(move)
        result.append(canonical(board))
        board.pop()
    return result


def neighborhood(positions: Iterable[Position], plies: int) -> set[Position]:
    """`positions` and every position up to `plies` moves beyond them."""
    result = set(positions)
    frontier = set(result)
    for _ in range(plies):
        frontier = {c for k in frontier for c in successors(k)} - result
        result |= frontier
    return result


def coverage(store: Store, label: str, reach: Mapping[Position, float], exits: Mapping[Position, float]) -> None:
    have = store.positions()
    covered = sum(p for k, p in exits.items() if k in have)
    total = sum(exits.values())
    share = f'{100 * covered / total:.1f}%' if total else 'n/a'
    print(
        f'{label}: evaluations for {sum(k in have for k in exits):,} of {len(exits):,} positions where '
        f'preparation ends ({share} of those games) and {sum(k in have for k in reach):,} of {len(reach):,} '
        'positions games reach',
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True, metavar='<command>')
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('reports', nargs='*', type=Path, help=f'Saved score JSON files (default: {" ".join(REPORTS)})')
    common.add_argument('--store', type=Path, default=STORE, help=f'Evaluation store (default: {STORE})')
    common.add_argument('--cache', default=DEFAULT_CACHE, help='Explorer cache the scores used')
    importing = commands.add_parser(
        'import', parents=[common], help='Search the Lichess evaluation export for the positions the reports use'
    )
    importing.add_argument('--export', type=Path, default=EXPORT, help=f'The export file (default: {EXPORT})')
    importing.add_argument(
        '--plies', type=int, default=1, help='Also keep positions up to this many moves beyond the repertoire'
    )
    commands.add_parser('status', parents=[common], help='Show how many of the positions have evaluations')
    args = parser.parse_args()
    paths = args.reports or [Path(p) for p in REPORTS]
    if args.command == 'import' and not args.export.exists():
        parser.exit(
            1,
            f'No evaluation export at {args.export}. Download {EXPORT_URL} (about 21 GB) there, or pass --export.\n',
        )
    try:
        analyses = [AnalysisContext(path) for path in paths]
        counts = [frequencies(analysis, args.cache) for analysis in analyses]
        with Store(args.store) as store:
            if args.command == 'import':
                # Every position where preparation ends is one move from the repertoire, so it is wanted even
                # with --plies 0.
                wanted = neighborhood((k for analysis in analyses for k in analysis.graph.nodes), args.plies)
                wanted |= {k for _, exits in counts for k in exits}
                added = import_export(store, args.export, wanted)
                print(f'Evaluation export: {added:,} evaluations added or deepened', flush=True)
            for analysis, (reach, exits) in zip(analyses, counts):
                coverage(store, analysis.saved['color'], reach, exits)
    except (ValueError, FileNotFoundError, zstandard.ZstdError) as exc:
        parser.exit(1, f'Evaluations failed: {exc}\nEvaluations found so far are kept.\n')


if __name__ == '__main__':
    main()
