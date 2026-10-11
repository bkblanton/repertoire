"""PGN parsing, position identity, policy resolution and chapter entries."""

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import chess
import chess.pgn

from .board_cache import canonical as key
from .board_cache import children, terminal_white, turn
from .schema import JsonObject, Position

# node -> move -> (target, policy weight); the weight is None at opponent turns. After `unroll`, nodes inside
# repetition loops also carry the loop history, and a target of None is a draw by threefold repetition.
Transitions = dict[Position, dict[str, tuple[Position | None, float | None]]]
# A loop unrolls into at most this many nodes; real repertoires need a handful.
MAX_LOOP_NODES = 100_000
# An explicit move, or move weights summing to one, at own-turn positions.
Policy = Mapping[Position, str | Mapping[str, float]]


class GraphChapter(TypedDict):
    id: str
    name: str
    url: str | None
    root: Position
    mainline: list[Position]


class InferredEntries(TypedDict):
    positions: list[Position]
    paths: list[list[str]]
    status: str


@dataclass
class Node:
    fen: str
    path: list[str]
    chapters: set[str] = field(default_factory=set)
    edges: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, set[str]] = field(default_factory=dict)
    chapter_moves: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Graph:
    nodes: dict[str, Node]
    chapters: list[GraphChapter]
    roots: list[str]


def read_games(path: str | Path) -> list[chess.pgn.Game]:
    """Every PGN game in a file, in order. The file is closed before any game is checked."""
    games = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        while (game := chess.pgn.read_game(stream)) is not None:
            games.append(game)
    return games


def parse(path: str | Path, exclusions: Collection[str] = ()) -> Graph:
    return parse_games(read_games(path), exclusions)


def parse_games(games: Iterable[chess.pgn.Game], exclusions: Collection[str] = ()) -> Graph:
    """The repertoire graph of PGN games, each one a chapter, in priority order."""
    nodes: dict[Position, Node] = {}
    chapters: list[GraphChapter] = []
    roots: list[Position] = []
    for game in games:
        if game.errors:
            raise ValueError(f"Malformed chapter: {game.headers}: {game.errors}")
        if game.headers.get("Variant", "Standard") not in ("Standard", "Chess"):
            raise ValueError("Only standard chess is supported")
        board = game.board()
        if not board.is_valid():
            raise ValueError(f"Invalid root position: {board.fen()}")
        cid = game.headers.get("ChapterURL", "").rstrip("/").split("/")[-1] or str(len(chapters) + 1)
        if cid in exclusions:
            continue
        if any(c["id"] == cid for c in chapters):
            cid += f"-copy-{len(chapters) + 1}"
        root = key(board)
        roots.append(root)
        mainline = [root]
        mb = board.copy()
        for move in game.mainline_moves():
            mb.push(move)
            mainline.append(key(mb))
        chapters.append(
            {
                "id": cid,
                "name": game.headers.get("ChapterName", game.headers.get("Event", cid)),
                "url": game.headers.get("ChapterURL"),
                "root": root,
                "mainline": mainline,
            }
        )
        stack: list[tuple[chess.pgn.GameNode, chess.Board, list[str]]] = [(game, board, [])]
        while stack:
            pgn, b, moves = stack.pop()
            k = key(b)
            node = nodes.setdefault(k, Node(b.fen(en_passant="legal"), moves))
            node.chapters.add(cid)
            pending = []
            for child in pgn.variations:
                if child.move not in b.legal_moves:
                    raise ValueError(f"Illegal move in {cid}: {child.move}")
                uci = child.move.uci()
                if f"{cid}:{k}:{uci}" in exclusions:
                    continue
                after = b.copy()
                san = b.san(child.move)
                after.push(child.move)
                node.edges[uci] = key(after)
                node.provenance.setdefault(uci, set()).add(cid)
                recorded = node.chapter_moves.setdefault(cid, [])
                if uci not in recorded:
                    recorded.append(uci)
                pending.append((child, after, moves + [san]))
            stack.extend(reversed(pending))
    if not chapters:
        raise ValueError("No included PGN chapters")
    return Graph(nodes, chapters, list(dict.fromkeys(roots)))


def conflicts(graph: Graph, color: bool) -> list[JsonObject]:
    return [
        {"position": k, "path": n.path, "choices": {m: sorted(n.provenance[m]) for m in n.edges}}
        for k, n in graph.nodes.items()
        if turn(k) == color and len(n.edges) > 1
    ]


def resolve(graph: Graph, color: bool, policy: Policy) -> Transitions:
    """Explicit overrides, otherwise first PGN move in first chapter order, with repetition loops unrolled.

    The scorer passes the winners of competing chapter alternatives as part of `policy`.
    """
    return unroll(policy_transitions(graph, color, policy))


def policy_transitions(graph: Graph, color: bool, policy: Policy) -> Transitions:
    """The moves `resolve` plays, between positions; a line can still return to an earlier position."""
    transitions: Transitions = {}
    for k, n in graph.nodes.items():
        if terminal_white(k) is not None:
            transitions[k] = {}
        elif turn(k) == color:
            if not n.edges:
                transitions[k] = {}
                continue
            chosen = policy.get(k)
            if chosen is None:
                chosen = next(iter(n.edges))
            weights = {chosen: 1.0} if isinstance(chosen, str) else chosen
            if (
                not weights
                or any(m not in n.edges or not 0 <= w <= 1 for m, w in weights.items())
                or abs(sum(weights.values()) - 1) > 1e-10
            ):
                raise ValueError(f"Invalid policy at {k}: {weights}")
            transitions[k] = {m: (n.edges[m], w) for m, w in weights.items() if w > 0}
        else:
            # Even the last prepared own move has cached opponent replies.
            # An immediate reply into any known board resumes preparation.
            transitions[k] = {m: (target, None) for m, target in n.edges.items()}
            for move, target in children(k).items():
                if target in graph.nodes:
                    transitions[k][move] = (target, None)
    return transitions


def chapter_alternatives(graph: Graph, color: bool, policy: Policy) -> dict[Position, list[str]]:
    """Own-turn boards where chapters record different first moves: each distinct move, in chapter order.

    These compete on score (see evaluate.select_alternatives). Side variations within one chapter do not
    compete, and an explicit policy override settles its board.
    """
    result = {}
    for k, n in graph.nodes.items():
        if turn(k) != color or k in policy or terminal_white(k) is not None:
            continue
        moves = list(dict.fromkeys(recorded[0] for recorded in n.chapter_moves.values() if recorded))
        if len(moves) > 1:
            result[k] = moves
    return result


def alternative_transitions(graph: Graph, color: bool, policy: Policy) -> Transitions:
    """The selected policy, plus every chapter's first move at each own-turn board.

    Every board any chapter alternative or chapter comparison policy can reach is reachable here, so evidence
    can be planned before scoring decides between alternatives. Own-turn boards may have several moves.
    """
    transitions = policy_transitions(graph, color, policy)
    for k, n in graph.nodes.items():
        if turn(k) == color and terminal_white(k) is None:
            for recorded in n.chapter_moves.values():
                if recorded and recorded[0] not in transitions[k]:
                    transitions[k][recorded[0]] = (n.edges[recorded[0]], None)
    return transitions


def reachable(transitions: Transitions, roots: Iterable[Position]) -> list[Position]:
    """Every board reachable from the roots, in discovery order; unlike topology, cycles are allowed."""
    seen: dict[Position, None] = {}
    pending = list(roots)
    while pending:
        k = pending.pop()
        if k in seen:
            continue
        seen[k] = None
        pending.extend(target for target, _ in transitions[k].values() if target is not None)
    return list(seen)


def chapter_policy_overrides(
    graph: Graph, color: bool, global_transitions: Transitions, chapter_id: str
) -> dict[Position, str]:
    """Prefer this chapter's first recorded own move; use the global policy elsewhere.

    Keep chapter-local ordering separately: filtering global edge order would
    incorrectly inherit an earlier chapter's variation order.
    """
    overrides = {}
    for k, node in graph.nodes.items():
        moves = node.chapter_moves.get(chapter_id, [])
        if moves and turn(k) == color:
            chosen = moves[0]
            # Compare moves and weights only: inside a repetition loop the target is a history node.
            if {m: w for m, (_, w) in global_transitions[k].items()} != {chosen: 1.0}:
                overrides[k] = chosen
    return overrides


def topology(transitions: Transitions, roots: Iterable[Position]) -> list[Position]:
    order: list[Position] = []
    visited: set[Position] = set()
    active: list[Position] = []

    def visit(k: Position) -> None:
        if k in active:
            raise ValueError(
                f"Reachable cycle; change policy or exclude the repeated line: {active[active.index(k) :] + [k]}"
            )
        if k in visited:
            return
        active.append(k)
        for target, _ in transitions[k].values():
            if target is not None:
                visit(target)
        active.pop()
        visited.add(k)
        order.append(k)

    for root in roots:
        visit(root)
    return order


def loops(transitions: Transitions) -> list[list[Position]]:
    """Every set of positions a game can move between and return to (strongly connected), each one sorted."""
    index: dict[Position, int] = {}
    low: dict[Position, int] = {}
    stack: list[Position] = []
    on_stack: set[Position] = set()
    found = []
    for start in transitions:
        if start in index:
            continue
        index[start] = low[start] = len(index)
        stack.append(start)
        on_stack.add(start)
        pending = [(start, iter(transitions[start].values()))]
        while pending:
            k, targets = pending[-1]
            for target, _ in targets:
                if target is None:
                    continue
                if target not in index:
                    index[target] = low[target] = len(index)
                    stack.append(target)
                    on_stack.add(target)
                    pending.append((target, iter(transitions[target].values())))
                    break
                if target in on_stack:
                    low[k] = min(low[k], index[target])
            else:
                pending.pop()
                if pending:
                    parent = pending[-1][0]
                    low[parent] = min(low[parent], low[k])
                if low[k] == index[k]:
                    component = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.append(member)
                        if member == k:
                            break
                    # A move always changes the position, so a single position never forms a loop.
                    if len(component) > 1:
                        found.append(sorted(component))
    return found


def unroll(transitions: Transitions) -> Transitions:
    """Transitions whose nodes inside repetition loops also record how often each loop position has occurred.

    A game is drawn when a position occurs for the third time, so inside a loop a node's future depends on
    the game's history. Only the loop's own positions matter: positions before the loop cannot recur, and
    a game that leaves a loop can never return to it. Such a node is written `position#counts`, one digit
    per loop position in sorted order (see board_cache.position_of). The plain position stands for its
    first occurrence with no other loop position played yet, which is how a game arrives from outside the
    loop. A move into a third occurrence has the target None. Without loops, `transitions` is returned
    unchanged.
    """
    found = loops(transitions)
    if not found:
        return transitions
    result = dict(transitions)
    for loop in found:
        place = {k: i for i, k in enumerate(loop)}

        def node(k: Position, counts: tuple[int, ...]) -> Position:
            return k if sum(counts) == 1 else f"{k}#{''.join(map(str, counts))}"

        done: set[Position] = set()
        for start in loop:
            pending = [(start, tuple(int(k == start) for k in loop))]
            while pending:
                k, counts = pending.pop()
                name = node(k, counts)
                if name in done:
                    continue
                done.add(name)
                if len(done) > MAX_LOOP_NODES:
                    raise ValueError(f"Repetition loop too large to unroll: {loop}")
                moves: dict[str, tuple[Position | None, float | None]] = {}
                for move, (target, weight) in transitions[k].items():
                    if target is None or target not in place:
                        moves[move] = (target, weight)
                    elif counts[place[target]] == 2:
                        moves[move] = (None, weight)
                    else:
                        after = list(counts)
                        after[place[target]] += 1
                        moves[move] = (node(target, tuple(after)), weight)
                        pending.append((target, tuple(after)))
                result[name] = moves
    return result


def automatic_entries(graph: Graph, color: bool, cid: str, root: Position) -> list[Position]:
    """The first positions on the chapter's lines that no other chapter continues from.

    At own turns, follow the chapter's first recorded move, as its comparison policy does. A chapter that only
    reaches a position, ending its line there, does not share it; this is how a chapter hands a line to another.
    A line ending on a position no other chapter reaches makes that position an entry.
    """
    entries, visited, pending = set(), set(), [root]
    while pending:
        k = pending.pop()
        if k in visited:
            continue
        visited.add(k)
        n = graph.nodes[k]
        moves = n.chapter_moves.get(cid)
        if any(recorded for other, recorded in n.chapter_moves.items() if other != cid):
            if moves and turn(k) == color:
                pending.append(n.edges[moves[0]])
            elif moves:
                pending.extend(n.edges[m] for m in moves)
        elif moves or n.chapters == {cid}:
            entries.add(k)
    return sorted(entries)


def infer_entries(graph: Graph, color: bool) -> dict[str, InferredEntries]:
    """Each chapter's automatic entry positions, with representative paths."""
    entries: dict[str, InferredEntries] = {}
    for c in graph.chapters:
        positions = automatic_entries(graph, color, c["id"], c["root"])
        entries[c["id"]] = {
            "positions": positions,
            "paths": [graph.nodes[k].path for k in positions],
            "status": "automatic" if positions else "no_position_only_this_chapter_continues_from",
        }
    return entries


def chapter_positions(graph: Graph, chapter_id: str, entries: Iterable[Position]) -> set[Position]:
    """The entries and every descendant through moves this chapter records, including shared positions.

    Describes what the chapter prepares. Reach and scores use the entries alone.
    """
    positions, pending = set(), list(entries)
    while pending:
        k = pending.pop()
        if k in positions:
            continue
        positions.add(k)
        node = graph.nodes[k]
        pending.extend(target for move, target in node.edges.items() if chapter_id in node.provenance[move])
    return positions
