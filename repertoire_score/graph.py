"""PGN parsing, position identity, policy resolution and chapter entries."""

from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.pgn

from .board_cache import canonical as key
from .board_cache import children, terminal_white, turn


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
    chapters: list[dict]
    roots: list[str]


def read_games(path):
    """Every PGN game in a file, in order. The file is closed before any game is checked."""
    games = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        while (game := chess.pgn.read_game(stream)) is not None:
            games.append(game)
    return games


def parse(path, exclusions=()):
    return parse_games(read_games(path), exclusions)


def parse_games(games, exclusions=()):
    """The repertoire graph of PGN games, each one a chapter, in priority order."""
    nodes, chapters, roots = {}, [], []
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
        stack = [(game, board, [])]
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


def conflicts(graph, color):
    return [
        {"position": k, "path": n.path, "choices": {m: sorted(n.provenance[m]) for m in n.edges}}
        for k, n in graph.nodes.items()
        if turn(k) == color and len(n.edges) > 1
    ]


def resolve(graph, color, policy):
    """Explicit overrides, otherwise first PGN move in first chapter order.

    The scorer passes the winners of competing chapter alternatives as part of `policy`.
    """
    transitions = {}
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


def chapter_alternatives(graph, color, policy):
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


def alternative_transitions(graph, color, policy):
    """The selected policy, plus every chapter's first move at each own-turn board.

    Every board any chapter alternative or chapter comparison policy can reach is reachable here, so evidence
    can be planned before scoring decides between alternatives. Own-turn boards may have several moves.
    """
    transitions = resolve(graph, color, policy)
    for k, n in graph.nodes.items():
        if turn(k) == color and terminal_white(k) is None:
            for recorded in n.chapter_moves.values():
                if recorded and recorded[0] not in transitions[k]:
                    transitions[k][recorded[0]] = (n.edges[recorded[0]], None)
    return transitions


def reachable(transitions, roots):
    """Every board reachable from the roots, in discovery order; unlike topology, cycles are allowed."""
    seen, pending = {}, list(roots)
    while pending:
        k = pending.pop()
        if k in seen:
            continue
        seen[k] = None
        pending.extend(target for target, _ in transitions[k].values())
    return list(seen)


def chapter_policy_overrides(graph, color, global_transitions, chapter_id):
    """Prefer this chapter's first recorded own move; use the global policy elsewhere.

    Keep chapter-local ordering separately: filtering global edge order would
    incorrectly inherit an earlier chapter's variation order.
    """
    overrides = {}
    for k, node in graph.nodes.items():
        moves = node.chapter_moves.get(chapter_id, [])
        if moves and turn(k) == color:
            chosen = moves[0]
            if global_transitions[k] != {chosen: (node.edges[chosen], 1.0)}:
                overrides[k] = chosen
    return overrides


def topology(transitions, roots):
    order, visited, active = [], set(), []

    def visit(k):
        if k in active:
            raise ValueError(
                f"Reachable cycle; change policy or exclude the repeated line: {active[active.index(k) :] + [k]}"
            )
        if k in visited:
            return
        active.append(k)
        for target, _ in transitions[k].values():
            visit(target)
        active.pop()
        visited.add(k)
        order.append(k)

    for root in roots:
        visit(root)
    return order


def chapter_frontier(graph, cid, root):
    """The first positions along the chapter's own moves that belong to no other chapter."""
    frontier, visited, pending = set(), set(), [root]
    while pending:
        k = pending.pop()
        if k in visited:
            continue
        visited.add(k)
        n = graph.nodes[k]
        if n.chapters == {cid}:
            frontier.add(k)
            continue
        pending.extend(target for move, target in n.edges.items() if cid in n.provenance[move])
    return frontier


def infer_entries(graph):
    """Find all first chapter-unique positions across every variation."""
    entries = {}
    for c in graph.chapters:
        cid = c["id"]
        frontier = chapter_frontier(graph, cid, c["root"])
        entries[cid] = {
            "positions": sorted(frontier),
            "candidates": sorted(frontier),
            "status": "inferred_unique_frontier"
            if len(frontier) == 1
            else "inferred_multiple_frontiers"
            if frontier
            else "automatic_fallback_needed",
        }
    return entries


def chapter_region(graph, chapter_id, anchors):
    """Chapter-owned descendants of explicit subject anchors, including shared nodes.

    Do not follow other chapters' continuations when defining membership. The
    evaluator still follows the complete repertoire after entering the region.
    """
    if not anchors:
        raise ValueError(f"Chapter region requires anchors: {chapter_id}")
    for k in anchors:
        if k not in graph.nodes or chapter_id not in graph.nodes[k].chapters:
            raise ValueError(f"Region anchor does not belong to chapter {chapter_id}: {k}")
    region, pending = set(), list(anchors)
    while pending:
        k = pending.pop()
        if k in region:
            continue
        region.add(k)
        node = graph.nodes[k]
        pending.extend(target for move, target in node.edges.items() if chapter_id in node.provenance[move])
    return region


def region_entries(transitions, roots, region):
    """All possible first arrivals, including later bypasses of earlier entries.

    Walk from roots, stopping each path on entry. Merely pruning entries that
    descend from another entry would incorrectly discard late transpositions.
    """
    frontier, visited, pending = set(), set(), list(roots)
    while pending:
        k = pending.pop()
        if k in visited:
            continue
        visited.add(k)
        if k in region:
            frontier.add(k)
        else:
            pending.extend(target for target, _ in transitions[k].values())
    return sorted(frontier)
