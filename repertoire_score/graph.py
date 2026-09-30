"""PGN parsing, position identity, policy resolution and chapter entries."""
from dataclasses import dataclass, field
from pathlib import Path
import chess
import chess.pgn


def key(board):
    return " ".join(board.fen(en_passant="legal").split()[:4])


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


def parse(path, exclusions=()):
    nodes, chapters, roots = {}, [], []
    with Path(path).open(encoding="utf-8-sig") as stream:
        while (game := chess.pgn.read_game(stream)) is not None:
            if game.errors:
                raise ValueError(f"Malformed chapter: {game.headers}: {game.errors}")
            if game.headers.get("Variant", "Standard") not in ("Standard", "Chess"):
                raise ValueError("Only standard chess is supported")
            board = game.board()
            if not board.is_valid():
                raise ValueError(f"Invalid root position: {board.fen()}")
            cid = game.headers.get("ChapterURL", "").rstrip("/").split("/")[-1] or str(len(chapters)+1)
            if cid in exclusions:
                continue
            if any(c["id"] == cid for c in chapters):
                cid += f"-copy-{len(chapters)+1}"
            root = key(board)
            roots.append(root)
            mainline = [root]
            mb = board.copy()
            for move in game.mainline_moves():
                mb.push(move)
                mainline.append(key(mb))
            chapters.append({"id": cid, "name": game.headers.get("ChapterName", game.headers.get("Event", cid)),
                             "url": game.headers.get("ChapterURL"), "root": root, "mainline": mainline})
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
    return [{"position": k, "path": n.path, "choices": {m: sorted(n.provenance[m]) for m in n.edges}}
            for k, n in graph.nodes.items() if chess.Board(n.fen).turn == color and len(n.edges) > 1]


def resolve(graph, color, policy):
    """Explicit overrides, otherwise first PGN move in first chapter order."""
    transitions = {}
    for k, n in graph.nodes.items():
        board = chess.Board(n.fen)
        if not n.edges:
            transitions[k] = {}
        elif board.turn == color:
            chosen = policy.get(k)
            if chosen is None:
                chosen = next(iter(n.edges))
            weights = {chosen: 1.0} if isinstance(chosen, str) else chosen
            if not weights or any(m not in n.edges or not 0 <= w <= 1 for m, w in weights.items()) or abs(sum(weights.values())-1) > 1e-10:
                raise ValueError(f"Invalid policy at {k}: {weights}")
            transitions[k] = {m: (n.edges[m], w) for m, w in weights.items() if w > 0}
        else:
            transitions[k] = {m: (target, None) for m, target in n.edges.items()}
            for move in board.legal_moves:
                after = board.copy()
                after.push(move)
                target = key(after)
                if target in graph.nodes:
                    transitions[k][move.uci()] = (target, None)
    return transitions


def chapter_policy_overrides(graph, color, global_transitions, chapter_id):
    """Prefer this chapter's first recorded own move; use the global policy elsewhere.

    Keep chapter-local ordering separately: filtering global edge order would
    incorrectly inherit an earlier chapter's variation order.
    """
    overrides = {}
    for k, node in graph.nodes.items():
        moves = node.chapter_moves.get(chapter_id, [])
        if moves and chess.Board(node.fen).turn == color:
            chosen = moves[0]
            if global_transitions[k] != {chosen: (node.edges[chosen], 1.0)}:
                overrides[k] = chosen
    return overrides


def topology(transitions, roots):
    order, visited, active = [], set(), []
    def visit(k):
        if k in active:
            raise ValueError(f"Reachable cycle; change policy or exclude the repeated line: {active[active.index(k):] + [k]}")
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


def infer_entries(graph):
    """Find all first chapter-unique positions across every variation."""
    entries = {}
    for c in graph.chapters:
        cid, frontier, visited = c["id"], set(), set()
        def walk(k):
            if k in visited:
                return
            visited.add(k)
            n = graph.nodes[k]
            if n.chapters == {cid}:
                frontier.add(k)
                return
            for move, target in n.edges.items():
                if cid in n.provenance[move]:
                    walk(target)
        walk(c["root"])
        entries[cid] = {"positions": sorted(frontier), "candidates": sorted(frontier),
                        "status": "inferred_unique_frontier" if len(frontier) == 1 else
                        "inferred_multiple_frontiers" if frontier else "automatic_fallback_needed"}
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
        pending.extend(target for move, target in node.edges.items()
                       if chapter_id in node.provenance[move])
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
