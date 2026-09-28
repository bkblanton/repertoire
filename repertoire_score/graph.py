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
                    pending.append((child, after, moves + [san]))
                stack.extend(reversed(pending))
    if not chapters:
        raise ValueError("No included PGN chapters")
    return Graph(nodes, chapters, list(dict.fromkeys(roots)))


def conflicts(graph, color):
    return [{"position": k, "path": n.path, "choices": {m: sorted(n.provenance[m]) for m in n.edges}}
            for k, n in graph.nodes.items() if chess.Board(n.fen).turn == color and len(n.edges) > 1]


def resolve(graph, color, policy):
    transitions, unresolved = {}, []
    for k, n in graph.nodes.items():
        board = chess.Board(n.fen)
        if not n.edges:
            transitions[k] = {}
        elif board.turn == color:
            chosen = policy.get(k)
            if chosen is None:
                if len(n.edges) > 1:
                    unresolved.append(k)
                    continue
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
    if unresolved:
        raise ValueError(f"{len(unresolved)} own-move policy conflicts. Configure policy by canonical position; run inspect for details.")
    return transitions


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
    """Only infer a unique chapter-specific frontier across every variation."""
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
        entries[cid] = {"positions": sorted(frontier) if len(frontier) == 1 else [],
                        "candidates": sorted(frontier), "status": "inferred_unique_frontier" if len(frontier) == 1 else "configuration_required"}
    return entries
