"""Evidence preparation, independent of network access."""

from dataclasses import dataclass, field
from typing import cast

from .board_cache import children, owner_outcome, turn
from .explorer import counts, validate
from .graph import Graph
from .schema import Position


@dataclass
class Branch:
    move: str | None = None
    target: str | None = None
    counts: list[int] = field(default_factory=lambda: [0, 0, 0])
    kind: str | None = None
    fixed_score: float | None = None
    weight: float | None = None


@dataclass
class ModelNode:
    mode: str
    branches: list[Branch]
    sample: int = 0
    potential_targets: list[str] = field(default_factory=list)


# Colors are booleans as in python-chess: True is White. A sampled branch is (probability, score); in the
# simulation tests both may be arrays of draws.
Sample = tuple[float, float | None]
Model = dict[Position, ModelNode]
Sampled = dict[Position, list[Sample]]
Evidence = dict[Position, dict]
Selected = dict[str, tuple[Position, float | None]]  # move -> (target, policy weight; None at opponent turns)


class MissingEvidence(ValueError):
    """A position the model needs has no cached table; distinct from a table with zero games."""


def prepare_node(k: Position, selected: Selected, color: bool, evidence: Evidence) -> ModelNode:
    """The model node for one position, from its selected transitions and cached table."""
    result = owner_outcome(k, color)
    if result is not None:
        return ModelNode("stop", [Branch(kind="theory_leaf" if not selected else "other_stop", fixed_score=result)])
    if selected and turn(k) == color:
        return ModelNode("own", [Branch(move=m, target=t, weight=w) for m, (t, w) in selected.items()])
    if k not in evidence:
        raise MissingEvidence(k)
    data = evidence[k]
    residual = validate(data, k)
    total = sum(counts(data))
    if turn(k) == color:
        return ModelNode("stop", [Branch(counts=counts(data), kind="theory_leaf")], total)
    if not total:
        return ModelNode(
            "stop", [Branch(kind="unresolved_distribution")], 0, [target for target, _ in selected.values()]
        )
    rows = {r["uci"]: counts(r) for r in data["moves"]}
    branches = []
    moves = children(k)
    for uci in sorted(moves):
        target = selected.get(uci, (None, None))[0]
        branches.append(
            Branch(
                move=uci,
                target=target,
                counts=rows.get(uci, [0, 0, 0]),
                kind=None if target else "deviation",
                fixed_score=None if target else owner_outcome(moves[uci], color),
            )
        )
    if sum(residual):
        branches.append(Branch(counts=residual, kind="no_recorded_continuation"))
    return ModelNode("opponent", branches, total)


def prepare(
    graph: Graph, transitions: dict[Position, Selected], order: list[Position], color: bool, evidence: Evidence
) -> Model:
    return {k: prepare_node(k, transitions[k], color, evidence) for k in order}


def arrival_moves(graph: Graph, color: bool, evidence: Evidence) -> dict[Position, list[tuple[Position, str]]]:
    """The recorded opponent moves leading to each own-turn position, as (parent, move), from cached parents."""
    result: dict[Position, list[tuple[Position, str]]] = {}
    for k, node in graph.nodes.items():
        if turn(k) != color and k in evidence:
            for move, target in node.edges.items():
                result.setdefault(target, []).append((k, move))
    return result


def move_counts(data: dict, move: str) -> list[int]:
    return next((counts(r) for r in data["moves"] if r["uci"] == move), [0, 0, 0])


def arrival_counts(graph: Graph, color: bool, evidence: Evidence) -> dict[Position, list[int]]:
    """Games reaching each own-turn position through a recorded opponent move: the sum of those moves' rows in
    the cached parent tables. Tables where you are to move and have a prepared move are never fetched."""
    result: dict[Position, list[int]] = {}
    for k, origins in arrival_moves(graph, color, evidence).items():
        rows = [move_counts(evidence[parent], move) for parent, move in origins]
        result[k] = [sum(column) for column in zip(*rows)]
    return result


def position_counts(arrivals: dict[Position, list[int]], evidence: Evidence, k: Position) -> list[int] | None:
    """Database games at an own-turn position with a prepared move: the opponent move rows that lead to it, or
    its own table where nothing leads to it (the starting position)."""
    if k in arrivals:
        return arrivals[k]
    return counts(evidence[k]) if k in evidence else None


def score(count: list[int], color: bool) -> float | None:
    total = sum(count)
    return ((count[0] if color else count[2]) + 0.5 * count[1]) / total if total else None


def node_empirical(n: ModelNode, color: bool) -> list[Sample]:
    """Each branch's observed probability and score, in the (probability, score) format of evaluate."""
    return [
        (
            cast(float, b.weight) if n.mode == "own" else sum(b.counts) / n.sample if n.mode == "opponent" else 1.0,
            b.fixed_score if b.fixed_score is not None else score(b.counts, color),
        )
        for b in n.branches
    ]


def empirical(model: Model, color: bool) -> Sampled:
    return {k: node_empirical(n, color) for k, n in model.items()}
