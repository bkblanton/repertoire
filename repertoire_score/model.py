"""Evidence preparation, independent of network access."""

from dataclasses import dataclass, field

from .board_cache import children, owner_outcome, turn
from .explorer import counts, validate


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


class MissingEvidence(ValueError):
    """A position the model needs has no cached table; distinct from a table with zero games."""


def prepare_node(k, selected, color, evidence):
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


def prepare(graph, transitions, order, color, evidence):
    return {k: prepare_node(k, transitions[k], color, evidence) for k in order}


def score(count, color):
    total = sum(count)
    return ((count[0] if color else count[2]) + 0.5 * count[1]) / total if total else None


def node_empirical(n, color):
    """Each branch's observed probability and score, in the (probability, score) format of evaluate."""
    return [
        (
            b.weight if n.mode == "own" else sum(b.counts) / n.sample if n.mode == "opponent" else 1.0,
            b.fixed_score if b.fixed_score is not None else score(b.counts, color),
        )
        for b in n.branches
    ]


def empirical(model, color):
    return {k: node_empirical(n, color) for k, n in model.items()}
