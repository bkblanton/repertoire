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


def outcome(board, color):
    if board.is_checkmate():
        return float(board.turn != color)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.5
    return None


def prepare(graph, transitions, order, color, evidence):
    model = {}
    for k in order:
        selected = transitions[k]
        result = owner_outcome(k, color)
        if result is not None:
            model[k] = ModelNode("stop", [Branch(kind="theory_leaf" if not selected else "other_stop", fixed_score=result)])
        elif selected and turn(k) == color:
            model[k] = ModelNode("own", [Branch(move=m, target=t, weight=w) for m, (t, w) in selected.items()])
        else:
            data = evidence[k]
            residual = validate(data, k)
            total = sum(counts(data))
            if turn(k) == color:
                model[k] = ModelNode("stop", [Branch(counts=counts(data), kind="theory_leaf")], total)
            elif not total:
                model[k] = ModelNode("stop", [Branch(kind="unresolved_distribution")], 0,
                                     [target for target, _ in selected.values()])
            else:
                rows = {r["uci"]: counts(r) for r in data["moves"]}
                branches = []
                moves = children(k)
                for uci in sorted(moves):
                    target = selected.get(uci, (None, None))[0]
                    branches.append(Branch(move=uci, target=target, counts=rows.get(uci, [0, 0, 0]),
                                           kind=None if target else "deviation",
                                           fixed_score=None if target else owner_outcome(moves[uci], color)))
                if sum(residual):
                    branches.append(Branch(counts=residual, kind="no_recorded_continuation"))
                model[k] = ModelNode("opponent", branches, total)
    return model


def score(count, color):
    total = sum(count)
    return ((count[0] if color else count[2])+0.5*count[1])/total if total else None


def empirical(model, color):
    return {k: [(b.weight if n.mode == "own" else sum(b.counts)/n.sample if n.mode == "opponent" else 1.0,
                 b.fixed_score if b.fixed_score is not None else score(b.counts, color)) for b in n.branches]
            for k, n in model.items()}
