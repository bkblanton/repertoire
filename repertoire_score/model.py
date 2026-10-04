"""Evidence preparation, independent of network access."""
from dataclasses import dataclass, field
import chess
import numpy as np
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
        n, selected = graph.nodes[k], transitions[k]
        board = chess.Board(n.fen)
        result = outcome(board, color)
        if result is not None:
            model[k] = ModelNode("stop", [Branch(kind="theory_leaf" if not selected else "other_stop", fixed_score=result)])
        elif selected and board.turn == color:
            model[k] = ModelNode("own", [Branch(move=m, target=t, weight=w) for m, (t, w) in selected.items()])
        else:
            data = evidence[k]
            residual = validate(data, k)
            total = sum(counts(data))
            if board.turn == color:
                model[k] = ModelNode("stop", [Branch(counts=counts(data), kind="theory_leaf")], total)
            elif not total:
                model[k] = ModelNode("stop", [Branch(kind="unresolved_distribution")], 0,
                                     [target for target, _ in selected.values()])
            else:
                rows = {r["uci"]: counts(r) for r in data["moves"]}
                branches = []
                for move in sorted(board.legal_moves, key=lambda m: m.uci()):
                    uci = move.uci()
                    after = board.copy()
                    after.push(move)
                    target = selected.get(uci, (None, None))[0]
                    branches.append(Branch(move=uci, target=target, counts=rows.get(uci, [0, 0, 0]),
                                           kind=None if target else "deviation",
                                           fixed_score=None if target else outcome(after, color)))
                if sum(residual):
                    branches.append(Branch(counts=residual, kind="no_recorded_continuation"))
                model[k] = ModelNode("opponent", branches, total)
    return model


def score(count, color):
    total = sum(count)
    return ((count[0] if color else count[2])+0.5*count[1])/total if total else None


def draws(model, color, simulations, seed, prior):
    """Sample each local evidence table once and reuse it at every reference."""
    rng = np.random.default_rng(seed)
    sampled = {}
    prior = np.asarray(prior, dtype=float)
    # User prior is in owner win/draw/loss order; tables use white/draw/black.
    table_prior = prior if color else prior[::-1]
    for k in sorted(model):
        node = model[k]
        local = []
        if node.mode == "opponent":
            alpha = np.asarray([b.counts for b in node.branches], dtype=float) + table_prior / len(node.branches)
            gamma = rng.gamma(alpha, size=(simulations, *alpha.shape))
            # Extremely weak unobserved cells can underflow to zero.
            row = gamma.sum(axis=2)
            joint_total = row.sum(axis=1)
            probabilities = row / joint_total[:, None]
            numerators = gamma[:, :, 0 if color else 2]+0.5*gamma[:, :, 1]
            scores = np.divide(numerators, row, out=np.full_like(row, 0.5), where=row>0)
            for j, b in enumerate(node.branches):
                local.append((probabilities[:, j], scores[:, j] if b.fixed_score is None else np.full(simulations, b.fixed_score)))
        elif node.mode == "own":
            local = [(b.weight, None) for b in node.branches]
        else:
            b = node.branches[0]
            if b.fixed_score is not None:
                values = np.full(simulations, b.fixed_score)
            else:
                theta = rng.dirichlet(np.asarray(b.counts)+table_prior, size=simulations)
                values = theta[:, 0 if color else 2] + theta[:, 1]/2
            local = [(1.0, values)]
        sampled[k] = local
    return sampled


def empirical(model, color):
    return {k: [(b.weight if n.mode == "own" else sum(b.counts)/n.sample if n.mode == "opponent" else 1.0,
                 b.fixed_score if b.fixed_score is not None else score(b.counts, color)) for b in n.branches]
            for k, n in model.items()}
