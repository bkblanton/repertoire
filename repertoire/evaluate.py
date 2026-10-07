"""DAG evaluation, first-entry weighting and probability conservation checks."""

from collections.abc import Callable, Collection, Mapping

import numpy as np

from .board_cache import children
from .graph import Graph, chapter_alternatives, resolve, topology
from .model import Branch, Evidence, Model, Sampled, node_empirical, prepare_node
from .schema import Position, ScoreSummary
from .status import Status

# Vector components: resolved contribution, unresolved mass, sparse contribution,
# sparse mass, leaf mass, deviation mass, other mass, prior-completed score.
KNOWN, UNKNOWN, SPARSE_KNOWN, SPARSE, LEAF, DEVIATION, OTHER, COMPLETED = range(8)

Weights = Mapping[Position, float]
Route = tuple[float, Position, tuple[str, ...]]  # (probability, root, moves)


def stopping_vector(
    branch: Branch, sampled_score: float | None, sparse_threshold: int, width: int | None = None
) -> np.ndarray:
    result = np.zeros(8 if width is None else (8, width))
    unresolved = branch.fixed_score is None and not sum(branch.counts)
    sparse = branch.fixed_score is None and sum(branch.counts) < sparse_threshold
    s = 0.5 if sampled_score is None else sampled_score
    result[COMPLETED] = s
    result[UNKNOWN] = float(unresolved)
    result[KNOWN] = 0 if unresolved else s
    result[SPARSE] = float(sparse)
    result[SPARSE_KNOWN] = 0 if unresolved or not sparse else s
    # Unresolved replaces the event's resolved category, keeping all masses disjoint.
    if not unresolved:
        category = LEAF if branch.kind == "theory_leaf" else DEVIATION if branch.kind == "deviation" else OTHER
        result[category] = 1
    return result


def fold(
    model: Model,
    order: list[Position],
    sampled: Sampled,
    stop_value: Callable,
    combine: Callable,
    values: dict | None = None,
) -> dict:
    """Values from the leaves up, in `order` (children first).

    Each position's value is `combine(position, parts)`, where `parts` holds (branch, probability, value) for every
    branch with nonzero probability: the target's value, or `stop_value(branch, sampled_score)` where play stops.
    Positions already in `values` are kept, so a lazily compiled order can be extended without recomputing.
    """
    values = {} if values is None else values
    for k in order:
        if k in values:
            continue
        parts = []
        for b, (p, s) in zip(model[k].branches, sampled[k]):
            if np.ndim(p) == 0 and p == 0:
                continue
            parts.append((b, p, values[b.target] if b.target is not None else stop_value(b, s)))
        values[k] = combine(k, parts)
    return values


def backward(
    model: Model, order: list[Position], sampled: Sampled, sparse_threshold: int, width: int | None = None
) -> dict[Position, np.ndarray]:
    """Each position's component vector. With `width`, every probability and score is an array of that many
    simulated draws and each vector has one column per draw; the tests use this to check the calculated uncertainty."""

    def combine(k, parts):
        v = np.zeros(8 if width is None else (8, width))
        for _, p, value in parts:
            v += p * value
        if not np.allclose(v[UNKNOWN] + v[LEAF] + v[DEVIATION] + v[OTHER], 1, atol=1e-9):
            raise AssertionError(f"Probability conservation failed at {k}")
        return v

    return fold(model, order, sampled, lambda b, s: stopping_vector(b, s, sparse_threshold, width), combine)


def select_alternatives(
    graph: Graph,
    color: bool,
    policy: dict,
    evidence: Evidence,
    roots: Collection[Position],
    sparse_threshold: int,
    fixed: Mapping[Position, str] | None = None,
) -> dict[Position, dict]:
    """Where chapters compete, play the alternative with the highest repertoire score.

    Boards are decided children first (backward induction), so each alternative is scored with the best choices
    below it, and together the choices maximize the score from every root at once. Unresolved evidence counts as
    the prior-completed score when comparing. Exact ties keep the earlier chapter's move. `fixed` settles boards
    in advance, as a comparison does at its decision points.

    Returns {position: {'selected': move, 'scores': {move: component vector}}} for every reachable contested board.
    """
    fixed = fixed or {}
    options = {k: moves for k, moves in chapter_alternatives(graph, color, policy).items() if k not in fixed}
    if not options:
        return {}
    transitions = resolve(graph, color, dict(policy, **fixed))
    for k, moves in options.items():
        transitions[k] = {m: (graph.nodes[k].edges[m], None) for m in moves}
    values, result = {}, {}
    for k in topology(transitions, list(roots)):
        selected = transitions[k]
        if k in options:
            scores = {m: values[target] for m, (target, _) in selected.items()}
            # max() keeps the first of equal scores, which is the earlier chapter's move.
            best = max(options[k], key=lambda m: scores[m][COMPLETED])
            result[k] = dict(selected=best, scores=scores)
            selected = {best: (selected[best][0], 1.0)}
        node = prepare_node(k, selected, color, evidence)
        value = np.zeros(8)
        for b, (p, s) in zip(node.branches, node_empirical(node, color)):
            if p:
                value += p * (values[b.target] if b.target is not None else stopping_vector(b, s, sparse_threshold))
        values[k] = value
    return result


def can_enter(model: Model, order: list[Position], entries: Collection[Position]) -> dict[Position, bool]:
    """Whether each position can still reach one of `entries`, including through unresolved replies."""
    result = {}
    for k in order:
        targets = [b.target for b in model[k].branches if b.target is not None] + model[k].potential_targets
        result[k] = k in entries or any(result[t] for t in targets)
    return result


def forward(
    model: Model,
    order: list[Position],
    sampled: Sampled,
    roots: Weights,
    stop_at: Collection[Position] = (),
    entering: dict[Position, bool] | None = None,
) -> tuple[dict[tuple[Position, int], float], dict[Position, float]]:
    """Propagate probability from the roots, stopping at `stop_at`.

    With `entering` (from can_enter), positions that cannot reach an entry are not expanded: their
    mass is absorbed whole and their individual stops are omitted. Entry masses are unchanged.
    Probabilities may be arrays of simulated draws; masses then become arrays of the same length.
    """
    mass = dict.fromkeys(order, 0.0)
    for k, weight in roots.items():
        mass[k] += weight
    stops, entries = {}, {}
    absorbed = 0.0
    for k in reversed(order):
        if k in stop_at:
            entries[k] = mass[k]
            continue
        if entering is not None and not entering[k]:
            absorbed += mass[k]
            continue
        for j, (b, (p, _)) in enumerate(zip(model[k].branches, sampled[k])):
            flow = mass[k] * p
            if b.target is not None:
                mass[b.target] += flow
            else:
                stops[(k, j)] = flow
    total = sum(stops.values(), 0.0) + sum(entries.values(), 0.0) + absorbed
    if not np.allclose(total, 1, atol=1e-9):
        raise AssertionError("Forward stopping mass does not sum to one")
    return stops, entries


def reaches(
    model: Model, order: list[Position], sampled: Sampled, roots: Weights, stop_at: Collection[Position] = ()
) -> dict[Position, float]:
    """Incoming probability at each position over every route from the roots; transpositions add up.

    Flow is not expanded past `stop_at`, so each of those positions holds its first-entry probability.
    """
    stop_at = set(stop_at)
    mass = dict.fromkeys(order, 0.0)
    for k, weight in roots.items():
        if weight:
            mass[k] += weight
    left = 0.0
    for k in reversed(order):
        if k in stop_at:
            left += mass[k]
            continue
        for b, (p, _) in zip(model[k].branches, sampled[k]):
            if b.target is None:
                left += mass[k] * p
            else:
                mass[b.target] += mass[k] * p
    if not np.allclose(left, sum(roots.values()), atol=1e-9):
        raise AssertionError("Forward reach does not conserve probability")
    return mass


def better_route(a: Route, b: Route) -> bool:
    """Whether route `a` beats `b`: more likely, with exact ties going to the earlier root and then earlier moves."""
    return a[0] > b[0] or (a[0] == b[0] and a[1:] < b[1:])


def best_routes(
    model: Model,
    order: list[Position],
    sampled: Sampled,
    roots: Weights,
    stop_at: Collection[Position] = (),
    replies: bool = False,
) -> dict[Position, Route]:
    """The most likely single route to each position, as (probability, root, moves).

    Probabilities always use all routes (see `reaches`); this route only labels a position. Routes are not
    extended past `stop_at`. With `replies`, the boards after unprepared opponent replies get routes too.
    """
    stop_at = set(stop_at)
    best = {k: (w, k, ()) for k, w in roots.items() if w > 0}
    for k in reversed(order):
        if k not in best or k in stop_at:
            continue
        probability, root, moves = best[k]
        for b, (p, _) in zip(model[k].branches, sampled[k]):
            if p <= 0:
                continue
            if b.target is not None:
                target = b.target
            elif replies and b.kind == "deviation":
                target = children(k)[b.move]
            else:
                continue
            candidate = probability * p, root, (*moves, b.move)
            if target not in best or better_route(candidate, best[target]):
                best[target] = candidate
    return best


def summarize(r: np.ndarray, posterior) -> ScoreSummary:
    """Empirical score fields from the raw component vector `r`, plus the posterior block from uncertainty.py."""
    u = float(r[UNKNOWN])
    return {
        "raw_empirical_score": float(r[KNOWN]) if u == 0 else None,
        "resolved_contribution": float(r[KNOWN]),
        "unresolved_mass": u,
        "conditional_bounds": [float(r[KNOWN]), float(r[KNOWN] + u)],
        "sparse_mass": float(r[SPARSE]),
        "sparse_sensitivity": [float(r[KNOWN] - r[SPARSE_KNOWN]), float(r[KNOWN] - r[SPARSE_KNOWN] + r[SPARSE])],
        "masses": {
            "theory_leaf": float(r[LEAF]),
            "deviation": float(r[DEVIATION]),
            "other_stop": float(r[OTHER]),
            "unresolved": u,
        },
        "posterior": posterior,
    }


def chapter_score(
    model: Model,
    order: list[Position],
    raw_sample: Sampled,
    values: dict[Position, np.ndarray],
    root_weights: Weights,
    entries: Collection[Position],
    posterior,
) -> dict:
    """Score conditional on first entry, with `posterior` an uncertainty.Posterior for the same model."""
    # Only positions that can still reach the chapter matter for its entry weights.
    entering = can_enter(model, order, entries)
    raw_stops, raw_entries = forward(model, order, raw_sample, root_weights, stop_at=entries, entering=entering)
    raw_reach = sum(raw_entries.values(), 0.0)
    r = sum((raw_entries[k] * values[k] for k in raw_entries), np.zeros(8))
    weights = {k: float(v / raw_reach) if raw_reach else None for k, v in raw_entries.items()}
    unresolved_entry_mass = sum(
        float(flow)
        for (k, j), flow in raw_stops.items()
        if model[k].branches[j].kind == "unresolved_distribution" and entering[k]
    )
    if unresolved_entry_mass > 0 and len(entries) > 1:
        return {
            "status": Status.UNRESOLVED_ENTRY_WEIGHTS,
            "entry_probability": None,
            "entry_probability_bounds": [float(raw_reach), float(raw_reach) + unresolved_entry_mass],
            "conditional_score_bounds": [
                min(values[k][KNOWN] for k in entries),
                max(values[k][KNOWN] + values[k][UNKNOWN] for k in entries),
            ],
        }
    chapter = posterior.chapter(root_weights, entries)
    if raw_reach <= 0:
        if len(entries) == 1:
            k = next(iter(entries))
            summary = summarize(values[k], posterior.mixture({k: 1.0}))
            summary["conditional_basis"] = "single entry position, even though root reach is zero"
        else:
            summary = {"status": Status.UNREACHABLE_MULTIPLE_ENTRIES}
    else:
        if chapter['summary'] is None:
            raise ValueError("Posterior first-entry weights are zero; use a stronger prior or explicit entries")
        summary = summarize(r / raw_reach, chapter['summary'])
    summary.update(
        entry_probability=float(raw_reach),
        posterior_entry_probability_mean=chapter['entry_probability'],
        first_entry_weights=weights,
    )
    if unresolved_entry_mass > 0:
        summary["entry_probability"] = None
        summary["posterior_entry_probability_mean"] = None
        summary["entry_probability_bounds"] = [float(raw_reach), float(raw_reach) + unresolved_entry_mass]
    return summary
