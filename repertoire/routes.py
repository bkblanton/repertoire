"""Exact cache-only depth distributions and first-arrival route examples."""

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, cast

from .board_cache import children, fen_number, move_text, next_number, position_of, san
from .evaluate import Route, Weights, better_route, by_position
from .schema import JsonObject, Position
from .status import Status

if TYPE_CHECKING:
    from .preparation import Evaluator

ENDING_TYPES = ('prepared_endpoint', 'unprepared_reply', 'game_over', 'other_stop', 'unresolved_distribution')


def depth_distribution(evaluator: 'Evaluator | None', starts: Weights) -> JsonObject:
    """Propagate board AND elapsed own-move count, retaining transposed histories.

    Missing leaf scores do not obscure depth. Missing reply distributions leave
    a finite interval between moves already played and the longest continuation.
    """
    if not starts:
        return dict(
            status=Status.UNRESOLVED_ENTRY_WEIGHTS,
            unit='own_moves',
            endings=[],
            survival=[],
            expected_moves=None,
            expected_bounds=None,
            median_moves=None,
            median_bounds=None,
        )
    if any(not math.isfinite(w) or w < 0 for w in starts.values()) or not math.isclose(
        sum(starts.values()), 1.0, abs_tol=1e-10
    ):
        raise ValueError('Depth distribution requires normalized nonnegative starting weights')
    evaluator = cast('Evaluator', evaluator)
    value = evaluator.evaluate(starts)
    starts = evaluator.entry_nodes(starts)
    mass: dict[Position, defaultdict[int, float]] = {k: defaultdict(float) for k in evaluator.values}
    for k, weight in starts.items():
        if weight:
            mass[k][0] += weight
    endings: defaultdict[int, dict[str, float]] = defaultdict(lambda: dict.fromkeys(ENDING_TYPES, 0.0))
    unknown: list[tuple[int, int, float]] = []
    lengths: dict[Position, int] = {}
    active: set[Position] = set()

    def remaining(k: Position) -> int:
        if k in lengths:
            return lengths[k]
        if k in active:
            raise ValueError('Repertoire contains a reachable cycle')
        active.add(k)
        node, fact = evaluator.graph.nodes[position_of(k)], evaluator.facts[position_of(k)]
        # Every prepared move, and every reply into a known board; a repetition ends the game.
        targets = {target for target, _ in evaluator.transitions[k].values() if target is not None}
        if fact['outcome'] is not None or (fact['turn'] == evaluator.color and not node.edges):
            result = 0
        elif fact['turn'] == evaluator.color:
            result = 1 + max((remaining(target) for target in targets), default=0)
        else:
            result = max((remaining(target) for target in targets), default=0)
        active.remove(k)
        lengths[k] = result
        return result

    for k in reversed(evaluator.values):
        node, fact = evaluator.graph.nodes[position_of(k)], evaluator.facts[position_of(k)]
        reward = int(bool(node.edges) and fact['turn'] == evaluator.color and fact['outcome'] is None)
        for depth, arrival in mass[k].items():
            for _, probability, target in evaluator.edges[k]:
                mass[target][depth + reward] += arrival * probability
            for _, probability, kind, _, fixed in evaluator.stops[k]:
                probability *= arrival
                if not probability:
                    continue
                ending = (
                    'game_over'
                    if fixed is not None
                    else 'prepared_endpoint'
                    if kind == 'theory_leaf'
                    else 'unprepared_reply'
                    if kind == 'deviation'
                    else 'unresolved_distribution'
                    if kind == 'unresolved_distribution'
                    else 'other_stop'
                )
                # A repetition can end the game on your own move, which counts as played.
                endings[depth + reward][ending] += probability
                if ending == 'unresolved_distribution':
                    unknown.append((depth, depth + remaining(k), probability))

    total = sum(sum(row.values()) for row in endings.values())
    if not math.isclose(total, sum(starts.values()), abs_tol=1e-10):
        raise AssertionError('Prepared-depth distribution does not conserve probability')
    known = {d: sum(v for kind, v in row.items() if kind != 'unresolved_distribution') for d, row in endings.items()}
    maximum = max([*endings, *[high for _, high, _ in unknown]], default=0)
    survival: list[JsonObject] = []
    for depth in range(maximum + 1):
        completed = sum(p for d, p in known.items() if d >= depth)
        low = completed + sum(p for d, _, p in unknown if d >= depth)
        high = completed + sum(p for _, d, p in unknown if d >= depth)
        survival.append(dict(own_moves=depth, probability=low if low == high else None, bounds=[low, high]))
    low = sum(row['bounds'][0] for row in survival[1:])
    high = sum(row['bounds'][1] for row in survival[1:])
    if not math.isclose(low, float(value[2]), abs_tol=1e-10):
        raise AssertionError('Survival curve does not reproduce expected prepared depth')

    def median(upper: bool) -> int | None:
        distribution = defaultdict(float, known)
        for minimum, maximum, p in unknown:
            distribution[maximum if upper else minimum] += p
        cumulative = 0.0
        for depth, p in sorted(distribution.items()):
            cumulative += p
            if cumulative >= 0.5 - 1e-12:
                return depth
        return None

    median_bounds = [median(False), median(True)]
    return dict(
        status=Status.RESOLVED if low == high else Status.UNRESOLVED_MOVE_DISTRIBUTION,
        unit='own_moves',
        expected_moves=low if low == high else None,
        expected_bounds=[low, high],
        median_moves=median_bounds[0] if median_bounds[0] == median_bounds[1] else None,
        median_bounds=median_bounds,
        unresolved_probability=sum(p for _, _, p in unknown),
        endings=[dict(own_moves=d, **endings[d]) for d in sorted(endings)],
        survival=survival,
    )


def first_entry_examples(
    evaluator: 'Evaluator',
    roots: Weights,
    entries: Iterable[Position],
    expected_probability: float | None = None,
    expected_weights: Mapping[Position, float] | None = None,
) -> JsonObject:
    """Sum ALL first arrivals; retain the most likely genuine route to each board.

    Example-route mass is only a subset of its entry-position mass. Paths stop
    on first chapter arrival, so a bypass example cannot pass an earlier entry.
    """
    entries = set(entries)
    mass = evaluator.reaches(roots, stop_at=entries)
    routes = evaluator.routes(roots, stop_at=entries)
    nodes = {k: mass[k] for k in reversed(evaluator.values) if position_of(k) in entries and mass[k]}
    # Inside a repetition loop an entry can be first reached in several history nodes.
    arrivals = by_position(nodes)
    witnesses: dict[Position, Route] = {}
    for k in nodes:
        position = position_of(k)
        if position not in witnesses or better_route(routes[k], witnesses[position]):
            witnesses[position] = routes[k]
    total = sum(arrivals.values())
    if expected_probability is not None and not math.isclose(total, expected_probability, abs_tol=1e-10):
        raise AssertionError('First-entry examples differ from saved chapter reach')
    if not total:
        return dict(status=Status.NO_REACHABLE_ENTRY, entry_probability=total, positions=[])
    if expected_weights is not None:
        for k in set(arrivals) | set(expected_weights):
            if not math.isclose(arrivals.get(k, 0.0) / total, expected_weights.get(k, 0.0), abs_tol=1e-10):
                raise AssertionError('First-entry examples differ from saved entry weights')
    rows: list[JsonObject] = []
    for k, arrival in sorted(arrivals.items(), key=lambda item: (-item[1], item[0])):
        probability, root, path = witnesses[k]
        position, number = position_of(root), fen_number(evaluator.graph.nodes[position_of(root)].fen)
        sans: list[str] = []
        text: list[str] = []
        for uci in path:
            if position in entries:
                raise AssertionError('Example route passes through an earlier chapter entry')
            sans.append(san(position, uci))
            text.append(move_text(position, number, uci))
            position, number = children(position)[uci], next_number(position, number)
        if position != k:
            raise AssertionError('Example route does not reach its entry board')
        rows.append(
            dict(
                position=k,
                conditional_first_entry_weight=arrival / total,
                example=dict(
                    root_position=position_of(root),
                    root_fen=evaluator.graph.nodes[position_of(root)].fen,
                    path_uci=list(path),
                    path_san=sans,
                    line=' '.join(text) or '(PGN root)',
                    root_probability=probability,
                    conditional_probability=probability / total,
                ),
            )
        )
    return dict(status=Status.RESOLVED, entry_probability=total, positions=rows)
