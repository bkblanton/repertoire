"""Unprepared replies after which one of your moves reaches a position you have prepared.

Preparation ends at an unprepared reply, and the model scores it with that reply's row in the parent table: the
database result of every game after it, whatever you played next. If one of your legal moves from there reaches a
prepared position, you could play it and continue your preparation. The change from doing so is

    your score at the prepared position - the database score after the reply,

which splits into the move (the prepared position's database score minus the score after the reply) and your
preparation (your score minus the prepared position's database score). The comparison is with average play after
the reply, not with your best alternative there: those tables are not fetched. A negative change is therefore a
reason not to transpose, while a positive one only says that transposing beats average play.

Nothing here changes the score; the model never follows a game back into preparation.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np

from .board_cache import children, move_text, position_of, san
from .evaluate import KNOWN, UNKNOWN
from .explorer import counts
from .graph import Graph
from .model import Evidence, Model, Sampled, score
from .schema import JsonObject, Position
from .uncertainty import Posterior, difference_interval, row_moments


def pieces(position: Position, white: bool) -> frozenset[tuple[int, str]]:
    """One side's pieces as (square index, letter), read from the FEN text without building a board."""
    result, square = set(), 0
    for char in position.split(' ', 1)[0].replace('/', ''):
        if char.isdigit():
            square += int(char)
        else:
            if char.isupper() == white:
                result.add((square, char))
            square += 1
    return frozenset(result)


def candidates(prepared: Iterable[Position], color: bool) -> dict[frozenset[tuple[int, str]], list[Position]]:
    """Prepared positions indexed by the opponent's pieces. One of your moves leaves those pieces as they were
    or removes one by capture, so a position whose opponent pieces match neither cannot be one move away."""
    index: defaultdict[frozenset[tuple[int, str]], list[Position]] = defaultdict(list)
    for k in prepared:
        index[pieces(k, not color)].append(k)
    return index


def one_move_targets(d: Position, index: Mapping[frozenset[tuple[int, str]], list[Position]], color: bool) -> bool:
    opponent = pieces(d, not color)
    return opponent in index or any(opponent - {piece} in index for piece in opponent)


def free_transpositions(
    graph: Graph,
    model: Model,
    sampled: Sampled,
    values: Mapping[Position, np.ndarray],
    reach: Mapping[Position, float],
    posterior: Posterior,
    evidence: Evidence,
    color: bool,
    lines: Mapping[Position, tuple[str, int]],
    sparse_threshold: int,
) -> list[JsonObject]:
    """One row per unprepared reply position and transposing move, most frequent and largest changes first."""
    # Reached unprepared replies by the position they lead to: (reach, row score, games, row variance, parent, move).
    exits: defaultdict[Position, list[tuple[float, float, int, float, Position, str]]] = defaultdict(list)
    for t, r in reach.items():
        node = model[t]
        if r <= 0 or node.mode != 'opponent':
            continue
        moves = children(position_of(t))
        for j, (b, (p, s)) in enumerate(zip(node.branches, sampled[t])):
            if b.kind != 'deviation' or b.move is None or b.fixed_score is not None or p <= 0 or s is None:
                continue
            variance = row_moments(posterior.alpha[t][j], posterior.owner)[1]
            exits[moves[b.move]].append((r * p, float(s), sum(b.counts), variance, t, b.move))
    order = {c['id']: i for i, c in enumerate(graph.chapters)}
    index = candidates((k for k, n in model.items() if n.mode == 'opponent' and position_of(k) in evidence), color)
    rows: list[JsonObject] = []
    for d, arrivals in exits.items():
        # Generating every legal move after each reply is the slow part; most replies cannot match.
        if not one_move_targets(d, index, color):
            continue
        weight = sum(a[0] for a in arrivals)
        database = sum(a[0] * a[1] for a in arrivals) / weight
        row_variance = sum((a[0] / weight) ** 2 * a[3] for a in arrivals)
        _, _, _, _, parent, reply = max(arrivals, key=lambda a: (a[0], a[4], a[5]))
        text, number = lines.get(parent, ('', 1))
        for move, target in children(d).items():
            prepared_node = model.get(target)
            if (
                prepared_node is None
                or prepared_node.mode != 'opponent'
                or target not in evidence
                or values[target][UNKNOWN] != 0
            ):
                continue
            prepared = float(values[target][KNOWN])
            pooled = score(counts(evidence[target]), color)
            if pooled is None:
                continue
            change = prepared - database
            variance = posterior.value_variance({target: 1.0}) + row_variance
            rows.append(
                dict(
                    id=f'{d}|{move}',
                    kind='opponent',
                    position=position_of(parent),
                    move=reply,
                    move_san=san(position_of(parent), reply),
                    line=(text + ' ' + move_text(position_of(parent), number, reply)).strip(),
                    exit_position=d,
                    transposing_move=move,
                    transposing_san=san(d, move),
                    transposition_target=target,
                    target_line=lines.get(target, ('', 0))[0],
                    target_chapters=sorted(graph.nodes[target].chapters, key=lambda c: (order.get(c, len(order)), c)),
                    reach=weight,
                    sample_count=sum(a[2] for a in arrivals),
                    sparse=sum(a[2] for a in arrivals) < sparse_threshold,
                    database_score=database,
                    target_database_score=pooled,
                    repertoire_score=prepared,
                    change_pp=100 * change,
                    move_change_pp=100 * (pooled - database),
                    preparation_change_pp=100 * (prepared - pooled),
                    change_interval_pp=difference_interval(change, variance),
                    weighted_change_pp=100 * weight * change,
                )
            )
    return sorted(rows, key=lambda r: (-abs(r['weighted_change_pp']), r['id']))
