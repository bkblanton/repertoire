"""Expected remaining repertoire-owner moves, with no discount or depth cutoff.

Prepared depth is reported in opponent moves answered: own prepared moves played in answer to an opponent move.
An own move at a repertoire root (the starting position, or a custom-FEN chapter root) answers nothing, so it is
left out; otherwise White would gain a free first move that Black never can. A chapter or opening entered on your
move still counts that move, which answers the opponent move before the entry. The raw count is kept beside it.
"""

import math
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import cast

from .board_cache import position_of
from .model import Model, Sampled
from .schema import ChapterScore, Position, PreparedDepth
from .status import Status

DepthBounds = dict[Position, tuple[float, float]]  # lower and upper expected remaining own moves


def prepared_depth_values(model: Model, order: Iterable[Position], sampled: Sampled) -> DepthBounds:
    """Return lower/upper expected-depth bounds for each canonical position.

    Our selected move earns one unit. Opponent branches weight continuation
    depth, and deviations and leaves earn zero. Missing outcome counts at a
    leaf do not affect depth. Missing move distributions retain finite bounds
    from the longest remaining prepared continuation instead of assuming zero.
    """
    values: DepthBounds = {}
    longest: dict[Position, float] = {}
    for k in order:
        node = model[k]
        targets = [b.target for b in node.branches if b.target is not None] + node.potential_targets
        reward = float(node.mode == 'own')
        longest[k] = reward + max((longest[t] for t in targets), default=0.0)
        if any(b.kind == 'unresolved_distribution' for b in node.branches):
            values[k] = (0.0, longest[k])
            continue
        low = high = reward
        for branch, (probability, _) in zip(node.branches, sampled[k]):
            if branch.target is not None:
                low += probability * values[branch.target][0]
                high += probability * values[branch.target][1]
        values[k] = (low, high)
    return values


def root_moves(model: Model, roots: Collection[Position]) -> set[Position]:
    """Root nodes where a prepared own move is played; that move answers no opponent move."""
    return {k for k, node in model.items() if node.mode == 'own' and position_of(k) in roots}


def summarize_depth(
    values: Mapping[Position, tuple[float, float]], weights: Mapping[Position, float], own: Collection[Position]
) -> PreparedDepth:
    """Expected depth from `weights`; `own` holds the starts whose first move is not an answer (see root_moves)."""
    if not weights or any(w is None or not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError('Prepared depth requires nonnegative, known starting weights')
    if not math.isclose(sum(weights.values()), 1, abs_tol=1e-9):
        raise ValueError('Prepared depth starting weights must sum to one')
    low = sum(w * values[k][0] for k, w in weights.items())
    high = sum(w * values[k][1] for k, w in weights.items())
    first = sum(w for k, w in weights.items() if k in own)
    return {
        'expected_answers': low - first if low == high else None,
        'answer_bounds': [low - first, high - first],
        'expected_moves': low if low == high else None,
        'conditional_bounds': [low, high],
        'first_move_share': first,
        'unit': 'own_moves',
        'status': Status.RESOLVED if low == high else Status.UNRESOLVED_MOVE_DISTRIBUTION,
    }


def chapter_prepared_depth(
    values: Mapping[Position, tuple[float, float]],
    entries: Sequence[Position],
    chapter_score: ChapterScore,
    own: Collection[Position],
    arrivals: Callable[[Mapping[Position, float]], Mapping[Position, float]] = lambda weights: weights,
) -> PreparedDepth:
    """Depth from a chapter's entries; `arrivals` spreads entry weights over the nodes games arrive at (see
    evaluate.entry_states)."""
    if not entries:
        return {
            'expected_answers': None,
            'expected_moves': None,
            'unit': 'own_moves',
            'status': Status.ENTRY_CONFIGURATION_REQUIRED,
        }
    # Matches the existing single-position conditional-score semantics even
    # when its root reach is zero or unknown.
    if len(entries) == 1:
        return summarize_depth(values, arrivals({entries[0]: 1.0}), own)
    weights = chapter_score.get('first_entry_weights', {})
    if any(weights.get(k) is None for k in entries):
        return {
            'expected_answers': None,
            'expected_moves': None,
            'unit': 'own_moves',
            'status': Status.UNRESOLVED_ENTRY_WEIGHTS,
            'answer_bounds': [
                min(values[k][0] - (k in own) for k in entries),
                max(values[k][1] - (k in own) for k in entries),
            ],
            'conditional_bounds': [min(values[k][0] for k in entries), max(values[k][1] for k in entries)],
        }
    return summarize_depth(values, arrivals({k: cast(float, weights[k]) for k in entries}), own)
