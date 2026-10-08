"""Expected remaining repertoire-owner moves, with no discount or depth cutoff."""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

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


def summarize_depth(values: Mapping[Position, tuple[float, float]], weights: Mapping[Position, float]) -> PreparedDepth:
    if not weights or any(w is None or not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError('Prepared depth requires nonnegative, known starting weights')
    if not math.isclose(sum(weights.values()), 1, abs_tol=1e-9):
        raise ValueError('Prepared depth starting weights must sum to one')
    low = sum(w * values[k][0] for k, w in weights.items())
    high = sum(w * values[k][1] for k, w in weights.items())
    return {
        'expected_moves': low if low == high else None,
        'conditional_bounds': [low, high],
        'unit': 'own_moves',
        'status': Status.RESOLVED if low == high else Status.UNRESOLVED_MOVE_DISTRIBUTION,
    }


def chapter_prepared_depth(
    values: Mapping[Position, tuple[float, float]], entries: Sequence[Position], chapter_score: ChapterScore
) -> PreparedDepth:
    if not entries:
        return {'expected_moves': None, 'unit': 'own_moves', 'status': Status.ENTRY_CONFIGURATION_REQUIRED}
    # Matches the existing single-position conditional-score semantics even
    # when its root reach is zero or unknown.
    if len(entries) == 1:
        return summarize_depth(values, {entries[0]: 1.0})
    weights = chapter_score.get('first_entry_weights', {})
    if any(weights.get(k) is None for k in entries):
        return {
            'expected_moves': None,
            'unit': 'own_moves',
            'status': Status.UNRESOLVED_ENTRY_WEIGHTS,
            'conditional_bounds': [min(values[k][0] for k in entries), max(values[k][1] for k in entries)],
        }
    return summarize_depth(values, {k: cast(float, weights[k]) for k in entries})
