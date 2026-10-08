"""Directed chapter reach after first entering another chapter."""

from collections.abc import Collection, Iterable, Mapping
from typing import cast

from .model import Model, Sampled
from .schema import Chapter, JsonObject, Position
from .status import Status


def hitting_bounds(
    model: Model, order: Iterable[Position], sampled: Sampled, destination: Collection[Position]
) -> dict[Position, tuple[float, float]]:
    """Probability of hitting a destination before stopping, from each node."""
    values: dict[Position, tuple[float, float]] = {}
    can_enter: dict[Position, bool] = {}
    for k in order:
        node = model[k]
        targets = [b.target for b in node.branches if b.target is not None] + node.potential_targets
        can_enter[k] = k in destination or any(can_enter[t] for t in targets)
        if k in destination:
            values[k] = (1.0, 1.0)
            continue
        low = high = 0.0
        for branch, (probability, _) in zip(node.branches, sampled[k]):
            if branch.target is not None:
                lower, upper = values[branch.target]
            else:
                lower = 0.0
                upper = float(branch.kind == 'unresolved_distribution' and can_enter[k])
            low += probability * lower
            high += probability * upper
        values[k] = (low, high)
    return values


def chapter_transitions(
    model: Model,
    order: Iterable[Position],
    sampled: Sampled,
    chapters: Iterable[Chapter],
    destinations: Mapping[str, Collection[Position]],
) -> list[JsonObject]:
    """Condition on source first arrival; include simultaneous destination entry."""
    rows: list[JsonObject] = []
    for target in chapters:
        cid = target['id']
        if not destinations[cid]:
            continue
        hits = hitting_bounds(model, order, sampled, set(destinations[cid]))
        for source in chapters:
            if source['id'] == cid:
                continue
            score = source['score']
            reach = score.get('entry_probability')
            weights = score.get('first_entry_weights', {})
            row: JsonObject = {
                'source_id': source['id'],
                'source_name': source['name'],
                'destination_id': cid,
                'destination_name': target['name'],
                'conditional_probability': None,
                'joint_probability': None,
            }
            if reach is not None and reach > 0 and weights and all(w is not None for w in weights.values()):
                low = sum(cast(float, w) * hits[k][0] for k, w in weights.items())
                high = sum(cast(float, w) * hits[k][1] for k, w in weights.items())
                row['conditional_bounds'] = [low, high]
                if high == low:
                    row.update(conditional_probability=low, joint_probability=reach * low, status=Status.RESOLVED)
                else:
                    row['status'] = Status.UNRESOLVED_DESTINATION_REACH
            else:
                row['status'] = Status.UNRESOLVED_SOURCE_REACH
            rows.append(row)
    return rows
