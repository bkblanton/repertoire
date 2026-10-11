"""Where a scope's edge over the database score comes from, in parts that add up exactly.

The edge is the repertoire score minus the baseline, the database score where the scope starts. With reach r
and database scores S, it splits into:

- **Your decisions:** for each prepared move, r(position) x (S after the move - S of the position). Unlike the
  gains elsewhere in the reports, these do not overlap: each counts only its own move.
- **Move orders at transpositions:** a prepared position's database score pools the opponent moves leading to
  it by game count, while the repertoire arrives through each move order in its own proportion. Each arrival
  adds its reach x (pooled score - score of its own move row).
- **Theory leaves:** a leaf is scored with its own table, which counts every move order; the move into it is
  compared with its row in the parent table.
- **Finished games:** checkmate, stalemate and insufficient material are scored exactly rather than with the
  database results of the move that reached them.

The terms telescope: at an opponent position, the database score is the probability-weighted average of its
move rows, so V(t) - S(t) is the sum over prepared replies of p x (V(after) - S(row)).
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import cast

import numpy as np

from .board_cache import move_text, owner_outcome, position_of
from .evaluate import KNOWN, UNKNOWN
from .explorer import counts
from .model import Evidence, Model, Sampled, position_counts, score
from .schema import JsonObject, Position
from .status import Status

# Own move numbers grouped as the opening choice, the early moves, the middle of a line and deep preparation.
MOVE_GROUPS = (('1', 1, 1), ('2-3', 2, 3), ('4-6', 4, 6), ('7+', 7, 10_000))
TOLERANCE = 1e-9


def start_score(k: Position, evidence: Evidence, color: bool) -> float | None:
    """The baseline at a starting position: its own table, as in the start and entry baselines."""
    k = position_of(k)
    fixed = owner_outcome(k, color)
    if fixed is not None:
        return fixed
    return score(counts(evidence[k]), color) if k in evidence else None


def ledger(
    model: Model,
    sampled: Sampled,
    values: Mapping[Position, np.ndarray],
    reach: Mapping[Position, float],
    starts: Mapping[Position, float],
    evidence: Evidence,
    arrivals: dict[Position, list[int]],
    color: bool,
    lines: Mapping[Position, tuple[str, int]],
) -> JsonObject:
    """The edge of one scope split into the terms above, in percentage points. `starts` are the scope's starting
    weights (the roots, or a chapter's first-entry weights); decision terms are per own move, keyed like the
    vulnerability rows (position|move)."""
    starting = {k: start_score(k, evidence, color) for k, w in starts.items() if w}
    reached = [k for k, r in reach.items() if r > 0]
    if any(s is None for s in starting.values()) or any(values[k][UNKNOWN] != 0 for k in reached):
        return dict(status=Status.UNRESOLVED_EVIDENCE)
    baseline = sum(starts[k] * float(s) for k, s in starting.items() if s is not None)
    total = sum(starts[k] * float(values[k][KNOWN]) for k in starting) - baseline
    decisions: dict[str, float] = {}
    finished = 0.0
    # Probability arriving at each position with the database score of the move row (or start) it came by.
    inflow: defaultdict[Position, list[tuple[float, float, Position | None, str | None]]] = defaultdict(list)
    for k, s in starting.items():
        inflow[k].append((starts[k], float(cast(float, s)), None, None))
    for k in reached:
        node, r = model[k], reach[k]
        if node.mode == 'own':
            reference = score(position_counts(arrivals, evidence, position_of(k)) or [0, 0, 0], color)
            for b, (p, _) in zip(node.branches, sampled[k]):
                if p <= 0 or reference is None:
                    continue
                if b.target is None:
                    # A move into a third occurrence of a position: a draw by repetition.
                    finished += r * p * (cast(float, b.fixed_score) - reference)
                    continue
                target = model.get(b.target)
                if target is not None and target.mode == 'opponent' and position_of(b.target) in evidence:
                    move = score(counts(evidence[position_of(b.target)]), color)
                    decisions[f'{k}|{b.move}'] = 100 * r * p * (float(move) - reference) if move is not None else 0.0
                else:
                    finished += r * p * (float(values[b.target][KNOWN]) - reference)
        elif node.mode == 'opponent':
            for b, (p, s) in zip(node.branches, sampled[k]):
                if p <= 0:
                    continue
                if b.target is not None:
                    inflow[b.target].append((r * p, float(cast(float, s)), k, b.move))
                elif b.fixed_score is not None:
                    row = score(b.counts, color)
                    finished += r * p * (b.fixed_score - row) if row is not None else 0.0
    move_orders, leaves = 0.0, 0.0
    transpositions: list[JsonObject] = []
    for u, arriving in inflow.items():
        node = model[u]
        if node.mode == 'own':
            pooled = score(position_counts(arrivals, evidence, position_of(u)) or [0, 0, 0], color)
            if pooled is None:
                continue
            effect = sum(w * (pooled - s) for w, s, _, _ in arriving)
            move_orders += effect
            if abs(effect) > 1e-15:
                transpositions.append(
                    dict(
                        position=position_of(u),
                        line=lines.get(u, ('', 0))[0],
                        effect_pp=100 * effect,
                        database_score=pooled,
                        arrivals=[arrival(parent, move, w, s, lines) for w, s, parent, move in arriving],
                    )
                )
        elif node.mode == 'stop':
            term = sum(w * (float(values[u][KNOWN]) - s) for w, s, _, _ in arriving)
            if node.branches[0].fixed_score is None:
                leaves += term
            else:
                finished += term
    result: JsonObject = dict(
        status=Status.RESOLVED,
        baseline_score=baseline,
        delta_pp=100 * total,
        decisions_pp=sum(decisions.values()),
        move_orders_pp=100 * move_orders,
        theory_leaves_pp=100 * leaves,
        finished_games_pp=100 * finished,
        decisions=decisions,
        transpositions=sorted(transpositions, key=lambda t: (-abs(t['effect_pp']), t['position'])),
    )
    parts = sum(result[k] for k in ('decisions_pp', 'move_orders_pp', 'theory_leaves_pp', 'finished_games_pp'))
    if not np.isclose(parts, result['delta_pp'], atol=TOLERANCE, rtol=0):
        raise AssertionError('Edge ledger does not reproduce the delta')
    return result


def arrival(
    parent: Position | None,
    move: str | None,
    weight: float,
    row_score: float,
    lines: Mapping[Position, tuple[str, int]],
) -> JsonObject:
    if parent is None or move is None:
        return dict(parent_position=None, move=None, line='', reach=weight, score=row_score)
    text, number = lines.get(parent, ('', 1))
    return dict(
        parent_position=position_of(parent),
        move=move,
        line=(text + ' ' + move_text(position_of(parent), number, move)).strip(),
        reach=weight,
        score=row_score,
    )


def move_number(label: str) -> int:
    """The full-move number of a move label such as '14.c3' or '8...e5'."""
    return int(label.split('.', 1)[0])


def summarize(result: JsonObject, rows: Iterable[JsonObject], top: int = 10) -> JsonObject:
    """Attach each own move's edge to its vulnerability row and add the groupings the reports show."""
    if result.get('status') != Status.RESOLVED:
        return result
    decisions = result.pop('decisions')
    own = [r for r in rows if r['kind'] == 'own']
    for row in own:
        row['edge_pp'] = decisions.get(row['id'])
    counted = [r for r in own if r.get('edge_pp') is not None]
    if not np.isclose(sum(r['edge_pp'] for r in counted), result['decisions_pp'], atol=TOLERANCE, rtol=0):
        raise AssertionError('Edge ledger rows do not add up to the decision total')
    groups = {name: 0.0 for name, _, _ in MOVE_GROUPS}
    for row in counted:
        number = move_number(row['move_label'])
        name = next(name for name, low, high in MOVE_GROUPS if low <= number <= high)
        groups[name] += row['edge_pp']
    ranked = sorted((r['edge_pp'] for r in counted), reverse=True)
    result.update(
        decision_count=len(counted),
        gaining_count=sum(e > 0 for e in ranked),
        costing_count=sum(e < 0 for e in ranked),
        gains_pp=sum(e for e in ranked if e > 0),
        costs_pp=sum(e for e in ranked if e < 0),
        top_decisions=top,
        top_decisions_pp=sum(ranked[:top]),
        by_move_number=[dict(moves=name, edge_pp=groups[name]) for name, _, _ in MOVE_GROUPS],
    )
    return result
