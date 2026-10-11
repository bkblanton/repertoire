"""What each of your decisions is worth against what it takes to know.

- **Dropping a decision** leaves preparation at that position, so it loses reach x total gain: the repertoire
  score after the move minus the database score of the position. It also removes every decision that can only be
  reached through it, the positions it dominates, so its cost in memorized moves is itself plus those.
- **Rarely practised moves** come up too seldom to stay fresh through play. A move reached with probability r is
  missing from the last 100 games with probability (1 - r) ** 100; the review list ranks reach x gain by that
  chance, so it favors valuable moves met about once per hundred games.

Values are in percentage points per game with that color; reports show them per 1,000 games.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from .evaluate import Weights, dominators
from .model import Model, Sampled
from .schema import JsonObject, Position

# Games looked back over for the review list, matching the reuse curve's 100-game point.
RECENT_GAMES = 100
# Moves met less often than this per game count as rare in the summary.
RARE_REACH = 0.001
# A pruning candidate takes at least this many decisions; single moves are the vulnerability tables' job.
PRUNE_MIN_DECISIONS = 3


def row_node(row: JsonObject) -> Position:
    """A vulnerability row's scoring node: its position, or inside a repetition loop the occurrence it names."""
    return cast(Position, row.get('node', row['position']))


def target_node(row: JsonObject) -> Position | None:
    """The scoring node after a vulnerability row's move, or None where the move ends the game."""
    return cast(Position | None, row.get('target_node', row['target']))


def supported(row: JsonObject) -> bool:
    """Comparisons resting on enough games at the move, its position and the position after it."""
    return not any(row.get(f, False) for f in ('sparse', 'parent_sparse', 'continuation_endpoint_sparse'))


def subtrees(idom: Mapping[Position, Position | None], own: Iterable[Position]) -> dict[Position, set[Position]]:
    """The own decisions each own decision dominates: those only reachable through it, itself included."""
    own = set(own)
    result: defaultdict[Position, set[Position]] = defaultdict(set)
    for k in own:
        node: Position | None = k if k in idom else None
        while node is not None:
            if node in own:
                result[node].add(k)
            node = idom[node]
    return dict(result)


def effort(
    model: Model,
    order: list[Position],
    sampled: Sampled,
    roots: Weights,
    rows: Sequence[JsonObject],
    ledger: JsonObject,
) -> JsonObject:
    """Attach each own move's drop value and cost to its row, and rank pruning and review candidates by row id."""
    idom = dominators(model, order, sampled, roots)
    own_rows = [r for r in rows if r['kind'] == 'own']
    dominated = subtrees(idom, (row_node(r) for r in own_rows))
    for row in own_rows:
        gain = row.get('local_gain_pp')
        row['decisions_dropped'] = len(dominated.get(row_node(row), ()))
        row['drop_value_pp'] = None if gain is None else row['branch_reach'] * gain
        row['review_priority_pp'] = (
            None if gain is None else row['branch_reach'] * gain * (1 - row['branch_reach']) ** RECENT_GAMES
        )
    candidates = sorted(
        (
            r
            for r in own_rows
            if supported(r) and r['drop_value_pp'] is not None and r['drop_value_pp'] > 0
            if r['decisions_dropped'] >= PRUNE_MIN_DECISIONS
        ),
        key=lambda r: (r['drop_value_pp'] / r['decisions_dropped'], r['id']),
    )
    # Dominated sets are nested or disjoint; keep the lowest-value set from each nest.
    pruning: list[str] = []
    taken: set[Position] = set()
    for row in candidates:
        members = dominated[row_node(row)]
        if not members & taken:
            pruning.append(row['id'])
            taken |= members
    review = [
        r['id']
        for r in sorted(
            (
                r
                for r in own_rows
                if supported(r) and r['review_priority_pp'] is not None and r['review_priority_pp'] > 0
            ),
            key=lambda r: (-r['review_priority_pp'], r['id']),
        )
    ]
    rare = [r for r in own_rows if r.get('edge_pp') is not None and r['branch_reach'] < RARE_REACH]
    return dict(
        recent_games=RECENT_GAMES,
        rare_reach=RARE_REACH,
        rare_decisions=len(rare),
        decisions=sum(r.get('edge_pp') is not None for r in own_rows),
        rare_edge_pp=sum(r['edge_pp'] for r in rare),
        delta_pp=ledger.get('delta_pp'),
        prune_min_decisions=PRUNE_MIN_DECISIONS,
        pruning=pruning,
        review=review,
    )


def chapter_efforts(chapters: Sequence[JsonObject], rows: Sequence[JsonObject]) -> list[JsonObject]:
    """Every chapter's effort row; needs the rows' chapter attribution."""
    own_rows = [r for r in rows if r['kind'] == 'own']
    return [chapter_effort(c, own_rows) for c in chapters]


def chapter_effort(chapter: JsonObject, own_rows: Iterable[JsonObject]) -> JsonObject:
    """A chapter's edge over all games with the color, and per move it records that games reach. A move recorded
    in several chapters counts in each, as you study it in each."""
    ledger = chapter.get('edge') or {}
    decisions = sum(chapter['id'] in (r.get('chapter_attribution') or {}).get('source_ids', []) for r in own_rows)
    probability, delta = chapter.get('entry_probability'), ledger.get('delta_pp')
    edge = None if probability is None or delta is None else probability * delta
    return dict(
        id=chapter['id'],
        decisions=decisions,
        edge_pp=edge,
        edge_per_decision_pp=None if edge is None or not decisions else edge / decisions,
    )
