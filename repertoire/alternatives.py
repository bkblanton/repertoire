"""Alternative preparation: where candidate chapters differ from a repertoire, and hypothetical repertoires.

A candidate chapter is followed along its own first moves until it plays a different move from the repertoire.
That board is a decision point. Its options are the repertoire's move (or no preparation) and each distinct
candidate move there; chapters choosing the same move pool their lines, and if they later disagree with each
other, that board is a nested decision point inside the pooled option. A scenario chooses one option at every
decision point: the chosen candidate lines are placed before the repertoire's chapters, every other candidate
line is left out, and the repertoire plays its usual moves elsewhere.

Nothing here reads the network; evidence is supplied by the caller.
"""

import itertools
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import chess.pgn

from .board_cache import canonical, position_of, terminal_white, turn
from .graph import Graph, Policy, Transitions, alternative_transitions, parse_games, reachable
from .schema import Position

if TYPE_CHECKING:
    from .compare import Scenario

# Scenario games carry these chapter IDs, so candidate chapters never collide with the repertoire's own IDs,
# which its configuration uses.
CANDIDATE_PREFIX = 'candidate-'

Reference = str | tuple[str, ...] | None  # the repertoire's move at a board (see repertoire_reference)


@dataclass
class Option:
    move: str | tuple | None  # UCI; None means no preparation; a tuple is a mixed repertoire policy
    chapters: list[str]  # candidate chapters whose lines continue through this move here
    source: str  # 'repertoire' for your own move, 'candidate' for a candidate chapter's


@dataclass
class Decision:
    id: int
    position: str
    options: list[Option]  # options[0] keeps the reference: your move, or the lead chapter's when nested
    parent: tuple[int, int] | None = None  # (decision id, option index) this nested decision lies inside
    nested: list[int] = field(default_factory=list)


def walk(
    graph: Graph,
    color: bool,
    cid: str,
    starts: Sequence[Position],
    reference: Callable[[Position], Reference],
    passed: Mapping[Position, list[str]],
) -> Iterator[tuple[Position, str]]:
    """Follow one chapter's first moves and recorded replies; yield (board, move) where it first differs."""
    seen, pending = set(), list(reversed(starts))
    while pending:
        k = pending.pop()
        if k in seen:
            continue
        seen.add(k)
        node = graph.nodes[k]
        moves = node.chapter_moves.get(cid, [])
        if turn(k) == color and terminal_white(k) is None:
            if not moves:
                continue
            if moves[0] != reference(k):
                yield k, moves[0]
                continue
            passed[k].append(cid)
            pending.append(node.edges[moves[0]])
        else:
            pending.extend(reversed([node.edges[m] for m in moves]))


def find_decisions(graph: Graph, color: bool, reference: Callable[[Position], Reference]) -> list[Decision]:
    """Every decision point of a candidate graph against `reference(board)`, the repertoire's move there."""
    decisions: list[Decision] = []

    def explore(
        chapters: Sequence[str],
        starts: Mapping[str, Sequence[Position]],
        ref: Callable[[Position], Reference],
        parent: tuple[int, int] | None,
        lead: str | None = None,
    ) -> None:
        found: dict[Position, dict[str, list[str]]]
        passed: defaultdict[Position, list[str]]
        found, passed = {}, defaultdict(list)
        for cid in chapters:
            for k, move in walk(graph, color, cid, starts[cid], ref, passed):
                found.setdefault(k, {}).setdefault(move, []).append(cid)
        for k, moves in found.items():
            if any(d.position == k for d in decisions):
                raise ValueError(
                    f'Candidate lines reach the decision at {" ".join(graph.nodes[k].path)} in two ways; '
                    'put competing lines in separate chapters that diverge at one board'
                )
            keeps = ref(k)
            agreeing = ([lead] if lead and graph.nodes[k].chapter_moves.get(lead, [None])[0] == keeps else []) + passed[
                k
            ]
            source = 'candidate' if lead and graph.nodes[k].chapter_moves.get(lead) else 'repertoire'
            decision = Decision(
                len(decisions),
                k,
                [Option(keeps, agreeing, source)] + [Option(m, cids, 'candidate') for m, cids in moves.items()],
                parent,
            )
            decisions.append(decision)
            if parent is not None:
                decisions[parent[0]].nested.append(decision.id)
            for index, option in enumerate(decision.options[1:], 1):
                if len(option.chapters) > 1:
                    # Inside a pooled option the earliest chapter is the reference for the others.
                    first, rest = option.chapters[0], option.chapters[1:]
                    target = graph.nodes[k].edges[cast(str, option.move)]

                    def nested_reference(
                        x: Position, first: str = first, ref: Callable[[Position], Reference] = ref
                    ) -> Reference:
                        recorded = graph.nodes[x].chapter_moves.get(first)
                        return recorded[0] if recorded else ref(x)

                    explore(rest, {c: [target] for c in rest}, nested_reference, (decision.id, index), first)

    explore([c['id'] for c in graph.chapters], {c['id']: [c['root']] for c in graph.chapters}, reference, None, None)
    return decisions


def repertoire_reference(transitions: Transitions) -> Callable[[Position], Reference]:
    """The repertoire's move at a board: one UCI move, None without preparation, or a tuple for a mixed policy."""

    def reference(k: Position) -> Reference:
        selected = transitions.get(k, {})
        if not selected:
            return None
        if len(selected) == 1:
            return next(iter(selected))
        return tuple(sorted(selected))

    return reference


class Choice(dict[int, int]):
    """Decision id -> option index; missing decisions keep option 0."""

    def active(self, decisions: Sequence[Decision]) -> list[Decision]:
        """Decisions that apply: top-level ones, and nested ones whose parent option is chosen."""
        result = []
        for d in decisions:
            parent = d.parent
            while parent is not None and self.get(parent[0], 0) == parent[1]:
                parent = decisions[parent[0]].parent
            if parent is None:
                result.append(d)
        return result

    def key(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted((d, i) for d, i in self.items() if i))


def copy_lines(
    source: chess.pgn.GameNode,
    target: chess.pgn.GameNode,
    board: chess.Board,
    keep: Mapping[Position, tuple[set[str], bool]],
    cid: str,
    inside: bool,
) -> bool:
    """Copy `source`'s variations into `target`, keeping candidate lines the scenario adopts.

    `keep` maps decision boards to (chapters allowed past it, whether those lines are adopted). Outside adopted
    lines only the paths leading to adopted lines are kept. Returns whether anything was kept.
    """
    kept = False
    for child in source.variations:
        board.push(child.move)
        k = canonical(board)
        child_inside = inside
        allowed = True
        if k in keep:
            chapters, adopted = keep[k]
            allowed = cid in chapters
            child_inside = adopted and allowed
        if allowed:
            node = target.add_variation(child.move, comment=child.comment, nags=child.nags)
            node.starting_comment = child.starting_comment
            if copy_lines(child, node, board, keep, cid, child_inside) or child_inside:
                kept = True
            else:
                target.variations.remove(node)
        board.pop()
    return kept


class Scenarios:
    """Hypothetical repertoires: candidate chapters, trimmed to the chosen options, before your own chapters."""

    def __init__(
        self,
        candidate: Graph,
        candidate_games: Sequence[chess.pgn.Game],
        decisions: list[Decision],
        repertoire_games: Sequence[chess.pgn.Game],
        color: bool,
        exclusions: Iterable[str] = (),
    ) -> None:
        self.candidate, self.games, self.decisions = candidate, candidate_games, decisions
        self.repertoire_games, self.color, self.exclusions = repertoire_games, color, list(exclusions)
        self.ids = [c['id'] for c in candidate.chapters]

    def keep(self, choice: Choice) -> dict[Position, tuple[set[str], bool]]:
        """Decision board -> (candidate chapters allowed past it, whether their lines there are adopted)."""
        active = {d.id for d in choice.active(self.decisions)}
        result: dict[Position, tuple[set[str], bool]] = {}
        for d in self.decisions:
            index = choice.get(d.id, 0) if d.id in active else 0
            option = d.options[index]
            # Keeping your move adopts nothing at a top-level board; candidate chapters that agree with it only
            # pass through towards later decisions.
            adopted = d.parent is not None or index > 0
            result[d.position] = (set(option.chapters), adopted and d.id in active)
        return result

    def candidate_games_for(self, choice: Choice, namespaced: bool = True) -> list[chess.pgn.Game]:
        """The candidate chapters trimmed to the chosen options; chapters with nothing adopted are dropped."""
        keep = self.keep(choice)
        result = []
        for cid, game in zip(self.ids, self.games):
            copy = chess.pgn.Game()
            copy.headers.update(game.headers)
            if namespaced:
                copy.headers['ChapterURL'] = CANDIDATE_PREFIX + cid
            copy.setup(game.board())
            copy.comment = game.comment
            if copy_lines(game, copy, game.board(), keep, cid, False):
                result.append(copy)
        return result

    def graph(self, choice: Choice) -> Graph:
        return parse_games([*self.candidate_games_for(choice), *self.repertoire_games], self.exclusions)

    def fixed(self, choice: Choice) -> dict[Position, str]:
        """The move each active decision plays; the scorer must not reconsider these boards."""
        result: dict[Position, str] = {}
        for d in choice.active(self.decisions):
            move = d.options[choice.get(d.id, 0)].move
            if isinstance(move, str):
                result[d.position] = move
        return result

    def superset(self, policy: Policy, roots: Iterable[Position]) -> tuple[Graph, Transitions, list[Position]]:
        """Every board any scenario can reach, with every candidate line included, for planning evidence."""
        graph = parse_games(
            [*(self._full(cid, game) for cid, game in zip(self.ids, self.games)), *self.repertoire_games],
            self.exclusions,
        )
        transitions = alternative_transitions(graph, self.color, policy)
        return graph, transitions, reachable(transitions, [*roots, *graph.roots])

    @staticmethod
    def _full(cid: str, game: chess.pgn.Game) -> chess.pgn.Game:
        copy = chess.pgn.Game()
        copy.headers.update(game.headers)
        copy.headers['ChapterURL'] = CANDIDATE_PREFIX + cid
        copy.setup(game.board())
        copy_lines(game, copy, game.board(), {}, cid, True)
        return copy


def required_tables(
    graph: Graph, transitions: Transitions, positions: Iterable[Position], color: bool
) -> list[Position]:
    """Tables a scenario may need: every opponent turn, and own turns some chapter leaves without a move."""
    result = []
    for k in positions:
        if terminal_white(k) is not None:
            continue
        node = graph.nodes.get(k)
        if turn(k) != color or not transitions[k] or node is None:
            result.append(k)
        elif any(not node.chapter_moves.get(c) for c in node.chapters):
            result.append(k)
    return result


def descendants(transitions: Transitions, starts: Iterable[Position]) -> set[Position]:
    return set(reachable(transitions, [k for k in starts if k in transitions]))


def changed(scenario: 'Scenario', baseline: 'Scenario') -> set[Position]:
    """Boards whose moves or replies differ between two evaluated scenarios."""
    keys = set(scenario.transitions) | set(baseline.transitions)
    return {position_of(k) for k in keys if scenario.transitions.get(k) != baseline.transitions.get(k)}


def groups(
    decisions: Sequence[Decision], footprints: Mapping[int, set[Position]], transitions: Transitions
) -> list[list[int]]:
    """Decision points whose options can affect each other, as lists of ids.

    `footprints[d]` is the union over d's options of the boards that option changes. Two decisions are linked
    when one's changes lie below the other's (a transposition), and nested decisions join their parents.
    """
    parent = list(range(len(decisions)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    below = {d.id: descendants(transitions, footprints[d.id]) for d in decisions}
    for d in decisions:
        if d.parent is not None:
            union(d.id, d.parent[0])
    for a, b in itertools.combinations(decisions, 2):
        if footprints[a.id] & below[b.id] or footprints[b.id] & below[a.id]:
            union(a.id, b.id)
    result: defaultdict[int, list[int]] = defaultdict(list)
    for d in decisions:
        result[find(d.id)].append(d.id)
    return sorted(result.values())


def assignments(
    decisions: Sequence[Decision], members: Iterable[Any], allowed: Callable[[Decision], Iterable[int]]
) -> Iterator[Choice]:
    """Every valid choice within a group: nested decisions vary only when their parent option is chosen."""
    members = [decisions[i] for i in members]
    top = [d for d in members if d.parent is None or d.parent[0] not in {m.id for m in members}]

    def expand(pending: list[Decision], choice: dict[int, int]) -> Iterator[Choice]:
        if not pending:
            yield Choice(choice)
            return
        d, rest = pending[0], pending[1:]
        for index in allowed(d):
            nested = [decisions[n] for n in d.nested] if index else []
            nested = [n for n in nested if n.parent == (d.id, index)]
            yield from expand(nested + rest, {**choice, d.id: index})

    yield from expand(top, {})


def count_assignments(
    decisions: Sequence[Decision], members: Iterable[int], allowed: Callable[[Decision], Iterable[int]], limit: int
) -> int:
    count = 0
    for _ in assignments(decisions, members, allowed):
        count += 1
        if count > limit:
            return count
    return count
