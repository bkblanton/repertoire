"""Coverage goals: learn the moves for positions that come up often enough, and leave rarer ones to the database.

A goal of 1 in N games keeps every position where you are to move that games reach at least once in N games,
counting only routes through kept positions (evaluate.coverage), together with the moves you play there. A position
reached less often is cut: the opponent's move into it becomes an unprepared reply, scored with its row in the
parent table like any other, so the trimmed repertoire needs no new tables. It is scored exactly like the full
repertoire, and the cost of each cut, the games arriving there times your score there minus the reply's database
score, adds up to the change.

The stage also writes the trimmed repertoire at your goal as a PGN made from your own export, keeping its chapters,
comments and annotations. Your studies, the export and every other analysis keep the full repertoire.
"""

import argparse
import io
import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import chess
import chess.pgn
import numpy as np

from .board_cache import canonical, fen_number, position_of, route_line, turn
from .context import DEFAULT_CACHE, AnalysisContext, write_companion
from .evaluate import (
    COMPLETED,
    KNOWN,
    UNKNOWN,
    Weights,
    backward,
    best_routes,
    by_position,
    coverage,
    stopping_vector,
    summarize,
)
from .graph import Transitions, chapter_ids, parse_games, read_games, resolve, topology
from .layout import report_directory
from .model import Model, Sampled, empirical, prepare
from .schema import JsonObject, Position
from .score import CONFIG, COVERAGE_KEY, load_config
from .uncertainty import Posterior, difference_interval, paired_variance

CONFIG_KEY = COVERAGE_KEY
DEFAULT_GOAL = 300
# The levels every report compares, as N in "1 in N games"; your own goal is added when it is not one of them.
GOALS = (100, 150, 200, 300, 400, 600, 800, 1000)
TRIMMED_DIRECTORY = 'trimmed'


def goal_from_config(config: Mapping[str, object]) -> int:
    """The coverage goal N of a configuration, 1 in N games, defaulting to 1 in 300."""
    value = config.get(CONFIG_KEY, DEFAULT_GOAL)
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(f'{CONFIG_KEY} must be a whole number of games, at least 2 (1 in N games): {value!r}')
    return value


def pgn_name(color: str, goal: int) -> str:
    return f'{color}-1-in-{goal}.pgn'


def pgn_path(score_path: str | Path, color: str, goal: int) -> Path:
    """Where the trimmed PGN goes: beside the reports, in their trimmed folder."""
    return report_directory(score_path) / TRIMMED_DIRECTORY / pgn_name(color, goal)


def trim(transitions: Transitions, cut: Collection[Position]) -> Transitions:
    """The repertoire without the cut nodes: an opponent move into one becomes an unprepared reply."""
    return {
        k: {m: (t, w) for m, (t, w) in moves.items() if t not in cut}
        for k, moves in transitions.items()
        if k not in cut
    }


@dataclass
class Repertoire:
    """One policy's scored model, from the repertoire roots."""

    transitions: Transitions
    order: list[Position]
    model: Model
    raw: Sampled
    values: dict[Position, np.ndarray]
    posterior: Posterior

    @property
    def own(self) -> list[Position]:
        return [k for k in self.order if self.model[k].mode == 'own']


@dataclass
class Level:
    goal: int | None
    reach: dict[Position, float]
    cut: set[Position]
    unknown: set[Position]
    repertoire: Repertoire
    root: np.ndarray
    # Per cut node: the change in its score vector, and the unresolved probability the cut removes (lost) or adds
    # through replies with no games (added).
    costs: dict[Position, np.ndarray]
    lost: dict[Position, float]
    added: dict[Position, float]

    def change(self) -> tuple[float | None, list[float]]:
        """The score change in points: exact when no unresolved evidence moves, otherwise its bounds."""
        known = -100 * float(sum((c[KNOWN] for c in self.costs.values()), 0.0))
        lost, added = 100 * sum(self.lost.values(), 0.0), 100 * sum(self.added.values(), 0.0)
        return (known if not lost and not added else None), [known - lost, known + added]


class Coverage:
    """The full repertoire of a saved score and its trimmed versions at each goal."""

    def __init__(self, analysis: AnalysisContext, cache: str | Path) -> None:
        self.analysis = analysis
        self.graph, self.color, self.roots = analysis.graph, analysis.color, cast(Weights, analysis.roots)
        self.sparse, self.prior = analysis.sparse_threshold, analysis.manifest['prior']
        self.evidence = analysis.read_evidence(cache, list(analysis.manifest['evidence']), required=True)
        self.full = self.repertoire(resolve(self.graph, self.color, analysis.policy))
        root = self.root_vector(self.full)
        saved = analysis.saved['overall']
        if not np.allclose(
            root[[KNOWN, UNKNOWN]], [saved['resolved_contribution'], saved['unresolved_mass']], atol=1e-12, rtol=0
        ):
            raise AssertionError('Reconstructed model differs from saved scores')
        self.reach, _, self.unknown = coverage(self.full.model, self.full.order, self.full.raw, self.roots, 0.0)
        # Your moves games reach: one per position, however many occurrences a repetition loop gives it.
        self.decisions = {position_of(k) for k in self.full.own if self.reach[k] > 0 or k in self.unknown}
        self.gradient = self.full.posterior.gradient(self.roots)

    def repertoire(self, transitions: Transitions) -> Repertoire:
        order = topology(transitions, list(self.roots))
        model = prepare(self.graph, transitions, order, self.color, self.evidence)
        raw = empirical(model, self.color)
        values = backward(model, order, raw, self.sparse)
        posterior = Posterior(model, order, self.color, self.prior, self.sparse)
        return Repertoire(transitions, order, model, raw, values, posterior)

    def root_vector(self, repertoire: Repertoire) -> np.ndarray:
        return cast(np.ndarray, sum(w * repertoire.values[k] for k, w in self.roots.items()))

    def level(self, goal: int | None) -> Level:
        reach, cut, unknown = coverage(
            self.full.model, self.full.order, self.full.raw, self.roots, 1 / goal if goal else 0.0
        )
        trimmed = self.repertoire(trim(self.full.transitions, cut)) if cut else self.full
        costs: dict[Position, np.ndarray] = {}
        lost: dict[Position, float] = {}
        added: dict[Position, float] = {}
        # Each opponent move into a cut node is now an unprepared reply; it costs the games arriving that way times
        # your score at the node minus the reply's own result.
        for k in trimmed.order:
            for b, (p, s) in zip(trimmed.model[k].branches, trimmed.raw[k]):
                if b.target is not None or b.move is None or not p:
                    continue
                target = self.full.transitions[k].get(b.move, (None, None))[0]
                if target in cut:
                    stop = stopping_vector(b, s, self.sparse)
                    change = reach[k] * p * (self.full.values[target] - stop)
                    costs[target] = costs[target] + change if target in costs else change
                    lost[target] = lost.get(target, 0.0) + reach[k] * p * float(self.full.values[target][UNKNOWN])
                    added[target] = added.get(target, 0.0) + reach[k] * p * float(stop[UNKNOWN])
        root = self.root_vector(trimmed)
        total = sum(costs.values(), np.zeros(8))
        if not np.allclose(self.root_vector(self.full) - root, total, atol=1e-9):
            raise AssertionError('Coverage cut costs do not reproduce the score change')
        return Level(goal, reach, cut, unknown, trimmed, root, costs, lost, added)

    def kept(self, level: Level) -> set[Position]:
        """Your moves the level keeps, among those games reach."""
        return {position_of(k) for k in level.repertoire.own} & self.decisions

    def row(self, level: Level) -> JsonObject:
        """One goal's line in the comparison: moves kept, games leaving earlier, and the score."""
        kept = self.kept(level)
        posterior = level.repertoire.posterior
        gradient = posterior.gradient(self.roots)
        mean = gradient['mean'][COMPLETED] - self.gradient['mean'][COMPLETED]
        change, bounds = level.change()
        return dict(
            goal=level.goal,
            threshold=1 / level.goal if level.goal else 0.0,
            decisions=len(kept),
            dropped=len(self.decisions - kept),
            leaving_earlier=float(sum((level.reach[k] for k in level.cut), 0.0)),
            unknown_reach_decisions=len({position_of(k) for k in level.unknown} & kept),
            score=summarize(level.root, posterior.mixture(self.roots)),
            change_pp=change,
            change_bounds_pp=bounds,
            change_interval_pp=[
                float(v)
                for v in difference_interval(
                    mean, paired_variance(self.full.posterior, self.gradient, posterior, gradient)
                )
            ],
        )

    def lines(self) -> dict[Position, tuple[str, list[str]]]:
        """One representative route to each node of the full repertoire: its move text and its moves."""
        result = {}
        for k, (_, root, moves) in best_routes(self.full.model, self.full.order, self.full.raw, self.roots).items():
            start = position_of(root)
            text, _, _ = route_line(start, fen_number(self.graph.nodes[start].fen), moves)
            result[k] = text, [root, *moves]
        return result

    def chapters_of(self, position: Position) -> list[str]:
        """The chapters recording a position, in chapter order."""
        recorded = self.graph.nodes[position].chapters
        return [c['id'] for c in self.graph.chapters if c['id'] in recorded]

    def details(self, level: Level) -> JsonObject:
        """At your goal: every cut and every move it drops, each move counted under the first cut on its most
        common route, so both add up to the totals."""
        routes = self.lines()
        full_reach = by_position(self.reach)
        frontier: dict[Position, Position] = {}
        dropped_rows = []
        nodes: dict[Position, Position] = {}
        for k in self.full.own:
            p = position_of(k)
            if p in self.decisions and (p not in nodes or self.reach[k] > self.reach[nodes[p]]):
                nodes[p] = k
        for p in sorted(self.decisions - self.kept(level)):
            text, (root, *moves) = routes[nodes[p]]
            node, first = root, None
            for move in moves:
                node = next(cast(Position, b.target) for b in self.full.model[node].branches if b.move == move)
                if node in level.cut:
                    first = position_of(node)
                    break
            if first is None:
                raise AssertionError('A dropped move has no cut on its route')
            frontier[p] = first
            dropped_rows.append(dict(position=p, line=text, reach=full_reach[p], cut=first))
        cut_reach: dict[Position, float] = {}
        cut_costs: dict[Position, float] = {}
        cut_nodes: dict[Position, Position] = {}
        lost: dict[Position, float] = {}
        added: dict[Position, float] = {}
        for k in level.cut:
            p = position_of(k)
            if level.reach[k] > 0:
                cut_reach[p] = cut_reach.get(p, 0.0) + level.reach[k]
                cut_nodes.setdefault(p, k)
            if k in level.costs:
                cut_costs[p] = cut_costs.get(p, 0.0) + 100 * float(level.costs[k][KNOWN])
                lost[p] = lost.get(p, 0.0) + 100 * level.lost[k]
                added[p] = added.get(p, 0.0) + 100 * level.added[k]
        counted: dict[Position, int] = {}
        for first in frontier.values():
            counted[first] = counted.get(first, 0) + 1
        cuts = []
        for p in sorted(cut_reach, key=lambda p: (-cut_costs.get(p, 0.0), p)):
            cost = cut_costs.get(p, 0.0)
            # A cut costs your score there minus the reply's result: removing unresolved games below it can only
            # raise the cost, and a reply with no games of its own can only lower it.
            exact = not lost.get(p) and not added.get(p)
            cuts.append(
                dict(
                    position=p,
                    line=routes[cut_nodes[p]][0],
                    reach=cut_reach[p],
                    full_reach=full_reach[p],
                    decisions_dropped=counted.get(p, 0),
                    cost_pp=cost if exact else None,
                    cost_bounds_pp=[cost - added.get(p, 0.0), cost + lost.get(p, 0.0)],
                    chapter_attribution=dict(source_ids=self.chapters_of(p)),
                )
            )
        kept = self.kept(level)
        chapters = []
        for c in self.graph.chapters:
            recorded = {p for p in self.decisions if c['id'] in self.graph.nodes[p].chapters}
            chapters.append(dict(id=c['id'], decisions=len(recorded), kept=len(recorded & kept)))
        # Recorded moves you never play: alternatives that lost, and moves at positions games never reach.
        played = {
            (position_of(k), m)
            for k in self.full.own
            if position_of(k) in self.decisions
            for m in self.full.transitions[k]
        }
        own_moves = {(k, m) for k, n in self.graph.nodes.items() if turn(k) == self.color for m in n.edges}
        return dict(
            cuts=cuts,
            dropped=dropped_rows,
            chapters=chapters,
            never_reached=len(own_moves - played),
        )


class Writer:
    """The trimmed repertoire as PGN, made from the original chapters.

    Each chapter keeps the moves you play at kept positions and the opponent moves into kept positions, with their
    comments and annotations. A kept position whose recorded line runs only through a cut one is attached where
    games still reach it, so no kept move is lost. Chapters that add nothing to the chapters before them are left out.
    """

    def __init__(self, coverage: Coverage, level: Level) -> None:
        self.color = coverage.color
        repertoire = level.repertoire
        self.kept = {position_of(k) for k in repertoire.order}
        self.moves: dict[Position, set[str]] = {}
        for k in repertoire.own:
            self.moves.setdefault(position_of(k), set()).update(repertoire.transitions[k])
        # Moves into each kept position, with the probability of games arriving that way.
        self.arrivals: dict[Position, list[tuple[float, Position, str]]] = {}
        for k in repertoire.order:
            for move, (target, _) in repertoire.transitions[k].items():
                if target is not None:
                    self.arrivals.setdefault(position_of(target), []).append((level.reach[k], position_of(k), move))
        self.order = list(dict.fromkeys(position_of(k) for k in reversed(repertoire.order)))
        self.transitions, self.roots = repertoire.transitions, coverage.roots
        self.policy = {
            position_of(k): {m: cast(float, w) for m, (_, w) in repertoire.transitions[k].items()}
            for k in repertoire.own
            if len(repertoire.transitions[k]) > 1
        }
        self.present: set[Position] = set()
        self.placed: set[tuple[Position, str]] = set()
        self.occurrence: dict[Position, tuple[chess.pgn.GameNode, chess.Board]] = {}
        self.moved: list[JsonObject] = []

    def keeps(self, k: Position, uci: str, target: Position) -> bool:
        if k not in self.kept:
            return False
        return uci in self.moves.get(k, ()) if turn(k) == self.color else target in self.kept

    def visit(self, node: chess.pgn.GameNode, board: chess.Board) -> Position:
        k = canonical(board)
        self.present.add(k)
        self.occurrence.setdefault(k, (node, board))
        return k

    def prune(self, game: chess.pgn.Game, cid: str, exclusions: Collection[str]) -> None:
        stack = [(cast(chess.pgn.GameNode, game), game.board())]
        while stack:
            node, board = stack.pop()
            k = self.visit(node, board)
            pending = []
            for child in list(node.variations):
                uci = child.move.uci()
                after = board.copy(stack=False)
                after.push(child.move)
                if f'{cid}:{k}:{uci}' in exclusions or not self.keeps(k, uci, canonical(after)):
                    node.remove_variation(child)
                    continue
                self.placed.add((k, uci))
                pending.append((cast(chess.pgn.GameNode, child), after))
            stack.extend(reversed(pending))

    def attach(self, position: Position, uci: str, sources: Mapping[tuple[Position, str], chess.pgn.ChildNode]) -> None:
        """Add a move at the first occurrence of `position`, with the kept part of a recorded line after it."""
        node, board = self.occurrence[position]
        stack: list[tuple[chess.pgn.GameNode, chess.Board, str, chess.pgn.ChildNode | None]] = [
            (node, board, uci, sources.get((position, uci)))
        ]
        while stack:
            parent, before, move, source = stack.pop()
            child = parent.add_variation(
                chess.Move.from_uci(move),
                comment=source.comment if source else '',
                starting_comment=source.starting_comment if source else '',
                nags=source.nags if source else (),
            )
            after = before.copy(stack=False)
            after.push(chess.Move.from_uci(move))
            self.placed.add((canonical(before), move))
            k = self.visit(child, after)
            if source is None:
                continue
            for variation in reversed(source.variations):
                v = variation.move.uci()
                following = after.copy(stack=False)
                following.push(variation.move)
                if self.keeps(k, v, canonical(following)):
                    stack.append((child, after, v, variation))

    def write(self, source: str | Path, exclusions: Collection[str]) -> tuple[str, JsonObject]:
        games, originals = read_games(source), read_games(source)
        ids = chapter_ids(games, exclusions)
        # The first recorded occurrence of every move, to copy its comments and the line after it.
        sources: dict[tuple[Position, str], chess.pgn.ChildNode] = {}
        for game, cid in zip(originals, ids):
            if cid is None:
                continue
            stack = [(cast(chess.pgn.GameNode, game), game.board())]
            while stack:
                node, board = stack.pop()
                k = canonical(board)
                for child in reversed(node.variations):
                    uci = child.move.uci()
                    if f'{cid}:{k}:{uci}' in exclusions:
                        continue
                    sources.setdefault((k, uci), child)
                    after = board.copy(stack=False)
                    after.push(child.move)
                    stack.append((child, after))
        chapters: list[tuple[chess.pgn.Game, str]] = []
        for game, cid in zip(games, ids):
            if cid is not None and canonical(game.board()) in self.kept:
                self.prune(game, cid, exclusions)
                chapters.append((game, cid))
        # Parents first, so a position is in place before anything is attached below it.
        for p in self.order:
            if p not in self.present:
                arrivals = sorted(
                    (-f, parent, m) for f, parent, m in self.arrivals.get(p, []) if parent in self.present
                )
                if not arrivals:
                    raise AssertionError(f'Kept position {p} cannot be placed in the trimmed PGN')
                _, parent, move = arrivals[0]
                self.attach(parent, move, sources)
                self.moved.append(dict(position=p, parent=parent, move=move))
            for move in sorted(self.moves.get(p, ())):
                if (p, move) not in self.placed:
                    self.attach(p, move, sources)
                    self.moved.append(dict(position=p, parent=p, move=move))
        written, left_out = [], []
        seen: set[tuple[Position, str]] = set()
        for game, cid in chapters:
            moves = recorded_moves(game)
            if moves <= seen:
                left_out.append(cid)
                continue
            seen |= moves
            written.append(game)
        left_out += [cid for cid in ids if cid is not None and cid not in {c for _, c in chapters}]
        # A StringExporter keeps adding to its output, so each game gets its own.
        exported = [
            game.accept(chess.pgn.StringExporter(headers=True, variations=True, comments=True)) for game in written
        ]
        text = '\n\n'.join(exported) + '\n'
        return text, dict(chapters_written=len(written), chapters_left_out=left_out, moved=self.moved)

    def reproduces(self, text: str, coverage: Coverage, level: Level) -> bool:
        """Whether the PGN, read back as a repertoire, plays exactly the trimmed repertoire and scores the same.

        Moves are compared position by position: where a cut breaks a repetition loop, the trimmed repertoire keeps
        the full one's loop occurrences while the PGN read back has plain positions, which play the same.
        """
        stream = io.StringIO(text)
        games = []
        while (game := chess.pgn.read_game(stream)) is not None:
            games.append(game)
        graph = parse_games(games)
        transitions = resolve(graph, self.color, self.policy)
        if shape(transitions, self.roots) != shape(self.transitions, self.roots):
            return False
        order = topology(transitions, list(self.roots))
        model = prepare(graph, transitions, order, self.color, coverage.evidence)
        values = backward(model, order, empirical(model, self.color), coverage.sparse)
        root = sum((w * values[k] for k, w in self.roots.items()), np.zeros(8))
        return bool(np.allclose(root, level.root, atol=1e-12, rtol=0))


def shape(transitions: Transitions, roots: Weights) -> dict[Position, set[tuple[str, Position | None]]]:
    """The moves played from each position reachable from the roots, and the positions they lead to; None where a
    move draws by repetition."""
    result: dict[Position, set[tuple[str, Position | None]]] = {}
    for k in topology(transitions, list(roots)):
        moves = {(m, None if t is None else position_of(t)) for m, (t, _) in transitions[k].items()}
        result.setdefault(position_of(k), set()).update(moves)
    return result


def recorded_moves(game: chess.pgn.Game) -> set[tuple[Position, str]]:
    result = set()
    stack = [(cast(chess.pgn.GameNode, game), game.board())]
    while stack:
        node, board = stack.pop()
        k = canonical(board)
        for child in node.variations:
            result.add((k, child.move.uci()))
            after = board.copy(stack=False)
            after.push(child.move)
            stack.append((child, after))
    return result


def goals(goal: int) -> list[int | None]:
    """Every level compared, rarest last, then every line."""
    return [*sorted({*GOALS, goal}), None]


def analyze(path: str | Path, cache: str | Path = DEFAULT_CACHE, goal: int = DEFAULT_GOAL) -> tuple[JsonObject, str]:
    """The coverage comparison and the trimmed PGN text at `goal`."""
    analysis = AnalysisContext(path)
    result = Coverage(analysis, cache)
    rows, chosen = [], None
    for level_goal in goals(goal):
        level = result.level(level_goal)
        rows.append(result.row(level))
        if level_goal == goal:
            chosen = level
    assert chosen is not None
    full = rows[-1]
    if full['dropped'] or full['leaving_earlier']:
        raise AssertionError('Every line must keep the whole repertoire')
    writer = Writer(result, chosen)
    exclusions = analysis.manifest['configuration'].get('exclude', [])
    text, pgn = writer.write(analysis.source, exclusions)
    if not writer.reproduces(text, result, chosen):
        raise AssertionError('The trimmed PGN does not reproduce the trimmed repertoire')
    analysis.require_source('coverage analysis')
    color = analysis.saved['color']
    value: JsonObject = dict(
        color=color,
        goal=goal,
        levels=rows,
        at_goal=dict(result.details(chosen), pgn=dict(pgn, path=f'{TRIMMED_DIRECTORY}/{pgn_name(color, goal)}')),
        validation=dict(
            costs_reproduce_change=True,
            every_line_reproduces_score=True,
            trimmed_pgn_reproduces_repertoire=True,
        ),
        manifest=analysis.companion_manifest(
            coverage=dict(
                goal=goal,
                rule='keep positions where you are to move that games reach at least once in N games through kept '
                'positions; cut positions are scored as unprepared replies with their parent table rows',
            )
        ),
    )
    return value, text


def run(path: str | Path, cache: str | Path = DEFAULT_CACHE, goal: int = DEFAULT_GOAL) -> JsonObject:
    """Save the coverage analysis and the trimmed PGN, removing a trimmed PGN left from an earlier goal."""
    value, text = analyze(path, cache, goal)
    target = pgn_path(path, value['color'], goal)
    target.parent.mkdir(parents=True, exist_ok=True)
    for old in target.parent.glob(f'{value["color"]}-1-in-*.pgn'):
        if old != target:
            old.unlink()
    temporary = target.with_suffix('.pgn.tmp')
    temporary.write_text(text, encoding='utf-8', newline='\n')
    temporary.replace(target)
    write_companion(path, 'coverage', value)
    return value


def configured_goal(color: str, config: str | Path | None = None) -> int:
    """The goal in a configuration file, by default configs/<color>.json, or 1 in 300 without one."""
    path = Path(config) if config else Path(CONFIG.format(color))
    return goal_from_config(load_config(path) if path.is_file() else {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('reports', nargs='+', type=Path, help='Saved score JSON files, with unchanged source PGNs')
    parser.add_argument('--cache', default=DEFAULT_CACHE)
    parser.add_argument(
        '--goal',
        type=int,
        help='Coverage goal N, 1 in N games (default: coverage_goal in configs/<color>.json, or 300)',
    )
    args = parser.parse_args()
    try:
        for path in args.reports:
            color = json.loads(path.read_text(encoding='utf-8'))['color']
            goal = goal_from_config({CONFIG_KEY: args.goal}) if args.goal is not None else configured_goal(color)
            value = run(path, args.cache, goal)
            row = next(r for r in value['levels'] if r['goal'] == goal)
            print(
                f'{color}: 1 in {goal} games keeps {row["decisions"]:,} of {row["decisions"] + row["dropped"]:,} '
                f'moves; trimmed PGN saved to {pgn_path(path, color, goal)}',
                flush=True,
            )
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f'Coverage analysis failed: {exc}\n')
    from .render import update_report_outputs

    update_report_outputs(args.reports[-1])
