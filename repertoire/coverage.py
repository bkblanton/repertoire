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
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chess
import chess.pgn
import numpy as np

from . import evals as evals_store
from .board_cache import canonical, children, fen_number, position_of, route_line, san, turn
from .chapter_order import (
    DEFAULT_ORDER,
    apply_order,
    configured_order,
    order_chapters,
    order_from_config,
)
from .chapter_order import (
    MODES as ORDER_MODES,
)
from .context import DEFAULT_CACHE, AnalysisContext, write_companion
from .engine import (
    expected_score,
    is_trivial_recapture,
    position_sharpness,
    pv_first_move,
)
from .evals import Evaluation, Store
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
from .graph import Graph, Transitions, chapter_ids, parse_games, read_games, resolve, topology
from .layout import report_directory
from .model import Model, Sampled, empirical, prepare
from .schema import JsonObject, Position
from .score import CONFIG, COVERAGE_KEY, CRITICAL_COVERAGE_KEY, KEEP_KEY, entry_positions, load_config
from .uncertainty import Posterior, difference_interval, paired_variance

CONFIG_KEY = COVERAGE_KEY
CRITICAL_KEY = CRITICAL_COVERAGE_KEY
KEEP_CONFIG_KEY = KEEP_KEY
DEFAULT_GOAL = 300
DEFAULT_CRITICAL: dict[str, Any] = {
    'enabled': True,
    'min_reach_goal': 2000,
    'only_move_gap_pp': 8.0,
    'only_move_risk_pp': 0.008,
    'engine_vs_db_pp': 8.0,
    'own_move_loss_pp': 5.0,
    'max_own_loss_pp': 3.0,
    'refutation_min_reach_goal': 5000,
}
# The levels every report compares, as N in "1 in N games"; your own goal is added when it is not one of them.
GOALS = (100, 150, 200, 300, 400, 600, 800, 1000)
TRIMMED_DIRECTORY = 'trimmed'


def goal_from_config(config: Mapping[str, object]) -> int:
    """The coverage goal N of a configuration, 1 in N games, defaulting to 1 in 300."""
    value = config.get(CONFIG_KEY, DEFAULT_GOAL)
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(f'{CONFIG_KEY} must be a whole number of games, at least 2 (1 in N games): {value!r}')
    return value


def critical_config_from_config(config: Mapping[str, object]) -> JsonObject:
    """Validate and normalize the optional `critical_coverage` block in a color configuration."""
    raw = config.get(CRITICAL_KEY)
    result = dict(DEFAULT_CRITICAL)
    if raw is None:
        return result
    if isinstance(raw, bool):
        result['enabled'] = raw
        return result
    if not isinstance(raw, Mapping):
        raise ValueError(f'{CRITICAL_KEY} must be a boolean or an object: {raw!r}')
    unknown = sorted(set(raw) - set(DEFAULT_CRITICAL))
    if unknown:
        raise ValueError(f'{CRITICAL_KEY} contains unknown options: {", ".join(unknown)}')
    for key, value in raw.items():
        if key == 'enabled':
            if not isinstance(value, bool):
                raise ValueError(f'{CRITICAL_KEY}.enabled must be a boolean: {value!r}')
            result[key] = value
        elif key in ('min_reach_goal', 'refutation_min_reach_goal'):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f'{CRITICAL_KEY}.{key} must be an integer >= 2: {value!r}')
            result[key] = value
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{CRITICAL_KEY}.{key} must be a non-negative number: {value!r}')
            result[key] = float(value)
    return result


def keep_from_config(config: Mapping[str, object], graph: Graph) -> list[Position]:
    """Pinned positions from `keep` in configuration, each a canonical FEN or a `{'path': ...}` object."""
    raw = config.get(KEEP_CONFIG_KEY, [])
    if not isinstance(raw, list):
        raise ValueError(f'{KEEP_CONFIG_KEY} must be a list of FEN strings or path objects: {raw!r}')
    for item in raw:
        valid = isinstance(item, str) or (
            isinstance(item, Mapping)
            and isinstance(item.get('path'), list)
            and all(isinstance(m, str) for m in item['path'])
        )
        if not valid:
            raise ValueError(f'Invalid {KEEP_CONFIG_KEY} entry: {item!r}')
    return entry_positions(raw, graph, label='Kept position')


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

    def level(self, goal: int | None, keep: Collection[Position] = ()) -> Level:
        reach, cut, unknown = coverage(
            self.full.model,
            self.full.order,
            self.full.raw,
            self.roots,
            1 / goal if goal else 0.0,
            keep=keep,
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

    def critical_protection(
        self,
        base: Level,
        critical_cfg: Mapping[str, Any],
        pinned: Collection[Position] = (),
        store: str | Path | None = None,
        order: Sequence[str] | None = None,
    ) -> tuple[Level, JsonObject]:
        """Rescue engine-critical and manually pinned lines below the coverage goal, keeping their routes intact."""
        pinned_list = sorted(set(pinned))
        base_kept = self.kept(base)
        dropped = self.decisions - base_kept
        empty_summary: JsonObject = dict(
            enabled=bool(critical_cfg['enabled']),
            config=dict(critical_cfg),
            pinned=pinned_list,
            decisions_added=0,
            ancestors_added=0,
            positions=[],
            row=None,
        )
        if not dropped or (not critical_cfg['enabled'] and not pinned_list):
            return base, empty_summary

        routes = self.lines()
        node_for_pos: dict[Position, Position] = {}
        for k in self.full.order:
            p = position_of(k)
            if p not in node_for_pos or self.reach[k] > self.reach[node_for_pos[p]]:
                node_for_pos[p] = k
        full_reach = by_position(self.reach)

        parent_info: dict[Position, tuple[Position, str]] = {}
        for p in self.decisions:
            _, (root, *moves) = routes[node_for_pos[p]]
            if not moves:
                continue
            cur = root
            for move in moves[:-1]:
                cur = next(cast(Position, b.target) for b in self.full.model[cur].branches if b.move == move)
            parent_info[p] = (position_of(cur), moves[-1])

        evaluations: dict[Position, Evaluation] = {}
        store_path = Path(evals_store.STORE if store is None else store)
        if critical_cfg['enabled'] and store_path.is_file():
            min_reach = 1 / critical_cfg['min_reach_goal']
            eval_wanted: set[Position] = set(base_kept)
            for k in base_kept:
                for move in self.full.transitions[node_for_pos[k]]:
                    if child := children(k).get(move):
                        eval_wanted.add(child)
            for k in dropped:
                eval_wanted.add(k)
                if full_reach[k] >= min_reach:
                    for move in children(k).values():
                        eval_wanted.add(move)
            with Store(store_path, read_only=True) as evals:
                evaluations = evals.read(sorted(eval_wanted))

        triggers: dict[Position, list[JsonObject]] = {}
        if critical_cfg['enabled'] and evaluations:
            min_reach = 1 / critical_cfg['min_reach_goal']
            gap_min = critical_cfg['only_move_gap_pp']
            risk_min = critical_cfg['only_move_risk_pp']
            eng_db_min = critical_cfg['engine_vs_db_pp']
            own_loss_min = critical_cfg['own_move_loss_pp']
            max_own_loss = critical_cfg['max_own_loss_pp']
            refut_min_reach = 1 / critical_cfg['refutation_min_reach_goal']

            def own_move_acceptable(pos: Position, target_score: float) -> bool:
                played = list(self.full.transitions[node_for_pos[pos]])
                child_map = children(pos)
                for move in played:
                    child_pos = child_map.get(move)
                    sc = expected_score(evaluations.get(child_pos), self.color) if child_pos else None
                    if sc is None or 100 * (target_score - sc) <= max_own_loss:
                        return True
                return False

            # 1. Only-move cliffs: sharp gap and material expected loss per game, excluding single-piece recaptures.
            for p in sorted(dropped):
                reach_p = full_reach[p]
                if reach_p < min_reach:
                    continue
                sh = position_sharpness(p, self.color, evaluations, allow_children=True)
                if not sh or sh['gap_pp'] < gap_min or reach_p * sh['gap_pp'] < risk_min:
                    continue
                played = self.full.transitions[node_for_pos[p]]
                if sh['best_move'] not in played:
                    child_map = children(p)
                    if not any(
                        (sc := expected_score(evaluations.get(child_map.get(m, '')), self.color)) is not None
                        and 100 * (sh['best_score'] - sc) <= max_own_loss
                        for m in played
                    ):
                        continue
                pi = parent_info.get(p)
                if pi and is_trivial_recapture(pi[0], pi[1], sh['best_move'], sh['second_move']):
                    continue
                triggers.setdefault(p, []).append(
                    dict(
                        kind='only_move',
                        gap_pp=sh['gap_pp'],
                        best_move_san=san(p, sh['best_move']),
                        second_move_san=san(p, sh['second_move']),
                        source=sh['source'],
                    )
                )

            # 2. Engine >> database collapses: pool non-sparse arrivals by transposed target position.
            arr_reach: dict[Position, float] = {}
            arr_db_sum: dict[Position, float] = {}
            for pk in self.full.order:
                if self.full.model[pk].mode != 'opponent' or self.reach[pk] <= 0:
                    continue
                for b, (prob, sc) in zip(self.full.model[pk].branches, self.full.raw[pk]):
                    if b.target is None or b.move is None or not prob or sc is None:
                        continue
                    if sum(b.counts) < self.sparse:
                        continue
                    tp = position_of(b.target)
                    if tp not in dropped:
                        continue
                    flow = self.reach[pk] * prob
                    arr_reach[tp] = arr_reach.get(tp, 0.0) + flow
                    arr_db_sum[tp] = arr_db_sum.get(tp, 0.0) + flow * sc
            for tp in sorted(arr_reach):
                r_arr = arr_reach[tp]
                if r_arr < min_reach:
                    continue
                db_sc = arr_db_sum[tp] / r_arr
                tsc = expected_score(evaluations.get(tp), self.color)
                if tsc is None:
                    continue
                eng_minus_db = 100 * (tsc - db_sc)
                if eng_minus_db < eng_db_min:
                    continue
                if float(self.full.values[node_for_pos[tp]][KNOWN]) - db_sc < 0.03:
                    continue
                if not own_move_acceptable(tp, tsc):
                    continue
                triggers.setdefault(tp, []).append(
                    dict(
                        kind='engine_vs_db',
                        engine_minus_db_pp=eng_minus_db,
                        engine_score=tsc,
                        database_score=db_sc,
                    )
                )

            # 3. Prepared refutations of kept own moves that the engine flags as inaccuracies or worse.
            for q in sorted(base_kept):
                s_before = expected_score(evaluations.get(q), self.color)
                if s_before is None:
                    continue
                child_map = children(q)
                for own_uci in self.full.transitions[node_for_pos[q]]:
                    after_q = child_map.get(own_uci)
                    if not after_q:
                        continue
                    s_after = expected_score(evaluations.get(after_q), self.color)
                    if s_after is None:
                        continue
                    loss_pp = 100 * (s_before - s_after)
                    if loss_pp < own_loss_min:
                        continue
                    refut = pv_first_move(after_q, evaluations.get(after_q))
                    ak = node_for_pos.get(after_q)
                    if not (refut and ak and refut in self.full.transitions.get(ak, {})):
                        continue
                    target = self.full.transitions[ak][refut][0]
                    if target is None:
                        continue
                    tp = position_of(target)
                    if tp in dropped and full_reach[tp] >= refut_min_reach:
                        triggers.setdefault(tp, []).append(
                            dict(
                                kind='refutation',
                                own_position=q,
                                own_move_san=san(q, own_uci),
                                own_loss_pp=loss_pp,
                                refutation_san=san(after_q, refut),
                            )
                        )

        # 4. Manual pinning from `keep`.
        def route_own_positions(pos: Position) -> list[Position]:
            _, (root, *moves) = routes[node_for_pos[pos]]
            result: list[Position] = []
            cur = root
            if self.full.model[cur].mode == 'own':
                result.append(position_of(cur))
            for move in moves:
                cur = next(cast(Position, b.target) for b in self.full.model[cur].branches if b.move == move)
                if self.full.model[cur].mode == 'own':
                    result.append(position_of(cur))
            return result

        for p in pinned_list:
            if p in dropped:
                triggers.setdefault(p, []).append(dict(kind='pinned'))
            elif p not in self.decisions and p in node_for_pos and self.reach[node_for_pos[p]] > 0:
                dropped_on_route = [anc for anc in route_own_positions(p) if anc in dropped]
                if dropped_on_route:
                    triggers.setdefault(dropped_on_route[-1], []).append(dict(kind='pinned'))

        if not triggers:
            return base, empty_summary

        keep_set: set[Position] = set(triggers)
        for p in triggers:
            keep_set.update(route_own_positions(p))

        protected_level = self.level(base.goal, keep=keep_set)
        protected_kept = self.kept(protected_level)
        added = protected_kept - base_kept
        if not added:
            return base, empty_summary

        protected_reach = by_position(protected_level.reach)
        rows = []
        for p in sorted(triggers, key=lambda pos: (-protected_reach.get(pos, 0.0), pos)):
            k = node_for_pos[p]
            own_sans = [san(p, m) for m in sorted(self.full.transitions[k])]
            rows.append(
                dict(
                    position=p,
                    line=routes[k][0],
                    reach=protected_reach.get(p, 0.0),
                    full_reach=full_reach[p],
                    own_move_san=' / '.join(own_sans),
                    reasons=triggers[p],
                    chapter_attribution=dict(source_ids=self.chapters_of(p, order)),
                )
            )
        summary: JsonObject = dict(
            enabled=bool(critical_cfg['enabled']),
            config=dict(critical_cfg),
            pinned=pinned_list,
            decisions_added=len(added),
            ancestors_added=len(added - set(triggers)),
            positions=rows,
            row=self.row(protected_level),
        )
        return protected_level, summary

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

    def chapters_of(self, position: Position, order: Sequence[str] | None = None) -> list[str]:
        """The chapters recording a position, in chapter order."""
        recorded = self.graph.nodes[position].chapters
        ids = order if order is not None else [c['id'] for c in self.graph.chapters]
        return [cid for cid in ids if cid in recorded]

    def details(self, level: Level, order: Sequence[str] | None = None) -> JsonObject:
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
                    chapter_attribution=dict(source_ids=self.chapters_of(p, order)),
                )
            )
        kept = self.kept(level)
        chapters = []
        ordered_chapters = apply_order(self.graph.chapters, order, lambda c: c['id'])
        for c in ordered_chapters:
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

    def write(
        self, source: str | Path, exclusions: Collection[str], order: Sequence[str] | None = None
    ) -> tuple[str, JsonObject]:
        games, originals = read_games(source), read_games(source)
        ids = chapter_ids(games, exclusions)
        included = [(game, original, cid) for game, original, cid in zip(games, originals, ids) if cid is not None]
        if order is not None:
            included = apply_order(included, order, lambda item: item[2])
        # The first recorded occurrence of every move, to copy its comments and the line after it.
        sources: dict[tuple[Position, str], chess.pgn.ChildNode] = {}
        for _, original, cid in included:
            stack = [(cast(chess.pgn.GameNode, original), original.board())]
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
        for game, _, cid in included:
            if canonical(game.board()) in self.kept:
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
        kept_ids = {c for _, c in chapters}
        left_out += [cid for _, _, cid in included if cid not in kept_ids]
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


def analyze(
    path: str | Path,
    cache: str | Path = DEFAULT_CACHE,
    goal: int = DEFAULT_GOAL,
    order: str = DEFAULT_ORDER,
    critical: Mapping[str, object] | bool | None = None,
    keep: Sequence[str | JsonObject] = (),
    store: str | Path | None = None,
) -> tuple[JsonObject, str]:
    """The coverage comparison and the trimmed PGN text at `goal`."""
    order = order_from_config({'chapter_order': order})
    analysis = AnalysisContext(path)
    result = Coverage(analysis, cache)
    critical_cfg = critical_config_from_config({CRITICAL_KEY: critical} if critical is not None else {})
    pinned = keep_from_config({KEEP_CONFIG_KEY: list(keep)}, result.graph) if keep else []
    ordering = order_chapters(
        analysis,
        result.evidence,
        order,
        (result.full.transitions, result.full.order, result.full.model, result.full.raw),
    )
    rows, base_chosen = [], None
    for level_goal in goals(goal):
        level = result.level(level_goal)
        rows.append(result.row(level))
        if level_goal == goal:
            base_chosen = level
    assert base_chosen is not None
    full = rows[-1]
    if full['dropped'] or full['leaving_earlier']:
        raise AssertionError('Every line must keep the whole repertoire')
    chosen, protected = result.critical_protection(
        base_chosen, critical_cfg, pinned, store=store, order=ordering['ids']
    )
    writer = Writer(result, chosen)
    exclusions = analysis.manifest['configuration'].get('exclude', [])
    text, pgn = writer.write(analysis.source, exclusions, ordering['ids'])
    if not writer.reproduces(text, result, chosen):
        raise AssertionError('The trimmed PGN does not reproduce the trimmed repertoire')
    analysis.require_source('coverage analysis')
    color = analysis.saved['color']
    details = result.details(chosen, ordering['ids'])
    details['protected'] = protected
    details['pgn'] = dict(pgn, path=f'{TRIMMED_DIRECTORY}/{pgn_name(color, goal)}')
    value: JsonObject = dict(
        color=color,
        goal=goal,
        chapter_order=ordering,
        levels=rows,
        at_goal=details,
        validation=dict(
            costs_reproduce_change=True,
            every_line_reproduces_score=True,
            trimmed_pgn_reproduces_repertoire=True,
        ),
        manifest=analysis.companion_manifest(
            coverage=dict(
                goal=goal,
                chapter_order=order,
                critical_coverage=critical_cfg,
                keep=pinned,
                rule='keep positions where you are to move that games reach at least once in N games through kept '
                'positions, plus criticality-protected and pinned positions with their routes; cut positions are '
                'scored as unprepared replies with their parent table rows',
            )
        ),
    )
    return value, text


def run(
    path: str | Path,
    cache: str | Path = DEFAULT_CACHE,
    goal: int = DEFAULT_GOAL,
    order: str = DEFAULT_ORDER,
    critical: Mapping[str, object] | bool | None = None,
    keep: Sequence[str | JsonObject] = (),
    store: str | Path | None = None,
) -> JsonObject:
    """Save the coverage analysis and the trimmed PGN, removing a trimmed PGN left from an earlier goal."""
    value, text = analyze(path, cache, goal, order, critical=critical, keep=keep, store=store)
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


def configured_critical(color: str, config: str | Path | None = None) -> JsonObject:
    """Critical-line protection settings from configs/<color>.json, or defaults without one."""
    path = Path(config) if config else Path(CONFIG.format(color))
    return critical_config_from_config(load_config(path) if path.is_file() else {})


def configured_keep(color: str, config: str | Path | None = None) -> list[Any]:
    """Raw pinned `keep` entries from configs/<color>.json, or empty without one."""
    path = Path(config) if config else Path(CONFIG.format(color))
    raw = load_config(path).get(KEEP_CONFIG_KEY, []) if path.is_file() else []
    if not isinstance(raw, list):
        raise ValueError(f'{KEEP_CONFIG_KEY} must be a list of FEN strings or path objects: {raw!r}')
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('reports', nargs='+', type=Path, help='Saved score JSON files, with unchanged source PGNs')
    parser.add_argument('--cache', default=DEFAULT_CACHE)
    parser.add_argument(
        '--goal',
        type=int,
        help='Coverage goal N, 1 in N games (default: coverage_goal in configs/<color>.json, or 300)',
    )
    parser.add_argument(
        '--chapter-order',
        choices=ORDER_MODES,
        help='Chapter order: study or opening (default: chapter_order in configs/<color>.json, or study)',
    )
    parser.add_argument(
        '--no-critical',
        action='store_true',
        help='Disable engine criticality protection (keep only positions meeting the reach goal and explicit pins)',
    )
    parser.add_argument('--store', type=Path, default=None, help=f'Evaluation store (default: {evals_store.STORE})')
    args = parser.parse_args()
    try:
        for path in args.reports:
            color = json.loads(path.read_text(encoding='utf-8'))['color']
            goal = goal_from_config({CONFIG_KEY: args.goal}) if args.goal is not None else configured_goal(color)
            order = args.chapter_order if args.chapter_order is not None else configured_order(color)
            critical_cfg = configured_critical(color)
            if args.no_critical:
                critical_cfg['enabled'] = False
            keep_cfg = configured_keep(color)
            value = run(path, args.cache, goal, order, critical=critical_cfg, keep=keep_cfg, store=args.store)
            row = (value['at_goal'].get('protected') or {}).get('row') or next(
                r for r in value['levels'] if r['goal'] == goal
            )
            print(
                f'{color}: 1 in {goal} games keeps {row["decisions"]:,} of {row["decisions"] + row["dropped"]:,} '
                f'moves; trimmed PGN saved to {pgn_path(path, color, goal)}',
                flush=True,
            )
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f'Coverage analysis failed: {exc}\n')
    from .render import update_report_outputs

    update_report_outputs(args.reports[-1])
