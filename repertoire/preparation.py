"""Cache-only stopping outcomes and their repertoire score contributions."""

import json
from collections.abc import Collection, Iterator, Sequence
from functools import cached_property
from pathlib import Path
from typing import NamedTuple, cast

import chess
import numpy as np

from .attribution import enrich
from .board_cache import children, fen_number, geometry, move_text, owner_outcome, position_of
from .context import DEFAULT_CACHE, AnalysisContext, stage_main
from .evaluate import Route, Weights, best_routes, entry_states, reaches
from .explorer import counts
from .graph import Graph, chapter_policy_overrides, resolve
from .model import Evidence, Model, Sampled, arrival_counts, node_empirical, position_counts, prepare_node, score
from .routes import depth_distribution, first_entry_examples
from .schema import JsonObject, Position
from .status import Status


class Nodes(dict[Position, float]):
    """Starting weights already given as scoring nodes, such as first arrivals found by a traversal, rather than
    as positions to be placed in the nodes games reach them in (see Evaluator.entry_nodes)."""


class Edge(NamedTuple):
    """A move that continues preparation: its probability and the canonical position it reaches."""

    move: str
    probability: float
    target: str


class Stop(NamedTuple):
    """Where preparation ends at a position: the move (if any), its probability, kind and evidence."""

    move: str | None
    probability: float
    kind: str
    counts: list
    fixed_score: float | None


class Evaluator:
    """Empirical continuation scores, depths and stopping events over the scoring model, from any starts.

    Positions are compiled on first use, children before parents, so `values` is always in postorder.
    Vector: resolved score, unresolved mass, prepared depth, sparse score, sparse mass.

    Keys are scoring nodes: positions, plus repetition history nodes inside loops (graph.unroll); use
    board_cache.position_of for the position. With `roots`, starts inside a loop are evaluated in the nodes
    games arrive at from them (evaluate.entry_states), as the saved chapter scores are.
    """

    model: Model
    sampled: Sampled
    values: dict[Position, np.ndarray]
    edges: dict[Position, list[Edge]]
    stops: dict[Position, list[Stop]]

    def __init__(
        self,
        graph: Graph,
        color: bool,
        evidence: Evidence,
        facts: dict,
        policy: dict | None = None,
        chapter: str | None = None,
        sparse: int = 30,
        roots: Weights | None = None,
    ) -> None:
        self.graph, self.color, self.evidence, self.facts = graph, color, evidence, facts
        self.policy, self.chapter, self.sparse, self.roots = policy or {}, chapter, sparse, roots
        transitions = resolve(graph, color, self.policy)
        if chapter is not None:
            # The chapter's own first recorded moves take precedence; the overall policy applies elsewhere.
            transitions = resolve(
                graph, color, dict(self.policy, **chapter_policy_overrides(graph, color, transitions, chapter))
            )
        self.transitions = transitions
        self.loops = {position_of(k) for k in transitions if position_of(k) != k}
        self.model, self.sampled, self.values, self.edges, self.stops = {}, {}, {}, {}, {}

    @cached_property
    def arrivals(self) -> dict[Position, list[int]]:
        return arrival_counts(self.graph, self.color, self.evidence)

    def database_counts(self, k: Position) -> tuple[list[int] | None, str]:
        """A position's database games and their source. Own-turn positions with a prepared move have no table
        of their own; they use the opponent move rows leading to them (see model.position_counts)."""
        k = position_of(k)
        if self.facts[k]['turn'] == self.color and self.graph.nodes[k].edges:
            sample = position_counts(self.arrivals, self.evidence, k)
            source = 'parent_move_rows' if k in self.arrivals else 'position'
        else:
            data = self.evidence.get(k)
            sample, source = (counts(data) if data is not None else None), 'position'
        return sample, source if sample is not None else 'unavailable'

    def own_choices(self, k: Position) -> dict[str, float]:
        return {m: cast(float, w) for m, (_, w) in self.transitions[k].items()}

    def stopping(self, sample: list[int], fixed: float | None = None) -> np.ndarray:
        n = sum(sample)
        s = fixed if fixed is not None else score(sample, self.color)
        if s is None:
            return np.array([0.0, 1.0, 0.0, 0.0, 1.0])
        sparse = fixed is None and n < self.sparse
        return np.array([s, 0.0, 0.0, s if sparse else 0.0, float(sparse)])

    def compile(self, start: Position) -> None:
        """Add every node reachable from `start` that is not yet compiled, children first."""
        if position_of(start) not in self.graph.nodes:
            raise ValueError('Chapter entry position is absent from the repertoire')
        if start in self.values:
            return
        active, pending = {start}, [(start, self.targets(start))]
        while pending:
            k, targets = pending[-1]
            target = next(targets, None)
            if target is None:
                pending.pop()
                active.discard(k)
                self._evaluate(k)
            elif target in active:
                raise ValueError('Repertoire contains a reachable cycle')
            elif target not in self.values:
                active.add(target)
                pending.append((target, self.targets(target)))

    def targets(self, k: Position) -> Iterator[Position]:
        return iter(t for t, _ in self.transitions[k].values() if t is not None)

    def _evaluate(self, k: Position) -> None:
        node = self.model[k] = prepare_node(k, self.transitions[k], self.color, self.evidence)
        sampled = self.sampled[k] = node_empirical(node, self.color)
        value, edges, stops = np.zeros(5), [], []
        for branch, (p, _) in zip(node.branches, sampled):
            if not p:
                continue
            if branch.target is not None:
                value += p * self.values[branch.target]
                edges.append(Edge(cast(str, branch.move), p, branch.target))
            else:
                value += p * self.stopping(branch.counts, branch.fixed_score)
                stops.append(Stop(branch.move, p, cast(str, branch.kind), branch.counts, branch.fixed_score))
        if node.mode == 'own':
            value[2] += 1
        self.values[k], self.edges[k], self.stops[k] = value, edges, stops

    def value(self, k: Position) -> np.ndarray:
        self.compile(k)
        return self.values[k]

    def entry_nodes(self, starts: Weights) -> Weights:
        """`starts` spread over the nodes games arrive at from the roots, where a start lies in a repetition loop.

        Starts given as `Nodes` are already scoring nodes and are kept as they are.
        """
        if isinstance(starts, Nodes) or self.roots is None or not self.loops.intersection(starts):
            return starts
        for k, w in self.roots.items():
            if w:
                self.compile(k)
        return Nodes(entry_states(self.model, list(self.values), self.sampled, self.roots, starts))

    def evaluate(self, starts: Weights) -> np.ndarray:
        starts = self.entry_nodes(starts)
        for k, w in starts.items():
            if w:
                self.compile(k)
        return sum((w * self.values[k] for k, w in starts.items() if w), np.zeros(5))

    def reaches(self, starts: Weights, stop_at: Collection[Position] = ()) -> dict[Position, float]:
        """Incoming probability at each compiled node; see evaluate.reaches."""
        self.evaluate(starts)
        return reaches(self.model, list(self.values), self.sampled, self.entry_nodes(starts), stop_at)

    def routes(
        self,
        starts: Weights,
        stop_at: Collection[Position] = (),
        replies: bool = False,
    ) -> dict[Position, Route]:
        """The most likely route to each node; see evaluate.best_routes."""
        self.evaluate(starts)
        return best_routes(self.model, list(self.values), self.sampled, self.entry_nodes(starts), stop_at, replies)


class Evaluators:
    """One Evaluator per distinct comparison policy. Chapters whose first recorded moves agree with the overall
    policy share its evaluator, so each position is compiled once per policy rather than once per chapter."""

    def __init__(
        self,
        graph: Graph,
        color: bool,
        evidence: Evidence,
        facts: dict,
        policy: dict | None = None,
        sparse: int = 30,
        roots: Weights | None = None,
    ) -> None:
        self.graph, self.color, self.evidence, self.facts = graph, color, evidence, facts
        self.policy, self.sparse, self.roots = policy or {}, sparse, roots
        self.transitions = resolve(graph, color, self.policy)
        self.shared: dict[str, Evaluator] = {}

    def __call__(self, chapter: str | None = None) -> Evaluator:
        """The evaluator for a chapter's comparison policy, or for the overall policy when `chapter` is None."""
        overrides = (
            {} if chapter is None else chapter_policy_overrides(self.graph, self.color, self.transitions, chapter)
        )
        key = json.dumps(overrides, sort_keys=True)
        if key not in self.shared:
            self.shared[key] = Evaluator(
                self.graph,
                self.color,
                self.evidence,
                self.facts,
                dict(self.policy, **overrides),
                sparse=self.sparse,
                roots=self.roots,
            )
        return self.shared[key]


def chess_facts(graph: Graph, color: bool, evidence: Evidence) -> dict[Position, JsonObject]:
    result: dict[Position, JsonObject] = {}
    for k in graph.nodes:
        facts = geometry(k)
        targets = dict(facts.moves)
        afters = {}
        possible = {target for target in targets.values() if target in graph.nodes}
        for row in evidence.get(k, {}).get('moves', []):
            target = targets[row['uci']]
            afters[row['uci']] = target, owner_outcome(target, color)
        result[k] = dict(turn=facts.turn, outcome=owner_outcome(k, color), after=afters, possible_targets=possible)
    return result


def line_text(path: Sequence[str], fen: str = chess.STARTING_FEN) -> str:
    board = chess.Board(fen)
    result = []
    for san in path:
        result.append(f'{board.fullmove_number}{"." if board.turn else "..."}{san}')
        board.push_san(san)
    return ' '.join(result) or '(PGN root)'


def position_lines(graph: Graph) -> dict[Position, str]:
    """Format each canonical board's first recorded route using its chapter root."""
    root_fens = {c['id']: graph.nodes[c['root']].fen for c in graph.chapters}
    lines: dict[Position, str] = {}
    for position, node in graph.nodes.items():
        chapter = next(c['id'] for c in graph.chapters if c['id'] in node.chapters)
        lines[position] = line_text(node.path, root_fens[chapter])
    return lines


def stopping_rows(
    evaluator: Evaluator, starts: Weights, baseline: float | None, lines: dict[Position, str]
) -> list[JsonObject]:
    mass = evaluator.reaches(starts)
    # Occurrences of a position in a repetition loop share one row per stop, with their reach added.
    stopping: dict[tuple[Position, str | None, str], tuple[Stop, float]] = {}
    for k, reach in mass.items():
        if reach <= 0:
            continue
        for stop in evaluator.stops[k]:
            identity = position_of(k), stop.move, stop.kind
            probability = reach * stop.probability
            if identity in stopping:
                probability += stopping[identity][1]
            stopping[identity] = stop, probability
    rows: list[JsonObject] = []
    for (k, move, kind), ((_, _, _, sample, fixed), probability) in stopping.items():
        number = fen_number(evaluator.graph.nodes[k].fen)
        line = lines.get(k, k)
        position = k
        if move:
            line += ' ' + move_text(k, number, move)
            position = children(k)[move]
        s = fixed if fixed is not None else score(sample, evaluator.color)
        rows.append(
            dict(
                position=position,
                parent_position=k,
                move=move,
                type=kind,
                line=line,
                reach=probability,
                score=s,
                sample_count=sum(sample),
                counts_white_draw_black=sample,
                contribution_pp=None if s is None else 100 * probability * s,
                baseline_contribution_pp=None if s is None or baseline is None else 100 * probability * (s - baseline),
                sparse=fixed is None and sum(sample) < evaluator.sparse,
                unresolved=s is None,
            )
        )
    root = evaluator.evaluate(starts)
    if not np.isclose(sum(r['reach'] for r in rows), 1.0, atol=1e-10) or not np.isclose(
        sum((r['contribution_pp'] or 0) / 100 for r in rows), root[0], atol=1e-10
    ):
        raise AssertionError('Stopping rows do not conserve probability or reproduce the score')
    if (
        baseline is not None
        and root[1] == 0
        and not np.isclose(sum(r['baseline_contribution_pp'] for r in rows), 100 * (root[0] - baseline), atol=1e-10)
    ):
        raise AssertionError('Stopping rows do not reproduce the baseline difference')
    return rows


def analyze(path: str | Path, cache: str | Path = DEFAULT_CACHE) -> JsonObject:
    analysis = AnalysisContext(path)
    graph, color, saved, manifest = analysis.graph, analysis.color, analysis.saved, analysis.manifest
    evidence = analysis.read_evidence(cache)
    facts = chess_facts(graph, color, evidence)
    policy, sparse = analysis.policy, analysis.sparse_threshold
    chapters = {c['id']: c for c in saved['chapters']}
    baselines = {c['id']: c.get('entry_baseline', {}).get('raw_score') for c in saved['chapters']}
    baselines[None] = saved.get('starting_position_reference', {}).get('owner_score')
    scopes = [
        dict(
            id=s['id'],
            name=s['name'],
            starts=s['starts'],
            baseline=baselines[s['chapter']],
            chapter=s['chapter'],
            expected=s['score'],
            policy_basis=s['policy_basis'],
        )
        for s in analysis.scopes()
    ]
    lines = position_lines(graph)
    evaluators = Evaluators(graph, color, evidence, facts, policy, sparse, analysis.roots)
    for scope in scopes:
        expected = scope.pop('expected')
        if not scope['starts']:
            scope['status'] = Status.UNRESOLVED_ENTRY_WEIGHTS
            scope['stops'] = []
            scope['depth_distribution'] = depth_distribution(None, {})
            continue
        evaluator = evaluators(scope['chapter'])
        value = evaluator.evaluate(scope['starts'])
        depth = expected.get('prepared_depth', {}).get('expected_moves')
        if (
            not np.isclose(value[0], expected['resolved_contribution'], atol=1e-10)
            or not np.isclose(value[1], expected['unresolved_mass'], atol=1e-10)
            or (depth is not None and not np.isclose(value[2], depth, atol=1e-10))
        ):
            raise AssertionError('Preparation analysis did not reproduce the saved scope score')
        scope['stops'] = stopping_rows(evaluator, scope['starts'], scope['baseline'], lines)
        scope['score'] = float(value[0]) if value[1] == 0 else None
        scope['prepared_depth'] = float(value[2]) if value[1] == 0 else None
        scope['status'] = Status.RESOLVED if value[1] == 0 else Status.UNRESOLVED_EVIDENCE
        scope['depth_distribution'] = depth_distribution(evaluator, scope['starts'])
        bounds = expected.get('prepared_depth', {}).get('conditional_bounds')
        if bounds is not None and not np.allclose(scope['depth_distribution']['expected_bounds'], bounds, atol=1e-10):
            raise AssertionError('Depth distribution did not reproduce the saved depth bounds')
        if scope['chapter']:
            chapter = chapters[scope['chapter']]
            scope['entry_routes'] = first_entry_examples(
                evaluator,
                manifest['root_weights'],
                [e['position'] for e in chapter['entries']],
                chapter['score'].get('entry_probability'),
                chapter['score'].get('first_entry_weights') or None,
            )
    analysis.require_source('preparation analysis')
    result = dict(
        color=saved['color'],
        scopes=scopes,
        manifest=analysis.companion_manifest(
            depth_distribution_definition='Remaining own moves from the same scope entry mixture; survival '
            'probabilities and exact stopping depths retain transposed '
            'elapsed-depth histories. Missing reply distributions have finite '
            'structural bounds. Missing leaf scores do not affect depth.',
            entry_example_definition='Highest-probability single root-to-first-arrival path for each entry '
            'board. All routes contribute to entry weights; example probabilities '
            'are subsets, conditional on chapter entry.',
        ),
        validation=dict(
            original_scores_reproduced=True,
            stopping_contributions_reproduced=True,
            baseline_attribution_reproduced=True,
            source_pgn_unchanged=True,
            depth_distribution_conserves_probability=True,
            survival_reproduces_prepared_depth=True,
            first_entry_routes_reproduced=True,
        ),
    )
    return enrich(result, graph)


def main() -> None:
    stage_main('preparation', analyze, __doc__)


if __name__ == '__main__':
    main()
