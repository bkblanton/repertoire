"""Cache-only stopping outcomes and their repertoire score contributions."""
import json
from typing import NamedTuple

import chess
import numpy as np

from .context import DEFAULT_CACHE, AnalysisContext, stage_main
from .explorer import counts
from .model import score
from .attribution import enrich
from .insights import depth_distribution, first_entry_examples
from .board_cache import children, fen_number, geometry, move_text, owner_outcome
from .status import Status


class MissingEvidence(ValueError):
    pass


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
    """Exact empirical continuation scores, depths, and stopping probabilities.

    Vector: resolved score, unresolved mass, prepared depth, sparse score, sparse mass.
    """
    def __init__(self, graph, color, evidence, facts, policy=None, chapter=None, sparse=30):
        self.graph, self.color, self.evidence, self.facts = graph, color, evidence, facts
        self.policy, self.chapter, self.sparse = policy or {}, chapter, sparse
        self.values, self.edges, self.stops, self.active = {}, {}, {}, set()
        self.checked, self.checking = set(), set()

    def check_structure(self, k):
        if k not in self.graph.nodes:
            raise ValueError('Chapter entry position is absent from the repertoire')
        if k in self.checked: return
        if k in self.checking: raise ValueError('Repertoire contains a reachable cycle')
        self.checking.add(k)
        node = self.graph.nodes[k]
        if self.facts[k]['outcome'] is None and (not node.edges or self.facts[k]['turn'] != self.color) and k not in self.evidence:
            self.checking.remove(k)
            raise MissingEvidence(k)
        targets = set()
        if self.facts[k]['outcome'] is None:
            if node.edges and self.facts[k]['turn'] == self.color:
                targets = {node.edges[m] for m in self.own_choices(k)}
            elif self.facts[k]['turn'] != self.color:
                targets = set(node.edges.values()) | (self.facts[k]['possible_targets'] & self.graph.nodes.keys())
        try:
            for target in targets: self.check_structure(target)
        finally:
            self.checking.remove(k)
        self.checked.add(k)

    def stopping(self, sample, fixed=None):
        n = sum(sample)
        s = fixed if fixed is not None else score(sample, self.color)
        if s is None:
            return np.array([0., 1., 0., 0., 1.])
        sparse = fixed is None and n < self.sparse
        return np.array([s, 0., 0., s if sparse else 0., float(sparse)])

    def own_choices(self, k):
        node = self.graph.nodes[k]
        chosen = self.policy.get(k, next(iter(node.edges)))
        local = node.chapter_moves.get(self.chapter, [])
        if local:
            chosen = local[0]
        weights = {chosen: 1.} if isinstance(chosen, str) else chosen
        if not weights or any(m not in node.edges or not 0 <= w <= 1 for m,w in weights.items()) or abs(sum(weights.values())-1)>1e-10:
            raise ValueError('Invalid configured move or mixture')
        return {m:w for m,w in weights.items() if w > 0}

    def value(self, k):
        if k in self.values:
            return self.values[k]
        if k not in self.graph.nodes:
            raise ValueError('Chapter entry position is absent from the repertoire')
        if k in self.active:
            raise ValueError('Repertoire contains a reachable cycle')
        self.active.add(k)
        node = self.graph.nodes[k]
        turn, terminal = self.facts[k]['turn'], self.facts[k]['outcome']
        edges, stops = [], []
        if terminal is not None:
            value = self.stopping([0,0,0], terminal)
            stops.append(Stop(None, 1., 'theory_leaf', [0,0,0], terminal))
        elif node.edges and turn == self.color:
            value = np.zeros(5)
            for move, p in self.own_choices(k).items():
                target = node.edges[move]
                value += p*self.value(target)
                edges.append(Edge(move, p, target))
            value[2] += 1
        else:
            if k not in self.evidence:
                raise MissingEvidence(k)
            data = self.evidence[k]
            total = sum(counts(data))
            if turn == self.color or not total:
                value = self.stopping(counts(data))
                stops.append(Stop(None, 1., 'theory_leaf' if turn == self.color else 'unresolved_distribution', counts(data), None))
            else:
                value, accounted = np.zeros(5), [0,0,0]
                for row in data['moves']:
                    sample = counts(row)
                    accounted = [a+b for a,b in zip(accounted, sample)]
                    p = sum(sample)/total
                    if not p:
                        continue
                    move = row['uci']
                    target, fixed = self.facts[k]['after'][move]
                    target = node.edges.get(move, target if target in self.graph.nodes else None)
                    if target:
                        value += p*self.value(target)
                        edges.append(Edge(move, p, target))
                    else:
                        value += p*self.stopping(sample, fixed)
                        stops.append(Stop(move, p, 'deviation', sample, fixed))
                residual = [a-b for a,b in zip(counts(data), accounted)]
                if sum(residual):
                    p = sum(residual)/total
                    value += p*self.stopping(residual)
                    stops.append(Stop(None, p, 'no_recorded_continuation', residual, None))
        self.active.remove(k)
        self.values[k], self.edges[k], self.stops[k] = value, edges, stops
        return value

    def evaluate(self, starts):
        for k,w in starts.items():
            if w: self.check_structure(k)
        return sum((w*self.value(k) for k,w in starts.items() if w), np.zeros(5))

    def reaches(self, starts):
        self.evaluate(starts)
        mass = dict.fromkeys(self.values, 0.)
        for k,w in starts.items():
            if w: mass[k] += w
        for k in reversed(self.values):
            for _,p,target in self.edges[k]:
                mass[target] += mass[k]*p
        stopped = sum(mass[k]*sum(s.probability for s in self.stops[k]) for k in mass)
        if not np.isclose(stopped, sum(starts.values()), atol=1e-10):
            raise AssertionError('Continuation evaluator probability conservation failed')
        return mass


def chess_facts(graph, color, evidence):
    result = {}
    for k,n in graph.nodes.items():
        facts = geometry(k)
        targets = dict(facts.moves)
        afters = {}
        possible = {target for target in targets.values() if target in graph.nodes}
        for row in evidence.get(k, {}).get('moves', []):
            target = targets[row['uci']]
            afters[row['uci']] = target, owner_outcome(target, color)
        result[k] = dict(turn=facts.turn, outcome=owner_outcome(k, color), after=afters, possible_targets=possible)
    return result


def line_text(path, fen=chess.STARTING_FEN):
    board = chess.Board(fen)
    result = []
    for san in path:
        result.append(f'{board.fullmove_number}{"." if board.turn else "..."}{san}')
        board.push_san(san)
    return ' '.join(result) or '(PGN root)'


def position_lines(graph):
    """Format each canonical board's first recorded route using its chapter root."""
    root_fens = {c['id']: graph.nodes[c['root']].fen for c in graph.chapters}
    lines = {}
    for position, node in graph.nodes.items():
        chapter = next(c['id'] for c in graph.chapters if c['id'] in node.chapters)
        lines[position] = line_text(node.path, root_fens[chapter])
    return lines


def stopping_rows(evaluator, starts, baseline, lines):
    mass = evaluator.reaches(starts)
    rows = []
    for k, reach in mass.items():
        if reach <= 0:
            continue
        number = fen_number(evaluator.graph.nodes[k].fen)
        for move,p,kind,sample,fixed in evaluator.stops[k]:
            line = lines.get(k, k)
            position = k
            if move:
                line += ' ' + move_text(k, number, move)
                position = children(k)[move]
            s = fixed if fixed is not None else score(sample, evaluator.color)
            probability = reach*p
            rows.append(dict(position=position, parent_position=k, move=move, type=kind, line=line,
                reach=probability, score=s, sample_count=sum(sample), counts_white_draw_black=sample,
                contribution_pp=None if s is None else 100*probability*s,
                baseline_contribution_pp=None if s is None or baseline is None else 100*probability*(s-baseline),
                sparse=fixed is None and sum(sample)<evaluator.sparse,
                unresolved=s is None))
    root = evaluator.evaluate(starts)
    if (not np.isclose(sum(r['reach'] for r in rows), 1., atol=1e-10)
            or not np.isclose(sum((r['contribution_pp'] or 0)/100 for r in rows), root[0], atol=1e-10)):
        raise AssertionError('Stopping rows do not conserve probability or reproduce the score')
    if baseline is not None and root[1] == 0 and not np.isclose(
            sum(r['baseline_contribution_pp'] for r in rows), 100*(root[0]-baseline), atol=1e-10):
        raise AssertionError('Stopping rows do not reproduce the baseline difference')
    return rows


def analyze(path, cache=DEFAULT_CACHE):
    analysis = AnalysisContext(path)
    graph, color, saved, manifest = analysis.graph, analysis.color, analysis.saved, analysis.manifest
    evidence = analysis.read_evidence(cache)
    facts = chess_facts(graph, color, evidence)
    policy, sparse = analysis.policy, analysis.sparse_threshold
    chapters = {c['id']: c for c in saved['chapters']}
    baselines = {c['id']: c.get('entry_baseline', {}).get('raw_score') for c in saved['chapters']}
    baselines[None] = saved.get('starting_position_reference', {}).get('owner_score')
    scopes = [dict(id=s['id'], name=s['name'], starts=s['starts'], baseline=baselines[s['chapter']], chapter=s['chapter'],
                   expected=s['score'], policy_basis=s['policy_basis']) for s in analysis.scopes()]
    lines = position_lines(graph)
    contexts = {}
    for scope in scopes:
        expected = scope.pop('expected')
        if not scope['starts']:
            scope['status'] = Status.UNRESOLVED_ENTRY_WEIGHTS; scope['stops'] = []
            scope['depth_distribution'] = depth_distribution(None, {})
            continue
        local = {k:n.chapter_moves[scope['chapter']][0] for k,n in graph.nodes.items()
                 if facts[k]['turn'] == color and n.chapter_moves.get(scope['chapter'])
                 and policy.get(k, next(iter(n.edges))) != n.chapter_moves[scope['chapter']][0]}
        identity = json.dumps(local, sort_keys=True)
        if identity not in contexts:
            contexts[identity] = Evaluator(graph, color, evidence, facts, dict(policy, **local), sparse=sparse)
        evaluator = contexts[identity]
        value = evaluator.evaluate(scope['starts'])
        depth = expected.get('prepared_depth', {}).get('expected_moves')
        if (not np.isclose(value[0], expected['resolved_contribution'], atol=1e-10)
                or not np.isclose(value[1], expected['unresolved_mass'], atol=1e-10)
                or (depth is not None and not np.isclose(value[2], depth, atol=1e-10))):
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
            region = (chapter.get('region') or {}).get('positions') or [e['position'] for e in chapter['entries']]
            scope['entry_routes'] = first_entry_examples(evaluator, manifest['root_weights'], region,
                chapter['score'].get('entry_probability'), chapter['score'].get('first_entry_weights') or None)
    analysis.require_source('preparation analysis')
    result = dict(color=saved['color'], scopes=scopes,
        manifest=analysis.companion_manifest(
            depth_distribution_definition='Remaining own moves from the same scope entry mixture; survival probabilities and exact stopping depths retain transposed elapsed-depth histories. Missing reply distributions have finite structural bounds. Missing leaf scores do not affect depth.',
            entry_example_definition='Highest-probability single root-to-first-arrival path for each entry board. All routes contribute to entry weights; example probabilities are subsets, conditional on chapter entry.'),
        validation=dict(original_scores_reproduced=True, stopping_contributions_reproduced=True,
                        baseline_attribution_reproduced=True, source_pgn_unchanged=True,
                        depth_distribution_conserves_probability=True,
                        survival_reproduces_prepared_depth=True, first_entry_routes_reproduced=True))
    return enrich(result, graph)


def main():
    stage_main('preparation', analyze, __doc__)


if __name__ == '__main__':
    main()
