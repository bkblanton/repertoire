"""Cache-only stopping outcomes and their repertoire score contributions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import chess
import numpy as np

from .explorer import Explorer, counts
from .graph import key, parse
from .model import outcome, score
from .attribution import enrich
from .insights import depth_distribution, first_entry_examples


class MissingEvidence(ValueError):
    pass


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
        if node.edges:
            if self.facts[k]['turn'] == self.color:
                targets = {node.edges[m] for m in self.own_choices(k)}
            else:
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
            stops.append((None, 1., 'theory_leaf', [0,0,0], terminal))
        elif node.edges and turn == self.color:
            value = np.zeros(5)
            for move, p in self.own_choices(k).items():
                target = node.edges[move]
                value += p*self.value(target)
                edges.append((move, p, target))
            value[2] += 1
        else:
            if k not in self.evidence:
                raise MissingEvidence(k)
            data = self.evidence[k]
            total = sum(counts(data))
            if not node.edges or not total:
                value = self.stopping(counts(data))
                stops.append((None, 1., 'theory_leaf' if not node.edges else 'unresolved_distribution', counts(data), None))
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
                    # The original scorer recognizes immediate transpositions only at nonleaves.
                    target = node.edges.get(move, target if target in self.graph.nodes else None)
                    if target:
                        value += p*self.value(target)
                        edges.append((move, p, target))
                    else:
                        value += p*self.stopping(sample, fixed)
                        stops.append((move, p, 'deviation', sample, fixed))
                residual = [a-b for a,b in zip(counts(data), accounted)]
                if sum(residual):
                    p = sum(residual)/total
                    value += p*self.stopping(residual)
                    stops.append((None, p, 'no_recorded_continuation', residual, None))
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
        stopped = sum(mass[k]*sum(s[1] for s in self.stops[k]) for k in mass)
        if not np.isclose(stopped, sum(starts.values()), atol=1e-10):
            raise AssertionError('Continuation evaluator probability conservation failed')
        return mass


def chess_facts(graph, color, evidence):
    result = {}
    for k,n in graph.nodes.items():
        board = chess.Board(n.fen)
        afters = {}
        possible = set()
        for move in board.legal_moves:
            after = board.copy(); after.push(move)
            target = key(after)
            if target in graph.nodes: possible.add(target)
        for row in evidence.get(k, {}).get('moves', []):
            after = board.copy(); after.push_uci(row['uci'])
            afters[row['uci']] = key(after), outcome(after, color)
        result[k] = dict(turn=board.turn, outcome=outcome(board, color), after=afters, possible_targets=possible)
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
        for move,p,kind,sample,fixed in evaluator.stops[k]:
            board = chess.Board(evaluator.graph.nodes[k].fen)
            line = lines.get(k, k)
            if move:
                line += f' {board.fullmove_number}{"." if board.turn else "..."}{board.san(chess.Move.from_uci(move))}'
                board.push_uci(move)
            s = fixed if fixed is not None else score(sample, evaluator.color)
            probability = reach*p
            rows.append(dict(position=key(board), parent_position=k, move=move, type=kind, line=line,
                reach=probability, score=s, sample_count=sum(sample), counts_white_draw_black=sample,
                contribution_pp=None if s is None else 100*probability*s,
                baseline_contribution_pp=None if s is None or baseline is None else 100*probability*(s-baseline),
                sparse=fixed is None and sum(sample)<evaluator.sparse,
                unresolved=s is None))
    root = evaluator.evaluate(starts)
    assert np.isclose(sum(r['reach'] for r in rows), 1., atol=1e-10)
    assert np.isclose(sum((r['contribution_pp'] or 0)/100 for r in rows), root[0], atol=1e-10)
    if baseline is not None and root[1] == 0:
        assert np.isclose(sum(r['baseline_contribution_pp'] for r in rows), 100*(root[0]-baseline), atol=1e-10)
    return rows


def analyze(path, cache='.cache/explorer'):
    path = Path(path)
    saved = json.loads(path.read_text(encoding='utf-8'))
    manifest = saved['manifest']; source = Path(manifest['input_path'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('PGN changed since scoring; regenerate scores first')
    graph = parse(source, manifest['configuration'].get('exclude', []))
    color = saved['color'] == 'white'
    explorer = Explorer(cache, manifest['filters'], offline=True)
    evidence, missing = {}, []
    try:
        for k in graph.nodes:
            try:
                evidence[k] = explorer.get(k)
            except ValueError as exc:
                if not str(exc).startswith('Offline cache miss:'): raise
                missing.append(k)
            if k in manifest['evidence'] and explorer.provenance.get(k) != manifest['evidence'][k]:
                raise ValueError('Saved score evidence changed; regenerate scores first')
    finally:
        explorer.close()
    facts = chess_facts(graph, color, evidence)
    policy = manifest['configuration'].get('policy', {})
    sparse = manifest['sparse_threshold']
    scopes = [dict(id='overall', name='Overall repertoire', starts=manifest['root_weights'],
                   baseline=saved.get('starting_position_reference', {}).get('owner_score'), chapter=None,
                   expected=saved['overall'], policy_basis='overall policy')]
    for chapter in saved['chapters']:
        weights = chapter['score'].get('first_entry_weights', {})
        if not weights and len(chapter['entries']) == 1:
            weights = {chapter['entries'][0]['position']: 1.}
        scopes.append(dict(id=chapter['id'], name=chapter['name'], starts={k:w for k,w in weights.items() if w},
            baseline=chapter.get('entry_baseline', {}).get('raw_score'), chapter=chapter['id'],
            expected=chapter['score'], policy_basis=chapter.get('policy_basis', 'overall policy')))
    lines = position_lines(graph)
    chapters = {c['id']: c for c in saved['chapters']}
    contexts = {}
    for scope in scopes:
        expected = scope.pop('expected')
        if not scope['starts']:
            scope['status'] = 'unresolved entry weights'; scope['stops'] = []
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
        assert np.isclose(value[0], expected['resolved_contribution'], atol=1e-10)
        assert np.isclose(value[1], expected['unresolved_mass'], atol=1e-10)
        if expected.get('prepared_depth', {}).get('expected_moves') is not None:
            assert np.isclose(value[2], expected['prepared_depth']['expected_moves'], atol=1e-10)
        scope['stops'] = stopping_rows(evaluator, scope['starts'], scope['baseline'], lines)
        scope['score'] = float(value[0]) if value[1] == 0 else None
        scope['prepared_depth'] = float(value[2]) if value[1] == 0 else None
        scope['status'] = 'resolved' if value[1] == 0 else 'unresolved evidence'
        scope['depth_distribution'] = depth_distribution(evaluator, scope['starts'])
        bounds = expected.get('prepared_depth', {}).get('conditional_bounds')
        if bounds is not None:
            assert np.allclose(scope['depth_distribution']['expected_bounds'], bounds, atol=1e-10)
        if scope['chapter']:
            chapter = chapters[scope['chapter']]
            region = (chapter.get('region') or {}).get('positions') or [e['position'] for e in chapter['entries']]
            scope['entry_routes'] = first_entry_examples(evaluator, manifest['root_weights'], region,
                chapter['score'].get('entry_probability'), chapter['score'].get('first_entry_weights') or None)
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('Source PGN changed during preparation analysis')
    result = dict(color=saved['color'], scopes=scopes,
        manifest=dict(created_at=datetime.now(timezone.utc).isoformat(), report_path=str(path.resolve()),
            report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), input_sha256=manifest['input_sha256'],
            input_path=str(source), filters=manifest['filters'], evidence=explorer.provenance,
            cache_only=True, network_requests=0, uncached_positions=missing, insights_schema_version=1,
            depth_distribution_definition='Remaining own moves from the same scope entry mixture; survival probabilities and exact stopping depths retain transposed elapsed-depth histories. Missing reply distributions have finite structural bounds. Missing leaf scores do not affect depth.',
            entry_example_definition='Highest-probability single root-to-first-arrival path for each entry board. All routes contribute to entry weights; example probabilities are subsets, conditional on chapter entry.'),
        validation=dict(original_scores_reproduced=True, stopping_contributions_reproduced=True,
                        baseline_attribution_reproduced=True, source_pgn_unchanged=True,
                        depth_distribution_conserves_probability=True,
                        survival_reproduces_prepared_depth=True, first_entry_routes_reproduced=True))
    return enrich(result, graph)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',nargs='+')
    parser.add_argument('--cache',default='.cache/explorer')
    args=parser.parse_args()
    for source in args.reports:
        path=Path(source); result=analyze(path,args.cache)
        path.with_suffix('.preparation.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        print(f"Generated {result['color']} stopping contributions for {len(result['scopes'])} scopes",flush=True)
    from .render import update_report_outputs
    update_report_outputs(args.reports[-1])


if __name__=='__main__':
    main()
