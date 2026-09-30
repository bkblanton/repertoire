"""Cache-only stopping-outcome strengths and counterfactual PGN subtree trims."""
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from .explorer import Explorer, counts
from .graph import Graph, Node, key, parse
from .model import outcome, score
from .report import pct


@dataclass
class Occurrence:
    position: str
    fen: str
    chapter: str
    path: list
    parent: int | None
    move: str | None
    children: list = field(default_factory=list)
    end: int = 0


class Study:
    """Preserve ordered PGN occurrences, including duplicate providers of an edge."""
    def __init__(self, path, exclusions=()):
        self.original = parse(path, exclusions)
        self.records, self.roots, self.chapters = [], [], []
        with Path(path).open(encoding='utf-8-sig') as stream:
            while (game := chess.pgn.read_game(stream)) is not None:
                cid = game.headers.get('ChapterURL', '').rstrip('/').split('/')[-1] or str(len(self.chapters)+1)
                if cid in exclusions:
                    continue
                if any(c['id'] == cid for c in self.chapters):
                    cid += f'-copy-{len(self.chapters)+1}'
                chapter = self.original.chapters[len(self.chapters)]
                assert chapter['id'] == cid
                self.chapters.append(chapter)

                def visit(node, board, path, parent=None):
                    i = len(self.records)
                    record = Occurrence(key(board), board.fen(en_passant='legal'), cid, path,
                                        parent, node.move.uci() if parent is not None else None)
                    self.records.append(record)
                    for child in node.variations:
                        move = child.move.uci()
                        if f'{cid}:{record.position}:{move}' in exclusions:
                            continue
                        after = board.copy(); after.push(child.move)
                        child_index = visit(child, after, path+[board.san(child.move)], i)
                        record.children.append(child_index)
                    record.end = len(self.records)
                    return i
                self.roots.append(visit(game, game.board(), []))

    def removed(self, cuts):
        return {j for i in cuts for j in range(i, self.records[i].end)}

    def graph(self, cuts=()):
        removed = self.removed(cuts)
        nodes = {}
        for i, r in enumerate(self.records):
            if i in removed:
                continue
            node = nodes.setdefault(r.position, Node(r.fen, r.path))
            node.chapters.add(r.chapter)
            for j in r.children:
                if j in removed:
                    continue
                child = self.records[j]
                node.edges[child.move] = child.position
                node.provenance.setdefault(child.move, set()).add(r.chapter)
                moves = node.chapter_moves.setdefault(r.chapter, [])
                if child.move not in moves:
                    moves.append(child.move)
        return Graph(nodes, self.chapters, list(dict.fromkeys(self.records[i].position for i in self.roots)))

    def candidates(self):
        """Every suffix cut, plus coordinated cuts of all copies of the same edge."""
        providers = defaultdict(list)
        for i, r in enumerate(self.records):
            if r.parent is not None:
                providers[self.records[r.parent].position, r.move].append(i)
                yield (i,)
        for indices in providers.values():
            if len(indices) > 1:
                yield tuple(indices)


class MissingEvidence(ValueError):
    pass


class Evaluator:
    """Exact empirical evaluator, sharing immutable chess/evidence facts across trims.

    Rebuild decisions and transposition links from the surviving PGN graph. No
    position value from the original repertoire is reused after a cut.
    Vector: resolved score, unresolved mass, prepared depth, sparse score, sparse mass.
    """
    def __init__(self, graph, color, evidence, facts, policy=None, chapter=None, sparse=30):
        self.graph, self.color, self.evidence, self.facts = graph, color, evidence, facts
        self.policy, self.chapter, self.sparse = policy or {}, chapter, sparse
        self.values, self.edges, self.stops, self.active = {}, {}, {}, set()
        self.checked, self.checking = set(), set()

    def check_structure(self, k):
        if k not in self.graph.nodes:
            raise ValueError('Trim removes a fixed chapter entry position')
        if k in self.checked: return
        if k in self.checking: raise ValueError('Trim activates a reachable cycle')
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
            raise ValueError('Trim conflicts with an explicit configured move or mixture')
        return {m:w for m,w in weights.items() if w > 0}

    def value(self, k):
        if k in self.values:
            return self.values[k]
        if k not in self.graph.nodes:
            raise ValueError('Trim removes a fixed chapter entry position')
        if k in self.active:
            raise ValueError('Trim activates a reachable cycle')
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
            raise AssertionError('Trim evaluator probability conservation failed')
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


def own_pairs(graph, color, facts):
    return {(k,m) for k,n in graph.nodes.items() if facts[k]['turn'] == color for m in n.edges}


def line_text(path, fen=chess.STARTING_FEN):
    board = chess.Board(fen)
    result = []
    for san in path:
        result.append(f'{board.fullmove_number}{"." if board.turn else "..."}{san}')
        board.push_san(san)
    return ' '.join(result) or '(PGN root)'


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
    study = Study(source, manifest['configuration'].get('exclude', []))
    graph = study.graph(); color = saved['color'] == 'white'
    assert {(k,tuple(n.edges)) for k,n in graph.nodes.items()} == {(k,tuple(n.edges)) for k,n in study.original.nodes.items()}
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
    roots = manifest['root_weights']
    scopes = [dict(id='overall', name='Overall repertoire', starts=roots,
                   baseline=saved.get('starting_position_reference', {}).get('owner_score'), chapter=None,
                   expected=saved['overall'], policy_basis='overall policy')]
    for c in saved['chapters']:
        weights = c['score'].get('first_entry_weights', {})
        if not weights and len(c['entries']) == 1:
            weights = {c['entries'][0]['position']: 1.}
        scopes.append(dict(id=c['id'], name=c['name'], starts={k:w for k,w in weights.items() if w},
            baseline=c.get('entry_baseline', {}).get('raw_score'), chapter=c['id'], expected=c['score'],
            policy_basis=c.get('policy_basis', 'overall policy'), trims=[]))
    initial_pairs = own_pairs(graph, color, facts)
    chapter_fens = {c['id']:graph.nodes[c['root']].fen for c in study.chapters}
    occurrence_lines = [line_text(r.path, chapter_fens[r.chapter]) for r in study.records]
    lines = {r.position:occurrence_lines[i] for i,r in reversed(list(enumerate(study.records)))}
    base_contexts = {}

    def context(g, chapter, contexts):
        # Deduplicate chapter policies, while recalculating precedence after each cut.
        for k,choice in policy.items():
            if k in g.nodes and g.nodes[k].edges and facts[k]['turn'] == color:
                weights = {choice:1.} if isinstance(choice,str) else choice
                if any(m not in g.nodes[k].edges for m in weights):
                    raise ValueError('Trim conflicts with an explicit configured move or mixture')
        local = {k:n.chapter_moves[chapter][0] for k,n in g.nodes.items()
                 if facts[k]['turn'] == color and chapter in n.chapter_moves and n.chapter_moves[chapter]
                 and policy.get(k, next(iter(n.edges))) != n.chapter_moves[chapter][0]}
        identity = json.dumps(local, sort_keys=True)
        if identity not in contexts:
            contexts[identity] = Evaluator(g,color,evidence,facts,dict(policy,**local),sparse=sparse)
        return contexts[identity]

    for scope in scopes:
        scope['trims'] = []; scope['unavailable_trims'] = []
        if not scope['starts']:
            scope['status'] = 'unresolved entry weights'; scope['stops'] = []; continue
        base = context(graph, scope['chapter'], base_contexts)
        v = base.evaluate(scope['starts'])
        expected = scope['expected']
        assert np.isclose(v[0], expected['resolved_contribution'], atol=1e-10)
        assert np.isclose(v[1], expected['unresolved_mass'], atol=1e-10)
        if expected.get('prepared_depth', {}).get('expected_moves') is not None:
            assert np.isclose(v[2], expected['prepared_depth']['expected_moves'], atol=1e-10)
        scope['_value'] = v; scope['_reach'] = base.reaches(scope['starts'])
        scope['_base'] = base
        scope['_selected_pairs'] = {(k,m) for k,reach in scope['_reach'].items() if reach>0 and facts[k]['turn']==color
                                    for m,p,_ in base.edges[k] if p>0}
        scope['stops'] = stopping_rows(base, scope['starts'], scope['baseline'], lines)
        scope['score'] = float(v[0]) if v[1] == 0 else None
        scope['prepared_depth'] = float(v[2]) if v[1] == 0 else None
        scope['status'] = 'resolved' if v[1] == 0 else 'unresolved evidence'

    proposals, seen = [], set()
    for cuts in study.candidates():
        removed = study.removed(cuts)
        identity = tuple(sorted(removed))
        if identity in seen: continue
        seen.add(identity)
        changed = study.graph(cuts)
        lost = initial_pairs-own_pairs(changed,color,facts)
        if not lost: continue
        active_scopes = [s for s in scopes if s.get('_selected_pairs', set()) & lost]
        if not active_scopes: continue
        first = study.records[cuts[0]]; parent = study.records[first.parent]
        proposal = dict(id=f'trim-{len(proposals)+1}', cut_occurrences=list(cuts),
            cut_type='one PGN occurrence' if len(cuts)==1 else 'all occurrences of this position/move',
            line=occurrence_lines[first.parent], remove_from=occurrence_lines[cuts[0]],
            retained_position=parent.position, removed_pgn_halfmoves=len(removed),
            unique_recorded_own_moves_removed=len(lost),
            affected_chapters=sorted({study.records[i].chapter for i in removed}),
            affected_pgn_leaves=sorted({study.records[i].position for i in removed if not study.records[i].children}),
            removed_own_pairs=[list(p) for p in sorted(lost)],
            edits=[dict(chapter_id=study.records[i].chapter, keep_path_san=study.records[study.records[i].parent].path,
                        remove_move_uci=study.records[i].move, remove_subtree_line=occurrence_lines[i]) for i in cuts])
        proposals.append(proposal)
        contexts = {}
        for scope in active_scopes:
            try:
                trial = context(changed, scope['chapter'], contexts)
                value = trial.evaluate(scope['starts'])
                new_reach = trial.reaches(scope['starts'])
            except (ValueError, MissingEvidence) as exc:
                for ctx in contexts.values():
                    ctx.active.clear(); ctx.checking.clear()
                scope['unavailable_trims'].append(dict(id=proposal['id'], reason=type(exc).__name__+': '+str(exc)))
                continue
            before = scope['_value']
            saved_moves = len(scope['_selected_pairs'] & lost)
            raw_cost = 100*(before[0]-value[0]) if before[1] == value[1] == 0 else None
            if raw_cost is not None and abs(raw_cost)<1e-10: raw_cost=0.
            # Conservative intervals over unknown scores, not statistical confidence intervals.
            cost_bounds = [100*(before[0]-value[0]-value[1]), 100*(before[0]+before[1]-value[0])]
            sample_sizes = []
            base = scope['_base']
            for current, other, reach_map in ((base,trial,scope['_reach']),(trial,base,new_reach)):
                for k,reach in reach_map.items():
                    if reach <= 0 or current.stops[k] == other.stops.get(k): continue
                    sample_sizes.extend(sum(sample) for _,p,_,sample,fixed in current.stops[k] if p>0 and fixed is None)
            scope['trims'].append(dict(id=proposal['id'], score_after=float(value[0]) if value[1]==0 else None,
                memorization_value_pp=raw_cost, trimming_gain_pp=None if raw_cost is None else -raw_cost,
                distinct_own_moves_saved=saved_moves,
                value_per_move_pp=None if raw_cost is None else raw_cost/saved_moves,
                expected_depth_reduction=float(before[2]-value[2]) if before[1]==value[1]==0 else None,
                conditional_cost_bounds_pp=cost_bounds, unresolved_mass_after=float(value[1]),
                minimum_changed_stop_observations=min(sample_sizes) if sample_sizes else None,
                changed_stop_sparse=any(n<sparse for n in sample_sizes),
                sparse_mass_before=float(before[4]), sparse_mass_after=float(value[4])))
        if len(proposals)%100 == 0:
            print(f'{saved["color"]}: evaluated {len(proposals)} distinct trim proposals', flush=True)

    for scope in scopes:
        resolved = [r for r in scope['trims'] if r['memorization_value_pp'] is not None]
        scope['harmful_preparation'] = sorted([r for r in resolved if r['memorization_value_pp']<0], key=lambda r:r['memorization_value_pp'])
        scope['low_value_preparation'] = sorted([r for r in resolved if r['memorization_value_pp']>=0], key=lambda r:(r['value_per_move_pp'],r['memorization_value_pp']))
        leaves = {r['position'] for r in scope['stops'] if r['type']=='theory_leaf'}
        scope['leaf_trim_choices'] = []
        by_id = {p['id']:p for p in proposals}
        for leaf in sorted(leaves):
            candidates = [r for r in resolved if leaf in by_id[r['id']]['affected_pgn_leaves']]
            beneficial = [r for r in candidates if r['trimming_gain_pp']>0]
            scope['leaf_trim_choices'].append(dict(position=leaf,
                best_gain=max(candidates,key=lambda r:r['trimming_gain_pp'])['id'] if candidates else None,
                shortest_beneficial=min(beneficial,key=lambda r:(by_id[r['id']]['removed_pgn_halfmoves'],-r['trimming_gain_pp']))['id'] if beneficial else None))
        for k in list(scope):
            if k.startswith('_') or k=='expected': del scope[k]
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('Source PGN changed during preparation analysis')
    return dict(color=saved['color'], scopes=scopes, proposals=proposals,
        manifest=dict(created_at=datetime.now(timezone.utc).isoformat(), report_path=str(path.resolve()),
            report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), input_sha256=manifest['input_sha256'],
            input_path=str(source), filters=manifest['filters'], evidence=explorer.provenance,
            cache_only=True, network_requests=0, uncached_positions=missing,
            memory_unit='Distinct own position/move pairs whose last PGN provider is removed and which were selected in this scope',
            comparison='Original chapter first-entry positions and weights held fixed; priorities recomputed from remaining PGN content'),
        validation=dict(original_scores_reproduced=True, stopping_contributions_reproduced=True,
                        baseline_attribution_reproduced=True, source_pgn_unchanged=True))


def number(value, precision=4):
    if value is None: return 'unresolved'
    if 0 < abs(value) < 10**(-precision): return f'{value:+.3g}'
    return f'{value:+.{precision}f}'


def render(result, top=15, chapter_top=3):
    proposals = {p['id']:p for p in result['proposals']}
    text = [f'# {result["color"].title()} preparation value and trimming', '',
        'Memorization value is score before trimming minus score after trimming. Negative values identify preparation that lowers the modeled score. Small positive values identify small benefits per memorized move. Trimming gain has the opposite sign. No cost threshold or depth discount is used.', '',
        'Each proposal deletes a PGN subtree in memory and rebuilds choices and transpositions. Shared moves count as saved only when their last provider is removed. Coordinated proposals explicitly remove all copies of the same edge. Broad cuts can remove several lines or switch the first available own move. Source PGNs are never edited.', '',
        'Chapter comparisons retain their original first-entry mixture and prefer that chapter\'s remaining first choices. Alternative chapter gains are not gains for the selected overall repertoire. Proposals overlap and their gains must not be added; evaluate any combination afresh.', '',
        'These are empirical screening estimates. A higher score after trimming means an earlier database fallback scores better, not that forgetting preparation improves personal play. Low-sample rows are marked sparse; trim gains do not have statistical confidence intervals. Missing observations retain unresolved bounds and missing cache entries are reported, with no network requests.', '']

    def strength(rows, count, title, reverse):
        rows = sorted([r for r in rows if r['baseline_contribution_pp'] is not None], key=lambda r:r['baseline_contribution_pp'], reverse=reverse)[:count]
        output = [title, '', '| Line / stopping position | Reach | Score | Contribution (pp) | Versus baseline (pp) | Observations |', '|---|---:|---:|---:|---:|---:|']
        for r in rows:
            output.append(f"| {r['line']} | {pct(r['reach'])} | {pct(r['score'])} | {number(r['contribution_pp'])} | {number(r['baseline_contribution_pp'])} | {r['sample_count']:,}{' (sparse)' if r['sparse'] else ''} |")
        return output if rows else [title,'','No resolved outcomes in this category.','']

    def trims(rows, count, title):
        output = [title, '', '| ID / stop after | Delete from | Own moves saved | Preparation value (pp) | Value per move (pp) | Trim gain (pp) | Depth reduction | Chapters / leaf positions affected | Minimum changed-stop observations |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows[:count]:
            p=proposals[row['id']]
            label=p['line']+(' [all copies]' if len(p['cut_occurrences'])>1 else '')
            sample_count = row.get('minimum_changed_stop_observations')
            support = ('n/a' if sample_count is None else str(sample_count))+(' (sparse)' if row.get('changed_stop_sparse') else '')
            output.append(f"| {row['id']}: {label} | {p['remove_from']} | {row['distinct_own_moves_saved']} | {number(row['memorization_value_pp'])} | {number(row['value_per_move_pp'],5)} | {number(row['trimming_gain_pp'])} | {number(row['expected_depth_reduction'],3)} | {len(p['affected_chapters'])} / {len(p['affected_pgn_leaves'])} | {support} |")
        return output if rows else [title,'','No resolved candidates in this category.','']

    for scope in result['scopes']:
        count = top if scope['id']=='overall' else chapter_top
        text += [f'## {scope["name"]}', '', f"Policy: {scope['policy_basis']}. Score: {pct(scope.get('score'))}; baseline: {pct(scope['baseline'])}.", '']
        if scope['status'] != 'resolved':
            text += [f"Status: {scope['status']}. Unknown outcomes are retained in JSON.", '']
        leaves = [r for r in scope['stops'] if r['type']=='theory_leaf']
        deviations = [r for r in scope['stops'] if r['type']=='deviation']
        for rows,label in ((leaves,'Prepared leaves'),(deviations,'Unprepared deviations')):
            text += strength(rows,count,f'### {label}: strongest baseline contributions',True)+['']
            text += strength(rows,count,f'### {label}: weakest baseline contributions',False)+['']
        text += trims(scope['harmful_preparation'],count,'### Trims that improve the modeled score')+['']
        text += trims(scope['low_value_preparation'],count,'### Smallest score cost per own move saved')+['']
        choices = {r['position']:r for r in scope['leaf_trim_choices']}
        trims_by_id = {r['id']:r for r in scope['trims']}
        weakest = sorted([r for r in leaves if r['baseline_contribution_pp'] is not None],key=lambda r:r['baseline_contribution_pp'])[:count]
        text += ['### Trim options for the weakest prepared leaves', '',
                 '| Leaf | Best-gain trim | Gain (pp) | Shortest beneficial trim |', '|---|---|---:|---|']
        for leaf in weakest:
            choice = choices[leaf['position']]
            best = choice['best_gain']; shortest = choice['shortest_beneficial']
            label = f"{best}: stop after {proposals[best]['line']}" if best else 'No available trim'
            short_label = f"{shortest}: stop after {proposals[shortest]['line']}" if shortest else 'None'
            gain = number(trims_by_id[best]['trimming_gain_pp']) if best else 'unavailable'
            text.append(f"| {leaf['line']} | {label} | {gain} | {short_label} |")
        text += ['', 'Best gain can be negative when every evaluated trim costs score. Shortest beneficial minimizes total PGN half-moves deleted across all affected copies. A broad cut may affect other leaves too; use its proposal ID to inspect the exact edits and affected chapters in JSON.', '']
        text += [f"Evaluated trims: {len(scope['trims'])}. Unresolved gains: {sum(r['memorization_value_pp'] is None for r in scope['trims'])}. Unavailable trims: {len(scope['unavailable_trims'])}. Full candidate edits, all leaf-to-trim mappings, unresolved bounds, evidence and affected chapters are in JSON.", '']
    text += ['## Reading the numbers', '',
        'Raw contribution is reach times stopping score, expressed in percentage points. Baseline contribution is reach times (stopping score minus scope baseline). Prepared leaves alone are only part of the score; adding all stopping outcomes reproduces the full score. The strongest/weakest tables use baseline contribution, so rare good positions are not mislabeled weak just because their raw contribution is small.', '',
        'The value-per-move denominator counts distinct recorded own position/move pairs removed from the study that were selected in this scope. Repeated visits or duplicate chapters do not create extra memorization units. Expected depth reduction is a separate frequency-weighted measure and can be negative if removing a first choice activates deeper alternative preparation.', '',
        f"Cache-only evaluation; network requests: 0. Distinct proposals: {len(result['proposals'])}. Original score, depth and stopping-contribution checks passed.", '']
    return '\n'.join(text)


def summary(results):
    lines=['# Preparation value reports', '',
           'Rankings of prepared leaves, unprepared deviations and opportunities to reduce memorization. Negative preparation value means the removed continuation lowers the modeled score; small positive value means it adds little benefit per distinct own move. Source studies are unchanged.', '',
           '| Color | Current repertoire score | Distinct trim proposals | Overall trims with modeled gains | Report |', '|---|---:|---:|---:|---|']
    for result,path in results:
        scope=result['scopes'][0]
        lines.append(f"| {result['color'].title()} | {pct(scope.get('score'))} | {len(result['proposals']):,} | {len(scope['harmful_preparation']):,} | [Overall and chapters]({Path(path).with_suffix('.preparation.md').name}) |")
    lines += ['', 'Proposals overlap and their gains must not be added. Sparse evidence is flagged. Trimming gains compare empirical population fallbacks with prepared continuations and do not establish personal improvement. Missing cache evidence is never fetched by this report.', '',
              '[Current score and chapter summary](summary.md) | [Vulnerabilities](vulnerabilities.md) | [Depth correlations](depth-delta-correlation.md)', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',nargs='+')
    parser.add_argument('--cache',default='.cache/explorer')
    parser.add_argument('--top',type=int,default=15)
    parser.add_argument('--chapter-top',type=int,default=3)
    args=parser.parse_args()
    if min(args.top,args.chapter_top)<1: parser.error('Table lengths must be positive')
    results=[]
    for source in args.reports:
        path=Path(source); result=analyze(path,args.cache)
        path.with_suffix('.preparation.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        output=path.with_suffix('.preparation.md')
        output.write_text(render(result,args.top,args.chapter_top),encoding='utf-8')
        results.append((result,path))
        print(f"Generated {result['color']} preparation report: {len(result['proposals'])} trim proposals",flush=True)
    (Path(args.reports[0]).parent/'preparation.md').write_text(summary(results),encoding='utf-8')


if __name__=='__main__':
    main()
