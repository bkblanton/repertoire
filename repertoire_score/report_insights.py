"""Cache-only gap priorities, branch-score spread, and move-comparison intervals."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import chess
import numpy as np

from . import SCHEMA_VERSION
from .evaluate import COMPLETED, backward
from .explorer import Explorer, counts, validate
from .graph import parse, resolve, topology
from .model import draws, prepare
from .openings import name_flow
from .preparation import Evaluator, chess_facts
from .spread import (assert_outcomes, mixture as spread_mixture, recursive_spread,
                     stopping_counts, stopping_spread)


def gap_priorities(metrics):
    combined = defaultdict(float)
    for row in (metrics or {}).get('gaps', []):
        combined[row['position']] += row['reach']
    gaps = [dict(position=k, reach=p) for k, p in combined.items()]
    repeat = sum(r['reach'] ** 2 for r in gaps)
    cumulative = 0.
    result = []
    for row in sorted(gaps, key=lambda r: (-r['reach'], r['position'])):
        share = row['reach'] ** 2 / repeat if repeat else 0.
        cumulative += share
        result.append(dict(position=row['position'], reach=row['reach'],
            repeat_probability_contribution=row['reach'] ** 2,
            repeat_probability_share=share, cumulative_repeat_probability_share=min(1., cumulative)))
    return dict(priorities=result, known_repeat_probability=repeat,
                share_basis='known first gaps' if (metrics or {}).get('unresolved_mass', 0) else 'all first gaps')


def branch_score_spread(stops, expected=None):
    """Retain incoming evidence cohorts, and separate within-board variation."""
    stops = [r for r in stops if r['reach'] > 0]
    mass = sum(r['reach'] for r in stops)
    known = sum(r['reach'] for r in stops if r['score'] is not None)
    if not stops or not math.isclose(mass, 1., abs_tol=1e-10) or known < 1 - 1e-10:
        return dict(status='unresolved stopping evidence', standard_deviation=None,
                    between_position_deviation=None, within_position_deviation=None,
                    resolved_mass=known, mean_score=None)
    mean = sum(r['reach'] * r['score'] for r in stops)
    if expected is not None and not math.isclose(mean, expected, abs_tol=1e-10):
        raise AssertionError('Branch-score mean does not reproduce repertoire score')
    groups = defaultdict(list)
    for row in stops:
        # Residual buckets have no exact next board, and stay separate.
        identity = (row['position'],) if row['type'] in ('theory_leaf', 'deviation', 'terminal') else (
            row['position'], row['type'], row.get('parent_position'), row.get('move'))
        groups[identity].append(row)
    between = within = 0.
    for rows in groups.values():
        weight = sum(r['reach'] for r in rows)
        if not weight:
            continue
        local = sum(r['reach'] * r['score'] for r in rows) / weight
        between += weight * (local - mean) ** 2
        within += sum(r['reach'] * (r['score'] - local) ** 2 for r in rows)
    total = sum(r['reach'] * (r['score'] - mean) ** 2 for r in stops)
    if not math.isclose(total, between + within, abs_tol=1e-12):
        raise AssertionError('Transposed-cohort variance decomposition failed')
    return dict(status='resolved', mean_score=mean, resolved_mass=known,
        standard_deviation=math.sqrt(max(0., total)),
        between_position_deviation=math.sqrt(max(0., between)),
        within_position_deviation=math.sqrt(max(0., within)),
        stopping_positions=len(groups), variance=total)


def move_decomposition(row):
    parent, move, continuation = (row.get(k) for k in ('reference_score', 'move_database_score', 'move_score'))
    if any(v is None for v in (parent, move, continuation)):
        return dict(database_move_gain_pp=None, continuation_gain_pp=None, total_gain_pp=None)
    a, b, total = 100 * (move-parent), 100 * (continuation-move), 100 * (continuation-parent)
    if not math.isclose(a+b, total, abs_tol=1e-10):
        raise AssertionError('Own-move gain decomposition does not add up')
    return dict(database_move_gain_pp=a, continuation_gain_pp=b, total_gain_pp=total)


def interval(values):
    return [float(v) for v in np.quantile(values, [.025, .975])]


def database_samples(data, position, chosen, color, simulations, prior, seed):
    """One joint move/result table keeps the move and parent scores correlated."""
    board = chess.Board(position + ' 0 1')
    moves = sorted(m.uci() for m in board.legal_moves)
    rows = {r['uci']: counts(r) for r in data['moves']}
    residual = validate(data, position)
    observations = [rows.get(move, [0, 0, 0]) for move in moves]
    if sum(residual):
        observations.append(residual)
    table_prior = np.asarray(prior if color else prior[::-1])
    alpha = np.asarray(observations, dtype=float) + table_prior / len(observations)
    derived_seed = int.from_bytes(hashlib.sha256(f'{seed}:{position}'.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(derived_seed)
    gamma = rng.gamma(alpha, size=(simulations, *alpha.shape))
    totals = gamma.sum(axis=2)
    numerators = gamma[:, :, 0 if color else 2] + .5 * gamma[:, :, 1]
    parent = numerators.sum(axis=1) / totals.sum(axis=1)
    selected = {}
    for move in chosen:
        j = moves.index(move)
        selected[move] = np.divide(numerators[:, j], totals[:, j],
            out=np.full(simulations, .5), where=totals[:, j] > 0)
    return parent, selected


def add_recursive_spreads(value, saved, supporting, graph, evidence, facts=None):
    """Add position, reply, chapter and opening spreads to one matching snapshot."""
    color = saved['color'] == 'white'
    manifest = saved['manifest']
    if facts is None:
        facts = chess_facts(graph, color, evidence)
    characters = {s['id']: s for s in supporting['character']['scopes']}
    preparations = {s['id']: s for s in supporting['preparation']['scopes']}
    comparisons = supporting['vulnerabilities']
    moves = {'overall': comparisons['overall'], **{s['id']: s for s in comparisons['chapters']}}
    insights = {s['id']: s for s in value['scopes']}
    profiles = defaultdict(list)
    profiles['{}'].append('overall')
    for chapter in saved['chapters']:
        profiles[json.dumps(chapter.get('policy_overrides', {}), sort_keys=True)].append(chapter['id'])
    for identity, scopes in profiles.items():
        evaluator = Evaluator(graph, color, evidence, facts,
            dict(manifest['configuration'].get('policy', {}), **json.loads(identity)))
        starts = {k: 1. for sid in scopes for k, p in preparations[sid]['starts'].items() if p > 0}
        evaluator.evaluate(starts)
        position_values = recursive_spread(evaluator)
        for sid in scopes:
            scope = insights[sid]
            result = spread_mixture([(p, position_values[k]) for k, p in preparations[sid]['starts'].items() if p > 0],
                                   characters[sid].get('outcomes', {}).get('resolved_score'))
            previous = scope['branch_score_spread']
            if previous.get('variance') is not None and (result['variance'] is None or not math.isclose(
                    previous['variance'], result['variance'], abs_tol=1e-10)):
                raise AssertionError('Recursive spread differs from the stopping-event ledger')
            if characters[sid].get('outcomes'):
                assert_outcomes(result, characters[sid]['outcomes'])
            previous.update(result, recursive=True)
            positions = {}
            for row in characters[sid].get('positions', []):
                metric = position_values.get(row['position'])
                if metric is None:
                    metric = stopping_spread(row['outcomes'])
                assert_outcomes(metric, row['outcomes'])
                positions[row['position']] = metric
            scope['position_spreads'] = positions
            for row in moves[sid].get('all_signed_rows', []):
                after = (position_values[row['target']] if row['prepared'] else stopping_counts(
                    row['counts_white_draw_black'], color,
                    row['move_score'] if row.get('score_basis') == 'terminal result' else None))
                scope['moves'][row['id']].update(move_spread=after,
                    reference_spread=position_values[row['position']] if row['kind'] == 'opponent' else None)
        if identity == '{}':
            opening_spreads = {}
            for row in supporting['openings']['openings']:
                all_parts, entries = [], {}
                for entry in row['entries']:
                    parts = [(origin['conditional_weight'],
                        position_values[entry['position']] if origin['parent'] is None else
                        stopping_counts(origin['counts_white_draw_black'], color, origin['fixed_outcome']))
                        for origin in entry['origins']]
                    all_parts.extend(parts)
                    entry_spread = spread_mixture([(p / entry['conditional_weight'], child) for p, child in parts],
                                                 entry['repertoire_score'])
                    if len(parts) == 1:
                        child = parts[0][1]
                        entry_spread.update(immediate_variance=child['immediate_variance'],
                                            immediate_standard_deviation=child['immediate_standard_deviation'])
                    assert_outcomes(entry_spread, entry['outcomes'])
                    entries[entry['position']] = entry_spread
                result = spread_mixture(all_parts, row['repertoire_score'])
                assert_outcomes(result, row['outcomes'])
                opening_spreads[row['id']] = dict(branch_score_spread=result, entries=entries)
            value['opening_spreads'] = opening_spreads
    value['manifest'].update(
        spread_definition='B(s) = sum(p * (B(child) + (score(child) - score(s))**2)); known stopping scores have B=0. '
            'Forced own moves inherit the child. Shared canonical continuations are reused. '
            'Entry mixtures and the color mixture include between-entry score variance; standard deviations are never averaged. '
            'Immediate reply spread uses only the current opponent response distribution and recursive child scores. '
            'It is unavailable at own turns, stopping boundaries, or incomplete named reply tables. '
            'Sparse evidence remains included; unresolved evidence remains unavailable. Outcome volatility is the existing WDL variance.',
        spreads_refreshed_at=datetime.now(timezone.utc).isoformat())
    value['validation'].update(recursive_spread_equals_stopping_ledger=True, outcome_variance_decomposition=True)
    return value


def analyze(path, cache='.cache/explorer'):
    path = Path(path)
    source_bytes = path.read_bytes()
    saved = json.loads(source_bytes)
    manifest = saved['manifest']
    source = Path(manifest['input_path'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('PGN differs from saved scores; regenerate scores first')
    supporting, hashes = {}, {}
    for family in ('preparation', 'character', 'vulnerabilities', 'openings'):
        target = path.with_suffix(f'.{family}.json')
        raw = target.read_bytes()
        value = json.loads(raw)
        if value['color'] != saved['color'] or any(value['manifest'].get(k) != expected for k, expected in (
                ('report_sha256', hashlib.sha256(source_bytes).hexdigest()),
                ('input_sha256', manifest['input_sha256']), ('filters', manifest['filters']))):
            raise ValueError(f'{family} differs from saved score snapshot')
        supporting[family], hashes[family] = value, hashlib.sha256(raw).hexdigest()
    char = {s['id']: s for s in supporting['character']['scopes']}
    prep = {s['id']: s for s in supporting['preparation']['scopes']}
    moves = supporting['vulnerabilities']
    scopes = [dict(id='overall', score=saved['overall'], moves=moves['overall'], overrides={}),
        *[dict(id=c['id'], score=c['score'], moves=next(s for s in moves['chapters'] if s['id'] == c['id']),
               overrides=c.get('policy_overrides', {})) for c in saved['chapters']]]
    results = {}
    for scope in scopes:
        sid = scope['id']
        results[sid] = dict(id=sid,
            gaps=gap_priorities(char[sid].get('gap_coverage')),
            branch_score_spread=branch_score_spread(prep[sid].get('stops', []), scope['score'].get('raw_empirical_score')),
            moves={})
    graph = parse(source, manifest['configuration'].get('exclude', []))
    color = saved['color'] == 'white'
    default_transitions = resolve(graph, color, manifest['configuration'].get('policy', {}))
    evidence = {}
    explorer = Explorer(cache, manifest['filters'], offline=True)
    try:
        for k, original in dict(manifest['evidence'], **moves['manifest']['evidence']).items():
            evidence[k] = explorer.get(k)
            if explorer.provenance[k] != original:
                raise ValueError('Cached evidence differs from saved analysis')
    finally:
        explorer.close()
    chosen = defaultdict(set)
    for scope in scopes:
        for row in scope['moves'].get('all_signed_rows', []):
            if row['kind'] == 'own': chosen[row['position']].add(row['move'])
    database = {k: database_samples(evidence[k], k, selected, color, manifest['simulations'],
                manifest['prior'], manifest['seed']) for k, selected in chosen.items()}
    facts = chess_facts(graph, color, evidence)
    exact = {k: r['exact_name'] for k, r in supporting['openings']['positions'].items() if r.get('exact_name')}
    chapters = {c['id']: c for c in saved['chapters']}
    profiles = defaultdict(list)
    for scope in scopes:
        profiles[json.dumps(scope['overrides'], sort_keys=True)].append(scope)
    roots = [*manifest['root_weights'], *[e['position'] for c in saved['chapters'] for e in c['entries']]]
    for identity, group in profiles.items():
        transitions = default_transitions if identity == '{}' else resolve(graph, color,
            dict(manifest['configuration'].get('policy', {}), **json.loads(identity)))
        order = topology(transitions, roots)
        model = prepare(graph, transitions, order, color, evidence)
        sampled = draws(model, color, manifest['simulations'], manifest['seed'], manifest['prior'])
        values = backward(model, order, sampled, manifest['sparse_threshold'], manifest['simulations'])
        branches = {k: {b.move: j for j, b in enumerate(n.branches) if b.move} for k, n in model.items()}
        evaluator = Evaluator(graph, color, evidence, facts,
            dict(manifest['configuration'].get('policy', {}), **json.loads(identity)))
        for scope in group:
            if scope['id'] != 'overall':
                chapter = chapters[scope['id']]
                region = (chapter.get('region') or {}).get('positions') or [e['position'] for e in chapter['entries']]
                flows = name_flow(evaluator, manifest['root_weights'], exact, stop_at=region)
                entry_sources = {}
                for entry in chapter['entries']:
                    weights = flows.get(entry['position'], {})
                    total = sum(weights.values())
                    if total:
                        entry_sources[entry['position']] = [dict(id=name, share=p/total) for name, p in
                            sorted(weights.items(), key=lambda item: (-item[1], item[0] or '')) if p > 0]
                results[scope['id']]['entry_opening_sources'] = entry_sources
            for row in scope['moves'].get('all_signed_rows', []):
                k, move = row['position'], row['move']
                j = branches[k][move]
                branch = model[k].branches[j]
                after = values[branch.target][COMPLETED] if branch.target else sampled[k][j][1]
                before = database[k][0] if row['kind'] == 'own' else values[k][COMPLETED]
                entry = dict(local_drop_interval_pp=interval(100 * (before-after)),
                             local_gain_interval_pp=interval(100 * (after-before)))
                if row['kind'] == 'own':
                    selected = database[k][1][move]
                    entry.update(move_decomposition(row),
                        database_move_gain_interval_pp=interval(100 * (selected-before)),
                        continuation_gain_interval_pp=interval(100 * (after-selected)))
                results[scope['id']]['moves'][row['id']] = entry
        del sampled, values, model
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('PGN changed during report insight analysis')
    result = dict(color=saved['color'], scopes=list(results.values()),
        manifest=dict(created_at=datetime.now(timezone.utc).isoformat(), schema_version=SCHEMA_VERSION,
            report_path=str(path.resolve()), report_sha256=hashlib.sha256(source_bytes).hexdigest(),
            input_path=str(source), input_sha256=manifest['input_sha256'], filters=manifest['filters'],
            supporting_sha256=hashes, cache_only=True, network_requests=0,
            simulations=manifest['simulations'], seed=manifest['seed'], prior=manifest['prior'],
            interval_definition='Approximate prior-completed 95% local model intervals, using the saved joint move/result sampler and shared transposition values. Own move and parent database scores share one joint table. Opponent frequencies and results are jointly sampled. Policies and population are fixed; historical game overlap and selection effects are not modeled. Weighted rankings use empirical reach.',
            spread_definition='Weighted standard deviation of stopping-event expected scores. Total variance equals between-canonical-position variance plus within-position incoming-evidence-cohort variance. No cutoff, sparse filter, prior or tunable parameter.'),
        validation=dict(source_pgn_unchanged=True, saved_scores_reproduced=True,
                        transposed_variance_decomposition=True, own_gain_decomposition=True))
    return add_recursive_spreads(result, saved, supporting, graph, evidence, facts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--cache', default='.cache/explorer')
    args = parser.parse_args()
    for path in args.reports:
        print(f'Generating {path.stem} report insights from cache', flush=True)
        value = analyze(path, args.cache)
        path.with_suffix('.insights.json').write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    from .render import update_report_outputs
    update_report_outputs(args.reports[-1])


if __name__ == '__main__':
    main()
