"""Reach-weighted preparation correlations with joint cached-evidence intervals."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .correlations import correlation, rank
from .evaluate import COMPLETED, KNOWN, UNKNOWN, backward
from .explorer import Explorer
from .graph import parse, resolve, topology
from .model import draws, empirical, prepare, score
from .report_insights import database_samples


METRICS = ('reach_weighted_pearson', 'reach_weighted_spearman', 'slope_pp_per_move', 'pearson', 'spearman')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weighted_rank(values, weights):
    """Weighted mid-CDF ranks, combining the weight of exact ties."""
    _, inverse = np.unique(values, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)
    return (np.cumsum(mass) - .5 * mass)[inverse] / weights.sum()


def metrics(depth, gain, weights):
    x, y, w = (np.asarray(a, dtype=float) for a in (depth, gain, weights))
    if len(x) < 2 or w.sum() <= 0:
        return dict.fromkeys(METRICS)
    dx, dy = x - np.average(x, weights=w), y - np.average(y, weights=w)
    xx = np.sum(w * dx * dx)
    result = dict(reach_weighted_pearson=correlation(x, y, w),
                  reach_weighted_spearman=correlation(weighted_rank(x, w), weighted_rank(y, w), w),
                  slope_pp_per_move=np.sum(w * dx * dy) / xx if xx > 0 else np.nan,
                  pearson=correlation(x, y), spearman=correlation(rank(x), rank(y)))
    return {k: float(v) if np.isfinite(v) else None for k, v in result.items()}


def depths(model, order, sampled, width=1):
    """Remaining own moves; a shared continuation has one value per draw."""
    values = {}
    for position in order:
        value = np.full(width, float(model[position].mode == 'own'))
        for branch, (probability, _) in zip(model[position].branches, sampled[position]):
            if branch.target is not None:
                value += probability * values[branch.target]
        values[position] = value
    return values


def reaches(model, order, sampled, roots, width=1):
    values = {position: np.zeros(width) for position in order}
    for position, weight in roots.items():
        values[position] += weight
    stopped = np.zeros(width)
    for position in reversed(order):
        for branch, (probability, _) in zip(model[position].branches, sampled[position]):
            flow = values[position] * probability
            if branch.target is None:
                stopped += flow
            else:
                values[branch.target] += flow
    if not np.allclose(stopped, sum(roots.values()), atol=1e-9):
        raise AssertionError('Correlation traversal does not conserve reach')
    return values


def analyze_color(path, cache, simulations, seed, batch_size=128):
    path = Path(path)
    move_path = path.with_suffix('.vulnerabilities.json')
    hashes = dict(score=digest(path), vulnerabilities=digest(move_path))
    saved = json.loads(path.read_bytes())
    moves = json.loads(move_path.read_bytes())
    manifest = saved['manifest']
    source = Path(manifest['input_path'])
    if digest(source) != manifest['input_sha256']:
        raise ValueError('PGN changed since scoring; regenerate scores before position correlations')
    if moves['color'] != saved['color'] or any(moves['manifest'].get(k) != expected for k, expected in (
            ('report_sha256', hashes['score']), ('input_sha256', manifest['input_sha256']), ('filters', manifest['filters']))):
        raise ValueError('Vulnerabilities do not match the saved score snapshot')
    if moves['manifest'].get('own_score_basis') != 'prepared repertoire continuation':
        raise ValueError('Own comparisons need a saved continuation refresh')
    graph = parse(source, manifest['configuration'].get('exclude', []))
    color = saved['color'] == 'white'
    transitions = resolve(graph, color, manifest['configuration'].get('policy', {}))
    roots = manifest['root_weights']
    order = topology(transitions, roots)
    evidence = {}
    explorer = Explorer(cache, manifest['filters'], offline=True)
    try:
        for position in manifest['evidence'].keys() & moves['manifest']['evidence'].keys():
            if manifest['evidence'][position] != moves['manifest']['evidence'][position]:
                raise ValueError('Scores and vulnerabilities use different cached evidence')
        for position, original in dict(manifest['evidence'], **moves['manifest']['evidence']).items():
            evidence[position] = explorer.get(position)
            if explorer.provenance[position] != original:
                raise ValueError('Cached evidence differs from saved scores or vulnerabilities')
    finally:
        explorer.close()
    model = prepare(graph, transitions, order, color, evidence)
    raw = empirical(model, color)
    values = backward(model, order, raw, manifest['sparse_threshold'])
    depth = depths(model, order, raw)
    reach = reaches(model, order, raw, roots)
    root_score = sum(weight * values[position] for position, weight in roots.items())
    if not np.allclose(root_score[[KNOWN, UNKNOWN], 0],
                       [saved['overall']['resolved_contribution'], saved['overall']['unresolved_mass']], atol=1e-10):
        raise AssertionError('Correlation model differs from saved repertoire score')
    root_depth = sum(weight * depth[position][0] for position, weight in roots.items())
    expected = saved['overall']['prepared_depth']['expected_moves']
    if expected is not None and not np.isclose(root_depth, expected, atol=1e-10):
        raise AssertionError('Position depths do not reproduce overall prepared depth')
    eligible, exclusions, seen = [], dict(sparse=0, unavailable=0, unreachable=0), set()
    for row in moves['overall']['all_signed_rows']:
        if row['kind'] != 'own':
            continue
        if row['id'] in seen:
            raise ValueError('Duplicate canonical own decision')
        seen.add(row['id'])
        if any(row.get(flag, False) for flag in ('sparse', 'parent_sparse', 'continuation_endpoint_sparse')):
            exclusions['sparse'] += 1
            continue
        if row['branch_reach'] <= 0:
            exclusions['unreachable'] += 1
            continue
        if row['move_score'] is None or row.get('move_database_score') is None:
            exclusions['unavailable'] += 1
            continue
        position, target = row['position'], row['target']
        branch = next((b for b in model[position].branches if b.move == row['move']), None)
        if model[position].mode != 'own' or branch is None or branch.target != target:
            raise ValueError('Saved own decision differs from the selected overall policy')
        probability = branch.weight
        weight = float(reach[position][0] * probability)
        after = float(values[target][KNOWN, 0])
        if values[target][UNKNOWN, 0] != 0 or not np.isclose(after, row['move_score'], atol=1e-10):
            raise ValueError('Saved continuation score differs from the cached model')
        if not np.isclose(weight, row['branch_reach'], atol=1e-10):
            raise ValueError('Saved decision reach differs from the cached model')
        cached_move = next(r for r in evidence[position]['moves'] if r['uci'] == row['move'])
        counts = [cached_move[k] for k in ('white', 'draws', 'black')]
        if not np.isclose(score(counts, color), row['move_database_score'], atol=1e-10):
            raise ValueError('Saved move baseline differs from the cached parent table')
        eligible.append(dict(id=row['id'], position=position, target=target, move=row['move'], line=row['line'],
                             chapters=row['chapters'], depth=float(depth[target][0]), reach=weight,
                             future_preparation_gain_pp=100 * (after - row['move_database_score']),
                             repertoire_score=after, move_database_score=row['move_database_score'],
                             games=row['sample_count'], probability=probability))
    def arrays(rows):
        return ([r['depth'] for r in rows], [r['future_preparation_gain_pp'] for r in rows], [r['reach'] for r in rows])
    point = metrics(*arrays(eligible))
    chosen = {}
    for row in eligible:
        chosen.setdefault(row['position'], set()).add(row['move'])
    samples = {k: [] for k in METRICS}
    for start in range(0, simulations, batch_size):
        width = min(batch_size, simulations - start)
        local_seed = seed + start // batch_size
        sampled = draws(model, color, width, local_seed, manifest['prior'])
        posterior = backward(model, order, sampled, manifest['sparse_threshold'], width)
        sampled_depth = depths(model, order, sampled, width)
        sampled_reach = reaches(model, order, sampled, roots, width)
        selected = {position: database_samples(evidence[position], position, selected_moves, color, width,
                                               manifest['prior'], local_seed)[1]
                    for position, selected_moves in chosen.items()}
        if eligible:
            x = np.stack([sampled_depth[r['target']] for r in eligible])
            y = np.stack([100 * (posterior[r['target']][COMPLETED] - selected[r['position']][r['move']]) for r in eligible])
            w = np.stack([sampled_reach[r['position']] * r['probability'] for r in eligible])
            for j in range(width):
                for name, value in metrics(x[:, j], y[:, j], w[:, j]).items():
                    if value is not None:
                        samples[name].append(value)
        print(f'{saved["color"]}: joint preparation draws {start + width}/{simulations}', flush=True)
    intervals = {}
    for name, sample in samples.items():
        available = point[name] is not None and len(sample) >= .99 * simulations
        intervals[name] = dict(bounds=np.quantile(sample, [.025, .975]).tolist() if available else None,
                               valid_replicates=len(sample), invalid_replicates=simulations - len(sample))
    nonzero = [r for r in eligible if r['depth'] > 0]
    sensitivity = dict(n=len(nonzero), **metrics(*arrays(nonzero)))
    for name, filename in (('score', path), ('vulnerabilities', move_path)):
        if digest(filename) != hashes[name]:
            raise ValueError('Saved inputs changed during preparation correlation analysis')
    if digest(source) != manifest['input_sha256']:
        raise ValueError('PGN changed during preparation correlation analysis')
    return dict(n=len(eligible), encounter_weight=sum(r['reach'] for r in eligible), **point,
                confidence_intervals_95=intervals, decisions=eligible, exclusions=exclusions,
                position_depths={k: float(v[0]) for k, v in depth.items()},
                sensitivity_without_zero_depth=sensitivity, validation=dict(root_score_reproduced=True,
                    root_depth_reproduced=float(root_depth), canonical_decisions_unique=True),
                provenance=dict(report_path=str(path.resolve()), report_sha256=hashes['score'],
                    vulnerabilities_sha256=hashes['vulnerabilities'], input_sha256=manifest['input_sha256'],
                    filters=manifest['filters'], prior=manifest['prior'], sparse_threshold=manifest['sparse_threshold']))


def analyze(paths, output, cache='.cache/explorer', simulations=2000, seed=20261005):
    if simulations < 1:
        raise ValueError('Use a positive simulation count')
    result = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(), network_requests=0,
                  interval_method='Joint Dirichlet evidence propagation for the fixed repertoire',
                  simulations=simulations, seed=seed, results={}, provenance={})
    for i, path in enumerate(paths):
        color = json.loads(Path(path).read_bytes())['color']
        if color in result['results']:
            raise ValueError('Supply one saved score file per color')
        value = analyze_color(path, cache, simulations, seed + 10000 * i)
        result['provenance'][color] = value.pop('provenance')
        result['results'][color] = value
    output = Path(output).with_suffix('.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+')
    parser.add_argument('--cache', default='.cache/explorer')
    parser.add_argument('--output')
    parser.add_argument('--simulations', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261005)
    args = parser.parse_args()
    output = args.output or Path(args.reports[0]).parent / 'prepared-depth-gain-correlation.json'
    analyze(args.reports, output, args.cache, args.simulations, args.seed)
    print(f'Saved {Path(output).resolve()}; network requests: 0')
    from .render import update_report_outputs
    update_report_outputs(args.reports[0])


if __name__ == '__main__':
    main()
