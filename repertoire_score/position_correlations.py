"""Reach-weighted correlations between future prepared depth and future preparation gain."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from . import SCHEMA_VERSION
from .stats import correlation, rank
from .evaluate import KNOWN, UNKNOWN, backward
from .context import DEFAULT_CACHE, AnalysisContext, file_sha256
from .graph import resolve, topology
from .model import empirical, prepare, score


METRICS = ('reach_weighted_pearson', 'reach_weighted_spearman', 'slope_pp_per_move', 'pearson', 'spearman')


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


def analyze_color(path, cache):
    analysis = AnalysisContext(path, ('vulnerabilities',))
    saved, manifest, moves = analysis.saved, analysis.manifest, analysis.companions['vulnerabilities']
    graph, color, roots = analysis.graph, analysis.color, analysis.roots
    transitions = resolve(graph, color, analysis.policy)
    order = topology(transitions, roots)
    compared = moves['manifest']['evidence']
    if any(manifest['evidence'][k] != compared[k] for k in manifest['evidence'].keys() & compared.keys()):
        raise ValueError('Scores and vulnerabilities use different cached evidence')
    evidence = analysis.read_evidence(cache, list(dict(manifest['evidence'], **compared)), required=True)
    if any(analysis.provenance[k] != original for k, original in compared.items()):
        raise ValueError('Cached evidence changed since the vulnerability analysis; rebuild it first')
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
    nonzero = [r for r in eligible if r['depth'] > 0]
    sensitivity = dict(n=len(nonzero), **metrics(*arrays(nonzero)))
    analysis.require_source('preparation correlation analysis')
    if (file_sha256(analysis.path) != analysis.report_sha256
            or file_sha256(analysis.path.with_suffix('.vulnerabilities.json')) != analysis.companion_hashes['vulnerabilities']):
        raise ValueError('Saved inputs changed during preparation correlation analysis')
    return dict(n=len(eligible), encounter_weight=sum(r['reach'] for r in eligible), **point,
                decisions=eligible, exclusions=exclusions,
                position_depths={k: float(v[0]) for k, v in depth.items()},
                sensitivity_without_zero_depth=sensitivity, validation=dict(root_score_reproduced=True,
                    root_depth_reproduced=float(root_depth), canonical_decisions_unique=True),
                provenance=dict(report_path=str(analysis.path.resolve()), report_sha256=analysis.report_sha256,
                    vulnerabilities_sha256=analysis.companion_hashes['vulnerabilities'], input_sha256=manifest['input_sha256'],
                    filters=manifest['filters'], prior=manifest['prior'], sparse_threshold=manifest['sparse_threshold']))


def analyze(paths, output, cache=DEFAULT_CACHE):
    result = dict(schema_version=SCHEMA_VERSION, created_at=datetime.now(timezone.utc).isoformat(), network_requests=0,
                  results={}, provenance={})
    for path in paths:
        color = json.loads(Path(path).read_bytes())['color']
        if color in result['results']:
            raise ValueError('Supply one saved score file per color')
        value = analyze_color(path, cache)
        result['provenance'][color] = value.pop('provenance')
        result['results'][color] = value
    output = Path(output).with_suffix('.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+')
    parser.add_argument('--cache', default=DEFAULT_CACHE)
    parser.add_argument('--output')
    args = parser.parse_args()
    output = args.output or Path(args.reports[0]).parent / 'prepared-depth-gain-correlation.json'
    analyze(args.reports, output, args.cache)
    print(f'Saved {Path(output).resolve()}; network requests: 0')
    from .render import update_report_outputs
    update_report_outputs(args.reports[0])


if __name__ == '__main__':
    main()
