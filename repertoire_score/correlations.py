"""Exploratory correlations with paired, transposition-cluster bootstrap intervals."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .graph import parse, resolve


METRICS = {
    'pearson': 'Linear correlation',
    'spearman': 'Rank correlation',
    'reach_weighted_pearson': 'Reach-weighted correlation',
    'partial_correlation_controlling_entry_baseline': 'Baseline-adjusted correlation',
    'depth_vs_repertoire_score': 'Depth vs repertoire score',
    'depth_vs_headroom_gain': 'Depth vs fraction of available improvement',
    'log_depth_vs_delta': 'Log-depth vs delta',
    'within_named_family_pearson': 'Within named opening families',
    'depth_vs_entry_baseline': 'Depth vs entry baseline',
    'slope_pp_per_move': 'Linear slope (% per own move)',
}


def rank(x):
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends-counts+1+ends)/2)[inverse]


def correlation(x, y, weights=None):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    if len(x) < 2 or w.sum() <= 0:
        return float('nan')
    dx, dy = x-np.average(x, weights=w), y-np.average(y, weights=w)
    denominator = np.sqrt(np.sum(w*dx*dx)*np.sum(w*dy*dy))
    return float(np.clip(np.sum(w*dx*dy)/denominator, -1, 1)) if denominator > 0 else float('nan')


def metrics(rows):
    x = np.array([r['depth'] for r in rows])
    y = np.array([r['delta'] for r in rows])
    b = np.array([r['baseline'] for r in rows])
    s = np.array([r['score'] for r in rows])
    w = np.array([r['reach'] for r in rows])
    result = dict(pearson=correlation(x,y), spearman=correlation(rank(x),rank(y)),
                  reach_weighted_pearson=correlation(x,y,w), depth_vs_repertoire_score=correlation(x,s),
                  depth_vs_headroom_gain=correlation(x,(s-b)/(1-b)), log_depth_vs_delta=correlation(np.log(x),y),
                  depth_vs_entry_baseline=correlation(x,b))
    columns = [np.ones(len(rows)), b]
    if len({r['color'] for r in rows}) > 1:
        columns.append(np.array([r['color']=='black' for r in rows]))
    controls = np.column_stack(columns)
    dx = x-controls@np.linalg.lstsq(controls,x,rcond=None)[0]
    dy = y-controls@np.linalg.lstsq(controls,y,rcond=None)[0]
    valid = (np.linalg.norm(dx) > 1e-12*max(1,np.linalg.norm(x)) and
             np.linalg.norm(dy) > 1e-12*max(1,np.linalg.norm(y)))
    result['partial_correlation_controlling_entry_baseline'] = correlation(dx,dy) if valid else float('nan')
    centered_x, centered_y = [], []
    for family in sorted({(r['color'],r['family']) for r in rows}):
        indices = np.array([(r['color'],r['family'])==family for r in rows])
        if indices.sum() > 1:
            centered_x.extend(x[indices]-x[indices].mean())
            centered_y.extend(y[indices]-y[indices].mean())
    result['within_named_family_pearson'] = correlation(centered_x,centered_y)
    dx = x-x.mean()
    result['slope_pp_per_move'] = float(np.dot(dx,y-y.mean())/np.dot(dx,dx)) if np.dot(dx,dx)>0 else float('nan')
    return result


def connected_groups(position_sets):
    """Connected components, including indirect overlap through a third chapter."""
    remaining, groups = set(range(len(position_sets))), []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        component, pending = {first}, [first]
        while pending:
            i = pending.pop()
            hits = {j for j in remaining if position_sets[i] & position_sets[j]}
            remaining -= hits
            component |= hits
            pending.extend(sorted(hits))
        groups.append(sorted(component))
    return groups


def report_rows(path):
    path = Path(path)
    report = json.loads(path.read_text(encoding='utf-8'))
    manifest, color = report['manifest'], report['color']
    source = Path(manifest['input_path'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError(f'PGN differs from saved report; reanalyze before clustering: {source}')
    graph = parse(source, manifest['configuration'].get('exclude', []))
    rows, position_sets, policies = [], [], {}
    for chapter in report['chapters']:
        overrides = chapter.get('policy_overrides', {})
        identity = json.dumps(overrides, sort_keys=True)
        if identity not in policies:
            policies[identity] = resolve(graph, color=='white', dict(manifest['configuration'].get('policy', {}), **overrides))
        chapter_transitions = policies[identity]
        score, baseline = chapter['score'], chapter['entry_baseline']
        row = dict(color=color, id=chapter['id'], name=chapter['name'], family=chapter['name'].split(':')[0],
                   depth=score['prepared_depth']['expected_moves'], delta=baseline['difference_pp'],
                   score=score['raw_empirical_score'], baseline=baseline['raw_score'], reach=score['entry_probability'])
        if any(row[k] is None or not np.isfinite(row[k]) for k in ('depth','delta','score','baseline','reach')):
            raise ValueError(f'Correlation input has unresolved chapter estimates: {chapter["name"]}')
        if row['depth'] <= 0 or row['baseline'] >= 1 or row['reach'] < 0:
            raise ValueError(f'Correlation variants require positive depth, baseline below one, nonnegative reach: {chapter["name"]}')
        rows.append(row)
        seen, pending = set(), [e['position'] for e in chapter['entries']]
        while pending:
            k = pending.pop()
            if k not in seen:
                seen.add(k)
                pending.extend(target for target, _ in chapter_transitions[k].values())
        position_sets.append(seen)
    groups = connected_groups(position_sets)
    for group_number, indices in enumerate(groups, 1):
        for i in indices:
            rows[i]['cluster'] = f'{color}-{group_number}'
    provenance = {'report_path':str(path.resolve()), 'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'input_sha256':manifest['input_sha256'], 'filters':manifest['filters']}
    return rows, provenance


def cluster_sample(rows, rng):
    """Keep every row and its variables together; resample whole clusters by color."""
    strata = {}
    for row in rows:
        strata.setdefault(row['color'], {}).setdefault(row['cluster'], []).append(row)
    sample = []
    for color in sorted(strata):
        groups = [strata[color][name] for name in sorted(strata[color])]
        for i in rng.integers(len(groups), size=len(groups)):
            sample.extend(groups[i])
    return sample


def bootstrap(rows, repetitions, seed):
    cluster_counts = {color:len({r['cluster'] for r in rows if r['color']==color}) for color in {r['color'] for r in rows}}
    if min(cluster_counts.values()) < 2:
        raise ValueError('At least two chapter clusters per color are required for bootstrap intervals')
    point = metrics(rows)
    rng = np.random.default_rng(seed)
    draws = {name:[] for name in METRICS}
    for _ in range(repetitions):
        draw = metrics(cluster_sample(rows,rng))
        for name, value in draw.items():
            if np.isfinite(value):
                draws[name].append(value)
    intervals = {}
    for name, values in draws.items():
        valid = len(values)
        # Do not silently condition on a large fraction of undefined replicates.
        available = valid >= .99*repetitions and np.isfinite(point[name])
        intervals[name] = {'bounds':np.quantile(values,[.025,.975]).tolist() if available else None,
                           'valid_replicates':valid, 'invalid_replicates':repetitions-valid,
                           'status':'available' if available else 'undefined_or_degenerate'}
    return dict(n=len(rows), cluster_count=sum(cluster_counts.values()), clusters_by_color=cluster_counts,
                **{k:(v if np.isfinite(v) else None) for k,v in point.items()}, confidence_intervals_95=intervals)


def cell(result, metric):
    value = result[metric]
    if value is None:
        return 'undefined'
    limits = result['confidence_intervals_95'][metric]['bounds']
    suffix = '%' if metric == 'slope_pp_per_move' else ''
    if limits is None:
        return f'{value:.3f}{suffix} (interval unavailable)'
    return f'{value:.3f}{suffix} [{limits[0]:.3f}{suffix}, {limits[1]:.3f}{suffix}]'


def markdown(result):
    results, method = result['results'], result['bootstrap']
    lines = ['# Prepared depth and repertoire improvement', '',
             '**Each estimate is followed by an approximate 95% cluster-bootstrap confidence interval.** These exploratory intervals describe variation across chapter groups, conditional on the saved model estimates. They do not propagate Lichess count uncertainty or establish causation.', '',
             'Depth is expected remaining own prepared moves after first entry. Delta is repertoire score minus the weighted entry baseline, in percentage points.', '',
             '| Repertoire | Chapters / groups | Linear correlation | Rank correlation | Reach-weighted correlation | Baseline-adjusted correlation |',
             '|---|---:|---:|---:|---:|---:|']
    primary = list(METRICS)[:4]
    for color,r in results.items():
        lines.append(f"| {color.title()} | {r['n']} / {r['cluster_count']} | " + ' | '.join(cell(r,k) for k in primary) + ' |')
    lines += ['', 'Linear correlation measures association along a straight line; rank correlation compares chapter orderings. Reach weighting gives frequently reached chapters more influence without making overlapping chapters independent.', '',
              'Baseline adjustment correlates residuals after separately fitting depth and delta against entry-baseline score and an intercept; the pooled fit also includes color. It is a sensitivity check answering a narrower question, not a correction that invalidates the ordinary correlation. Baseline regressions, ranks and reach-weight normalization are recomputed in every bootstrap replicate.', '',
              '## Other variants', '', '| Measure | White | Black | Pooled |', '|---|---:|---:|---:|']
    for metric,label in list(METRICS.items())[4:]:
        lines.append(f'| {label} | ' + ' | '.join(cell(results[c],metric) if c in results else 'not supplied' for c in ('white','black','pooled')) + ' |')
    lines += ['', 'Fraction of available improvement means (score - baseline) / (1 - baseline). The named-family comparison subtracts each color/family mean, grouping by the chapter-name prefix before the colon. It is distinct from the transposition groups used for resampling. The slope is descriptive and does not predict the effect of adding one move.', '']
    if 'white_without_portuguese' in result.get('sensitivity',{}):
        sensitivity = result['sensitivity']['white_without_portuguese']
        lines += ['## Portuguese Gambit sensitivity', '',
                  'The primary results retain every chapter. Removing the White Portuguese Gambit outlier produces:', '',
                  '| Measure | Estimate and 95% interval |', '|---|---:|']
        for metric in primary:
            lines.append(f'| {METRICS[metric]} | {cell(sensitivity,metric)} |')
    lines += ['', '## Interval method and limitations', '',
              f"Used {method['repetitions']:,} paired cluster-bootstrap replicates per analysis, seed {method['seed']}. Bounds are the 2.5th and 97.5th percentiles of the resampled statistic. No new Lichess requests were made.", '',
              'Chapters belong to the same group when their reachable post-entry theory shares any canonical position, directly or through another chapter. Reachability uses each chapter comparison policy and the complete merged graph, including structurally possible theory moves with zero observed frequency. Entry weights also use that comparison policy for alternative chapters. Repeated introductory positions before chapter entry are excluded. All chapters and variables in a group are resampled together. Each color draws its original number of groups with replacement; pooled resampling is stratified by color. The number of chapter rows can vary because groups have different sizes.', '',
              'These intervals treat transposition groups as exchangeable independent units. There are few groups, their sizes differ, and the repertoire is deliberately selected rather than randomly sampled. Shared upstream frequencies and overlapping historical games can also create dependence across groups. Therefore nominal 95% coverage is not guaranteed; these are exploratory intervals, not a full uncertainty analysis of the fixed repertoire. More bootstrap replicates improve numerical precision, not the number of independent groups.', '',
              'The fixed saved chapter values have an exactly computable correlation. The intervals ask how that association varies under repeated sampling of similar chapter groups; they do not measure uncertainty about that arithmetic. A separate joint evidence-model analysis would be required to estimate uncertainty from finite Lichess observations for these fixed chapters.', '',
              'Undefined replicates are counted in JSON. An interval is withheld if more than 1% of replicates are undefined or its original statistic is undefined. These are individual 95% intervals, not simultaneous intervals adjusted for trying multiple variants. No significance claim or causal effect follows merely from excluding zero.', '',
              'The method follows the cluster-resampling principle discussed by [Cameron and Miller](https://faculty.econ.ucdavis.edu/faculty/cameron/research/Cameron_Miller_JHR_2014_July_09.pdf); the percentile construction is described in the [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). The chapter-group rule is specific to this analysis.', '',
              '## Resampling groups', '']
    for group in result['clusters']:
        lines.append(f"- {group['id']}: " + '; '.join(group['chapters']))
    return '\n'.join(lines)+'\n'


def analyze(paths, output, repetitions=20000, seed=20260929):
    rows, provenance = [], {}
    for path in paths:
        new_rows, source = report_rows(path)
        color = new_rows[0]['color']
        if color in provenance:
            raise ValueError('Supply only one report per color')
        provenance[color] = source
        rows.extend(new_rows)
    results = {}
    for i,color in enumerate(('white','black','pooled')):
        sample = rows if color=='pooled' else [r for r in rows if r['color']==color]
        if not sample or (color=='pooled' and len(provenance)<2):
            continue
        print(f'Bootstrap {color}: {len(sample)} chapters, {repetitions:,} replicates',flush=True)
        results[color] = bootstrap(sample,repetitions,seed+i)
    sensitivity = {}
    if any(r['color']=='white' and r['id']=='wchap001' for r in rows):
        sample = [r for r in rows if r['color']=='white' and r['id']!='wchap001']
        print('Bootstrap White excluding Portuguese: sensitivity check',flush=True)
        sensitivity['white_without_portuguese'] = bootstrap(sample,repetitions,seed+3)
    clusters = [{'id':cid,'chapters':[r['name'] for r in rows if r['cluster']==cid]} for cid in sorted({r['cluster'] for r in rows})]
    result = {'method':'Exploratory, color-stratified, paired transposition-cluster percentile bootstrap',
              'bootstrap':{'repetitions':repetitions,'seed':seed,'confidence_level':.95},
              'provenance':provenance,'results':results,'sensitivity':sensitivity,'chapters':rows,'clusters':clusters}
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(f'Saved correlation estimates and intervals: {output.with_suffix(".json").resolve()}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',nargs='+')
    parser.add_argument('--output',help='JSON output prefix (default: beside the score data)')
    parser.add_argument('--bootstrap-samples',type=int,default=20000)
    parser.add_argument('--seed',type=int,default=20260929)
    parser.add_argument('--plot',action='store_true',help='Also write a standalone HTML scatterplot; requires Plotly')
    args = parser.parse_args()
    if args.bootstrap_samples < 1000:
        parser.error('Use at least 1000 bootstrap replicates')
    output = Path(args.output) if args.output else Path(args.reports[0]).parent/'depth-delta-correlation'
    result = analyze(args.reports,output,args.bootstrap_samples,args.seed)
    if args.plot:
        from .correlation_plot import plot
        plot(result,output.with_suffix('.html'))
    from .render import update_report_outputs
    update_report_outputs(args.reports[-1])


if __name__ == '__main__':
    main()
