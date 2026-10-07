"""Cache-only gap priorities, branch-score spread, and move-comparison intervals."""
from collections import defaultdict
from datetime import datetime, timezone
import json
import math

import numpy as np

from .evaluate import COMPLETED
from .uncertainty import (METHOD as UNCERTAINTY_METHOD, Posterior, comparison_interval, dirichlet_variance,
                          row_moments)
from .board_cache import children
from .context import DEFAULT_CACHE, AnalysisContext, stage_main
from .explorer import counts, validate
from .graph import resolve, topology
from .model import prepare
from .openings import name_flow
from .preparation import Evaluator, chess_facts
from .spread import (assert_outcomes, mixture as spread_mixture, recursive_spread,
                     stopping_counts, stopping_spread)
from .status import Status


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
        return dict(status=Status.UNRESOLVED_STOPPING_EVIDENCE, standard_deviation=None,
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
    return dict(status=Status.RESOLVED, mean_score=mean, resolved_mass=known,
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


def database_table(data, position, color, prior):
    """Legal moves and Dirichlet parameters for one cached position; reusable across sampling batches."""
    moves = sorted(children(position))
    rows = {r['uci']: counts(r) for r in data['moves']}
    residual = validate(data, position)
    observations = [rows.get(move, [0, 0, 0]) for move in moves]
    if sum(residual):
        observations.append(residual)
    table_prior = np.asarray(prior if color else prior[::-1])
    return moves, np.asarray(observations, dtype=float) + table_prior / len(observations)


class LocalComparisons:
    """Means and 95% intervals of local score changes, from one policy's exact posterior.

    A reply's or move's result split within its row is independent of how often it is played, so its
    own score has an exact Beta-like distribution. When a comparison involves such a score, its interval
    keeps that score's skewed shape, which matters for replies with only a few games.
    """

    def __init__(self, posterior, database, owner):
        self.posterior, self.database, self.owner = posterior, database, owner
        self.influence = posterior.influence()
        self._variance = {}

    def value_variance(self, k):
        if k not in self._variance:
            self._variance[k] = self.posterior.value_variance({k: 1.})
        return self._variance[k]

    def leaf(self, k):
        """A position whose value is one table's own result score, or None."""
        node = self.posterior.model[k]
        if node.mode == 'stop' and k in self.posterior.alpha:
            return row_moments(self.posterior.alpha[k][0], self.owner)
        return None

    def opponent(self, k, j):
        """Repertoire value before an opponent reply minus the value after it: mean and 95% interval."""
        posterior, branch = self.posterior, self.posterior.model[k].branches[j]
        before = posterior.values[k][COMPLETED, 0]
        reach = self.influence[k]
        # The value before the reply already contains the reply's own share of the score after it.
        component, coefficient = None, posterior.sample[k][j][0] - 1
        if branch.target is not None:
            after = posterior.values[branch.target][COMPLETED, 0]
            downstream = self.influence[branch.target]
            variance = sum((reach.get(m, 0.) - downstream.get(m, 0.)) ** 2 * posterior.table_variance(m)
                           for m in sorted(set(reach) | set(downstream)))
            component = self.leaf(branch.target)
        elif branch.fixed_score is not None:
            after, variance = branch.fixed_score, self.value_variance(k)
        else:
            # The reply's score covaries with the value before it only through its own share of that value.
            probability = posterior.sample[k][j][0]
            component = row_moments(posterior.alpha[k][j], self.owner)
            after = component[0]
            variance = self.value_variance(k) + component[1] * (1 - 2 * probability)
        drop = before - after
        return drop, comparison_interval(drop, variance, component, coefficient)

    def own(self, k, move, target):
        """Parent database score, the selected move's database score and the continuation after it."""
        moves, alpha = self.database[k]
        weights = alpha / alpha.sum()
        parent = float((weights * self.owner).sum())
        parent_variance = dirichlet_variance(alpha, np.broadcast_to(self.owner, alpha.shape))
        row = moves.index(move)
        selected, selected_variance = row_moments(alpha[row], self.owner)
        share = float(weights[row].sum())
        after = self.posterior.values[target][COMPLETED, 0]
        after_variance = self.value_variance(target)
        drop = parent - after
        # Keep the skewed shape of whichever bounded score varies most: the parent's or a leaf continuation's.
        leaf = self.leaf(target)
        component, coefficient = ((parent, parent_variance), 1.) if leaf is None or parent_variance >= leaf[1] else (leaf, -1.)
        return dict(
            drop=(drop, comparison_interval(drop, parent_variance + after_variance, component, coefficient)),
            # The parent score already contains the move's own share of its database score.
            database_gain=comparison_interval(selected - parent, parent_variance + selected_variance * (1 - 2 * share),
                                              (selected, selected_variance), 1 - share),
            continuation_gain=comparison_interval(after - selected, after_variance + selected_variance,
                                                  (selected, selected_variance), -1))


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


def analyze(path, cache=DEFAULT_CACHE):
    analysis = AnalysisContext(path, ('preparation', 'character', 'vulnerabilities', 'openings'))
    saved, manifest, supporting = analysis.saved, analysis.manifest, analysis.companions
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
    graph, color = analysis.graph, analysis.color
    default_transitions = resolve(graph, color, analysis.policy)
    compared = moves['manifest']['evidence']
    evidence = analysis.read_evidence(cache, list(dict(manifest['evidence'], **compared)), required=True)
    if any(analysis.provenance[k] != original for k, original in compared.items()):
        raise ValueError('Cached evidence changed since the vulnerability analysis; rebuild it first')
    chosen = defaultdict(set)
    for scope in scopes:
        for row in scope['moves'].get('all_signed_rows', []):
            if row['kind'] == 'own': chosen[row['position']].add(row['move'])
    database = {k: database_table(evidence[k], k, color, manifest['prior']) for k in chosen}
    owner = np.array([1., .5, 0.] if color else [0., .5, 1.])
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
        comparisons = LocalComparisons(Posterior(model, order, color, manifest['prior'], manifest['sparse_threshold']),
                                       database, owner)
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
                if row['kind'] == 'own':
                    parts = comparisons.own(k, move, model[k].branches[j].target)
                    _, drop = parts['drop']
                else:
                    _, drop = comparisons.opponent(k, j)
                entry = dict(local_drop_interval_pp=drop, local_gain_interval_pp=[-drop[1], -drop[0]])
                if row['kind'] == 'own':
                    entry.update(move_decomposition(row),
                                 database_move_gain_interval_pp=parts['database_gain'],
                                 continuation_gain_interval_pp=parts['continuation_gain'])
                results[scope['id']]['moves'][row['id']] = entry
        del comparisons, model
    analysis.require_source('report insight analysis')
    result = dict(color=saved['color'], scopes=list(results.values()),
        manifest=analysis.companion_manifest(supporting_sha256=analysis.companion_hashes,
            prior=manifest['prior'], uncertainty_method=UNCERTAINTY_METHOD,
            interval_definition='Approximate prior-completed 95% local model intervals: exact posterior means and first-order variances from each cached Dirichlet table, combined through shared transposition values. Own move and parent database scores share one table, so their covariance is included. Policies and population are fixed; historical game overlap and selection effects are not modeled. Weighted rankings use empirical reach.',
            spread_definition='Weighted standard deviation of stopping-event expected scores. Total variance equals between-canonical-position variance plus within-position incoming-evidence-cohort variance. No cutoff, sparse filter, prior or tunable parameter.'),
        validation=dict(source_pgn_unchanged=True, saved_scores_reproduced=True,
                        transposed_variance_decomposition=True, own_gain_decomposition=True))
    return add_recursive_spreads(result, saved, supporting, graph, evidence, facts)


def main():
    stage_main('insights', analyze, __doc__)


if __name__ == '__main__':
    main()
