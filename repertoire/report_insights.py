"""Cache-only gap priorities, branch-score spread, and move-comparison intervals."""

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np

from .board_cache import children
from .context import DEFAULT_CACHE, AnalysisContext, selected_policy, stage_main
from .evaluate import COMPLETED
from .explorer import counts, validate
from .graph import Graph, resolve, topology
from .model import Evidence, arrival_moves, move_counts, prepare
from .openings import name_flow
from .preparation import Evaluator, chess_facts
from .schema import JsonObject, Position
from .spread import assert_outcomes, recursive_spread, stopping_counts, stopping_spread
from .spread import mixture as spread_mixture
from .status import Status
from .uncertainty import METHOD as UNCERTAINTY_METHOD
from .uncertainty import Posterior, comparison_interval, dirichlet_variance, row_moments


class OwnComparison(TypedDict):
    drop: tuple[float, list[float]]
    database_gain: list[float] | None
    continuation_gain: list[float] | None


def gap_priorities(metrics: JsonObject | None) -> JsonObject:
    combined: defaultdict[Position, float] = defaultdict(float)
    for row in (metrics or {}).get('gaps', []):
        combined[row['position']] += row['reach']
    gaps: list[JsonObject] = [dict(position=k, reach=p) for k, p in combined.items()]
    repeat = sum(r['reach'] ** 2 for r in gaps)
    cumulative = 0.0
    result: list[JsonObject] = []
    # Reaches equal up to rounding noise rank by position, so the running share does not depend on summation order.
    for row in sorted(gaps, key=lambda r: (-round(r['reach'], 15), r['position'])):
        share = row['reach'] ** 2 / repeat if repeat else 0.0
        cumulative += share
        result.append(
            dict(
                position=row['position'],
                reach=row['reach'],
                repeat_probability_contribution=row['reach'] ** 2,
                repeat_probability_share=share,
                cumulative_repeat_probability_share=min(1.0, cumulative),
            )
        )
    return dict(
        priorities=result,
        known_repeat_probability=repeat,
        share_basis='known first gaps' if (metrics or {}).get('unresolved_mass', 0) else 'all first gaps',
    )


def branch_score_spread(stops: Iterable[JsonObject], expected: float | None = None) -> JsonObject:
    """Retain incoming evidence cohorts, and separate within-board variation."""
    stops = [r for r in stops if r['reach'] > 0]
    mass = sum(r['reach'] for r in stops)
    known = sum(r['reach'] for r in stops if r['score'] is not None)
    if not stops or not math.isclose(mass, 1.0, abs_tol=1e-10) or known < 1 - 1e-10:
        return dict(
            status=Status.UNRESOLVED_STOPPING_EVIDENCE,
            standard_deviation=None,
            between_position_deviation=None,
            within_position_deviation=None,
            resolved_mass=known,
            mean_score=None,
        )
    mean = sum(r['reach'] * r['score'] for r in stops)
    if expected is not None and not math.isclose(mean, expected, abs_tol=1e-10):
        raise AssertionError('Branch-score mean does not reproduce repertoire score')
    groups: defaultdict[tuple[Any, ...], list[JsonObject]] = defaultdict(list)
    for row in stops:
        # Residual buckets have no exact next board, and stay separate.
        identity = (
            (row['position'],)
            if row['type'] in ('theory_leaf', 'deviation', 'terminal')
            else (row['position'], row['type'], row.get('parent_position'), row.get('move'))
        )
        groups[identity].append(row)
    between = within = 0.0
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
    return dict(
        status=Status.RESOLVED,
        mean_score=mean,
        resolved_mass=known,
        standard_deviation=math.sqrt(max(0.0, total)),
        between_position_deviation=math.sqrt(max(0.0, between)),
        within_position_deviation=math.sqrt(max(0.0, within)),
        stopping_positions=len(groups),
        variance=total,
    )


def move_decomposition(row: JsonObject) -> JsonObject:
    parent: Any
    move: Any
    continuation: Any
    parent, move, continuation = (row.get(k) for k in ('reference_score', 'move_database_score', 'move_score'))
    if any(v is None for v in (parent, move, continuation)):
        return dict(database_move_gain_pp=None, continuation_gain_pp=None, total_gain_pp=None)
    a, b, total = 100 * (move - parent), 100 * (continuation - move), 100 * (continuation - parent)
    if not math.isclose(a + b, total, abs_tol=1e-10):
        raise AssertionError('Own-move gain decomposition does not add up')
    return dict(database_move_gain_pp=a, continuation_gain_pp=b, total_gain_pp=total)


def database_table(
    data: JsonObject, position: Position, color: bool, prior: Sequence[float]
) -> tuple[list[str], np.ndarray]:
    """Legal moves and Dirichlet parameters for one cached position; reusable across sampling batches."""
    moves = sorted(children(position))
    rows = {r['uci']: counts(r) for r in data['moves']}
    residual = validate(data, position)
    observations = [rows.get(move, [0, 0, 0]) for move in moves]
    if sum(residual):
        observations.append(residual)
    table_prior = np.asarray(prior if color else prior[::-1])
    return moves, np.asarray(observations, dtype=float) + table_prior / len(observations)


def parent_moments(
    k: Position,
    origins: Sequence[tuple[Position, str]] | None,
    evidence: Evidence,
    color: bool,
    prior: Sequence[float],
) -> tuple[float, float] | None:
    """Mean and variance of an own-turn position's database score, as in model.position_counts: the opponent
    move rows leading to it, weighted by their games, or its own table where nothing leads to it. Each row is
    from a different table, so their scores are independent."""
    owner = np.array([1.0, 0.5, 0.0] if color else [0.0, 0.5, 1.0])
    if origins:
        parts = []
        for parent, move in origins:
            moves, alpha = database_table(evidence[parent], parent, color, prior)
            parts.append((sum(move_counts(evidence[parent], move)), row_moments(alpha[moves.index(move)], owner)))
        total = sum(n for n, _ in parts)
        if not total:
            return None
        mean = sum(n * m for n, (m, _) in parts) / total
        return mean, sum((n / total) ** 2 * v for n, (_, v) in parts)
    if k not in evidence:
        return None
    _, alpha = database_table(evidence[k], k, color, prior)
    cells = np.broadcast_to(owner, alpha.shape)
    return float((alpha * cells).sum() / alpha.sum()), dirichlet_variance(alpha, cells)


class LocalComparisons:
    """Means and 95% intervals of local score changes, from one policy's exact posterior.

    A reply's or move's result split within its row is independent of how often it is played, so its
    own score has an exact Beta-like distribution. When a comparison involves such a score, its interval
    keeps that score's skewed shape, which matters for replies with only a few games.
    """

    def __init__(
        self, posterior: Posterior, database: Mapping[Position, tuple[float, float]], owner: np.ndarray
    ) -> None:
        self.posterior, self.database, self.owner = posterior, database, owner
        self.influence = posterior.influence()
        self._variance: dict[Position, float] = {}

    def value_variance(self, k: Position) -> float:
        if k not in self._variance:
            self._variance[k] = self.posterior.value_variance({k: 1.0})
        return self._variance[k]

    def leaf(self, k: Position) -> tuple[float, float] | None:
        """A position whose value is one table's own result score, or None."""
        node = self.posterior.model[k]
        if node.mode == 'stop' and k in self.posterior.alpha:
            return row_moments(self.posterior.alpha[k][0], self.owner)
        return None

    def opponent(self, k: Position, j: int) -> tuple[float, list[float]]:
        """Repertoire value before an opponent reply minus the value after it: mean and 95% interval."""
        posterior, branch = self.posterior, self.posterior.model[k].branches[j]
        before = posterior.values[k][COMPLETED]
        reach = self.influence[k]
        # The value before the reply already contains the reply's own share of the score after it.
        component: tuple[float, float] | None
        component, coefficient = None, posterior.sample[k][j][0] - 1
        if branch.target is not None:
            after = posterior.values[branch.target][COMPLETED]
            downstream = self.influence[branch.target]
            variance = sum(
                (reach.get(m, 0.0) - downstream.get(m, 0.0)) ** 2 * posterior.table_variance(m)
                for m in sorted(set(reach) | set(downstream))
            )
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

    def own(self, k: Position, target: Position) -> OwnComparison | None:
        """Parent database score, the selected move's database score and the continuation after it.

        The parent score pools the opponent move rows leading to the position (`database[k]`). The move's score
        is the whole table after it, which the continuation also uses, so those two covary. Tables upstream of
        the position are independent of everything after it.
        """
        if k not in self.database:
            return None
        parent, parent_variance = self.database[k]
        after = self.posterior.values[target][COMPLETED]
        after_variance = self.value_variance(target)
        drop = parent - after
        # Keep the skewed shape of whichever bounded score varies most: the parent's or a leaf continuation's.
        leaf = self.leaf(target)
        component, coefficient = (
            ((parent, parent_variance), 1.0) if leaf is None or parent_variance >= leaf[1] else (leaf, -1.0)
        )
        result: OwnComparison = dict(
            drop=(drop, comparison_interval(drop, parent_variance + after_variance, component, coefficient)),
            database_gain=None,
            continuation_gain=None,
        )
        if self.posterior.model[target].mode == 'opponent':
            alpha = self.posterior.alpha[target]
            owner = np.broadcast_to(self.owner, alpha.shape)
            selected = float((alpha * owner).sum() / alpha.sum())
            selected_variance = dirichlet_variance(alpha, owner)
            continuation_variance = (
                after_variance
                - self.posterior.table_variance(target)
                + dirichlet_variance(alpha, self.posterior.cells(target) - owner)
            )
            gain_component, gain_coefficient = (
                ((selected, selected_variance), 1.0)
                if selected_variance >= parent_variance
                else ((parent, parent_variance), -1.0)
            )
            result['database_gain'] = comparison_interval(
                selected - parent, parent_variance + selected_variance, gain_component, gain_coefficient
            )
            result['continuation_gain'] = comparison_interval(after - selected, continuation_variance)
        return result


def add_recursive_spreads(
    value: JsonObject,
    saved: JsonObject,
    supporting: Mapping[str, JsonObject],
    graph: Graph,
    evidence: Evidence,
    facts: dict[Position, JsonObject] | None = None,
) -> JsonObject:
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
    profiles: defaultdict[str, list[str]] = defaultdict(list)
    profiles['{}'].append('overall')
    for chapter in saved['chapters']:
        profiles[json.dumps(chapter.get('policy_overrides', {}), sort_keys=True)].append(chapter['id'])
    for identity, scopes in profiles.items():
        evaluator = Evaluator(graph, color, evidence, facts, dict(selected_policy(manifest), **json.loads(identity)))
        starts = {k: 1.0 for sid in scopes for k, p in preparations[sid]['starts'].items() if p > 0}
        evaluator.evaluate(starts)
        position_values = recursive_spread(evaluator)
        for sid in scopes:
            scope = insights[sid]
            result = spread_mixture(
                [(p, position_values[k]) for k, p in preparations[sid]['starts'].items() if p > 0],
                characters[sid].get('outcomes', {}).get('resolved_score'),
            )
            previous = scope['branch_score_spread']
            if previous.get('variance') is not None and (
                result['variance'] is None or not math.isclose(previous['variance'], result['variance'], abs_tol=1e-10)
            ):
                raise AssertionError('Recursive spread differs from the stopping-event ledger')
            if characters[sid].get('outcomes'):
                assert_outcomes(result, characters[sid]['outcomes'])
            previous.update(result, recursive=True)
            positions: dict[Position, JsonObject] = {}
            for row in characters[sid].get('positions', []):
                metric = position_values.get(row['position'])
                if metric is None:
                    metric = stopping_spread(row['outcomes'])
                assert_outcomes(metric, row['outcomes'])
                positions[row['position']] = metric
            scope['position_spreads'] = positions
            for row in moves[sid].get('all_signed_rows', []):
                after = (
                    position_values[row['target']]
                    if row['prepared']
                    else stopping_counts(
                        row['counts_white_draw_black'],
                        color,
                        row['move_score'] if row.get('score_basis') == 'terminal result' else None,
                    )
                )
                scope['moves'][row['id']].update(
                    move_spread=after,
                    reference_spread=position_values[row['position']] if row['kind'] == 'opponent' else None,
                )
        if identity == '{}':
            opening_spreads: dict[str, JsonObject] = {}
            for row in supporting['openings']['openings']:
                all_parts: list[tuple[float, JsonObject]]
                entries: dict[Position, JsonObject]
                all_parts, entries = [], {}
                for entry in row['entries']:
                    parts = [
                        (
                            origin['conditional_weight'],
                            position_values[entry['position']]
                            if origin['parent'] is None
                            else stopping_counts(origin['counts_white_draw_black'], color, origin['fixed_outcome']),
                        )
                        for origin in entry['origins']
                    ]
                    all_parts.extend(parts)
                    entry_spread = spread_mixture(
                        [(p / entry['conditional_weight'], child) for p, child in parts], entry['repertoire_score']
                    )
                    if len(parts) == 1:
                        child = parts[0][1]
                        entry_spread.update(
                            immediate_variance=child['immediate_variance'],
                            immediate_standard_deviation=child['immediate_standard_deviation'],
                        )
                    assert_outcomes(entry_spread, entry['outcomes'])
                    entries[entry['position']] = entry_spread
                result = spread_mixture(all_parts, row['repertoire_score'])
                assert_outcomes(result, row['outcomes'])
                opening_spreads[row['id']] = dict(branch_score_spread=result, entries=entries)
            value['opening_spreads'] = opening_spreads
    value['manifest'].update(
        spread_definition='B(s) = sum(p * (B(child) + (score(child) - score(s))**2)); known stopping scores have B=0. '
        'Forced own moves inherit the child. Shared canonical continuations are reused. '
        'Entry mixtures and the color mixture include between-entry score '
        'variance; standard deviations are never averaged. '
        'Immediate reply spread uses only the current opponent response distribution and recursive child scores. '
        'It is unavailable at own turns, stopping boundaries, or incomplete named reply tables. '
        'Sparse evidence remains included; unresolved evidence remains '
        'unavailable. Outcome volatility is the existing WDL variance.',
        spreads_refreshed_at=datetime.now(UTC).isoformat(),
    )
    value['validation'].update(recursive_spread_equals_stopping_ledger=True, outcome_variance_decomposition=True)
    return value


def analyze(path: str | Path, cache: str | Path = DEFAULT_CACHE) -> JsonObject:
    analysis = AnalysisContext(path, ('preparation', 'character', 'vulnerabilities', 'openings'))
    saved, manifest, supporting = analysis.saved, analysis.manifest, analysis.companions
    char = {s['id']: s for s in supporting['character']['scopes']}
    prep = {s['id']: s for s in supporting['preparation']['scopes']}
    moves = supporting['vulnerabilities']
    scopes: list[JsonObject] = [
        dict(id='overall', score=saved['overall'], moves=moves['overall'], overrides={}),
        *[
            dict(
                id=c['id'],
                score=c['score'],
                moves=next(s for s in moves['chapters'] if s['id'] == c['id']),
                overrides=c.get('policy_overrides', {}),
            )
            for c in saved['chapters']
        ],
    ]
    results: dict[str, JsonObject] = {}
    for scope in scopes:
        sid = scope['id']
        results[sid] = dict(
            id=sid,
            gaps=gap_priorities(char[sid].get('gap_coverage')),
            branch_score_spread=branch_score_spread(
                prep[sid].get('stops', []), scope['score'].get('raw_empirical_score')
            ),
            moves={},
        )
    graph, color = analysis.graph, analysis.color
    default_transitions = resolve(graph, color, analysis.policy)
    compared = moves['manifest']['evidence']
    evidence = analysis.read_evidence(cache, list(dict(manifest['evidence'], **compared)), required=True)
    if any(analysis.provenance[k] != original for k, original in compared.items()):
        raise ValueError('Cached evidence changed since the vulnerability analysis; rebuild it first')
    chosen = {
        row['position'] for scope in scopes for row in scope['moves'].get('all_signed_rows', []) if row['kind'] == 'own'
    }
    origins = arrival_moves(graph, color, evidence)
    database = {
        k: moments
        for k in chosen
        if (moments := parent_moments(k, origins.get(k), evidence, color, manifest['prior'])) is not None
    }
    owner = np.array([1.0, 0.5, 0.0] if color else [0.0, 0.5, 1.0])
    facts = chess_facts(graph, color, evidence)
    exact = {k: r['exact_name'] for k, r in supporting['openings']['positions'].items() if r.get('exact_name')}
    chapters = {c['id']: c for c in saved['chapters']}
    profiles: defaultdict[str, list[JsonObject]] = defaultdict(list)
    for scope in scopes:
        profiles[json.dumps(scope['overrides'], sort_keys=True)].append(scope)
    roots = [*manifest['root_weights'], *[e['position'] for c in saved['chapters'] for e in c['entries']]]
    for identity, group in profiles.items():
        transitions = (
            default_transitions
            if identity == '{}'
            else resolve(graph, color, dict(selected_policy(manifest), **json.loads(identity)))
        )
        order = topology(transitions, roots)
        model = prepare(graph, transitions, order, color, evidence)
        comparisons = LocalComparisons(
            Posterior(model, order, color, manifest['prior'], manifest['sparse_threshold']), database, owner
        )
        branches = {k: {b.move: j for j, b in enumerate(n.branches) if b.move} for k, n in model.items()}
        evaluator = Evaluator(graph, color, evidence, facts, dict(selected_policy(manifest), **json.loads(identity)))
        for scope in group:
            if scope['id'] != 'overall':
                chapter = chapters[scope['id']]
                entries = [e['position'] for e in chapter['entries']]
                flows = name_flow(evaluator, manifest['root_weights'], exact, stop_at=entries)
                entry_sources: dict[Position, list[JsonObject]] = {}
                for entry in chapter['entries']:
                    weights = flows.get(entry['position'], {})
                    total = sum(weights.values())
                    if total:
                        entry_sources[entry['position']] = [
                            dict(id=name, share=p / total)
                            for name, p in sorted(weights.items(), key=lambda item: (-item[1], item[0] or ''))
                            if p > 0
                        ]
                results[scope['id']]['entry_opening_sources'] = entry_sources
            for row in scope['moves'].get('all_signed_rows', []):
                k, move = row['position'], row['move']
                j = branches[k][move]
                if row['kind'] == 'own':
                    parts = comparisons.own(k, cast(Position, model[k].branches[j].target))
                    drop = None if parts is None else parts['drop'][1]
                else:
                    _, drop = comparisons.opponent(k, j)
                entry = dict(
                    local_drop_interval_pp=drop, local_gain_interval_pp=None if drop is None else [-drop[1], -drop[0]]
                )
                if row['kind'] == 'own':
                    entry.update(
                        move_decomposition(row),
                        database_move_gain_interval_pp=None if parts is None else parts['database_gain'],
                        continuation_gain_interval_pp=None if parts is None else parts['continuation_gain'],
                    )
                results[scope['id']]['moves'][row['id']] = entry
        del comparisons, model
    analysis.require_source('report insight analysis')
    result: JsonObject = dict(
        color=saved['color'],
        scopes=list(results.values()),
        manifest=analysis.companion_manifest(
            supporting_sha256=analysis.companion_hashes,
            prior=manifest['prior'],
            uncertainty_method=UNCERTAINTY_METHOD,
            interval_definition='Approximate prior-completed 95% local model intervals: exact '
            'posterior means and first-order variances from each cached Dirichlet '
            'table, combined through shared transposition values. The database score '
            'after an own move and the continuation share one table, so their covariance is '
            'included. Policies and population are fixed; historical game overlap '
            'and selection effects are not modeled. Weighted rankings use '
            'empirical reach.',
            spread_definition='Weighted standard deviation of stopping-event expected scores. Total '
            'variance equals between-canonical-position variance plus '
            'within-position incoming-evidence-cohort variance. No cutoff, sparse '
            'filter, prior or tunable parameter.',
        ),
        validation=dict(
            source_pgn_unchanged=True,
            saved_scores_reproduced=True,
            transposed_variance_decomposition=True,
            own_gain_decomposition=True,
        ),
    )
    return add_recursive_spreads(result, saved, supporting, graph, evidence, facts)


def main() -> None:
    stage_main('insights', analyze, __doc__)


if __name__ == '__main__':
    main()
