import math

import numpy as np
import pytest

from repertoire_score.evaluate import COMPLETED, UNKNOWN, backward, forward
from repertoire_score.report_insights import LocalComparisons, database_table
from repertoire_score.uncertainty import beta_quantile, dirichlet_variance, score_interval
from helpers import data, graph, position, setup

PGN = '1. Nf3 d5 2. g3 Nf6 3. Bg2 (3. Bh3) *\n\n1. g3 Nf6 2. Nf3 d5 3. Bg2 *'


def evidence():
    # Moderate counts, two transposed routes into one board, deviations and sparse leaves.
    return {position('Nf3'): data(61, 20, 40, [('d7d5', 40, 15, 25), ('g8f6', 12, 3, 9), ('c7c5', 8, 2, 6), ('b7b6', 1, 0, 0)]),
            position('Nf3 d5 g3'): data(50, 10, 40, [('g8f6', 30, 6, 24), ('c7c6', 20, 4, 16)]),
            position('g3'): data(40, 10, 50, [('g8f6', 30, 6, 30), ('d7d5', 10, 4, 20)]),
            position('g3 Nf6 Nf3'): data(45, 15, 40, [('d7d5', 25, 10, 25), ('g7g6', 20, 5, 15)]),
            position('Nf3 d5 g3 Nf6 Bg2'): data(30, 5, 25),
            position('Nf3 d5 g3 Nf6 Bh3'): data(3, 1, 4)}


def simulate(posterior, rng, n):
    """Brute-force Dirichlet draws of every table, in the branch format used by backward and forward."""
    sample = {}
    for k in posterior.order:
        node = posterior.model[k]
        if k not in posterior.alpha:
            sample[k] = posterior.sample[k]
            continue
        alpha = posterior.alpha[k]
        theta = rng.dirichlet(alpha.ravel(), size=n).reshape(n, *alpha.shape)
        rows = theta.sum(axis=2)
        scores = (theta @ posterior.owner) / rows
        if node.mode == 'stop':
            sample[k] = [(1., scores[:, 0])]
        else:
            sample[k] = [(rows[:, j], b.fixed_score if b.fixed_score is not None else scores[:, j])
                         for j, b in enumerate(node.branches)]
    return sample


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    g = graph(tmp_path_factory.mktemp('uncertainty'), PGN)
    root = g.roots[0]
    model, order, raw, posterior, values, _ = setup(g, True, evidence(), {root: {'g1f3': .6, 'g2g3': .4}})
    rng = np.random.default_rng(3)
    n = 40000
    sample = simulate(posterior, rng, n)
    return dict(g=g, root=root, model=model, order=order, posterior=posterior, sample=sample, n=n,
                simulated=backward(model, order, sample, 30, n))


def test_posterior_means_are_exact_and_variances_match_simulation(fixture):
    posterior, simulated, root, n = fixture['posterior'], fixture['simulated'], fixture['root'], fixture['n']
    for k in (root, position('Nf3'), position('g3'), position('Nf3 d5 g3 Nf6')):
        draws = simulated[k][COMPLETED]
        exact = posterior.values[k][COMPLETED]
        assert abs(draws.mean() - exact) < 4 * draws.std() / math.sqrt(n)
        assert posterior.value_variance({k: 1.}) == pytest.approx(draws.var(), rel=.06)
    summary = posterior.mixture({root: 1.})
    low, high = np.quantile(simulated[root][COMPLETED], [.025, .975])
    assert summary['credible_interval_95'] == pytest.approx([low, high], abs=.003)
    assert summary['mean'] == pytest.approx(posterior.values[root][COMPLETED])
    # Legal moves nobody has played get prior mass, and stop unresolved: the mean is exact too.
    unresolved = simulated[root][UNKNOWN]
    assert summary['unresolved_mass_mean'] == posterior.values[root][UNKNOWN] > 0
    assert abs(unresolved.mean() - summary['unresolved_mass_mean']) < 4 * unresolved.std() / math.sqrt(n)


def test_chapter_score_with_uncertain_entry_weights_matches_simulation(fixture):
    posterior, model, order, sample, n = (fixture[k] for k in ('posterior', 'model', 'order', 'sample', 'n'))
    root = fixture['root']
    # The shared board is reached from both first moves, so the entry weights are uncertain too.
    entries = [position('Nf3 d5 g3 Nf6'), position('g3 Nf6 Nf3 d5')]
    _, masses = forward(model, order, sample, {root: 1.}, stop_at=entries)
    total = sum(masses.values())
    draws = sum(masses[k] * fixture['simulated'][k][COMPLETED] for k in masses) / total
    chapter = posterior.chapter({root: 1.}, entries)
    assert chapter['entry_probability'] == pytest.approx(total.mean(), abs=4 * total.std() / math.sqrt(n))
    assert chapter['summary']['mean'] == pytest.approx(draws.mean(), abs=.002)
    assert chapter['summary']['standard_deviation'] ** 2 == pytest.approx(draws.var(), rel=.08)


def test_local_comparisons_match_simulation(fixture):
    posterior, model, sample, simulated = (fixture[k] for k in ('posterior', 'model', 'sample', 'simulated'))
    comparisons = LocalComparisons(posterior, {}, posterior.owner)
    k = position('Nf3')
    checked = 0
    # A prepared reply, an unprepared reply with 16 games, and one with a single game, whose skewed
    # three-outcome score is only approximately Beta-shaped.
    tolerance = {'d7d5': .5, 'c7c5': 2.5, 'b7b6': 5.}
    for j, branch in enumerate(model[k].branches):
        if branch.move not in tolerance:
            continue
        after = simulated[branch.target][COMPLETED] if branch.target else sample[k][j][1]
        drop = simulated[k][COMPLETED] - after
        mean, interval = comparisons.opponent(k, j)
        assert mean == pytest.approx(drop.mean(), abs=.003)
        assert interval == pytest.approx(100 * np.quantile(drop, [.025, .975]), abs=tolerance[branch.move])
        checked += 1
    assert checked == 3


def test_own_move_comparisons_include_the_shared_table_covariance():
    k = position('')
    table = data(120, 20, 60, [('e2e4', 60, 10, 30), ('d2d4', 60, 10, 30)])
    moves, alpha = database_table(table, k, True, [.5] * 3)
    owner = np.array([1., .5, 0.])
    parent = dirichlet_variance(alpha, np.broadcast_to(owner, alpha.shape))
    weights = alpha / alpha.sum()
    row = moves.index('e2e4')
    selected = alpha[row] @ owner / alpha[row].sum()
    gradient = np.zeros_like(alpha)
    gradient[row] = (owner - selected) / weights[row].sum()
    move = dirichlet_variance(alpha, gradient)
    difference = dirichlet_variance(alpha, gradient - np.broadcast_to(owner, alpha.shape))
    # The move's own games are part of the parent table, so the difference varies less than independent scores would.
    assert 0 < difference < parent + move
    _, reversed_alpha = database_table(table, k, False, [.5] * 3)
    black = np.array([0., .5, 1.])
    assert (weights * owner).sum() + (reversed_alpha / reversed_alpha.sum() * black).sum() == pytest.approx(1)


def test_beta_quantiles_and_intervals():
    for p in (.025, .5, .975):
        assert beta_quantile(p, 1., 1.) == pytest.approx(p, abs=1e-12)
        assert beta_quantile(p, 3., 1.) == pytest.approx(p ** (1 / 3), abs=1e-12)
        assert beta_quantile(p, 1., 4.) == pytest.approx(1 - (1 - p) ** .25, abs=1e-12)
    assert score_interval(.5, 0.) == [.5, .5]
    low, high = score_interval(.9, .002)
    assert 0 < low < .9 < high < 1 and high - .9 < .9 - low
    # Very large samples use the normal approximation.
    assert score_interval(.5, 1e-8) == pytest.approx([.5 - 1.96e-4, .5 + 1.96e-4], abs=1e-6)


def test_paired_policy_difference_matches_simulation(fixture):
    from repertoire_score.uncertainty import paired_variance
    root, n = fixture['root'], fixture['n']
    a = fixture['posterior']
    model, order, _, b, _, _ = setup(fixture['g'], True, evidence(), {root: {'g1f3': .1, 'g2g3': .9}})
    # Same evidence draws, different own-move weights: the comparison shares every table.
    sample = {k: b.sample[k] if model[k].mode == 'own' else fixture['sample'][k] for k in order}
    difference = backward(model, order, sample, 30, n)[root][COMPLETED] - fixture['simulated'][root][COMPLETED]
    first, second = a.gradient({root: 1.}), b.gradient({root: 1.})
    assert second['mean'][COMPLETED] - first['mean'][COMPLETED] == pytest.approx(difference.mean(), abs=4 * difference.std() / math.sqrt(n))
    assert paired_variance(a, first, b, second) == pytest.approx(difference.var(), rel=.08)
    # Pairing removes most of the shared variance.
    assert paired_variance(a, first, b, second) < a.variance(first) + b.variance(second)


@pytest.mark.parametrize('rows, tolerance', [([('e2e4', 60, 10, 30), ('d2d4', 60, 10, 30)], 1.5), ([('e2e4', 5, 1, 2)], 4.)])
def test_own_move_comparisons_match_simulation(fixture, rows, tolerance):
    # The second case is a sparse move that makes up every game at its parent; unplayed moves' skewed
    # prior-only scores make its database gain only roughly normal. Report tables filter such rows.
    posterior, simulated, n = fixture['posterior'], fixture['simulated'], fixture['n']
    k, target = 'parent', position('Nf3')
    totals = [sum(r[i] for r in rows) for i in (1, 2, 3)]
    database = {k: database_table(data(*totals, rows), position(''), True, [.5] * 3)}
    moves, alpha = database[k]
    theta = np.random.default_rng(5).dirichlet(alpha.ravel(), size=n).reshape(n, *alpha.shape)
    parent = theta.sum(axis=1) @ posterior.owner
    row = theta[:, moves.index('e2e4')]
    selected = row @ posterior.owner / row.sum(axis=1)
    after = simulated[target][COMPLETED]
    parts = LocalComparisons(posterior, database, posterior.owner).own(k, 'e2e4', target)
    for interval, draws in ((parts['drop'][1], parent - after), (parts['database_gain'], selected - parent),
                            (parts['continuation_gain'], after - selected)):
        assert interval == pytest.approx(100 * np.quantile(draws, [.025, .975]), abs=tolerance)
