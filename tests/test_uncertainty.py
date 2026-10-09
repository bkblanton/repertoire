import math

import numpy as np
import pytest
from helpers import data, graph, position, setup

from repertoire.evaluate import COMPLETED, UNKNOWN, backward, forward
from repertoire.report_insights import LocalComparisons, database_table, parent_moments
from repertoire.uncertainty import beta_quantile, dirichlet_variance, row_moments, score_interval

PGN = '1. Nf3 d5 2. g3 Nf6 3. Bg2 (3. Bh3) *\n\n1. g3 Nf6 2. Nf3 d5 3. Bg2 *'


def evidence():
    # Moderate counts, two transposed routes into one board, deviations and sparse leaves.
    return {
        position('Nf3'): data(
            61, 20, 40, [('d7d5', 40, 15, 25), ('g8f6', 12, 3, 9), ('c7c5', 8, 2, 6), ('b7b6', 1, 0, 0)]
        ),
        position('Nf3 d5 g3'): data(50, 10, 40, [('g8f6', 30, 6, 24), ('c7c6', 20, 4, 16)]),
        position('g3'): data(40, 10, 50, [('g8f6', 30, 6, 30), ('d7d5', 10, 4, 20)]),
        position('g3 Nf6 Nf3'): data(45, 15, 40, [('d7d5', 25, 10, 25), ('g7g6', 20, 5, 15)]),
        position('Nf3 d5 g3 Nf6 Bg2'): data(30, 5, 25),
        position('Nf3 d5 g3 Nf6 Bh3'): data(3, 1, 4),
    }


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
            sample[k] = [(1.0, scores[:, 0])]
        else:
            sample[k] = [
                (rows[:, j], b.fixed_score if b.fixed_score is not None else scores[:, j])
                for j, b in enumerate(node.branches)
            ]
    return sample


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    g = graph(tmp_path_factory.mktemp('uncertainty'), PGN)
    root = g.roots[0]
    model, order, raw, posterior, values, _ = setup(g, True, evidence(), {root: {'g1f3': 0.6, 'g2g3': 0.4}})
    rng = np.random.default_rng(3)
    n = 40000
    sample = simulate(posterior, rng, n)
    return dict(
        g=g,
        root=root,
        model=model,
        order=order,
        posterior=posterior,
        sample=sample,
        n=n,
        simulated=backward(model, order, sample, 30, n),
    )


def test_posterior_means_are_exact_and_variances_match_simulation(fixture):
    posterior, simulated, root, n = fixture['posterior'], fixture['simulated'], fixture['root'], fixture['n']
    for k in (root, position('Nf3'), position('g3'), position('Nf3 d5 g3 Nf6')):
        draws = simulated[k][COMPLETED]
        exact = posterior.values[k][COMPLETED]
        assert abs(draws.mean() - exact) < 4 * draws.std() / math.sqrt(n)
        assert posterior.value_variance({k: 1.0}) == pytest.approx(draws.var(), rel=0.06)
    summary = posterior.mixture({root: 1.0})
    low, high = np.quantile(simulated[root][COMPLETED], [0.025, 0.975])
    assert summary['credible_interval_95'] == pytest.approx([low, high], abs=0.003)
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
    _, masses = forward(model, order, sample, {root: 1.0}, stop_at=entries)
    total = sum(masses.values())
    draws = sum(masses[k] * fixture['simulated'][k][COMPLETED] for k in masses) / total
    chapter = posterior.chapter({root: 1.0}, entries)
    assert chapter['entry_probability'] == pytest.approx(total.mean(), abs=4 * total.std() / math.sqrt(n))
    assert chapter['summary']['mean'] == pytest.approx(draws.mean(), abs=0.002)
    assert chapter['summary']['standard_deviation'] ** 2 == pytest.approx(draws.var(), rel=0.08)


def test_local_comparisons_match_simulation(fixture):
    posterior, model, sample, simulated = (fixture[k] for k in ('posterior', 'model', 'sample', 'simulated'))
    comparisons = LocalComparisons(posterior, {}, posterior.owner)
    k = position('Nf3')
    checked = 0
    # A prepared reply, an unprepared reply with 16 games, and one with a single game, whose skewed
    # three-outcome score is only approximately Beta-shaped.
    tolerance = {'d7d5': 0.5, 'c7c5': 2.5, 'b7b6': 5.0}
    for j, branch in enumerate(model[k].branches):
        if branch.move not in tolerance:
            continue
        after = simulated[branch.target][COMPLETED] if branch.target else sample[k][j][1]
        drop = simulated[k][COMPLETED] - after
        mean, interval = comparisons.opponent(k, j)
        assert mean == pytest.approx(drop.mean(), abs=0.003)
        assert interval == pytest.approx(100 * np.quantile(drop, [0.025, 0.975]), abs=tolerance[branch.move])
        checked += 1
    assert checked == 3


def test_parent_moments_pool_arrival_rows_by_games():
    prior, owner = [0.5] * 3, np.array([1.0, 0.5, 0.0])
    first, second = position(''), position('e4 e5')
    evidence = {
        first: data(40, 10, 50, [('e2e4', 20, 5, 5), ('d2d4', 20, 5, 45)]),
        second: data(30, 0, 70, [('g1f3', 5, 0, 5), ('b1c3', 25, 0, 65)]),
    }
    parts = []
    for k, move in ((first, 'e2e4'), (second, 'g1f3')):
        moves, alpha = database_table(evidence[k], k, True, prior)
        parts.append(row_moments(alpha[moves.index(move)], owner))
    mean, variance = parent_moments('own', [(first, 'e2e4'), (second, 'g1f3')], evidence, True, prior)
    # Weighted by their 30 and 10 games; rows from different tables are independent.
    assert mean == pytest.approx(0.75 * parts[0][0] + 0.25 * parts[1][0])
    assert variance == pytest.approx(0.75**2 * parts[0][1] + 0.25**2 * parts[1][1])
    # With nothing leading to it, a position uses its own whole table.
    _, alpha = database_table(evidence[first], first, True, prior)
    whole = parent_moments(first, None, evidence, True, prior)
    assert whole == pytest.approx(
        ((alpha / alpha.sum() * owner).sum(), dirichlet_variance(alpha, np.broadcast_to(owner, alpha.shape)))
    )
    assert parent_moments('own', [(first, 'c2c4')], evidence, True, prior) is None
    assert parent_moments(second, None, {}, True, prior) is None


def test_beta_quantiles_and_intervals():
    for p in (0.025, 0.5, 0.975):
        assert beta_quantile(p, 1.0, 1.0) == pytest.approx(p, abs=1e-12)
        assert beta_quantile(p, 3.0, 1.0) == pytest.approx(p ** (1 / 3), abs=1e-12)
        assert beta_quantile(p, 1.0, 4.0) == pytest.approx(1 - (1 - p) ** 0.25, abs=1e-12)
    assert score_interval(0.5, 0.0) == [0.5, 0.5]
    low, high = score_interval(0.9, 0.002)
    assert 0 < low < 0.9 < high < 1 and high - 0.9 < 0.9 - low
    # Very large samples use the normal approximation.
    assert score_interval(0.5, 1e-8) == pytest.approx([0.5 - 1.96e-4, 0.5 + 1.96e-4], abs=1e-6)


def test_paired_policy_difference_matches_simulation(fixture):
    from repertoire.uncertainty import paired_variance

    root, n = fixture['root'], fixture['n']
    a = fixture['posterior']
    model, order, _, b, _, _ = setup(fixture['g'], True, evidence(), {root: {'g1f3': 0.1, 'g2g3': 0.9}})
    # Same evidence draws, different own-move weights: the comparison shares every table.
    sample = {k: b.sample[k] if model[k].mode == 'own' else fixture['sample'][k] for k in order}
    difference = backward(model, order, sample, 30, n)[root][COMPLETED] - fixture['simulated'][root][COMPLETED]
    first, second = a.gradient({root: 1.0}), b.gradient({root: 1.0})
    assert second['mean'][COMPLETED] - first['mean'][COMPLETED] == pytest.approx(
        difference.mean(), abs=4 * difference.std() / math.sqrt(n)
    )
    assert paired_variance(a, first, b, second) == pytest.approx(difference.var(), rel=0.08)
    # Pairing removes most of the shared variance.
    assert paired_variance(a, first, b, second) < a.variance(first) + b.variance(second)


@pytest.mark.parametrize(
    'rows, tolerance', [([('e2e4', 60, 10, 30), ('d2d4', 60, 10, 30)], 1.5), ([('e2e4', 5, 1, 2)], 4.0)]
)
def test_own_move_comparisons_match_simulation(fixture, rows, tolerance):
    # The parent score is the opponent's 1.e4 row leading to the position; the second case is a sparse row,
    # whose skewed score makes the drop only roughly normal. Report tables filter such rows. The move's database
    # score is the table after it, which the continuation shares.
    posterior, sample, simulated, n = fixture['posterior'], fixture['sample'], fixture['simulated'], fixture['n']
    k, target, before = 'parent', position('Nf3'), position('')
    totals = [sum(r[i] for r in rows) for i in (1, 2, 3)]
    evidence = {before: data(*totals, rows)}
    database = {k: parent_moments(k, [(before, 'e2e4')], evidence, True, [0.5] * 3)}
    moves, alpha = database_table(evidence[before], before, True, [0.5] * 3)
    theta = np.random.default_rng(5).dirichlet(alpha.ravel(), size=n).reshape(n, *alpha.shape)
    row = theta[:, moves.index('e2e4')]
    parent = row @ posterior.owner / row.sum(axis=1)
    selected = sum(p * s for p, s in sample[target])
    after = simulated[target][COMPLETED]
    parts = LocalComparisons(posterior, database, posterior.owner).own(k, target)
    assert parts is not None
    for interval, draws in (
        (parts['drop'][1], parent - after),
        (parts['database_gain'], selected - parent),
        (parts['continuation_gain'], after - selected),
    ):
        assert interval == pytest.approx(100 * np.quantile(draws, [0.025, 0.975]), abs=tolerance)
