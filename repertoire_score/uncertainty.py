"""How much the scores could move because the Lichess database is a finite sample, computed without simulation.

Each cached position's results are given a Dirichlet posterior: its observed move/result counts plus the
configured prior. Every score is a sum over paths of products of probabilities taken from different
positions (a path visits a position at most once), so:

- Posterior means are exact: one backward pass with each table replaced by its posterior mean.
- Variances are first-order: each table's own contribution, weighted by its squared influence on the
  score. Interactions between two tables are omitted; they shrink with the product of both sample sizes.
- 95% intervals match a Beta distribution to a score's mean and variance (or a normal one for differences).
"""
import math
from functools import lru_cache

import numpy as np

from .evaluate import COMPLETED, KNOWN, LEAF, DEVIATION, OTHER, UNKNOWN, backward, can_enter

Z95 = 1.959963984540054
METHOD = ('Dirichlet posterior per cached table (observed counts plus the configured prior): exact posterior means, '
          'first-order variances that omit interactions between tables, and Beta-moment 95% intervals.')


def dirichlet_variance(alpha, gradient):
    """Variance of sum(theta * gradient) when theta ~ Dirichlet(alpha)."""
    total = alpha.sum()
    weights = alpha / total
    mean = float((weights * gradient).sum())
    return max(0., (float((weights * gradient * gradient).sum()) - mean * mean) / (total + 1))


def _continued_fraction(a, b, x):
    tiny, c, d = 1e-300, 1., 1. - (a + b) * x / (a + 1)
    d = 1. / (d if abs(d) > tiny else tiny)
    result = d
    for m in range(1, 10000):
        for numerator in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                          -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1. + numerator * d
            d = 1. / (d if abs(d) > tiny else tiny)
            c = 1. + numerator / c
            c = c if abs(c) > tiny else tiny
            result *= d * c
        if abs(d * c - 1.) < 1e-14:
            break
    return result


def beta_cdf(x, a, b):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0:
        return 0.
    if x >= 1:
        return 1.
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _continued_fraction(a, b, x) / a
    return 1. - front * _continued_fraction(b, a, 1. - x) / b


@lru_cache(maxsize=65536)
def beta_quantile(p, a, b):
    low, high = 0., 1.
    for _ in range(60):
        middle = (low + high) / 2
        low, high = (middle, high) if beta_cdf(middle, a, b) < p else (low, middle)
    return (low + high) / 2


def score_interval(mean, variance):
    """Approximate 95% interval for a score between 0 and 1 with the given mean and variance."""
    if variance <= 0 or not 0 < mean < 1:
        return [mean, mean]
    concentration = mean * (1 - mean) / variance - 1
    if concentration <= 0:
        sd = math.sqrt(variance)
        return [max(0., mean - Z95 * sd), min(1., mean + Z95 * sd)]
    a, b = mean * concentration, (1 - mean) * concentration
    if min(a, b) > 1000:
        # Practically normal; the series for the Beta quantile also converges slowly here.
        sd = math.sqrt(variance)
        return [mean - Z95 * sd, mean + Z95 * sd]
    return [beta_quantile(.025, a, b), beta_quantile(.975, a, b)]


def difference_interval(mean, variance, scale=100.):
    """Approximate 95% interval for a difference of scores, in points of `scale`."""
    sd = math.sqrt(max(variance, 0.))
    return [scale * (mean - Z95 * sd), scale * (mean + Z95 * sd)]


def row_moments(row, owner):
    """Mean and exact variance of one row's owner score, given its Dirichlet parameters (white/draw/black)."""
    total = row.sum()
    weights = row / total
    mean = float(weights @ owner)
    return mean, max(0., (float(weights @ (owner * owner)) - mean * mean) / (total + 1))


def comparison_interval(mean, variance, component=None, coefficient=1., scale=100.):
    """95% interval for a difference of scores, in points of `scale`.

    `component` is (mean, variance) of one bounded score whose fluctuations enter the difference times
    `coefficient`. Its skewed Beta shape is kept; the rest of the variance is treated as normal and
    combined in quadrature.
    """
    if component is None:
        return difference_interval(mean, variance, scale)
    low, high = score_interval(*component)
    below, above = abs(coefficient) * (component[0] - low), abs(coefficient) * (high - component[0])
    if coefficient < 0:
        below, above = above, below
    rest = Z95 * Z95 * max(variance - coefficient * coefficient * component[1], 0.)
    return [scale * (mean - math.sqrt(below * below + rest)), scale * (mean + math.sqrt(above * above + rest))]


def summary(mean, variance):
    """The saved `posterior` block for a score: mean component vector (see evaluate) and score variance."""
    unresolved = float(mean[UNKNOWN])
    return {
        "mean": float(mean[COMPLETED]),
        "standard_deviation": math.sqrt(max(variance, 0.)),
        "credible_interval_95": score_interval(float(mean[COMPLETED]), variance),
        "label": ("prior-completed estimate; missing distributions stop unresolved without assuming opponent play"
                  if unresolved > 0 else "posterior estimate"),
        "unresolved_mass_mean": unresolved,
        "resolved_contribution_mean": float(mean[KNOWN]),
        "conditional_bounds_mean": [float(mean[KNOWN]), float(mean[KNOWN] + mean[UNKNOWN])],
        "masses_mean": {"theory_leaf": float(mean[LEAF]), "deviation": float(mean[DEVIATION]),
                        "other_stop": float(mean[OTHER]), "unresolved": unresolved},
        "method": METHOD,
    }


class Posterior:
    """Posterior means and first-order variances for one policy's model (see model.prepare)."""

    def __init__(self, model, order, color, prior, sparse_threshold):
        self.model, self.order = model, order
        self.owner = np.array([1., .5, 0.] if color else [0., .5, 1.])
        # The prior is in owner win/draw/loss order; tables use white/draw/black.
        table_prior = np.asarray(prior if color else prior[::-1], dtype=float)
        self.alpha, self.sample = {}, {}
        for k in order:
            node = model[k]
            if node.mode == 'own':
                self.sample[k] = [(b.weight, None) for b in node.branches]
            elif node.mode == 'opponent':
                # The prior's total strength is shared across all legal moves and any residual bucket.
                alpha = np.asarray([b.counts for b in node.branches], dtype=float) + table_prior / len(node.branches)
                rows, total = alpha.sum(axis=1), alpha.sum()
                self.alpha[k] = alpha
                self.sample[k] = [(float(rows[j] / total),
                                   b.fixed_score if b.fixed_score is not None else float(alpha[j] @ self.owner / rows[j]))
                                  for j, b in enumerate(node.branches)]
            else:
                b = node.branches[0]
                if b.fixed_score is not None:
                    self.sample[k] = [(1., b.fixed_score)]
                else:
                    alpha = np.asarray(b.counts, dtype=float)[None, :] + table_prior
                    self.alpha[k] = alpha
                    self.sample[k] = [(1., float(alpha[0] @ self.owner / alpha.sum()))]
        # Exact posterior means of every component, because each path uses each table at most once.
        self.values = backward(model, order, self.sample, sparse_threshold)
        self._cells, self._variance, self._influence = {}, {}, None

    def cells(self, k):
        """The value each cell of table k contributes, at posterior means (rows x white/draw/black)."""
        if k not in self._cells:
            node = self.model[k]
            if node.mode == 'stop':
                cells = self.owner[None, :]
            else:
                cells = np.array([np.full(3, self.values[b.target][COMPLETED, 0]) if b.target is not None
                                  else np.full(3, b.fixed_score) if b.fixed_score is not None else self.owner
                                  for b in node.branches])
            self._cells[k] = cells
        return self._cells[k]

    def table_variance(self, k):
        if k not in self._variance:
            self._variance[k] = dirichlet_variance(self.alpha[k], self.cells(k))
        return self._variance[k]

    def influence(self):
        """For each position, the mean probability of reaching each table position from it."""
        if self._influence is None:
            result = {}
            for k in self.order:
                reach = {k: 1.} if k in self.alpha else {}
                for b, (p, _) in zip(self.model[k].branches, self.sample[k]):
                    if b.target is not None and p:
                        for m, value in result[b.target].items():
                            reach[m] = reach.get(m, 0.) + p * value
                result[k] = reach
            self._influence = result
        return self._influence

    def combined_influence(self, starts):
        influence, total = self.influence(), {}
        for k, weight in starts.items():
            for m, value in influence[k].items():
                total[m] = total.get(m, 0.) + weight * value
        return total

    def value_variance(self, starts):
        """Variance of a fixed mixture of position values, such as the repertoire root."""
        return sum(c * c * self.table_variance(m) for m, c in self.combined_influence(starts).items())

    def mixture(self, starts):
        mean = sum((w * self.values[k][:, 0] for k, w in starts.items()), np.zeros(8))
        return summary(mean, self.value_variance(starts))

    def gradient(self, roots, entries=None):
        """A score's posterior mean and its sensitivity to every table, as cell arrays.

        Without `entries`, the score is the fixed mixture `roots` of position values. With `entries`, it
        is the score conditional on first entering one of them, whose weights are themselves uncertain.
        Returns None when no entry can be reached.
        """
        if entries is None:
            mean = sum((w * self.values[k][:, 0] for k, w in roots.items()), np.zeros(8))
            cells = {m: c * self.cells(m) for m, c in self.combined_influence(roots).items()}
            return dict(mean=mean, entry_probability=1., cells=cells)
        entries = set(entries)
        entering = can_enter(self.model, self.order, entries)
        upstream = dict.fromkeys(self.order, 0.)
        for k, weight in roots.items():
            upstream[k] += weight
        for k in reversed(self.order):
            if k in entries or not entering[k] or not upstream[k]:
                continue
            for b, (p, _) in zip(self.model[k].branches, self.sample[k]):
                if b.target is not None:
                    upstream[b.target] += upstream[k] * p
        starts = {k: upstream[k] for k in entries if upstream[k] > 0}
        probability = sum(starts.values())
        if probability <= 0:
            return None
        mean = sum((w * self.values[k][:, 0] for k, w in starts.items()), np.zeros(8)) / probability
        score = mean[COMPLETED]
        # Value carried through the first entry (G) and probability of entering (H) from each position.
        carried, enters = {}, {}
        for k in self.order:
            if not entering[k]:
                continue
            if k in entries:
                carried[k], enters[k] = self.values[k][COMPLETED, 0], 1.
            else:
                pairs = [(p, b.target) for b, (p, _) in zip(self.model[k].branches, self.sample[k]) if b.target is not None]
                carried[k] = sum(p * carried.get(t, 0.) for p, t in pairs)
                enters[k] = sum(p * enters.get(t, 0.) for p, t in pairs)
        downstream = self.combined_influence(starts)
        cells = {}
        for m in set(downstream) | {k for k, w in upstream.items() if w and k in self.alpha and k not in entries}:
            gradient = downstream.get(m, 0.) * self.cells(m)
            if m not in entries and upstream[m] and self.model[m].mode == 'opponent':
                # Changing this table moves probability between entries and away from the chapter.
                rows = [carried.get(b.target, 0.) - score * enters.get(b.target, 0.) if b.target is not None else 0.
                        for b in self.model[m].branches]
                gradient = gradient + upstream[m] * np.asarray(rows)[:, None]
            cells[m] = gradient / probability
        return dict(mean=mean, entry_probability=probability, cells=cells)

    def variance(self, gradient):
        return sum(dirichlet_variance(self.alpha[m], g) for m, g in gradient['cells'].items())

    def chapter(self, roots, entries):
        """Score conditional on first entering one of `entries`, whose weights are themselves uncertain."""
        gradient = self.gradient(roots, entries)
        if gradient is None:
            return dict(entry_probability=0., summary=None)
        return dict(entry_probability=gradient['entry_probability'],
                    summary=summary(gradient['mean'], self.variance(gradient)))

    def stop_interval(self, k, j):
        """Posterior mean and 95% interval of one stopping event's score, from its own result row."""
        branch = self.model[k].branches[j]
        if branch.fixed_score is not None:
            return branch.fixed_score, [branch.fixed_score, branch.fixed_score]
        mean, variance = row_moments(self.alpha[k][j if self.model[k].mode == 'opponent' else 0], self.owner)
        return mean, score_interval(mean, variance)


def paired_variance(a, a_gradient, b, b_gradient, a_scale=1., b_scale=1.):
    """Variance of b_scale * (score b) - a_scale * (score a) for two policies over the same cached tables.

    The scales allow a transformed difference, such as a logit (centipawn) change, by the delta method.
    """
    total = 0.
    for m in set(a_gradient['cells']) | set(b_gradient['cells']):
        alpha = b.alpha[m] if m in b.alpha else a.alpha[m]
        if m in a.alpha and m in b.alpha and not np.array_equal(a.alpha[m], b.alpha[m]):
            raise ValueError('Paired scores use different evidence for the same position')
        gradient = 0.
        if m in b_gradient['cells']:
            gradient = gradient + b_scale * b_gradient['cells'][m]
        if m in a_gradient['cells']:
            gradient = gradient - a_scale * a_gradient['cells'][m]
        total += dirichlet_variance(alpha, np.broadcast_to(gradient, alpha.shape))
    return total
