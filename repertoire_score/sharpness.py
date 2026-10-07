"""Recursive repertoire WDL and normalized outcome variance, using saved evidence."""

import math

import numpy as np

from .evaluate import fold

FIELDS = ('win_probability', 'draw_probability', 'loss_probability', 'unresolved_probability')


def sharpness(win, draw, loss):
    """Variance of a 1 / 0.5 / 0 game score, scaled to its maximum of 100."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in (win, draw, loss)) or not math.isclose(
        win + draw + loss, 1.0, abs_tol=1e-9
    ):
        raise ValueError('Sharpness requires a complete normalized WDL distribution')
    # Equivalent to 400 * (win + draw / 4 - (win + draw / 2) ** 2),
    # but avoids cancellation near certain outcomes.
    return min(100.0, 100 * (4 * win * loss + draw * (win + loss)))


def stopping_wdl(counts, color, fixed=None):
    """Owner-relative WDL; absent observations retain unresolved probability."""
    if fixed is not None:
        if fixed not in (0.0, 0.5, 1.0):
            raise ValueError('A terminal result must be a loss, draw or win')
        return np.array([float(fixed == 1), float(fixed == 0.5), float(fixed == 0), 0.0])
    total = sum(counts)
    if not total:
        return np.array([0.0, 0.0, 0.0, 1.0])
    owner_counts = counts if color else counts[::-1]
    return np.array([*(n / total for n in owner_counts), 0.0])


def summarize(distribution):
    distribution = list(map(float, distribution))
    total = sum(distribution)
    if any(not math.isfinite(p) or p < 0 or p > 1 + 1e-9 for p in distribution) or not math.isclose(
        total, 1.0, abs_tol=1e-9
    ):
        raise AssertionError('Recursive WDL probability conservation failed')
    # Correct only floating-point accumulation error, including unresolved mass.
    win, draw, loss, unresolved = (p / total for p in distribution)
    result = dict(zip(FIELDS, (win, draw, loss, unresolved)))
    result['resolved_score'] = win + draw / 2
    result['sharpness'] = sharpness(win, draw, loss) if unresolved == 0 else None
    return result


def recursive_wdl(evaluator, values=None):
    """Win, draw, loss and unresolved probabilities for every compiled position, over the scorer's model.

    Pass the previous result as `values` to extend it after the evaluator compiles more positions.
    """

    def combine(position, parts):
        value = np.zeros(4)
        for _, probability, child in parts:
            value += probability * child
        expected = evaluator.values[position]
        if (
            not np.isclose(value.sum(), 1.0, atol=1e-9)
            or not math.isclose(value[0] + value[1] / 2, expected[0], abs_tol=1e-10)
            or not math.isclose(value[3], expected[1], abs_tol=1e-10)
        ):
            raise AssertionError('Recursive WDL did not reproduce repertoire score and unresolved mass')
        return value

    return fold(
        evaluator.model,
        list(evaluator.values),
        evaluator.sampled,
        lambda b, _: stopping_wdl(b.counts, evaluator.color, b.fixed_score),
        combine,
        values,
    )


def scope_outcomes(evaluator, starts, values=None):
    evaluator.evaluate(starts)
    if values is None:
        values = recursive_wdl(evaluator)
    return summarize(sum((weight * values[position] for position, weight in starts.items() if weight), np.zeros(4)))
