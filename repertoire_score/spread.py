"""Recursive variation in prepared continuation scores, using empirical replies."""
import math

from .sharpness import stopping_wdl, summarize


def stopping_spread(outcomes):
    """A stopping score is fixed; game outcomes can still vary after preparation."""
    unresolved = outcomes['unresolved_probability']
    known = unresolved == 0
    return dict(mean_score=outcomes['resolved_score'] if known else None,
        variance=0. if known else None, standard_deviation=0. if known else None,
        within_stop_outcome_variance=outcomes['sharpness'] / 400 if known else None,
        outcome_variance=outcomes['sharpness'] / 400 if known else None,
        resolved_mass=1 - unresolved, immediate_variance=None, immediate_standard_deviation=None,
        basis='stopping score; no modeled future replies')


def stopping_counts(sample, color, fixed=None):
    return stopping_spread(summarize(stopping_wdl(sample, color, fixed)))


def mixture(parts, expected=None):
    """Mix variances and means; never average standard deviations or fill gaps."""
    parts = [(p, child) for p, child in parts if p > 0]
    mass = sum(p for p, _ in parts)
    known = sum(p * child['resolved_mass'] for p, child in parts)
    if not parts or not math.isclose(mass, 1., abs_tol=1e-10) or any(
            child['variance'] is None for _, child in parts):
        return dict(mean_score=None, variance=None, standard_deviation=None,
            within_stop_outcome_variance=None, outcome_variance=None,
            resolved_mass=known, immediate_variance=None, immediate_standard_deviation=None,
            basis='unresolved entry or stopping evidence')
    mean = sum(p * child['mean_score'] for p, child in parts)
    if expected is not None and not math.isclose(mean, expected, abs_tol=1e-10):
        raise AssertionError('Recursive spread did not reproduce the saved score')
    local = sum(p * (child['mean_score'] - mean) ** 2 for p, child in parts)
    future = sum(p * child['variance'] for p, child in parts)
    within = sum(p * child['within_stop_outcome_variance'] for p, child in parts)
    variance = local + future
    return dict(mean_score=mean, variance=variance, standard_deviation=math.sqrt(variance),
        local_variance=local, future_variance=future, within_stop_outcome_variance=within,
        outcome_variance=variance + within, resolved_mass=known,
        immediate_variance=None, immediate_standard_deviation=None, basis='recursive prepared continuations')


def recursive_spread(evaluator):
    """Follow the scorer's postorder; each transposition has one continuation."""
    values = {}
    for k, expected in evaluator.values.items():
        parts = [(p, values[target]) for _, p, target in evaluator.edges[k]]
        parts += [(p, stopping_counts(sample, evaluator.color, fixed))
                  for _, p, _, sample, fixed in evaluator.stops[k]]
        value = mixture(parts, float(expected[0]) if expected[1] == 0 else None)
        opponent = evaluator.facts[k]['turn'] != evaluator.color and evaluator.facts[k]['outcome'] is None
        reply_mass = sum(p for move, p, _ in evaluator.edges[k] if move is not None)
        reply_mass += sum(p for move, p, _, _, _ in evaluator.stops[k] if move is not None)
        value['reply_distribution_coverage'] = reply_mass if opponent else None
        if opponent and math.isclose(reply_mass, 1., abs_tol=1e-10) and value['variance'] is not None:
            value['immediate_variance'] = value['local_variance']
            value['immediate_standard_deviation'] = math.sqrt(value['local_variance'])
        values[k] = value
    return values


def assert_outcomes(value, outcomes):
    if value['variance'] is None:
        if outcomes['unresolved_probability'] == 0:
            raise AssertionError('Resolved WDL has unresolved recursive spread')
        return
    if not math.isclose(value['mean_score'], outcomes['resolved_score'], abs_tol=1e-10):
        raise AssertionError('Recursive spread and WDL scores differ')
    if not math.isclose(value['outcome_variance'], outcomes['sharpness'] / 400, abs_tol=1e-10):
        raise AssertionError('Law of total variance did not reproduce outcome volatility')
