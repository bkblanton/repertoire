"""Recursive variation in prepared continuation scores, using empirical replies."""

import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from .board_cache import position_of
from .evaluate import fold
from .model import Branch
from .schema import JsonObject, Position
from .sharpness import stopping_wdl, summarize

if TYPE_CHECKING:
    from .preparation import Evaluator


def stopping_spread(outcomes: JsonObject) -> JsonObject:
    """A stopping score is fixed; game outcomes can still vary after preparation."""
    unresolved = outcomes['unresolved_probability']
    known = unresolved == 0
    return dict(
        mean_score=outcomes['resolved_score'] if known else None,
        variance=0.0 if known else None,
        standard_deviation=0.0 if known else None,
        within_stop_outcome_variance=outcomes['sharpness'] / 400 if known else None,
        outcome_variance=outcomes['sharpness'] / 400 if known else None,
        resolved_mass=1 - unresolved,
        immediate_variance=None,
        immediate_standard_deviation=None,
        basis='stopping score; no modeled future replies',
    )


def stopping_counts(sample: Sequence[int], color: bool, fixed: float | None = None) -> JsonObject:
    return stopping_spread(summarize(stopping_wdl(sample, color, fixed)))


def mixture(parts: Iterable[tuple[float, JsonObject]], expected: float | None = None) -> JsonObject:
    """Mix variances and means; never average standard deviations or fill gaps."""
    parts = [(p, child) for p, child in parts if p > 0]
    mass = sum(p for p, _ in parts)
    known = sum(p * child['resolved_mass'] for p, child in parts)
    if not parts or not math.isclose(mass, 1.0, abs_tol=1e-10) or any(child['variance'] is None for _, child in parts):
        return dict(
            mean_score=None,
            variance=None,
            standard_deviation=None,
            within_stop_outcome_variance=None,
            outcome_variance=None,
            resolved_mass=known,
            immediate_variance=None,
            immediate_standard_deviation=None,
            basis='unresolved entry or stopping evidence',
        )
    mean = sum(p * child['mean_score'] for p, child in parts)
    if expected is not None and not math.isclose(mean, expected, abs_tol=1e-10):
        raise AssertionError('Recursive spread did not reproduce the saved score')
    local = sum(p * (child['mean_score'] - mean) ** 2 for p, child in parts)
    future = sum(p * child['variance'] for p, child in parts)
    within = sum(p * child['within_stop_outcome_variance'] for p, child in parts)
    variance = local + future
    return dict(
        mean_score=mean,
        variance=variance,
        standard_deviation=math.sqrt(variance),
        local_variance=local,
        future_variance=future,
        within_stop_outcome_variance=within,
        outcome_variance=variance + within,
        resolved_mass=known,
        immediate_variance=None,
        immediate_standard_deviation=None,
        basis='recursive prepared continuations',
    )


def recursive_spread(evaluator: 'Evaluator') -> dict[Position, JsonObject]:
    """Recursive score spread for every compiled position; each transposition has one continuation."""

    def combine(k: Position, parts: list[tuple[Branch, float, JsonObject]]) -> JsonObject:
        expected = evaluator.values[k]
        value = mixture([(p, child) for _, p, child in parts], float(expected[0]) if expected[1] == 0 else None)
        fact = evaluator.facts[position_of(k)]
        opponent = fact['turn'] != evaluator.color and fact['outcome'] is None
        reply_mass = sum(p for b, p, _ in parts if b.move is not None)
        value['reply_distribution_coverage'] = reply_mass if opponent else None
        if opponent and math.isclose(reply_mass, 1.0, abs_tol=1e-10) and value['variance'] is not None:
            value['immediate_variance'] = value['local_variance']
            value['immediate_standard_deviation'] = math.sqrt(value['local_variance'])
        return value

    return fold(
        evaluator.model,
        list(evaluator.values),
        evaluator.sampled,
        lambda b, _: stopping_counts(b.counts, evaluator.color, b.fixed_score),
        combine,
    )


def assert_outcomes(value: JsonObject, outcomes: JsonObject) -> None:
    if value['variance'] is None:
        if outcomes['unresolved_probability'] == 0:
            raise AssertionError('Resolved WDL has unresolved recursive spread')
        return
    if not math.isclose(value['mean_score'], outcomes['resolved_score'], abs_tol=1e-10):
        raise AssertionError('Recursive spread and WDL scores differ')
    if not math.isclose(value['outcome_variance'], outcomes['sharpness'] / 400, abs_tol=1e-10):
        raise AssertionError('Law of total variance did not reproduce outcome volatility')
