import math

import pytest

from repertoire_score.preparation import Evaluator, chess_facts, stopping_rows
from repertoire_score.report_insights import branch_score_spread
from repertoire_score.sharpness import recursive_wdl, summarize
from repertoire_score.spread import assert_outcomes, mixture, recursive_spread, stopping_counts
from helpers import data, graph, position


def test_same_outcome_volatility_can_hide_very_different_reply_spreads():
    uniform = mixture([(.5, stopping_counts([50, 0, 50], True))] * 2)
    uneven = mixture([(.5, stopping_counts([90, 0, 10], True)), (.5, stopping_counts([10, 0, 90], True))])
    assert uniform['mean_score'] == uneven['mean_score'] == .5
    assert uniform['outcome_variance'] == pytest.approx(uneven['outcome_variance'])
    assert uniform['variance'] == 0
    assert uneven['standard_deviation'] == pytest.approx(.4)
    assert uneven['variance'] + uneven['within_stop_outcome_variance'] == pytest.approx(.25)


@pytest.mark.parametrize('color', [True, False])
def test_recursion_includes_deeper_spread_even_when_immediate_scores_agree(tmp_path, color):
    if color:
        g = graph(tmp_path, '1. e4 e5 2. Nf3 Nc6 3. Bb5 *')
        parent, deep = position('e4'), position('e4 e5 Nf3')
        evidence = {parent: data(50, 0, 50, [('e7e5', 25, 0, 25), ('c7c5', 25, 0, 25)]),
            deep: data(50, 0, 50, [('b8c6', 45, 0, 5), ('d7d6', 5, 0, 45)]),
            position('e4 e5 Nf3 Nc6 Bb5'): data(90, 0, 10)}
    else:
        g = graph(tmp_path, '1. e4 c5 2. Nf3 Nc6 3. d4 *')
        parent, deep = position('e4 c5'), position('e4 c5 Nf3 Nc6')
        evidence = {g.roots[0]: data(50, 0, 50, [('e2e4', 50, 0, 50)]),
            parent: data(50, 0, 50, [('g1f3', 25, 0, 25), ('b1c3', 25, 0, 25)]),
            deep: data(50, 0, 50, [('d2d4', 5, 0, 45), ('f1b5', 45, 0, 5)]),
            position('e4 c5 Nf3 Nc6 d4'): data(10, 0, 90)}
    evaluator = Evaluator(g, color, evidence, chess_facts(g, color, evidence))
    evaluator.evaluate({g.roots[0]: 1.})
    result = recursive_spread(evaluator)
    root, local = result[g.roots[0]], result[parent]
    assert root['variance'] == local['variance'] == pytest.approx(.08)
    assert local['immediate_standard_deviation'] == pytest.approx(0.)
    assert local['future_variance'] == pytest.approx(.08)
    assert root['standard_deviation'] == pytest.approx(math.sqrt(.08))
    assert result[deep]['immediate_variance'] == pytest.approx(.16)
    for k, outcomes in recursive_wdl(evaluator).items():
        assert_outcomes(result[k], summarize(outcomes))
    gaps = ('e4 c5', 'e4 e5 Nf3 d6') if color else ('e4 c5 Nc3', 'e4 c5 Nf3 Nc6 Bb5')
    assert all(position(moves) not in evidence for moves in gaps)


def test_transpositions_reuse_values_and_recursive_spread_matches_stop_ledger(tmp_path):
    g = graph(tmp_path, '1. Nf3 d5 2. g3 Nf6 3. Bg2 *\n\n1. g3 Nf6 2. Nf3 d5 3. Bg2 *')
    evidence = {k: data(60, 0, 40, [(m, 60, 0, 40) for m in n.edges]) for k, n in g.nodes.items()}
    shared = position('Nf3 d5 g3 Nf6 Bg2')
    evidence[shared] = data(50, 0, 50, [('c7c5', 45, 0, 5), ('e7e6', 5, 0, 45)])
    evaluator = Evaluator(g, True, evidence, chess_facts(g, True, evidence),
                          {g.roots[0]: {'g1f3': .4, 'g2g3': .6}})
    evaluator.evaluate({g.roots[0]: 1.})
    values = recursive_spread(evaluator)
    assert len(values) == len(evaluator.values)
    assert values[shared]['immediate_standard_deviation'] == pytest.approx(.4)
    assert values[g.roots[0]]['standard_deviation'] == pytest.approx(.4)
    lines = {k: ' '.join(n.path) for k, n in g.nodes.items()}
    ledger = stopping_rows(evaluator, {g.roots[0]: 1.}, None, lines)
    assert branch_score_spread(ledger)['variance'] == pytest.approx(values[g.roots[0]]['variance'])


def test_entry_and_color_mixtures_include_between_mean_variance():
    a, b = stopping_counts([100, 0, 0], True), stopping_counts([0, 0, 100], True)
    result = mixture([(.5, a), (.5, b)])
    assert result['standard_deviation'] == .5
    assert result['outcome_variance'] == .25
    assert result['within_stop_outcome_variance'] == 0


def test_zero_data_is_unresolved_and_residual_does_not_claim_complete_reply_spread(tmp_path):
    g = graph(tmp_path, '1. e4 *')
    k = position('e4')
    evidence = {k: data(60, 0, 40, [('e7e5', 40, 0, 10)])}
    evaluator = Evaluator(g, True, evidence, chess_facts(g, True, evidence))
    evaluator.evaluate({g.roots[0]: 1.})
    values = recursive_spread(evaluator)
    assert values[k]['reply_distribution_coverage'] == .5
    assert values[k]['immediate_standard_deviation'] is None
    assert values[k]['standard_deviation'] == pytest.approx(.2)
    evidence[k] = data(0, 0, 0)
    evaluator = Evaluator(g, True, evidence, chess_facts(g, True, evidence))
    evaluator.evaluate({g.roots[0]: 1.})
    assert recursive_spread(evaluator)[k]['variance'] is None
    assert mixture([(.5, stopping_counts([0, 0, 0], True)), (.5, stopping_counts([1, 0, 0], True))])['variance'] is None


@pytest.mark.parametrize('fixed', [0., .5, 1.])
def test_terminal_scores_have_no_branch_or_outcome_variance(fixed):
    result = stopping_counts([0, 0, 0], False, fixed)
    assert result['variance'] == result['outcome_variance'] == 0
