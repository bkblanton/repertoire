import math

import chess
import pytest

from repertoire_score.gaps import distribution
from repertoire_score.preparation import Evaluator, chess_facts
from test_model import graph, data, position


def evaluator(g, evidence, policy=None, color=True):
    return Evaluator(g, color, evidence, chess_facts(g, color, evidence), policy)


def test_endpoint_reply_distribution_and_chapter_weight(tmp_path):
    g = graph(tmp_path, '1. e4 *')
    evidence = {position('e4'): data(100, 0, 0, [('e7e5', 50, 0, 0), ('c7c5', 30, 0, 0), ('d7d5', 20, 0, 0)])}
    result = distribution(evaluator(g, evidence), {g.roots[0]: 1.}, .2)
    assert result['equivalent_gap_reach'] == pytest.approx(math.sqrt(.38))
    assert result['repeat_gap_probability'] == pytest.approx(.38)
    assert result['weighted_equivalent_gap_reach'] == pytest.approx(.2 * math.sqrt(.38))
    assert result['gap_mass'] == pytest.approx(1.)
    assert result['unresolved_mass'] == 0
    assert {r['position'] for r in result['gaps']}.isdisjoint(evidence)
    assert distribution(evaluator(g, evidence), {g.roots[0]: 1.}, 0)['weighted_equivalent_gap_reach'] == 0
    assert distribution(evaluator(g, evidence), {g.roots[0]: 1.}, None)['weighted_equivalent_gap_reach'] is None


def test_transposed_first_gaps_are_merged_before_squaring(tmp_path):
    g = graph(tmp_path, '1. Nf3 d5 2. g3 Nf6 *\n\n1. g3 Nf6 2. Nf3 d5 *')
    evidence = {k: data(100, 0, 0, [(m, 100, 0, 0) for m in n.edges])
                for k, n in g.nodes.items() if not chess.Board(n.fen).turn}
    policy = {g.roots[0]: {'g1f3': .4, 'g2g3': .6}}
    result = distribution(evaluator(g, evidence, policy), {g.roots[0]: 1.})
    assert result['equivalent_gap_reach'] == pytest.approx(1.)
    assert result['gaps'] == [dict(position=position('Nf3 d5 g3 Nf6'), reach=1.)]


def test_cached_endpoint_transposition_rejoins_prepared_reply(tmp_path):
    g = graph(tmp_path, '1. Nf3 d5 2. g3 *\n\n1. g3 Nf6 2. Nf3 d5 3. Bg2 *')
    evidence = {position('Nf3'): data(100, 0, 0, [('d7d5', 100, 0, 0)]),
                position('Nf3 d5 g3'): data(100, 0, 0, [('g8f6', 100, 0, 0)]),
                position('g3'): data(100, 0, 0, [('g8f6', 100, 0, 0)]),
                position('g3 Nf6 Nf3'): data(100, 0, 0, [('d7d5', 100, 0, 0)]),
                position('g3 Nf6 Nf3 d5 Bg2'): data(100, 0, 0, [('c7c5', 100, 0, 0)])}
    result = distribution(evaluator(g, evidence, {g.roots[0]: {'g1f3': .4, 'g2g3': .6}}), {g.roots[0]: 1.})
    assert result['distinct_gaps'] == 1
    assert result['gaps'][0]['position'] == position('Nf3 d5 g3 Nf6 Bg2 c5')
    assert result['gap_mass'] == pytest.approx(1.)


@pytest.mark.parametrize('recorded,total,bounds', [(60, 100, [.6, 1.]), (0, 0, [0., 1.])])
def test_unidentified_and_missing_distributions_remain_bounded(tmp_path, recorded, total, bounds):
    g = graph(tmp_path, '1. e4 *')
    evidence = {position('e4'): data(total, 0, 0, [('e7e5', recorded, 0, 0)])}
    result = distribution(evaluator(g, evidence), {g.roots[0]: 1.})
    assert result['equivalent_gap_reach'] is None
    assert result['equivalent_gap_reach_bounds'] == pytest.approx(bounds)
    assert result['unresolved_mass'] == pytest.approx(1 - recorded / total if total else 1)


@pytest.mark.parametrize('loop_probability', [.5, 1.])
def test_endpoint_cycles_conserve_first_gap_mass(tmp_path, loop_probability):
    g = graph(tmp_path, '1. Nf3 Nf6 2. Ng1 *')
    evidence = {position('Nf3'): data(100, 0, 0, [('g8f6', 100, 0, 0)]),
                position('Nf3 Nf6 Ng1'): data(100, 0, 0, [('f6g8', int(100 * loop_probability), 0, 0),
                                                         ('c7c5', int(100 * (1 - loop_probability)), 0, 0)])}
    result = distribution(evaluator(g, evidence), {g.roots[0]: 1.})
    assert result['validation']['cyclic_components'] == 1
    if loop_probability < 1:
        assert result['gap_mass'] == pytest.approx(1.)
        assert result['equivalent_gap_reach'] == pytest.approx(1.)
    else:
        assert result['unresolved_mass'] == pytest.approx(1.)
        assert result['equivalent_gap_reach_bounds'] == [0., 1.]


def test_game_over_is_not_an_unprepared_gap(tmp_path):
    g = graph(tmp_path, '1. f3 e5 2. g4 Qh4# *')
    evidence = {position('f3'): data(0, 0, 100, [('e7e5', 0, 0, 100)]),
                position('f3 e5 g4'): data(0, 0, 100, [('d8h4', 0, 0, 100)])}
    result = distribution(evaluator(g, evidence), {g.roots[0]: 1.})
    assert result['terminal_mass'] == pytest.approx(1.)
    assert result['equivalent_gap_reach'] == 0


def test_black_uses_white_reply_rows(tmp_path):
    g = graph(tmp_path, '1. e4 c5 *')
    evidence = {position(''): data(100, 0, 0, [('e2e4', 100, 0, 0)]),
                position('e4 c5'): data(100, 0, 0, [('g1f3', 80, 0, 0), ('b1c3', 20, 0, 0)])}
    result = distribution(evaluator(g, evidence, color=False), {g.roots[0]: 1.})
    assert result['equivalent_gap_reach'] == pytest.approx(math.sqrt(.68))
    assert {r['position'] for r in result['gaps']} == {position('e4 c5 Nf3'), position('e4 c5 Nc3')}
