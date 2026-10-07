import pytest

from repertoire_score.preparation import Evaluator, chess_facts
from repertoire_score.sharpness import recursive_wdl, scope_outcomes, sharpness, stopping_wdl
from helpers import data, graph, position


@pytest.mark.parametrize('wdl, expected', [
    ((.5, 0., .5), 100.), ((.05, .9, .05), 10.),
    ((.95, 0., .05), 19.), ((.95, .05, 0.), 4.75),
    ((1., 0., 0.), 0.), ((0., 1., 0.), 0.), ((0., 0., 1.), 0.),
])
def test_drawish_winning_and_certain_outcomes(wdl, expected):
    assert sharpness(*wdl) == pytest.approx(expected)
    assert sharpness(*wdl[::-1]) == pytest.approx(expected)


@pytest.mark.parametrize('wdl', [(.5, .5, .5), (-.1, .5, .6), (float('nan'), 0., 0.)])
def test_invalid_wdl_is_not_silently_completed(wdl):
    with pytest.raises(ValueError):
        sharpness(*wdl)


def test_recursion_changes_draw_rate_even_when_database_score_is_identical(tmp_path):
    g = graph(tmp_path, '1. e4 e5 2. Nf3 *')
    parent, leaf = position('e4'), position('e4 e5 Nf3')
    evidence = {parent: data(40, 20, 40, [('e7e5', 20, 0, 20), ('c7c5', 10, 10, 20)]),
                leaf: data(0, 100, 0)}
    evaluator = Evaluator(g, True, evidence, chess_facts(g, True, evidence))
    result = scope_outcomes(evaluator, {g.roots[0]: 1.})
    # 40% prepared draws + 40% cached unprepared outcomes + 20% residual outcomes.
    assert [result[k] for k in ('win_probability', 'draw_probability', 'loss_probability')] == pytest.approx([.2, .6, .2])
    assert result['resolved_score'] == pytest.approx(.5)
    assert result['sharpness'] == pytest.approx(40.)
    assert sharpness(.4, .2, .4) == pytest.approx(80.)  # parent database results
    values = recursive_wdl(evaluator)
    assert values[g.roots[0]] == pytest.approx(values[parent])  # own e4 is forced
    assert position('e4 c5') not in evidence  # no unprepared child data is needed


def test_multiple_entries_mix_wdl_before_sharpness(tmp_path):
    g = graph(tmp_path, '1. e4 e5 *\n\n1. d4 d5 *')
    a, b = position('e4 e5'), position('d4 d5')
    evidence = {a: data(100, 0, 0), b: data(0, 0, 100)}
    evaluator = Evaluator(g, True, evidence, chess_facts(g, True, evidence))
    assert scope_outcomes(evaluator, {a: 1.})['sharpness'] == 0.
    assert scope_outcomes(evaluator, {b: 1.})['sharpness'] == 0.
    assert scope_outcomes(evaluator, {a: .5, b: .5})['sharpness'] == 100.


def test_transposition_reuses_one_continuation(tmp_path):
    g = graph(tmp_path, '1. Nf3 d5 2. g3 Nf6 3. Bg2 *\n\n1. g3 Nf6 2. Nf3 d5 3. Bg2 *')
    evidence = {k: data(100, 0, 0, [(m, 100, 0, 0) for m in n.edges]) for k, n in g.nodes.items()}
    shared = position('Nf3 d5 g3 Nf6 Bg2')
    evidence[shared] = data(30, 40, 30)
    policy = {g.roots[0]: {'g1f3': .4, 'g2g3': .6}}
    evaluator = Evaluator(g, True, evidence, chess_facts(g, True, evidence), policy)
    result = scope_outcomes(evaluator, {g.roots[0]: 1.})
    assert result['sharpness'] == pytest.approx(60.)
    assert result['draw_probability'] == pytest.approx(.4)
    assert result['unresolved_probability'] == 0.
    assert len(recursive_wdl(evaluator)) == len(g.nodes)


def test_black_perspective_and_missing_outcomes(tmp_path):
    g = graph(tmp_path, '1. e4 c5 2. Nf3 *')
    root, leaf = g.roots[0], position('e4 c5 Nf3')
    evidence = {root: data(100, 0, 0, [('e2e4', 100, 0, 0)]),
                position('e4 c5'): data(100, 0, 0, [('g1f3', 100, 0, 0)]),
                leaf: data(10, 20, 70)}
    evaluator = Evaluator(g, False, evidence, chess_facts(g, False, evidence))
    result = scope_outcomes(evaluator, {root: 1.})
    assert result['win_probability'] == pytest.approx(.7)
    assert result['loss_probability'] == pytest.approx(.1)
    assert result['resolved_score'] == pytest.approx(.8)
    assert result['sharpness'] == pytest.approx(44.)
    evidence[leaf] = data(0, 0, 0)
    result = scope_outcomes(Evaluator(g, False, evidence, chess_facts(g, False, evidence)), {root: 1.})
    assert result['unresolved_probability'] == 1.
    assert result['sharpness'] is None


def test_partially_unresolved_wdl_is_not_renormalized(tmp_path):
    from helpers import sample
    g, evidence = sample(tmp_path)
    evidence[position('e4 e5 Nf3 Nc6 Bb5')] = data(0, 0, 0)
    evaluator = Evaluator(g, True, evidence, chess_facts(g, True, evidence))
    result = scope_outcomes(evaluator, {g.roots[0]: 1.})
    assert result['unresolved_probability'] == pytest.approx(.8)
    assert result['win_probability'] == pytest.approx(.06)
    assert result['loss_probability'] == pytest.approx(.14)
    assert result['resolved_score'] == pytest.approx(.06)
    assert result['sharpness'] is None


@pytest.mark.parametrize('fixed, expected', [(1., [1, 0, 0, 0]), (.5, [0, 1, 0, 0]), (0., [0, 0, 1, 0])])
def test_terminal_results_need_no_observations(fixed, expected):
    assert stopping_wdl([0, 0, 0], False, fixed) == pytest.approx(expected)


def test_certain_results_tolerate_only_probability_accumulation_rounding():
    from repertoire_score.sharpness import summarize
    assert summarize([1. + 1e-12, 0., 0., 0.])['sharpness'] == 0.
    with pytest.raises(AssertionError, match='conservation'):
        summarize([1.01, 0., 0., 0.])
