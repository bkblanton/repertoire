import chess
import pytest
from repertoire_score.baseline import chapter_entry_baseline
from repertoire_score.graph import key


def two_positions():
    b = chess.Board()
    first = key(b)
    b.push_san('e4')
    return first, key(b)


def test_multiple_entries_use_first_entry_weights_not_sample_counts():
    a, b = two_positions()
    evidence = {a: {'white': 8, 'draws': 0, 'black': 2},
                b: {'white': 200, 'draws': 0, 'black': 800}}
    chapter = {'raw_empirical_score': .5, 'first_entry_weights': {a: .25, b: .75}}
    white = chapter_entry_baseline([a, b], chapter, evidence, True, {})
    black = chapter_entry_baseline([a, b], chapter, evidence, False, {})
    assert white['raw_score'] == pytest.approx(.35)
    assert white['difference_pp'] == pytest.approx(15)
    assert black['raw_score'] == pytest.approx(.65)
    assert black['difference_pp'] == pytest.approx(-15)


def test_zero_data_keeps_its_full_weight_and_zero_weight_does_not_block():
    a, b = two_positions()
    evidence = {a: {'white': 8, 'draws': 0, 'black': 2},
                b: {'white': 0, 'draws': 0, 'black': 0}}
    chapter = {'raw_empirical_score': .5, 'first_entry_weights': {a: .25, b: .75}}
    baseline = chapter_entry_baseline([a, b], chapter, evidence, True, {})
    assert baseline['raw_score'] is None and baseline['difference_pp'] is None
    assert baseline['conditional_bounds'] == pytest.approx([.2, .95])
    chapter['first_entry_weights'] = {a: 1., b: 0.}
    assert chapter_entry_baseline([a, b], chapter, evidence, True, {})['raw_score'] == .8


def test_unknown_multiple_entry_weights_are_not_replaced_by_uniform_weights():
    a, b = two_positions()
    assert chapter_entry_baseline([a, b], {}, {}, True, {})['status'] == 'unresolved_first_entry_weights'
    with pytest.raises(ValueError, match='summing to one'):
        chapter_entry_baseline([a, b], {'first_entry_weights': {a: .5, b: .8}}, {}, True, {})


def test_single_terminal_entry_needs_no_database_evidence():
    position = '7k/6Q1/6K1/8/8/8/8/8 b - -'
    baseline = chapter_entry_baseline([position], {'raw_empirical_score': 1.}, {}, True, {})
    assert baseline['raw_score'] == 1.
    assert baseline['difference_pp'] == 0.
