import json

import chess
import pytest

from repertoire_score.graph import chapter_region
from repertoire_score.insights import depth_distribution, first_entry_examples
from repertoire_score.preparation import Evaluator, chess_facts
from repertoire_score.consolidated import (Chapters, cp_change, entry_routes_section, games_per_encounter,
    generate, load, own_priorities, summary_own_priorities)
from test_model import graph, data, position
from test_preparation import sample
from test_consolidated import complete


def evaluator(g, evidence, color=True):
    return Evaluator(g, color, evidence, chess_facts(g, color, evidence))


def test_survival_and_stopping_depths_reproduce_expected_depth(tmp_path):
    g, evidence = sample(tmp_path)
    result = depth_distribution(evaluator(g, evidence), {g.roots[0]: 1.})
    endings = {r['own_moves']: r for r in result['endings']}
    assert endings[2]['unprepared_reply'] == pytest.approx(.2)
    assert endings[3]['other_stop'] == pytest.approx(.8)  # no recorded reply rows
    assert [r['probability'] for r in result['survival']] == pytest.approx([1, 1, 1, .8])
    assert result['expected_moves'] == pytest.approx(2.8)
    assert result['median_moves'] == 3
    assert sum(sum(r[t] for t in ('prepared_endpoint', 'unprepared_reply', 'game_over', 'other_stop', 'unresolved_distribution'))
               for r in result['endings']) == pytest.approx(1)


def test_depth_is_known_even_when_leaf_score_has_no_observations(tmp_path):
    g, evidence = sample(tmp_path)
    evidence[position('e4 e5 Nf3 Nc6 Bb5')] = data(0, 0, 0)
    e = evaluator(g, evidence)
    assert e.evaluate({g.roots[0]: 1.})[1] == pytest.approx(.8)
    result = depth_distribution(e, {g.roots[0]: 1.})
    assert result['status'] == 'resolved'
    assert result['expected_moves'] == pytest.approx(2.8)


def test_missing_reply_distribution_keeps_finite_depth_bounds(tmp_path):
    g, evidence = sample(tmp_path)
    evidence[position('e4 e5 Nf3')] = data(0, 0, 0)
    result = depth_distribution(evaluator(g, evidence), {g.roots[0]: 1.})
    assert result['status'] == 'unresolved_move_distribution'
    assert result['expected_moves'] is None
    assert result['expected_bounds'] == pytest.approx([2, 3])
    assert result['survival'][2]['probability'] == 1
    assert result['survival'][3]['probability'] is None
    assert result['survival'][3]['bounds'] == [0, 1]
    assert result['median_bounds'] == [2, 3]


def test_transposed_arrivals_retain_different_elapsed_depths(tmp_path):
    g, evidence = sample(tmp_path)
    starts = {position('e4'): .5, position('e4 e5 Nf3'): .5}
    result = depth_distribution(evaluator(g, evidence), starts)
    endings = {r['own_moves']: r for r in result['endings']}
    assert endings[0]['unprepared_reply'] == pytest.approx(.1)
    assert endings[1]['unprepared_reply'] == pytest.approx(.1)
    assert endings[1]['other_stop'] == pytest.approx(.4)
    assert endings[2]['other_stop'] == pytest.approx(.4)
    assert result['expected_moves'] == pytest.approx(1.3)
    assert result['median_moves'] == 1


def test_black_depth_counts_black_moves_and_residuals_have_their_own_bucket(tmp_path):
    g, evidence = sample(tmp_path)
    # White replies are probabilities for the Black repertoire; its own moves
    # remain forced, regardless of their database popularity.
    for k, n in g.nodes.items():
        evidence[k] = data(50, 0, 50, [(m, 50, 0, 50) for m in n.edges][:1])
    result = depth_distribution(evaluator(g, evidence, False), {g.roots[0]: 1.})
    assert result['expected_moves'] == 2
    evidence[position('')] = data(60, 0, 40, [('e2e4', 30, 0, 20)])
    result = depth_distribution(evaluator(g, evidence, False), {g.roots[0]: 1.})
    assert result['expected_moves'] == 1
    assert result['endings'][0]['other_stop'] == .5
    assert result['endings'][1]['prepared_endpoint'] == .5


def quiet_fixture(tmp_path, paulsen=False):
    text = '1. e4 e5 2. Nc3 Nf6 3. g3 Nc6 4. Bg2 Bc5 *\n\n'
    text += '1. e4 e5 2. Nc3 Nf6 3. g3 Bc5 4. Bg2 Nc6 *'
    if paulsen: text += '\n\n1. e4 e5 2. Nc3 Nc6 3. g3 Bc5 4. Bg2 d6 *'
    g = graph(tmp_path, text)
    evidence = {}
    for k, n in g.nodes.items():
        weight = 100 // max(1, len(n.edges))
        evidence[k] = data(50, 0, 50, [(m, weight // 2, 0, weight // 2) for m in n.edges])
    evidence[position('e4 e5 Nc3 Nf6 g3')] = data(50, 0, 50, [('b8c6', 30, 0, 30), ('f8c5', 20, 0, 20)])
    if paulsen:
        evidence[position('e4 e5 Nc3 Nc6 g3 Bc5 Bg2')] = data(50, 0, 50, [('g8f6', 50, 0, 50)])
    region = chapter_region(g, '1', [position('e4 e5 Nc3 Nf6 g3 Nc6')])
    return g, evidence, region


def test_entry_examples_follow_real_bypasses_and_reproduce_entry_weights(tmp_path):
    g, evidence, region = quiet_fixture(tmp_path)
    anchor = position('e4 e5 Nc3 Nf6 g3 Nc6')
    later = position('e4 e5 Nc3 Nf6 g3 Nc6 Bg2 Bc5')
    result = first_entry_examples(evaluator(g, evidence), {g.roots[0]: 1.}, region, 1., {anchor: .6, later: .4})
    entry = next(r for r in result['positions'] if r['position'] == later)
    assert entry['example']['path_san'] == ['e4', 'e5', 'Nc3', 'Nf6', 'g3', 'Bc5', 'Bg2', 'Nc6']
    assert entry['example']['conditional_probability'] == pytest.approx(.4)
    assert entry['conditional_first_entry_weight'] == pytest.approx(.4)
    board = chess.Board(entry['example']['root_fen'])
    for move in entry['example']['path_uci']:
        assert position_from_board(board) not in region
        board.push_uci(move)
    assert position_from_board(board) == later


def position_from_board(board):
    return ' '.join(board.fen(en_passant='legal').split()[:4])


def test_entry_examples_merge_unrecorded_immediate_transpositions(tmp_path):
    g, evidence, region = quiet_fixture(tmp_path, paulsen=True)
    anchor = position('e4 e5 Nc3 Nf6 g3 Nc6')
    later = position('e4 e5 Nc3 Nf6 g3 Nc6 Bg2 Bc5')
    assert 'g8f6' not in g.nodes[position('e4 e5 Nc3 Nc6 g3 Bc5 Bg2')].edges
    result = first_entry_examples(evaluator(g, evidence), {g.roots[0]: 1.}, region, 1., {anchor: .3, later: .7})
    row = next(r for r in result['positions'] if r['position'] == later)
    assert row['conditional_first_entry_weight'] == pytest.approx(.7)
    assert row['example']['conditional_probability'] == pytest.approx(.5)
    assert row['example']['path_san'] == ['e4', 'e5', 'Nc3', 'Nc6', 'g3', 'Bc5', 'Bg2', 'Nf6']


def test_own_summary_ranking_weights_reach_filters_sparse_and_preserves_cp():
    def row(line, reach, gain, **extra):
        return dict(line=line, branch_reach=reach, local_gain_pp=gain, local_drop_pp=gain,
                    parent_sample_count=100, reference_score=.5, move_score=.6, **extra)
    rows = [row('rare', .0001, 30), row('common', .1, 1), row('sparse', 1, 100, sparse=True)]
    ranked = own_priorities(rows, strongest=True)
    assert [r['line'] for r in ranked] == ['common', 'rare']
    assert games_per_encounter(.0001) == '10,000'
    assert games_per_encounter(.5) == '2.0'
    assert games_per_encounter(0) == 'never'
    assert games_per_encounter(None) == 'unavailable'
    refs = Chapters(dict(color='white', chapters=[]))
    rendered = '\n'.join(summary_own_priorities(ranked, refs, strongest=True))
    assert '| 10.00%<br>1 per 10 games |' in rendered
    assert cp_change(.6, .5) + ' cp)' in rendered and '+0.1000%' in rendered
    assert 'sparse' not in rendered


def test_new_insights_are_automated_in_report_and_summary(complete):
    path, _ = complete
    bundle = load([path])[0]
    assert all('depth_distribution' in s for s in bundle['preparation']['scopes'])
    generate([path])
    full = (path.parent / 'report.md').read_text(encoding='utf-8')
    summary = (path.parent / 'summary.md').read_text(encoding='utf-8')
    assert full.count('##### Prepared-depth distribution') == len(bundle['report']['chapters'])
    assert 'Entry-position weight' in full and 'Example-route weight' in full
    assert 'any position in this chapter, through any move order' in full
    assert 'Own moves with the largest weighted' in summary
    assert 'Avg games per encounter' in full and 'Avg games per encounter' in summary
    assert 'Median prepared depth' in summary
    assert 'Depth and improvement' not in summary
    assert 'white-prepared-depth' in full and 'report.md#white-prepared-depth' in summary
    assert all(r.get('chapter_attribution') for s in bundle['preparation']['scopes']
               for r in s.get('entry_routes', {}).get('positions', []))
