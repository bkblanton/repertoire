import json
from types import SimpleNamespace

import chess
import pytest
from helpers import cache_row, data, graph, position, run_fixture

from repertoire_score import score as cli
from repertoire_score.graph import chapter_policy_overrides, resolve
from repertoire_score.vulnerabilities import analyze as vulnerabilities


def test_chapter_local_mainline_order_is_independent_of_global_edge_order(tmp_path):
    g = graph(tmp_path, '1. e4 (1. d4) *\n\n1. d4 (1. e4) *')
    global_policy = resolve(g, chess.WHITE, {})
    assert list(global_policy[g.roots[0]]) == ['e2e4']
    override = chapter_policy_overrides(g, chess.WHITE, global_policy, '2')
    assert override == {g.roots[0]: 'd2d4'}
    assert list(resolve(g, chess.WHITE, override)[g.roots[0]]) == ['d2d4']
    assert list(resolve(g, chess.WHITE, {g.roots[0]: 'd2d4'})[g.roots[0]]) == ['d2d4']


def test_competing_chapters_play_the_higher_scoring_alternative_in_any_order(tmp_path, monkeypatch):
    decision = position('e4 e6 d4 d5')
    first, cache = run_fixture(tmp_path, monkeypatch, common_entry=True, policy=None)
    # Advance scores 40% and the Tarrasch chapters 86% after 2...d5, so 3.Nd2 is played although Advance is first.
    assert first['overall']['raw_empirical_score'] == pytest.approx(0.86)
    assert first['manifest']['selected_alternatives'] == {decision: 'b1d2'}
    [row] = first['alternatives']
    assert row['position'] == decision and row['selected'] == 'b1d2' and row['reach'] == 1
    assert {o['san']: (o['score'], o['chapters']) for o in row['options']} == {
        'e5': (pytest.approx(0.4), ['advance']),
        'Nd2': (pytest.approx(0.86), ['tarrasch', 'split']),
    }
    conflict = next(c for c in first['diagnostics']['policy_conflicts'] if c['position'] == decision)
    assert conflict['resolution'] == 'highest repertoire score among chapter alternatives'
    chapters = {c['id']: c for c in first['chapters']}
    assert chapters['advance']['score']['raw_empirical_score'] == pytest.approx(0.4)
    assert chapters['advance']['score']['overall_policy_entry_probability'] == 1  # shared entry, not selected move
    for cid in ('tarrasch', 'split'):
        c = chapters[cid]
        assert c['score']['raw_empirical_score'] == pytest.approx(0.86)
        assert c['score']['entry_probability'] == 1
        assert c['entry_baseline']['raw_score'] == 0.5
        assert c['score']['prepared_depth']['expected_moves'] == 3
    assert first['manifest']['policy_profile_count'] == 2
    # Later stages replay the saved choice rather than the first chapter's move.
    overall = vulnerabilities(tmp_path / 'white.json', cache)['overall']
    assert {r['move_san'] for r in overall['all_signed_rows'] if r['position'] == decision} == {'Nd2'}
    reordered, _ = run_fixture(tmp_path, monkeypatch, common_entry=True, alternative_first=True, policy=None)
    assert reordered['overall']['raw_empirical_score'] == pytest.approx(0.86)
    assert {c['id']: c['score']['raw_empirical_score'] for c in reordered['chapters']} == pytest.approx(
        {c['id']: c['score']['raw_empirical_score'] for c in first['chapters']}
    )


def test_explicit_policy_overrides_competing_alternatives(tmp_path, monkeypatch):
    forced, _ = run_fixture(tmp_path, monkeypatch, common_entry=True)
    assert forced['overall']['raw_empirical_score'] == pytest.approx(0.4)
    assert forced['manifest']['selected_alternatives'] == {}
    assert forced['alternatives'] == []
    chapters = {c['id']: c for c in forced['chapters']}
    assert chapters['tarrasch']['score']['raw_empirical_score'] == pytest.approx(0.86)


def test_unselected_chapters_have_conditional_scores_and_separate_reach(tmp_path, monkeypatch):
    report, cache = run_fixture(tmp_path, monkeypatch)
    chapters = {c['id']: c for c in report['chapters']}
    for cid, expected_reach, expected_score in [('tarrasch', 0.6, 0.9), ('split', 0.4, 0.8)]:
        c = chapters[cid]
        assert c['score']['entry_probability'] == pytest.approx(expected_reach)
        assert c['score']['overall_policy_entry_probability'] == 0
        assert c['score']['raw_empirical_score'] == pytest.approx(expected_score)
        assert c['score']['prepared_depth']['expected_moves'] == 1
    rendered = (tmp_path / 'report.md').read_text(encoding='utf-8')
    assert not (tmp_path / 'white.md').exists()
    assert '**(alternative)**' in rendered and 'Overall-policy reach' in rendered
    summary = (tmp_path / 'summary.md').read_text()
    assert 'Tarrasch Nf6' in summary
    assert (
        '**(alternative)**<br>[Lichess study](https://lichess.org/study/test/tarrasch) | 60.0%<br>Overall 0.0%'
        in summary
    )
    assert 'Overall-policy reach' in summary
    result = vulnerabilities(tmp_path / 'white.json', cache)
    assert result['manifest']['candidate_child_queries'] == 0
    assert result['manifest']['parent_tables_fetched'] == 0
    assert len(result['chapters']) == 3
    assert result['validation']['chapter_scores_reproduced']
    selected = {r['move_san'] for r in result['overall']['all_signed_rows'] if r['position'] == position('e4 e6 d4 d5')}
    assert selected == {'e5'}
    assert next(c for c in result['chapters'] if c['id'] == 'tarrasch')['repertoire_score'] == pytest.approx(0.9)


def test_alternative_multiple_entries_use_its_own_policy_weights(tmp_path, monkeypatch):
    report, _ = run_fixture(tmp_path, monkeypatch, multiple_entries=True)
    c = next(c for c in report['chapters'] if c['id'] == 'tarrasch')
    assert c['score']['first_entry_weights'] == pytest.approx(
        {position('e4 e6 d4 d5 Nd2 Nf6'): 0.6, position('e4 e6 d4 d5 Nd2 c5'): 0.4}
    )
    assert c['score']['entry_probability'] == 1
    assert c['score']['overall_policy_entry_probability'] == 0
    assert c['score']['raw_empirical_score'] == pytest.approx(0.86)
    assert c['entry_baseline']['raw_score'] == 0.5
    assert c['score']['prepared_depth']['expected_moves'] == 1


def test_automatic_entry_falls_back_when_only_discarded_variations_are_unique(tmp_path, monkeypatch):
    # The second chapter's only unique position is its discarded d4 alternative.
    graph(tmp_path, '1. e4 e5 *\n\n1. e4 (1. d4) e5 *')
    cache = tmp_path / 'cache'
    cache.mkdir()
    for k, row in {
        position(''): data(50, 0, 50),
        position('e4'): data(50, 0, 50, [('e7e5', 50, 0, 50)]),
        position('e4 e5'): data(70, 0, 30),
    }.items():
        cache_row(cache, k, row)
    args = SimpleNamespace(
        config=None,
        pgn=str(tmp_path / 'fixture.pgn'),
        color='white',
        output=str(tmp_path / 'white'),
        command='run',
        cache=str(cache),
        offline=True,
        refresh=False,
        prior=[0.5] * 3,
        sparse_threshold=30,
        tolerance=1,
    )
    cli.analyze(args)
    report = json.loads((tmp_path / 'white.json').read_text())
    assert [c['score']['raw_empirical_score'] for c in report['chapters']] == pytest.approx([0.7, 0.7])
