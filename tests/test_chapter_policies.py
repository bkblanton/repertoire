import json
from types import SimpleNamespace

import chess
import httpx
import pytest

from repertoire_score import __main__ as cli
from repertoire_score.graph import chapter_policy_overrides, resolve
from repertoire_score.vulnerabilities import analyze as vulnerabilities
from test_model import data, graph, position
from test_vulnerabilities import cache_row


ADVANCE = '[ChapterURL "https://lichess.org/study/test/advance"]\n[ChapterName "Advance"]\n\n1. e4 e6 2. d4 d5 3. e5 c5 4. c3 *'
TARRASCH = '[ChapterURL "https://lichess.org/study/test/tarrasch"]\n[ChapterName "Tarrasch Nf6"]\n\n1. e4 e6 2. d4 d5 3. Nd2 Nf6 4. e5 *'
SPLIT = '[ChapterURL "https://lichess.org/study/test/split"]\n[ChapterName "Tarrasch c5"]\n\n1. e4 e6 2. d4 d5 3. Nd2 c5 4. exd5 *'


def run_fixture(tmp_path, monkeypatch, common_entry=False, alternative_first=False, multiple_entries=False):
    def forbidden(*args, **kwargs):
        raise AssertionError('A fully cached comparison must not use the network')
    monkeypatch.setattr(httpx.Client, 'request', forbidden)
    parts = [TARRASCH, SPLIT, ADVANCE] if alternative_first else [ADVANCE, TARRASCH, SPLIT]
    g = graph(tmp_path, '\n\n'.join(parts))
    cache = tmp_path/'cache'
    cache.mkdir(exist_ok=True)
    leaves = {position('e4 e6 d4 d5 e5 c5 c3'): 40,
              position('e4 e6 d4 d5 Nd2 Nf6 e5'): 90,
              position('e4 e6 d4 d5 Nd2 c5 exd5'): 80}
    for k, n in g.nodes.items():
        if k in leaves:
            evidence = data(leaves[k], 0, 100-leaves[k])
        else:
            moves = list(n.edges)
            weights = [60, 40] if k == position('e4 e6 d4 d5 Nd2') else [100//len(moves)]*len(moves)
            evidence = data(50, 0, 50, [(move, weight//2, 0, weight//2) for move, weight in zip(moves, weights)])
        cache_row(cache, k, evidence)
    config = tmp_path/'config.json'
    configuration = {'entries': {c['id']: [{'path': ['e4', 'e6']}] for c in g.chapters}} if common_entry else {}
    if multiple_entries:
        configuration.setdefault('entries', {})['tarrasch'] = [
            {'path': ['e4', 'e6', 'd4', 'd5', 'Nd2', move]} for move in ('Nf6', 'c5')]
    config.write_text(json.dumps(configuration))
    args = SimpleNamespace(config=str(config), pgn=str(tmp_path/'fixture.pgn'), color='white',
                           output=str(tmp_path/'white'), command='run', cache=str(cache), offline=True,
                           refresh=False, simulations=100, seed=1, prior=[.5]*3, sparse_threshold=30, tolerance=1)
    cli.analyze(args)
    return json.loads((tmp_path/'white.json').read_text()), cache


def test_chapter_local_mainline_order_is_independent_of_global_edge_order(tmp_path):
    g = graph(tmp_path, '1. e4 (1. d4) *\n\n1. d4 (1. e4) *')
    global_policy = resolve(g, chess.WHITE, {})
    assert list(global_policy[g.roots[0]]) == ['e2e4']
    override = chapter_policy_overrides(g, chess.WHITE, global_policy, '2')
    assert override == {g.roots[0]: 'd2d4'}
    assert list(resolve(g, chess.WHITE, override)[g.roots[0]]) == ['d2d4']
    assert list(resolve(g, chess.WHITE, {g.roots[0]: 'd2d4'})[g.roots[0]]) == ['d2d4']


def test_alternatives_with_shared_entry_keep_split_continuations_and_reorder(tmp_path, monkeypatch):
    first, cache = run_fixture(tmp_path, monkeypatch, common_entry=True)
    assert first['overall']['raw_empirical_score'] == pytest.approx(.4)
    chapters = {c['id']: c for c in first['chapters']}
    assert chapters['advance']['score']['raw_empirical_score'] == pytest.approx(.4)
    for cid in ('tarrasch', 'split'):
        c = chapters[cid]
        assert c['score']['raw_empirical_score'] == pytest.approx(.86)
        assert c['score']['entry_probability'] == 1
        assert c['score']['overall_policy_entry_probability'] == 1  # shared entry, not selected move
        assert c['entry_baseline']['raw_score'] == .5
        assert c['score']['prepared_depth']['expected_moves'] == 3
    assert first['manifest']['policy_profile_count'] == 2
    comparison = next(c for c in vulnerabilities(tmp_path/'white.json', cache)['chapters'] if c['id'] == 'tarrasch')
    selected = {r['move_san'] for r in comparison['all_signed_rows'] if r['position'] == position('e4 e6 d4 d5')}
    assert selected == {'Nd2'}
    reordered, _ = run_fixture(tmp_path, monkeypatch, common_entry=True, alternative_first=True)
    assert reordered['overall']['raw_empirical_score'] == pytest.approx(.86)
    assert {c['id']: c['score']['raw_empirical_score'] for c in reordered['chapters']} == pytest.approx(
        {c['id']: c['score']['raw_empirical_score'] for c in first['chapters']})


def test_unselected_chapters_have_conditional_scores_and_separate_reach(tmp_path, monkeypatch):
    report, cache = run_fixture(tmp_path, monkeypatch)
    chapters = {c['id']: c for c in report['chapters']}
    for cid, expected_reach, expected_score in [('tarrasch', .6, .9), ('split', .4, .8)]:
        c = chapters[cid]
        assert c['score']['entry_probability'] == pytest.approx(expected_reach)
        assert c['score']['overall_policy_entry_probability'] == 0
        assert c['score']['raw_empirical_score'] == pytest.approx(expected_score)
        assert c['score']['prepared_depth']['expected_moves'] == 1
    rendered = (tmp_path/'report.md').read_text(encoding='utf-8')
    assert not (tmp_path/'white.md').exists()
    assert '**(alternative)**' in rendered and 'Overall-policy reach' in rendered
    summary = (tmp_path/'summary.md').read_text()
    assert 'Tarrasch Nf6' in summary and '**(alternative)** | 60.00%<br>Overall 0.00%' in summary
    assert 'Overall-policy reach' in summary
    result = vulnerabilities(tmp_path/'white.json', cache)
    assert result['manifest']['candidate_child_queries'] == 0
    assert result['manifest']['parent_tables_fetched'] == 0
    assert len(result['chapters']) == 3
    assert result['validation']['chapter_scores_reproduced']
    selected = {r['move_san'] for r in result['overall']['all_signed_rows'] if r['position'] == position('e4 e6 d4 d5')}
    assert selected == {'e5'}
    assert next(c for c in result['chapters'] if c['id'] == 'tarrasch')['repertoire_score'] == pytest.approx(.9)


def test_alternative_multiple_entries_use_its_own_policy_weights(tmp_path, monkeypatch):
    report, _ = run_fixture(tmp_path, monkeypatch, multiple_entries=True)
    c = next(c for c in report['chapters'] if c['id'] == 'tarrasch')
    assert c['score']['first_entry_weights'] == pytest.approx({position('e4 e6 d4 d5 Nd2 Nf6'): .6,
                                                            position('e4 e6 d4 d5 Nd2 c5'): .4})
    assert c['score']['entry_probability'] == 1
    assert c['score']['overall_policy_entry_probability'] == 0
    assert c['score']['raw_empirical_score'] == pytest.approx(.86)
    assert c['entry_baseline']['raw_score'] == .5
    assert c['score']['prepared_depth']['expected_moves'] == 1


def test_automatic_entry_falls_back_when_only_discarded_variations_are_unique(tmp_path, monkeypatch):
    # The second chapter's only unique position is its discarded d4 alternative.
    graph(tmp_path, '1. e4 e5 *\n\n1. e4 (1. d4) e5 *')
    cache = tmp_path/'cache'
    cache.mkdir()
    for k, row in {position(''): data(50, 0, 50), position('e4'): data(50, 0, 50, [('e7e5', 50, 0, 50)]),
                   position('e4 e5'): data(70, 0, 30)}.items():
        cache_row(cache, k, row)
    args = SimpleNamespace(config=None, pgn=str(tmp_path/'fixture.pgn'), color='white', output=str(tmp_path/'white'),
                           command='run', cache=str(cache), offline=True, refresh=False, simulations=100,
                           seed=1, prior=[.5]*3, sparse_threshold=30, tolerance=1)
    cli.analyze(args)
    report = json.loads((tmp_path/'white.json').read_text())
    assert [c['score']['raw_empirical_score'] for c in report['chapters']] == pytest.approx([.7, .7])
