import hashlib
import json
import math
import re
from pathlib import Path

import httpx
import numpy as np
import pytest

from repertoire_score import character, openings, preparation, vulnerabilities
from repertoire_score.consolidated import Chapters, gap_section, generate, opening_table
from repertoire_score.graph import resolve
from repertoire_score.report_insights import (analyze, branch_score_spread, database_samples,
    gap_priorities, move_decomposition, opening_summary_groups, refresh_opening_groups, refresh_spreads)
from test_chapter_policies import run_fixture
from test_model import data, graph, position


def test_gap_shares_merge_transposed_arrivals_before_squaring():
    result = gap_priorities(dict(gaps=[dict(position='same', reach=.1), dict(position='other', reach=.2),
                                        dict(position='same', reach=.2)]))
    a, b = result['priorities']
    assert a['reach'] == pytest.approx(.3) and a['position'] == 'same'
    assert a['repeat_probability_share'] == pytest.approx(.09/.13)
    assert result['known_repeat_probability'] == pytest.approx(.13)
    assert sum(r['repeat_probability_share'] for r in result['priorities']) == pytest.approx(1.)
    assert b['cumulative_repeat_probability_share'] == pytest.approx(1.)
    partial = gap_priorities(dict(gaps=[dict(position='a', reach=.1)], unresolved_mass=.9))
    assert partial['share_basis'] == 'known first gaps'


def test_branch_spread_separates_transposed_cohorts_and_preserves_duplicates():
    def stop(p, s, board):
        return dict(reach=p, score=s, position=board, type='deviation')
    rows = [stop(.5, .2, 'same'), stop(.5, .8, 'same')]
    result = branch_score_spread(rows, .5)
    assert result['standard_deviation'] == pytest.approx(.3)
    assert result['between_position_deviation'] == 0
    assert result['within_position_deviation'] == pytest.approx(.3)
    rows[1]['position'] = 'different'
    result = branch_score_spread(rows, .5)
    assert result['between_position_deviation'] == pytest.approx(.3)
    assert result['within_position_deviation'] == 0
    split = [stop(.25, .2, 'same'), stop(.25, .2, 'same'), rows[1]]
    assert branch_score_spread(split, .5)['standard_deviation'] == result['standard_deviation']
    assert branch_score_spread([stop(.5, None, 'unknown'), stop(.5, .8, 'known')])['standard_deviation'] is None


def test_move_gain_components_sum_even_when_the_database_move_hurts():
    result = move_decomposition(dict(reference_score=.5, move_database_score=.4, move_score=.6))
    assert result == pytest.approx(dict(database_move_gain_pp=-10, continuation_gain_pp=20, total_gain_pp=10))
    assert move_decomposition(dict(reference_score=.5, move_database_score=None, move_score=.6))['total_gain_pp'] is None


def test_parent_and_move_samples_share_joint_evidence_are_reproducible_and_color_relative():
    k = position('')
    table = data(120, 20, 60, [('e2e4', 60, 10, 30), ('d2d4', 60, 10, 30)])
    a, moves = database_samples(table, k, ['e2e4'], True, 4000, [.5]*3, 7)
    repeat, same = database_samples(table, k, ['e2e4'], True, 4000, [.5]*3, 7)
    black, reversed_moves = database_samples(table, k, ['e2e4'], False, 4000, [.5]*3, 7)
    np.testing.assert_array_equal(a, repeat)
    np.testing.assert_array_equal(moves['e2e4'], same['e2e4'])
    np.testing.assert_allclose(a + black, 1.)
    np.testing.assert_allclose(moves['e2e4'] + reversed_moves['e2e4'], 1.)
    assert np.cov(a, moves['e2e4'])[0, 1] > 0


def test_opening_grouping_requires_identical_entry_cohorts_not_equal_percentages(tmp_path):
    g = graph(tmp_path, '1. Nf3 d5 *\n\n1. g3 d5 *')
    def opening(name, k, parents=()):
        return dict(id=name, name=name, parent_ids=list(parents), repertoire_score=.6,
                    entries=[dict(position=k, entry_probability=1.)])
    rows = [opening('Family', g.roots[0]), opening('Family: Main', position('Nf3'), ['Family']),
            opening('Family: Other', position('g3'), ['Family'])]
    groups = opening_summary_groups(rows, g, resolve(g, True, {}), True)
    assert len(groups) == 2
    assert groups[0] == dict(ids=['Family', 'Family: Main'], representative='Family: Main')
    assert groups[1]['ids'] == ['Family: Other']


@pytest.mark.parametrize('pgn, earlier, later, color', [
    ('1. e4 e5 2. Nc3 Nc6 3. g3 *', 'e4 e5 Nc3 Nc6', 'e4 e5 Nc3 Nc6 g3', True),
    ('1. e4 e5 2. Nc3 Nf6 3. g3 *', 'e4 e5 Nc3 Nf6', 'e4 e5 Nc3 Nf6 g3', True),
    ('1. d4 d5 2. c4 e6 *', 'd4 d5 c4', 'd4 d5 c4 e6', False),
])
def test_sibling_openings_group_by_forced_moves_and_choose_downstream_baseline(tmp_path, pgn, earlier, later, color):
    g = graph(tmp_path, pgn)
    def opening(name, moves):
        return dict(id=name, name=name, parent_ids=['Family'], repertoire_score=.6,
                    entries=[dict(position=position(moves), entry_probability=.2)])
    upstream, downstream = opening('Family: Very long upstream name', earlier), opening('Family: Short', later)
    expected = [dict(ids=[upstream['id'], downstream['id']], representative=downstream['id'])]
    for rows in ([upstream, downstream], [downstream, upstream]):
        assert opening_summary_groups(rows, g, resolve(g, color, {}), color) == expected


def test_grouping_requires_all_transposed_entries_and_their_weights_to_match(tmp_path):
    g = graph(tmp_path, '1. e4 e5 2. Nc3 Nc6 3. g3 *\n\n1. e4 e5 2. Nc3 Nf6 3. g3 *')
    def opening(name, entries):
        return dict(id=name, name=name, parent_ids=['Vienna'], repertoire_score=.6,
                    entries=[dict(position=position(moves), entry_probability=p) for moves, p in entries])
    upstream = opening('Defense', [('e4 e5 Nc3 Nc6', .05), ('e4 e5 Nc3 Nc6', .15), ('e4 e5 Nc3 Nf6', .1)])
    downstream = opening('Variation', [('e4 e5 Nc3 Nc6 g3', .2), ('e4 e5 Nc3 Nf6 g3', .1)])
    assert opening_summary_groups([downstream, upstream], g, resolve(g, True, {}), True) == [
        dict(ids=['Defense', 'Variation'], representative='Variation')]
    for entries in ([('e4 e5 Nc3 Nc6 g3', .3)],
                    [('e4 e5 Nc3 Nc6 g3', .1), ('e4 e5 Nc3 Nf6 g3', .2)]):
        different = opening('Variation', entries)
        assert len(opening_summary_groups([upstream, different], g, resolve(g, True, {}), True)) == 2


def test_converging_routes_are_not_the_same_opening_entry_cohort(tmp_path):
    g = graph(tmp_path, '1. e4 e5 2. Nc3 Nc6 3. g3 *\n\n1. e4 Nc6 2. g3 e5 3. Nc3 *')
    rows = [dict(id=str(i), name=str(i), parent_ids=['Family'], repertoire_score=.6,
                 entries=[dict(position=position(moves), entry_probability=.2)])
            for i, moves in enumerate(('e4 e5 Nc3 Nc6', 'e4 Nc6 g3 e5'))]
    # Both forced replies reach the same board, but neither named entry leads to the other.
    assert len(opening_summary_groups(rows, g, resolve(g, True, {}), True)) == 2


def test_grouping_does_not_assume_opponent_moves_or_mixed_own_choices(tmp_path):
    g = graph(tmp_path, '1. e4 e5 *\n\n1. d4 d5 *')
    def opening(name, moves):
        return dict(id=name, name=name, parent_ids=['Family'], repertoire_score=.6,
                    entries=[dict(position=position(moves), entry_probability=1.)])
    parent, e4, e5 = opening('Family', ''), opening('Family: e4', 'e4'), opening('Family: e5', 'e4 e5')
    assert len(opening_summary_groups([e4, e5], g, resolve(g, True, {}), True)) == 2
    mixed = resolve(g, True, {position(''): {'e2e4': .5, 'd2d4': .5}})
    assert len(opening_summary_groups([parent, e4], g, mixed, True)) == 2
    different = dict(e4, repertoire_score=.7)
    assert len(opening_summary_groups([parent, different], g, resolve(g, True, {}), True)) == 2
    empty = [dict(r, entries=[]) for r in (parent, e4)]
    assert len(opening_summary_groups(empty, g, resolve(g, True, {}), True)) == 2


def test_summary_group_uses_downstream_statistics_counts_reach_once_and_links_both_labels():
    def opening(name, baseline):
        return dict(id=name, name=name, reach=.2, entry_baseline=dict(raw_score=baseline),
                    repertoire_score=.6, outcomes=dict(sharpness=90))
    upstream = opening('Vienna Game: Max Lange Defense', .5)
    downstream = opening('Vienna Game: Paulsen Variation', .55)
    refs = Chapters(dict(color='white', chapters=[]), dict(openings=[upstream, downstream]))
    row = dict(downstream, _summary_ids=[upstream['id'], downstream['id']])
    text = '\n'.join(opening_table([row], refs, {upstream['id']: 'first', downstream['id']: 'second'}, True))
    assert '[Vienna Game: Max Lange Defense](#first)<br>↳ [Paulsen Variation](#second)' in text
    assert '| 20.00% | 55.00% | 60.00% | +5.00%' in text and ' cp)' not in text
    assert text.count('20.00%') == 1 and '50.00%' not in text


def test_gap_priority_table_retains_full_denominator_and_five_summary_rows():
    positions = [dict(position=str(i), line=f'gap {i}', reach=.1, database_score=.5,
                      games=100, kind='unprepared_reply', to_move='white') for i in range(10)]
    metrics = dict(equivalent_gap_reach=math.sqrt(.1), gaps=positions, unresolved_mass=0)
    scope = dict(id='overall', positions=positions, gap_coverage=metrics, gap_priorities=gap_priorities(metrics))
    text = '\n'.join(gap_section(scope, refs=Chapters(dict(color='white', chapters=[])), top=5, compact=True))
    assert text.count('| [gap ') == 5
    assert '**50.00%**' in text
    assert text.count('10.00% | 50.00%') == 5


def test_cached_insights_preserve_scores_cache_and_source_and_reject_stale_inputs(tmp_path, monkeypatch):
    saved, cache = run_fixture(tmp_path, monkeypatch, common_entry=True)
    path = tmp_path/'white.json'
    for name, module in [('vulnerabilities', vulnerabilities), ('preparation', preparation), ('character', character), ('openings', openings)]:
        path.with_suffix(f'.{name}.json').write_text(json.dumps(module.analyze(path, cache)))
    preserved = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [path, *cache.glob('*.json')]}
    source = Path(saved['manifest']['input_path'])
    source_before = source.read_bytes()
    def forbidden(*args, **kwargs): raise AssertionError('network forbidden')
    monkeypatch.setattr(httpx.Client, 'request', forbidden)
    result = analyze(path, cache)
    again = analyze(path, cache)
    assert result['scopes'] == again['scopes']
    assert result['manifest']['network_requests'] == 0
    for scope in result['scopes']:
        for sources in scope.get('entry_opening_sources', {}).values():
            assert sum(r['share'] for r in sources) == pytest.approx(1.)
        for row in scope['moves'].values():
            lo, hi = row['local_drop_interval_pp']
            assert -100 <= lo <= hi <= 100
            if row.get('total_gain_pp') is not None:
                assert row['database_move_gain_pp'] + row['continuation_gain_pp'] == pytest.approx(row['total_gain_pp'])
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in preserved.items())
    assert source.read_bytes() == source_before
    path.with_suffix('.insights.json').write_text(json.dumps(result))
    with monkeypatch.context() as patch:
        patch.setattr('repertoire_score.report_insights.database_samples', forbidden)
        patch.setattr('repertoire_score.report_insights.Explorer.get', forbidden)
        refreshed = refresh_opening_groups(path)
    assert refreshed['scopes'] == result['scopes']
    assert refreshed['opening_summary_groups'] == result['opening_summary_groups']
    assert refreshed['manifest']['created_at'] == result['manifest']['created_at']
    with monkeypatch.context() as patch:
        patch.setattr('repertoire_score.report_insights.database_samples', forbidden)
        spread_refresh = refresh_spreads(path, cache)
    assert spread_refresh['scopes'] == result['scopes']
    assert spread_refresh['opening_spreads'] == result['opening_spreads']
    assert spread_refresh['manifest']['recursive_spread_schema_version'] == 1
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in preserved.items())
    source.write_bytes(source_before + b'\n')
    with pytest.raises(ValueError, match='PGN differs from saved scores'):
        refresh_opening_groups(path)
    source.write_bytes(source_before)
    data = path.with_suffix('.preparation.json')
    d = json.loads(data.read_bytes()); d['manifest']['report_sha256'] = 'stale'; data.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='differs from saved score snapshot'):
        analyze(path, cache)
    with pytest.raises(ValueError, match='differs from saved report insights'):
        refresh_opening_groups(path)


def test_summary_is_narrow_and_new_insights_and_entry_explanations_are_visible(complete):
    path, _ = complete
    generate([path])
    full = (path.parent/'report.md').read_text(encoding='utf-8')
    summary = (path.parent/'summary.md').read_text(encoding='utf-8')
    for header in summary.splitlines():
        if header.startswith('| ') and not header.startswith('| ---'):
            assert len(header.strip('|').split('|')) <= 10
    assert '**Score uncertainty**' in summary and 'Sparse-evidence sensitivity' in summary
    assert '<summary>Score uncertainty</summary>' not in summary
    assert '### Score uncertainty' not in full + summary
    assert 'combined-score-uncertainty' not in full
    assert '### Score and evidence limits' in full
    uncertainty = summary.split('**Score uncertainty**', 1)[1].split('**Data snapshot**', 1)[0]
    assert '| Approximate 95% score interval |' in uncertainty
    assert (summary.index('| Repertoire |') < summary.index('## White repertoire')
            < summary.index('<summary>Evidence and definitions</summary>')
            < summary.index('**Score uncertainty**'))
    assert 'Branch score spread' in full and 'Move gain and preparation gain' in full
    page = (path.parent / 'chapters' / 'W1.md').read_text(encoding='utf-8')
    assert 'Where this chapter starts' in page
    assert page.index('Where this chapter starts') < page.index('## Where preparation ends') < page.index('## Exact first-entry positions')
    assert 'Gain split: move / prep' in full and '95%:' in full
    anchors = set(re.findall(r'<a id="([^"]+)"', full))
    assert all(a in anchors for a in re.findall(r'\]\(report.md#([^)]+)\)', summary))
