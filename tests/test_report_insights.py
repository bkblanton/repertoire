import hashlib
import json
import math
import re
from pathlib import Path

import httpx
import pytest
from helpers import run_fixture

from repertoire import character, openings, preparation, vulnerabilities
from repertoire.report.generate import generate
from repertoire.report.links import Chapters
from repertoire.report.sections import gap_section
from repertoire.report_insights import analyze, branch_score_spread, gap_priorities, move_decomposition


def test_gap_shares_merge_transposed_arrivals_before_squaring():
    result = gap_priorities(
        dict(
            gaps=[dict(position='same', reach=0.1), dict(position='other', reach=0.2), dict(position='same', reach=0.2)]
        )
    )
    a, b = result['priorities']
    assert a['reach'] == pytest.approx(0.3) and a['position'] == 'same'
    assert a['repeat_probability_share'] == pytest.approx(0.09 / 0.13)
    assert result['known_repeat_probability'] == pytest.approx(0.13)
    assert sum(r['repeat_probability_share'] for r in result['priorities']) == pytest.approx(1.0)
    assert b['cumulative_repeat_probability_share'] == pytest.approx(1.0)
    partial = gap_priorities(dict(gaps=[dict(position='a', reach=0.1)], unresolved_mass=0.9))
    assert partial['share_basis'] == 'known first gaps'


def test_branch_spread_separates_transposed_cohorts_and_preserves_duplicates():
    def stop(p, s, board):
        return dict(reach=p, score=s, position=board, type='deviation')

    rows = [stop(0.5, 0.2, 'same'), stop(0.5, 0.8, 'same')]
    result = branch_score_spread(rows, 0.5)
    assert result['standard_deviation'] == pytest.approx(0.3)
    assert result['between_position_deviation'] == 0
    assert result['within_position_deviation'] == pytest.approx(0.3)
    rows[1]['position'] = 'different'
    result = branch_score_spread(rows, 0.5)
    assert result['between_position_deviation'] == pytest.approx(0.3)
    assert result['within_position_deviation'] == 0
    split = [stop(0.25, 0.2, 'same'), stop(0.25, 0.2, 'same'), rows[1]]
    assert branch_score_spread(split, 0.5)['standard_deviation'] == result['standard_deviation']
    assert branch_score_spread([stop(0.5, None, 'unknown'), stop(0.5, 0.8, 'known')])['standard_deviation'] is None


def test_move_gain_components_sum_even_when_the_database_move_hurts():
    result = move_decomposition(dict(reference_score=0.5, move_database_score=0.4, move_score=0.6))
    assert result == pytest.approx(dict(database_move_gain_pp=-10, continuation_gain_pp=20, total_gain_pp=10))
    assert (
        move_decomposition(dict(reference_score=0.5, move_database_score=None, move_score=0.6))['total_gain_pp'] is None
    )


def test_gap_priority_table_retains_full_denominator_and_five_summary_rows():
    positions = [
        dict(
            position=str(i),
            line=f'gap {i}',
            reach=0.1,
            database_score=0.5,
            games=100,
            kind='unprepared_reply',
            to_move='white',
        )
        for i in range(10)
    ]
    metrics = dict(equivalent_gap_reach=math.sqrt(0.1), gaps=positions, unresolved_mass=0)
    scope = dict(id='overall', positions=positions, gap_coverage=metrics, gap_priorities=gap_priorities(metrics))
    text = '\n'.join(gap_section(scope, refs=Chapters(dict(color='white', chapters=[])), top=5))
    assert text.count('| [gap ') == 5
    assert '**50.0%**' in text
    assert text.count('10.0% | 50.0%') == 5


def test_cached_insights_preserve_scores_cache_and_source_and_reject_stale_inputs(tmp_path, monkeypatch):
    saved, cache = run_fixture(tmp_path, monkeypatch, common_entry=True)
    path = tmp_path / 'white.json'
    for name, module in [
        ('vulnerabilities', vulnerabilities),
        ('preparation', preparation),
        ('character', character),
        ('openings', openings),
    ]:
        path.with_suffix(f'.{name}.json').write_text(json.dumps(module.analyze(path, cache)))
    preserved = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [path, *cache.glob('*.json')]}
    source = Path(saved['manifest']['input_path'])
    source_before = source.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError('network forbidden')

    monkeypatch.setattr(httpx.Client, 'request', forbidden)
    result = analyze(path, cache)
    again = analyze(path, cache)
    assert result['scopes'] == again['scopes']
    assert result['manifest']['network_requests'] == 0
    for scope in result['scopes']:
        for sources in scope.get('entry_opening_sources', {}).values():
            assert sum(r['share'] for r in sources) == pytest.approx(1.0)
        for row in scope['moves'].values():
            lo, hi = row['local_drop_interval_pp']
            assert -100 <= lo <= hi <= 100
            if row.get('total_gain_pp') is not None:
                assert row['database_move_gain_pp'] + row['continuation_gain_pp'] == pytest.approx(row['total_gain_pp'])
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in preserved.items())
    assert source.read_bytes() == source_before
    source.write_bytes(source_before + b'\n')
    with pytest.raises(ValueError, match='PGN changed since scoring'):
        analyze(path, cache)
    source.write_bytes(source_before)
    data = path.with_suffix('.preparation.json')
    d = json.loads(data.read_bytes())
    d['manifest']['report_sha256'] = 'stale'
    data.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='belongs to a different score snapshot'):
        analyze(path, cache)


def test_summary_is_narrow_and_new_insights_and_entry_explanations_are_visible(complete):
    path, _ = complete
    generate([path])
    full = (path.parent / 'report.md').read_text(encoding='utf-8')
    summary = (path.parent / 'summary.md').read_text(encoding='utf-8')
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
    assert (
        summary.index('| Repertoire |')
        < summary.index('## White repertoire')
        < summary.index('<summary>Evidence and definitions</summary>')
        < summary.index('**Score uncertainty**')
    )
    assert 'Branch score spread' in full and 'Move gain and preparation gain' in full
    page = (path.parent / 'chapters' / 'W1.md').read_text(encoding='utf-8')
    assert 'Where this chapter starts' in page
    assert (
        page.index('Where this chapter starts')
        < page.index('## Where preparation ends')
        < page.index('## Exact first-entry positions')
    )
    assert '| Gain split |' in full and '95%:' in full
    anchors = set(re.findall(r'<a id="([^"]+)"', full))
    assert all(a in anchors for a in re.findall(r'\]\(report.md#([^)]+)\)', summary))
