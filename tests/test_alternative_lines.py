import json
from types import SimpleNamespace

import httpx
import pytest
from helpers import cache_row, data, graph, position

from repertoire import score as cli

ITALIAN = (
    '[ChapterURL "https://lichess.org/study/test/italian"]\n[ChapterName "Italian"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bc4 *'
)
BISHOP = (
    '[ChapterURL "https://lichess.org/study/test/bishop"]\n[ChapterName "Bishop first"]\n\n1. e4 e5 2. Bc4 Nc6 3. Nf3 *'
)


def run(tmp_path, monkeypatch, text=f'{ITALIAN}\n\n{BISHOP}'):
    """2.Nf3 and 2.Bc4 compete after 1.e4 e5 and meet again after 2...Nc6 3.Bc4 / 3.Nf3 (60% for White).

    After 2.Nf3, 2...d6 (40%) is unprepared and scores 50%: 2.Nf3 scores 0.6 * 0.6 + 0.4 * 0.5 = 56%.
    After 2.Bc4, 2...Nf6 (30%) is unprepared and scores 30%: 2.Bc4 scores 0.7 * 0.6 + 0.3 * 0.3 = 51%.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError('A fully cached run must not use the network')

    monkeypatch.setattr(httpx.Client, 'request', forbidden)
    g = graph(tmp_path, text)
    cache = tmp_path / 'cache'
    cache.mkdir()
    tables = {
        position('e4'): data(50, 0, 50, [('e7e5', 50, 0, 50)]),
        position('e4 e5 Nf3'): data(56, 0, 44, [('b8c6', 36, 0, 24), ('d7d6', 20, 0, 20)]),
        position('e4 e5 Bc4'): data(51, 0, 49, [('b8c6', 42, 0, 28), ('g8f6', 9, 0, 21)]),
        position('e4 e5 Nf3 Nc6 Bc4'): data(60, 0, 40),
    }
    for k in g.nodes:
        cache_row(cache, k, tables.get(k, data(50, 0, 50)))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({}))
    args = SimpleNamespace(
        config=str(config),
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
    return json.loads((tmp_path / 'white.json').read_text())


def test_unplayed_move_is_split_into_lines_that_add_up_to_the_difference(tmp_path, monkeypatch):
    report = run(tmp_path, monkeypatch)
    [row] = report['alternatives']
    assert row['selected'] == 'g1f3'
    played, bishop = (next(o for o in row['options'] if o['move'] == m) for m in ('g1f3', 'f1c4'))
    assert played['score'] == pytest.approx(0.56) and bishop['score'] == pytest.approx(0.51)
    assert played['lines'] == []
    lines = {(r['kind'], r['path'][-1]): r for r in bishop['lines']}
    assert set(lines) == {('meet', 'Nf3'), ('stop', 'Nf6'), ('stop', 'd6')}
    meet = lines[('meet', 'Nf3')]
    # The shared position, labeled by the 2.Bc4 move order, with its share after each move.
    assert meet['path'] == ['e4', 'e5', 'Bc4', 'Nc6', 'Nf3'] and meet['position'] == position('e4 e5 Nf3 Nc6 Bc4')
    assert meet['played_path'] == ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4']
    assert lines[('stop', 'Nf6')]['played_path'] is None
    assert (meet['share'], meet['played_share'], meet['score']) == pytest.approx((0.7, 0.6, 0.6))
    assert meet['effect'] == pytest.approx(0.1 * (0.6 - 0.56))
    assert sorted(meet['chapters']) == ['bishop', 'italian']
    nf6, d6 = lines[('stop', 'Nf6')], lines[('stop', 'd6')]
    assert (nf6['stop'], nf6['move'], nf6['share'], nf6['played_share']) == ('deviation', 'g8f6', 0.3, 0)
    assert nf6['effect'] == pytest.approx(0.3 * (0.3 - 0.56)) and nf6['chapters'] == ['bishop']
    assert (d6['share'], d6['played_share'], d6['score']) == pytest.approx((0, 0.4, 0.5))
    assert d6['effect'] == pytest.approx(-0.4 * (0.5 - 0.56))
    assert sum(r['effect'] for r in bishop['lines']) == pytest.approx(bishop['score'] - played['score'])
    assert [r['path'][-1] for r in bishop['lines']] == ['Nf6', 'd6', 'Nf3']  # largest effect first


def test_alternatives_are_rendered_in_the_report_summary_and_chapter_pages(tmp_path, monkeypatch):
    run(tmp_path, monkeypatch)
    full = (tmp_path / 'report.md').read_text(encoding='utf-8')
    section = full[full.index('### White alternatives') :]
    section = section[: section.index('\n### ', 1)]
    assert section.index('| Position |') < section.index('#### 2. Bc4 instead of 2. Nf3 after 1. e4 e5')
    assert 'Bc4 · [lines](#white-alternatives-1-f1c4)' in section
    assert '<a id="white-alternatives-1-f1c4"></a>' in section
    assert '| Line | After 2. Bc4 | After 2. Nf3 | Score from there | Effect here |' in section
    assert 'After 2. Bc4, 70.0% of games transpose into positions 2. Nf3 also reaches (60.0% after 2. Nf3)' in section
    meet = '<br>transposition, also 2. Nf3 Nc6 3. Bc4 · [W1](chapters/W1.md), [W2](chapters/W2.md)'
    assert meet + ' | 70.0% | 60.0% | 60.0% | +0.40% |' in section
    assert '<br>unprepared in [W2](chapters/W2.md) | 30.0% |  | 30.0% | -7.80% |' in section
    assert '| **Total** | **100.0%** | **100.0%** |  | **-5.00%** |' in section
    assert full.index('### White chapters') < full.index('### White alternatives')
    summary = (tmp_path / 'summary.md').read_text(encoding='utf-8')
    assert '<summary>Alternatives (1)</summary>' in summary
    assert '[lines](report.md#white-alternatives-1-f1c4)' in summary
    assert '#### 2. Bc4 instead of' not in summary
    for page in ('W1', 'W2'):
        text = (tmp_path / 'chapters' / f'{page}.md').read_text(encoding='utf-8')
        anchor = f'white-chapter-{page[1:]}-alternatives'
        assert f'<a id="{anchor}"></a>\n\n## Alternatives' in text
        assert f'[Alternatives](#{anchor})' in text
        assert text.index('## Where this chapter starts') < text.index('## Alternatives')
        assert '### 2. Bc4 instead of 2. Nf3 after 1. e4 e5' in text
    bishop = (tmp_path / 'chapters' / 'W2.md').read_text(encoding='utf-8')
    assert '[How its moves compare](#white-chapter-2-alternatives)' in bishop


def test_side_variations_within_a_chapter_compete(tmp_path, monkeypatch):
    side = (
        '[ChapterURL "https://lichess.org/study/test/italian"]\n[ChapterName "Italian"]\n\n'
        '1. e4 e5 2. Bc4 (2. Nf3 Nc6 3. Bc4) 2... Nc6 3. Nf3 *'
    )
    report = run(tmp_path, monkeypatch, side)
    [row] = report['alternatives']
    # The side variation 2.Nf3 scores higher than the main line 2.Bc4, so it is played.
    assert row['selected'] == 'g1f3' and report['manifest']['selected_alternatives'] == {row['position']: 'g1f3'}
    options = {o['san']: o for o in row['options']}
    assert (options['Nf3']['sideline'], options['Bc4']['sideline']) == (True, False)
    assert options['Nf3']['chapters'] == options['Bc4']['chapters'] == ['italian']
    assert sum(r['effect'] for r in options['Bc4']['lines']) == pytest.approx(0.51 - 0.56)
    assert report['overall']['raw_empirical_score'] == pytest.approx(0.56)
    full = (tmp_path / 'report.md').read_text(encoding='utf-8')
    assert '| Nf3 **(played)** | [W1](chapters/W1.md) (side variation) | 56.0% |' in full
    assert '#### 2. Bc4 instead of 2. Nf3 after 1. e4 e5' in full


def test_moves_whose_lines_never_meet_are_split_by_the_replies(tmp_path, monkeypatch):
    side = (
        '[ChapterURL "https://lichess.org/study/test/italian"]\n[ChapterName "Italian"]\n\n'
        '1. e4 e5 2. Bc4 (2. Nf3 Nc6) *'
    )
    report = run(tmp_path, monkeypatch, side)
    [row] = report['alternatives']
    # 2.Bc4 has no prepared replies (70% at 60%, 30% at 30%: 51%); 2.Nf3 Nc6 is unanswered (50%), 2...d6 50%.
    assert row['selected'] == 'f1c4'
    knight = next(o for o in row['options'] if o['san'] == 'Nf3')
    assert knight['sideline'] and knight['score'] == pytest.approx(0.5)
    lines = {(r['kind'], r['path'][-1]): (r['share'], r['played_share']) for r in knight['lines']}
    assert lines == pytest.approx(
        {('only', 'Nc6'): (0.6, 0), ('stop', 'd6'): (0.4, 0), ('stop', 'Nc6'): (0, 0.7), ('stop', 'Nf6'): (0, 0.3)}
    )
    assert sum(r['effect'] for r in knight['lines']) == pytest.approx(0.5 - 0.51)
