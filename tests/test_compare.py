import json
import re
from types import SimpleNamespace

import httpx
import pytest
from helpers import cache_row, data, position

from repertoire_score import compare, studies
from repertoire_score import score as cli
from repertoire_score.alternatives import Choice, find_decisions, repertoire_reference
from repertoire_score.graph import parse, parse_games, read_games, resolve

REPERTOIRE = '''[ChapterURL "https://lichess.org/study/mine0001/tarrasc1"]
[ChapterName "Tarrasch"]

1. e4 e6 2. d4 d5 3. Nd2 Nf6 4. e5 *

[ChapterURL "https://lichess.org/study/mine0001/vienna01"]
[ChapterName "Vienna"]

1. e4 e5 2. Nc3 *
'''

ADVANCE = '''[ChapterURL "https://lichess.org/study/cand0001/advance1"]
[ChapterName "Advance"]
[StudyName "French ideas"]

1. e4 e6 2. d4 d5 3. e5 c5 4. c3 *
'''

SCHLECHTER = '''[ChapterURL "https://lichess.org/study/cand0001/schlech1"]
[ChapterName "Schlechter"]

1. e4 e6 2. d4 d5 3. Bd3 c5 *
'''

KING_KNIGHT = '''[ChapterURL "https://lichess.org/study/cand0001/knight01"]
[ChapterName "King's knight"]

1. e4 e5 2. Nf3 Nc6 *
'''

ADVANCE_NF3 = '''[ChapterURL "https://lichess.org/study/cand0001/advance2"]
[ChapterName "Advance Nf3"]

1. e4 e6 2. d4 d5 3. e5 c5 4. Nf3 *
'''

# White's score where each line ends.
LEAVES = {
    'e4 e6 d4 d5 Nd2 Nf6 e5': 60,
    'e4 e6 d4 d5 e5 c5 c3': 40,
    'e4 e6 d4 d5 e5 c5 Nf3': 80,
    'e4 e6 d4 d5 Bd3 c5': 70,
    'e4 e5 Nc3': 55,
    'e4 e5 Nf3 Nc6': 50,
}


def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('A cached comparison must not use the network')

    monkeypatch.setattr(httpx.Client, 'request', forbidden)


def scored(tmp_path, candidate_text):
    """Score the repertoire offline, with tables cached for every repertoire and candidate position."""
    (tmp_path / 'repertoire.pgn').write_text(REPERTOIRE, encoding='utf-8')
    (tmp_path / 'candidate.pgn').write_text(candidate_text, encoding='utf-8')
    union = parse_games([*read_games(tmp_path / 'candidate.pgn'), *read_games(tmp_path / 'repertoire.pgn')])
    cache = tmp_path / 'cache'
    cache.mkdir()
    leaves = {position(k): v for k, v in LEAVES.items()}
    for k, node in union.nodes.items():
        if k in leaves:
            cache_row(cache, k, data(leaves[k], 0, 100 - leaves[k]))
        else:
            moves = list(node.edges)
            weight = 100 // len(moves)
            cache_row(cache, k, data(50, 0, 50, [(m, weight // 2, 0, weight // 2) for m in moves]))
    folder = tmp_path / 'reports' / 'data'
    folder.mkdir(parents=True)
    cli.analyze(
        SimpleNamespace(
            config=None,
            pgn=str(tmp_path / 'repertoire.pgn'),
            color='white',
            output=str(folder / 'white'),
            command='run',
            cache=str(cache),
            offline=True,
            refresh=False,
            prior=[0.5] * 3,
            sparse_threshold=30,
            tolerance=1,
        )
    )
    return folder / 'white.json', cache


def run(tmp_path, monkeypatch, candidate_text, **options):
    path, cache = scored(tmp_path, candidate_text)
    no_network(monkeypatch)
    done = compare.run(
        [str(tmp_path / 'candidate.pgn')], color='white', score=path, cache=str(cache), offline=True, **options
    )
    return json.loads((path.parent / 'comparisons' / f"{done['name']}.json").read_text()), done['page']


def test_competing_and_independent_alternatives(tmp_path, monkeypatch):
    result, page = run(tmp_path, monkeypatch, '\n\n'.join([ADVANCE, SCHLECHTER, KING_KNIGHT]))
    assert result['name'] == 'french-ideas'
    decisions = {d['line']: d for d in result['decisions']}
    french = decisions['1.e4 1...e6 2.d4 2...d5']
    # Your 3.Nd2 and the two candidate chapters compete at one board.
    assert [(o['label'], o['source']) for o in french['options']] == [
        ('3.Nd2', 'repertoire'),
        ('3.e5', 'candidate'),
        ('3.Bd3', 'candidate'),
    ]
    assert [o['score'] for o in french['options']] == pytest.approx([0.6, 0.4, 0.7])
    assert french['options'][0]['repertoire_chapters'] == [dict(id='tarrasc1', name='Tarrasch', page='W1')]
    assert french['options'][2]['improving'] and french['options'][2]['all']
    vienna = decisions['1.e4 1...e5']
    assert [o['score'] for o in vienna['options']] == pytest.approx([0.55, 0.5])
    # Keeping 2.Nc3 is better, but "all alternatives" always switches.
    assert vienna['options'][0]['improving'] and vienna['options'][1]['all']
    scenarios = result['scenarios']
    assert scenarios['current']['score'] == pytest.approx(0.575)
    assert scenarios['improving']['change'] == pytest.approx(0.5 * 0.1)
    assert scenarios['all']['change'] == pytest.approx(0.5 * 0.1 - 0.5 * 0.05)
    # The two decisions share no positions, so their gains add up.
    assert result['validation'] == dict(
        current_score_reproduced=True,
        improving_additive=True,
        all_additive=True,
        adopt_pgn_reproduces_selection=True,
    )
    assert result['entry']['position'] == position('e4')
    adopt = read_games(page.with_name('french-ideas.adopt.pgn'))
    assert [g.headers['ChapterName'] for g in adopt] == ['Schlechter']
    text = page.read_text(encoding='utf-8')
    assert '**Adopt 3.Bd3 against 2...d5**; keep 2.Nc3 against 1...e5.' in text
    assert 'the candidate chapters compete; all alternatives uses the higher-scoring 3.Bd3' in text
    assert '[W1](../chapters/W1.md)' in text and '—' not in text
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', text))
    assert set(re.findall(r'\]\(#([^)]+)\)', text)) <= anchors
    for block in re.findall(r'(?m)(?:^\|.*\|\n)+', text):
        widths = {len(row.split('|')) for row in block.splitlines()}
        assert len(widths) == 1


def test_chapters_sharing_a_move_compete_further_down(tmp_path, monkeypatch):
    result, _ = run(tmp_path, monkeypatch, '\n\n'.join([ADVANCE, ADVANCE_NF3]), name='advance')
    outer, inner = result['decisions']
    assert [o['label'] for o in outer['options']] == ['3.Nd2', '3.e5']
    assert [c['name'] for c in outer['options'][1]['chapters']] == ['Advance', 'Advance Nf3']
    # Inside the pooled 3.e5 option, the earlier chapter's 4.c3 is the reference for 4.Nf3.
    assert inner['parent'] == [outer['id'], 1]
    assert [(o['label'], o['source']) for o in inner['options']] == [('4.c3', 'candidate'), ('4.Nf3', 'candidate')]
    assert [o['score'] for o in inner['options']] == pytest.approx([0.4, 0.8])
    assert outer['options'][1]['improving'] and inner['options'][1]['improving']
    assert result['scenarios']['improving']['entry_score'] == pytest.approx(0.8)
    assert result['groups'][0]['improving']['combinations'] == 3


def test_decisions_follow_each_chapter_against_the_repertoire(tmp_path):
    (tmp_path / 'repertoire.pgn').write_text(REPERTOIRE, encoding='utf-8')
    repertoire = parse(tmp_path / 'repertoire.pgn')
    reference = repertoire_reference(resolve(repertoire, True, {}))
    candidate_path = tmp_path / 'candidate.pgn'
    # The first chapter agrees with 3.Nd2 and adds 4...Nfd7 5.c3, where you have no move after 4.e5.
    candidate_path.write_text(
        '[ChapterName "Agrees"]\n\n1. e4 e6 2. d4 d5 3. Nd2 Nf6 4. e5 Nfd7 5. c3 *\n\n' + SCHLECHTER,
        encoding='utf-8',
    )
    candidate = parse(candidate_path)
    decisions = find_decisions(candidate, True, reference)
    by_line = {' '.join(candidate.nodes[d.position].path): d for d in decisions}
    assert set(by_line) == {'e4 e6 d4 d5', 'e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7'}
    added = by_line['e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7']
    assert added.options[0].move is None and added.options[1].move == 'c2c3'
    # The agreeing chapter passes through 3.Nd2, so keeping your move keeps its route to the later decision.
    assert by_line['e4 e6 d4 d5'].options[0].chapters == ['1']
    assert Choice({added.id: 1}).active(decisions) == decisions


def test_study_urls_are_exported_with_orientation_and_chapter_links(tmp_path, monkeypatch):
    monkeypatch.setenv('LICHESS_TOKEN', 'test')
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, text='[Orientation "black"]\n[StudyName "Caro"]\n\n1. e4 c6 *\n')

    client = httpx.Client(transport=httpx.MockTransport(handler))
    games, name, sources = compare.load_inputs(
        ['https://lichess.org/study/abcdEFGH/chap0001'], directory=tmp_path, client=client
    )
    assert str(seen[0].url).startswith('https://lichess.org/api/study/abcdEFGH/chap0001.pgn')
    assert seen[0].url.params['orientation'] == 'true'
    assert name == 'caro' and compare.infer_color(games) == 'black'
    assert sources[0]['kind'] == 'lichess' and (tmp_path / 'caro.pgn').is_file()
    # A later run without export reuses the saved copy.
    again, _, _ = compare.load_inputs(['https://lichess.org/study/abcdEFGH/chap0001'], 'caro', False, tmp_path)
    assert len(again) == 1
    assert studies.chapter_url('abcdEFGH') == 'https://lichess.org/api/study/abcdEFGH.pgn'
    with pytest.raises(ValueError, match='No saved export'):
        compare.load_inputs(['https://lichess.org/study/zzzzzzzz'], 'other', False, tmp_path)


def test_saved_comparisons_rerun_and_appear_in_the_summary(tmp_path, monkeypatch):
    result, page = run(tmp_path, monkeypatch, '\n\n'.join([ADVANCE, SCHLECHTER]), name='french')
    path = tmp_path / 'reports' / 'data' / 'white.json'
    registry = tmp_path / 'comparisons.json'
    outside = tmp_path / 'outside'
    outside.mkdir()
    candidate = outside / 'study.pgn'
    candidate.write_text(ADVANCE, encoding='utf-8')
    monkeypatch.chdir(tmp_path / 'reports')
    compare.register(registry, 'french', 'white', [str(candidate)], '1.e4 e6 2.d4 d5', directory='candidates')
    # A file outside the working folder is copied in, so the registry never names a local folder.
    saved = compare.load_registry(registry)
    assert saved == [dict(name='french', color='white', sources=['candidates/french.pgn'], entry='1.e4 e6 2.d4 d5')]
    assert (tmp_path / 'reports' / 'candidates' / 'french.pgn').read_text(encoding='utf-8') == ADVANCE
    compare.run_registered(registry, score=path, cache=str(tmp_path / 'cache'), offline=True)
    rerun = json.loads((path.parent / 'comparisons' / 'french.json').read_text())
    assert rerun['entry']['basis'] == 'given' and len(rerun['chapters']) == 1
    from repertoire_score.report.generate import generate

    generate([path])
    summary = (tmp_path / 'reports' / 'summary.md').read_text(encoding='utf-8')
    assert '## Comparisons' in summary and '[French ideas](comparisons/french.md)' in summary
    # A comparison made from an older score is marked out of date rather than shown as current.
    data = json.loads(path.read_text())
    data['manifest']['created_at'] = 'changed'
    path.write_text(json.dumps(data))
    generate([path], strict=False)
    assert '**Out of date:**' in (tmp_path / 'reports' / 'summary.md').read_text(encoding='utf-8')
