import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import chess.pgn
import httpx
import pytest
from helpers import cache_row, data, graph, position

from repertoire import build, chapter_order, coverage
from repertoire import score as cli
from repertoire.context import AnalysisContext


def chess_games(text):
    stream, games = io.StringIO(text), []
    while (game := chess.pgn.read_game(stream)) is not None:
        games.append(game)
    return games


def check_links(directory):
    pages = list(Path(directory).rglob('*.md'))
    anchors = {p.resolve(): set(re.findall(r'<a id="([^"]+)"></a>', p.read_text(encoding='utf-8'))) for p in pages}
    for page in pages:
        for target in re.findall(r'\]\(([^)\s]+)\)', page.read_text(encoding='utf-8')):
            if target.startswith('https://'):
                continue
            name, _, anchor = target.partition('#')
            destination = (page.parent / name).resolve() if name else page.resolve()
            assert destination.exists(), (page, target)
            assert not anchor or anchor in anchors[destination], (page, target)


def score_pgn(tmp_path, monkeypatch, pgn_text, tables, *, color='white', config=None, name='fixture'):
    monkeypatch.setattr(httpx.Client, 'request', lambda *a, **k: pytest.fail('Must stay offline'))
    pgn = tmp_path / f'{name}.pgn'
    pgn.write_text(pgn_text, encoding='utf-8')
    g = graph(tmp_path, pgn_text)
    cache = tmp_path / 'cache'
    cache.mkdir(exist_ok=True)
    for k in {*g.nodes, *tables}:
        rows = tables.get(k, [])
        if rows is None:
            cache_row(cache, k, data(0, 0, 0, []))
            continue
        totals = [sum(r[i] for r in rows) for i in (1, 2, 3)] if rows else [50, 0, 50]
        cache_row(cache, k, data(*totals, rows))
    cfg = tmp_path / f'{name}.config.json'
    cfg.write_text(json.dumps(config or {}), encoding='utf-8')
    out = tmp_path / f'{name}_{color}'
    cli.analyze(
        SimpleNamespace(
            config=str(cfg),
            pgn=str(pgn),
            color=color,
            output=str(out),
            command='run',
            cache=str(cache),
            offline=True,
            refresh=False,
            prior=[0.5] * 3,
            sparse_threshold=30,
            tolerance=1,
        )
    )
    return out.with_suffix('.json'), cache


def test_chapter_order_config_validation():
    assert chapter_order.order_from_config({}) == 'study'
    assert chapter_order.order_from_config({'chapter_order': 'study'}) == 'study'
    assert chapter_order.order_from_config({'chapter_order': 'opening'}) == 'opening'
    for bad in ('reach', '', True, 1, None, ['opening']):
        with pytest.raises(ValueError, match='chapter_order'):
            chapter_order.order_from_config({'chapter_order': bad})


def test_white_recursive_move_prefix_grouping_and_ordering(tmp_path, monkeypatch):
    # Study order is deliberately scrambled across 1...e5, 1...e6 and 1...c5.
    # French (e6) has two sub-chapters (20% + 18% = 38%) + a multi-entry 2nd-move sidelines chapter (4%) -> 42% union.
    # Open (e5) is one 30% chapter: bigger than any single French chapter, smaller than the French group.
    # Sicilian (c5) has two sub-chapters (15% + 10% = 25% union).
    pgn = '\n\n'.join(
        [
            '[ChapterURL "https://lichess.org/study/t/fr-side"]\n[ChapterName "French sidelines"]\n\n'
            '1. e4 e6 2. d4 b6 3. Nc3 * (2... a6 3. Nf3 *)',
            '[ChapterURL "https://lichess.org/study/t/sic-d6"]\n[ChapterName "Sicilian d6"]\n\n'
            '1. e4 c5 2. Nc3 d6 3. g3 *',
            '[ChapterURL "https://lichess.org/study/t/open"]\n[ChapterName "Open"]\n\n1. e4 e5 2. Nf3 Nc6 *',
            '[ChapterURL "https://lichess.org/study/t/fr-nf6"]\n[ChapterName "French Tarrasch Nf6"]\n\n'
            '1. e4 e6 2. d4 d5 3. Nd2 Nf6 4. e5 *',
            '[ChapterURL "https://lichess.org/study/t/sic-nc6"]\n[ChapterName "Sicilian Nc6"]\n\n'
            '1. e4 c5 2. Nc3 Nc6 3. g3 *',
            '[ChapterURL "https://lichess.org/study/t/fr-c5"]\n[ChapterName "French Tarrasch c5"]\n\n'
            '1. e4 e6 2. d4 d5 3. Nd2 c5 4. Ngf3 *',
        ]
    )
    # Build valid PGN variation syntax for fr-side.
    pgn = pgn.replace('1. e4 e6 2. d4 b6 3. Nc3 * (2... a6 3. Nf3 *)', '1. e4 e6 2. d4 b6 (2... a6 3. Nf3) 3. Nc3 *')
    tables = {
        position('e4'): [('e7e6', 210, 0, 210), ('e7e5', 150, 0, 150), ('c7c5', 125, 0, 125), ('d7d5', 15, 0, 15)],
        position('e4 e6 d4'): [('d7d5', 190, 0, 190), ('b7b6', 15, 0, 15), ('a7a6', 5, 0, 5)],
        position('e4 e6 d4 d5 Nd2'): [('c7c5', 100, 0, 100), ('g8f6', 90, 0, 90)],
        position('e4 c5 Nc3'): [('b8c6', 75, 0, 75), ('d7d6', 50, 0, 50)],
        position('e4 e5 Nf3'): [('b8c6', 150, 0, 150)],
    }
    score_path, cache = score_pgn(tmp_path, monkeypatch, pgn, tables)
    analysis = AnalysisContext(score_path)
    evidence = analysis.read_evidence(cache, required=True)
    result = chapter_order.order_chapters(analysis, evidence, 'opening')
    assert result['mode'] == 'opening'
    # Outer: French (42%) > Open (30%) > Sicilian (25%).
    # Inside French: 2...d5 group (38%: fr-c5 20% > fr-nf6 18%) > fr-side (4%, anchored at 1.e4 e6 2.d4).
    # Inside Sicilian: sic-nc6 (15%) > sic-d6 (10%).
    assert result['ids'] == ['fr-c5', 'fr-nf6', 'fr-side', 'open', 'sic-nc6', 'sic-d6']
    assert chapter_order.order_chapters(analysis, evidence, 'study')['ids'] == [
        'fr-side',
        'sic-d6',
        'open',
        'fr-nf6',
        'sic-nc6',
        'fr-c5',
    ]


def test_parent_anchored_high_reach_chapter_sorts_before_smaller_subopening(tmp_path, monkeypatch):
    # A broad chapter split 50/50 at 1.e4 e6 2.d4 (2...b6 12% + 2...a6 12% = 24%) stays anchored at 2.d4 and beats
    # a smaller 2...d5 chapter (10%).
    pgn = (
        '[ChapterURL "https://lichess.org/study/t/sub"]\n[ChapterName "Sub"]\n\n'
        '1. e4 e6 2. d4 d5 3. Nd2 *\n\n'
        '[ChapterURL "https://lichess.org/study/t/broad"]\n[ChapterName "Broad"]\n\n'
        '1. e4 e6 2. d4 b6 (2... a6 3. Nf3) 3. Nc3 *'
    )
    tables = {
        position('e4'): [('e7e6', 50, 0, 50)],
        position('e4 e6 d4'): [('b7b6', 12, 0, 12), ('a7a6', 12, 0, 12), ('d7d5', 10, 0, 10), ('c7c5', 66, 0, 66)],
    }
    score_path, cache = score_pgn(tmp_path, monkeypatch, pgn, tables)
    analysis = AnalysisContext(score_path)
    result = chapter_order.order_chapters(analysis, analysis.read_evidence(cache, required=True), 'opening')
    assert result['ids'] == ['broad', 'sub']


def test_black_groups_by_white_first_move_and_recurses(tmp_path, monkeypatch):
    # Black faces 1.d4 (45%: 2.c4 30% + 2.Bf4 15%) and 1.e4 (40%: 2.Nf3 25% + 2.Nc3 15%) and 1.c4 (15%).
    pgn = '\n\n'.join(
        [
            '[ChapterURL "https://lichess.org/study/t/eng"]\n[ChapterName "English"]\n\n1. c4 e5 *',
            '[ChapterURL "https://lichess.org/study/t/d4-bf4"]\n[ChapterName "London"]\n\n1. d4 d5 2. Bf4 Nf6 *',
            '[ChapterURL "https://lichess.org/study/t/e4-nc3"]\n[ChapterName "Vienna"]\n\n1. e4 e5 2. Nc3 Nf6 *',
            '[ChapterURL "https://lichess.org/study/t/d4-c4"]\n[ChapterName "QG"]\n\n1. d4 d5 2. c4 e6 *',
            '[ChapterURL "https://lichess.org/study/t/e4-nf3"]\n[ChapterName "Open"]\n\n1. e4 e5 2. Nf3 Nc6 *',
        ]
    )
    tables = {
        position(''): [('d2d4', 225, 0, 225), ('e2e4', 200, 0, 200), ('c2c4', 75, 0, 75)],
        position('d4 d5'): [('c2c4', 150, 0, 150), ('c1f4', 75, 0, 75)],
        position('e4 e5'): [('g1f3', 125, 0, 125), ('b1c3', 75, 0, 75)],
    }
    score_path, cache = score_pgn(tmp_path, monkeypatch, pgn, tables, color='black')
    analysis = AnalysisContext(score_path)
    result = chapter_order.order_chapters(analysis, analysis.read_evidence(cache, required=True), 'opening')
    assert result['ids'] == ['d4-c4', 'd4-bf4', 'e4-nf3', 'e4-nc3', 'eng']


def test_union_reach_does_not_double_count_overlapping_chapters(tmp_path, monkeypatch):
    # Two French chapters overlap: fr-root enters at 1.e4 e6 (30%) and fr-deep enters at 1.e4 e6 2.d4 d5 (25%).
    # Summing chapter reaches would give 55%, beating Sicilian (40%), but the true union first-entry reach is 30%.
    pgn = (
        '[ChapterURL "https://lichess.org/study/t/fr-root"]\n[ChapterName "French root"]\n\n'
        '1. e4 e6 2. d4 d5 *\n\n'
        '[ChapterURL "https://lichess.org/study/t/fr-deep"]\n[ChapterName "French deep"]\n\n'
        '1. e4 e6 2. d4 d5 3. Nd2 Nf6 *\n\n'
        '[ChapterURL "https://lichess.org/study/t/sic"]\n[ChapterName "Sicilian"]\n\n'
        '1. e4 c5 2. Nf3 d6 *'
    )
    tables = {
        position('e4'): [('c7c5', 200, 0, 200), ('e7e6', 150, 0, 150), ('e7e5', 150, 0, 150)],
        position('e4 e6 d4'): [('d7d5', 125, 0, 125), ('c7c5', 25, 0, 25)],
    }
    config = {'entries': {'fr-root': [{'path': ['e4', 'e6']}], 'fr-deep': [{'path': ['e4', 'e6', 'd4', 'd5']}]}}
    score_path, cache = score_pgn(tmp_path, monkeypatch, pgn, tables, config=config)
    analysis = AnalysisContext(score_path)
    result = chapter_order.order_chapters(analysis, analysis.read_evidence(cache, required=True), 'opening')
    assert result['ids'] == ['sic', 'fr-root', 'fr-deep']


def test_alternative_chapters_do_not_inflate_group_and_follow_played_chapters(tmp_path, monkeypatch):
    # Competing French alternatives (3.Nd2 wins over 3.e5; 3.e5 appears earlier in study order).
    # French overall union reach is 30% (NOT 60%), so Sicilian (40%) stays ahead of French.
    # Inside 1.e4 e6 2.d4 d5, played 3.Nd2 precedes unplayed 3.e5, and both precede 2...b6 sideline (3%).
    # Unplayed move-1 alternatives (1.d4 d5 15% comp, 1.d4 Nf6 25% comp) sit at the end ordered by comparison reach.
    pgn = '\n\n'.join(
        [
            '[ChapterURL "https://lichess.org/study/t/d4-d5"]\n[ChapterName "Alt d4 d5"]\n\n1. d4 d5 2. c4 *',
            '[ChapterURL "https://lichess.org/study/t/fr-adv"]\n[ChapterName "French Advance alt"]\n\n'
            '1. e4 e6 2. d4 d5 3. e5 *',
            '[ChapterURL "https://lichess.org/study/t/fr-side"]\n[ChapterName "French sideline"]\n\n'
            '1. e4 e6 2. d4 b6 3. Nc3 *',
            '[ChapterURL "https://lichess.org/study/t/fr-tar"]\n[ChapterName "French Tarrasch"]\n\n'
            '1. e4 e6 2. d4 d5 3. Nd2 *',
            '[ChapterURL "https://lichess.org/study/t/d4-nf6"]\n[ChapterName "Alt d4 Nf6"]\n\n1. d4 Nf6 2. c4 *',
            '[ChapterURL "https://lichess.org/study/t/sic"]\n[ChapterName "Sicilian"]\n\n1. e4 c5 2. Nf3 d6 *',
        ]
    )
    tables = {
        position('e4'): [('c7c5', 200, 0, 200), ('e7e6', 165, 0, 165), ('e7e5', 135, 0, 135)],
        position('e4 c5 Nf3'): [('d7d6', 80, 0, 20)],
        position('e4 e6 d4'): [('d7d5', 150, 0, 150), ('b7b6', 15, 0, 15)],
        position('e4 e6 d4 d5 Nd2'): [('g8f6', 80, 0, 20)],
        position('e4 e6 d4 d5 e5'): [('c7c5', 20, 0, 80)],
        position('d4'): [('g8f6', 25, 0, 75), ('d7d5', 15, 0, 65)],
        position('d4 d5 c4'): [('e7e6', 20, 0, 80)],
        position('d4 Nf6 c4'): [('e7e6', 20, 0, 80)],
    }
    score_path, cache = score_pgn(tmp_path, monkeypatch, pgn, tables)
    analysis = AnalysisContext(score_path)
    assert analysis.manifest['selected_alternatives'][position('e4 e6 d4 d5')] == 'b1d2'
    result = chapter_order.order_chapters(analysis, analysis.read_evidence(cache, required=True), 'opening')
    assert result['ids'] == ['sic', 'fr-tar', 'fr-adv', 'fr-side', 'd4-nf6', 'd4-d5']


def test_shared_explicit_entry_places_played_chapter_before_earlier_alternative(tmp_path, monkeypatch):
    # Losing alternative appears first in study order, shares explicit entry 1.e4 e6 with winning chapter.
    pgn = (
        '[ChapterURL "https://lichess.org/study/t/alt"]\n[ChapterName "Losing alt"]\n\n'
        '1. e4 e6 2. d4 d5 3. e5 *\n\n'
        '[ChapterURL "https://lichess.org/study/t/main"]\n[ChapterName "Winning main"]\n\n'
        '1. e4 e6 2. d4 d5 3. Nd2 *'
    )
    tables = {
        position('e4'): [('e7e6', 50, 0, 50)],
        position('e4 e6 d4'): [('d7d5', 50, 0, 50)],
        position('e4 e6 d4 d5 Nd2'): [('g8f6', 80, 0, 20)],
        position('e4 e6 d4 d5 e5'): [('c7c5', 20, 0, 80)],
    }
    config = {'entries': {'alt': [{'path': ['e4', 'e6']}], 'main': [{'path': ['e4', 'e6']}]}}
    score_path, cache = score_pgn(tmp_path, monkeypatch, pgn, tables, config=config)
    analysis = AnalysisContext(score_path)
    assert analysis.manifest['selected_alternatives'][position('e4 e6 d4 d5')] == 'b1d2'
    result = chapter_order.order_chapters(analysis, analysis.read_evidence(cache, required=True), 'opening')
    assert result['ids'] == ['main', 'alt']


def test_edge_cases_zero_reach_entry_unresolved_custom_fen_and_loop(tmp_path, monkeypatch):
    # 1. fr-zero has two entries: 3...Nf6 (positive reach) and 1...a6 (0 observed games in Explorer).
    #    The zero-reach entry must be ignored so fr-zero stays anchored inside 1.e4 e6, not pulled to 1.e4.
    # 2. fr-unres has an entry behind an unresolved (0-game) reply table; structural fallback keeps it under 1.e4 e6.
    # 3. loop contains a repetition loop (2.Nc3 Qf6 3.Nd5 Qd8 4.Nc3).
    # 4. custom is a disconnected custom-FEN chapter.
    custom_fen = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2'
    # Make custom_fen genuinely disconnected from 1.e4 (no route from starting position).
    custom_fen = '4k3/8/8/8/8/8/4P3/4K3 w - - 0 1'
    pgn = '\n\n'.join(
        [
            f'[SetUp "1"]\n[FEN "{custom_fen}"]\n[ChapterURL "https://lichess.org/study/t/custom"]\n'
            '[ChapterName "Endgame"]\n\n1. e3 Kf7 *',
            '[ChapterURL "https://lichess.org/study/t/fr-unres"]\n[ChapterName "French unresolved"]\n\n'
            '1. e4 e6 2. d4 c5 3. d5 *',
            '[ChapterURL "https://lichess.org/study/t/loop"]\n[ChapterName "Loop"]\n\n'
            '1. e4 e5 2. Nc3 Qf6 3. Nd5 Qd8 4. Nc3 *',
            '[ChapterURL "https://lichess.org/study/t/fr-zero"]\n[ChapterName "French main"]\n\n'
            '1. e4 e6 (1... a6 2. d4) 2. d4 d5 3. Nd2 Nf6 *',
        ]
    )
    tables = {
        position('e4'): [('e7e5', 300, 0, 300), ('e7e6', 200, 0, 200)],
        position('e4 e6 d4'): [('d7d5', 150, 0, 150), ('c7c5', 50, 0, 50)],
        position('e4 e6 d4 c5'): None,  # unresolved empty table -> 0 known reach into 3.d5
        position('e4 e5 Nc3'): [('d8f6', 100, 0, 100)],
        position('e4 e5 Nc3 Qf6 Nd5'): [('f6d8', 50, 0, 50), ('f6c6', 50, 0, 50)],
    }
    score_path, cache = score_pgn(tmp_path, monkeypatch, pgn, tables)
    analysis = AnalysisContext(score_path)
    result = chapter_order.order_chapters(analysis, analysis.read_evidence(cache, required=True), 'opening')
    # Loop (60%) > French group (40%: fr-zero 30% > fr-unres 0% structurally under 1.e4 e6 d4) > custom (0%).
    assert result['ids'] == ['loop', 'fr-zero', 'fr-unres', 'custom']


def test_opening_order_in_build_reports_trimmed_pgn_and_compare(tmp_path, monkeypatch):
    monkeypatch.setattr(httpx.Client, 'request', lambda *a, **k: pytest.fail('Batch must stay offline'))
    # Study order: Rare (10%) then Common (80%) then French (10%, beats Rare? Give French 20% and Rare 10%).
    # Wait: make Common (e5) 60%, French (e6) 30%, Rare (d6) 10%.
    white_pgn = '\n\n'.join(
        [
            '[ChapterURL "https://lichess.org/study/w/rare"]\n[ChapterName "Rare Pirc"]\n\n1. e4 d6 2. d4 Nf6 3. Nc3 *',
            '[ChapterURL "https://lichess.org/study/w/french"]\n[ChapterName "French"]\n\n1. e4 e6 2. d4 d5 3. Nd2 *',
            '[ChapterURL "https://lichess.org/study/w/open"]\n[ChapterName "Open Games"]\n\n'
            '1. e4 e5 2. Nf3 Nc6 3. Bb5 *',
        ]
    )
    black_pgn = '[ChapterURL "https://lichess.org/study/b/one"]\n[ChapterName "Black main"]\n\n1. e4 e5 2. Nf3 Nc6 *'
    candidate_pgn = (
        '[ChapterURL "https://lichess.org/study/cand/c1"]\n[ChapterName "Italian candidate"]\n\n'
        '1. e4 e5 2. Nf3 Nc6 3. Bc4 *'
    )
    white = tmp_path / 'white.pgn'
    black = tmp_path / 'black.pgn'
    cand = tmp_path / 'candidate.pgn'
    white.write_text(white_pgn, encoding='utf-8')
    black.write_text(black_pgn, encoding='utf-8')
    cand.write_text(candidate_pgn, encoding='utf-8')

    cache = tmp_path / 'cache'
    cache.mkdir()
    all_nodes = {
        *graph(tmp_path, white_pgn).nodes,
        *graph(tmp_path, black_pgn).nodes,
        *graph(tmp_path, candidate_pgn).nodes,
    }
    tables = {
        position(''): [('e2e4', 500, 0, 500)],
        position('e4'): [('e7e5', 300, 0, 300), ('e7e6', 150, 0, 150), ('d7d6', 50, 0, 50)],
        position('e4 e5 Nf3'): [('b8c6', 300, 0, 300)],
        position('e4 e6 d4'): [('d7d5', 150, 0, 150)],
        position('e4 d6 d4'): [('g8f6', 50, 0, 50)],
        position('e4 e5 Nf3 Nc6 Bb5'): [('a7a6', 150, 0, 150)],
        position('e4 e5 Nf3 Nc6 Bc4'): [('f8c5', 200, 0, 100)],
    }
    for k in all_nodes:
        rows = tables.get(k, [])
        totals = [sum(r[i] for r in rows) for i in (1, 2, 3)] if rows else [50, 0, 50]
        cache_row(cache, k, data(*totals, rows))

    configs = {color: tmp_path / f'{color}.config.json' for color in ('white', 'black')}
    configs['white'].write_text(json.dumps({'policy': {}, 'coverage_goal': 100}), encoding='utf-8')
    configs['black'].write_text(json.dumps({'policy': {}}), encoding='utf-8')
    comparisons = tmp_path / 'comparisons.json'
    comparisons.write_text(
        json.dumps({'comparisons': [{'name': 'italian', 'color': 'white', 'sources': [str(cand)]}]}),
        encoding='utf-8',
    )

    def run():
        return build.build(
            white,
            black,
            white_config=configs['white'],
            black_config=configs['black'],
            directory=tmp_path / 'data',
            cache=cache,
            offline=True,
            comparisons=comparisons,
        )

    run()
    # In study order: Rare Pirc is W1, French is W2, Open Games is W3.
    trimmed_study = (tmp_path / 'trimmed' / 'white-1-in-100.pgn').read_text(encoding='utf-8')
    assert [g.headers['ChapterName'] for g in chess_games(trimmed_study)] == ['Rare Pirc', 'French', 'Open Games']
    assert '# W1. Rare Pirc' in (tmp_path / 'chapters' / 'W1.md').read_text(encoding='utf-8')
    assert '[W3](../chapters/W3.md)' in (tmp_path / 'comparisons' / 'italian.md').read_text(encoding='utf-8')
    assert 'Chapters are ordered by opening line' not in (tmp_path / 'report.md').read_text(encoding='utf-8')
    score_bytes = (tmp_path / 'data' / 'white.json').read_bytes()

    # Switch White to chapter_order: "opening" -> only white.coverage, comparison.italian and render should rebuild.
    configs['white'].write_text(
        json.dumps({'policy': {}, 'coverage_goal': 100, 'chapter_order': 'opening'}), encoding='utf-8'
    )
    reordered = run()
    assert {s['step'] for s in reordered['steps'] if s['status'] == 'built'} == {
        'white.coverage',
        'comparison.italian',
        'render',
    }
    assert (tmp_path / 'data' / 'white.json').read_bytes() == score_bytes
    assert json.loads(score_bytes)['manifest']['configuration'] == {'policy': {}}

    # Trimmed PGN is now Open Games (60%), French (30%), Rare Pirc (10%).
    trimmed_opening = (tmp_path / 'trimmed' / 'white-1-in-100.pgn').read_text(encoding='utf-8')
    assert [g.headers['ChapterName'] for g in chess_games(trimmed_opening)] == ['Open Games', 'French', 'Rare Pirc']

    # Chapter pages, navigation, tables, glossary note, and comparison link all reflect the new order.
    w1 = (tmp_path / 'chapters' / 'W1.md').read_text(encoding='utf-8')
    w2 = (tmp_path / 'chapters' / 'W2.md').read_text(encoding='utf-8')
    w3 = (tmp_path / 'chapters' / 'W3.md').read_text(encoding='utf-8')
    assert '# W1. Open Games' in w1 and 'Next: [W2](W2.md)' in w1
    assert '# W2. French' in w2 and 'Previous: [W1](W1.md)' in w2 and 'Next: [W3](W3.md)' in w2
    assert '# W3. Rare Pirc' in w3 and 'Previous: [W2](W2.md)' in w3
    report = (tmp_path / 'report.md').read_text(encoding='utf-8')
    summary = (tmp_path / 'summary.md').read_text(encoding='utf-8')
    assert report.index('W1. Open Games') < report.index('W2. French') < report.index('W3. Rare Pirc')
    assert 'Chapters are ordered by opening line and group reach' in report
    assert 'Chapters are ordered by opening line and group reach' in summary
    assert '\u2014' not in report + summary + w1 + w2 + w3
    assert '[W1](../chapters/W1.md)' in (tmp_path / 'comparisons' / 'italian.md').read_text(encoding='utf-8')
    check_links(tmp_path)

    # Changing only coverage_goal keeps the chapter order IDs unchanged, so comparison.italian is reused.
    configs['white'].write_text(
        json.dumps({'policy': {}, 'coverage_goal': 5, 'chapter_order': 'opening'}), encoding='utf-8'
    )
    goal_only = run()
    assert {s['step'] for s in goal_only['steps'] if s['status'] == 'built'} == {'white.coverage', 'render'}
    # At 1 in 5 (20%), Rare Pirc (10%) is cut to 1.e4, which Open Games before it already covers, so it is left out.
    trimmed_cut = (tmp_path / 'trimmed' / 'white-1-in-5.pgn').read_text(encoding='utf-8')
    assert [g.headers['ChapterName'] for g in chess_games(trimmed_cut)] == ['Open Games', 'French']
    cov_data = json.loads((tmp_path / 'data' / 'white.coverage.json').read_text(encoding='utf-8'))
    assert cov_data['at_goal']['pgn']['chapters_left_out'] == ['rare']


def test_coverage_standalone_respects_order_override(tmp_path, monkeypatch):
    pgn = (
        '[ChapterURL "https://lichess.org/study/w/rare"]\n[ChapterName "Rare"]\n\n1. e4 d6 2. d4 *\n\n'
        '[ChapterURL "https://lichess.org/study/w/common"]\n[ChapterName "Common"]\n\n1. e4 e5 2. Nf3 *'
    )
    tables = {position('e4'): [('e7e5', 400, 0, 400), ('d7d6', 100, 0, 100)]}
    score_path, cache = score_pgn(tmp_path, monkeypatch, pgn, tables)
    study_cov, study_text = coverage.analyze(score_path, cache, goal=100, order='study')
    open_cov, open_text = coverage.analyze(score_path, cache, goal=100, order='opening')
    assert study_cov['chapter_order'] == {'mode': 'study', 'ids': ['rare', 'common']}
    assert open_cov['chapter_order'] == {'mode': 'opening', 'ids': ['common', 'rare']}
    assert [g.headers['ChapterName'] for g in chess_games(study_text)] == ['Rare', 'Common']
    assert [g.headers['ChapterName'] for g in chess_games(open_text)] == ['Common', 'Rare']
    assert [c['id'] for c in open_cov['at_goal']['chapters']] == ['common', 'rare']


def test_transposing_entry_keeps_multi_entry_chapter_under_dag_common_ancestor(tmp_path, monkeypatch):
    # Modeled on Vienna Mieses Sidelines: mieses-side has entries at 3...Bc5 and 3...d6 under 2...Nf6 3.g3, while
    # vienna-side also reaches 3...d6 via 2...d6 3.g3 Nf6 with higher single-route probability. Because 2...Nf6 3.g3
    # is still a positive-probability ancestor of all of mieses-side's entries, mieses-side stays grouped with
    # mieses-main under 2...Nf6 3.g3 instead of being pulled up to 2.Nc3 and separated by vienna-side and anderssen.
    pgn = '\n\n'.join(
        [
            '[ChapterURL "https://lichess.org/study/t/mieses-main"]\n[ChapterName "Mieses Mainline"]\n\n'
            '1. e4 e5 2. Nc3 Nf6 3. g3 d5 4. exd5 Nxd5 *',
            '[ChapterURL "https://lichess.org/study/t/anderssen"]\n[ChapterName "Anderssen"]\n\n'
            '1. e4 e5 2. Nc3 Bc5 3. g3 *',
            '[ChapterURL "https://lichess.org/study/t/vienna-side"]\n[ChapterName "Vienna Sidelines"]\n\n'
            '1. e4 e5 2. Nc3 d6 3. g3 f5 (3... Nf6) *',
            '[ChapterURL "https://lichess.org/study/t/mieses-side"]\n[ChapterName "Mieses Sidelines"]\n\n'
            '1. e4 e5 2. Nc3 Nf6 3. g3 Bc5 (3... d6 4. Bg2) 4. Bg2 *',
            '[ChapterURL "https://lichess.org/study/t/paulsen"]\n[ChapterName "Paulsen"]\n\n'
            '1. e4 e5 2. Nc3 Nc6 3. g3 Bc5 *',
        ]
    )
    tables = {
        position('e4'): [('e7e5', 500, 0, 500)],
        position('e4 e5 Nc3'): [
            ('b8c6', 200, 0, 200),
            ('g8f6', 150, 0, 150),
            ('d7d6', 100, 0, 100),
            ('f8c5', 50, 0, 50),
        ],
        position('e4 e5 Nc3 Nc6 g3'): [('f8c5', 200, 0, 200)],
        position('e4 e5 Nc3 Nf6 g3'): [('f8c5', 90, 0, 90), ('d7d5', 40, 0, 40), ('d7d6', 20, 0, 20)],
        position('e4 e5 Nc3 Nf6 g3 d5 exd5'): [('f6d5', 40, 0, 40)],
        # Via 2...d6 3.g3, 3...Nf6 has 60/100 * 20% = 12% single-route prob vs 20/150 * 30% = 4% via 2...Nf6 3.g3 d6.
        position('e4 e5 Nc3 d6 g3'): [('g8f6', 60, 0, 60), ('f7f5', 40, 0, 40)],
    }
    score_path, cache = score_pgn(tmp_path, monkeypatch, pgn, tables)
    analysis = AnalysisContext(score_path)
    result = chapter_order.order_chapters(analysis, analysis.read_evidence(cache, required=True), 'opening')
    # Mieses group (mieses-side 34% + mieses-main 8% = 42%) > Paulsen (40%) > Vienna Sidelines (20%) > Anderssen (10%).
    # Without DAG common-ancestor anchoring, mieses-side would pull up to 2.Nc3 and leave mieses-main (8%) last.
    assert result['ids'] == ['mieses-side', 'mieses-main', 'paulsen', 'vienna-side', 'anderssen']


def test_majority_flow_ignores_minor_alternative_entry(tmp_path, monkeypatch):
    # Modeled on Black's Accelerated Dragon: maroczy has 75% of its reach in 3.d4 cxd4 4.Nxd4 g6 5.c4 and 25% in
    # 3.c4. Majority-flow (>50%) anchors maroczy under 4.Nxd4 g6 5.c4 so all three Accelerated Dragon chapters stay
    # grouped together ahead of rossolimo (3.Bb5), while maroczy still contributes its full union reach (both entries).
    pgn = '\n\n'.join(
        [
            '[ChapterURL "https://lichess.org/study/t/rossolimo"]\n[ChapterName "Rossolimo"]\n\n'
            '1. e4 c5 2. Nf3 Nc6 3. Bb5 g6 *',
            '[ChapterURL "https://lichess.org/study/t/maroczy"]\n[ChapterName "Maroczy Bind"]\n\n'
            '1. e4 c5 2. Nf3 Nc6 3. d4 (3. c4 g6) 3... cxd4 4. Nxd4 g6 5. c4 Bg7 *',
            '[ChapterURL "https://lichess.org/study/t/exchange"]\n[ChapterName "Exchange"]\n\n'
            '1. e4 c5 2. Nf3 Nc6 3. d4 cxd4 4. Nxd4 g6 5. Nxc6 bxc6 *',
            '[ChapterURL "https://lichess.org/study/t/modern"]\n[ChapterName "Modern"]\n\n'
            '1. e4 c5 2. Nf3 Nc6 3. d4 cxd4 4. Nxd4 g6 5. Nc3 Bg7 *',
        ]
    )
    tables = {
        position(''): [('e2e4', 500, 0, 500)],
        position('e4 c5'): [('g1f3', 500, 0, 500)],
        position('e4 c5 Nf3 Nc6'): [('d2d4', 350, 0, 350), ('f1b5', 130, 0, 130), ('c2c4', 20, 0, 20)],
        position('e4 c5 Nf3 Nc6 d4 cxd4'): [('f3d4', 350, 0, 350)],
        position('e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6'): [
            ('b1c3', 200, 0, 200),
            ('d4c6', 90, 0, 90),
            ('c2c4', 60, 0, 60),
        ],
    }
    score_path, cache = score_pgn(tmp_path, monkeypatch, pgn, tables, color='black')
    analysis = AnalysisContext(score_path)
    result = chapter_order.order_chapters(analysis, analysis.read_evidence(cache, required=True), 'opening')
    assert result['ids'] == ['modern', 'exchange', 'maroczy', 'rossolimo']


def test_majority_flow_keeps_scattered_sidelines_at_parent_and_avoids_plurality_misfiling(tmp_path, monkeypatch):
    # Modeled on White's Closed Sicilian: sidelines (36%) is scattered across four 2nd/3rd-move entries (max entry
    # 12% under 2...e6, i.e. 33% of the chapter) while korchnoi (15%) is under 2...e6 and fianchetto (46%, with 72%
    # flowing through 2...Nc6 3.g3) + botvinnik (3%) form a 49% group. Pure single-top-entry would merge sidelines +
    # korchnoi under 2...e6 (51%) ahead of fianchetto; majority-flow keeps sidelines at 2.Nc3 so fianchetto + botvinnik
    # stay first, followed by sidelines and korchnoi.
    pgn = '\n\n'.join(
        [
            '[ChapterURL "https://lichess.org/study/t/korchnoi"]\n[ChapterName "Korchnoi"]\n\n'
            '1. e4 c5 2. Nc3 e6 3. Nge2 Nc6 *',
            '[ChapterURL "https://lichess.org/study/t/sidelines"]\n[ChapterName "Sidelines"]\n\n'
            '1. e4 c5 2. Nc3 e6 (2... d6 3. g3) (2... a6 3. g3) (2... b6 3. g3) 3. Nge2 a6 *',
            '[ChapterURL "https://lichess.org/study/t/botvinnik"]\n[ChapterName "Botvinnik"]\n\n'
            '1. e4 c5 2. Nc3 Nc6 3. g3 g6 4. Bg2 e5 *',
            '[ChapterURL "https://lichess.org/study/t/fianchetto"]\n[ChapterName "Fianchetto"]\n\n'
            '1. e4 c5 2. Nc3 Nc6 (2... g6 3. g3) 3. g3 g6 (3... Nf6 4. Bg2) 4. Bg2 Bg7 *',
        ]
    )
    tables = {
        position('e4'): [('c7c5', 500, 0, 500)],
        position('e4 c5 Nc3'): [
            ('b8c6', 180, 0, 180),
            ('e7e6', 135, 0, 135),
            ('g7g6', 65, 0, 65),
            ('d7d6', 45, 0, 45),
            ('a7a6', 40, 0, 40),
            ('b7b6', 35, 0, 35),
        ],
        position('e4 c5 Nc3 Nc6 g3'): [('g7g6', 100, 0, 100), ('g8f6', 80, 0, 80)],
        position('e4 c5 Nc3 Nc6 g3 g6 Bg2'): [('f8g7', 85, 0, 85), ('e7e5', 15, 0, 15)],
        position('e4 c5 Nc3 e6 Nge2'): [('b8c6', 75, 0, 75), ('a7a6', 60, 0, 60)],
    }
    score_path, cache = score_pgn(tmp_path, monkeypatch, pgn, tables)
    analysis = AnalysisContext(score_path)
    result = chapter_order.order_chapters(analysis, analysis.read_evidence(cache, required=True), 'opening')
    # fianchetto (46% reach, 72% flows through 2...Nc6 3.g3) + botvinnik (3%) form a 49% 2...Nc6 group;
    # sidelines (36%, top entry 12% = 33% of chapter) stays at 2.Nc3; korchnoi (15%) is under 2...e6.
    assert result['ids'] == ['fianchetto', 'botvinnik', 'sidelines', 'korchnoi']


def test_exact_fifty_fifty_split_stays_at_parent_node(tmp_path, monkeypatch):
    # A chapter split exactly 50/50 across 2.c3 and 2.Nf3 Nc6 3.c3 must stay at 1.e4 c5 rather than joining 2.Nf3 Nc6.
    pgn = '\n\n'.join(
        [
            '[ChapterURL "https://lichess.org/study/t/alapin"]\n[ChapterName "Alapin"]\n\n'
            '1. e4 c5 2. c3 (2. Nf3 Nc6 3. c3 d5) 2... d5 *',
            '[ChapterURL "https://lichess.org/study/t/open"]\n[ChapterName "Open"]\n\n1. e4 c5 2. Nf3 Nc6 3. d4 *',
            '[ChapterURL "https://lichess.org/study/t/bowdler"]\n[ChapterName "Bowdler"]\n\n1. e4 c5 2. Bc4 e6 *',
        ]
    )
    tables = {
        position(''): [('e2e4', 500, 0, 500)],
        position('e4 c5'): [('g1f3', 200, 0, 200), ('f1c4', 200, 0, 200), ('c2c3', 100, 0, 100)],
        position('e4 c5 Nf3 Nc6'): [('d2d4', 100, 0, 100), ('c2c3', 100, 0, 100)],
    }
    score_path, cache = score_pgn(tmp_path, monkeypatch, pgn, tables, color='black')
    analysis = AnalysisContext(score_path)
    # alapin is 20% via 2.c3 and 20% via 2.Nf3 Nc6 3.c3 (exact 50/50 split -> stays at 1.e4 c5 with 40% reach!).
    # Under 1.e4 c5: bowdler (40%, study index 2) vs alapin (40%, study index 0) vs open (20%, under 2.Nf3 Nc6).
    # Because alapin does NOT get pulled into 2.Nf3 Nc6, alapin (40%) and bowdler (40%) both beat open (20%).
    result = chapter_order.order_chapters(analysis, analysis.read_evidence(cache, required=True), 'opening')
    assert result['ids'] == ['alapin', 'bowdler', 'open']
